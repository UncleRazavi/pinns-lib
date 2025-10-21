import torch
from typing import Callable, Optional, Dict


class Trainer:
    """A trainer for PINNs with curriculum learning, physics regularization,
    hybrid solver hooks and transfer-learning helpers.

    Features added:
    - curriculum: schedule that adjusts n_collocation/n_bc/n_ic over epochs
    - physics_reg: optional callable regularizer added to the loss
    - hybrid_loss: optional callable that provides extra supervision (e.g., FD/FEM)
    - pretrain / fine_tune: utilities to pretrain on another problem and fine-tune
    """

    def __init__(self, model, problem, lr=1e-3, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.problem = problem
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Optional hooks / callables set by user
        # physics_reg(model, problem, x_int, t_int, **kwargs) -> scalar tensor
        self.physics_reg: Optional[Callable] = None
        # hybrid_loss(model, inputs, targets) -> scalar tensor
        self.hybrid_loss: Optional[Callable] = None
        # Sampler hook: callable(n_samples, domain) -> (x_tensor, t_tensor)
        self.sampler = None

        # Adaptive loss weighting state
        # keys: 'pde', 'ic', 'bc', 'hybrid', 'reg'
        self.adaptive_weights = {k: 1.0 for k in ("pde", "ic", "bc", "hybrid", "reg")}
        self.adaptive_alpha = 0.12  # update step size exponent
        self.adaptive_eps = 1e-8

    def sample_points(self, n_collocation=1000, n_bc=200, n_ic=200):
        """Return tensors (x_int,t_int,x_bc,t_bc,x_ic,t_ic) on the configured device.

        All returned tensors have requires_grad=True where needed for PDE derivatives.
        """
        x_min, x_max = self.problem.domain["x"]
        t_min, t_max = self.problem.domain["t"]

        # Collocation sampling: use sampler hook if provided
        if callable(self.sampler):
            x_int, t_int = self.sampler(n_collocation, self.problem.domain)
            x_int = x_int.to(self.device).view(-1, 1).requires_grad_(True)
            t_int = t_int.to(self.device).view(-1, 1).requires_grad_(True)
        else:
            x_int = (torch.rand(n_collocation, 1, device=self.device) * (x_max - x_min) + x_min).requires_grad_(True)
            t_int = (torch.rand(n_collocation, 1, device=self.device) * (t_max - t_min) + t_min).requires_grad_(True)

        x_ic = (torch.rand(n_ic, 1, device=self.device) * (x_max - x_min) + x_min).requires_grad_(True)
        t_ic = torch.zeros_like(x_ic, device=self.device).requires_grad_(True)

        x_bc = torch.cat([torch.full((n_bc // 2, 1), x_min, device=self.device), torch.full((n_bc // 2, 1), x_max, device=self.device)], dim=0).requires_grad_(True)
        t_bc = (torch.rand(n_bc, 1, device=self.device) * (t_max - t_min) + t_min).requires_grad_(True)

        return x_int, t_int, x_bc, t_bc, x_ic, t_ic

    def _compute_components(self, x_int, t_int, x_bc, t_bc, x_ic, t_ic):
        """Compute loss components separately so adaptive weighting can inspect them.

        Returns dict: {'pde': tensor, 'ic': tensor or None, 'bc': tensor or None}
        """
        comps = {}
        # PDE residual
        res = self.problem.pde_residual(self.model, x_int, t_int)
        comps['pde'] = torch.mean(res ** 2)

        # IC
        if self.problem.initial_condition is not None and x_ic is not None and t_ic is not None:
            u_pred = self.model(torch.cat([x_ic, t_ic], dim=1))
            u_true = self.problem.initial_condition(x_ic)
            comps['ic'] = torch.mean((u_pred - u_true) ** 2)
        else:
            comps['ic'] = None

        # BC (Dirichlet only)
        loss_bc = None
        if self.problem.boundary_conditions and x_bc is not None and t_bc is not None:
            loss_bc_val = 0.0
            for bc_type, location, value in self.problem.boundary_conditions:
                if bc_type.lower() == 'dirichlet':
                    u_pred = self.model(torch.cat([x_bc, t_bc], dim=1))
                    if callable(value):
                        u_val = value(x_bc)
                    else:
                        u_val = value
                    loss_bc_val = loss_bc_val + torch.mean((u_pred - u_val) ** 2)
            loss_bc = loss_bc_val
        comps['bc'] = loss_bc

        return comps

    def _apply_curriculum(self, epoch: int, curriculum: Optional[Dict]):
        """Compute schedule values for sampling counts based on curriculum.

        curriculum: dict with keys 'start', 'end', 'milestones' and optional 'schedule' callable
        Expected shape example:
            {
                'n_collocation': (start, end),
                'n_bc': (start, end),
                'n_ic': (start, end),
                'schedule': lambda frac: int(start + (end-start)*frac)
            }
        """
        if curriculum is None:
            return None

        def interp(pair, frac, schedule=None):
            start, end = pair
            if schedule is None:
                return int(start + (end - start) * frac)
            return int(schedule(start, end, frac))

        total_epochs = curriculum.get("epochs", 1)
        frac = min(max((epoch - 1) / max(total_epochs - 1, 1), 0.0), 1.0)
        schedule = curriculum.get("schedule", None)

        n_coll = interp(curriculum.get("n_collocation", (1000, 1000)), frac, schedule)
        n_bc = interp(curriculum.get("n_bc", (200, 200)), frac, schedule)
        n_ic = interp(curriculum.get("n_ic", (200, 200)), frac, schedule)

        return n_coll, n_bc, n_ic

    def fit(self,
            epochs=1000,
            n_collocation=1000,
            n_bc=200,
            n_ic=200,
            verbose=100,
            curriculum: Optional[Dict] = None,
            physics_reg_weight: float = 0.0,
            hybrid_weight: float = 0.0):
        """Train the model.

        Args:
            curriculum: optional dict controlling progressive sampling counts.
            physics_reg_weight: scale for physics-based regularizer (if self.physics_reg provided)
            hybrid_weight: scale for hybrid supervision loss (if self.hybrid_loss provided)
        """
        self.model.train()
        loss_history = []

        for epoch in range(1, epochs + 1):
            # Curriculum scheduling
            if curriculum is not None:
                n_coll, n_bc_cur, n_ic_cur = self._apply_curriculum(epoch, curriculum)
            else:
                n_coll, n_bc_cur, n_ic_cur = n_collocation, n_bc, n_ic

            x_int, t_int, x_bc, t_bc, x_ic, t_ic = self.sample_points(n_coll, n_bc_cur, n_ic_cur)

            # compute components
            comps = self._compute_components(x_int, t_int, x_bc, t_bc, x_ic, t_ic)

            # optional physics / hybrid components
            reg_comp = None
            if physics_reg_weight > 0.0 and self.physics_reg is not None:
                reg_comp = self.physics_reg(self.model, self.problem, x_int, t_int)

            hybrid_comp = None
            if hybrid_weight > 0.0 and self.hybrid_loss is not None:
                hybrid_comp = self.hybrid_loss(self.model)

            # Adaptive loss weighting: compute gradient norms per component and update weights
            # controlled by self.adaptive_alpha; only if any adaptive flag is set via curriculum dict
            if curriculum and curriculum.get('adaptive_weighting', False):
                # gather active comps
                active = {}
                if comps.get('pde') is not None:
                    active['pde'] = comps['pde']
                if comps.get('ic') is not None:
                    active['ic'] = comps['ic']
                if comps.get('bc') is not None:
                    active['bc'] = comps['bc']
                if hybrid_comp is not None:
                    active['hybrid'] = hybrid_comp
                if reg_comp is not None:
                    active['reg'] = reg_comp

                grad_norms = {}
                for i, (name, comp) in enumerate(active.items()):
                    # compute gradient of this component w.r.t. model params
                    grads = torch.autograd.grad(comp, [p for p in self.model.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
                    gnorm = torch.tensor(0.0, device=self.device)
                    for g in grads:
                        if g is not None:
                            gnorm = gnorm + g.detach().norm() ** 2
                    grad_norms[name] = torch.sqrt(gnorm + 1e-16).item()

                if len(grad_norms) > 0:
                    avg = float(sum(grad_norms.values()) / len(grad_norms))
                    # update weights
                    for name, gnorm in grad_norms.items():
                        old = self.adaptive_weights.get(name, 1.0)
                        new = old * (avg / (gnorm + self.adaptive_eps)) ** self.adaptive_alpha
                        self.adaptive_weights[name] = float(new)
                    # normalize to keep mean weight ~1
                    vals = [v for k, v in self.adaptive_weights.items() if k in grad_norms]
                    if sum(vals) > 0:
                        factor = len(vals) / sum(vals)
                        for k in grad_norms.keys():
                            self.adaptive_weights[k] = float(self.adaptive_weights[k] * factor)

            # build final weighted loss
            total = 0.0
            # pde
            w_pde = self.adaptive_weights.get('pde', 1.0)
            total = total + w_pde * comps['pde']
            # ic
            if comps.get('ic') is not None:
                w_ic = self.adaptive_weights.get('ic', 1.0)
                total = total + w_ic * comps['ic']
            # bc
            if comps.get('bc') is not None:
                w_bc = self.adaptive_weights.get('bc', 1.0)
                total = total + w_bc * comps['bc']
            # hybrid
            if hybrid_comp is not None:
                w_h = self.adaptive_weights.get('hybrid', 1.0)
                total = total + hybrid_weight * w_h * hybrid_comp
            # reg
            if reg_comp is not None:
                w_r = self.adaptive_weights.get('reg', 1.0)
                total = total + physics_reg_weight * w_r * reg_comp

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

            loss_history.append(float(total.item()))

            if epoch % verbose == 0 or epoch == 1:
                # display optional curriculum state
                if curriculum is not None:
                    print(f"Epoch {epoch}/{epochs}, Loss: {total.item():.6f}, n_coll={n_coll}, n_bc={n_bc_cur}, n_ic={n_ic_cur}")
                    if curriculum.get('adaptive_weighting', False):
                        print(" adaptive_weights:", {k: round(v, 4) for k, v in self.adaptive_weights.items()})
                else:
                    print(f"Epoch {epoch}/{epochs}, Loss: {total.item():.6f}")

        return loss_history

    # Transfer-learning helpers
    def pretrain(self, other_problem, epochs=100, lr=1e-3, device=None, **fit_kwargs):
        """Pretrain model on a simpler PDEProblem (in-place)."""
        device = device or self.device
        original_problem = self.problem
        original_optimizer = self.optimizer

        self.problem = other_problem
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        history = self.fit(epochs=epochs, **fit_kwargs)

        # restore
        self.problem = original_problem
        self.optimizer = original_optimizer
        return history

    def fine_tune(self, epochs=100, lr=1e-4, device=None, **fit_kwargs):
        """Fine-tune the model on the trainer's current problem."""
        device = device or self.device
        original_optimizer = self.optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        history = self.fit(epochs=epochs, **fit_kwargs)

        self.optimizer = original_optimizer
        return history
