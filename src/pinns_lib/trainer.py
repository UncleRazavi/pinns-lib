import torch
from typing import Callable, Optional, Dict


class Trainer:
    """
    Trainer for PINNs.

    Improvements:
    - Only interior points require gradients (faster, less memory).
    - Safer boundary sampling when n_bc is odd.
    - Keeps adaptive weighting feature, but avoids unnecessary graph retention.
    """

    def __init__(self, model, problem, lr=1e-3, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.problem = problem
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.physics_reg: Optional[Callable] = None
        self.hybrid_loss: Optional[Callable] = None
        self.sampler = None

        self.adaptive_weights = {k: 1.0 for k in ("pde", "ic", "bc", "hybrid", "reg")}
        self.adaptive_alpha = 0.12
        self.adaptive_eps = 1e-8

    def sample_points(self, n_collocation=1000, n_bc=200, n_ic=200):
        x_min, x_max = self.problem.domain["x"]
        t_min, t_max = self.problem.domain["t"]

        # Collocation sampling
        if callable(self.sampler):
            x_int, t_int = self.sampler(n_collocation, self.problem.domain)
            x_int = x_int.to(self.device).view(-1, 1).requires_grad_(True)
            t_int = t_int.to(self.device).view(-1, 1).requires_grad_(True)
        else:
            x_int = (torch.rand(n_collocation, 1, device=self.device) * (x_max - x_min) + x_min).requires_grad_(True)
            t_int = (torch.rand(n_collocation, 1, device=self.device) * (t_max - t_min) + t_min).requires_grad_(True)

        # IC points (no gradients needed for basic IC loss)
        x_ic = (torch.rand(n_ic, 1, device=self.device) * (x_max - x_min) + x_min)
        t_ic = torch.zeros_like(x_ic, device=self.device)

        # BC points: split as evenly as possible between x_min and x_max
        n_left = n_bc // 2
        n_right = n_bc - n_left
        x_bc = torch.cat(
            [torch.full((n_left, 1), x_min, device=self.device),
             torch.full((n_right, 1), x_max, device=self.device)],
            dim=0,
        )
        t_bc = (torch.rand(n_bc, 1, device=self.device) * (t_max - t_min) + t_min)

        return x_int, t_int, x_bc, t_bc, x_ic, t_ic

    def _compute_components(self, x_int, t_int, x_bc, t_bc, x_ic, t_ic):
        comps = {}

        res = self.problem.pde_residual(self.model, x_int, t_int)
        comps["pde"] = torch.mean(res ** 2)

        if self.problem.initial_condition is not None and x_ic is not None and t_ic is not None:
            u_pred = self.model(torch.cat([x_ic, t_ic], dim=1))
            u_true = self.problem.initial_condition(x_ic)
            comps["ic"] = torch.mean((u_pred - u_true) ** 2)
        else:
            comps["ic"] = None

        loss_bc = None
        if self.problem.boundary_conditions and x_bc is not None and t_bc is not None:
            loss_bc_val = 0.0
            inp_bc = torch.cat([x_bc, t_bc], dim=1)
            u_pred_all = self.model(inp_bc)

            for bc_type, location, value in self.problem.boundary_conditions:
                if str(bc_type).lower() != "dirichlet":
                    continue

                # mask boundary points at this location (tolerant)
                loc = torch.as_tensor(location, dtype=x_bc.dtype, device=x_bc.device)
                mask = (x_bc - loc).abs() <= 1e-12
                mask = mask.view(-1, 1)

                if mask.any():
                    u_pred = u_pred_all[mask]
                    if callable(value):
                        u_val = value(x_bc)[mask]
                    else:
                        u_val = torch.as_tensor(value, dtype=u_pred.dtype, device=u_pred.device).expand_as(u_pred)
                    loss_bc_val = loss_bc_val + torch.mean((u_pred - u_val) ** 2)

            loss_bc = loss_bc_val

        comps["bc"] = loss_bc
        return comps

    def _apply_curriculum(self, epoch: int, curriculum: Optional[Dict]):
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

        return max(n_coll, 1), max(n_bc, 0), max(n_ic, 0)

    def fit(
        self,
        epochs=1000,
        n_collocation=1000,
        n_bc=200,
        n_ic=200,
        verbose=100,
        curriculum: Optional[Dict] = None,
        physics_reg_weight: float = 0.0,
        hybrid_weight: float = 0.0,
        grad_clip: float = 0.0,
    ):
        self.model.train()
        loss_history = []

        params = [p for p in self.model.parameters() if p.requires_grad]

        for epoch in range(1, epochs + 1):
            if curriculum is not None:
                n_coll, n_bc_cur, n_ic_cur = self._apply_curriculum(epoch, curriculum)
            else:
                n_coll, n_bc_cur, n_ic_cur = n_collocation, n_bc, n_ic

            x_int, t_int, x_bc, t_bc, x_ic, t_ic = self.sample_points(n_coll, n_bc_cur, n_ic_cur)
            comps = self._compute_components(x_int, t_int, x_bc, t_bc, x_ic, t_ic)

            reg_comp = None
            if physics_reg_weight > 0.0 and self.physics_reg is not None:
                reg_comp = self.physics_reg(self.model, self.problem, x_int, t_int)

            hybrid_comp = None
            if hybrid_weight > 0.0 and self.hybrid_loss is not None:
                hybrid_comp = self.hybrid_loss(self.model)

            if curriculum and curriculum.get("adaptive_weighting", False):
                active = {}
                if comps.get("pde") is not None:
                    active["pde"] = comps["pde"]
                if comps.get("ic") is not None:
                    active["ic"] = comps["ic"]
                if comps.get("bc") is not None:
                    active["bc"] = comps["bc"]
                if hybrid_comp is not None:
                    active["hybrid"] = hybrid_comp
                if reg_comp is not None:
                    active["reg"] = reg_comp

                grad_norms = {}
                for name, comp in active.items():
                    grads = torch.autograd.grad(comp, params, retain_graph=True, allow_unused=True)
                    g2 = 0.0
                    for g in grads:
                        if g is not None:
                            g2 = g2 + float(g.detach().norm().item()) ** 2
                    grad_norms[name] = (g2 ** 0.5)

                if grad_norms:
                    avg = sum(grad_norms.values()) / len(grad_norms)
                    for name, gnorm in grad_norms.items():
                        old = self.adaptive_weights.get(name, 1.0)
                        new = old * (avg / (gnorm + self.adaptive_eps)) ** self.adaptive_alpha
                        self.adaptive_weights[name] = float(new)

                    vals = [self.adaptive_weights[k] for k in grad_norms.keys()]
                    factor = len(vals) / (sum(vals) + 1e-16)
                    for k in grad_norms.keys():
                        self.adaptive_weights[k] *= factor

            total = 0.0
            total = total + self.adaptive_weights.get("pde", 1.0) * comps["pde"]
            if comps.get("ic") is not None:
                total = total + self.adaptive_weights.get("ic", 1.0) * comps["ic"]
            if comps.get("bc") is not None:
                total = total + self.adaptive_weights.get("bc", 1.0) * comps["bc"]
            if hybrid_comp is not None:
                total = total + hybrid_weight * self.adaptive_weights.get("hybrid", 1.0) * hybrid_comp
            if reg_comp is not None:
                total = total + physics_reg_weight * self.adaptive_weights.get("reg", 1.0) * reg_comp

            self.optimizer.zero_grad(set_to_none=True)
            total.backward()

            if grad_clip and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))

            self.optimizer.step()
            loss_history.append(float(total.item()))

            if epoch % verbose == 0 or epoch == 1:
                if curriculum is not None:
                    msg = f"Epoch {epoch}/{epochs}, Loss: {total.item():.6f}, n_coll={n_coll}, n_bc={n_bc_cur}, n_ic={n_ic_cur}"
                    print(msg)
                    if curriculum.get("adaptive_weighting", False):
                        print(" adaptive_weights:", {k: round(v, 4) for k, v in self.adaptive_weights.items()})
                else:
                    print(f"Epoch {epoch}/{epochs}, Loss: {total.item():.6f}")

        return loss_history

    def pretrain(self, other_problem, epochs=100, lr=1e-3, **fit_kwargs):
        original_problem = self.problem
        original_optimizer = self.optimizer

        self.problem = other_problem
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        history = self.fit(epochs=epochs, **fit_kwargs)

        self.problem = original_problem
        self.optimizer = original_optimizer
        return history

    def fine_tune(self, epochs=100, lr=1e-4, **fit_kwargs):
        original_optimizer = self.optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        history = self.fit(epochs=epochs, **fit_kwargs)

        self.optimizer = original_optimizer
        return history
