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

    def sample_points(self, n_collocation=1000, n_bc=200, n_ic=200):
        """Return tensors (x_int,t_int,x_bc,t_bc,x_ic,t_ic) on the configured device.

        All returned tensors have requires_grad=True where needed for PDE derivatives.
        """
        x_min, x_max = self.problem.domain["x"]
        t_min, t_max = self.problem.domain["t"]

        x_int = (torch.rand(n_collocation, 1, device=self.device) * (x_max - x_min) + x_min).requires_grad_(True)
        t_int = (torch.rand(n_collocation, 1, device=self.device) * (t_max - t_min) + t_min).requires_grad_(True)

        x_ic = (torch.rand(n_ic, 1, device=self.device) * (x_max - x_min) + x_min).requires_grad_(True)
        t_ic = torch.zeros_like(x_ic, device=self.device).requires_grad_(True)

        x_bc = torch.cat([torch.full((n_bc // 2, 1), x_min, device=self.device), torch.full((n_bc // 2, 1), x_max, device=self.device)], dim=0).requires_grad_(True)
        t_bc = (torch.rand(n_bc, 1, device=self.device) * (t_max - t_min) + t_min).requires_grad_(True)

        return x_int, t_int, x_bc, t_bc, x_ic, t_ic

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

            self.optimizer.zero_grad()
            loss = self.problem.loss(self.model, x_int, t_int, x_bc=x_bc, t_bc=t_bc, x_ic=x_ic, t_ic=t_ic)

            # physics-based regularization
            if physics_reg_weight > 0.0 and self.physics_reg is not None:
                reg = self.physics_reg(self.model, self.problem, x_int, t_int)
                loss = loss + physics_reg_weight * reg

            # hybrid supervision loss (e.g., FD/FEM partial labels or preconditioning)
            if hybrid_weight > 0.0 and self.hybrid_loss is not None:
                hybrid = self.hybrid_loss(self.model)
                loss = loss + hybrid_weight * hybrid

            loss.backward()
            self.optimizer.step()

            loss_history.append(loss.item())

            if epoch % verbose == 0 or epoch == 1:
                # display optional curriculum state
                if curriculum is not None:
                    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}, n_coll={n_coll}, n_bc={n_bc_cur}, n_ic={n_ic_cur}")
                else:
                    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

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
