import torch


class PDEProblem:
    """
    PDE problem helper for PINNs.

    Improvements:
    - Uses BC 'location' properly by applying BC loss only to the boundary subset
      (works with Trainer's sampling at x_min and x_max).
    - Avoids re-computing boundary prediction for each BC when possible.
    """

    def __init__(self, order, coefficients=None, domain=None, initial_condition=None, boundary_conditions=None):
        self.order = int(order)
        self.coefficients = coefficients or {}
        self.domain = domain or {}
        self.initial_condition = initial_condition
        self.boundary_conditions = boundary_conditions or []

    def pde_residual(self, model, x, t):
        """
        Default residual: u_t + a*u_x - nu*u_xx.
        x,t: (N,1) with requires_grad=True
        """
        a = float(self.coefficients.get("a", 0.0))
        nu = float(self.coefficients.get("nu", 0.0))

        inp = torch.cat([x, t], dim=1)
        u = model(inp)

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        if self.order >= 2:
            u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        else:
            u_xx = torch.zeros_like(u)

        return u_t + a * u_x - nu * u_xx

    def _bc_mask(self, x_bc: torch.Tensor, location) -> torch.Tensor:
        """
        Create a boolean mask selecting boundary points at 'location'.
        Trainer samples exactly at x_min and x_max, so exact compare works.
        For safety, use a tolerance.
        """
        if location is None:
            return torch.ones_like(x_bc, dtype=torch.bool)

        loc = torch.as_tensor(location, dtype=x_bc.dtype, device=x_bc.device)
        tol = 1e-12
        return (x_bc - loc).abs() <= tol

    def loss(self, model, x_int, t_int, x_bc=None, t_bc=None, x_ic=None, t_ic=None):
        # PDE residual loss
        res = self.pde_residual(model, x_int, t_int)
        loss_pde = torch.mean(res ** 2)
        total_loss = loss_pde

        # IC loss
        if self.initial_condition is not None and x_ic is not None and t_ic is not None:
            u_pred_ic = model(torch.cat([x_ic, t_ic], dim=1))
            u_true_ic = self.initial_condition(x_ic)
            total_loss = total_loss + torch.mean((u_pred_ic - u_true_ic) ** 2)

        # BC loss (Dirichlet only, respects location)
        if self.boundary_conditions and x_bc is not None and t_bc is not None:
            inp_bc = torch.cat([x_bc, t_bc], dim=1)
            u_pred_all = model(inp_bc)

            loss_bc = 0.0
            for bc_type, location, value in self.boundary_conditions:
                if str(bc_type).lower() != "dirichlet":
                    continue
                mask = self._bc_mask(x_bc, location).view(-1, 1)
                if mask.any():
                    u_pred = u_pred_all[mask]
                    if callable(value):
                        u_val = value(x_bc)[mask]
                    else:
                        u_val = torch.as_tensor(value, dtype=u_pred.dtype, device=u_pred.device)
                        u_val = u_val.expand_as(u_pred)
                    loss_bc = loss_bc + torch.mean((u_pred - u_val) ** 2)

            total_loss = total_loss + loss_bc

        return total_loss
