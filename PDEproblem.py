import torch


class PDEProblem:
    """Simplified PDE problem helper for PINNs.

    This class stores PDE metadata and computes a simple MSE loss composed of:
    - PDE residual loss
    - initial condition (IC) MSE (if provided)
    - boundary condition (BC) MSE for Dirichlet BCs (if provided)
    """

    def __init__(self, order, coefficients=None, domain=None, initial_condition=None, boundary_conditions=None):
        self.order = order
        self.coefficients = coefficients or {}
        self.domain = domain or {}
        self.initial_condition = initial_condition
        self.boundary_conditions = boundary_conditions or []

    def pde_residual(self, model, x, t):
        """Compute residual u_t + a*u_x - nu*u_xx.

        x and t must be tensors with requires_grad=True and shape (N,1).
        model accepts torch.cat([x,t], dim=1) and returns u with shape (N,1).
        """
        a = self.coefficients.get("a", 0.0)
        nu = self.coefficients.get("nu", 0.0)

        inp = torch.cat([x, t], dim=1)
        u = model(inp)

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

        if self.order >= 2:
            u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        else:
            u_xx = torch.zeros_like(u)

        residual = u_t + a * u_x - nu * u_xx
        return residual

    def loss(self, model, x_int, t_int, x_bc=None, t_bc=None, x_ic=None, t_ic=None):
        """Compute MSE loss: PDE residual + optional IC and BC terms.

        All tensor arguments are expected to be shaped (N,1).
        """
        # PDE residual loss
        res = self.pde_residual(model, x_int, t_int)
        loss_pde = torch.mean(res ** 2)

        total_loss = loss_pde

        # initial condition loss
        if self.initial_condition is not None and x_ic is not None and t_ic is not None:
            u_pred = model(torch.cat([x_ic, t_ic], dim=1))
            u_true = self.initial_condition(x_ic)
            loss_ic = torch.mean((u_pred - u_true) ** 2)
            total_loss = total_loss + loss_ic

        # boundary conditions (Dirichlet)
        loss_bc = 0.0
        if self.boundary_conditions and x_bc is not None and t_bc is not None:
            for bc_type, location, value in self.boundary_conditions:
                if bc_type.lower() == "dirichlet":
                    u_pred = model(torch.cat([x_bc, t_bc], dim=1))
                    if callable(value):
                        u_val = value(x_bc)
                    else:
                        u_val = value
                    loss_bc = loss_bc + torch.mean((u_pred - u_val) ** 2)

        total_loss = total_loss + loss_bc
        return total_loss