import torch


class Trainer:
    """A minimal trainer for PINNs using Adam and the PDEProblem.loss API.

    This keeps sampling and the training loop tiny and easy to read.
    """

    def __init__(self, model, problem, lr=1e-3, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.problem = problem
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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

    def fit(self, epochs=1000, n_collocation=1000, n_bc=200, n_ic=200, verbose=100):
        self.model.train()
        loss_history = []

        for epoch in range(1, epochs + 1):
            x_int, t_int, x_bc, t_bc, x_ic, t_ic = self.sample_points(n_collocation, n_bc, n_ic)

            self.optimizer.zero_grad()
            loss = self.problem.loss(self.model, x_int, t_int, x_bc=x_bc, t_bc=t_bc, x_ic=x_ic, t_ic=t_ic)
            loss.backward()
            self.optimizer.step()

            loss_history.append(loss.item())

            if epoch % verbose == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

        return loss_history
