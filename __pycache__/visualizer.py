import torch
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, model, problem, device="cpu"):
        self.model = model.to(device)
        self.problem = problem
        self.device = device

    def plot_loss(self, loss_history):
        plt.figure(figsize=(6,4))
        plt.semilogy(loss_history, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (log scale)")
        plt.title("Training Loss")
        plt.legend()
        plt.show()

    def plot_solution_1d(self, resolution=100, time=0.5):
        """
        Plot u(x,t) vs x at fixed t.
        """
        x_min, x_max = self.problem.domain["x"]
        x = torch.linspace(x_min, x_max, resolution).view(-1,1).to(self.device)
        t = torch.full_like(x, time).to(self.device)
        xt = torch.cat([x,t], dim=1)

        with torch.no_grad():
            u_pred = self.model(xt).cpu().numpy()

        plt.figure(figsize=(6,4))
        plt.plot(x.cpu().numpy(), u_pred, label=f"t={time}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title("PINN Solution at Fixed Time")
        plt.legend()
        plt.show()

    def plot_solution_2d(self, resolution=100):
        """
        Plot heatmap of u(x,t) over domain.
        """
        x_min, x_max = self.problem.domain["x"]
        t_min, t_max = self.problem.domain["t"]

        x = torch.linspace(x_min, x_max, resolution)
        t = torch.linspace(t_min, t_max, resolution)
        X, T = torch.meshgrid(x, t, indexing="ij")

        xt = torch.cat([X.reshape(-1,1), T.reshape(-1,1)], dim=1).to(self.device)

        with torch.no_grad():
            U = self.model(xt).cpu().numpy().reshape(resolution,resolution)

        plt.figure(figsize=(6,5))
        plt.imshow(U, extent=[t_min, t_max, x_min, x_max],
                   origin="lower", aspect="auto", cmap="jet")
        plt.colorbar(label="u(x,t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("PINN Solution u(x,t)")
        plt.show()
