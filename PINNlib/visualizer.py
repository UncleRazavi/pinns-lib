import torch
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Simple visualizer for 1D PINN problems.

    Methods:
        plot_loss(loss_history)
        plot_1d(model, problem, t=0.0, resolution=200)
        plot_2d(model, problem, resolution=100)
    """

    @staticmethod
    def plot_loss(loss_history, figsize=(6, 4)):
        plt.figure(figsize=figsize)
        plt.plot(loss_history)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.title("Training Loss")
        plt.grid(alpha=0.3)
        plt.show()

    @staticmethod
    def plot_1d(model, problem, t=0.0, resolution=200, device="cpu"):
        """Plot u(x,t) vs x at a fixed time t."""
        x_min, x_max = problem.domain["x"]
        x = torch.linspace(x_min, x_max, resolution, device=device).view(-1, 1)
        t_tensor = torch.full_like(x, float(t))
        inp = torch.cat([x, t_tensor], dim=1).to(device)

        with torch.no_grad():
            u = model(inp).cpu().numpy().squeeze()

        plt.figure(figsize=(6, 4))
        plt.plot(x.cpu().numpy(), u, lw=2)
        plt.xlabel("x")
        plt.ylabel(f"u(x, t={t})")
        plt.title(f"Solution at t={t}")
        plt.grid(alpha=0.3)
        plt.show()

    @staticmethod
    def plot_2d(model, problem, resolution=100, device="cpu", cmap="viridis"):
        """Plot u(x,t) as a 2D colormap over the domain using pcolormesh.

        The x-axis will be t and y-axis will be x to match common PDE plots.
        """
        x_min, x_max = problem.domain["x"]
        t_min, t_max = problem.domain["t"]

        x = np.linspace(x_min, x_max, resolution)
        t = np.linspace(t_min, t_max, resolution)
        X, T = np.meshgrid(x, t, indexing="xy")

        XT = np.stack([X.ravel(), T.ravel()], axis=1)
        XT_tensor = torch.tensor(XT, dtype=torch.float32, device=device)

        with torch.no_grad():
            U = model(XT_tensor).cpu().numpy().reshape(resolution, resolution)

        # pcolormesh expects grid edges; use extent for imshow-like layout
        plt.figure(figsize=(7, 5))
        plt.pcolormesh(t, x, U.T, shading="auto", cmap=cmap)
        plt.colorbar(label="u(x,t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Solution u(x,t)")
        plt.show()
