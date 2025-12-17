<<<<<<< HEAD:visualizer.py
import torch
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Simple visualizer for 1D PINN problems.

    Supports both instance-based usage:
        viz = Visualizer(model, problem)
        viz.plot_1d(t=0.5)

    and direct/static usage:
        Visualizer.plot_1d(model, problem, t=0.5)
    """

    def __init__(self, model=None, problem=None, device="cpu"):
        self.model = model
        self.problem = problem
        self.device = device

    # Static helpers (can be used without creating an instance)
    @staticmethod
    def plot_loss(loss_history, figsize=(6, 4), figure_name="Figure 1", save=False):
        plt.figure(figsize=figsize)
        plt.plot(loss_history)
        plt.gcf().canvas.manager.set_window_title(figure_name)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.title("Training Loss")
        plt.grid(alpha=0.3)
        if save:
            plt.savefig(figure_name)
        plt.show()

    @staticmethod
    def plot_1d(model, problem, t=0.0, resolution=200, device="cpu", figure_name="Figure 1", save=False):
        """Plot u(x,t) vs x at a fixed time t."""
        x_min, x_max = problem.domain["x"]
        x = torch.linspace(x_min, x_max, resolution, device=device).view(-1, 1)
        t_tensor = torch.full_like(x, float(t))
        inp = torch.cat([x, t_tensor], dim=1).to(device)

        with torch.no_grad():
            u = model(inp).cpu().numpy().squeeze()

        plt.figure(figsize=(6, 4))
        plt.plot(x.cpu().numpy(), u, lw=2)
        plt.gcf().canvas.manager.set_window_title(figure_name)
        plt.xlabel("x")
        plt.ylabel(f"u(x, t={t})")
        plt.title(f"Solution at t={t}")
        plt.grid(alpha=0.3)
        if save:
            plt.savefig(figure_name)
        plt.show()

    @staticmethod
    def plot_2d(model, problem, resolution=100, device="cpu", cmap="viridis", figure_name="Figure 1", save=False):
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

        plt.figure(figsize=(7, 5))
        plt.pcolormesh(t, x, U.T, shading="auto", cmap=cmap)
        plt.gcf().canvas.manager.set_window_title(figure_name)
        plt.colorbar(label="u(x,t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Solution u(x,t)")
        if save:
            plt.savefig(figure_name)
        plt.show()

    # Instance wrappers (convenience methods that use stored model/problem)
    def show_loss(self, loss_history):
        return Visualizer.plot_loss(loss_history)

    def plot(self, t=0.0, resolution=200):
        if self.model is None or self.problem is None:
            raise ValueError("Visualizer requires model and problem for instance plotting")
        return Visualizer.plot_1d(self.model, self.problem, t=t, resolution=resolution, device=self.device)

    def plot_field(self, resolution=100):
        if self.model is None or self.problem is None:
            raise ValueError("Visualizer requires model and problem for instance plotting")
        return Visualizer.plot_2d(self.model, self.problem, resolution=resolution, device=self.device)
=======
import torch
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    @staticmethod
    def plot_loss(loss_history, figsize=(6, 4)):
        plt.figure(figsize=figsize)
        plt.plot(loss_history)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.title("Training Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_1d(model, problem, t=0.0, resolution=200, device="cpu"):
        x_min, x_max = problem.domain["x"]
        x = torch.linspace(x_min, x_max, resolution, device=device).view(-1, 1)
        t_tensor = torch.full_like(x, float(t))
        inp = torch.cat([x, t_tensor], dim=1)

        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            u = model(inp).cpu().numpy().squeeze()
        model.train(model_was_training)

        plt.figure(figsize=(6, 4))
        plt.plot(x.cpu().numpy(), u, lw=2)
        plt.xlabel("x")
        plt.ylabel(f"u(x, t={t})")
        plt.title(f"Solution at t={t}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_2d(model, problem, resolution=100, device="cpu", cmap="viridis"):
        x_min, x_max = problem.domain["x"]
        t_min, t_max = problem.domain["t"]

        x = np.linspace(x_min, x_max, resolution)
        t = np.linspace(t_min, t_max, resolution)
        X, T = np.meshgrid(x, t, indexing="xy")

        XT = np.stack([X.ravel(), T.ravel()], axis=1)
        XT_tensor = torch.tensor(XT, dtype=torch.float32, device=device)

        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            U = model(XT_tensor).cpu().numpy().reshape(resolution, resolution)
        model.train(model_was_training)

        plt.figure(figsize=(7, 5))
        plt.pcolormesh(t, x, U.T, shading="auto", cmap=cmap)
        plt.colorbar(label="u(x,t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Solution u(x,t)")
        plt.tight_layout()
        plt.show()
>>>>>>> d0de537 (Optimize PINN core, add examples, improve repo structure):src/pinns_lib/visualizer.py
