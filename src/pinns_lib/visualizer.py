from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class Visualizer:
    """Simple visualizer for 1D PINN problems.

    Supports both instance-based usage:
        viz = Visualizer(model, problem)
        viz.plot_1d(t=0.5)

    and direct/static usage:
        Visualizer.plot_1d(model, problem, t=0.5)
    """

    def __init__(self, model=None, problem=None, device: str = "cpu"):
        self.model = model
        self.problem = problem
        self.device = device

    @staticmethod
    def _save_or_show(*, save_path: Optional[str | Path], show: bool):
        import matplotlib.pyplot as plt

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_loss(
        loss_history,
        figsize=(6, 4),
        *,
        save_path: Optional[str | Path] = None,
        show: Optional[bool] = None,
    ):
        import matplotlib.pyplot as plt

        if show is None:
            show = save_path is None

        plt.figure(figsize=figsize)
        plt.plot(loss_history)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.title("Training Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        Visualizer._save_or_show(save_path=save_path, show=bool(show))

    @staticmethod
    def plot_1d(
        model,
        problem,
        *,
        t: float = 0.0,
        resolution: int = 200,
        device: str = "cpu",
        save_path: Optional[str | Path] = None,
        show: Optional[bool] = None,
    ):
        import matplotlib.pyplot as plt

        if show is None:
            show = save_path is None

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
        Visualizer._save_or_show(save_path=save_path, show=bool(show))

    @staticmethod
    def plot_2d(
        model,
        problem,
        *,
        resolution: int = 100,
        device: str = "cpu",
        cmap: str = "viridis",
        save_path: Optional[str | Path] = None,
        show: Optional[bool] = None,
    ):
        import matplotlib.pyplot as plt

        if show is None:
            show = save_path is None

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
        Visualizer._save_or_show(save_path=save_path, show=bool(show))

    def show_loss(self, loss_history, *, save_path: Optional[str | Path] = None, show: Optional[bool] = None):
        return Visualizer.plot_loss(loss_history, save_path=save_path, show=show)

    def plot(self, *, t: float = 0.0, resolution: int = 200, save_path: Optional[str | Path] = None, show: Optional[bool] = None):
        if self.model is None or self.problem is None:
            raise ValueError("Visualizer requires model and problem for instance plotting")
        return Visualizer.plot_1d(self.model, self.problem, t=t, resolution=resolution, device=self.device, save_path=save_path, show=show)

    def plot_field(self, *, resolution: int = 100, save_path: Optional[str | Path] = None, show: Optional[bool] = None):
        if self.model is None or self.problem is None:
            raise ValueError("Visualizer requires model and problem for instance plotting")
        return Visualizer.plot_2d(self.model, self.problem, resolution=resolution, device=self.device, save_path=save_path, show=show)
