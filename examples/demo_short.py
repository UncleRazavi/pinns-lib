"""A tiny runnable demo for quick smoke test of the library.

Runs a few training steps on a tiny network and visualizes the result.
"""
import torch
from PDEproblem import PDEProblem
from PINNModel import PINNModel
from trainer import Trainer
from visualizer import Visualizer
from utils import set_seed


def main():
    set_seed(0)

    # tiny PDE and model for a quick smoke test
    pde = PDEProblem(
        order=2,
        coefficients={"nu": 0.01},
        domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        initial_condition=lambda x: torch.sin(torch.pi * x),
        boundary_conditions=[("dirichlet", 0.0, lambda x: torch.zeros_like(x)), ("dirichlet", 1.0, lambda x: torch.zeros_like(x))]
    )

    model = PINNModel(input_dim=2, output_dim=1, layers=[32, 32], activation="tanh")

    trainer = Trainer(model, pde, lr=1e-3, device="cpu")
    loss_history = trainer.fit(epochs=200, n_collocation=400, n_bc=80, n_ic=80, verbose=50)

    viz = Visualizer(model, pde, device="cpu")
    viz.show_loss(loss_history)
    viz.plot(t=0.5)
    viz.plot_field(resolution=80)


if __name__ == "__main__":
    main()
