from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pinns_lib import PDEProblem, PINNModel, Trainer, Visualizer, set_seed



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "png" / "heat")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--fast", action="store_true", help="Use fewer epochs/samples for quicker runs.")
    args = parser.parse_args()

    set_seed(123)

    # Define Heat Equation PDE (simple metadata for the example)
    pde = PDEProblem(
        order=2,
        coefficients={"nu": 0.01},
        domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        initial_condition=lambda x: torch.sin(torch.pi * x),
        boundary_conditions=[("dirichlet", 0.0, lambda x: torch.zeros_like(x)), ("dirichlet", 1.0, lambda x: torch.zeros_like(x))],
    )

    # PINN model
    model = PINNModel(input_dim=2, output_dim=1, layers=[64, 64, 64], activation="tanh")

    # Trainer
    trainer = Trainer(model, pde, lr=1e-3, device="cpu")

    if args.fast:
        epochs, n_collocation = 300, 400
    else:
        epochs, n_collocation = args.epochs, 1000

    # Train (returns loss history)
    loss_history = trainer.fit(epochs=epochs, n_collocation=n_collocation)

    # Visualize
    outdir = args.outdir
    Visualizer.plot_loss(loss_history, save_path=outdir / "loss.png", show=False)
    Visualizer.plot_1d(model, pde, t=0.5, save_path=outdir / "solution_1d_t0.5.png", show=False)
    Visualizer.plot_2d(model, pde, resolution=100, save_path=outdir / "solution_2d.png", show=False)


if __name__ == "__main__":
    main()
