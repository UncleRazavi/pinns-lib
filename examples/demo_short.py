"""A tiny runnable demo for quick smoke test of the library.

Runs a few training steps on a tiny network and visualizes the result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pinns_lib import PDEProblem, PINNModel, Trainer, Visualizer, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "png" / "demo_short")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fast", action="store_true", help="Use fewer epochs/samples for quicker runs.")
    args = parser.parse_args()

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
    if args.fast:
        epochs, n_collocation, n_bc, n_ic = 50, 200, 40, 40
    else:
        epochs, n_collocation, n_bc, n_ic = args.epochs, 400, 80, 80

    loss_history = trainer.fit(epochs=epochs, n_collocation=n_collocation, n_bc=n_bc, n_ic=n_ic, verbose=50)

    viz = Visualizer(model, pde, device="cpu")
    outdir = args.outdir
    viz.show_loss(loss_history, save_path=outdir / "loss.png", show=False)
    viz.plot(t=0.5, save_path=outdir / "solution_1d_t0.5.png", show=False)
    viz.plot_field(resolution=80, save_path=outdir / "solution_2d.png", show=False)


if __name__ == "__main__":
    main()
