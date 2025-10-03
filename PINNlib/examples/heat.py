import torch
from PDEproblem import PDEProblem
from PINNModel import PINNModel
from trainer import Trainer
from visualizer import Visualizer
from utils import set_seed

set_seed(123)

# Define Heat Equation PDE (simple metadata for the example)
pde = PDEProblem(
    order=2,
    coefficients={"nu": 0.01},
    domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
    initial_condition=lambda x: torch.sin(torch.pi * x),
    boundary_conditions=[("dirichlet", 0.0, lambda x: torch.zeros_like(x)), ("dirichlet", 1.0, lambda x: torch.zeros_like(x))]
)

# PINN model
model = PINNModel(input_dim=2, output_dim=1, layers=[64, 64, 64], activation="tanh")

# Trainer
trainer = Trainer(model, pde, lr=1e-3, device="cpu")

# Train (returns loss history)
loss_history = trainer.fit(epochs=2000, n_collocation=1000)

# Visualize
viz = Visualizer()
viz.plot_loss(loss_history)
viz.plot_1d(model, pde, t=0.5)
viz.plot_2d(model, pde, resolution=100)
