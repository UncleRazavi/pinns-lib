# PINNs-Lib

**PINNs-Lib** is a Python library for solving partial differential equations (PDEs) using **Physics-Informed Neural Networks (PINNs)**.  
It allows users to define PDEs with coefficients, boundary and initial conditions, and automatically trains a neural network to approximate the solution.

---

## Features

- Define PDEs via `PDEProblem` with order and coefficients
- Flexible neural network architectures with `PINNModel`
- Automatic computation of PDE residual + IC/BC losses
- Trainer class with point sampling and optimizer support
- Visualization tools for solutions and loss curves
- Utilities for reproducibility, model save/load, and error metrics
- Example scripts for common PDEs (Heat, Burgers)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/pinns-lib.git
cd pinns-lib
```
## Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage :

```python

import torch
from pinns import PDEProblem, PINNModel, Trainer
from pinns.visualizer import Visualizer
from pinns.utils import set_seed

set_seed(123)

# Define PDE
pde = PDEProblem(
    order=2,
    coefficients={"alpha": 0.01},
    domain={"x": (0,1), "t": (0,1)},
    initial_condition=lambda x: torch.sin(torch.pi * x),
    boundary_conditions=[("dirichlet", "x=0", 0), ("dirichlet","x=1",0)]
)

# Create PINN model
model = PINNModel(input_dim=2, output_dim=1, layers=[64,64,64], activation="tanh")

# Train model
trainer = Trainer(model, pde, optimizer="adam", lr=1e-3)
trainer.fit(epochs=2000, n_collocation=1000, n_bc=200, n_ic=200)

# Visualize solution
viz = Visualizer(model, pde)
viz.plot_solution_1d(time=0.5)
viz.plot_solution_2d(resolution=100)
```
