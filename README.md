# PINNs-Lib

A small, easy-to-use Python library for solving PDEs with Physics-Informed Neural Networks (PINNs).

This project provides a lightweight toolkit to define PDE problems, build PINN models, train them, and visualize results.

Key components
- `PDEproblem.PDEProblem` — describe the PDE (order, coefficients, domain, IC/BC)
- `PINNModel.PINNModel` — flexible fully-connected network (optional Fourier features)
- `trainer.Trainer` — simple training loop with point sampling and Adam optimizer
- `visualizer.Visualizer` — quick plotting helpers (loss, 1D line, 2D field)
- `utils` — helpers: seeding, save/load, error metrics

---

## Quick features

- Define PDEs with coefficients, IC and BCs
- Compute PDE residual + IC/BC MSE automatically
- Minimal trainer that returns loss history
- Simple, dependency-light visualization with matplotlib
- Small examples: `examples/heat.py`, `examples/burgers.py`

---

## Install

Recommended: create a virtual environment and install dependencies from `requirements.txt`.

Windows (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or simply:

```bash
pip install -r requirements.txt
```

---

## Minimal example (Heat equation)

This example matches the small API used in the repository. Run it from the repository root.

```python
import torch
from PDEproblem import PDEProblem
from PINNModel import PINNModel
from trainer import Trainer
from visualizer import Visualizer
from utils import set_seed

set_seed(123)

pde = PDEProblem(
    order=2,
    coefficients={"nu": 0.01},
    domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
    initial_condition=lambda x: torch.sin(torch.pi * x),
    boundary_conditions=[("dirichlet", 0.0, lambda x: torch.zeros_like(x)), ("dirichlet", 1.0, lambda x: torch.zeros_like(x))]
)

model = PINNModel(input_dim=2, output_dim=1, layers=[64, 64, 64], activation="tanh")

trainer = Trainer(model, pde, lr=1e-3, device="cpu")
loss_history = trainer.fit(epochs=2000, n_collocation=1000)

Visualizer.plot_loss(loss_history)
Visualizer.plot_1d(model, pde, t=0.5)
Visualizer.plot_2d(model, pde, resolution=100)
```

---
