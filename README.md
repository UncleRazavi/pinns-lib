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

## Training & Optimization Enhancements

This release adds several trainer features to support more advanced workflows:

- Curriculum Learning: progressively increase training difficulty (collocation / BC / IC counts) via a schedule.
- Physics-Based Regularization: attach problem-specific regularizers (energy conservation, monotonicity penalties) and weight them in the loss.
- Hybrid Solvers: provide a hook to include additional supervision or preconditioning terms (e.g., coarse finite-difference or FEM solutions).
- Transfer Learning: helpers to pretrain on simpler PDEs and fine-tune on a target problem.

These features are available through the `Trainer` API. See short examples below.

### Curriculum learning example

```python
from trainer import Trainer
from utils import linear_schedule

# create model and pde as usual
trainer = Trainer(model, pde, lr=1e-3, device="cpu")

curriculum = {
    "epochs": 2000,
    "n_collocation": (200, 2000),
    "n_bc": (20, 200),
    "n_ic": (20, 200),
    # optional custom schedule: (start,end,frac)->int
    # "schedule": lambda s,e,frac: int(s + (e-s)*(frac**0.7))
}

loss_history = trainer.fit(epochs=2000, verbose=100, curriculum=curriculum)
```

### Physics-based regularizer example

```python
from utils import energy_regularizer

trainer.physics_reg = energy_regularizer  # signature: (model, problem, x_int, t_int) -> torch scalar
loss_history = trainer.fit(epochs=1000, physics_reg_weight=1e-3)
```

### Hybrid solver hook example

```python
def hybrid_loss(model):
    # compute supervision from a coarse FD solver or preconditioner
    return torch.tensor(0.0, device="cpu")

trainer.hybrid_loss = hybrid_loss
loss_history = trainer.fit(epochs=1000, hybrid_weight=0.1)
```

### Transfer learning example

```python
# pretrain on a simpler PDEProblem
trainer.pretrain(problem_simple, epochs=200, lr=1e-3, n_collocation=500)

# fine-tune on the main problem
trainer.fine_tune(epochs=500, lr=1e-4, n_collocation=2000)
```

## Adaptive Loss Weighting & Sampling

Two additional enhancements are included:

- Adaptive Loss Weighting: an optional GradNorm-style update that adjusts component weights (PDE / IC / BC / hybrid / reg) during training. Enable by passing `curriculum={'adaptive_weighting': True, ...}` to `Trainer.fit`. The trainer prints the current adaptive weights when verbose.

- Domain Sampling Strategies: built-in samplers in `utils.py`:
    - `latin_hypercube_sampler(n, domain)` — Latin Hypercube sampling.
    - `sobol_sampler(n, domain)` — Sobol sequence (falls back to LHS if unavailable).
    - `adaptive_resample(model, problem, candidate_x, candidate_t, n_new)` — pick new points where PDE residual is largest.

Usage example with custom sampler and adaptive weighting:

```python
from utils import sobol_sampler

trainer.sampler = lambda n, domain: sobol_sampler(n, domain)

curriculum = {
        'epochs': 1000,
        'n_collocation': (500, 2000),
        'adaptive_weighting': True
}

loss_history = trainer.fit(epochs=1000, curriculum=curriculum, verbose=100)
```



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

This example matches the structure used in the repository. Run it from the repository root.

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
## Physics-Informed Neural Networks (PINNs) – Mathematical Background

Physics-Informed Neural Networks (PINNs) are a class of neural networks designed to **solve partial differential equations (PDEs)** by incorporating the physics (the governing equations) directly into the training process.

### 1. PDE Formulation

Consider a PDE defined over a domain $$\( \Omega \times [0, T] \)$$:

$$\[
\mathcal{N}[u(x,t); \lambda] = 0, \quad x \in \Omega, \, t \ in [0,T]\
\]$$

with:
- $$\( u(x,t) \)$$ — the unknown solution function
- $$\( \mathcal{N}[\cdot] \)$$ — differential operator (e.g., $$\( u_t + a u_x - \nu u_{xx} \))$$
- $$\( \lambda \)$$ — PDE coefficients/parameters
- Boundary and initial conditions:


$$
\[
u(x,0) = u_0(x), \quad u|_{\partial \Omega} = g(x,t)
\]
$$

---

### 2. PINN Approach

A PINN approximates $$\( u(x,t) \)$$ with a neural network:

$$\[
u_\theta(x,t) \approx u(x,t)
\]$$

where $$\( \theta \)$$ are the trainable parameters (weights and biases) of the network.

Instead of supervised labels, the **loss function is defined by the PDE itself**:

1. **PDE Residual Loss**:  

$$\[
\mathcal{L}_\text{PDE} = \frac{1}{N_f} \sum_{i=1}^{N_f} \big| \mathcal{N}[u_\theta(x_i, t_i)] \big|^2
\]$$

where $$\( (x_i, t_i) \)$$ are randomly sampled **collocation points** in the domain.

2. **Initial Condition Loss**:  

$$\[
\mathcal{L}_\text{IC} = \frac{1}{N_0} \sum_{i=1}^{N_0} \big| u_\theta(x_i,0) - u_0(x_i) \big|^2
\]$$

3. **Boundary Condition Loss**:  

$$\[
\mathcal{L}_\text{BC} = \frac{1}{N_b} \sum_{i=1}^{N_b} \big| u_\theta(x_i, t_i) - g(x_i, t_i) \big|^2
\]$$

The **total loss** minimized during training is:

$$\[
\mathcal{L}_\text{total} = \mathcal{L}_\text{PDE} + \mathcal{L}_\text{IC} + \mathcal{L}_\text{BC}
\]$$

---

### 3. How it works

1. Randomly sample collocation points in the domain and boundary/initial points.  
2. Forward pass through the network to predict $$\( u_\theta(x,t) \)$$.  
3. Use **automatic differentiation** to compute derivatives $$(\( u_t, u_x, u_{xx}, \dots \))$$.  
4. Compute residuals according to the PDE and IC/BC.  
5. Backpropagate the **total loss** to update network parameters.  
6. Repeat until convergence.

---

### 4. Advantages of PINNs

- Mesh-free method: works for arbitrary domains and high-dimensional PDEs.  
- Can handle noisy or sparse data combined with physics constraints.  
- Easily incorporates parametric PDEs and inverse problems.  
- No need for discretization like finite difference or finite element methods.

---

### References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686–707.
- Karniadakis, G. E., et al. (2021). *Physics-informed machine learning*. Nature Reviews Physics, 3, 422–440.


## License

This project is licensed under the [MIT License](LICENSE) — see the file for details.


