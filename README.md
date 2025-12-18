# PINNs-Lib

  

**PINNs-Lib** is a lightweight Python library for solving **partial differential equations (PDEs)** using **Physics-Informed Neural Networks (PINNs)**.

  

The goal of this project is to provide a **minimal, readable, and extensible** PINN implementation that is easy to understand, modify, and experiment with—without heavy abstractions or unnecessary dependencies.

  

---

  

## Features

  

- Simple API to define PDE problems, domains, and conditions

- Fully-connected PINN models with optional Fourier features

- Automatic PDE residual computation using PyTorch autograd

- Flexible training loop with modern PINN enhancements

- Built-in visualization utilities (loss curves, 1D and 2D solutions)

- Small, runnable examples for common PDEs (heat equation, Burgers’ equation)

  

---

  
  

---

  

## Core Components

  

### PDEProblem

Defines the PDE, domain, coefficients, and boundary/initial conditions.

  

- Supports first- and second-order PDEs

- Handles PDE residual, IC loss, and Dirichlet BC loss

- Clean separation between physics definition and training logic

  

### PINNModel

A fully-connected neural network used to approximate the PDE solution.

  

- Configurable depth and width

- Supports `tanh`, `relu`, and `sin` activations

- Optional Fourier feature encoding for high-frequency solutions

- Stable weight initialization

  

### Trainer

Manages sampling, loss computation, optimization, and training.

  

Supports:

- Curriculum learning (progressive increase in difficulty)

- Adaptive loss weighting (GradNorm-style)

- Physics-based regularization

- Hybrid supervision hooks

- Transfer learning (pretrain / fine-tune)

- Custom domain samplers (Sobol, LHS, adaptive resampling)

  

### Visualizer

Simple plotting utilities built on matplotlib.

  

- Training loss (log scale)

- 1D solution slices at fixed time

- 2D space–time solution fields

  

### utils

Helper utilities:

- Random seed control

- Sampling strategies

- Regularizers

- Error metrics

- Save / load helpers

  

---

  

## Installation

  

Create a virtual environment (recommended), then install dependencies:

  

```bash

pip install -r requirements.txt

```

## Required dependencies:

  

- PyTorch

  

- NumPy

  

- Matplotlib

  

## Minimal Example — Heat Equation

```python

import torch

from pinns_lib import PDEProblem, PINNModel, Trainer, Visualizer, set_seed

  

set_seed(123)

  

pde = PDEProblem(

    order=2,

    coefficients={"nu": 0.01},

    domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},

    initial_condition=lambda x: torch.sin(torch.pi * x),

    boundary_conditions=[

        ("dirichlet", 0.0, lambda x: torch.zeros_like(x)),

        ("dirichlet", 1.0, lambda x: torch.zeros_like(x)),

    ],

)

  

model = PINNModel(

    input_dim=2,

    output_dim=1,

    layers=[64, 64, 64],

    activation="tanh",

)

  

trainer = Trainer(model, pde, lr=1e-3)

loss_history = trainer.fit(epochs=2000, n_collocation=1000)

  

Visualizer.plot_loss(loss_history)

Visualizer.plot_1d(model, pde, t=0.5)

Visualizer.plot_2d(model, pde, resolution=100)

```

# Advanced Training Features

## Curriculum Learning

- gradually increase the number of training points during training:

```python

curriculum = {

    "epochs": 2000,

    "n_collocation": (200, 2000),

    "n_bc": (20, 200),

    "n_ic": (20, 200),

}

  

trainer.fit(epochs=2000, curriculum=curriculum)

```

## Adaptive Loss Weighting

- Automatically balances PDE, IC, BC, hybrid, and regularization losses:

``` python

curriculum = {

    "epochs": 1000,

    "n_collocation": (500, 2000),

    "adaptive_weighting": True,

}

  

trainer.fit(epochs=1000, curriculum=curriculum)

```

## Physics-Based Regularization

- Attach a custom regularizer:

``` python

from pinns_lib.utils import energy_regularizer

  

trainer.physics_reg = energy_regularizer

trainer.fit(epochs=1000, physics_reg_weight=1e-3)

```

## Hybrid Supervision

  

- Add auxiliary supervision (e.g. from a coarse solver):

```python

def hybrid_loss(model):

    return torch.tensor(0.0)

  

trainer.hybrid_loss = hybrid_loss

trainer.fit(epochs=1000, hybrid_weight=0.1)

```

  

## Transfer Learning

- Pretrain on a simpler PDE, then fine-tune:

  

```python

trainer.pretrain(problem_simple, epochs=200, lr=1e-3)

trainer.fine_tune(epochs=500, lr=1e-4)

```

## Sampling Strategies

  

# Available samplers in utils.py:

  

- Latin Hypercube sampling

  

- Sobol sequence sampling

  

- Adaptive residual-based resampling

  

### 1. PDE Formulation

Consider a PDE defined over a domain $$( \Omega \times [0, T] )$$:

$$[
\mathcal{N}[u(x,t); \lambda] = 0, \quad x \in \Omega, \, t \ in \ [0,T]\
]$$

with:
- $$( u(x,t) )$$  the unknown solution function
- $$( \mathcal{N}[\cdot] )$$ differential operator (e.g., $$( u_t + a u_x - \nu u_{xx} ))$$
- $$( \lambda )$$  PDE coefficients/parameters
- Boundary and initial conditions:


$$
[
u(x,0) = u_0(x), \quad u|_{\partial \Omega} = g(x,t)
]
$$

---

### 2. PINN Approach

A PINN approximates $$( u(x,t) )$$ with a neural network:

$$[
u_\theta(x,t) \approx u(x,t)
]$$

where $$( \theta )$$ are the trainable parameters (weights and biases) of the network.

Instead of supervised labels, the **loss function is defined by the PDE itself**:

1. **PDE Residual Loss**:  

$$[
\mathcal{L}_\text{PDE} = \frac{1}{N_f} \sum_{i=1}^{N_f} \big| \mathcal{N}[u_\theta(x_i, t_i)] \big|^2
]$$

where $$( (x_i, t_i) )$$ are randomly sampled **collocation points** in the domain.

2. **Initial Condition Loss**:  

$$[
\mathcal{L}_\text{IC} = \frac{1}{N_0} \sum_{i=1}^{N_0} \big| u_\theta(x_i,0) - u_0(x_i) \big|^2
]$$

3. **Boundary Condition Loss**:  

$$[
\mathcal{L}_\text{BC} = \frac{1}{N_b} \sum_{i=1}^{N_b} \big| u_\theta(x_i, t_i) - g(x_i, t_i) \big|^2
]$$

The **total loss** minimized during training is:

$$[
\mathcal{L}_\text{total} = \mathcal{L}_\text{PDE} + \mathcal{L}_\text{IC} + \mathcal{L}_\text{BC}
]$$

---

  

### References - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686–707. - Karniadakis, G. E., et al. (2021). *Physics-informed machine learning*. Nature Reviews Physics, 3, 422–440.


## License

This project is licensed under the [MIT License](LICENSE) — see the file for details.

