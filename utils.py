import torch
import numpy as np

def set_seed(seed=42):
    """Set all seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def l2_error(u_pred, u_true):
    """L2 error between predicted and true solution."""
    return np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)

def save_model(model, path="pinn_model.pth"):
    """Save model weights."""
    torch.save(model.state_dict(), path)

def load_model(model, path="pinn_model.pth", device="cpu"):
    """Load model weights."""
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def linear_schedule(start: int, end: int, frac: float) -> int:
    """Simple linear schedule for curriculum interpolation.

    Returns integer interpolated value between start and end.
    """
    return int(start + (end - start) * frac)


def energy_regularizer(model, problem, x_int, t_int):

    """A small example of a physics-based regularizer.
    """
    # compute u^2 integrated over sampled points as a proxy energy
    inp = torch.cat([x_int, t_int], dim=1)
    u = model(inp)
    energy = torch.mean(u ** 2)
    return energy


def latin_hypercube_sampler(n_samples: int, domain: dict):
    """Generate Latin Hypercube samples in 1D x and 1D t domain.

    Returns (x_tensor, t_tensor) as float32 torch tensors of shape (n_samples,)
    """
    x_min, x_max = domain["x"]
    t_min, t_max = domain["t"]

    # LHS in 2 dims
    rng = np.random.default_rng()
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.uniform(size=n_samples)
    a = cut[:n_samples]
    b = cut[1:n_samples + 1]
    points = a + u * (b - a)
    # shuffle per dimension
    idx = rng.permutation(n_samples)
    x = points[idx]
    idx2 = rng.permutation(n_samples)
    t = points[idx2]

    x = x_min + x * (x_max - x_min)
    t = t_min + t * (t_max - t_min)

    return torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)


def sobol_sampler(n_samples: int, domain: dict, scramble: bool = True):
    """Generate Sobol sequence samples (uses torch.quasirandom.SobolEngine when available).

    Returns (x_tensor, t_tensor).
    """
    try:
        from torch.quasirandom import SobolEngine
        eng = SobolEngine(dimension=2, scramble=scramble)
        samp = eng.draw(n_samples)
        x_min, x_max = domain["x"]
        t_min, t_max = domain["t"]
        x = x_min + samp[:, 0] * (x_max - x_min)
        t = t_min + samp[:, 1] * (t_max - t_min)
        return x.to(dtype=torch.float32), t.to(dtype=torch.float32)
    except Exception:
        # fallback to LHS if Sobol is not available
        return latin_hypercube_sampler(n_samples, domain)


def adaptive_resample(model, problem, candidate_x, candidate_t, n_new):
    """Adaptive resampling: score candidate points by PDE residual and sample n_new high-residual points.

    candidate_x, candidate_t are torch tensors shape (M,1) or (M,)
    Returns (x_new, t_new) tensors of shape (n_new,)
    """
    # Ensure shapes (M,1)
    X = candidate_x.view(-1, 1).detach().requires_grad_(True)
    T = candidate_t.view(-1, 1).detach().requires_grad_(True)
    with torch.no_grad():
        res = problem.pde_residual(model, X, T)
        scores = torch.abs(res).cpu().numpy().ravel()

    # sample top-n_new by residual
    idx = np.argsort(-scores)[:n_new]
    x_new = candidate_x.view(-1)[idx]
    t_new = candidate_t.view(-1)[idx]
    return x_new.to(dtype=torch.float32), t_new.to(dtype=torch.float32)
