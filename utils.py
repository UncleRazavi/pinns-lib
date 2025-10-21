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
