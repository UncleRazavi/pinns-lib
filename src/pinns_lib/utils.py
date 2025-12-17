import torch
import numpy as np


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_error(u_pred, u_true):
    u_pred = np.asarray(u_pred)
    u_true = np.asarray(u_true)
    denom = np.linalg.norm(u_true) + 1e-12
    return np.linalg.norm(u_pred - u_true) / denom


def save_model(model, path="pinn_model.pth"):
    torch.save(model.state_dict(), path)


def load_model(model, path="pinn_model.pth", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def linear_schedule(start: int, end: int, frac: float) -> int:
    return int(start + (end - start) * frac)


def energy_regularizer(model, problem, x_int, t_int):
    inp = torch.cat([x_int, t_int], dim=1)
    u = model(inp)
    return torch.mean(u ** 2)


def latin_hypercube_sampler(n_samples: int, domain: dict):
    x_min, x_max = domain["x"]
    t_min, t_max = domain["t"]

    rng = np.random.default_rng()
    cut = np.linspace(0, 1, n_samples + 1)

    u1 = rng.uniform(size=n_samples)
    u2 = rng.uniform(size=n_samples)

    a = cut[:n_samples]
    b = cut[1:n_samples + 1]

    x = a + u1 * (b - a)
    t = a + u2 * (b - a)

    x = x[rng.permutation(n_samples)]
    t = t[rng.permutation(n_samples)]

    x = x_min + x * (x_max - x_min)
    t = t_min + t * (t_max - t_min)

    return torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)


def sobol_sampler(n_samples: int, domain: dict, scramble: bool = True):
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
        return latin_hypercube_sampler(n_samples, domain)


def adaptive_resample(model, problem, candidate_x, candidate_t, n_new: int):
    """
    Fix: do NOT wrap in torch.no_grad() because PDE residual needs autograd.
    We compute residuals with grad enabled, then detach to score.
    """
    X = candidate_x.view(-1, 1).detach().clone().requires_grad_(True)
    T = candidate_t.view(-1, 1).detach().clone().requires_grad_(True)

    model_was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            res = problem.pde_residual(model, X, T)
            scores = torch.abs(res).detach().cpu().numpy().ravel()
    finally:
        model.train(model_was_training)

    idx = np.argsort(-scores)[: int(n_new)]
    x_new = candidate_x.view(-1)[idx]
    t_new = candidate_t.view(-1)[idx]
    return x_new.to(dtype=torch.float32), t_new.to(dtype=torch.float32)
