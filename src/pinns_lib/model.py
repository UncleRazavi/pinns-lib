import math
import torch
import torch.nn as nn


class SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class PINNModel(nn.Module):
    """
    Fully-connected PINN with optional Fourier feature encoding.

    Fixes vs original:
    - Fourier matrix B registered as a buffer (moves with .to(device), saved in state_dict)
    - Correct Fourier projection scaling (no double 2π)
    - 'sin' activation implemented as nn.Module (works inside nn.Sequential)
    - Optional weight init for stability
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        layers=None,
        activation: str = "tanh",
        fourier_features: bool = False,
        num_frequencies: int = 10,
        fourier_scale: float = 1.0,
        init: str = "xavier",
    ):
        super().__init__()
        if layers is None:
            layers = [64, 64, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fourier_features = bool(fourier_features)
        self.num_frequencies = int(num_frequencies)
        self.fourier_scale = float(fourier_scale)

        net_input_dim = input_dim

        # Fourier features: register B as a buffer so it is moved with model.to(device)
        if self.fourier_features:
            B = torch.randn(input_dim, self.num_frequencies) * self.fourier_scale
            self.register_buffer("B", B)
            net_input_dim = 2 * self.num_frequencies  # sin+cos

        layers_dim = [net_input_dim] + list(layers) + [output_dim]
        modules = []
        for i in range(len(layers_dim) - 1):
            lin = nn.Linear(layers_dim[i], layers_dim[i + 1])
            modules.append(lin)
            if i < len(layers_dim) - 2:
                modules.append(self.get_activation(activation))
        self.net = nn.Sequential(*modules)

        self._init_weights(init)

    def _init_weights(self, mode: str):
        mode = (mode or "").lower()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if mode == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif mode == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                # Bias init
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_activation(self, name: str) -> nn.Module:
        name = (name or "").lower()
        if name == "tanh":
            return nn.Tanh()
        if name == "relu":
            return nn.ReLU()
        if name == "sin":
            return SinActivation()
        raise ValueError(f"Unknown activation {name}")

    def encode_fourier(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, input_dim)
        returns: (N, 2*num_frequencies)
        """
        # Ensure float
        x = x.to(dtype=torch.float32)
        # Projection: (N,input_dim) @ (input_dim,K) => (N,K)
        x_proj = (2.0 * math.pi) * (x @ self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier_features:
            x = self.encode_fourier(x)
        return self.net(x)
