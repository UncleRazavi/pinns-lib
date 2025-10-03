import torch
import torch.nn as nn
import math

class PINNModel(nn.Module):
    
    def __init__(self, input_dim=2, output_dim=1, 
                 layers=[64, 64, 64], 
                 activation="tanh", 
                 fourier_features=False, num_frequencies=10):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fourier_features = fourier_features
        self.num_frequencies = num_frequencies

        # Handle Fourier features
        if fourier_features:
            self.B = torch.randn(input_dim, num_frequencies) * 2.0 * math.pi
            input_dim = 2 * num_frequencies  # sin + cos expansion

        # Build fully connected network
        layers_dim = [input_dim] + layers + [output_dim]
        net = []
        for i in range(len(layers_dim)-1):
            net.append(nn.Linear(layers_dim[i], layers_dim[i+1]))
            if i < len(layers_dim)-2:  # no activation in last layer
                net.append(self.get_activation(activation))
        self.net = nn.Sequential(*net)

    def get_activation(self, name):
        if name == "tanh":
            return nn.Tanh()
        elif name == "relu":
            return nn.ReLU()
        elif name == "sin":
            return torch.sin
        else:
            raise ValueError(f"Unknown activation {name}")

    def encode_fourier(self, x):
        """Map input to Fourier feature space."""
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x):
        if self.fourier_features:
            x = self.encode_fourier(x)
        return self.net(x)
