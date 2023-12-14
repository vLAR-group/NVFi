import numpy as np
import torch
import torch.nn as nn


class SineActivation(nn.Module):
    def __init__(self, a=1., adaptive=True):
        super(SineActivation, self).__init__()

        a = torch.tensor([a], dtype=torch.float32)
        if adaptive:
            self.register_parameter('a', nn.Parameter(a))
        else:
            self.register_buffer('a', a)

    def forward(self, x):
        return torch.sin(self.a * x)


class PositionEncoder(nn.Module):
    
    def __init__(self, encode_dim, log_sampling=True):
        super(PositionEncoder, self).__init__()

        self.encode_dim = encode_dim
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0,
                self.encode_dim - 1,
                self.encode_dim,
                dtype=torch.float32
            )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (self.encode_dim - 1),
                self.encode_dim,
                dtype=torch.float32
            )
        self.register_buffer('frequency_bands', frequency_bands)
        
    def forward(self, x):

        encoding = [x]

        for freq in self.frequency_bands:
            encoding.append(torch.sin(x * freq))
            encoding.append(torch.cos(x * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


class BaseMLP(nn.Module):

    def __init__(self, input_dim=3, output_dim=3, encode_dim=10, layers=8, hidden=256, skip_in=(4,), bias=1.0,
                 log_sampling=True, geometric_init=True, siren=True):
        super(BaseMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encode_dim = encode_dim
        self.skip_in = skip_in if skip_in is not None else ()
        self.encode_input_dim = (2 * encode_dim + 1) * input_dim

        if siren:
            activation = SineActivation
        else:
            activation = nn.ReLU

        if encode_dim > 0:
            self.pos_encoder = PositionEncoder(encode_dim, log_sampling)
        else:
            self.pos_encoder = lambda x: x

        self.layer_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.encode_input_dim, hidden),
                activation()
            )
        ])
        for l in range(1, layers):
            if l in self.skip_in:
                c_in = self.encode_input_dim + hidden
            else:
                c_in = hidden
            lin = nn.Linear(c_in, hidden)
            if geometric_init:
                if l == layers - 1:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(c_in), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif l == 0: # multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, input_dim:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :input_dim], 0.0, np.sqrt(2) / np.sqrt(hidden))
                elif l + 1 in self.skip_in: # multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(hidden))
                    torch.nn.init.constant_(lin.weight[:, :self.encode_input_dim], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(hidden))
            self.layer_list.append(nn.Sequential(lin, activation()))

        lin = nn.Linear(hidden, output_dim)
        torch.nn.init.constant_(lin.bias, 0.0)
        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(hidden))
        self.layer_list.append(lin)

    def forward(self, x):
        pos = self.pos_encoder(x)

        z = pos
        for l, layer in enumerate(self.layer_list):
            if l in self.skip_in:
                z = torch.cat([pos, z], dim=-1)
            z = layer(z)

        return z
