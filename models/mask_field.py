import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedding:
    def __init__(self,
                 n_freq=4,
                 log_sample=True,
                 input_dim=3,
                 include_input=True):
        self.embed_fns = []
        self.output_dim = 0

        # Identity mapping
        if include_input:
            self.embed_fns.append(lambda x: x)
            self.output_dim += input_dim

        # Fourier embedding
        if log_sample:
            freq_bands = 2.**torch.linspace(0., n_freq-1, steps=n_freq)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(n_freq-1), steps=n_freq)
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq : torch.sin(freq * x))
            self.embed_fns.append(lambda x, freq=freq : torch.cos(freq * x))
            self.output_dim += 2 * input_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class MaskField(nn.Module):
    def __init__(self,
                 n_layer=8,
                 n_dim=256,
                 input_dim=3,
                 skips=[4],
                 mask_dim=2,
                 mask_act='softmax',
                 point_embed=False):
        super().__init__()
        self.skips = skips
        self.mask_dim = mask_dim

        if point_embed:
            self.point_embed = FourierEmbedding(input_dim=input_dim)
            input_dim = self.point_embed.output_dim
        else:
            self.point_embed = None

        self.point_fc = nn.ModuleList()
        self.point_fc.append(nn.Linear(input_dim, n_dim))
        for l in range(n_layer - 1):
            if l in skips:
                self.point_fc.append(nn.Linear(n_dim + input_dim, n_dim))
            else:
                self.point_fc.append(nn.Linear(n_dim, n_dim))
        self.mask_fc = nn.Linear(n_dim, mask_dim)

        # Output activations
        act_fns = {'identity': lambda x: x,
                   'sigmoid': lambda x: torch.sigmoid(x),
                   'softmax': lambda x: F.softmax(x, dim=1)}
        self.mask_act = act_fns[mask_act]

    def forward(self, point):
        if self.point_embed is not None:
            point = self.point_embed(point)
        else:
            point = point

        h = point
        for l in range(len(self.point_fc)):
            h = self.point_fc[l](h)
            h = F.relu(h)
            if l in self.skips:
                h = torch.cat([point, h], 1)
        
        mask = self.mask_fc(h)
        mask = self.mask_act(mask)
        return mask


if __name__ == '__main__':
    torch.manual_seed(0)

    model = MaskField()
    print(model.mask_fc.bias)

    point = torch.randn(8, 32, 3)
    point = point.reshape(-1, 3)

    mask = model(point)
    print(mask)
    print(mask.shape)
