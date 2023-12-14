import torch
import torch.nn as nn
from functorch import vmap, jacrev
import numpy as np
import einops


try:
    from .base_network import BaseMLP, SineActivation, PositionEncoder
except ImportError:
    from base_network import BaseMLP, SineActivation, PositionEncoder


def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


class VelocityAABB(nn.Module):

    def __init__(self, vel_net, eps=-0.03):
        super(VelocityAABB, self).__init__()
        self.vel_net = vel_net
        self.eps = eps

    def forward(self, xt):
        vel = torch.zeros_like(xt[..., :3])
        pts = xt[..., :3]
        mask_outbbox = ((pts < -1 + self.eps) | (pts > 1 - self.eps)).any(dim=-1)
        vel[~mask_outbbox] = self.vel_net.get_vel(xt[~mask_outbbox])[..., :3]
        return vel


class VelocityAABBSur(nn.Module):

    def __init__(self, vel_net, aabb, surround):
        super(VelocityAABBSur, self).__init__()
        self.vel_net = vel_net
        self.aabb = aabb
        self.surround = surround
        # normalize surround by aabb as bounds
        self.bounds = (surround - aabb[0]) * 2 / (aabb[1] - aabb[0]) - 1

    def forward(self, xt):
        vel = torch.zeros_like(xt[..., :3])
        pts = xt[..., :3]
        mask_outbbox = ((pts < self.bounds[0]) | (pts > self.bounds[1])).any(dim=-1)
        vel[~mask_outbbox] = self.vel_net.get_vel(xt[~mask_outbbox])[..., :3]
        return vel


class VelBasis(nn.Module):
    def __init__(self):
        super(VelBasis, self).__init__()
        encode_dim = 3
        in_dim = 4 + 4 * 2 * encode_dim
        hidden_dim = 128
        self.weight_net = nn.Sequential(PositionEncoder(encode_dim), nn.Linear(in_dim, hidden_dim), nn.SiLU())
        for i in range(4):
            self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, 6)))
        self.a_weight_net = nn.Sequential(PositionEncoder(encode_dim), nn.Linear(in_dim, hidden_dim), nn.ReLU())
        for i in range(4):
            self.a_weight_net.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.a_weight_net.append(nn.Sequential(nn.Linear(hidden_dim, 6)))

    def forward(self, xt):
        v_basis, a_basis = self.get_basis(xt)
        weights = self.weight_net(xt)
        a_weights = self.a_weight_net(xt)
        v = torch.einsum('...ij,...i->...j', v_basis, weights)
        a = torch.einsum('...ij,...i->...j', a_basis, a_weights)
        return torch.cat([v, a], dim=-1)

    def get_vel(self, xt):
        v_basis, _ = self.get_basis(xt)
        weights = self.weight_net(xt)
        v = torch.einsum('...ij,...i->...j', v_basis, weights)
        return v

    def get_basis(self, xt):
        x, y, z = xt[..., 0], xt[..., 1], xt[..., 2]
        zeros = xt[..., -1] * 0.
        ones = zeros + 1.

        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)
        b4 = torch.stack([zeros, z, -y], dim=-1)
        b5 = torch.stack([-z, zeros, x], dim=-1)
        b6 = torch.stack([y, -x, zeros], dim=-1)

        a4 = torch.stack([zeros, -y, -z], dim=-1)
        a5 = torch.stack([-x, zeros, -z], dim=-1)
        a6 = torch.stack([-x, -y, zeros], dim=-1)
        return torch.stack([b1, b2, b3, b4, b5, b6], dim=-2), torch.stack([b1, b2, b3, a4, a5, a6], dim=-2)

