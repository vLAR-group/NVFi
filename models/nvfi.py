import einops
from einops.layers.torch import Rearrange
import numpy as np
import torch
import torch.nn as nn
from functorch import vmap, jacrev

import time
import tqdm

try:
    from .tensorf_keyframe import *
except ImportError:
    from tensorf_keyframe import *


class NVFi(nn.Module):

    def __init__(self, config, device, aabb, res_cur, near_far):
        super(NVFi, self).__init__()
        self.config = config.nvfi

        self.nvfi = eval(self.config.model_name)(
            aabb, res_cur, device, near_far=near_far, cfg=config.nvfi
        )

    def render_ray(self, t, ray_o, ray_d, white_bg=True, ndc_ray=False):
        return self.nvfi(t, ray_o, ray_d, white_bg, ndc_ray)

    def render_ray_transfer(self, t, ray_o, ray_d, white_bg=True, ndc_ray=False):
        return self.nvfi(t, ray_o, ray_d, white_bg, ndc_ray, transfer_vel=True)

    def update_nvfi_kwargs(self, kwargs):
        for k, v in kwargs.items():
            self.nvfi.__dict__[k] = v

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001, lr_init_velocity=0.001):
        grad_vars = self.nvfi.get_optparam_groups(lr_init_spatialxyz, lr_init_network)
        # grad_vars += [{'params': self.velocity_field.parameters(), 'lr': lr_init_velocity}]
        return grad_vars

    def get_vel_loss(self, n_pts=32768.):
        min_corner, max_corner = self.nvfi.aabb
        points = torch.rand(int(n_pts), 3, device=self.nvfi.device) * (max_corner - min_corner) + min_corner
        points = self.nvfi.normalize_coord(points)
        t = torch.rand(int(n_pts), 1, device=self.nvfi.device) 
        xyzt = torch.cat([points, t], dim=-1)

        # get the occupancy
        with torch.no_grad():
            time_scale_factor = self.nvfi.tmax / (self.nvfi.num_keyframes - 1)
            base_times = torch.round(
                (t / time_scale_factor).clamp(0.0, self.nvfi.num_keyframes - 1)
            ) * time_scale_factor

            points_prev = self.nvfi.integrate_pos(points, t, base_times)
            points_prev = torch.cat([points_prev, self.nvfi.normalize_time_coord(base_times)], dim=-1)
            sigma_feature = self.nvfi.compute_densityfeature(points_prev)
            sigma = self.nvfi.feature2density(sigma_feature, {})
            alpha = 1 - torch.exp(-sigma * 0.01 * 25)
            alpha[alpha >= self.nvfi.alphaMask_thres] = 1.
            alpha[alpha < self.nvfi.alphaMask_thres] = 0.

            xyzt = xyzt[alpha.squeeze() > 0.5]

        if xyzt.shape[0] == 0:
            return 0.

        def u_func(xyzt):
            u = self.nvfi.vel_net(xyzt)
            return u, u
        jac, u = vmap(jacrev(u_func, argnums=0, has_aux=True))(xyzt)
        vel, a = u[..., :3], u[..., 3:]

        # calculate the divergence
        divergence = jac[..., 0, 0] + jac[..., 1, 1] + jac[..., 2, 2]
        # calculate the transport equation
        transport = einops.einsum(jac[..., :3, :3], vel, '... o i, ... i -> ... o') + jac[..., :3, 3] - a

        loss = (
            torch.mean(divergence ** 2) * 5
            + torch.mean(transport ** 2) * 0.1
        )
        return loss
