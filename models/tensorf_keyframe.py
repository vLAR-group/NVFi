#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .tensorf_base import TensorBase
from .base_network import PositionEncoder, SineActivation, BaseMLP
from .velocity_field import *

from .tensorf_model_utils import (
    AlphaGridMask,
    raw2alpha,
    alpha2weights,
    scale_shift_color_all,
    scale_shift_color_one,
    transform_color_all,
    transform_color_one,
    DensityRender,
    DensityLinearRender,
    DensityFourierRender,
    RGBIdentityRender,
    RGBtFourierRender,
    RGBtLinearRender
)


class TensorVMKeyframeTimeKplane(TensorBase):
    def __init__(self, aabb, gridSize, device, near_far, cfg):
        self.matModeSpace = [[0, 1], [0, 2], [1, 2]]
        self.matModeTime = [[2, 3], [1, 3], [0, 3]]
        self.num_keyframes = cfg.num_keyframes
        self.tmax = cfg.tmax
        # self.dt = cfg.dt
        self.dt = 0.51
        self.time_scale_factor = self.tmax / (self.num_keyframes - 1) if self.num_keyframes > 1 else 1

        # Number of outputs for color and density
        if cfg.shadingMode == "RGBtLinear":
            cfg.data_dim_color = (2) * 3
        elif cfg.shadingMode == "RGBtFourier":
            cfg.data_dim_color = (self.frames_per_keyframe * 2 + 1) * 3

        self.densityMode = cfg.densityMode

        if self.densityMode == "Density":
            self.data_dim_density = 1
        elif self.densityMode == "DensityLinear":
            self.data_dim_density = 2
        elif self.densityMode == "DensityFourier":
            self.data_dim_density = self.frames_per_keyframe * 2 + 1

        super().__init__(aabb, gridSize, device, near_far, cfg)

        if "MLP" in self.shadingMode:
            self.opt_group = {
                "color": [
                    self.density_plane_space,
                    self.density_plane_time,
                    self.app_plane_space,
                    self.app_plane_time,
                ],
                "color_impl": [self.basis_mat, self.basis_mat_density],
            }
        else:
            self.opt_group = {
                "color": [
                    self.density_plane_space,
                    self.density_plane_time,
                    self.app_plane_space,
                    self.app_plane_time,
                    self.basis_mat,
                    self.basis_mat_density,
                ],
            }

        if isinstance(self.renderModule, torch.nn.Module):
            if "MLP" in self.shadingMode:
                self.opt_group["color_impl"] += [self.renderModule]
            else:
                self.opt_group["color_impl"] = [self.renderModule]

        self.use_vel = cfg.use_vel
        if self.use_vel:
            self.vel_net = VelBasis()

            try:
                eps = cfg.eps
            except:
                eps = 0.03
            try:
                sur_x, sur_y, sur_z = [torch.tensor(cfg[sur]) for sur in ['sur_x', 'sur_y', 'sur_z']]
                surround = torch.stack([sur_x, sur_y, sur_z], dim=-1).to(self.device)
                self.vel = VelocityAABBSur(self.vel_net, self.aabb, surround)
                print('VelocityAABBSur')
            except:
                self.vel = VelocityAABB(self.vel_net, eps)
                print('VelocityAABB')

        self.mask_field = None

        if 'contract_ray' in cfg and cfg.contract_ray:
            self.contract_ray = True
        else:
            self.contract_ray = False

    def init_svd_volume(self, device):
        if self.fea2denseAct == 'softplus':
            self.density_plane_space, self.density_plane_time = self.init_one_svd_density(
                self.density_n_comp, self.gridSize, self.num_keyframes, 0.8, device
            )
        else:
            self.density_plane_space, self.density_plane_time = self.init_one_svd_density(
                self.density_n_comp, self.gridSize, self.num_keyframes, 0.5, device
            )

        self.app_plane_space, self.app_plane_time = self.init_one_svd(
            self.app_n_comp, self.gridSize, self.num_keyframes, 0.1, device
        )
        self.basis_mat = torch.nn.Linear(
            self.app_n_comp[0], self.app_dim, bias=False
        ).to(device)
        self.basis_mat_density = torch.nn.Linear(
            self.density_n_comp[0], self.data_dim_density, bias=False
        ).to(device)

    def init_one_svd(self, n_component, gridSize, numFrames, scale, device):
        plane_coef_space, plane_coef_time = [], []

        for i in range(len(self.matModeSpace)):
            mat_id_space_0, mat_id_space_1 = self.matModeSpace[i]
            mat_id_time_0, mat_id_time_1 = self.matModeTime[i]

            new_plane = nn.Parameter(torch.empty(1, n_component[i], gridSize[mat_id_space_1], gridSize[mat_id_space_0]))
            nn.init.uniform_(new_plane, a=0.1, b=0.5)
            plane_coef_space.append(scale * new_plane)

            plane_coef_time.append(
                torch.nn.Parameter(torch.ones(1, n_component[i], numFrames, gridSize[mat_id_time_0]))
            )

        return torch.nn.ParameterList(plane_coef_space).to(
            device
        ), torch.nn.ParameterList(plane_coef_time).to(device)

    def init_one_svd_density(self, n_component, gridSize, numFrames, scale, device):
        plane_coef_space, plane_coef_time = [], []

        for i in range(len(self.matModeSpace)):
            mat_id_space_0, mat_id_space_1 = self.matModeSpace[i]
            mat_id_time_0, mat_id_time_1 = self.matModeTime[i]

            if self.fea2denseAct == 'softplus':
                new_plane = nn.Parameter(
                    torch.empty(1, n_component[i], gridSize[mat_id_space_1], gridSize[mat_id_space_0])
                )
                nn.init.uniform_(new_plane, a=0.1, b=0.5)
                plane_coef_space.append(scale * new_plane)
                plane_coef_time.append(
                    torch.nn.Parameter(torch.ones(1, n_component[i], numFrames, gridSize[mat_id_time_0])
                    )
                )
            else:
                new_plane = nn.Parameter(
                    torch.empty(1, n_component[i], gridSize[mat_id_space_1], gridSize[mat_id_space_0])
                )
                nn.init.uniform_(new_plane, a=0.1, b=0.5)
                plane_coef_space.append(scale * new_plane)
                plane_coef_time.append(
                    torch.nn.Parameter(
                        torch.ones(1, n_component[i], numFrames, gridSize[mat_id_time_0]).clamp(1e-2, 1e8)
                    )
                )

        return torch.nn.ParameterList(plane_coef_space).to(
            device
        ), torch.nn.ParameterList(plane_coef_time).to(device)

    def density_L1(self):
        total = 0

        for idx in range(len(self.density_plane_space)):
            if self.density_plane_space[idx].shape[1] == 0:
                continue

            total = (
                    total
                    + torch.mean(torch.abs(self.density_plane_space[idx]))
                    + torch.mean(torch.abs(1 - self.density_plane_time[idx]))
                    # + torch.mean(torch.abs(self.app_plane_space[idx]))
                    # + torch.mean(torch.abs(self.app_plane_time[idx]))
            )

        return total

    def TV_loss_density(self, reg):
        total = 0

        for idx in range(len(self.density_plane_space)):
            if self.density_plane_space[idx].shape[1] == 0 or self.density_plane_time[idx].shape[1] == 0:
                continue

            total = (
                total + reg(self.density_plane_space[idx]) * 1e-2
                + ((reg(self.density_plane_time[idx], t=True) * 1e-2) if self.num_keyframes > 1 else 0)
            )

        return total

    def TV_loss_app(self, reg):
        total = 0

        for idx in range(len(self.app_plane_space)):
            if self.density_plane_space[idx].shape[1] == 0 or self.density_plane_time[idx].shape[1] == 0:
                continue

            total = (
                total + reg(self.app_plane_space[idx]) * 1e-2
                # + reg(self.app_plane_time[idx], t=True) * 1e-2
            )

        return total

    def compute_densityfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane_space = torch.stack(
            (
                xyz_sampled[:, self.matModeSpace[0]],
                xyz_sampled[:, self.matModeSpace[1]],
                xyz_sampled[:, self.matModeSpace[2]],
            )
        ).view(3, -1, 1, 2)

        coordinate_plane_time = torch.stack(
            (
                xyz_sampled[:, self.matModeTime[0]],
                xyz_sampled[:, self.matModeTime[1]],
                xyz_sampled[:, self.matModeTime[2]],
            )
        ).view(3, -1, 1, 2)

        plane_coef_space, plane_coef_time = 1., 1.

        for idx_plane, (plane_space, plane_time) in enumerate(
                zip(self.density_plane_space, self.density_plane_time)
        ):
            if self.density_plane_space[idx_plane].shape[1] == 0:
                continue

            cur_plane = F.grid_sample(
                plane_space, coordinate_plane_space[[idx_plane]], align_corners=True
            ).view(-1, xyz_sampled.shape[0])
            cur_time = F.grid_sample(
                plane_time, coordinate_plane_time[[idx_plane]], align_corners=True
            ).view(-1, xyz_sampled.shape[0])

            plane_coef_space = plane_coef_space * cur_plane
            plane_coef_time = plane_coef_time * cur_time

        if self.densityMode != "Density":
            return self.basis_mat_density((plane_coef_space * plane_coef_time).T)
        else:
            return torch.sum((plane_coef_space * plane_coef_time), dim=0).unsqueeze(-1)

    def compute_appfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane_space = torch.stack(
            (
                xyz_sampled[:, self.matModeSpace[0]],
                xyz_sampled[:, self.matModeSpace[1]],
                xyz_sampled[:, self.matModeSpace[2]],
            )
        ).view(3, -1, 1, 2)

        coordinate_plane_time = torch.stack(
            (
                xyz_sampled[:, self.matModeTime[0]],
                xyz_sampled[:, self.matModeTime[1]],
                xyz_sampled[:, self.matModeTime[2]],
            )
        ).view(3, -1, 1, 2)

        plane_coef_space, plane_coef_time = 1., 1.

        for idx_plane, (plane_space, plane_time) in enumerate(
                zip(self.app_plane_space, self.app_plane_time)
        ):
            if self.density_plane_space[idx_plane].shape[1] == 0:
                continue

            cur_plane = F.grid_sample(
                plane_space, coordinate_plane_space[[idx_plane]], align_corners=True
            ).view(-1, *xyz_sampled.shape[:1])
            cur_time = F.grid_sample(
                plane_time, coordinate_plane_time[[idx_plane]], align_corners=True
            ).view(-1, *xyz_sampled.shape[:1])

            plane_coef_space = plane_coef_space * cur_plane
            plane_coef_time = plane_coef_time * cur_time

        return self.basis_mat((plane_coef_space * plane_coef_time).T)

    def feature2density(self, density_features: torch.Tensor, x: Dict[str, torch.Tensor]):
        if self.densityMode == "Density":
            density_features = DensityRender(density_features, x)
        elif self.densityMode == "DensityLinear":
            density_features = DensityLinearRender(density_features, x)
        elif self.densityMode == "DensityFourier":
            density_features = DensityFourierRender(density_features, x)

        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)
        elif self.fea2denseAct == "relu_abs":
            return F.relu(torch.abs(density_features))

    @torch.no_grad()
    def up_sampling_VM(self, n_component, plane_coef_space, plane_coef_time, res_target, numFrames):

        for i in range(len(self.matModeSpace)):
            mat_id_space_0, mat_id_space_1 = self.matModeSpace[i]
            mat_id_time_0, mat_id_time_1 = self.matModeTime[i]

            if self.density_plane_space[i].shape[1] == 0:
                plane_coef_space[i] = torch.nn.Parameter(
                    plane_coef_space[i].data.new_zeros(1, n_component[i], res_target[mat_id_space_1],
                                                       res_target[mat_id_space_0]),
                )
                plane_coef_time[i] = torch.nn.Parameter(
                    plane_coef_time[i].data.new_zeros(1, n_component[i], numFrames, res_target[mat_id_time_0])
                )
            else:
                plane_coef_space[i] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef_space[i].data,
                        size=(res_target[mat_id_space_1], res_target[mat_id_space_0]),
                        mode="bilinear",
                        align_corners=True,
                    )
                )
                plane_coef_time[i] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef_time[i].data,
                        size=(numFrames, res_target[mat_id_time_0]),
                        mode="bilinear",
                        align_corners=True,
                    )
                )

        return plane_coef_space, plane_coef_time

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, new_keyframes):
        self.num_keyframes = new_keyframes
        self.app_plane_space, self.app_plane_time = self.up_sampling_VM(
            self.app_n_comp, self.app_plane_space, self.app_plane_time, res_target, self.num_keyframes
        )
        self.density_plane_space, self.density_plane_time = self.up_sampling_VM(
            self.density_n_comp,
            self.density_plane_space,
            self.density_plane_time,
            res_target,
            self.num_keyframes,
        )
        self.update_stepSize(res_target)
        print(f"upsamping to {res_target} time {new_keyframes}")

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200), transfer=False):

        alpha, dense_xyz = self.getDenseAlpha(gridSize, transfer=transfer)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(
            gridSize[::-1]
        )
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]
        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(
            f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100)
        )
        return new_aabb

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")

        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)

        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.matModeSpace)):
            mat_id_space_0, mat_id_space_1 = self.matModeSpace[i]
            mat_id_time_0, mat_id_time_1 = self.matModeTime[i]

            self.density_plane_space[i] = torch.nn.Parameter(
                self.density_plane_space[i].data[
                ...,
                t_l[mat_id_space_1]: b_r[mat_id_space_1],
                t_l[mat_id_space_0]: b_r[mat_id_space_0],
                ]
            )
            self.density_plane_time[i] = torch.nn.Parameter(
                self.density_plane_time[i].data[
                ..., :, t_l[mat_id_time_0]: b_r[mat_id_time_0]
                ]
            )
            self.app_plane_space[i] = torch.nn.Parameter(
                self.app_plane_space[i].data[
                ...,
                t_l[mat_id_space_1]: b_r[mat_id_space_1],
                t_l[mat_id_space_0]: b_r[mat_id_space_0],
                ]
            )
            self.app_plane_time[i] = torch.nn.Parameter(
                self.app_plane_time[i].data[
                ..., :, t_l[mat_id_time_0]: b_r[mat_id_time_0]
                ]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None, transfer=False):
        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ),
            -1,
        ).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[..., 0])

        # for t in np.linspace(0, self.tmax, self.num_keyframes):
        for t in np.linspace(0, 59, 60) / 60:
            cur_alpha = torch.zeros_like(alpha)
            times = torch.ones_like(dense_xyz[..., -1:]) * t
            if transfer:
                base_times = torch.zeros_like(times)
            else:
                base_times = times
            time_offset = times - base_times

            for i in range(gridSize[0]):
                cur_xyz = dense_xyz[i].view(-1, 3)
                cur_base_times = base_times[i].view(-1, 1)
                cur_times = times[i].view(-1, 1)
                cur_time_offset = time_offset[i].view(-1, 1)

                cur_xyzt = torch.cat([cur_xyz, cur_times], -1)
                cur_alpha[i] = self.compute_alpha(
                    cur_xyzt, self.stepSize, times=cur_times, time_offset=cur_time_offset, transfer=transfer
                ).view((gridSize[1], gridSize[2]))

            alpha = torch.maximum(alpha, cur_alpha)

        return alpha, dense_xyz

    def normalize_time_coord(self, time):
        # return (time * self.time_scale_factor + self.time_pixel_offset) * 2 - 1
        if self.num_keyframes == 1 or self.tmax == 0:
            return time * 0
        else:
            return time * 2 / self.tmax - 1

    def compute_alpha(self, xyzt_locs, length=0.01, times=None, time_offset=None, transfer=False):
        sigma = torch.zeros(xyzt_locs.shape[:-1], device=xyzt_locs.device)

        points, t = self.normalize_coord(xyzt_locs[..., :3]), xyzt_locs[..., -1:]

        time_scale_factor = self.tmax / (self.num_keyframes - 1) if self.num_keyframes > 1 else 1
        if transfer:
            base_times = torch.zeros_like(t)
        else:
            base_times = torch.round(
                (t / time_scale_factor).clamp(0.0, self.num_keyframes - 1)
            ) * time_scale_factor

        points_prev = self.integrate_pos(points, t, base_times)
        xyzt_sampled = torch.cat([points_prev, self.normalize_time_coord(base_times)], dim=-1)

        sigma_feature = self.compute_densityfeature(xyzt_sampled)
        sigma = self.feature2density(
            sigma_feature,
            {
                "num_keyframes": self.num_keyframes,
                "times": times,
                "time_offset": time_offset,
                "weights": torch.ones_like(times),
            },
        )

        alpha = 1 - torch.exp(-sigma * length).view(xyzt_locs.shape[:-1])

        return alpha

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [{'params': self.density_plane_space, 'lr': lr_init_spatialxyz},
                     {'params': self.density_plane_time, 'lr': lr_init_spatialxyz},
                     {'params': self.app_plane_space, 'lr': lr_init_spatialxyz},
                     {'params': self.app_plane_time, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr': lr_init_network},
                     {'params': self.basis_mat_density.parameters(), 'lr': lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        if self.use_vel:
            grad_vars += [{'params': self.vel.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def characteristic_loss(self, n_pts, t):
        points = torch.rand(int(n_pts), 3, device=self.device) * 2 -1
        if t > 0:
            t = torch.ones(int(n_pts), 1, device=self.device) * t
            time_scale_factor = self.tmax / (self.num_keyframes - 1) if self.num_keyframes > 1 else 1
            t = torch.round(
                (t / time_scale_factor).clamp(0.0, self.num_keyframes - 1)
            ) * time_scale_factor
        else:
            t = torch.ones(int(n_pts), 1, device=self.device) * self.tmax / (self.num_keyframes - 1)
        t0 = t * 0.
        with torch.no_grad():
            points0 = self.integrate_pos(points, t, t0)

        density_feat_t = self.compute_densityfeature(torch.cat([points, self.normalize_time_coord(t)], dim=-1))
        app_feat_t = self.compute_appfeature(torch.cat([points, self.normalize_time_coord(t)], dim=-1))

        density_feat_0 = self.compute_densityfeature(torch.cat([points0, self.normalize_time_coord(t0)], dim=-1))
        app_feat_0 = self.compute_appfeature(torch.cat([points0, self.normalize_time_coord(t0)], dim=-1))

        character_loss = torch.mean((density_feat_t - density_feat_0) ** 2) + torch.mean((app_feat_t - app_feat_0) ** 2)
        return character_loss

    def integrate_pos(self, pos_init, t, base_times):
        # dt_max = min(0.5 * self.tmax / (self.num_keyframes - 1), self.dt)
        dt_max = 0.5 * self.tmax / (self.num_keyframes - 1) if self.num_keyframes > 1 else 1
        # dt_max = self.dt
        dt_max = torch.ones_like(t) * dt_max

        # xyz_keyframe = self.vel_ode.advect(t, base_times, pos_init, dt_max[...,0])
        # return xyz_keyframe

        time_offset = t - base_times
        if self.training:
            xyz_prev = pos_init
        else:
            xyz_prev = pos_init.clone()

        t_curr = t
        unfinished = (time_offset.abs() > 0).squeeze(-1)
        while unfinished.any():
            # get time step
            dt = time_offset[unfinished].sign() * torch.minimum(time_offset[unfinished].abs(), dt_max[unfinished])
            # Runge-Kutta 2
            xyzt_prev = torch.cat([xyz_prev[unfinished], t_curr[unfinished]], dim=-1)
            velocity = self.vel(xyzt_prev)
            p_mid = xyz_prev[unfinished] - 0.5 * dt * velocity
            t_mid = t_curr[unfinished] - 0.5 * dt
            pt_mid = torch.cat([p_mid, t_mid], dim=-1)

            xyz_cur = xyz_prev[unfinished] - dt * self.vel(pt_mid)
            if isinstance(self.vel, VelocityAABBSur):
                mask_outbbox = ((xyz_cur < self.vel.bounds[0]) | (xyz_cur > self.vel.bounds[1])).any(dim=-1)
                xyz_cur[mask_outbbox] = xyz_prev[unfinished][mask_outbbox]
            xyz_prev[unfinished] = xyz_cur
            time_offset[unfinished] = time_offset[unfinished] - dt
            t_curr[unfinished] = t_curr[unfinished] - dt
            unfinished = (time_offset.abs() > 0).squeeze(-1)

        return xyz_prev

    def forward(self, t, ray_o, ray_d, white_bg=True, ndc_ray=False, N_samples=-1, transfer_vel=False):

        viewdirs = ray_d
        # sample points
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(ray_o, viewdirs, N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            if self.contract_ray:
                xyz_sampled, z_vals, ray_valid = self.sample_ray_contracted(ray_o, viewdirs, N_samples=N_samples)
                dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
                viewdirs_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
                dists = dists * viewdirs_norm
                viewdirs = viewdirs / viewdirs_norm
            else:
                xyz_sampled, z_vals, ray_valid = self.sample_ray(ray_o, viewdirs, N_samples=N_samples)
                dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        t = torch.ones_like(ray_o[..., -1:]) * t
        t = t.view(-1, 1, 1).expand(*xyz_sampled.shape[:-1], 1)

        xyz_sampled = self.normalize_coord(xyz_sampled)

        return self.render_pts(t, xyz_sampled, z_vals, ray_valid, dists, viewdirs,
                               white_bg=white_bg, transfer_vel=transfer_vel, ndc_ray=ndc_ray)

    def render_pts(self, t, xyz_sampled, z_vals, ray_valid, dists, viewdirs, white_bg=True, transfer_vel=False,
                   ndc_ray=False):

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        time_scale_factor = self.tmax / (self.num_keyframes - 1) if self.num_keyframes > 1 else 1
        if transfer_vel:
            base_times = torch.zeros_like(t)
            time_offset = t
        else:
            base_times = torch.round(
                (t / time_scale_factor).clamp(0.0, self.num_keyframes - 1)
            ) * time_scale_factor
            time_offset = t - base_times

        if self.alphaMask is not None and not self.training:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        xyzt_sampled = torch.cat(
            [
                xyz_sampled,
                self.normalize_time_coord(t),
            ],
            dim=-1,
        )

        sigma = torch.zeros(xyzt_sampled.shape[:-1], device=xyzt_sampled.device)
        rgb = torch.zeros((*xyzt_sampled.shape[:-1], 3), device=xyzt_sampled.device)
        if self.mask_field is not None:
            mask = torch.zeros((*xyzt_sampled.shape[:-1], self.mask_field.mask_dim), device=xyzt_sampled.device)
        else:
            mask = torch.zeros((*xyzt_sampled.shape[:-1], 3), device=xyzt_sampled.device)

        if ray_valid.any():
            if self.use_vel:
                # xyzt_unnorm = torch.cat([xyz_sampled, t], dim=-1)
                xyz_prev = xyz_sampled
                # xyz_next = xyz_sampled
                key = torch.isclose(t, base_times)
                not_key = (~key[..., 0]) & ray_valid
                if not_key.any():
                    # xyzt_prev = torch.cat([xyz_prev, t], dim=-1)
                    # velocity[not_key] = self.vel(xyzt_prev[not_key])
                    # p_mid = xyz_prev[not_key] - 0.5 * time_offset[not_key] * velocity[not_key]
                    # t_mid = base_times[not_key] + 0.5 * time_offset[not_key]
                    # pt_mid = torch.cat([p_mid, t_mid], dim=-1)
                    # xyz_prev[not_key] = xyz_prev[not_key] - time_offset[not_key] * self.vel(pt_mid)
                    xyz_prev[not_key] = self.integrate_pos(xyz_prev[not_key], t[not_key], base_times[not_key])
                xyzt_eval = torch.cat(
                    [
                        xyz_prev,
                        self.normalize_time_coord(base_times),
                    ],
                    dim=-1,
                )
                out_range = (base_times[...,0] == self.tmax) & not_key
                if out_range.any():
                    xyzt_sampled[out_range] = xyzt_eval[out_range]
            else:
                xyzt_eval = xyzt_sampled
            sigma_feature = self.compute_densityfeature(xyzt_eval[ray_valid])

            validsigma = self.feature2density(
                sigma_feature,
                {
                    "num_keyframes": self.num_keyframes,
                    "times": t,
                    "time_offset": time_offset
                }
            )
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            # app_features = self.compute_appfeature(xyzt_sampled[app_mask])
            app_features = self.compute_appfeature(xyzt_eval[app_mask])
            valid_rgbs = self.renderModule(
                # xyzt_sampled[...,:-1][app_mask],
                xyzt_eval[..., :-1][app_mask],
                viewdirs[app_mask],
                app_features,
                {
                    "num_keyframes": self.num_keyframes,
                    "times": t[app_mask],
                    "time_offset": time_offset[app_mask],
                },
            )
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (self.training and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        # with torch.no_grad():
        depth_map = torch.sum(weight * z_vals, -1) + (1. - acc_map) * self.near_far[1]
            # depth_map = depth_map + (1. - acc_map) * ray_d[..., -1]

        if self.mask_field is not None:
            if app_mask.any():
                valid_mask = self.mask_field(xyzt_eval[app_mask][..., :-1])
                mask[app_mask] = valid_mask
        mask_map = torch.sum(weight[..., None] * mask, -2)

        return rgb_map, depth_map, acc_map, weight, mask_map  # rgb, sigma, alpha, weight, bg_weight

