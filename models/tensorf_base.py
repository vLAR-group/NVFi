#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from .tensorf_model_utils import cal_n_samples, N_to_reso

from .tensorf_model_utils import (
    AlphaGridMask,
    DensityRender,
    DensityLinearRender,
    DensityFourierRender,
    RGBRender,
    RGBIdentityRender,
    RGBtLinearRender,
    RGBtFourierRender,
    SHRender,
    raw2alpha,
    alpha2weights,
    positional_encoding
)


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super().__init__()
        self.opt_group = "color_impl"

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, kwargs):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_PE(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, pospe=6, featureC=128):
        super().__init__()
        self.opt_group = "color_impl"

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + inChanel  #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, kwargs):
        indata = [features, viewdirs, pts]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, featureC=128):
        super().__init__()
        self.opt_group = "color_impl"

        self.in_mlpC = (3 + 2 * viewpe * 3) + inChanel
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, kwargs):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, near_far, cfg):
        super(TensorBase, self).__init__()

        self.cfg = cfg
        self.device = device

        self.register_buffer('aabb', aabb.to(self.device))

        self.gridSize = gridSize

        self.step_ratio = cfg.step_ratio
        self.max_n_samples = cfg.max_n_samples
        self.near_far = near_far

        self.density_n_comp = cfg.density_n_comp
        self.app_n_comp = cfg.appearance_n_comp
        self.app_dim = cfg.app_dim

        self.density_shift = cfg.density_shift
        self.distance_scale = cfg.distance_scale
        self.alphaMask = None
        self.alphaMask_thres = cfg.alphaMask_thres
        self.rayMarch_weight_thres = cfg.rayMarch_weight_thres

        self.fea2denseAct = cfg.fea2denseAct

        ### Filtering parameters

        self.update_stepSize(self.gridSize)

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]

        self.init_svd_volume(self.device)

        self.shadingMode = cfg.shadingMode
        self.pos_pe = cfg.pos_pe
        self.view_pe = cfg.view_pe
        self.fea_pe = cfg.fea_pe
        self.featureC = cfg.featureC

        self.init_render_func(
            self.shadingMode,
            self.pos_pe,
            self.view_pe,
            self.fea_pe,
            self.featureC,
            self.device,
        )

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == "MLP_PE":
            self.renderModule = MLPRender_PE(
                self.app_dim, view_pe, pos_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea":
            self.renderModule = MLPRender_Fea(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP":
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == "SH":
            self.renderModule = SHRender
        elif shadingMode == "RGB":
            assert self.app_dim == 3
            self.renderModule = RGBRender
        elif shadingMode == "RGBIdentity":
            assert self.app_dim == 3
            self.renderModule = RGBIdentityRender
        elif shadingMode == "RGBtLinear":
            self.renderModule = RGBtLinearRender
        elif shadingMode == "RGBtFourier":
            self.renderModule = RGBtFourierRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = min(
            self.max_n_samples, int((self.aabbDiag / self.stepSize).item()) + 1
        )
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled):
        pass

    def compute_appfeature(self, xyz_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        kwargs = {
            "aabb": self.aabb,
            "gridSize": self.gridSize.tolist() if not isinstance(self.gridSize, list) else self.gridSize,
            "density_n_comp": self.density_n_comp,
            "appearance_n_comp": self.app_n_comp,
            "app_dim": self.app_dim,
            "density_shift": self.density_shift,
            "alphaMask_thres": self.alphaMask_thres,
            "fea2denseAct": self.fea2denseAct,
            "near_far": self.near_far,
            "step_ratio": self.step_ratio,
            "shadingMode": self.shadingMode,
            "pos_pe": self.pos_pe,
            "view_pe": self.view_pe,
            "fea_pe": self.fea_pe,
            "featureC": self.featureC,
            "num_keyframes": self.num_keyframes
        }
        if self.alphaMask is not None:
            kwargs |= {"alphaMask_grid": self.alphaMask.gridSize}
        return kwargs

    def sample_ray_ndc(self, rays_o, rays_d, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)

        if self.training:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )
        return rays_pts, interpx, ~mask_outbbox

    def valid_mask(self, rays_pts):
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )
        return ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        if ((self.aabb[0] <= rays_o) & (rays_o <= self.aabb[1])).any():
            t_min = torch.ones_like(rays_o[..., 0]) * near
        else:
            vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
            rate_a = (self.aabb[1] - rays_o) / vec
            rate_b = (self.aabb[0] - rays_o) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if self.training:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = t_min[..., None] + step

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray_contracted(self, rays_o, rays_d, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        inner_N_samples = N_samples - N_samples // 2
        outer_N_samples = N_samples // 2
        # inner
        interpx_inner = (
            torch.linspace(near, 2.0, inner_N_samples + 1).unsqueeze(0).to(rays_o)
        )
        if self.training:
            interpx_inner[:, :-1] += (
                                             torch.rand_like(interpx_inner).to(rays_o)
                                             * ((2.0 - near) / inner_N_samples)
                                     )[:, :-1]
        interpx_inner = (interpx_inner[:, 1:] + interpx_inner[:, :-1]) * 0.5
        # sample outer
        rng = torch.arange(outer_N_samples + 1)[None].float()
        if self.training:
            rng[:, :-1] += (torch.rand_like(rng).to(rng))[:, :-1]
        rng = torch.flip(rng, [1])
        rng = (rng[:, 1:] + rng[:, :-1]) * 0.5
        interpx_outer = 1.0 / (
                1 / far + (1 / 2.0 - 1 / far) * rng / outer_N_samples
        ).to(rays_o.device)
        interpx = torch.cat((interpx_inner, interpx_outer), -1)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]

        rays_pts_norm, _ = torch.max(torch.abs(rays_pts.clone()), dim=-1)
        contract_mask = rays_pts_norm > 1.0
        rays_pts[contract_mask] = (2 - 1 / rays_pts_norm[contract_mask])[..., None] * (
                rays_pts[contract_mask] / rays_pts_norm[contract_mask][..., None]
        )

        mask_outbbox = torch.zeros_like(rays_pts[..., 0]) > 0
        return rays_pts, interpx, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
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
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1, 3), 0.01).view(
                (gridSize[1], gridSize[2])
            )
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
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
    def filtering_rays(
        self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5, bbox_only=False
    ):
        print("========> filtering rays ...")
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(
                    -1
                )  # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(
                    -1
                )  # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _, _ = self.sample_ray(rays_o, rays_d, N_samples=N_samples)
                mask_inbbox = (
                    self.alphaMask.sample_alpha(xyz_sampled).view(
                        xyz_sampled.shape[:-1]
                    )
                    > 0
                ).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(
            f"Ray filtering done! takes {time.time()-tt :.3f} s. ray mask ratio: {torch.sum(mask_filtered) / N:.3f}"
            +
            f". Remaining {torch.sum(mask_filtered)} / {N} rays. "
        )
        return all_rays[mask_filtered], all_rgbs[mask_filtered], mask_filtered

    def feature2density(self, density_features, **kwargs):
        if "weights" in kwargs:
            density_features = density_features * kwargs["weights"].view(
                *density_features.shape
            )

        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)
        elif self.fea2denseAct == "relu_abs":
            return F.relu(torch.abs(density_features))

    def compute_alpha(self, xyz_locs, length=0.01):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=torch.bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            valid_sigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = valid_sigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha

    def set_iter(self, iteration):
        self.cur_iter = iteration

        if not self.training:
            return

        self.needs_opt_reset = False

        # Pruning
        if iteration in self.update_AlphaMask_list:
            # Update BBOX
            reso_mask = tuple(self.gridSize)

            if reso_mask[0] > 200:
                reso_mask = (200, 200, 200)

            new_aabb = self.updateAlphaMask(reso_mask)

            # Update regularization weights
            if iteration == self.update_AlphaMask_list[0]:
                self.shrink(new_aabb)

        # Upsampling
        if iteration in self.upsamp_list:
            if self.use_grid_size_upsample:
                print("Before:", self.N_voxel_list, iteration)

                reso_cur = []
                for i in range(3):
                    reso_cur.append(self.N_voxel_list[i].pop(0))

                print("After:", self.N_voxel_list, iteration)
            else:
                print("Before:", self.N_voxel_list, iteration)
                n_voxels = self.N_voxel_list.pop(0)
                print("After:", self.N_voxel_list, iteration)
                reso_cur = N_to_reso(n_voxels, self.aabb)

            self.nSamples = min(
                self.max_n_samples, cal_n_samples(reso_cur, self.step_ratio)
            )
            self.upsample_volume_grid(reso_cur)

            if self.cfg.lr_upsample_reset:
                self.needs_opt_reset = True

    def forward(self, rays_chunk):
        # Sample points
        viewdirs = rays_chunk[:, 3:6]

        if self.ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3], rays_chunk[:, 3:6], N_samples=self.nSamples
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            dists = dists * torch.norm(rays_chunk[:, 3:6], dim=-1, keepdim=True)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(
                rays_chunk[:, :3], rays_chunk[:, 3:6], N_samples=self.nSamples
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~alpha_mask
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            valid_sigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = valid_sigma

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)
        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(
                xyz_sampled[app_mask], viewdirs[app_mask], app_features, {}
            )
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if self.white_bg or (self.training and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]

        # return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight
        return rgb_map


class TensorVMSplit(TensorBase):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)

        if "MLP" in self.shadingMode:
            self.opt_group = {
                "color": [
                    self.density_line,
                    self.density_plane,
                    self.app_line,
                    self.app_plane,
                ],
                "color_impl": [self.basis_mat],
            }
        else:
            self.opt_group = {
                "color": [
                    self.density_line,
                    self.density_plane,
                    self.app_line,
                    self.app_plane,
                    self.basis_mat,
                ],
            }

        if isinstance(self.renderModule, torch.nn.Module):
            if "color_impl" not in self.opt_group:
                self.opt_group["color_impl"] = [self.renderModule]
            else:
                self.opt_group["color_impl"] += [self.renderModule]

    def init_svd_volume(self, device):
        if self.fea2denseAct == 'softplus':
            self.density_plane, self.density_line = self.init_one_svd_density(
                self.density_n_comp, self.gridSize, 0.1, device
            )
        else:
            self.density_plane, self.density_line = self.init_one_svd_density(
                self.density_n_comp, self.gridSize, 1e-2, device
            )
        self.app_plane, self.app_line = self.init_one_svd(
            self.app_n_comp, self.gridSize, 0.1, device
        )
        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp), self.app_dim, bias=False
        ).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            if self.cfg.shadingMode == 'RGBIdentity':
                plane_coef.append(
                    torch.nn.Parameter(
                        scale
                        * torch.randn(
                            (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                        )
                    )
                )  #
                line_coef.append(
                    torch.nn.Parameter(
                        scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))
                    )
                )
            else:
                plane_coef.append(
                    torch.nn.Parameter(
                        scale
                        * torch.randn(
                            (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                        )
                    )
                )  #
                line_coef.append(
                    torch.nn.Parameter(
                        scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))
                    )
                )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_coef
        ).to(device)

    def init_one_svd_density(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            if self.fea2denseAct == 'softplus':
                plane_coef.append(
                    torch.nn.Parameter(
                        scale
                        * torch.randn(
                            (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                        )
                    )
                )  #
                line_coef.append(
                    torch.nn.Parameter(
                        scale
                        * torch.randn(
                            (1, n_component[i], gridSize[vec_id], 1)
                        )
                    )
                )
            elif self.fea2denseAct == 'relu':
                plane_coef.append(
                    torch.nn.Parameter(
                        scale * torch.rand(
                            (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                        ).clamp(1e-2, 1e8)
                    )
                )
                line_coef.append(
                    torch.nn.Parameter(
                        scale * torch.rand(
                            (1, n_component[i], gridSize[vec_id], 1)
                        ).clamp(1e-2, 1e8)
                    )
                )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_coef
        ).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatialxyz},
            {"params": self.density_plane, "lr": lr_init_spatialxyz},
            {"params": self.app_line, "lr": lr_init_spatialxyz},
            {"params": self.app_plane, "lr": lr_init_spatialxyz},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [
                {"params": self.renderModule.parameters(), "lr": lr_init_network}
            ]
        return grad_vars

    def vectorDiffs(self, vector_comps):
        total = 0

        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]

            dotp = torch.matmul(
                vector_comps[idx].view(n_comp, n_size),
                vector_comps[idx].view(n_comp, n_size).transpose(-1, -2),
            )
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            if self.density_plane[idx].shape[1] == 0:
                continue

            total = (
                total
                + torch.mean(torch.abs(self.density_plane[idx]))
                + torch.mean(torch.abs(self.density_line[idx]))
            )  # + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            if self.density_plane[idx].shape[1] == 0:
                continue

            total = (
                total + reg(self.density_plane[idx]) * 1e-2
            )  # + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            if self.app_plane[idx].shape[1] == 0:
                continue

            total = (
                total + reg(self.app_plane[idx]) * 1e-2
            )  # + reg(self.app_line[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack(
            (
                xyz_sampled[:, self.matMode[0]],
                xyz_sampled[:, self.matMode[1]],
                xyz_sampled[:, self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_sampled[:, self.vecMode[0]],
                xyz_sampled[:, self.vecMode[1]],
                xyz_sampled[:, self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            if self.density_plane[idx_plane].shape[1] == 0:
                continue

            plane_coef_point = F.grid_sample(
                self.density_plane[idx_plane],
                coordinate_plane[[idx_plane]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(
                self.density_line[idx_plane],
                coordinate_line[[idx_plane]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(
                plane_coef_point * line_coef_point, dim=0
            )
            # sigma_feature = sigma_feature + torch.mean(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature
        # return sigma_feature / len(self.density_plane)

    def compute_appfeature(self, xyz_sampled):
        # return xyz_sampled.new_zeros(xyz_sampled.shape[0], self.app_dim)

        # plane + line basis
        coordinate_plane = torch.stack(
            (
                xyz_sampled[..., self.matMode[0]],
                xyz_sampled[..., self.matMode[1]],
                xyz_sampled[..., self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.app_plane)):
            if self.app_plane[idx_plane].shape[1] == 0:
                continue

            plane_coef_point.append(
                F.grid_sample(
                    self.app_plane[idx_plane],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.app_line[idx_plane],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(
            line_coef_point
        )

        return self.basis_mat((plane_coef_point * line_coef_point).T)
        # return self.basis_mat((plane_coef_point * line_coef_point).T) / plane_coef_point.shape

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            if plane_coef[i].shape[1] > 0:
                plane_coef[i] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef[i].data,
                        size=(res_target[mat_id_1], res_target[mat_id_0]),
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            if line_coef[i].shape[1] > 0:
                line_coef[i] = torch.nn.Parameter(
                    F.interpolate(
                        line_coef[i].data,
                        size=(res_target[vec_id], 1),
                        mode="bilinear",
                        align_corners=True,
                    )
                )

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(
            self.app_plane, self.app_line, res_target
        )
        self.density_plane, self.density_line = self.up_sampling_VM(
            self.density_plane, self.density_line, res_target
        )

        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (
            xyz_max - self.aabb[0]
        ) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[
                    ..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]
                ]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[
                    ..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]
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
        self.register_buffer('aabb', new_aabb)
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))


class TensorCP(TensorBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.opt_group = {
            "color": [self.density_line, self.app_line],
            "color_impl": [self.basis_mat],
        }

        if isinstance(self.renderModule, torch.nn.Module):
            self.opt_group["color_impl"] += [self.renderModule]

    def init_svd_volume(self, device):
        self.density_line = self.init_one_svd(
            self.density_n_comp[0], self.gridSize, 0.2, device
        )
        self.app_line = self.init_one_svd(
            self.app_n_comp[0], self.gridSize, 0.2, device
        )
        self.basis_mat = torch.nn.Linear(
            self.app_n_comp[0], self.app_dim, bias=False
        ).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(
                    scale * torch.randn((1, n_component, gridSize[vec_id], 1))
                )
            )
        return torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatialxyz},
            {"params": self.app_line, "lr": lr_init_spatialxyz},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [
                {"params": self.renderModule.parameters(), "lr": lr_init_network}
            ]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):

        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        line_coef_point = F.grid_sample(
            self.density_line[0], coordinate_line[[0]], align_corners=True
        ).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(
            self.density_line[1], coordinate_line[[1]], align_corners=True
        ).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(
            self.density_line[2], coordinate_line[[2]], align_corners=True
        ).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(line_coef_point, dim=0)

        return sigma_feature

    def compute_appfeature(self, xyz_sampled):

        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        line_coef_point = F.grid_sample(
            self.app_line[0], coordinate_line[[0]], align_corners=True
        ).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(
            self.app_line[1], coordinate_line[[1]], align_corners=True
        ).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(
            self.app_line[2], coordinate_line[[2]], align_corners=True
        ).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    density_line_coef[i].data,
                    size=(res_target[vec_id], 1),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    app_line_coef[i].data,
                    size=(res_target[vec_id], 1),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(
            self.density_line, self.app_line, res_target
        )

        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (
            xyz_max - self.aabb[0]
        ) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.register_buffer('aabb', new_aabb)
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0

        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3

        return total

