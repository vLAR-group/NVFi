import numpy as np
import torch
from torch.utils.data import Dataset
import time


class BatchedRays(object):
    def __init__(self, all_targets, all_poses, all_times, H, W, focal, near, far, ndc=False):
        self.target_images = all_targets
        self.poses = all_poses
        self.times = all_times
        self.H = H
        self.W = W
        self.focal = focal
        self.near, self.far = near, far
        self.ndc = ndc

        self.all_rays, self.all_pixels, self.all_ts = [], [], []
        for pose, target, times in zip(self.poses, self.target_images, all_times):
            t = torch.ones_like(target[..., :1]) * times
            camera = Camera(pose, self.H, self.W, self.focal, target, self.near, self.far, self.ndc, t)
            rays = camera.rays
            self.all_rays.append(torch.cat([rays.ray_origins.view(-1, 3), rays.ray_directions.view(-1, 3)], dim=-1))
            self.all_pixels.append(target.view(-1, 3))
            self.all_ts.append(t.view(-1, 1))
        self.all_rays = torch.cat(self.all_rays, dim=0)
        self.all_pixels = torch.cat(self.all_pixels, dim=0)
        self.all_ts = torch.cat(self.all_ts, dim=0)

    def __len__(self):
        return len(self.all_rays)


class Ray(torch.nn.Module):

    def __init__(self, ray_o, ray_d, near, far, t=None):
        super(Ray, self).__init__()
        self.restore_shape = ray_o.shape[:-1]

        self.register_buffer('ray_origins', ray_o)
        self.register_buffer('ray_directions', ray_d)
        self.register_buffer('near', near * torch.ones_like(self.ray_directions[..., :1]))
        self.register_buffer('far', far * torch.ones_like(self.ray_directions[..., :1]))
        self.num_rays = ray_o.reshape(-1, 3).shape[0]
        if t is None:
            self.register_buffer('t', torch.zeros_like(ray_o[...,:1]))
        else:
            self.register_buffer('t', t)

    def update_near_far(self, near, far):
        self.near = near
        self.far = far

    def points_sampling(self, n_points, lindisp=False, perturb=True):
        t_vals = torch.linspace(0.0, 1.0, n_points, dtype=torch.float32, device=self.ray_directions.device)

        if lindisp:
            z_vals = 1.0 / (1.0 / self.near * (1.0 - t_vals) + 1.0 / self.far * t_vals)
        else:
            z_vals = self.near * (1.0 - t_vals) + self.far * t_vals

        if perturb:
            # Get intervals between samples.
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
            lower = torch.cat((z_vals[..., :1], mids), dim=-1)
            # Stratified samples in those intervals.
            t_rand = torch.rand(*z_vals.shape, dtype=self.ray_origins.dtype, device=self.ray_origins.device)
            z_vals = lower + (upper - lower) * t_rand

        points = self.ray_origins[..., None, :] + self.ray_directions[..., None, :] * z_vals[..., :, None]
        # self.restore_shape = points.shape[:-1]
        self.z_vals = z_vals

        # points shape [n_rays, n_points, 3]
        return points


class Camera(object):

    def __init__(self, pose, height, width, focal, target, near, far, ndc=False, t=None, dpt=None):
        """
        @param height
        @param width
        @param focal
        @param pose: 4X4
                |SO(3) trans|
                |  0    1   |
        @param target: target image
        @param near
        @param far
        """
        self.pose = pose
        self.height = height
        self.width = width
        self.focal = focal
        self.target = target
        self.near, self.far = near, far
        self.ndc = ndc
        self.t = t
        self.dpt = dpt

        ii, jj = torch.meshgrid(
            torch.arange(height, device=self.pose.device),
            torch.arange(width, device=self.pose.device),
            indexing='ij'
        )
        self.coords = torch.stack([ii, jj], dim=-1).reshape(-1, 2)
        ray_origins, ray_directions = self.get_ray_bundle()
        self.rays = Ray(ray_origins, ray_directions, near, far, t)

    def get_ray_bundle(self):

        X, Y = torch.meshgrid(
            torch.arange(
                self.width, dtype=self.pose.dtype, device=self.pose.device
            ),
            torch.arange(
                self.height, dtype=self.pose.dtype, device=self.pose.device
            ),
            indexing='xy'
        )

        directions = torch.stack([
                (X - self.width * 0.5) / self.focal,
                -(Y - self.height * 0.5) / self.focal,
                -torch.ones_like(X),
        ], dim=-1,)

        ray_directions = torch.sum(
            directions[..., None, :] * self.pose[:3, :3], dim=-1
        )
        ray_origins = self.pose[:3, -1].expand(ray_directions.shape)

        if self.ndc:
            ray_origins, ray_directions = self.get_ndc_rays(self.height, self.width, ray_origins, ray_directions)

        return ray_origins, ray_directions

    def get_ndc_rays(self, H, W, rays_o: torch.Tensor, rays_d: torch.Tensor):
        # Shift ray origins to near plane
        t = - (self.near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # Projection
        o0 = -1. / (W / (2. * self.focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1. / (H / (2. * self.focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * self.near / rays_o[..., 2]

        d0 = -1. / (W / (2. * self.focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1. / (H / (2. * self.focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2. * self.near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)

        return rays_o, rays_d

    def sample_rays(self, n_rays):
        select_inds = np.random.choice(self.coords.shape[0], size=n_rays, replace=False)
        select_coords = self.coords[select_inds]

        sample_ray_o = self.rays.ray_origins[select_coords[:, 0], select_coords[:, 1], :]
        sample_ray_d = self.rays.ray_directions[select_coords[:, 0], select_coords[:, 1], :]
        sample_ray = Ray(sample_ray_o, sample_ray_d, self.near, self.far)
        target_pixels = self.target[select_coords[:, 0], select_coords[:, 1], :]
        if self.dpt is not None:
            target_dpts = self.dpt[select_coords[:, 0], select_coords[:, 1]]
            return sample_ray, target_pixels, target_dpts
        else:

            return sample_ray, target_pixels


if __name__ == '__main__':
    pose_test = torch.tensor([
        [1, 0, 0, 5],
        [0, 1, 0, 5],
        [0, 0, 1, 5],

        [0, 0, 0, 0]
    ]).float()
    H = 400
    W = 800
    focal_test = 300
    begin_time = time.time()
    np.random.randint(150)
    camera = Camera(pose_test, H, W, focal_test, torch.rand(H, W, 3), 0, 10, ndc=True)
    rays, target = camera.sample_rays(4096)
    end_time = time.time()
    all_rays = camera.rays
    sample_ray, target_pixels = camera.sample_rays(100)
    sample_ray_points = sample_ray.points_sampling(64)
    all_rays_points = all_rays.points_sampling(64)
    print(end_time - begin_time)
    exit()
