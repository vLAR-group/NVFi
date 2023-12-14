import torch
import torch.nn as nn


class Renderer(nn.Module):

    def __init__(self, tensorf, batch_size, test_batch_size, ray_chunk, distance_scale=1,
                 lindisp=False, perturb=True, tensorf_sample=True, ndc=False):
        super(Renderer, self).__init__()

        self.tensorf = tensorf

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.lindisp = lindisp
        self.perturb = perturb
        self.distance_scale = distance_scale
        self.tensorf_sample = tensorf_sample
        self.ndc = ndc
        self.ray_chunk = ray_chunk

    def forward(self, t, rays, white_background=False, transfer_vel=False):
        # ray_o, ray_d = rays.ray_origins.reshape(-1, 3), rays.ray_directions.reshape(-1, 3)
        ray_o, ray_d = rays.ray_origins, rays.ray_directions
        ray_o, ray_d = ray_o.view(-1, 3), ray_d.view(-1, 3)

        N_rays_all = ray_o.shape[0]
        rgb_map, depth_map, acc_map, weights, velocity = [], [], [], [], []
        for chunk_idx in range(N_rays_all // self.ray_chunk + int(N_rays_all % self.ray_chunk > 0)):
            r_o = ray_o[chunk_idx * self.ray_chunk:(chunk_idx + 1) * self.ray_chunk]
            r_d = ray_d[chunk_idx * self.ray_chunk:(chunk_idx + 1) * self.ray_chunk]

            if transfer_vel:
                rgb, depth, acc, w, vel = self.tensorf.render_ray_transfer(t, r_o, r_d, white_background, self.ndc)
            else:
                rgb, depth, acc, w, vel = self.tensorf.render_ray(t, r_o, r_d, white_background, self.ndc)

            rgb_map.append(rgb)
            depth_map.append(depth)
            acc_map.append(acc)
            weights.append(w)
            velocity.append(vel)

        rgb_map = torch.cat(rgb_map, 0)
        depth_map = torch.cat(depth_map, 0)
        acc_map = torch.cat(acc_map, 0)
        weights = torch.cat(weights, 0)
        velocity = torch.cat(velocity, 0)

        rgb_map = rgb_map.reshape(*rays.restore_shape, 3)
        depth_map = depth_map.reshape(*rays.restore_shape)
        acc_map = acc_map.reshape(*rays.restore_shape)
        weights = weights.reshape(*rays.restore_shape, -1)
        velocity = velocity.reshape(*rays.restore_shape, 3)

        return rgb_map, depth_map, acc_map, weights, velocity

    def render(self, t, rays, white_background=False, mode='train', transfer_vel=False):
        if mode == 'train':
            self.tensorf.train()
            return self.forward(t, rays, white_background)
        else:
            self.tensorf.eval()
            with torch.no_grad():
                return self.forward(t, rays, white_background, transfer_vel=transfer_vel)

