import argparse
import os.path
import yaml
from tqdm import tqdm

import wandb

from models import *
from utils import *


def load_model_checkpoint(cfg, checkpoint, device):
    aabb = checkpoint["nvfi_kwarg"]['aabb'].to(device)
    res_cur = checkpoint["nvfi_kwarg"]['gridSize']
    near_far = [cfg.dataset.near, cfg.dataset.far]
    cfg.nvfi.num_keyframes = checkpoint["nvfi_kwarg"]['num_keyframes']
    nvfi = NVFi(cfg, device, aabb, res_cur, near_far).to(device)
    nvfi.update_nvfi_kwargs(checkpoint["nvfi_kwarg"])
    # nvfi.nvfi.upsample_volume_grid(nvfi.nvfi.gridSize, nvfi.nvfi.num_keyframes)
    try:
        alpha_aabb = checkpoint["model_state_dict"]["nvfi.alphaMask.alpha_aabb"]
        alpha_volume = checkpoint["model_state_dict"]["nvfi.alphaMask.alpha_volume"]
        nvfi.nvfi.alphaMask = AlphaGridMask(device, alpha_aabb.to(device), alpha_volume.to(device))
    except:
        pass
    nvfi.load_state_dict(checkpoint["model_state_dict"])
    renderer = Renderer(
        nvfi, cfg.renderer.batch_size, cfg.renderer.test_batch_size, cfg.renderer.n_rays, cfg.renderer.distance_scale,
        tensorf_sample=cfg.renderer.tensorf_sample
    ).to(device)

    return nvfi, renderer


def balanced_sample(xyz, object_bounds):
    fg = (xyz[:, 0] > object_bounds[0, 0]) & (xyz[:, 0] < object_bounds[0, 1]) & \
         (xyz[:, 1] > object_bounds[1, 0]) & (xyz[:, 1] < object_bounds[1, 1]) & \
         (xyz[:, 2] > object_bounds[2, 0]) & (xyz[:, 2] < object_bounds[2, 1])
    bg = ~fg
    xyz_fg = xyz[fg]
    n_fg_point = xyz_fg.shape[0]

    # Downsample the background points
    xyz_bg = xyz[bg]
    n_bg_point = xyz_bg.shape[0]
    if n_bg_point > n_fg_point:
        idx = np.random.choice(n_bg_point, n_fg_point, replace=False)
        xyz_bg = xyz_bg[idx]

    xyz = torch.cat([xyz_fg, xyz_bg], 0)
    return xyz


if __name__ == '__main__':
    # Fix the random seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the pre-trained NVFi model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yaml) config file."
    )
    parser.add_argument(
        "--checkpoint", type=int, default=0, help="Path to load saved checkpoint from."
    )
    parser.add_argument(
        '--use_wandb', dest='use_wandb', default=False, action='store_true', help='Use WANDB for logging'
    )
    config_args = parser.parse_args()

    with open(config_args.config, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    device = cfg.experiment.device

    ckpt = load_checkpoint(cfg, config_args.checkpoint)
    nvfi, renderer = load_model_checkpoint(cfg, ckpt, device)

    vel_net = nvfi.nvfi.vel
    kplane = nvfi.nvfi

    n_sample_res = cfg.segmentation.n_sample_res   # 96
    volume_bounds = kplane.aabb.reshape(2, 3).transpose(0, 1).cpu()
    try:
        object_bounds = vel_net.surround.reshape(2, 3).transpose(0, 1).cpu()
    except:
        object_bounds = None
    max_t = kplane.tmax
    min_t = cfg.segmentation.min_t


    # Create the MaskField model
    n_object = cfg.segmentation.n_object
    model = MaskField(n_layer=4,
                      n_dim=128,
                      input_dim=3,
                      skips=[],
                      mask_dim=n_object,
                      mask_act='softmax').to(device)

    # Hyperparameters for the mask field
    n_iters = cfg.segmentation.n_iters
    lrate = cfg.segmentation.lrate
    lrate_decay = cfg.segmentation.lrate_decay
    lrate_decay_step = cfg.segmentation.lrate_decay_step
    save_freq = cfg.segmentation.save_freq

    loss_smooth_w = cfg.segmentation.loss_smooth_w

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lrate, betas=(0.9, 0.999))

    exp_name = cfg.wandb.name + f'_k={n_object}'
    # Create wandb logger
    if config_args.use_wandb:
        wandb.init(project=cfg.wandb.project,
                   name=exp_name,
                   config=None)

    exp_base = os.path.join('logs_segm', exp_name)
    os.makedirs(exp_base, exist_ok=True)

    tbar = tqdm(total=n_iters)
    for it in range(1, n_iters + 1):
        xyz = sample_volume_points(volume_bounds, n_sample_res, perturb=True).to(device)
        xyz = xyz.reshape(-1, 3)
        xyz = kplane.normalize_coord(xyz)
        n_point = xyz.shape[0]

        t0 = torch.zeros(n_point, 1).to(device)
        t0_norm = kplane.normalize_time_coord(t0)

        with torch.no_grad():
            # Query the sigma
            xyzt0_norm = torch.cat([xyz, t0_norm], dim=1)
            sigma_feature = kplane.compute_densityfeature(xyzt0_norm)
            sigma = kplane.feature2density(sigma_feature, {})

            # Select valid points
            dists = 0.01
            alpha = 1.0 - torch.exp(-sigma * dists)

            app_mask = alpha > (kplane.alphaMask_thres * cfg.segmentation.alpha_scale)
            # if 'dining' in exp_name:
            #     app_mask = alpha > (kplane.alphaMask_thres * 10)
            # else:
            #     app_mask = alpha > kplane.alphaMask_thres
            xyz = xyz[app_mask]

            # Balance the number of points in FG / BG
            xyz_org = (xyz + 1) / kplane.invaabbSize + kplane.aabb[0]
            if object_bounds:
                xyz_org = balanced_sample(xyz_org, object_bounds)
            xyz = kplane.normalize_coord(xyz_org)

            n_point = xyz.shape[0]
            t0 = torch.zeros(n_point, 1).to(device)

            # Sample a time step
            t = min_t + (max_t - min_t) * torch.rand(1).to(device)
            t = t0 + t

            # Query the motion
            xyz2 = kplane.integrate_pos(xyz.clone(), t0, t)

            # xyz = (xyz + 1) / kplane.invaabbSize + kplane.aabb[0]
            # xyz2 = (xyz2 + 1) / kplane.invaabbSize + kplane.aabb[0]

            flow = xyz2 - xyz

            # import open3d as o3d
            # from point_visual_util import build_pointcloud_segm
            # xyz = xyz.cpu().numpy()
            # flow = flow.cpu().numpy()
            # pcds = []
            # pcds.append(build_pointcloud_segm(xyz, np.zeros(xyz.shape[0], dtype=np.int32)))
            # pcds.append(build_pointcloud_segm(xyz + flow, np.ones(xyz.shape[0], dtype=np.int32)))
            # o3d.visualization.draw_geometries(pcds)

        # Query prediction of object mask
        mask = model(xyz)

        # Compute dynamic loss
        xyz = xyz.unsqueeze(0)      # [1, n_point, 3]
        mask = mask.unsqueeze(0)        # [1, n_point, n_object]
        flow = flow.unsqueeze(0)        # [1, n_point, 3]
        loss_dynamic, _ = dynamic_loss(xyz, mask, flow)

        # Compute smooth loss
        # loss_smooth = smooth_loss(xyz, mask)
        loss_smooth = smooth_loss(xyz, mask, k=4, radius=0.01)

        # Compute entropy loss
        loss_entropy = entropy_loss(mask)

        # Compute total loss
        if it < cfg.segmentation.smooth_iter:
            loss = loss_dynamic
        else:
            loss = loss_dynamic + loss_smooth_w * loss_smooth

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save model checkpoint
        if it % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(exp_base, 'model_%06d.pth.tar'%(it)))

        # Decay learning rate
        new_lrate = lrate * (lrate_decay ** (it / lrate_decay_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Logging
        train_log = {"train_loss": loss_dynamic.item(),
                     "smooth_loss": loss_smooth.item(),
                     "entropy_loss": loss_entropy.item(),
                     "lrate": new_lrate}
        if config_args.use_wandb:
            wandb.log(train_log)

        tbar.set_description('Loss: %.4f'%(loss.item()))
        tbar.update()