import argparse
import os.path

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
import wandb
import yaml
import sys
import matplotlib.pyplot as plt

import time

from models import *
from utils import *
from datasets import *


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


def render_test_transfer(config_args, checkpoint, checkpoint2, split='test'):
    with open(config_args.config, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    if config_args.full_res:
        cfg.dataset.half_res = not config_args.full_res

    device = cfg.experiment.device

    ckpt = load_checkpoint(cfg, checkpoint)
    nvfi, renderer = load_model_checkpoint(cfg, ckpt, device)

    with open(config_args.config2, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg2 = CfgNode(cfg_dict)

    if config_args.full_res:
        cfg2.dataset.half_res = not config_args.full_res

    ckpt2 = load_checkpoint(cfg2, checkpoint2)
    nvfi2, renderer2 = load_model_checkpoint(cfg2, ckpt2, device)

    nvfi.nvfi.vel = nvfi2.nvfi.vel
    nvfi.nvfi.vel_net = nvfi2.nvfi.vel_net

    nvfi.eval()
    renderer.eval()
    print("computing alpha mask ... ", end="")
    nvfi.nvfi.updateAlphaMask(nvfi.nvfi.gridSize, transfer=True)
    print("done")

    all_targets, all_poses, all_times, counts, render_poses, render_times, (H, W, focal) = load_blender_data(
        basedir=cfg.dataset.basedir,
        half_res=cfg.dataset.half_res,
        testskip=cfg.dataset.test_skip,
        white_background=cfg.dataset.white_background
    )
    print(f'rendering in shape {H} x {W}, half_res: {cfg.dataset.half_res}')
    test_poses = all_poses[split]
    test_targets = all_targets[split]
    test_times = all_times[split]

    savedir = os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name, 'transfer', split + '_img')
    os.makedirs(savedir, exist_ok=True)
    img_preds = []
    with torch.no_grad():
        for idx in tqdm.trange(len(test_poses)):
            pose = test_poses[idx]
            target = test_targets[idx]
            t = test_times[idx]
            camera = Camera(pose, H, W, focal, target, cfg.dataset.near, cfg.dataset.far)

            rgb_map, depth_map, acc_map, weights, velocity = renderer.render(
                t, camera.rays.to(device), white_background=cfg.dataset.white_background, mode='test', transfer_vel=True
            )
            img = rgb_map.cpu().numpy()
            img = (img * 255.).astype(np.uint8)
            img_preds.append(img)

            filename = os.path.join(savedir, f'r_{idx :03d}.png')
            imageio.imwrite(filename, img)

    estim_dir = os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name, 'transfer', split + '_img')
    # gt_dir = os.path.join(cfg.dataset.basedir, "test")

    estim = read_images_in_dir(estim_dir)
    gt = all_targets[split].permute(0, 3, 1, 2)

    estim = torch.Tensor(estim).cuda()
    gt = torch.Tensor(gt).cuda()

    errors = estim_error(estim, gt)
    save_error(errors, os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name, 'transfer'))
    print(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yaml) config file."
    )
    parser.add_argument(
        "--config2", type=str, required=True, help="Path to (.yaml) config file."
    )
    parser.add_argument(
        "--checkpoint", type=int, default=0, help="Path to load saved checkpoint from."
    )
    parser.add_argument(
        "--checkpoint2", type=int, default=0, help="Path to load saved checkpoint from."
    )
    parser.add_argument(
        "--full_res", action='store_true', help="whether to evaluate on full res"
    )

    config_args = parser.parse_args()

    if config_args.checkpoint == 0:
        checkpoint = -1
    else:
        checkpoint = config_args.checkpoint

    if config_args.checkpoint2 == 0:
        checkpoint2 = -1
    else:
        checkpoint2 = config_args.checkpoint2

    render_test_transfer(config_args, checkpoint, checkpoint2)


