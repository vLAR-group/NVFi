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


def main(config_args):
    # =================================================================================================================
    # =============================================== Preparation
    # =================================================================================================================

    # Read config file
    with open(config_args.config, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    if config_args.full_res:
        cfg.dataset.half_res = not config_args.full_res

    if config_args.wandb:
        if config_args.static:
            name = cfg.wandb.name
        else:
            name = cfg.wandb.name + '-nvfi'

        wandb.init(project=cfg.wandb.project, name=name, config=cfg, notes=cfg.wandb.notes)

    train_fp16 = config_args.disable_fp32

    # Setup logging on device
    if config_args.checkpoint != 0:
        logdir = os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name, 'from_checkpoint')
    else:
        logdir = os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        f.write(cfg.dump())

    # Set seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run
    device = cfg.experiment.device

    # Model initialization
    # ---- init model
    bbox_x, bbox_y, bbox_z = [torch.tensor(cfg.nvfi[bbox]) for bbox in ['bbox_x', 'bbox_y', 'bbox_z']]
    aabb = torch.stack([bbox_x, bbox_y, bbox_z], dim=-1).to(device)
    res_cur = N_to_reso(cfg.nvfi.N_voxel_init, aabb)
    reso_mask = res_cur
    near_far = [cfg.dataset.near, cfg.dataset.far]
    t_max = cfg.nvfi.tmax
    if config_args.checkpoint != 0:
        ckpt = load_checkpoint(cfg, config_args.checkpoint)
        nvfi, renderer = load_model_checkpoint(cfg, ckpt, device)
    else:
        nvfi = NVFi(cfg, device, aabb, res_cur, near_far).to(device)
        renderer = Renderer(
            nvfi, cfg.renderer.batch_size, cfg.renderer.test_batch_size, cfg.renderer.n_rays,
            cfg.renderer.distance_scale, tensorf_sample=cfg.renderer.tensorf_sample, ndc=cfg.renderer.ndc
        )

    # ---- setup optimizer
    if config_args.static:
        nvfi.nvfi.requires_grad_(True)
        grad_vars = nvfi.nvfi.get_optparam_groups(cfg.experiment.lr_grid, cfg.experiment.lr_net)
    elif config_args.static_dynamic:
        print("Train static and dynamic altogether")
        nvfi.nvfi.requires_grad_(True)
        grad_vars = nvfi.nvfi.get_optparam_groups(cfg.experiment.lr_grid, cfg.experiment.lr_net)
    else:
        nvfi.nvfi.requires_grad_(False)
        grad_vars = [{'params': nvfi.velocity_field.parameters(), 'lr': cfg.experiment.lr_vel}]
    if cfg.experiment.lr_decay_iters > 0:
        lr_factor = cfg.experiment.lr_decay_target_ratio ** (1 / cfg.experiment.lr_decay_iters)
    else:
        cfg.experiment.lr_decay_iters = cfg.experiment.train_iters
        lr_factor = cfg.experiment.lr_decay_target_ratio ** (1 / cfg.experiment.train_iters)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
    gscaler = torch.cuda.amp.GradScaler(enabled=train_fp16)

    # Initialize the voxel resolution for keyframe radiance field
    N_voxel_list = torch.round(torch.exp(
        torch.linspace(
            np.log(cfg.nvfi.N_voxel_init),
            np.log(cfg.nvfi.N_voxel_final),
            len(cfg.nvfi.upsamp_list) + 1
        )
    )).long().tolist()[1:]
    keyframe_list = torch.round(torch.exp(
        torch.linspace(
            np.log(cfg.nvfi.num_keyframes),
            np.log(cfg.nvfi.num_keyframes_end),
            len(cfg.nvfi.upsamp_list) + 1
        )
    )).long().tolist()[1:]

    # Loading data
    all_targets, all_poses, all_times, counts, render_poses, render_times, (H, W, focal) = load_blender_data(
        basedir=cfg.dataset.basedir,
        half_res=cfg.dataset.half_res,
        testskip=cfg.dataset.test_skip,
        white_background=cfg.dataset.white_background
    )
    train_rays = BatchedRays(all_targets['train'], all_poses['train'], all_times['train'],
                             H, W, focal, cfg.dataset.near, cfg.dataset.far, cfg.renderer.ndc)
    allrays, allrgbs, allts = train_rays.all_rays, train_rays.all_pixels, train_rays.all_ts

    # Set up regulation loss
    vel_reg_weight = cfg.experiment.vel_reg_weight
    vel_reg_n_pts = cfg.experiment.vel_reg_n_pts
    print(f'initia velocity loss weight {vel_reg_weight} with {vel_reg_n_pts} points')
    L1_reg_weight = cfg.experiment.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = cfg.experiment.TV_weight_density, cfg.experiment.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    # =================================================================================================================
    # =========================================== Training + Validating
    # =================================================================================================================
    pbar = tqdm.tqdm(range(cfg.experiment.train_iters), miniters=cfg.pbar.progress_refresh_rate, file=sys.stdout)
    for epoch in pbar:

        nvfi.train()
        renderer.train()

        with torch.cuda.amp.autocast(enabled=train_fp16):

            if not config_args.static and not config_args.keyframe:
                
                # rgb loss for random time t
                all_idx = counts['train']
                idx = np.random.randint(all_idx)

                target = all_targets['train'][idx].to(device)
                pose = all_poses['train'][idx]
                t = all_times['train'][idx]
                camera = Camera(pose, H, W, focal, target, cfg.dataset.near, cfg.dataset.far)
                rays, target = camera.sample_rays(cfg.renderer.n_rays)

                rgb_map, depth_map, acc_map, weights, velocity = renderer.render(
                    t, rays.to(device), white_background=cfg.dataset.white_background, mode='train'
                )

                rgb_loss = torch.nn.functional.mse_loss(rgb_map[..., :3], target[..., :3].to(device))
                loss = rgb_loss
                rgb_loss_t = rgb_loss.item()

            else:
                rgb_loss, rgb_loss_t = torch.zeros(1,1), torch.zeros(1,1)
                loss = 0

            # rgb loss for canonical space, t = 0
            if config_args.static:
                idx = np.random.randint(counts['init'])
                target_init = all_targets['init'][idx].to(device)
                pose_init = all_poses['init'][idx]
                camera = Camera(pose_init, H, W, focal, target_init, cfg.dataset.near, cfg.dataset.far)
                rays, target = camera.sample_rays(cfg.renderer.n_rays)
                rgb_map, depth_map, acc_map, weights, velocity = renderer.render(
                    0, rays.to(device), white_background=cfg.dataset.white_background, mode='train'
                )

                rgb_loss0 = torch.nn.functional.mse_loss(rgb_map[..., :3], target[..., :3])
                loss += 1. * rgb_loss0
            elif config_args.static_dynamic:
                all_time = torch.tensor(all_times['train'])
                t = all_time
                time_scale_factor = nvfi.nvfi.tmax / (nvfi.nvfi.num_keyframes - 1)
                base_times = torch.round(
                    (t / time_scale_factor).clamp(0.0, nvfi.nvfi.num_keyframes - 1)
                ) * time_scale_factor
                time_offset = t - base_times
                key = t.isclose(base_times)
                valid_index = torch.where(key)[0].numpy().tolist()

                idx = np.random.choice(valid_index)
                target_key = all_targets['train'][idx].to(device)
                pose_key = all_poses['train'][idx]
                t_key = all_times['train'][idx]
                camera = Camera(pose_key, H, W, focal, target_key, cfg.dataset.near, cfg.dataset.far)
                rays, target = camera.sample_rays(cfg.renderer.n_rays)
                rgb_map, depth_map, acc_map, weights, velocity = renderer.render(
                    t_key, rays.to(device), white_background=cfg.dataset.white_background, mode='train'
                )
                rgb_loss0 = torch.nn.functional.mse_loss(rgb_map[..., :3], target[..., :3])
                loss += 1. * rgb_loss0
            else:
                rgb_loss0 = torch.zeros(1,1)

            # regularization
            if config_args.static or config_args.static_dynamic or config_args.keyframe:
                if L1_reg_weight > 0:
                    L1_reg_weight *= lr_factor
                    loss_reg_L1 = nvfi.nvfi.density_L1()
                    loss += L1_reg_weight * loss_reg_L1
                    if config_args.wandb:
                        wandb.log({'train_reg_L1': loss_reg_L1.detach().item()}, step=epoch)
                if TV_weight_density > 0:
                    TV_weight_density *= lr_factor
                    loss_tv = nvfi.nvfi.TV_loss_density(tvreg) * TV_weight_density
                    loss += loss_tv
                    if config_args.wandb:
                        wandb.log({'reg_tv_density': loss_tv.detach().item()}, step=epoch)
                if TV_weight_app > 0:
                    TV_weight_app *= lr_factor
                    loss_tv = nvfi.nvfi.TV_loss_app(tvreg) * TV_weight_app
                    loss += loss_tv
                    if config_args.wandb:
                        wandb.log({'reg_tv_app': loss_tv.detach().item()}, step=epoch)

            if (not config_args.static) or config_args.static_dynamic :
                if vel_reg_weight > 0:
                    vel_reg_weight *= lr_factor
                    loss_vel = nvfi.get_vel_loss(vel_reg_n_pts)
                    if loss_vel > 0:
                        loss += vel_reg_weight * loss_vel
                        if config_args.wandb:
                            wandb.log({'train_vel_reg': loss_vel.detach().item()}, step=epoch)
                    else:
                        if config_args.wandb:
                            wandb.log({'train_vel_reg': loss_vel}, step=epoch)

        optimizer.zero_grad(set_to_none=True)
        gscaler.scale(loss).backward()
        gscaler.step(optimizer)
        scale = gscaler.get_scale()
        gscaler.update()

        # update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # log
        psnr = mse2psnr(rgb_loss0.item())
        psnr_t = mse2psnr(rgb_loss_t)
        if config_args.wandb:
            wandb.log({
                'rgb_loss_t': rgb_loss_t,
                'rgb_loss_0': rgb_loss0.item(),
                'psnr_0': psnr,
                'psnr_t': psnr_t,
            }, step=epoch)
        if epoch % cfg.experiment.print_every == 0 or epoch == cfg.experiment.train_iters - 1:
            tqdm.tqdm.write(
                f"[TRAIN] Iter: {epoch} Loss: {loss.item():.6f}"
                + f" PSNR: {psnr:.2f} "
                + f" PSNR_t: {psnr_t:.2f}"
            )
        if epoch % cfg.pbar.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iter {epoch :05d}:'
                + f' psnr = {float(np.mean(psnr)):.2f}|{float(np.mean(psnr_t)):.2f}'
                + f' loss = {loss.item():.6f}'
            )

        # Validation
        if epoch % cfg.experiment.validate_every == 0 or epoch == cfg.experiment.train_iters - 1:
            with torch.no_grad():
                loss_val = 0.
                psnr_val = 0.

                images_val = []
                depth_val = []

                if config_args.static:
                    idx = np.random.randint(counts['val'])
                    target = all_targets['val'][idx].to(device)
                    pose = all_poses['val'][idx]
                    t_list = [0]
                else:
                    idx = np.random.randint(counts['val'])
                    target = all_targets['val'][idx].to(device)
                    pose = all_poses['val'][idx]
                    t_list = [all_times['val'][idx]] + np.linspace(0, 1., 5).tolist()

                camera = Camera(pose, H, W, focal, target, cfg.dataset.near, cfg.dataset.far)
                for v_frame, t in enumerate(t_list):
                    rgb_map, depth_map, acc_map, weights, velocity = renderer.render(
                        t, camera.rays.to(device), white_background=cfg.dataset.white_background, mode='test'
                    )
                    images_val.append(rgb_map.cpu().numpy())
                    pred_depth = depth_map.cpu().numpy()
                    pred_depth = (pred_depth - cfg.dataset.near) / (cfg.dataset.far - cfg.dataset.near)
                    pred_depth = (pred_depth.clip(0, 1) * 255).astype(np.uint8)
                    depth_val.append(pred_depth)

                    if v_frame == 0:
                        rgb_loss = torch.nn.functional.mse_loss(rgb_map[..., :3], target[..., :3])
                        loss_val += rgb_loss.cpu().item()
                        psnr = mse2psnr(rgb_loss.cpu().item())
                        psnr_val += psnr

            pred_vedio = np.stack(images_val)
            pred_vedio = einops.rearrange(pred_vedio, 't h w c -> t c h w')
            pred_vedio = (pred_vedio * 255).astype(np.uint8)
            if config_args.wandb:
                wandb.log({
                    'val_rgb_loss': loss_val,
                    'val_psnr': psnr_val,
                    'val_target': wandb.Image(target.cpu().numpy(), caption='gt'),
                    'val_rgb': [wandb.Image(pred_image, caption=f"rgb at {t_list[i]}")
                                for i, pred_image in enumerate(images_val)],
                    'val_depth': [wandb.Image(pred_depth, caption=f"depth at {t_list[i]}")
                                  for i, pred_depth in enumerate(depth_val)],
                    'val_vedio': wandb.wandb.Video(pred_vedio, fps=1, format="gif")
                }, step=epoch)

            del rgb_loss, rgb_map, depth_map, acc_map, weights, target, camera, velocity

            tqdm.tqdm.write(
                f"[VALIDATION] Iter: {epoch} Loss: {loss_val} PSNR: {psnr_val}"
            )

        if config_args.static or config_args.static_dynamic:
            if epoch in cfg.nvfi.update_AlphaMask_list:

                if res_cur[0] * res_cur[1] * res_cur[2]<256**3:# update volume resolution
                    reso_mask = res_cur
                new_aabb = nvfi.nvfi.updateAlphaMask(tuple(reso_mask))
                nvfi.nvfi.shrink(new_aabb)
                if epoch == cfg.nvfi.update_AlphaMask_list[0]:
                    L1_reg_weight = cfg.experiment.L1_weight_reset
                    print("continuing L1_reg_weight", L1_reg_weight)

            if epoch in cfg.nvfi.upsamp_list:
                n_voxels = N_voxel_list.pop(0)
                res_cur = N_to_reso(n_voxels, nvfi.nvfi.aabb)
                keyframe_cur = keyframe_list.pop(0)
                nvfi.nvfi.upsample_volume_grid(res_cur, keyframe_cur)
                if cfg.experiment.lr_upsample_reset:
                    print("reset lr to initial")
                    lr_scale = 1
                else:
                    lr_scale = cfg.experiment.lr_decay_target_ratio ** (epoch / cfg.experiment.train_iters)
                grad_vars = nvfi.get_optparam_groups(
                    cfg.experiment.lr_grid * lr_scale, cfg.experiment.lr_net * lr_scale,
                    cfg.experiment.lr_vel * cfg.experiment.lr_decay_target_ratio ** (epoch / cfg.experiment.train_iters)
                )
                optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        if (epoch != 0 and epoch % cfg.experiment.save_every == 0) or epoch == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "model_state_dict": nvfi.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "nvfi_kwarg": nvfi.nvfi.get_kwargs()
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, f"model_{epoch :05d}.ckpt"),
            )
            tqdm.tqdm.write("================== Saved Checkpoint =================")


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


def render_test_evaluation(config_args, checkpoint, split='test'):
    with open(config_args.config, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    if config_args.full_res:
        cfg.dataset.half_res = not config_args.full_res

    train_fp16 = config_args.disable_fp32

    device = cfg.experiment.device

    ckpt = load_checkpoint(cfg, checkpoint, ext)
    nvfi, renderer = load_model_checkpoint(cfg, ckpt, device)

    nvfi.eval()
    renderer.eval()
    print("computing alpha mask ... ", end="")
    nvfi.nvfi.updateAlphaMask(nvfi.nvfi.gridSize)
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

    savedir = os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name, split + '_img')
    os.makedirs(savedir, exist_ok=True)
    img_preds = []
    with torch.no_grad():
        for idx in tqdm.trange(len(test_poses)):
            pose = test_poses[idx]
            target = test_targets[idx]
            t = test_times[idx]
            camera = Camera(pose, H, W, focal, target, cfg.dataset.near, cfg.dataset.far)

            rgb_map, depth_map, acc_map, weights, velocity = renderer.render(
                t, camera.rays.to(device), white_background=cfg.dataset.white_background, mode='test'
            )
            img = rgb_map.cpu().numpy()
            img = (img * 255.).astype(np.uint8)
            img_preds.append(img)

    for idx, img in enumerate(img_preds):
        filename = os.path.join(savedir, f'r_{idx :03d}.png')
        imageio.imwrite(filename, img)

    estim_dir = os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name, split + '_img')
    # gt_dir = os.path.join(cfg.dataset.basedir, "test")

    estim = read_images_in_dir(estim_dir)
    gt = all_targets[split].permute(0, 3, 1, 2)

    estim = torch.Tensor(estim).cuda()
    gt = torch.Tensor(gt).cuda()

    errors = estim_error(estim, gt)
    save_error(errors, os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name))
    print(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yaml) config file."
    )
    parser.add_argument(
        "--checkpoint", type=int, default=0, help="Path to load saved checkpoint from."
    )
    parser.add_argument(
        "--wandb", action='store_true', help="whether to turn on the usage of wandb"
    )
    parser.add_argument(
        "--disable_fp32", action='store_true', help="whether to turn off the training of float32"
    )
    parser.add_argument(
        "--not_train", action='store_true', help="whether to train the model"
    )
    parser.add_argument(
        "--eval_val", action='store_true', help="whether to evaluate on val set"
    )
    parser.add_argument(
        "--eval_test", action='store_true', help="whether to evaluate on test set"
    )
    parser.add_argument(
        "--full_res", action='store_true', help="whether to evaluate on full res"
    )
    parser.add_argument(
        "--static", action='store_true', help="whether to train init frame only"
    )
    parser.add_argument(
        "--vel", action='store_true', help="whether to train velocity field only"
    )
    parser.add_argument(
        "--static_dynamic", action='store_true', help="whether to train init frame and deformation field together"
    )

    config_args = parser.parse_args()

    if not config_args.not_train:
        main(config_args)

    if config_args.eval_val:
        if config_args.checkpoint == 0:
            checkpoint = -1
        else:
            checkpoint = config_args.checkpoint
        render_test_evaluation(config_args, checkpoint, 'val')

    if config_args.eval_test:
        if config_args.checkpoint == 0:
            checkpoint = -1
        else:
            checkpoint = config_args.checkpoint
        render_test_evaluation(config_args, checkpoint)

