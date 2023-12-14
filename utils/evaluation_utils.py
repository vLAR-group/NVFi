import os.path
import glob
import warnings
import numpy as np
import torch


def compute_depth_loss(pred, gt):
    t_pred = torch.median(pred)
    s_pred = torch.mean(torch.abs(pred - t_pred))
    pred_norm = (pred - t_pred) / (s_pred + 1e-6)

    t_gt = torch.median(gt)
    s_gt = torch.mean(torch.abs(gt - t_gt))
    gt_norm = (gt - t_gt) / (s_gt + 1e-6)

    return torch.mean((pred_norm - gt_norm) ** 2)


def load_checkpoint(cfg, checkpoint, ext=''):
    if ext == '':
        ckpt_path = os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name, '*.ckpt')
    else:
        ckpt_path = os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name, ext, '*.ckpt')
    ckpts = glob.glob(ckpt_path)

    if checkpoint > 0:
        decimals = len(ckpts[0].split('/')[-1].rstrip('.ckpt').lstrip('model_'))
        fname = ('model_{ckpt:0' + str(decimals) + 'd}.ckpt').format(ckpt=checkpoint)
        ckpt_fname = os.path.join(cfg.experiment.logdir, cfg.wandb.project, cfg.wandb.name + ext, fname)

        if ckpt_fname not in ckpts:
            ckpt_fname = np.sort(ckpts)[-1]
            warnings.warn(f'No checkpoint: {checkpoint}, try to use the latest one {ckpt_fname}!')
        else:
            print(f'Successfully found checkpoint {ckpt_fname} !')
    elif checkpoint == -1:
        ckpt_fname = np.sort(ckpts)[-1]
        print(f'Successfully found checkpoint {ckpt_fname} !')
    else:
        raise ValueError('Checkpoint is 0')

    return torch.load(ckpt_fname)

