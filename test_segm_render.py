import argparse
import os.path as osp
import yaml
from tqdm import tqdm

import einops
import numpy as np
import torch

from models import *
from utils import *
from datasets import *

from train_segm import load_model_checkpoint


if __name__ == "__main__":
    # Load the pre-trained TensoRF model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yaml) config file."
    )
    parser.add_argument(
        "--checkpoint", type=int, default=0, help="Path to load saved checkpoint from."
    )
    parser.add_argument(
        "--ckpt_segm", type=int, default=0, help="Path to load saved checkpoint from."
    )
    parser.add_argument(
        '--load_saved', dest='load_saved', default=False, action='store_true', help='Load pre-saved rendering results'
    )
    config_args = parser.parse_args()

    with open(config_args.config, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)


    # Load blender data
    # basedir = cfg.dataset.basedir.replace('data', 'data_segm')
    basedir = cfg.dataset.basedir.replace('data', 'data_segm_allframe')
    test_targets, test_poses, test_segms, test_times, counts, render_poses, render_times, (H, W, focal) = load_blender_data_segm(
        basedir=basedir,
        half_res=cfg.dataset.half_res,
        testskip=cfg.dataset.test_skip,
        white_background=cfg.dataset.white_background
    )
    print(f'rendering in shape {H} x {W}, half_res: {cfg.dataset.half_res}')
    split = 'test'
    n_view_test = len(test_poses)

    # Specify the path to save rendered images
    exp_name = config_args.config.split('/')[-1][:-5] + '_k=%d' % (config_args.n_object)
    exp_base = os.path.join('logs_segm', exp_name)
    # save_render_base = osp.join(exp_base, 'test_%06d' % (config_args.ckpt_segm))
    # save_render_base = osp.join(exp_base, 'test_%06d_allframe' % (config_args.ckpt_segm))
    save_render_base = osp.join(exp_base, 'test_%06d_allframe_k4' % (config_args.ckpt_segm))
    os.makedirs(save_render_base, exist_ok=True)

    device = 'cuda:0'  # cfg.experiment.device


    """
    Render with trained NeRF model & Mask field
    """
    if not config_args.load_saved:
        ckpt = load_checkpoint(cfg, config_args.checkpoint)
        nvfi, renderer = load_model_checkpoint(cfg, ckpt, device)

        vel_net = nvfi.nvfi.vel
        kplane = nvfi.tensorf

        # Load the pre-trained MaskField model
        n_object = cfg.segmentation.n_object
        model = MaskField(n_layer=4,
                          n_dim=128,
                          input_dim=3,
                          skips=[],
                          mask_dim=n_object,
                          mask_act='softmax').to(device)
        weight_path = osp.join(exp_base, 'model_%06d.pth.tar' % (config_args.ckpt_segm))
        model.load_state_dict(torch.load(weight_path))

        nvfi.nvfi.mask_field = model
        renderer.tensorf = nvfi

        # Traverse the testing set
        tbar = tqdm(total=n_view_test)
        for vid in range(n_view_test):
            pose = test_poses[vid]
            target = test_targets[vid]
            t = test_times[vid]
            camera = Camera(pose, H, W, focal, target, cfg.dataset.near, cfg.dataset.far)

            with torch.no_grad():
                rgb_map, depth_map, acc_map, weights, segm_map = renderer.render(
                    t, camera.rays.to(device), white_background=cfg.dataset.white_background, mode='test', transfer_vel=True
                )

            segm_map = segm_map.cpu().numpy()
            save_path = osp.join(save_render_base, 'r_%03d_segm.npy' % (vid))
            np.save(save_path, segm_map)
            segm_map = segm_map.argmax(-1)
            segm_map_vis = build_segm_vis(segm_map)
            save_path = osp.join(save_render_base, 'r_%03d_segm_vis.png' % (vid))
            segm_map_vis = (segm_map_vis * 255).astype(np.uint8)
            imageio.imwrite(save_path, segm_map_vis)

            tbar.update(1)


    """
    Compute quantitative metrics
    """
    # Load predicted segmentation maps
    pred_segms = []
    for vid in range(n_view_test):
        pred_segm_file = osp.join(save_render_base, 'r_%03d_segm.npy' % (vid))
        pred_segm = np.load(pred_segm_file)
        pred_segms.append(pred_segm)
    pred_segms = np.stack(pred_segms, 0)

    # Align the object order in GT & Pred
    gt_segms_all = np.reshape(test_segms.cpu().numpy(), (-1))
    gt_segms_all = compress_label(gt_segms_all)
    pred_segms_all = np.reshape(pred_segms, (-1, config_args.n_object))
    pred_segms_all = pred_segms_all.argmax(-1)
    pred_segms_all = compress_label(pred_segms_all)
    pred_segms_aligned = align_insts(gt_segms_all, pred_segms_all)
    pred_segms_aligned = pred_segms_aligned.reshape(-1, H, W)

    # mbs_eval = ClusteringMetrics(spec=[ClusteringMetrics.IOU, ClusteringMetrics.RI])
    mbs_eval = ClusteringMetrics(spec=[ClusteringMetrics.IOU])
    ap_eval_meter = {'Pred_IoU': [], 'Pred_Matched': [], 'Confidence': [], 'N_GT_Inst': []}
    mbs_eval_meter = {'IoU': [], 'RI': []}

    tbar = tqdm(total=n_view_test)
    for vid in range(n_view_test):
        target_segm = torch.Tensor(test_segms[vid])
        pred_segm = torch.Tensor(pred_segms[vid])

        target_segm = target_segm.reshape(-1).unsqueeze(0)
        pred_segm = pred_segm.reshape(-1, config_args.n_object).unsqueeze(0)

        # Accumulate for AP, PQ, F1, Pre, Rec
        Pred_IoU, Pred_Matched, Confidence, N_GT_Inst = accumulate_eval_results(target_segm, pred_segm)
        ap_eval_meter['Pred_IoU'].append(Pred_IoU)
        ap_eval_meter['Pred_Matched'].append(Pred_Matched)
        ap_eval_meter['Confidence'].append(Confidence)
        ap_eval_meter['N_GT_Inst'].append(N_GT_Inst)

        # mIoU & RI metrics
        per_scan_mbs = mbs_eval(pred_segm, target_segm.long())
        mbs_eval_meter['IoU'].append(per_scan_mbs['iou'])
        # mbs_eval_meter['RI'].append(np.mean(per_scan_mbs['ri']))

        # Save visualization of rendered segmentation maps
        pred_segm = pred_segms_aligned[vid].reshape(H, W)
        segm_map = pred_segm
        segm_map_vis = build_segm_vis(segm_map, with_background=True)
        save_path = osp.join(save_render_base, 'r_%03d_segm_vis.png' % (vid))
        segm_map_vis = (segm_map_vis * 255).astype(np.uint8)
        imageio.imwrite(save_path, segm_map_vis)

        tbar.update(1)

    # Evaluate
    print('Evaluation on %s:' % (exp_name))
    Pred_IoU = np.concatenate(ap_eval_meter['Pred_IoU'])
    Pred_Matched = np.concatenate(ap_eval_meter['Pred_Matched'])
    Confidence = np.concatenate(ap_eval_meter['Confidence'])
    N_GT_Inst = np.sum(ap_eval_meter['N_GT_Inst'])
    AP = calculate_AP(Pred_Matched, Confidence, N_GT_Inst, plot=False)
    print('AveragePrecision@50:', AP)
    PQ, F1, Pre, Rec = calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst)
    print('PanopticQuality@50:', PQ, 'F1-score@50:', F1, 'Prec@50:', Pre, 'Recall@50:', Rec)
    # IoU, RI = np.mean(mbs_eval_meter['IoU']), np.mean(mbs_eval_meter['RI'])
    # print('mIoU:', IoU, 'RI:', RI)
    IoU = np.mean(mbs_eval_meter['IoU'])
    print('mIoU:', IoU)