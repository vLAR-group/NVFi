import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def compress_label(segm):
    segm_ids, segm_inv = np.unique(segm, return_inverse=True)
    return segm_inv


def align_insts(gt_segm, segm):
    gt_ids = np.unique(gt_segm)
    n_gt_inst = gt_ids.size
    pred_ids = np.unique(segm)
    n_pred_inst = pred_ids.size
    n_inst = max(n_gt_inst, n_pred_inst)

    # Match with larges overlap
    overlap = np.zeros([n_inst, n_inst]).astype(float)
    for i in range(n_gt_inst):
        for j in range(n_pred_inst):
            overlap[i, j] = np.sum(np.logical_and(gt_segm == gt_ids[i], segm == pred_ids[j]))
    row_ind, col_ind = linear_sum_assignment(overlap, maximize=True)

    segm_aligned = np.zeros_like(segm)
    for i in range(n_inst):
        segm_aligned[segm == col_ind[i]] = row_ind[i]
    return segm_aligned