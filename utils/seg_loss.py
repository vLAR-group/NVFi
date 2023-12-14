import torch

from pytorch3d.ops import knn_points, knn_gather


def fit_motion_svd_batch(pc1, pc2, mask=None):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param mask: (B, N) torch.Tensor.
    :return:
        R_base: (B, 3, 3) torch.Tensor.
        t_base: (B, 3) torch.Tensor.
    """
    n_batch, n_point, _ = pc1.size()

    if mask is None:
        pc1_mean = torch.mean(pc1, dim=1, keepdim=True)   # (B, 1, 3)
        pc2_mean = torch.mean(pc2, dim=1, keepdim=True)   # (B, 1, 3)
    else:
        pc1_mean = torch.einsum('bnd,bn->bd', pc1, mask) / torch.sum(mask, dim=1, keepdim=True)   # (B, 3)
        pc1_mean.unsqueeze_(1)
        pc2_mean = torch.einsum('bnd,bn->bd', pc2, mask) / torch.sum(mask, dim=1, keepdim=True)
        pc2_mean.unsqueeze_(1)

    pc1_centered = pc1 - pc1_mean
    pc2_centered = pc2 - pc2_mean

    if mask is None:
        S = torch.bmm(pc1_centered.transpose(1, 2), pc2_centered)
    else:
        S = pc1_centered.transpose(1, 2).bmm(torch.diag_embed(mask).bmm(pc2_centered))

    # If mask is not well-defined, S will be ill-posed.
    # We just return an identity matrix.
    valid_batches = ~torch.isnan(S).any(dim=1).any(dim=1)
    R_base = torch.eye(3, device=pc1.device).unsqueeze(0).repeat(n_batch, 1, 1)
    t_base = torch.zeros((n_batch, 3), device=pc1.device)

    if valid_batches.any():
        S = S[valid_batches, ...]
        u, s, v = torch.svd(S, some=False, compute_uv=True)
        R = torch.bmm(v, u.transpose(1, 2))
        det = torch.det(R)

        # Correct reflection matrix to rotation matrix
        diag = torch.ones_like(S[..., 0], requires_grad=False)
        diag[:, 2] = det
        R = v.bmm(torch.diag_embed(diag).bmm(u.transpose(1, 2)))

        pc1_mean, pc2_mean = pc1_mean[valid_batches], pc2_mean[valid_batches]
        t = pc2_mean.squeeze(1) - torch.bmm(R, pc1_mean.transpose(1, 2)).squeeze(2)

        R_base[valid_batches] = R
        t_base[valid_batches] = t

    return R_base, t_base


def dynamic_loss(pc, mask, flow):
    """
    :param pc: (B, N, 3) torch.Tensor.
    :param mask: (B, N, K) torch.Tensor.
    :param flow: (B, N, 3) torch.Tensor.
    :return:
        loss: () torch.Tensor.
    """
    n_batch, n_point, n_object = mask.size()
    pc2 = pc + flow
    mask = mask.transpose(1, 2).reshape(n_batch * n_object, n_point)
    pc_rep = pc.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)
    pc2_rep = pc2.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)

    # Estimate the rigid transformation
    object_R, object_t = fit_motion_svd_batch(pc_rep, pc2_rep, mask)

    # Apply the estimated rigid transformation onto point cloud
    pc_transformed = torch.einsum('bij,bnj->bni', object_R, pc_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
    pc_transformed = pc_transformed.reshape(n_batch, n_object, n_point, 3).detach()
    mask = mask.reshape(n_batch, n_object, n_point)

    # Measure the discrepancy of per-point flow
    mask = mask.unsqueeze(-1)
    pc_transformed = (mask * pc_transformed).sum(1)
    loss = (pc_transformed - pc2).norm(p=2, dim=-1)
    return loss.mean(), pc_transformed


def smooth_loss(pc, mask, k=16, radius=0.1, loss_norm=1):
    """
    :param pc: (B, N, 3) torch.Tensor.
    :param mask: (B, N, K) torch.Tensor.
    :return:
        loss: () torch.Tensor.
    """
    dist, idx, _ = knn_points(pc, pc, K=k)
    tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, k).to(idx.device)
    idx[dist > radius] = tmp_idx[dist > radius]
    nn_mask = knn_gather(mask, idx.detach())
    loss = (mask.unsqueeze(2) - nn_mask).norm(p=loss_norm, dim=-1)
    return loss.mean()


def entropy_loss(mask, epsilon=1e-5):
    """
    :param mask: (B, N, K) torch.Tensor.
    :return:
        loss: () torch.Tensor.
    """
    loss = - (mask * torch.log(mask.clamp(epsilon)))
    loss = loss.sum(-1)
    return loss.mean()


def rank_loss(mask):
    """
    :param mask: (B, N, K) torch.Tensor.
    :return:
        loss: () torch.Tensor.
    """
    loss = mask.norm(p='nuc', dim=(1, 2))
    return loss.mean()