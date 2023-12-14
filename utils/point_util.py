import torch


def sample_volume_points(volume_bounds, n_sample_point=64, perturb=False):
    """
    :param volume_bounds: (3, 2) torch.Tensor.
    :return:
        points: (N, N, N, 3)
    """
    t_vals = torch.linspace(0., 1., steps=n_sample_point+1).unsqueeze(1)   # (N+1, 1)
    xyz_vals = volume_bounds[:, 0] * (1 - t_vals) + volume_bounds[:, 1] * t_vals  # (N+1, 3)
    lower, upper = xyz_vals[:-1], xyz_vals[1:]      # (N, 3)

    if perturb:
        t_rand = torch.rand(n_sample_point, 3)
        points = lower + (upper - lower) * t_rand       # (N, 3)
    else:
        points = 0.5 * (lower + upper)      # (N, 3)

    x, y, z = torch.meshgrid(points[:, 0], points[:, 1], points[:, 2])
    points = torch.stack([x, y, z], 3)
    return points