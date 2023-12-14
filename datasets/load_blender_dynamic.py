import os
import torch
import numpy as np
import imageio
import json
import cv2
from collections import OrderedDict

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def rodrigues_mat_to_rot(R):
    eps = 1e-16
    trc = np.trace(R)
    trc2 = (trc - 1.) / 2.
    # sinacostrc2 = np.sqrt(1 - trc2 * trc2)
    s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if (1 - trc2 * trc2) >= eps:
        tHeta = np.arccos(trc2)
        tHetaf = tHeta / (2 * (np.sin(tHeta)))
    else:
        tHeta = np.real(np.arccos(trc2))
        tHetaf = 0.5 / (1 - tHeta / 6)
    omega = tHetaf * s
    return omega


def rodrigues_rot_to_mat(r):
    wx, wy, wz = r
    theta = np.sqrt(wx * wx + wy * wy + wz * wz)
    a = np.cos(theta)
    b = (1 - np.cos(theta)) / (theta * theta)
    c = np.sin(theta) / theta
    R = np.zeros([3, 3])
    R[0, 0] = a + b * (wx * wx)
    R[0, 1] = b * wx * wy - c * wz
    R[0, 2] = b * wx * wz + c * wy
    R[1, 0] = b * wx * wy + c * wz
    R[1, 1] = a + b * (wy * wy)
    R[1, 2] = b * wy * wz - c * wx
    R[2, 0] = b * wx * wz - c * wy
    R[2, 1] = b * wz * wy + c * wx
    R[2, 2] = a + b * (wz * wz)
    return R


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, white_background=True):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = OrderedDict()
    all_poses = OrderedDict()
    all_times = OrderedDict()
    counts = OrderedDict()
    for s in splits:
        all_imgs |= {s: OrderedDict()}
        all_poses |= {s: OrderedDict()}
        all_times |= {s: OrderedDict()}
        counts |= {s: OrderedDict()}

        meta = metas[s]

        for cam, data in meta['data'].items():
            imgs = []
            times = []
            if s == 'train':
                imgs_init, poses_init, times_init = [], [], []
            # if s=='train' or testskip==0:
            #     skip = 2  # if you remove/change this 2, also change the /2 in the times vector
            # else:
            skip = testskip

            all_poses[s][cam] = torch.tensor(data['transform_matrix'], dtype=torch.float32)
            for t, frame in enumerate(data['frames'][::skip]):
                fname = os.path.join(basedir, frame['file_path'] + '.png')

                image = imageio.v2.imread(fname)
                image = torch.tensor(image, dtype=torch.float32)/ 255.  # keep all 4 channels (RGBA)
                if white_background:
                    image = image[..., :-1] * image[..., -1:] + (1.0 - image[..., -1:])
                else:
                    image = image[..., :-1] * image[..., -1:]
                imgs.append(image)

                cur_time = frame['time'] if 'time' in frame else float(t) / (len(data['frames'][::skip]) - 1)
                times.append(cur_time)

                if s == 'train' and cur_time == 0.:
                    imgs_init.append(image)
                    poses_init.append(torch.tensor(data['transform_matrix'], dtype=torch.float32))
                    times_init.append(cur_time)

        # assert times[0] == 0, "Time must start at 0"

            counts[s][cam] = len(imgs)
            all_imgs[s][cam] = torch.stack(imgs)
            all_times[s][cam] = times

    counts['init'] = len(imgs_init)
    all_imgs['init'] = torch.stack(imgs_init)
    all_poses['init'] = poses_init
    all_times['init'] = times_init

    H, W = all_imgs['test'][cam][0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format('render'))):
        with open(os.path.join(basedir, 'transforms_{}.json'.format('render')), 'r') as fp:
            meta = json.load(fp)
        render_poses = []
        for frame in meta['frames']:
            render_poses.append(torch.tensor(frame['transform_matrix'], dtype=torch.float32))
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0)
                                    for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        for split, datas in all_imgs.items():
            if split == 'init':
                if white_background:
                    imgs_half_res = torch.zeros(len(datas), H, W, 3)
                else:
                    imgs_half_res = torch.zeros(len(datas), H, W, 3)
                for i, img in enumerate(datas):
                    imgs_half_res[i] = torch.tensor(cv2.resize(img.numpy(), (W, H), interpolation=cv2.INTER_AREA))
                all_imgs[split] = imgs_half_res
            else:
                for cam, imgs in datas.items():
                    if white_background:
                        imgs_half_res = torch.zeros(len(imgs), H, W, 3)
                    else:
                        imgs_half_res = torch.zeros(len(imgs), H, W, 3)
                    for i, img in enumerate(imgs):
                        imgs_half_res[i] = torch.tensor(cv2.resize(img.numpy(), (W, H), interpolation=cv2.INTER_AREA))
                    all_imgs[split][cam] = imgs_half_res
            # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    H, W = int(H), int(W)

    return all_imgs, all_poses, all_times, counts, render_poses, render_times, [H, W, focal]
