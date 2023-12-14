import torch
import numpy as np
import einops


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


class RayImportanceSampler:
    def __init__(self, all_poses, all_targets, all_times, batch):
        self.poses = all_poses
        self.targets = all_targets
        self.times = all_times
        self.batch = batch
        self.cam_N = len(all_poses)
        self.T = len(list(all_times.values())[0])

    def get_isg_weights(self, gamma=0.02):
        weights = []
        self.cam_idx = {}
        for i, (cam, imgs) in enumerate(self.targets.items()):
            median_img = imgs.median(0, keepdims=True).values
            weight = imgs - median_img
            weight = weight ** 2 # + 0.0002
            weight = weight / (weight + gamma ** 2)
            weight = weight.mean(dim=-1)
            weights.append(weight)
            self.cam_idx[i] = cam

        weights = torch.stack(weights)
        self.weights = weights # / einops.reduce(weights, 'cam t H W -> 1 t 1 1', 'sum')
        self.weights = einops.rearrange(self.weights, 'cam t H W -> t (cam H W)')
        self.length = self.weights.shape[-1]

    def nextids(self):
        # cam_id = np.random.randint(self.cam_N)
        t_id = np.random.randint(self.T)
        # ids = np.random.choice(self.length, p=self.weights[cam_id][t_id], replace=False, size=self.batch)
        # return cam_id, self.cam_idx[cam_id], t_id, ids
        # ids = np.random.choice(self.length, p=self.weights[t_id], replace=False, size=self.batch)
        ids = torch.multinomial(self.weights[t_id], num_samples=self.batch)
        return t_id, ids


class PatchSampler(object):

    def __init__(self, n_random_poses):
        self.n_random_poses = n_random_poses
        self.random_poses = torch.tensor(self._generate_random_poses())

    def _generate_random_poses(self):
        """Generates random poses."""
        def sample_on_sphere(n_samples, only_upper=True, radius=4.03112885717555):
            p = np.random.randn(n_samples, 3)
            if only_upper:
                p[:, -1] = abs(p[:, -1])
            p = p / np.linalg.norm(p, axis=-1, keepdims=True) * radius
            return p

        def create_look_at(eye, target=np.array([0, 0, 0]),
                           up=np.array([0, 0, 1]), dtype=np.float32):
            """Creates lookat matrix."""
            eye = eye.reshape(-1, 3).astype(dtype)
            target = target.reshape(-1, 3).astype(dtype)
            up = up.reshape(-1, 3).astype(dtype)

            def normalize_vec(x, eps=1e-9):
                return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

            forward = normalize_vec(target - eye)
            side = normalize_vec(np.cross(forward, up))
            up = normalize_vec(np.cross(side, forward))

            up = up * np.array([1., 1., 1.]).reshape(-1, 3)
            forward = forward * np.array([-1., -1., -1.]).reshape(-1, 3)

            rot = np.stack([side, up, forward], axis=-1).astype(dtype)
            return rot

        origins = sample_on_sphere(self.n_random_poses)
        rotations = create_look_at(origins)
        random_poses = np.concatenate([rotations, origins[:, :, None]], axis=-1)

        return np.stack(random_poses, axis=0)
