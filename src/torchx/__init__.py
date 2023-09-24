import torch
import numpy as np


def to_device(xs, device):
    if isinstance(xs, dict):
        xs = {name: to_device(gt, device) for name, gt in xs.items()}
    elif isinstance(xs, list):
        xs = [to_device(t, device) for t in xs]
    else:
        xs = xs.to(device)
    return xs


def convert_to_torch(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return torch.from_numpy(x)


def target_to_torch(target):
    if isinstance(target, dict):
        return {k: convert_to_torch(t) for k, t in target.items()}
    return convert_to_torch(target)
