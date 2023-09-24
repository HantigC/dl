from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from src.stdx import transpose
from src.torchx import target_to_torch


def pad_collate(batch):
    imgs, targets = transpose(batch)
    _, h_max, w_max = np.max([np.array(img.shape) for img in imgs], axis=0)
    padded_imgs = []

    padded_targets = defaultdict(list)
    for img, target in zip(imgs, targets):
        target = target_to_torch(target)
        _, h, w = img.shape
        pad_h, pad_w = h_max - h, w_max - w
        img = F.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
        padded_imgs.append(img)
        if len(target):
            target["boxes"] = target["boxes"] * torch.from_numpy(
                np.array([h / h_max, w / w_max, h / h_max, w / w_max])
            )
        for name, v in target.items():
            padded_targets[name].append(v)

    imgs = torch.stack(padded_imgs)
    return imgs, padded_targets
