import numpy as np
import torch


def to_numpy(img: torch.Tensor) -> np.ndarray:
    return np.transpose(img.numpy(), (1, 2, 0))


def to_torch(img: np.ndarray) -> torch.Tensor:
    img = torch.from_numpy(img)
    img = torch.permute(img, (2, 0, 1))
    img = img.type(torch.float32)
    return img
