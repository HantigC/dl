import cv2
import numpy as np
import torch


def to_rgb(img):
    return np.repeat(img[..., None], 3, axis=2)


def to_numpy(img):
    return np.array(img)


def resize(img, size=(224, 224)):
    return cv2.resize(img, size)


def to_torch(img):
    return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype(np.float32))
