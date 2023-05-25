from typing import Union
import numpy as np
import torch

from light_torch.utils.device import to_numpy_cpu


def prob_to_pred(prob: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    prob = to_numpy_cpu(prob)
    if prob.ndim == 2:
        prob = np.argmax(prob, axis=1)
    return prob
