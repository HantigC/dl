import torch
import numpy as np


def make_grid(grid_size, height=1, width=1, as_torch=True):
    h, w = grid_size
    xs = np.linspace(0, width, w + 1)
    ys = np.linspace(0, height, h + 1)
    ws = np.stack(np.meshgrid(ys, xs), axis=2)
    grid = np.concatenate([ws[:-1, :-1], ws[1:, 1:]], axis=-1)
    grid = grid.reshape(-1, 4)
    if as_torch:
        return torch.from_numpy(grid)
    return grid


def torch_compute_iou_tl_br(st_tlbr, nd_tlbr):
    tl = torch.maximum(st_tlbr[:, None, :2], nd_tlbr[:, :2])
    br = torch.minimum(st_tlbr[:, None, 2:], nd_tlbr[:, 2:])

    intersection = torch.clamp((br - tl), min=0).prod(-1)

    st_area = compute_area(st_tlbr).unsqueeze(-1)
    nd_area = compute_area(nd_tlbr)
    union = st_area + nd_area + intersection
    iou = intersection / union
    return iou


def np_compute_iou_tl_br(st_tlbr, nd_tlbr):
    tl = np.maximum(st_tlbr[:, None, :2], nd_tlbr[:, :2])
    br = np.minimum(st_tlbr[:, None, 2:], nd_tlbr[:, 2:])
    intersection = np.maximum((br - tl), 0).prod(-1)

    st_area = compute_area(st_tlbr)
    st_area = np.expand_dims(st_area, axis=-1)
    nd_area = compute_area(nd_tlbr)

    union = st_area + nd_area + intersection
    iou = intersection / union
    return iou


def compute_area(yxyx):
    return (yxyx[:, 2:] - yxyx[:, :2]).prod(1)


def yxyx_to_xywh(yxyx):
    yxyx = np.array(yxyx)
    hw = yxyx[:, 2:] - yxyx[:, :2]
    xywh = np.stack([yxyx[:, 1], yxyx[:, 0], hw[:, 1], hw[:, 0]], axis=-1)
    return xywh


def xywh_to_yxyx(xywh):
    xywh = np.array(xywh)
    wh = xywh[:, 2:] + xywh[:, :2]
    yxyx = np.stack([xywh[:, 1], xywh[:, 0], wh[:, 1], wh[:, 0]], axis=-1)
    return yxyx
