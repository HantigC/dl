import torch
import numpy as np


def make_grid(grid_size, height=1, width=1):
    h, w = grid_size
    xs = torch.linspace(0, width, w + 1)
    ys = torch.linspace(0, height, h + 1)
    ws = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=2)
    grid = torch.cat([ws[:-1, :-1], ws[1:, 1:]], dim=-1).view(-1, 4)
    return grid


def compute_iou_tl_br(st_tlbr, nd_tlbr):
    tl = torch.maximum(st_tlbr[:, None, :2], nd_tlbr[:, :2])
    br = torch.minimum(st_tlbr[:, None, 2:], nd_tlbr[:, 2:])

    intersection = torch.clamp((br - tl), min=0).prod(-1)

    st_area = compute_area(st_tlbr).unsqueeze(-1)
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
    yxyx = np.concatenate([xywh[:, 1], xywh[:, 0], wh[:, 1], wh[:, 0]], axis=-1)
    return yxyx
