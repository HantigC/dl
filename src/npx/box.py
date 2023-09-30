import numpy as np
from src.utils.box import compute_area


def compute_iou_tl_br(st_tlbr, nd_tlbr):
    tl = np.maximum(st_tlbr[:, None, :2], nd_tlbr[:, :2])
    br = np.minimum(st_tlbr[:, None, 2:], nd_tlbr[:, 2:])

    intersection = np.maximum((br - tl), 0).prod(-1)
    st_area = compute_area(st_tlbr)
    st_area = np.expand_dims(st_area, axis=-1)
    nd_area = compute_area(nd_tlbr)
    union = st_area + nd_area - intersection
    iou = intersection / union
    return iou


def yxyx_to_xywh(yxyx):
    yxyx = np.array(yxyx)
    hw = yxyx[:, 2:] - yxyx[:, :2]
    xywh = np.stack([yxyx[:, 1], yxyx[:, 0], hw[:, 1], hw[:, 0]], axis=-1)
    return xywh


def yxhw_to_xywh(yxhw):
    yxhw = np.array(yxhw)
    xywh = np.stack([yxhw[:, 1], yxhw[:, 0], yxhw[:, 3], yxhw[:, 2]], axis=-1)
    return xywh


def xywh_to_yxyx(xywh):
    xywh = np.array(xywh)
    wh = xywh[:, 2:] + xywh[:, :2]
    yxyx = np.stack([xywh[:, 1], xywh[:, 0], wh[:, 1], wh[:, 0]], axis=-1)
    return yxyx


def yxhw_to_yxyx(xywh):
    xywh = np.array(xywh)
    wh = xywh[:, 2:] + xywh[:, :2]
    yxyx = np.stack([xywh[:, 0], xywh[:, 1], wh[:, 0], wh[:, 1]], axis=-1)
    return yxyx


def xywh_to_yxhw(xywh):
    xywh = np.array(xywh)
    yxhw = np.stack([xywh[:, 1], xywh[:, 0], xywh[:, 3], xywh[:, 2]], axis=-1)
    return yxhw


def make_grid(grid_size, height=1, width=1):
    h, w = grid_size
    xs = np.linspace(0, width, w + 1)
    ys = np.linspace(0, height, h + 1)
    ws = np.stack(np.meshgrid(ys, xs), axis=2)
    grid = np.concatenate([ws[:-1, :-1], ws[1:, 1:]], axis=-1)
    grid = grid.reshape(-1, 4)
    return grid
