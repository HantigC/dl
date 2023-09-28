import torch
from src.utils.box import compute_area


def compute_iou_tl_br(st_tlbr, nd_tlbr):
    tl = torch.maximum(st_tlbr[:, None, :2], nd_tlbr[:, :2])
    br = torch.minimum(st_tlbr[:, None, 2:], nd_tlbr[:, 2:])

    intersection = torch.clamp((br - tl), min=0).prod(-1)
    st_area = compute_area(st_tlbr)
    st_area = st_area.unsqueeze(-1)
    nd_area = compute_area(nd_tlbr)
    union = st_area + nd_area + intersection
    iou = intersection / union
    return iou


def make_grid(grid_size, height=1, width=1):
    h, w = grid_size
    xs = torch.linspace(0, width, w + 1)
    ys = torch.linspace(0, height, h + 1)
    ws = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=2)
    grid = torch.cat([ws[:-1, :-1], ws[1:, 1:]], dim=-1)
    grid = grid.view(-1, 4)
    return grid


def yxhw_to_yxyx(yxhw, min=0, max=1):
    br = yxhw[:, :2] + yxhw[:, 2:]
    yxyx = torch.cat([yxhw[:, :2], br], dim=1)
    torch.clamp_(yxyx, min=min, max=max)
    return yxyx


def yxyx_to_xywh(yxyx, min=0, max=1):
    hw = yxyx[:, 2:] - yxyx[:, :2]
    xywh = torch.stack([yxyx[:, 1], yxyx[:, 0], hw[:, 1], hw[:, 0]], dim=-1)
    torch.clamp_(yxyx, min=min, max=max)
    return xywh


def yxyx_to_yxhw(yxyx, min=0, max=1):
    hw = yxyx[:, 2:] - yxyx[:, :2]
    xywh = torch.stack([yxyx[:, 0], yxyx[:, 1], hw[:, 0], hw[:, 1]], dim=-1)
    torch.clamp_(yxyx, min=min, max=max)
    return xywh


def xywh_to_yxyx(xywh, min=0, max=1):
    wh = xywh[:, 2:] + xywh[:, :2]
    yxyx = torch.stack([xywh[:, 1], xywh[:, 0], wh[:, 1], wh[:, 0]], dim=-1)
    torch.clamp_(yxyx, min=min, max=max)
    return yxyx
