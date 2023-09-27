from collections import defaultdict
import torch
from src.torchx.box import compute_iou_tl_br, yxhw_to_yxyx


def nms_yxyx(boxes, labels, scores, iou_threshold=0.5):
    score_value, score_indices = torch.sort(scores, descending=True, stable=True)
    keep = defaultdict(list)
    while len(score_indices) > 0:
        selected = score_indices[0]

        keep["boxes"].append(boxes[selected])
        keep["lables"].append(labels[selected])
        keep["scores"].append(scores[selected])

        boxes[score_indices[0]].unsqueeze(0)

        iou = compute_iou_tl_br(
            boxes[score_indices[1:]], boxes[score_indices[0]].unsqueeze(0)
        )
        mask = (iou < iou_threshold).squeeze_()
        score_indices = score_indices[1:][mask]

    return {k: torch.stack(v) for k, v in keep.items()}


def nms_yxhw(boxes, labels, scores, iou_threshold=0.5):
    return nms_yxyx(yxhw_to_yxyx(boxes), labels, scores, iou_threshold=iou_threshold)
