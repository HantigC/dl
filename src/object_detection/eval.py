from collections import defaultdict
import numpy as np
from src.npx.box import compute_iou_tl_br


class MeanAveragePrecision:
    def __init__(
        self,
        iou_thresholds=None,
        recall_thresholds=None,
    ):
        if iou_thresholds is None:
            iou_thresholds = np.linspace(
                0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
            )
            iou_thresholds = np.round(iou_thresholds, 2)
        if recall_thresholds is None:
            recall_thresholds = np.linspace(
                0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
            )
            recall_thresholds = np.round(recall_thresholds, 2)

        self.iou_thresholds = iou_thresholds
        self.recall_thresholds = recall_thresholds

        self.category_counters = None
        self.scores_per_category = None
        self.tp_per_class_per_th = None
        self.reinit()

    def reinit(self):
        self.category_counters = defaultdict(int)
        self.scores_per_category = defaultdict(list)
        self.tp_per_class_per_th = defaultdict(lambda: defaultdict(list))

    def accumulate(self):
        sorted_score_indices = {}
        for category, scores in self.scores_per_category.items():
            sorted_score_indices[category] = np.argsort(scores)

        map_per_th = {}
        for th, tp_per_class in self.tp_per_class_per_th.items():
            aps = []
            for label, tp in tp_per_class.items():
                tp = np.array(tp)
                tp = tp[sorted_score_indices[label]]
                fp = (tp == 0).cumsum()
                tp = tp.cumsum()
                if self.category_counters[label] == 0:
                    recall = 0
                else:
                    recall = tp / self.category_counters[label]

                precision = tp / (tp + fp + np.spacing(1))
                precision = precision.tolist()
                for i in range(len(tp) - 1, 0, -1):
                    if precision[i] > precision[i - 1]:
                        precision[i - 1] = precision[i]

                ap = self._get_precission_at_recall(recall, precision)
                aps.append(ap)
            map_per_th[f"mAP_{th}"] = np.mean(aps)
        return map_per_th

    def _get_precission_at_recall(self, recall, precision):
        recall_indices = np.searchsorted(recall, self.recall_thresholds, side="left")
        precision_at = np.zeros_like(self.recall_thresholds)
        try:
            for pidx, idx in enumerate(recall_indices):
                precision_at[pidx] = precision[idx]
        except IndexError:
            pass
        return np.mean(precision_at)

    def add_batch(self, gt, pred):
        for pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels in zip(
            pred["boxes"], pred["scores"], pred["labels"], gt["boxes"], gt["labels"]
        ):
            self._compute(
                np.array(pred_boxes),
                np.array(pred_scores),
                np.array(pred_labels),
                np.array(gt_boxes),
                np.array(gt_labels),
            )

    def _compute(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        indices = np.argsort(pred_scores)
        ious = compute_iou_tl_br(pred_boxes, gt_boxes)

        for gt_label in gt_labels:
            self.category_counters[gt_label] += 1
        for pred_label, pred_score in zip(pred_labels, pred_scores):
            self.scores_per_category[pred_label].append(pred_score)

        for iou_th in self.iou_thresholds:
            already_ioud = [False for _ in range(len(gt_boxes))]
            for pred_idx in indices:
                max_gt_idx = -1
                max_iou = iou_th
                for gt_idx, box in enumerate(gt_boxes):
                    if gt_labels[gt_idx] != pred_labels[pred_idx]:
                        continue

                    if already_ioud[gt_idx]:
                        continue

                    iou_scalar = ious[pred_idx, gt_idx]

                    if iou_scalar < max_iou:
                        continue
                    max_iou = iou_scalar
                    max_gt_idx = gt_idx

                if max_gt_idx != -1:
                    already_ioud[max_gt_idx] = True
                    self.tp_per_class_per_th[iou_th][pred_labels[pred_idx]].append(1)
                else:
                    self.tp_per_class_per_th[iou_th][pred_labels[pred_idx]].append(0)
