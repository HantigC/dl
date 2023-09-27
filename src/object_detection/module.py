from collections import defaultdict
from typing import Any, Callable, Mapping
from torch import nn
from light_torch.module.base import Module

from light_torch.eval.collector import StepCollector, WindowCollector
from src.torchx import to_device
from . import nms_yxhw


class ObjectDetectionModule(Module):
    def __init__(
        self,
        model: nn.Module,
        loss,
        label_mapper: Mapping[str, int],
        device=None,
        iou_threshold=0.5,
        transform: Callable[[Any], Any] = None,
        transform_y=None,
    ):
        super().__init__()
        self.label_mapper = label_mapper
        self.id_mapper = {id_: label for label, id_ in label_mapper.items()}
        self.model = model
        self.iou_threshold = iou_threshold
        self.transform = transform
        self.transform_y = transform_y
        self.loss = loss
        self.metrics_board = None
        if device is None:
            device = next(self.model.parameters()).device
        self.device = device
        self.window_collector = WindowCollector()
        self.step_collector = StepCollector()

    def get_label_num(self):
        return len(self.label_mapper)

    def label_to_id(self, label):
        return self.label_mapper[label]

    def id_to_label(self, idd):
        return self.id_mapper[idd]

    def set_device(self, device):
        self.device = device
        super().to(self.device)

    def on_train_begin(self):
        self.window_collector = WindowCollector()
        self.step_collector = StepCollector()

    def on_val_begin(self):
        self.window_collector = WindowCollector()
        self.step_collector = StepCollector()

    def _compute_loss(self, batch, batch_idx=None, epoch_idx=None):
        if self.transform is not None:
            batch = self.transform.fit(batch)

        xs, ys_gt = batch
        ys_gt = to_device(ys_gt, self.device)

        ys_pred = self.forward(xs)
        loss_value = self.loss(ys_pred, ys_gt)
        if isinstance(loss_value, dict):
            for loss_name, value in loss_value.items():
                self.step_collector.add_value(loss_name, value.item())
                self.window_collector.add_value(loss_name, value.item())
                self.log(loss_name, self.window_collector.get_value(loss_name))

            loss_value = sum(loss_value.values())

        self.step_collector.add_value("loss", loss_value.item())
        self.window_collector.add_value("loss", loss_value.item())

        self.log("loss", self.window_collector.get_value("loss"))

        return loss_value

    def train_step(self, batch, batch_idx=None, epoch_idx=None):
        return self._compute_loss(batch, batch_idx, epoch_idx)

    def forward(self, x):
        if self.transform is not None:
            x = self.transform.transform(x)

        y = self.model(to_device(x, self.device))
        return y

    def predict(self, x):
        y = self.forward(x)
        if self.transform_y is not None:
            y = self.transform_y(y)
        if not self.training:
            predicts = defaultdict(list)
            for pred_boxes, pred_labels, pred_scores in zip(
                y["boxes"], y["labels"], y["scores"]
            ):
                nmsed_y = nms_yxhw(
                    pred_boxes, pred_labels, pred_scores, self.iou_threshold
                )
                for k, v in nmsed_y.items():
                    predicts[k].append(v)
            return dict(predicts)
        return y

    def val_step(self, batch, batch_idx=None, epoch_idx=None):
        return self._compute_loss(batch, batch_idx, epoch_idx)

    def get_report(self):
        return self.step_collector.get_summary()
