from typing import Any, Callable, Mapping
from torch import nn
from torch.nn import functional as F
from light_torch.module.base import Module

from light_torch.eval.collector import StepCollector, WindowCollector
from .eval import ClassificationMetrics


class ClassificationModule(Module):
    def __init__(
        self,
        model: nn.Module,
        loss,
        label_mapper: Mapping[str, int],
        transform: Callable[[Any], Any] = None,
        softmax=True,
    ):
        super().__init__()
        self.label_mapper = label_mapper
        self.id_mapper = {id_: label for label, id_ in label_mapper.items()}
        self.model = model
        self.transform = transform
        self.loss = loss
        self.metrics_board = None
        self.device = next(self.model.parameters()).device
        self.softmax = softmax
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
        self.model.to(device)

    def on_train_begin(self):
        self.metrics_board = ClassificationMetrics()
        self.window_collector = WindowCollector()
        self.step_collector = StepCollector()

    def on_val_begin(self):
        self.metrics_board = ClassificationMetrics()
        self.window_collector = WindowCollector()
        self.step_collector = StepCollector()

    def train_step(self, batch, batch_idx=None, epoch_idx=None):
        if self.transform is not None:
            batch = self.transform.fit(batch)

        xs, ys_gt = batch
        ys_gt = ys_gt.to(self.device)
        ys_pred = self.forward(xs)
        loss_value = self.loss(ys_pred, ys_gt)

        self.metrics_board.addps(ys_gt, ys_pred)

        self.step_collector.add_value("loss", loss_value.item())
        self.window_collector.add_value("loss", loss_value.item())

        self.log("accuracy", self.metrics_board.accuracy())
        self.log("loss", self.window_collector.get_value("loss"))
        return loss_value

    def forward(self, x):
        y = self.model(x.to(self.device))
        if self.softmax:
            y = F.softmax(y, dim=1)
        return y

    def predict(self, batch):
        if self.transform is not None:
            batch = self.transform.transform(batch)
        return self.model(batch.to(self.device))

    def val_step(self, batch, batch_idx=None, epoch_idx=None):
        if self.transform is not None:
            batch = self.transform.transform(batch)

        xs, ys_gt = batch
        ys_gt = ys_gt.to(self.device)
        ys_pred = self.forward(xs)

        loss_value = self.loss(ys_pred, ys_gt)

        self.metrics_board.addps(ys_gt, ys_pred)

        self.step_collector.add_value("loss", loss_value.item())
        self.window_collector.add_value("loss", loss_value.item())

        self.log("accuracy", self.metrics_board.accuracy())
        self.log("loss", self.window_collector.get_value("loss"))
        return loss_value

    def get_report(self):
        self.step_collector.add_value("accuracy", self.metrics_board.accuracy())
        return self.step_collector.get_summary()
