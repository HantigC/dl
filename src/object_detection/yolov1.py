from typing import Any, Callable, Mapping
from collections import OrderedDict, defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from light_torch.module import Module

from src.layers.xtended import Conv2dSamePadding
from src.layers.lazy import LazyConv2d
from src.torchx.box import make_grid, compute_iou_tl_br, yxhw_to_yxyx
from src.torchx import to_device
from .eval import MeanAveragePrecision

from . import nms_yxhw

CPU_DEVICE = torch.device("cpu")


class YoloV1Backbone(nn.Module):
    """docstring for YoloBackbone."""

    def __init__(self):
        super().__init__()
        self.st_conv = Conv2dSamePadding(3, 64, kernel_size=(7, 7), stride=2)
        self.st_maxpol = nn.MaxPool2d((2, 2), 2)
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("conv2", Conv2dSamePadding(64, 192, kernel_size=(3, 3))),
                    ("activation2", nn.LeakyReLU()),
                    ("pool2", nn.MaxPool2d((2, 2), 2)),
                    (
                        "identity_conv31",
                        Conv2dSamePadding(192, 128, kernel_size=(1, 1)),
                    ),
                    ("conv31", Conv2dSamePadding(128, 256, kernel_size=(3, 3))),
                    ("activation31", nn.LeakyReLU()),
                    (
                        "identity_conv32",
                        Conv2dSamePadding(256, 256, kernel_size=(1, 1)),
                    ),
                    ("conv32", Conv2dSamePadding(256, 512, kernel_size=(3, 3))),
                    ("activation32", nn.LeakyReLU()),
                    ("pool3", nn.MaxPool2d((2, 2), 2)),
                    (
                        "identity_conv41",
                        Conv2dSamePadding(512, 256, kernel_size=(1, 1)),
                    ),
                    ("conv41", Conv2dSamePadding(256, 512, kernel_size=(3, 3))),
                    ("activation41", nn.LeakyReLU()),
                    (
                        "identity_conv42",
                        Conv2dSamePadding(512, 256, kernel_size=(1, 1)),
                    ),
                    ("conv42", Conv2dSamePadding(256, 512, kernel_size=(3, 3))),
                    ("activation42", nn.LeakyReLU()),
                    (
                        "identity_conv43",
                        Conv2dSamePadding(512, 256, kernel_size=(1, 1)),
                    ),
                    ("conv43", Conv2dSamePadding(256, 512, kernel_size=(3, 3))),
                    ("activation43", nn.LeakyReLU()),
                    (
                        "identity_conv44",
                        Conv2dSamePadding(512, 256, kernel_size=(1, 1)),
                    ),
                    ("conv44", Conv2dSamePadding(256, 512, kernel_size=(3, 3))),
                    ("activation44", nn.LeakyReLU()),
                    (
                        "identity_conv45",
                        Conv2dSamePadding(512, 512, kernel_size=(1, 1)),
                    ),
                    ("conv45", Conv2dSamePadding(512, 1024, kernel_size=(3, 3))),
                    ("activation45", nn.LeakyReLU()),
                    ("pool4", nn.MaxPool2d((2, 2), 2)),
                    (
                        "identity_conv51",
                        Conv2dSamePadding(1024, 512, kernel_size=(1, 1)),
                    ),
                    ("conv51", Conv2dSamePadding(512, 1024, kernel_size=(3, 3))),
                    ("activation51", nn.LeakyReLU()),
                    (
                        "identity_conv52",
                        Conv2dSamePadding(1024, 512, kernel_size=(1, 1)),
                    ),
                    ("conv52", Conv2dSamePadding(512, 1024, kernel_size=(3, 3))),
                    ("activation52", nn.LeakyReLU()),
                    ("conv53", Conv2dSamePadding(1024, 1024, kernel_size=(3, 3))),
                    ("activation53", nn.LeakyReLU()),
                    ("pool5", nn.MaxPool2d((2, 2), 2)),
                    ("conv61", Conv2dSamePadding(1024, 1024, kernel_size=(3, 3))),
                    ("activation61", nn.LeakyReLU()),
                    ("conv62", Conv2dSamePadding(1024, 1024, kernel_size=(3, 3))),
                    ("activation62", nn.LeakyReLU()),
                ]
            )
        )

    def forward(self, x):
        x = self.st_maxpol(self.st_conv(x))
        x = self.net(x)
        return x


class YoloV1(nn.Module):
    """docstring for Yolo."""

    def __init__(self, classes_num, num_boxes=2, grid_size=(7, 7), backbone=None):
        super().__init__()
        if backbone is None:
            backbone = YoloV1Backbone()

        self.backbone = backbone
        self.to_grid = nn.AdaptiveAvgPool2d(grid_size)

        self.classes_num = classes_num
        self.grid_size = grid_size
        self.grid_area = grid_size[0] * grid_size[1]
        self.num_boxes = num_boxes
        self.output_size = 5 * num_boxes + classes_num

        self.conf = LazyConv2d(self.output_size, kernel_size=(1, 1))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.forward = self._train_forward
        else:
            self.forward = self._eval_forward
        return self

    def eval(self):
        super().eval()
        self.forward = self._eval_forward
        return self

    @staticmethod
    def train_to_eval(y):
        _, indices = torch.max(y["labels"], -1)

        labels = indices.unsqueeze(-1)
        num_boxes = y["boxes"].shape[-2]
        batch_size = y["boxes"].shape[0]
        labels = torch.tile(labels, (1, 1, num_boxes))
        labels = labels.reshape(batch_size, -1)

        y["labels"] = labels
        y["boxes"] = y["boxes"].reshape(batch_size, -1, 4)
        y["scores"] = y["scores"].reshape(batch_size, -1)
        return y

    def _eval_forward(self, x):
        y = self._train_forward(x)
        y = self.train_to_eval(y)
        return y

    def _train_forward(self, x):
        x = self.backbone(x)
        x = self.to_grid(x)
        x = self.conf(x)
        batch_size = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.output_size)
        labels = x[..., -self.classes_num :]
        boxes = x[..., : -self.classes_num].reshape(batch_size, self.grid_area, -1, 5)
        confidences = boxes[..., 4:]
        boxes = boxes[..., :4]

        return {
            "boxes": torch.sigmoid(boxes),
            "labels": labels,
            "scores": torch.sigmoid(confidences),
        }

    forward = _train_forward


class YoloV1Loss(nn.Module):
    """docstring for YoloV1Loss."""

    def __init__(self, lambda_coord=5, lambda_noobj=0.5, grid=None, grid_size=(7, 7)):
        if grid is None:
            if grid_size is None:
                raise ValueError(
                    "At least one of `grid_size` and `grid` should be provided"
                )
            grid = make_grid(grid_size)
        super().__init__()

        self.register_buffer("grid", grid)
        self.register_buffer("lambda_coord", torch.tensor(lambda_coord))
        self.register_buffer("lambda_noobj", torch.tensor(lambda_noobj))

    def forward(self, preds, gts):
        classification_losses = []
        box_losses = []
        confidence_losses = []

        for gt_labels, gt_boxes, pred_labels, pred_boxes, pred_confidences in zip(
            gts["labels"],
            gts["boxes"],
            preds["labels"],
            preds["boxes"],
            preds["scores"],
        ):
            max_iou, max_iou_indices = compute_iou_tl_br(
                self.grid, yxhw_to_yxyx(gt_boxes)
            ).max(1)
            classification_losses.append(
                self._classification_loss(
                    pred_labels, gt_labels, max_iou, max_iou_indices
                )
            )
            confidence_losses.append(self._confidence_loss(pred_confidences, max_iou))
            box_losses.append(
                self._box_loss(gt_boxes, pred_boxes, max_iou, max_iou_indices)
            )
        return {
            "box_loss": torch.mean(torch.stack(box_losses)),
            "classification_loss": torch.mean(torch.stack(classification_losses)),
            "confidence_loss": torch.mean(torch.stack(confidence_losses)),
        }

    def _classification_loss(self, pred_labels, gt_labels, max_iou, max_iou_indices):
        entropies = F.cross_entropy(
            pred_labels, gt_labels[max_iou_indices], reduction="none"
        )
        entropies = entropies[max_iou > 0]
        class_loss = entropies.sum()
        return class_loss

    def _box_loss(self, gt_boxes, pred_boxes, max_iou, max_iou_indices):
        gt_boxess = torch.tile(gt_boxes[max_iou_indices].unsqueeze(1), dims=(1, 2, 1))
        dydx = gt_boxess[..., :2] - pred_boxes[..., :2]
        dydx = dydx[max_iou > 0]
        dhdw = torch.sqrt(gt_boxess[..., 2:]) - torch.sqrt(pred_boxes[..., 2:])
        dhdw = dhdw[max_iou > 0]
        pos_loss = torch.pow(dydx, 2).sum()
        size_loss = torch.pow(dhdw, 2).sum()
        return self.lambda_coord * (pos_loss + size_loss)

    def _confidence_loss(self, confidences, max_iou_values):
        confidence_mask = max_iou_values > 0
        gt_confidences = (confidence_mask).type(torch.float32)
        gt_confidences = torch.tile(gt_confidences.unsqueeze(-1), (1, 2))
        confidences = confidences[0].squeeze()

        confidence_loss = torch.pow(
            (gt_confidences - confidences)[confidence_mask],
            2,
        )
        not_confidence_loss = self.lambda_noobj * torch.pow(
            (gt_confidences - confidences)[confidence_mask],
            2,
        )
        return confidence_loss.sum() + not_confidence_loss.sum()


class YoloV1ClassLoss(nn.Module):
    def __init__(self, grid_size=None, grid=None):
        if grid is None:
            if grid_size is None:
                raise ValueError(
                    "At least one of `grid_size` and `grid` should be provided"
                )
            grid = make_grid(grid_size)
        super().__init__()
        self.grid = grid

    def forward(
        self,
        pred_labels,
        gt_labels_list,
        max_ious=None,
        max_ious_indices=None,
        gt_boxes_list=None,
    ):
        entropies = []

        for pred_labels, gt_labels, gt_boxes in zip(
            pred_labels, gt_labels_list, gt_boxes_list
        ):
            max_iou, max_iou_indices = compute_iou_tl_br(self.grid, gt_boxes).max(1)
            entropy = F.cross_entropy(
                pred_labels, gt_labels[max_iou_indices], reduction="none"
            )
            entropy = entropy[max_iou > 0]
            entropies.append(entropy)

        loss_value = torch.sum(torch.hstack(entropies))
        return loss_value


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
        super().__init__(model, device)
        self.label_mapper = label_mapper
        self.id_mapper = {id_: label for label, id_ in label_mapper.items()}
        self.model = model
        self.iou_threshold = iou_threshold
        self.transform = transform
        self.transform_y = transform_y
        self.loss = loss
        self.mean_ap = MeanAveragePrecision()

    def get_label_num(self):
        return len(self.label_mapper)

    def label_to_id(self, label):
        return self.label_mapper[label]

    def id_to_label(self, idd):
        return self.id_mapper[idd]

    def _log_agg_loss(self, loss):
        if isinstance(loss, dict):
            self.log("loss", loss)
            loss = sum(loss.values())
        elif isinstance(loss, list):
            loss = sum(loss)
        self.log("loss", loss)
        return loss

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
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

    def train_step(self, batch, batch_idx=None, epoch_idx=None):
        xs, ys_gt = batch
        ys_gt = to_device(ys_gt, self.device)

        ys_pred = self.forward(xs)
        loss_value = self.loss(ys_pred, ys_gt)
        loss_value = self._log_agg_loss(loss_value)

        return loss_value

    def on_val_begin(self):
        self.mean_ap.reinit()

    def val_step(self, batch, batch_idx=None, epoch_idx=None):
        xs, ys_gt = batch
        ys_gt = to_device(ys_gt, self.device)

        ys_pred = self.forward(xs)
        loss_value = self.loss(ys_pred, ys_gt)
        loss_value = self._log_agg_loss(loss_value)
        ys_pred = to_device(ys_pred, CPU_DEVICE)
        ys_gt = to_device(ys_gt, CPU_DEVICE)
        ys_pred = YoloV1.train_to_eval(ys_pred)
        self.mean_ap.add_batch(ys_gt, ys_pred)
        return loss_value

    def on_val_end(self):
        self.log(value=self.mean_ap.accumulate())
