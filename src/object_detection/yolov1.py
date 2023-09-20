from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from src.layers.xtended import Conv2dSamePadding
from src.layers.lazy import LazyConv2d
from .utils.box import make_grid, compute_iou_tl_br


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

    def forward(self, x):
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
            "boxes": boxes,
            "labels": labels,
            "confidences": confidences,
        }


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
        gt_boxes_list,
    ):
        max_iou_list = []
        max_iou_indices_list = []
        entropies = []

        for pred_labels, gt_labels, gt_boxes in zip(
            pred_labels, gt_labels_list, gt_boxes_list
        ):
            max_iou, max_iou_indices = compute_iou_tl_br(self.grid, gt_boxes).max(1)
            max_iou_list.append(max_iou)
            max_iou_indices_list.append(gt_labels[max_iou_indices])
            entropy = F.cross_entropy(
                pred_labels, gt_labels[max_iou_indices], reduction="none"
            )
            entropy = entropy[max_iou > 0]
            entropies.append(entropy)

        loss_value = torch.mean(torch.hstack(entropies))

        return loss_value
