from itertools import cycle, islice

import torch
from torch import nn
from src.layers.linear import LinearDefered


def init_if_none(x, default):
    return x if x is not None else default


class VGGBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_nums: int,
        ends_with_1x1: bool = False,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._conv_nums = conv_nums
        self._ends_with_1x1 = ends_with_1x1

        self.net = nn.Sequential(*self._build_net())

    def _build_net(self):
        channels_sequence = cycle([self._out_channels])
        channels_sequence = islice(channels_sequence, self._conv_nums)
        layers = []
        prev_channels = self._in_channels
        for out_channels in channels_sequence:
            layers.append(
                nn.Conv2d(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=3, padding=1
                )
            )
            layers.append(nn.ReLU())
            prev_channels = out_channels
        if self._ends_with_1x1:
            layers.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1
                )
            )
        return layers

    def forward(self, input):
        return self.net(input)


VGG_A = [
    {"in_channels": 3, "out_channels": 64, "conv_nums": 1},
    {"out_channels": 128, "conv_nums": 1},
    {"out_channels": 256, "conv_nums": 2},
    {"out_channels": 512, "conv_nums": 2},
    {"out_channels": 512, "conv_nums": 2},
]

VGG_B = [
    {"in_channels": 3, "out_channels": 64, "conv_nums": 2},
    {"out_channels": 128, "conv_nums": 2},
    {"out_channels": 256, "conv_nums": 2},
    {"out_channels": 512, "conv_nums": 2},
    {"out_channels": 512, "conv_nums": 2},
]


VGG_C = [
    {"in_channels": 3, "out_channels": 64, "conv_nums": 2},
    {"out_channels": 128, "conv_nums": 2},
    {"out_channels": 256, "conv_nums": 2, "ends_with_1x1": True},
    {"out_channels": 512, "conv_nums": 2, "ends_with_1x1": True},
    {"out_channels": 512, "conv_nums": 2, "ends_with_1x1": True},
]

VGG_D = [
    {"in_channels": 3, "out_channels": 64, "conv_nums": 2},
    {"out_channels": 128, "conv_nums": 2},
    {"out_channels": 256, "conv_nums": 3},
    {"out_channels": 512, "conv_nums": 3},
    {"out_channels": 512, "conv_nums": 3},
]

VGG_E = [
    {"in_channels": 3, "out_channels": 64, "conv_nums": 2},
    {"out_channels": 128, "conv_nums": 2},
    {"out_channels": 256, "conv_nums": 4},
    {"out_channels": 512, "conv_nums": 4},
    {"out_channels": 512, "conv_nums": 4},
]

DEFAULT_CONFIG = VGG_C


class VGG(nn.Module):
    def __init__(
        self,
        classes_num,
        configuration=None,
    ):
        super().__init__()
        self.fcn = nn.Sequential(
            *self._build_fcn(init_if_none(configuration, DEFAULT_CONFIG))
        )
        self.ffn = nn.Sequential(
            *[
                LinearDefered(out_features=4096),
                nn.Linear(in_features=4096, out_features=4096),
                nn.Linear(in_features=4096, out_features=classes_num),
            ]
        )

    def _build_block(self, layer_config, prev_in_channels):
        in_channels = init_if_none(layer_config.get("in_channels"), prev_in_channels)
        out_channels = init_if_none(layer_config["out_channels"], in_channels)
        conv_nums = init_if_none(layer_config["conv_nums"], in_channels)
        ends_with_1x1 = init_if_none(layer_config.get("ends_with_1x1"), False)
        return VGGBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_nums=conv_nums,
            ends_with_1x1=ends_with_1x1,
        )

    def _build_fcn(self, configuration):
        prev_channels = configuration[0]["in_channels"]
        layers = []
        for layer_config in configuration:
            layers.append(self._build_block(layer_config, prev_channels))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = layer_config["out_channels"]
        return layers

    def forward(self, input):
        return self.ffn(torch.flatten(self.fcn(input), start_dim=1))


def make_vgg_11_a(classes_num):
    return VGG(classes_num, configuration=VGG_A)


def make_vgg_13_b(classes_num):
    return VGG(classes_num, configuration=VGG_B)


def make_vgg_16_c(classes_num):
    return VGG(classes_num, configuration=VGG_C)


def make_vgg_16_d(classes_num):
    return VGG(classes_num, configuration=VGG_D)


def make_vgg_19_e(classes_num):
    return VGG(classes_num, configuration=VGG_E)
