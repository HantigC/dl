import torch
from torch import nn
from collections import OrderedDict
from src.layers.lazy import LazyConv2d
from src.layers.xtended import Conv2dSamePadding


class ResConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.conv1 = Conv2dSamePadding(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
            )
            self.shortcut = nn.Sequential(
                Conv2dSamePadding(in_channels, out_channels, kernel_size=1, stride=2),
                nn.ReLU(),
            )
        else:
            self.conv1 = Conv2dSamePadding(
                in_channels, out_channels, kernel_size=kernel_size, stride=1
            )
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    Conv2dSamePadding(in_channels, out_channels, kernel_size=1),
                    nn.ReLU(),
                )
            else:
                self.shortcut = nn.Sequential()
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2dSamePadding(
            out_channels, out_channels, kernel_size=kernel_size
        )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.end_relu = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return self.end_relu(out + self.shortcut(x))


def repeat_conv_block(block, no, channels_in, channels_out, **kwargs):
    layers = [block(channels_in, channels_out, **kwargs)]
    for _ in range(no - 1):
        layers.append(block(channels_out, channels_out, **kwargs))

    return nn.Sequential(*layers)


class Resnet(nn.Module):
    """docstring for Resnet"""

    def __init__(
        self,
        first_kernel_size=7,
        first_channels=64,
        kernel_size=(3, 3),
        channels_growth_rate=2,
        block_sizes=(3, 4, 6, 3),
        mini_bloc=None,
    ):
        super().__init__()
        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.first_channels = first_channels
        self.channels_growth_rate = channels_growth_rate
        self.block_sizes = block_sizes
        self.conv = Conv2dSamePadding(3, first_channels, kernel_size=first_kernel_size)
        self.net = self._make_net()

    def _make_net(self):
        channels = self.first_channels
        next_channels = int(channels * self.channels_growth_rate)
        layers = [
            (
                "block_no_1",
                repeat_conv_block(
                    ResConv2d,
                    self.block_sizes[0],
                    channels,
                    channels,
                    kernel_size=self.kernel_size,
                ),
            )
        ]
        for num, block_size in enumerate(self.block_sizes[1:], 2):
            layers.append(
                (
                    f"downsample_no_{num}",
                    ResConv2d(
                        channels,
                        next_channels,
                        kernel_size=self.kernel_size,
                        downsample=True,
                    ),
                ),
            )
            layers.append(
                (
                    f"block_no_{num}",
                    repeat_conv_block(
                        ResConv2d,
                        block_size - 1,
                        next_channels,
                        next_channels,
                        kernel_size=self.kernel_size,
                    ),
                ),
            )
            channels = next_channels
            next_channels = int(channels * self.channels_growth_rate)
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        out = self.conv(x)
        out = self.net(out)
        return out


class PredAvgModule(nn.Module):
    def __init__(self, out_channels, in_channels=None):
        super().__init__()
        self.ga = nn.AdaptiveAvgPool2d(1)
        self.conv = LazyConv2d(out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.ga(x)
        x = torch.flatten(x, start_dim=1)
        return x


class ResnetClassification(nn.Module):
    def __init__(
        self,
        classes,
        first_kernel_size=7,
        first_channels=64,
        kernel_size=(3, 3),
        channels_growth_rate=2,
        block_sizes=(3, 4, 6, 3),
        mini_bloc=None,
    ):
        super().__init__()
        self.resnet = Resnet(
            first_kernel_size,
            first_channels,
            kernel_size,
            channels_growth_rate,
            block_sizes,
            mini_bloc,
        )
        self.to_flatt = PredAvgModule(classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.to_flatt(x)
        return x
