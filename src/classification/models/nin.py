from typing import NamedTuple, List, Union
from torch import nn

from ...layers.lazy import LazyConv2d


class MlpConv(nn.Module):
    def __init__(self, out_channels_list, kernel_size, stride, padding):
        super().__init__()

        self.net = self._build_net(out_channels_list, kernel_size, stride, padding)

    def _build_net(self, out_channels_list, kernel_size, stride, padding):
        layers = [
            LazyConv2d(
                out_channels=out_channels_list[0],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
        ]
        for out_channels in out_channels_list[1:]:
            layers.extend(
                [
                    LazyConv2d(
                        out_channels=out_channels,
                        kernel_size=1,
                    ),
                    nn.ReLU(),
                ]
            )
        return nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class LayerConfg(NamedTuple):
    channels_list: List[int]
    kernel_size: int
    stride: int
    padding: int


NIN_CONFIGURATION = [
    LayerConfg([96, 96, 96], 11, 4, 0),
    LayerConfg([256, 256, 256], 5, 1, 2),
    LayerConfg([384, 384, 384], 3, 1, 1),
]


class NiN(nn.Module):
    def __init__(
        self, classes_num: int, configuration: List[LayerConfg] = NIN_CONFIGURATION
    ) -> None:
        super().__init__()

        layers = self._build_from_configuration(configuration)
        layers.extend(
            [
                nn.Dropout(0.5),
                MlpConv([classes_num, classes_num, classes_num], 3, 1, 1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def _build_from_configuration(
        self, configuration: List[LayerConfg]
    ) -> List[Union[MlpConv, nn.MaxPool2d]]:
        layers = []
        for layer_config in configuration:
            layers.extend(
                [
                    MlpConv(
                        out_channels_list=layer_config.channels_list,
                        kernel_size=layer_config.kernel_size,
                        stride=layer_config.stride,
                        padding=layer_config.padding,
                    ),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ]
            )
        return layers

    def forward(self, input):
        return self.net(input)
