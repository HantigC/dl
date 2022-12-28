from torch import nn

from ...layers.lazy import LazyConv2d


class MlpConv(nn.Module):
    def __init__(self, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            LazyConv2d(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            LazyConv2d(
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            LazyConv2d(
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.net(input)


class NiN(nn.Module):
    def __init__(self, classes_num: int):
        super().__init__()
        self.net = nn.Sequential(
            MlpConv(96, 11, 4, 0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MlpConv(256, 5, 1, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MlpConv(384, 3, 1, 1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            MlpConv(classes_num, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, input):
        return self.net(input)
