from torch import nn


class LazyLinear(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=True):
        super().__init__()
        self.linear = None
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        if in_features is not None:
            self.linear = nn.Linear(in_features, out_features, bias)

    def _get_linear(self, x):
        if self.linear is None:
            self.linear = nn.Linear(x.shape[1], self.out_features, self.bias)
            self.linear.to(x.device)
        return self.linear

    def forward(self, x):
        return self._get_linear(x)(x)


class LazyConv2d(nn.Module):
    def __init__(self, out_channels, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.kwargs = kwargs
        self.net = None
        if kwargs.get("in_channels") is not None:
            self.net = nn.Conv2d(out_channels=out_channels, **kwargs)

    def _get_conv(self, input):
        if self.net is None:
            self.net = nn.Conv2d(
                in_channels=input.size(1), out_channels=self.out_channels, **self.kwargs
            )
            self.net.to(input.device)
        return self.net

    def forward(self, input):
        return self._get_conv(input)(input)
