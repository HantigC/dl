from torch import nn


class LinearDefered(nn.Module):

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
