from torch import nn


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kernel_size = kwargs["kernel_size"]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size, (tuple, list)):
            raise ValueError(
                f"kernel size should be int or tuple. Actual type: {kernel_size.__class__.__name__}"
            )
        kwargs["padding"] = [k // 2 for k in kernel_size]
        super().__init__(*args, **kwargs)
