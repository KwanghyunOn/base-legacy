import torch
from torch import nn

from .actnorm import ActNorm2d


class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [
            ((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)
        ],
        "valid": lambda kernel, stride: [0 for _ in kernel],
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding="same",
        do_actnorm=True,
        weight_std=0.05,
    ):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding="same",
        logscale_factor=3,
    ):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class SRFlowConvNet(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers):

        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(num_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))
        super().__init__(*layers)
