import typing
import random
import math

import torch
from torch import nn


def default_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int=1,
        padding: typing.Optional[int]=None,
        bias=True,
        padding_mode: str='zeros'):

    if padding is None:
        padding = (kernel_size - 1) // 2

    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        padding_mode=padding_mode,
    )
    return conv


def append_module(m, name, n_feats):
    if name is None:
        return

    if name == 'batch':
        m.append(nn.BatchNorm2d(n_feats))
    elif name == 'layer':
        m.append(nn.GroupNorm(1, n_feats))
    elif name == 'instance':
        m.append(nn.InstanceNorm2d(n_feats))

    if name == 'relu':
        m.append(nn.ReLU(True))
    elif name == 'lrelu':
        m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    elif name == 'prelu':
        m.append(nn.PReLU())


class ResBlock(nn.Sequential):
    '''
    Make a residual block which consists of Conv-(Norm)-Act-Conv-(Norm).

    Args:
        n_feats (int): Conv in/out_channels.
        kernel_size (int): Conv kernel_size.
        norm (<None> or 'batch' or 'layer'): Norm function.
        act (<'relu'> or 'lrelu' or 'prelu'): Activation function.
        res_scale (float, optional): Residual scaling.
        conv (funcion, optional): A function for making a conv layer.

    Note:
        Residual scaling:
        From Szegedy et al.,
        "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"
        See https://arxiv.org/pdf/1602.07261.pdf for more detail.

        To modify stride, change the conv function.
    '''

    def __init__(
            self,
            n_feats: int,
            kernel_size: int,
            norm: typing.Optional[str]=None,
            act: str='relu',
            res_scale: float=1,
            res_prob: float=1,
            padding_mode: str='zeros',
            conv=default_conv) -> None:

        bias = norm is None
        m = []
        for i in range(2):
            m.append(conv(
                n_feats,
                n_feats,
                kernel_size,
                bias=bias,
                padding_mode=padding_mode,
            ))
            append_module(m, norm, n_feats)
            if i == 0:
                append_module(m, act, n_feats)

        super().__init__(*m)
        self.res_scale = res_scale
        self.res_prob = res_prob
        return

    def forward(self, x):
        if self.training and random.random() > self.res_prob:
            return x

        x = x + self.res_scale * super(ResBlock, self).forward(x)
        return x


class Upsampler(nn.Sequential):
    '''
    Make an upsampling block using sub-pixel convolution
    
    Args:

    Note:
        From Shi et al.,
        "Real-Time Single Image and Video Super-Resolution
        Using an Efficient Sub-pixel Convolutional Neural Network"
        See https://arxiv.org/pdf/1609.05158.pdf for more detail
    '''

    def __init__(
            self,
            scale: int,
            n_feats: int,
            norm: typing.Optional[str]=None,
            act: typing.Optional[str]=None,
            bias: bool=True,
            padding_mode: str='zeros',
            conv=default_conv):

        bias = norm is None
        m = []
        log_scale = math.log(scale, 2)
        # check if the scale is power of 2
        if int(log_scale) == log_scale:
            for _ in range(int(log_scale)):
                m.append(conv(
                    n_feats,
                    4 * n_feats,
                    3,
                    bias=bias,
                    padding_mode=padding_mode,
                ))
                m.append(nn.PixelShuffle(2))
                append_module(m, norm, n_feats)
                append_module(m, act, n_feats)
        elif scale == 3:
            m.append(conv(
                n_feats,
                9 * n_feats,
                3,
                bias=bias,
                padding_mode=padding_mode,
            ))
            m.append(nn.PixelShuffle(3))
            append_module(m, norm, n_feats)
            append_module(m, act, n_feats)
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    '''
    EDSR model

    Note:
        From Lim et al.,
        "Enhanced Deep Residual Networks for Single Image Super-Resolution"
        See https://arxiv.org/pdf/1707.02921.pdf for more detail.
    '''

    def __init__(
            self,
            scale: int=4,
            depth: int=16,
            n_colors: int=3,
            n_feats: int=64,
            res_scale: float=1,
            res_prob: float=1,
            conv=default_conv):

        super().__init__()
        self.n_colors = n_colors
        self.conv = conv(n_colors, n_feats, 3)
        resblock = lambda: ResBlock(
            n_feats, 3, conv=conv, res_scale=res_scale, res_prob=res_prob,
        )
        m = [resblock() for _ in range(depth)]
        m.append(conv(n_feats, n_feats, 3))
        self.resblocks = nn.Sequential(*m)
        self.recon = nn.Sequential(
            Upsampler(scale, n_feats, conv=conv),
            conv(n_feats, n_colors, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x + self.resblocks(x)
        x = self.recon(x)
        return x 
