import math

import torch
from torch import nn
import torch.nn.functional as F

from .rrdb import RRDBNet, RRDB


class Conv2dZero(nn.Conv2d):
    """
    Zero initialized convolution layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.data.zero_()
        self.bias.data.zero_()


class ConvNet(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers):

        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
        ]
        for _ in range(num_hidden_layers - 1):
            layers.extend(
                [
                    nn.Conv2d(
                        hidden_channels, hidden_channels, kernel_size=1, bias=False
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=False),
                ]
            )
        layers.extend(
            [
                Conv2dZero(hidden_channels, out_channels, kernel_size=3, padding=1),
            ]
        )

        super(ConvNet, self).__init__(*layers)


class RRDBEncoder(nn.Module):
    def __init__(self, scale, num_rrdb_blocks, block_idxs):
        super().__init__()
        self.rrdb = RRDBNet(
            in_nc=3,
            out_nc=3,
            nf=64,
            nb=num_rrdb_blocks,
            gc=32,
            scale=scale,
            block_idxs=block_idxs,
        )
        self.num_levels = round(math.log(scale, 2))
        self.keymap = {
            -1: "fea_up0",
            0: "fea_up1",
            1: "fea_up2",
            2: "fea_up4",
            3: "fea_up8",
        }

    def forward(self, x):
        rrdb_results = self.rrdb(x, get_steps=True)
        context = {}
        for scale_level in range(-1, self.num_levels):
            k = self.keymap[scale_level]
            shape = rrdb_results[k].shape[-2:]
            context[scale_level] = torch.cat(
                [rrdb_results[k], F.interpolate(rrdb_results["concat"], shape)], dim=1
            )
        return context

    def out_channels(self):
        return self.rrdb.nf * (1 + len(self.rrdb.block_idxs))

    def load_pretrained_encoder(self, state_dict):
        self.rrdb.load_state_dict(state_dict)


class SimpleRRDB(nn.Module):
    def __init__(self, in_channels=3, num_rrdb_blocks=[8, 8], nf=64, gc=32):
        super().__init__()
        self.nf = nf
        self.conv_head = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=True)
        self.rrdb0 = nn.Sequential(
            *[RRDB(nf=nf, gc=gc) for _ in range(num_rrdb_blocks[0])]
        )
        self.rrdb1 = nn.Sequential(
            *[RRDB(nf=nf, gc=gc) for _ in range(num_rrdb_blocks[1])]
        )
        self.conv_tail = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        feat0 = self.conv_head(x)
        feat1 = self.rrdb0(feat0)
        feat2 = self.conv_tail(self.rrdb1(feat1)) + feat0
        return torch.cat([feat1, feat2], dim=1)

    def out_channels(self):
        return self.nf * 2


class UpsamplerNet(nn.Sequential):
    def __init__(self, log_scale, in_channels):
        layers = []
        for _ in range(log_scale):
            layers.append(
                nn.Conv2d(
                    in_channels, 4 * in_channels, kernel_size=3, padding=1, bias=True
                )
            )
            layers.append(nn.PixelShuffle(2))
        super().__init__(*layers)


class DownsamplerNet(nn.Sequential):
    def __init__(self, log_scale, in_channels):
        layers = []
        for _ in range(log_scale):
            layers.append(nn.PixelUnshuffle(2))
            layers.append(
                nn.Conv2d(
                    4 * in_channels, in_channels, kernel_size=3, padding=1, bias=True
                )
            )
        super().__init__(*layers)


class SRSimpleRRDB(nn.Module):
    def __init__(self, scale, **rrdb_kwargs):
        super().__init__()
        self.scale = scale
        log_scale = round(math.log2(scale))
        self.rrdb = SimpleRRDB(**rrdb_kwargs)
        feat_channels = self.rrdb.out_channels()
        self.upsampler = nn.Sequential(
            UpsamplerNet(log_scale=log_scale, in_channels=feat_channels),
            nn.Conv2d(
                in_channels=feat_channels,
                out_channels=3,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

    def forward(self, x):
        x = self.rrdb(x)
        x = self.upsampler(x)
        return x
