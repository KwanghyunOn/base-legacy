import math

from survae.distributions import Distribution, ConditionalDistribution
from survae.flows import ConditionalFlow
from survae.transforms import (
    ConditionalTransform,
    ConditionalSequentialTransform,
    Squeeze2d,
    ActNormBijection2d,
    Conv1x1,
    ConditionalAffineBijection,
)

from .distributions import NormalAuxSampleDist
from .layers import ConditionalAffineCoupling, ConditionalSplit2d
from .nets import RRDBEncoder, SimpleRRDB, UpsamplerNet, DownsamplerNet
from .srflow_modules import SRFlowConvNet, Conv2dZeros


class SRFlowStep(ConditionalSequentialTransform):
    def __init__(self, in_channels, context_channels):
        transforms = [
            ActNormBijection2d(in_channels),
            Conv1x1(in_channels),
            self.get_affine_injector(in_channels, context_channels),
            self.get_affine_coupling(in_channels, context_channels),
        ]
        super(SRFlowStep, self).__init__(transforms=transforms)

    @staticmethod
    def get_affine_injector(
        in_channels, context_channels, hidden_channels=64, num_hidden_layers=1
    ):
        context_net = SRFlowConvNet(
            in_channels=context_channels,
            out_channels=in_channels * 2,
            hidden_channels=hidden_channels,
            num_hidden_layers=num_hidden_layers,
        )
        return ConditionalAffineBijection(context_net)

    @staticmethod
    def get_affine_coupling(
        in_channels, context_channels, hidden_channels=64, num_hidden_layers=1
    ):
        cond_channels = in_channels // 2
        coupling_net = SRFlowConvNet(
            in_channels=cond_channels + context_channels,
            out_channels=(in_channels - cond_channels) * 2,
            hidden_channels=hidden_channels,
            num_hidden_layers=num_hidden_layers,
        )
        return ConditionalAffineCoupling(
            coupling_net=coupling_net, context_net=None, num_condition=cond_channels
        )


class TransitionStep(ConditionalSequentialTransform):
    def __init__(self, in_channels):
        transforms = [ActNormBijection2d(in_channels), Conv1x1(in_channels)]
        super(TransitionStep, self).__init__(transforms=transforms)


class SRFlowLevel(ConditionalSequentialTransform):
    def __init__(
        self, scale_level, in_channels, context_channels, num_flow_steps, split, temp
    ):
        transforms = []

        # Squeeze
        transforms.append(Squeeze2d(factor=2))
        in_channels = 4 * in_channels

        # Flow steps
        transforms.append(TransitionStep(in_channels))
        for _ in range(num_flow_steps):
            transforms.append(SRFlowStep(in_channels, context_channels))

        # Split
        if split:
            transforms.append(self.get_split2d(in_channels, context_channels, temp))

        super(SRFlowLevel, self).__init__(transforms=transforms)

        # scale_level = log_2(context_shape / lr_shape)
        self.scale_level = scale_level

    @staticmethod
    def get_split2d(in_channels, context_channels, temp):
        out_channels = in_channels // 2
        cond_channels = in_channels - out_channels
        split_dist = NormalAuxSampleDist(
            loc=0, scale=1, loc_sample=0, scale_sample=temp
        )
        context_net = Conv2dZeros(
            in_channels=out_channels + context_channels, out_channels=cond_channels * 2
        )
        return ConditionalSplit2d(
            split_dist=split_dist,
            context_net=context_net,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def _preprocess_context(self, context):
        """
        Overrides ConditionalSequentialTransform._preprocess_context()
        """
        return context[self.scale_level]


class SRFlowModel(ConditionalFlow):
    def __init__(self, scale, base_dist, context_channels, num_flow_steps, temp):
        transforms = []
        num_levels = round(math.log(scale, 2))
        in_channels = 3

        self.scale_2d = scale  # Ratio of the 2d shape of LR image and latent variable
        for scale_level in reversed(range(-1, num_levels)):
            if scale_level > 0:
                split = True
                out_channels = 2 * in_channels
            else:
                split = False
                out_channels = 4 * in_channels

            transforms.append(
                SRFlowLevel(
                    scale_level=scale_level,
                    in_channels=in_channels,
                    context_channels=context_channels,
                    num_flow_steps=num_flow_steps,
                    split=split,
                    temp=temp,
                )
            )
            in_channels = out_channels
            self.scale_2d /= 2
        self.out_channels = out_channels

        super().__init__(base_dist=base_dist, transforms=transforms)

    def sample(self, context, lr_shape):
        b, c, h, w = lr_shape
        shape = (b, self.out_channels, int(h * self.scale_2d), int(w * self.scale_2d))
        if isinstance(self.base_dist, ConditionalDistribution):
            z = self.base_dist.sample(context, shape)
        else:
            z = self.base_dist.sample(shape)
        for transform in reversed(self.transforms):
            if isinstance(transform, ConditionalTransform):
                z = transform.inverse(z, context)
            else:
                z = transform.inverse(z)
        return z


class SRFlow(Distribution):
    def __init__(self, scale, sample_temp, num_flow_steps):
        super().__init__()
        self.lr_encoder = RRDBEncoder(
            scale, num_rrdb_blocks=23, block_idxs=[1, 8, 15, 22]
        )
        context_channels = self.lr_encoder.out_channels()
        base_dist = self.get_base_dist(sample_temp)
        self.flow = SRFlowModel(
            scale, base_dist, context_channels, num_flow_steps, temp=sample_temp
        )

    @staticmethod
    def get_base_dist(temp):
        return NormalAuxSampleDist(loc=0, scale=1, loc_sample=0, scale_sample=temp)

    def log_prob(self, hr, lr):
        lr_context = self.lr_encoder(lr)
        return self.flow.log_prob(hr, lr_context)

    def sample(self, lr):
        lr_context = self.lr_encoder(lr)
        sr = self.flow.sample(lr_context, lr.shape)
        sr = sr.clamp(min=-1.0, max=1.0)
        return sr

    def loglik_bpd(self, hr, lr):
        return -self.log_prob(hr, lr).sum() / (math.log(2) * hr.shape.numel())
    
    def forward(self, hr=None, lr=None, sample=False):
        if sample:
            return self.sample(lr)
        else:
            return self.loglik_bpd(hr, lr)


class SimpleSRFlowLevel(SRFlowLevel):
    def __init__(
        self, scale_level, in_channels, context_channels, num_flow_steps, split, temp
    ):
        super().__init__(
            scale_level, in_channels, context_channels, num_flow_steps, split, temp
        )
        if self.scale_level >= 0:
            self.context_net = UpsamplerNet(
                log_scale=scale_level, in_channels=context_channels
            )
        else:
            self.context_net = DownsamplerNet(
                log_scale=(-scale_level), in_channels=context_channels
            )

    def _preprocess_context(self, context):
        return self.context_net(context)


class SimpleSRFlowModel(ConditionalFlow):
    def __init__(self, scale, base_dist, context_channels, num_flow_steps, temp):
        transforms = []
        num_levels = round(math.log(scale, 2))
        in_channels = 3

        self.scale_2d = scale  # Ratio of the 2d shape of LR image and latent variable
        for scale_level in reversed(range(-1, num_levels)):
            if scale_level > 0:
                split = True
                out_channels = 2 * in_channels
            else:
                split = False
                out_channels = 4 * in_channels

            transforms.append(
                SimpleSRFlowLevel(
                    scale_level=scale_level,
                    in_channels=in_channels,
                    context_channels=context_channels,
                    num_flow_steps=num_flow_steps,
                    split=split,
                    temp=temp,
                )
            )
            in_channels = out_channels
            self.scale_2d /= 2
        self.out_channels = out_channels

        super().__init__(base_dist=base_dist, transforms=transforms)

    def sample(self, context, lr_shape):
        b, c, h, w = lr_shape
        shape = (b, self.out_channels, int(h * self.scale_2d), int(w * self.scale_2d))
        if isinstance(self.base_dist, ConditionalDistribution):
            z = self.base_dist.sample(context, shape)
        else:
            z = self.base_dist.sample(shape)
        for transform in reversed(self.transforms):
            if isinstance(transform, ConditionalTransform):
                z = transform.inverse(z, context)
            else:
                z = transform.inverse(z)
        return z


class SimpleSRFlow(Distribution):
    def __init__(self, scale, sample_temp, num_flow_steps):
        super().__init__()
        self.lr_encoder = SimpleRRDB()
        context_channels = self.lr_encoder.out_channels()
        base_dist = self.get_base_dist(sample_temp)
        self.flow = SimpleSRFlowModel(
            scale, base_dist, context_channels, num_flow_steps, temp=sample_temp
        )

    @staticmethod
    def get_base_dist(temp):
        return NormalAuxSampleDist(loc=0, scale=1, loc_sample=0, scale_sample=temp)

    def log_prob(self, hr, lr):
        lr_context = self.lr_encoder(lr)
        return self.flow.log_prob(hr, lr_context)

    def sample(self, lr):
        lr_context = self.lr_encoder(lr)
        sr = self.flow.sample(lr_context, lr.shape)
        sr = sr.clamp(min=-1.0, max=1.0)
        return sr

    def loglik_bpd(self, hr, lr):
        return -self.log_prob(hr, lr).sum() / (math.log(2) * hr.shape.numel())

    def forward(self, hr=None, lr=None, sample=False):
        if sample:
            return self.sample(lr)
        else:
            return self.loglik_bpd(hr, lr)
