import torch

from survae.distributions import ConditionalDistribution
from survae.transforms import (
    Surjection,
    ConditionalTransform,
    ConditionalAffineBijection,
    ConditionalAffineCouplingBijection,
    ConditionalSurjection,
)


class ConditionalAffineCoupling(ConditionalAffineCouplingBijection):
    def __init__(self, coupling_net, context_net, num_condition):
        super().__init__(coupling_net, context_net, num_condition=num_condition)

    def _unconstrained_scale_and_shift(self, elementwise_params):
        unconstrained_scale, shift = torch.chunk(elementwise_params, chunks=2, dim=1)
        return unconstrained_scale, shift


class ConditionalSplit2d(ConditionalSurjection):
    stochastic_forward = False

    def __init__(self, split_dist, context_net, in_channels, out_channels, dim=1):
        super().__init__()
        self.split_dist = split_dist
        self.transform = ConditionalAffineBijection(context_net)
        self.dim = dim
        self.out_channels = out_channels
        self.cond_channels = in_channels - out_channels

    def split_input(self, input):
        split_proportions = (
            self.out_channels,
            input.shape[self.dim] - self.out_channels,
        )
        return torch.split(input, split_proportions, dim=self.dim)

    def forward(self, x, context):
        z, x2 = self.split_input(x)
        context = torch.cat([z, context], dim=self.dim)
        log_prob = torch.zeros(x.shape[0], device=x.device)

        if isinstance(self.transform, ConditionalTransform):
            x2, ldj = self.transform(x2, context)
        else:
            x2, ldj = self.transfrom(x2)
        log_prob += ldj

        if isinstance(self.split_dist, ConditionalDistribution):
            log_prob += self.split_dist.log_prob(x2, context)
        else:
            log_prob += self.split_dist.log_prob(x2)

        return z, log_prob

    def inverse(self, z, context):
        context = torch.cat([z, context], dim=self.dim)
        shape = (self.cond_channels, *z.shape[2:])
        if isinstance(self.split_dist, ConditionalDistribution):
            x2 = self.split_dist.sample(context, shape)
        else:
            x2 = self.split_dist.sample(shape)

        if isinstance(self.transform, ConditionalTransform):
            x2 = self.transform.inverse(x2, context)
        else:
            x2 = self.transfrom.inverse(x2)
        x = torch.cat([z, x2], dim=self.dim)
        return x


class ChannelAugment(Surjection):
    stochastic_forward = True

    def __init__(self, encoder, in_channels, aug_channels, split_dim=1):
        super().__init__()
        self.encoder = encoder
        self.in_channels = in_channels
        self.aug_channels = aug_channels
        self.split_dim = split_dim
        self.cond = isinstance(self.encoder, ConditionalDistribution)

    def split_z(self, z):
        assert z.shape[self.split_dim] == self.in_channels + self.aug_channels
        split_proportions = (self.in_channels, self.aug_channels)
        return torch.split(z, split_proportions, dim=self.split_dim)

    def forward(self, x):
        shape = (x.shape[0], self.aug_channels, *x.shape[2:4])
        if self.cond:
            z2, logqz2 = self.encoder.sample_with_log_prob(context=x, shape=shape)
        else:
            z2, logqz2 = self.encoder.sample_with_log_prob(shape)
        z = torch.cat([x, z2], dim=self.split_dim)
        ldj = -logqz2
        return z, ldj

    def inverse(self, z):
        x, z2 = self.split_z(z)
        return x


class ReLU(Surjection):
    stochastic_forward = False

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.cond = isinstance(self.decoder, ConditionalDistribution)

    def split_input(self, x):
        zero_tensor = torch.zeros(1).to(x.device)
        pos = torch.where(x >= 0, x, zero_tensor)
        neg = torch.where(x < 0, x, zero_tensor)
        return pos, neg

    def forward(self, x):
        z, x_neg = self.split_input(x)
        if self.cond:
            ldj = self.decoder.log_prob(x_neg, context=z)
        else:
            ldj = self.decoder.log_prob(x_neg)
        return z, ldj

    def inverse(self, z):
        z_pos = torch.where(z >= 0, z, 0.0)
        if self.cond:
            x_neg = self.decoder.sample(shape=z.shape, context=z_pos)
        else:
            x_neg = self.decoder.sample(shape=z.shape)
        x = torch.where(z >= 0, z_pos, x_neg)
        return x


class CouplingReLU(ConditionalSurjection):
    stochastic_forward = False

    def __init__(self, decoder, cond_channels, dim=1):
        super().__init__()
        self.decoder = decoder
        self.cond_channels = cond_channels
        self.dim = dim

    def split_input(self, x):
        split_proportions = (self.cond_channels, x.shape[self.dim] - self.cond_channels)
        x_cond, x2 = x.split(split_proportions, dim=self.dim)
        zero_tensor = torch.zeros(1).to(x2.device)
        x2_pos = torch.where(x2 >= 0.0, x2, zero_tensor)
        x2_neg = torch.where(x2 < 0.0, x2, zero_tensor)
        return x_cond, x2_pos, x2_neg

    def forward(self, x, context):
        x_cond, x2_pos, x2_neg = self.split_input(x)
        context = torch.cat([x_cond, x2_pos, context], dim=self.dim)
        ldj = self.decoder.log_prob(x2_neg, context=context)
        z = torch.cat([x_cond, x2_pos], dim=self.dim)
        print(f"ldj: {ldj.mean():10.3f}")
        return z, ldj

    def inverse(self, z, context):
        z_cond, z2_pos, z2_neg = self.split_input(z)
        context = torch.cat([z_cond, z2_pos, context], dim=self.dim)
        x2_neg = self.decoder.sample(shape=z2_neg.shape, context=context)
        x2 = torch.where(z2_pos >= 0, z2_pos, x2_neg)
        x = torch.cat([z_cond, x2], dim=self.dim)
        return x
