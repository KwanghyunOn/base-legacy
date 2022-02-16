import math
import torch
from torch.distributions import Normal

from survae.distributions import Distribution, ConditionalDistribution
from survae.utils import sum_except_batch


class GeneralNormal(Distribution):
    """A multivariate Normal with scalar mean and covariance."""

    def __init__(self, loc=0.0, scale=1.0):
        super().__init__()
        self.register_buffer("loc", torch.tensor(loc))
        self.register_buffer("scale", torch.tensor(scale))

    def log_prob(self, x):
        var = self.scale**2
        log_scale = math.log(self.scale)
        elementwise_log_prob = (
            -((x - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
        return sum_except_batch(elementwise_log_prob)

    def sample(self, shape):
        return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def sample_with_log_prob(self, shape):
        samples = self.sample(shape)
        log_prob = self.log_prob(samples)
        return samples, log_prob


class NormalAuxSampleDist(Distribution):
    """
    A multivariate Normal with scalar mean and covariance.
    Can use different normal distribution for sampling.
    """

    def __init__(self, loc, scale, loc_sample, scale_sample, shape=None):
        super().__init__()
        self.register_buffer("loc", torch.tensor(loc, dtype=torch.float))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))
        self.register_buffer("loc_sample", torch.tensor(loc_sample, dtype=torch.float))
        self.register_buffer(
            "scale_sample", torch.tensor(scale_sample, dtype=torch.float)
        )

    def log_prob(self, x):
        var = self.scale**2
        log_scale = math.log(self.scale)
        elementwise_log_prob = (
            -((x - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
        return sum_except_batch(elementwise_log_prob)

    def sample(self, shape):
        return torch.normal(
            self.loc_sample.expand(shape), self.scale_sample.expand(shape)
        )


class ReLUNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=1):
        super().__init__()
        self.net = net
        self.split_dim = split_dim

    def cond_dist(self, context):
        params = self.net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)
        # print(f"mean: {mean.mean():10.3f}, log_std: {log_std.mean():10.3f}")
        return Normal(loc=mean, scale=log_std.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        ldj = dist.log_prob(x)
        ldj[x >= 0] = 0.0
        return sum_except_batch(ldj)

    def sample(self, shape, context):
        dist = self.cond_dist(context)
        z = dist.rsample(shape=shape)
        z[z >= 0] = 0.0
        return z

    def sample_with_log_prob(self, shape, context):
        dist = self.cond_dist(context)
        z = dist.rsample(shape=shape)
        z[z >= 0] = 0.0
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev
