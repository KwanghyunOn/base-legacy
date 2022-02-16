import torch
from torch import nn
from collections.abc import Iterable
from survae.transforms import Transform


class ConditionalTransform(Transform):
    """Base class for ConditionalTransform"""

    has_inverse = True

    @property
    def bijective(self):
        raise NotImplementedError()

    @property
    def stochastic_forward(self):
        raise NotImplementedError()

    @property
    def stochastic_inverse(self):
        raise NotImplementedError()

    @property
    def lower_bound(self):
        return self.stochastic_forward

    def forward(self, x, context):
        """
        Forward transform.
        Computes `z = f(x|context)` and `log|det J|` for `J = df(x|context)/dx`
        such that `log p_x(x|context) = log p_z(f(x|context)) + log|det J|`.

        Args:
            x: Tensor, shape (batch_size, ...)
            context: Tensor, shape (batch_size, ...).

        Returns:
            z: Tensor, shape (batch_size, ...)
            ldj: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def inverse(self, z, context):
        """
        Inverse transform.
        Computes `x = f^{-1}(z|context)`.

        Args:
            z: Tensor, shape (batch_size, ...)
            context: Tensor, shape (batch_size, ...).

        Returns:
            x: Tensor, shape (batch_size, ...)
        """
        raise NotImplementedError()


class ConditionalSequentialTransform(ConditionalTransform):
    """
    Chains multiple ConditionalTransform objects sequentially.

    Args:
        transforms: Transform or iterable with each element being a Transform object
    """

    def __init__(self, transforms):
        super(ConditionalSequentialTransform, self).__init__()
        if isinstance(transforms, Transform):
            transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.has_inverse = all(transform.has_inverse for transform in transforms)
        self.transforms = nn.ModuleList(transforms)

    @property
    def bijective(self):
        return all(transform.bijective for transform in self.transforms)

    @property
    def stochastic_forward(self):
        return any(transform.stochastic_forward for transform in self.transforms)

    @property
    def stochastic_inverse(self):
        return any(transform.stochastic_inverse for transform in self.transforms)

    def _preprocess_context(self, context):
        return context

    def forward(self, x, context):
        context = self._preprocess_context(context)
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                x, ldj = transform(x, context)
            else:
                x, ldj = transform(x)
            log_prob += ldj
        return x, log_prob

    def inverse(self, z, context):
        context = self._preprocess_context(context)
        for transform in reversed(self.transforms):
            if isinstance(transform, ConditionalTransform):
                z = transform.inverse(z, context)
            else:
                z = transform.inverse(z)
        return z
