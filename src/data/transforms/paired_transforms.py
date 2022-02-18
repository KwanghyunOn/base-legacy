import random
import abc

import torch
import torchvision.transforms.functional as F


class PairedTransform(metaclass=abc.ABCMeta):
    """
    Abstract class for paired transforms.
    """


class Compose(PairedTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_dict):
        for t in self.transforms:
            if isinstance(t, PairedTransform):
                img_dict = t(img_dict)
            else:
                for k in img_dict:
                    img_dict[k] = t(img_dict[k])
        return img_dict

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RandomHorizontalFlip(torch.nn.Module, PairedTransform):
    """Horizontally flip the given images randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img_dict):
        """
        Args:
            img_dict (Dictionary of PIL Image or Tensor): Images to be flipped.

        Returns:
            Dictionary of PIL Image or Tensor: Randomly flipped images.
        """
        if torch.rand(1) < self.p:
            img_dict = {k: F.hflip(img) for k, img in img_dict.items()}
        return img_dict

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomCrop(PairedTransform):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_dict):
        """
        Args:
            img_dict (Dictionary): Dictionary of images(numpy)
        """
        h = min(img.shape[0] for img in img_dict.values())
        w = min(img.shape[1] for img in img_dict.values())
        iy = random.randrange(0, h - self.size + 1)
        ix = random.randrange(0, w - self.size + 1)

        for k, img in img_dict.items():
            scale = img.shape[0] // h
            tx = scale * ix
            ty = scale * iy
            tp = scale * self.size
            img_dict[k] = img[ty : ty + tp, tx : tx + tp]

        return img_dict

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)


class CentorCrop(PairedTransform):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_dict):
        """
        Args:
            img_dict (Dictionary): Dictionary of images(numpy)
        """
        h = min(img.shape[0] for img in img_dict.values())
        w = min(img.shape[1] for img in img_dict.values())
        iy = (h - self.size + 1) // 2
        ix = (w - self.size + 1) // 2

        for k, img in img_dict.items():
            scale = img.shape[0] // h
            tx = scale * ix
            ty = scale * iy
            tp = scale * self.size
            img_dict[k] = img[ty : ty + tp, tx : tx + tp]

        return img_dict

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)


class ModCrop(PairedTransform):
    def __init__(self, modulo):
        self.m = modulo

    def __call__(self, img_dict):
        """
        Args:
            img_dict (Dictionary): Dictionary of images(numpy)
        """
        h = min(img.shape[0] for img in img_dict.values())
        w = min(img.shape[1] for img in img_dict.values())
        size_h = (h // self.m) * self.m
        size_w = (w // self.m) * self.m
        iy = (h - size_h + 1) // 2
        ix = (w - size_w + 1) // 2

        for k, img in img_dict.items():
            scale = img.shape[0] // h
            tx = scale * ix
            ty = scale * iy
            tpx = scale * size_w
            tpy = scale * size_h
            img_dict[k] = img[ty : ty + tpy, tx : tx + tpx]

        return img_dict

    def __repr__(self):
        return self.__class__.__name__ + "(modulo={})".format(self.m)
