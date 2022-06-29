from torchvision import transforms as T
from . import paired_transforms as PT


class SRTransform(PT.Compose):
    def __init__(self, train_size, eval_size, train=True):
        if train:
            transforms = [
                PT.RandomCrop(size=train_size),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
                PT.RandomHorizontalFlip(),
            ]
            super().__init__(transforms)

        else:
            transforms = [
                PT.CentorCrop(size=eval_size),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ]
            super().__init__(transforms)