from torch import nn
from torchvision.models import resnet18, resnet34, resnet50
from typing import Any


class ResNet(nn.Module):
    def __init__(
        self, resnet_type: str = "resnet18", pretrained: bool = False, **kwargs: Any
    ) -> None:
        super().__init__()
        if resnet_type == "resnet18":
            self.model = resnet18(pretrained=pretrained, **kwargs)
        elif resnet_type == "resnet34":
            self.model = resnet34(pretrained=pretrained, **kwargs)
        elif resnet_type == "resnet50":
            self.model = resnet50(pretrained=pretrained, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
