from torchvision import transforms


class ImgClsTransform(transforms.Compose):
    def __init__(self, train):
        transform = [transforms.ToPILImage(),
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        super().__init__(transform)
