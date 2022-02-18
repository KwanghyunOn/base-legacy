import os
import pickle
import numpy as np
from torch.utils.data import Dataset


class CIFAR10(Dataset):

    """
    Root directory should look like
        root/
            batches.meta
            data_batch_1
            data_batch_2
            ...
            data_batch_5
            test_batch
    """

    def __init__(self, root, transform=None, train=True):
        super().__init__()
        self._init_data(root, train)
        self.keys = {"img", "label", "class_name"}
        self.transform = transform
        self.train = train

    def _init_data(self, root, train):
        def unpickle(file):
            with open(file, "rb") as fo:
                dict = pickle.load(fo, encoding="bytes")
            return dict

        self.imgs = []
        self.labels = []
        if train:
            for i in range(1, 6):
                data_dict = unpickle(os.path.join(root, f"data_batch_{i}"))
                self.imgs.extend(data_dict[b"data"])
                self.labels.extend(data_dict[b"labels"])
        else:
            data_dict = unpickle(os.path.join(root, f"test_batch"))
            self.imgs.extend(data_dict[b"data"])
            self.labels.extend(data_dict[b"labels"])

        self.imgs = (
            np.stack(self.imgs, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        )

        meta_dict = unpickle(os.path.join(root, f"batches.meta"))
        self.class_names = [
            class_name.decode("utf-8") for class_name in meta_dict[b"label_names"]
        ]

    def __getitem__(self, index):
        data = {
            "img": self.imgs[index],
            "label": self.labels[index],
            "class_name": self.class_names[self.labels[index]],
        }
        if self.transform is not None:
            data["img"] = self.transform(data["img"])
        return data

    def __len__(self):
        return len(self.imgs)
