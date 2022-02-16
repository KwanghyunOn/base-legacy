from importlib import import_module
from torch.utils.data import DataLoader


def get_transform(transform_name, **transform_kwargs):
    if transform_name is None:
        return None
    module, attr = transform_name.rsplit(".", 1)
    transform_cls = getattr(import_module("." + module, "data.transforms"), attr)
    return transform_cls(**transform_kwargs)


def get_dataset(dataset_name, **dataset_kwargs):
    module, attr = dataset_name.rsplit(".", 1)
    dataset_cls = getattr(import_module("." + module, "data.datasets"), attr)
    return dataset_cls(**dataset_kwargs)


def get_dataloader(transform_name, transform_kwargs, dataset_name, dataset_kwargs, loader_kwargs):
    train_transform = get_transform(transform_name, **transform_kwargs, train=True)
    eval_transform = get_transform(transform_name, **transform_kwargs, train=False)
    train_dataset = get_dataset(dataset_name, **dataset_kwargs, transform=train_transform, train=True)
    eval_dataset = get_dataset(dataset_name, **dataset_kwargs, transform=eval_transform, train=False)
    
    if "eval_kwargs" in loader_kwargs:
        eval_kwargs = loader_kwargs.pop("eval_kwargs")
        eval_loader_kwargs = loader_kwargs.copy()
        eval_loader_kwargs.update(eval_kwargs)
    else:
        eval_loader_kwargs = loader_kwargs
    train_loader = DataLoader(train_dataset, **loader_kwargs, shuffle=True)
    eval_loader = DataLoader(eval_dataset, **eval_loader_kwargs, shuffle=False)
    return train_loader, eval_loader
