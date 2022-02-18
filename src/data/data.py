from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


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


def get_dataloader(
    transform_name, transform_kwargs, dataset_name, dataset_kwargs, loader_kwargs
):
    if "eval_kwargs" in loader_kwargs:
        eval_kwargs = loader_kwargs.pop("eval_kwargs")
        loader_eval_kwargs = loader_kwargs.copy()
        loader_eval_kwargs.update(eval_kwargs)
    else:
        loader_eval_kwargs = loader_kwargs

    if transform_name:
        transform_train = get_transform(transform_name, **transform_kwargs, train=True)
        transform_eval = get_transform(transform_name, **transform_kwargs, train=False)
    else:
        transform_train = None
        transform_eval = None

    dataset_train = get_dataset(
        dataset_name, **dataset_kwargs, transform=transform_train, train=True
    )
    dataset_eval = get_dataset(
        dataset_name, **dataset_kwargs, transform=transform_eval, train=False
    )
    loader_train = DataLoader(dataset_train, **loader_kwargs, shuffle=True)
    loader_eval = DataLoader(dataset_eval, **loader_eval_kwargs, shuffle=False)
    return loader_train, loader_eval


def get_dataloader_ddp(
    transform_name, transform_kwargs, dataset_name, dataset_kwargs, loader_kwargs
):
    if "eval_kwargs" in loader_kwargs:
        eval_kwargs = loader_kwargs.pop("eval_kwargs")
        loader_eval_kwargs = loader_kwargs.copy()
        loader_eval_kwargs.update(eval_kwargs)
    else:
        loader_eval_kwargs = loader_kwargs

    if transform_name:
        transform_train = get_transform(transform_name, **transform_kwargs, train=True)
        transform_eval = get_transform(transform_name, **transform_kwargs, train=False)
    else:
        transform_train = None
        transform_eval = None

    dataset_train = get_dataset(
        dataset_name, **dataset_kwargs, transform=transform_train, train=True
    )
    dataset_eval = get_dataset(
        dataset_name, **dataset_kwargs, transform=transform_eval, train=False
    )
    sampler_train = DistributedSampler(dataset_train, shuffle=True)
    sampler_eval = DistributedSampler(dataset_eval, shuffle=False)
    loader_train = DataLoader(
        dataset_train, **loader_kwargs, sampler=sampler_train, pin_memory=True
    )
    loader_eval = DataLoader(
        dataset_eval, **loader_eval_kwargs, sampler=sampler_eval, pin_memory=True
    )
    return loader_train, loader_eval
