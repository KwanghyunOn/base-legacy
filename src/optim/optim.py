import torch


def get_optimizer(optim_name, optim_kwargs, params):
    if optim_name == "Adam":
        return torch.optim.Adam(params, **optim_kwargs)
    elif optim_name == "SGD":
        return torch.optim.SGD(params, **optim_kwargs)
