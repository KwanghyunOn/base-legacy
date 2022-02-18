import os
import torch
from torch import distributed


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = distributed.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k, v in sorted(input_dict.items()):
            names.append(k)
            values.append(v)
        values = torch.stack(values, dim=0)
        distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def is_dist_avail_and_initialized():
    if not distributed.is_available():
        return False
    if not distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return distributed.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ["LOCAL_SIZE"])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
