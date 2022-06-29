import os
import torch
import torch.distributed as dist


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = dist.get_world_size()
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
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ["LOCAL_SIZE"])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def is_master_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_master_process():
        torch.save(*args, **kwargs)


def gather_and_concat(x, dim=0):
    if not is_dist_avail_and_initialized():
        return x
    x_all = [torch.zeros_like(x) for _ in range(get_world_size())]
    dist.all_gather(x_all, x)
    return torch.cat(x_all, dim=dim)


def wait():
    if dist.is_initialized():
        dist.barrier()