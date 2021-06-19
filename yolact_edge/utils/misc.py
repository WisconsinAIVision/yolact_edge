import torch
import torch.distributed as dist


def is_distributed_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_distributed_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_distributed_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def barrier():
    if is_distributed_initialized():
        dist.barrier()