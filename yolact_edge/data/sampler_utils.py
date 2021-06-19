import torch
from torch.utils.data.sampler import Sampler, BatchSampler
import torch.distributed as dist
import itertools

"""
Modified based on https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/samplers/distributed_sampler.py
"""


class InfiniteSampler(Sampler):
    def __init__(self, dataset, seed, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.size = len(dataset)
        self.seed = seed
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        start = self.rank
        yield from itertools.islice(self.infinite_indices(), start, None, self.num_replicas)

    def infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g, device='cpu')
            else:
                yield from torch.arange(self.size, device='cpu')


def build_batch_data_sampler(sampler, images_per_batch):
    batch_sampler = BatchSampler(sampler, images_per_batch, drop_last=True)
    return batch_sampler
