
import math
import bisect

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, ConcatDataset, BatchSampler


def default_worker_init_fn(worker_id):
    np.random.seed(worker_id)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=12, num_workers=4, distributed=False):

        if distributed:
            sampler = DistributedSampler(dataset, shuffle=True, num_replicas=2)
            batch_sampler = BatchSampler(sampler, batch_size//2, False)
            torch.utils.data.DataLoader.__init__(
                self, dataset=dataset, batch_sampler=batch_sampler,
                num_workers=num_workers, pin_memory=False,
                drop_last=True, collate_fn=None,
                worker_init_fn=default_worker_init_fn)

        else:
            torch.utils.data.DataLoader.__init__(
                self, dataset,
                batch_size=batch_size, num_workers=num_workers,
                drop_last=True, shuffle=True,
                pin_memory=True, collate_fn=None,
                worker_init_fn=default_worker_init_fn)


class SuccessiveRandomSampler(Sampler):
    '''Random Sampler that yields sorted data in successive ranges.
    Args:
        dataset: Dataset used for sampling.
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch




