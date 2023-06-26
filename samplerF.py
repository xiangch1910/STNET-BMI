import torch.utils.data.sampler as sampler

class OverLoadSampler(sampler.Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(self.data_source))

    def __len__(self):
        return self.data_source



