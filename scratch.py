from pytorch_balanced_sampler.sampler import SamplerFactory
import itertools
import numpy as np
import torch
from torch.utils.data import DistributedSampler, DataLoader
from pytorch_balanced_sampler.dist_sampler_wrapper import DistributedSamplerWrapper
from collections import defaultdict
# which sample indices belong to each of 4 classes
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch_utils.misc import InfiniteSampler
from pytorch_balanced_sampler import ImbalancedDatasetSampler2
from torch.utils.data import WeightedRandomSampler

data = 200 * [0] + 50 * [1] + 30 * [2] + 10 * [3] + 5 * [4]

class_counts = [200,50,30,10,5]
class_counts = np.array(class_counts) ** 0.25
weight = 1.0 / class_counts
weight /= weight.max()
print(weight)
samples_weight = np.array([weight[c] for c in data])
samples_weight = torch.from_numpy(samples_weight).type('torch.DoubleTensor')


# set the iterator infinitely large number and wrap it with dist sampler
imb_sampler = WeightedRandomSampler(samples_weight, num_samples=32, replacement=True)

iter_0 = iter(DistributedSamplerWrapper(imb_sampler, num_replicas=2, rank=0))
iter_1 = iter(DistributedSamplerWrapper(imb_sampler, num_replicas=2, rank=1))


stat_a=defaultdict(int)
stat_b=defaultdict(int)

while True:
    a = next(iter_0)
    b = next(iter_1)
    print(a,b)