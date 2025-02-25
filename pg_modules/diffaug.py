# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def DiffAugment(x, policy='', p=None, c=None, channels_first=True):
    probs = torch.matmul(c, p)
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for pol in policy.split(','):
            for f in AUGMENT_FNS[pol]:
                mask = torch.rand(probs.size(0), dtype=probs.dtype, device=probs.device) <= probs
                x = f(x, mask)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x, mask=None):
    out = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    if mask is not None:
        x[mask] = out[mask]
        return x
    return out


def rand_saturation(x, mask=None):
    x_mean = x.mean(dim=1, keepdim=True)
    out = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    if mask is not None:
        x[mask] = out[mask]
        return x
    return out


def rand_contrast(x, mask=None):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    out = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    if mask is not None:
        x[mask] = out[mask]
        return x
    return out


def rand_translation(x, mask=None, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    out = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    if mask is not None:
        x[mask] = out[mask]
        return x
    return out


def rand_cutout(x, mask=None, ratio=0.2):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask_ = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask_[grid_batch, grid_x, grid_y] = 0
    out = x * mask_.unsqueeze(1)
    if mask is not None:
        x[mask] = out[mask]
        return x
    return out


def mix(x1, x2, lam):
    return lam * x1 + (1.0 - lam) * x2


def mixup(x, c, alpha=0.0):
    if alpha == 0:
        return x, c

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    lam = np.random.beta(alpha, alpha)
    mix_x, mix_c = mix(x, x[index, ...], lam), mix(c, c[index, ...], lam)
    # lam = torch.empty(batch_size, 1, 1, 1).uniform_(0, alpha).to(x.device)
    # mix_x, mix_c = mix(x, x[index, ...], lam), mix(c, c[index, ...], lam[:, :, 0, 0])
    return mix_x, mix_c


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
