from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.5, 0.5, 0.5)
CIFAR10_STD  = (0.5, 0.5, 0.5)


def build_cifar10_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for CIFAR-10.

    Images are normalised to [-1, 1] (mean=0.5, std=0.5 per channel).
    Train split uses random-crop + horizontal-flip augmentation.
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    root = str(Path(data_root))
    trainset       = datasets.CIFAR10(root, train=True,  download=True, transform=train_tf)
    trainset_plain = datasets.CIFAR10(root, train=True,  download=True, transform=test_tf)
    testset        = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    n     = len(trainset)
    n_val = int(n * val_fraction)
    idx   = np.random.default_rng(seed).permutation(n)

    pin = torch.cuda.is_available()
    kw  = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)

    train_loader = DataLoader(Subset(trainset,       idx[n_val:]), shuffle=True,  **kw)
    val_loader   = DataLoader(Subset(trainset_plain, idx[:n_val]), shuffle=False, **kw)
    test_loader  = DataLoader(testset,                             shuffle=False, **kw)

    return train_loader, val_loader, test_loader
