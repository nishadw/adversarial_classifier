from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_cifar10_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
    archive_path: str | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for CIFAR-10.

    Data resolution order:
      1. Already-extracted cifar-10-batches-py/ directory  (fastest)
      2. Local tar.gz archive at archive_path (or data_root/cifar-10-python.tar.gz)
      3. Download from the web if download=True
    """
    root      = Path(data_root)
    cifar_dir = root / "cifar-10-batches-py"
    root.mkdir(parents=True, exist_ok=True)

    if not cifar_dir.exists():
        ap = Path(archive_path) if archive_path else root / "cifar-10-python.tar.gz"
        if ap.exists():
            print(f"Extracting CIFAR-10 from {ap} ...")
            with tarfile.open(ap, "r:*") as tar:
                tar.extractall(path=root)
        elif not download:
            raise FileNotFoundError(
                f"CIFAR-10 not found at {root}. "
                "Provide archive_path or set download=True."
            )

    need_download = download and not cifar_dir.exists()

    aug   = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()])
    plain = transforms.ToTensor()

    aug_ds   = datasets.CIFAR10(str(root), train=True,  transform=aug,   download=need_download)
    plain_ds = datasets.CIFAR10(str(root), train=True,  transform=plain, download=False)
    test_ds  = datasets.CIFAR10(str(root), train=False, transform=plain, download=False)

    idx   = np.random.default_rng(seed).permutation(len(aug_ds))
    n_val = int(len(aug_ds) * val_fraction)

    pin = torch.cuda.is_available()
    kw  = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)

    train_loader = DataLoader(Subset(aug_ds,   idx[n_val:]),  shuffle=True,  **kw)
    val_loader   = DataLoader(Subset(plain_ds, idx[:n_val]),  shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,                        shuffle=False, **kw)

    return train_loader, val_loader, test_loader
