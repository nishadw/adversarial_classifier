from __future__ import annotations

import tarfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from torchvision import datasets, transforms


def _extract_cifar10_archive_if_needed(
    data_root: Path,
    archive_path: Path | None,
    download_if_missing: bool,
) -> None:
    cifar_dir = data_root / "cifar-10-batches-py"
    if cifar_dir.exists() or archive_path is None:
        return

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with tarfile.open(archive_path, "r:*") as tar:
        names = tar.getnames()
        if not any(name.startswith("cifar-10-batches-py/") for name in names):
            if download_if_missing:
                return
            raise ValueError(
                "Archive does not look like CIFAR-10 python format. "
                "Expected folder 'cifar-10-batches-py/'."
            )
        tar.extractall(path=data_root)


def build_cifar10_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    archive_path: str | None = None,
    download_if_missing: bool = False,
    val_fraction: float = 0.1,
) -> tuple[list, list, list]:
    """Build CIFAR-10 data loaders as lists of (images, labels) JAX arrays.

    Returns (train_batches, val_batches, test_batches). The validation split is taken
    from the end of the original training set with fraction `val_fraction`.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    archive = Path(archive_path) if archive_path else None
    _extract_cifar10_archive_if_needed(root, archive, download_if_missing=download_if_missing)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )
    test_transform = transforms.Compose([])

    train_dataset = datasets.CIFAR10(
        root=str(root),
        train=True,
        transform=train_transform,
        download=download_if_missing,
    )
    test_dataset = datasets.CIFAR10(
        root=str(root),
        train=False,
        transform=test_transform,
        download=download_if_missing,
    )

    # Convert full train dataset to arrays first, then split into train/val and batch.
    train_imgs = []
    train_lbls = []
    for i in range(len(train_dataset)):
        img, lbl = train_dataset[i]
        train_imgs.append(np.array(img, dtype=np.float32) / 255.0)
        train_lbls.append(int(lbl))

    train_imgs = np.stack(train_imgs, axis=0)  # (N, H, W, C)
    train_lbls = np.array(train_lbls, dtype=np.int32)

    # split
    n_total = train_imgs.shape[0]
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    train_imgs_arr = train_imgs[:n_train]
    train_lbls_arr = train_lbls[:n_train]
    val_imgs_arr = train_imgs[n_train:]
    val_lbls_arr = train_lbls[n_train:]

    def make_batches_from_arrays(imgs_arr, lbls_arr, batch_size):
        batches = []
        for i in range(0, imgs_arr.shape[0], batch_size):
            bs = imgs_arr[i : i + batch_size]
            bl = lbls_arr[i : i + batch_size]
            batches.append((jnp.array(bs, dtype=jnp.float32), jnp.array(bl, dtype=jnp.int32)))
        return batches

    train_batches = make_batches_from_arrays(train_imgs_arr, train_lbls_arr, batch_size)
    val_batches = make_batches_from_arrays(val_imgs_arr, val_lbls_arr, batch_size)

    # test dataset -> batches
    test_imgs = []
    test_lbls = []
    for i in range(len(test_dataset)):
        img, lbl = test_dataset[i]
        test_imgs.append(np.array(img, dtype=np.float32) / 255.0)
        test_lbls.append(int(lbl))

    test_imgs = np.stack(test_imgs, axis=0)
    test_lbls = np.array(test_lbls, dtype=np.int32)
    test_batches = make_batches_from_arrays(test_imgs, test_lbls, batch_size)

    return train_batches, val_batches, test_batches
