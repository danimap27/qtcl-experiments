"""
Split CIFAR-10 dataset for continual learning benchmarks.

Partitions CIFAR-10 into T sequential binary classification tasks.
Default: 5 tasks pairing classes in order (0,1), (2,3), (4,5), (6,7), (8,9).

Each task is split into train / val / test with configurable fractions:
    subset_fraction:   fraction of full data to use (e.g. 0.2 = 20%)
    train_test_split:  within subset, fraction for train+val (e.g. 0.8)
    train_val_split:   within train+val, fraction for train (e.g. 0.8)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional


CIFAR10_CLASS_PAIRS: List[Tuple[int, int]] = [
    (0, 1),  # airplane vs automobile
    (2, 3),  # bird vs cat
    (4, 5),  # deer vs dog
    (6, 7),  # frog vs horse
    (8, 9),  # ship vs truck
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class BinarySubset(Dataset):
    """Dataset wrapper that filters to two classes and relabels them 0/1."""

    def __init__(self, base: Dataset, indices: List[int],
                 class_a: int, class_b: int):
        self.base = base
        self.indices = indices
        self.class_a = class_a
        self.class_b = class_b

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base[self.indices[idx]]
        return x, (0 if y == self.class_a else 1)


class SplitCIFAR10:
    """
    Generates T sequential binary classification tasks from CIFAR-10
    with train / val / test splits.
    """

    def __init__(
        self,
        root: str = "./data/datasets",
        n_tasks: int = 5,
        image_size: int = 32,
        subset_fraction: float = 0.2,
        train_test_split: float = 0.8,
        train_val_split: float = 0.8,
        seed: int = 0,
    ):
        if not 2 <= n_tasks <= 5:
            raise ValueError(f"n_tasks must be in [2, 5], got {n_tasks}")
        self.n_tasks = n_tasks
        self.class_pairs = CIFAR10_CLASS_PAIRS[:n_tasks]
        self.subset_fraction  = subset_fraction
        self.train_test_split = train_test_split
        self.train_val_split  = train_val_split
        self.seed = seed

        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)

        train_tf = transforms.Compose([
            transforms.Resize(image_size) if image_size != 32 else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(image_size) if image_size != 32 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # Pool together the official train and test splits, then re-split.
        # Datasets must be pre-downloaded via data/download_datasets.py
        # to avoid concurrent download races across SLURM array tasks.
        self._pool_train_tf      = datasets.CIFAR10(root, train=True,  download=False, transform=train_tf)
        self._pool_eval_tf       = datasets.CIFAR10(root, train=True,  download=False, transform=eval_tf)
        self._pool_test_train_tf = datasets.CIFAR10(root, train=False, download=False, transform=train_tf)
        self._pool_test_eval_tf  = datasets.CIFAR10(root, train=False, download=False, transform=eval_tf)

    def _build_indices(self, task_id: int):
        ca, cb = self.class_pairs[task_id]

        # Use .targets directly (avoids loading and transforming every image)
        train_targets = np.asarray(self._pool_train_tf.targets)
        test_targets  = np.asarray(self._pool_test_train_tf.targets)
        train_pool = [(int(i), "train") for i in np.where(np.isin(train_targets, [ca, cb]))[0]]
        test_pool  = [(int(i), "test")  for i in np.where(np.isin(test_targets,  [ca, cb]))[0]]
        all_idx = train_pool + test_pool

        rng = np.random.default_rng(self.seed + task_id)
        rng.shuffle(all_idx)

        # Apply subset fraction
        n_total = int(len(all_idx) * self.subset_fraction)
        all_idx = all_idx[:n_total]

        # Train+val vs test
        n_train_val = int(n_total * self.train_test_split)
        train_val_idx = all_idx[:n_train_val]
        test_idx      = all_idx[n_train_val:]

        # Train vs val
        n_train = int(len(train_val_idx) * self.train_val_split)
        train_idx = train_val_idx[:n_train]
        val_idx   = train_val_idx[n_train:]

        return train_idx, val_idx, test_idx, ca, cb

    def _make_subset(self, idx_list, ca, cb, mode: str):
        """mode: 'train' uses train transforms, others use eval transforms."""
        # Build per-base index lists
        train_pool_indices = [i for (i, src) in idx_list if src == "train"]
        test_pool_indices  = [i for (i, src) in idx_list if src == "test"]

        if mode == "train":
            base_train_pool = self._pool_train_tf
            base_test_pool  = self._pool_test_train_tf
        else:
            base_train_pool = self._pool_eval_tf
            base_test_pool  = self._pool_test_eval_tf

        ds_a = BinarySubset(base_train_pool, train_pool_indices, ca, cb) if train_pool_indices else None
        ds_b = BinarySubset(base_test_pool,  test_pool_indices,  ca, cb) if test_pool_indices  else None
        if ds_a and ds_b:
            return torch.utils.data.ConcatDataset([ds_a, ds_b])
        return ds_a or ds_b

    def get_task(self, task_id: int) -> Tuple[Dataset, Dataset, Dataset]:
        """Return (train_dataset, val_dataset, test_dataset)."""
        if task_id >= self.n_tasks:
            raise IndexError(f"task_id {task_id} out of range for n_tasks={self.n_tasks}")
        train_idx, val_idx, test_idx, ca, cb = self._build_indices(task_id)
        return (
            self._make_subset(train_idx, ca, cb, mode="train"),
            self._make_subset(val_idx,   ca, cb, mode="eval"),
            self._make_subset(test_idx,  ca, cb, mode="eval"),
        )

    def task_description(self, task_id: int) -> str:
        ca, cb = self.class_pairs[task_id]
        return f"{CIFAR10_CLASSES[ca]} vs {CIFAR10_CLASSES[cb]}"


def get_task_loaders(
    dataset: SplitCIFAR10,
    task_id: int,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_ds, val_ds, test_ds = dataset.get_task(task_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
