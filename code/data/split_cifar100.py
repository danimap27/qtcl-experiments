"""
Split CIFAR-100 dataset for continual learning benchmarks.

Uses CIFAR-100 superclass labels (20 superclasses → 10 binary tasks).
Each task is split into train / val / test with configurable fractions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional


CIFAR100_SUPERCLASS_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers",
    "fruit and vegetables", "household electrical devices",
    "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes",
    "large omnivores and herbivores", "medium-sized mammals",
    "non-insect invertebrates", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2",
]

FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

SUPERCLASS_TASK_PAIRS: List[Tuple[int, int]] = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
    (10, 11), (12, 13), (14, 15), (16, 17), (18, 19),
]


class BinaryCoarseSubset(Dataset):
    """Filter CIFAR-100 to two superclasses and relabel as 0/1."""

    def __init__(self, base: Dataset, indices: List[int],
                 coarse_a: int, coarse_b: int, fine_to_coarse: List[int]):
        self.base = base
        self.indices = indices
        self.coarse_a = coarse_a
        self.coarse_b = coarse_b
        self.fine_to_coarse = fine_to_coarse

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, fine = self.base[self.indices[idx]]
        return x, (0 if self.fine_to_coarse[fine] == self.coarse_a else 1)


class SplitCIFAR100:
    """Generates T sequential binary classification tasks from CIFAR-100 superclasses."""

    def __init__(
        self,
        root: str = "./data/datasets",
        n_tasks: int = 10,
        image_size: int = 32,
        subset_fraction: float = 0.2,
        train_test_split: float = 0.8,
        train_val_split: float = 0.8,
        seed: int = 0,
    ):
        if not 2 <= n_tasks <= 10:
            raise ValueError(f"n_tasks must be in [2, 10], got {n_tasks}")
        self.n_tasks = n_tasks
        self.task_pairs = SUPERCLASS_TASK_PAIRS[:n_tasks]
        self.subset_fraction  = subset_fraction
        self.train_test_split = train_test_split
        self.train_val_split  = train_val_split
        self.seed = seed

        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)

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

        # Pre-downloaded via data/download_datasets.py
        self._pool_train_tf      = datasets.CIFAR100(root, train=True,  download=False, transform=train_tf)
        self._pool_eval_tf       = datasets.CIFAR100(root, train=True,  download=False, transform=eval_tf)
        self._pool_test_train_tf = datasets.CIFAR100(root, train=False, download=False, transform=train_tf)
        self._pool_test_eval_tf  = datasets.CIFAR100(root, train=False, download=False, transform=eval_tf)

    def _build_indices(self, task_id: int):
        ca, cb = self.task_pairs[task_id]

        train_pool = [(i, "train") for i, (_, y) in enumerate(self._pool_train_tf)
                      if FINE_TO_COARSE[y] in (ca, cb)]
        test_pool  = [(i, "test")  for i, (_, y) in enumerate(self._pool_test_train_tf)
                      if FINE_TO_COARSE[y] in (ca, cb)]
        all_idx = train_pool + test_pool

        rng = np.random.default_rng(self.seed + task_id)
        rng.shuffle(all_idx)

        n_total = int(len(all_idx) * self.subset_fraction)
        all_idx = all_idx[:n_total]

        n_train_val = int(n_total * self.train_test_split)
        train_val_idx = all_idx[:n_train_val]
        test_idx      = all_idx[n_train_val:]

        n_train = int(len(train_val_idx) * self.train_val_split)
        train_idx = train_val_idx[:n_train]
        val_idx   = train_val_idx[n_train:]

        return train_idx, val_idx, test_idx, ca, cb

    def _make_subset(self, idx_list, ca, cb, mode: str):
        train_pool_indices = [i for (i, src) in idx_list if src == "train"]
        test_pool_indices  = [i for (i, src) in idx_list if src == "test"]

        if mode == "train":
            base_train_pool = self._pool_train_tf
            base_test_pool  = self._pool_test_train_tf
        else:
            base_train_pool = self._pool_eval_tf
            base_test_pool  = self._pool_test_eval_tf

        ds_a = BinaryCoarseSubset(base_train_pool, train_pool_indices, ca, cb, FINE_TO_COARSE) if train_pool_indices else None
        ds_b = BinaryCoarseSubset(base_test_pool,  test_pool_indices,  ca, cb, FINE_TO_COARSE) if test_pool_indices  else None
        if ds_a and ds_b:
            return torch.utils.data.ConcatDataset([ds_a, ds_b])
        return ds_a or ds_b

    def get_task(self, task_id: int) -> Tuple[Dataset, Dataset, Dataset]:
        if task_id >= self.n_tasks:
            raise IndexError(f"task_id {task_id} out of range for n_tasks={self.n_tasks}")
        train_idx, val_idx, test_idx, ca, cb = self._build_indices(task_id)
        return (
            self._make_subset(train_idx, ca, cb, mode="train"),
            self._make_subset(val_idx,   ca, cb, mode="eval"),
            self._make_subset(test_idx,  ca, cb, mode="eval"),
        )

    def task_description(self, task_id: int) -> str:
        ca, cb = self.task_pairs[task_id]
        return f"{CIFAR100_SUPERCLASS_NAMES[ca]} vs {CIFAR100_SUPERCLASS_NAMES[cb]}"


def get_task_loaders(
    dataset,
    task_id: int,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = dataset.get_task(task_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
