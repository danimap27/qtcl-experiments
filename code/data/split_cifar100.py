"""
Split CIFAR-100 dataset for continual learning benchmarks.

Uses CIFAR-100 superclass labels (20 superclasses → 10 binary tasks).
Each task is a binary classification between two superclasses.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional


# CIFAR-100 superclass names (20 superclasses, paired into 10 binary tasks)
CIFAR100_SUPERCLASS_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers",
    "fruit and vegetables", "household electrical devices",
    "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes",
    "large omnivores and herbivores", "medium-sized mammals",
    "non-insect invertebrates", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2",
]

# CIFAR-100 superclass index for each fine class
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

# 10 binary tasks: pairs of superclasses
SUPERCLASS_TASK_PAIRS: List[Tuple[int, int]] = [
    (0, 1),   # aquatic mammals vs fish
    (2, 3),   # flowers vs food containers
    (4, 5),   # fruit and vegetables vs household electrical devices
    (6, 7),   # household furniture vs insects
    (8, 9),   # large carnivores vs large man-made outdoor things
    (10, 11), # large natural outdoor scenes vs large omnivores and herbivores
    (12, 13), # medium-sized mammals vs non-insect invertebrates
    (14, 15), # people vs reptiles
    (16, 17), # small mammals vs trees
    (18, 19), # vehicles 1 vs vehicles 2
]


class BinaryCoarseSubset(Dataset):
    """Filter CIFAR-100 to two superclasses and relabel as 0/1."""

    def __init__(
        self,
        dataset: Dataset,
        coarse_a: int,
        coarse_b: int,
        fine_to_coarse: List[int],
        max_samples: Optional[int] = None,
    ):
        self.dataset = dataset
        self.coarse_a = coarse_a
        self.coarse_b = coarse_b
        self.fine_to_coarse = fine_to_coarse

        indices = [
            i for i, (_, label) in enumerate(dataset)
            if fine_to_coarse[label] in (coarse_a, coarse_b)
        ]
        if max_samples is not None:
            rng = np.random.default_rng(0)
            rng.shuffle(indices)
            indices = indices[:max_samples]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, fine_label = self.dataset[self.indices[idx]]
        coarse_label = self.fine_to_coarse[fine_label]
        binary_label = 0 if coarse_label == self.coarse_a else 1
        return x, binary_label


class SplitCIFAR100:
    """
    Generates T sequential binary classification tasks from CIFAR-100 superclasses.

    Args:
        root:               Directory to download / cache CIFAR-100.
        n_tasks:            Number of tasks (2–10).
        image_size:         Spatial resolution (default 32).
        samples_per_task_train: Optional training sample cap per task.
        samples_per_task_val:   Optional validation sample cap per task.
    """

    def __init__(
        self,
        root: str = "./data/datasets",
        n_tasks: int = 10,
        image_size: int = 32,
        samples_per_task_train: Optional[int] = None,
        samples_per_task_val: Optional[int] = None,
    ):
        if not 2 <= n_tasks <= 10:
            raise ValueError(f"n_tasks must be in [2, 10], got {n_tasks}")
        self.n_tasks = n_tasks
        self.task_pairs = SUPERCLASS_TASK_PAIRS[:n_tasks]
        self.max_train = samples_per_task_train
        self.max_val = samples_per_task_val

        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)

        train_tf = transforms.Compose([
            transforms.Resize(image_size) if image_size != 32 else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(image_size) if image_size != 32 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self._train_base = datasets.CIFAR100(root, train=True,  download=True, transform=train_tf)
        self._val_base   = datasets.CIFAR100(root, train=False, download=True, transform=val_tf)

    def get_task(self, task_id: int) -> Tuple[Dataset, Dataset]:
        if task_id >= self.n_tasks:
            raise IndexError(f"task_id {task_id} out of range for n_tasks={self.n_tasks}")
        ca, cb = self.task_pairs[task_id]
        train_ds = BinaryCoarseSubset(self._train_base, ca, cb, FINE_TO_COARSE, self.max_train)
        val_ds   = BinaryCoarseSubset(self._val_base,   ca, cb, FINE_TO_COARSE, self.max_val)
        return train_ds, val_ds

    def task_description(self, task_id: int) -> str:
        ca, cb = self.task_pairs[task_id]
        return f"{CIFAR100_SUPERCLASS_NAMES[ca]} vs {CIFAR100_SUPERCLASS_NAMES[cb]}"


def get_task_loaders(
    dataset,
    task_id: int,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = dataset.get_task(task_id)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
