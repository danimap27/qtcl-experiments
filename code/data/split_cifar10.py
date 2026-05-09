"""
Split CIFAR-10 dataset for continual learning benchmarks.

Partitions CIFAR-10 into T sequential binary classification tasks.
Default: 5 tasks pairing classes in order (0,1), (2,3), (4,5), (6,7), (8,9).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
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

    def __init__(self, dataset: Dataset, class_a: int, class_b: int,
                 max_samples: Optional[int] = None):
        self.dataset = dataset
        self.class_a = class_a
        self.class_b = class_b

        indices = [
            i for i, (_, label) in enumerate(dataset)
            if label in (class_a, class_b)
        ]
        if max_samples is not None:
            rng = np.random.default_rng(0)
            rng.shuffle(indices)
            indices = indices[:max_samples]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.dataset[self.indices[idx]]
        label = 0 if y == self.class_a else 1
        return x, label


class SplitCIFAR10:
    """
    Generates T sequential binary classification tasks from CIFAR-10.

    Args:
        root:               Directory to download / cache CIFAR-10.
        n_tasks:            Number of tasks (2–5; uses first n_tasks class pairs).
        image_size:         Spatial resolution to resize images (default 32).
        samples_per_task_train: Optional cap on training samples per task.
        samples_per_task_val:   Optional cap on validation samples per task.
    """

    def __init__(
        self,
        root: str = "./data/datasets",
        n_tasks: int = 5,
        image_size: int = 32,
        samples_per_task_train: Optional[int] = None,
        samples_per_task_val: Optional[int] = None,
    ):
        if not 2 <= n_tasks <= 5:
            raise ValueError(f"n_tasks must be in [2, 5], got {n_tasks}")
        self.n_tasks = n_tasks
        self.class_pairs = CIFAR10_CLASS_PAIRS[:n_tasks]
        self.max_train = samples_per_task_train
        self.max_val = samples_per_task_val

        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)

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

        self._train_base = datasets.CIFAR10(root, train=True,  download=True, transform=train_tf)
        self._val_base   = datasets.CIFAR10(root, train=False, download=True, transform=val_tf)

    def get_task(self, task_id: int) -> Tuple[Dataset, Dataset]:
        """Return (train_dataset, val_dataset) for a single task."""
        if task_id >= self.n_tasks:
            raise IndexError(f"task_id {task_id} out of range for n_tasks={self.n_tasks}")
        ca, cb = self.class_pairs[task_id]
        train_ds = BinarySubset(self._train_base, ca, cb, self.max_train)
        val_ds   = BinarySubset(self._val_base,   ca, cb, self.max_val)
        return train_ds, val_ds

    def task_description(self, task_id: int) -> str:
        ca, cb = self.class_pairs[task_id]
        return f"{CIFAR10_CLASSES[ca]} vs {CIFAR10_CLASSES[cb]}"


def get_task_loaders(
    dataset: SplitCIFAR10,
    task_id: int,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for the given task."""
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
