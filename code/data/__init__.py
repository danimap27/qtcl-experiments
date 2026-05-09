from .split_cifar10  import SplitCIFAR10,  get_task_loaders as _loaders10
from .split_cifar100 import SplitCIFAR100, get_task_loaders as _loaders100
from torch.utils.data import DataLoader
from typing import Tuple


def get_task_loaders(dataset, task_id: int, batch_size: int = 32, num_workers: int = 2
                     ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    if isinstance(dataset, SplitCIFAR10):
        return _loaders10(dataset, task_id, batch_size, num_workers)
    elif isinstance(dataset, SplitCIFAR100):
        return _loaders100(dataset, task_id, batch_size, num_workers)
    else:
        raise TypeError(f"Unknown dataset type: {type(dataset)}")


__all__ = ["SplitCIFAR10", "SplitCIFAR100", "get_task_loaders"]
