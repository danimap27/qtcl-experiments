"""
Elastic Weight Consolidation (EWC) for continual learning.

Implements diagonal Fisher Information approximation and EWC loss penalty
as described in Kirkpatrick et al. (2017).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import copy


class EWC:
    """
    Elastic Weight Consolidation regulariser.

    After training on each task, call `update(model, loader, device)` to
    compute the diagonal Fisher Information and store the optimal parameters.
    During training on a new task, call `penalty(model)` to get the EWC loss term.
    """

    def __init__(self, lambda_ewc: float = 5000.0):
        self.lambda_ewc = lambda_ewc
        self._fisher: List[Dict[str, torch.Tensor]] = []
        self._optima: List[Dict[str, torch.Tensor]] = []

    def update(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        n_samples: int = 200,
    ) -> None:
        """
        Compute diagonal Fisher Information for the current task and store optimal params.

        Args:
            model:     Trained model (backbone + head) after convergence on task t.
            loader:    Training data loader for task t.
            device:    Compute device.
            n_samples: Number of samples used for Fisher estimation.
        """
        model.eval()
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        criterion = nn.CrossEntropyLoss()
        count = 0

        for x, y in loader:
            if count >= n_samples:
                break
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            model.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2) * batch_size

            count += batch_size

        count = max(count, 1)
        for name in fisher:
            fisher[name] /= count

        self._fisher.append(fisher)
        self._optima.append({
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        })

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty across all previous tasks.

        Returns:
            Scalar tensor: (lambda/2) * sum_k sum_i F_k[i] * (theta_i - theta*_k[i])^2
        """
        if not self._fisher:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        params = dict(model.named_parameters())

        for fisher, optima in zip(self._fisher, self._optima):
            for name, f in fisher.items():
                if name not in params:
                    continue
                param = params[name]
                opt   = optima[name].to(param.device)
                f_dev = f.to(param.device)
                loss  = loss + (f_dev * (param - opt).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * loss

    def n_tasks_seen(self) -> int:
        return len(self._fisher)
