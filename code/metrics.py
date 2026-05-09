"""
Continual learning evaluation metrics.

Implements Average Accuracy (AA), Average Forgetting (AF),
and Backward Transfer (BWT) as defined in Chaudhry et al. (2018).
"""

import numpy as np
from typing import List


class CLMetrics:
    """
    Tracks accuracy matrix A[t][j] = accuracy on task j after training on task t.
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        # A[t][j]: accuracy on task j right after completing training on task t
        self.A: List[List[float]] = []

    def record(self, row: List[float]) -> None:
        """Record a full evaluation row after completing training on task t."""
        assert len(row) == self.n_tasks, f"Expected {self.n_tasks} values, got {len(row)}"
        self.A.append(list(row))

    def average_accuracy(self) -> float:
        """AA = mean accuracy over all tasks after training on all T tasks."""
        if not self.A:
            return 0.0
        final_row = self.A[-1]
        return float(np.mean(final_row))

    def average_forgetting(self) -> float:
        """
        AF = mean over tasks 1..T-1 of (max accuracy ever achieved on task j) - (final accuracy).
        """
        if len(self.A) < 2:
            return 0.0
        T = self.n_tasks
        forgetting = []
        for j in range(T - 1):
            # max accuracy ever achieved on task j (up to when we last trained it)
            max_acc = max(self.A[t][j] for t in range(len(self.A)) if j < len(self.A[t]))
            final_acc = self.A[-1][j]
            forgetting.append(max_acc - final_acc)
        return float(np.mean(forgetting))

    def backward_transfer(self) -> float:
        """
        BWT = mean over tasks 1..T-1 of (final accuracy on task j) - (accuracy right after training task j).
        Negative BWT = forgetting; positive BWT = beneficial backward transfer.
        """
        if len(self.A) < 2:
            return 0.0
        T = self.n_tasks
        bwt = []
        for j in range(T - 1):
            if j < len(self.A):
                a_T_j  = self.A[-1][j]     # final accuracy on task j
                a_j_j  = self.A[j][j]      # accuracy on task j right after training it
                bwt.append(a_T_j - a_j_j)
        return float(np.mean(bwt)) if bwt else 0.0

    def summary(self) -> dict:
        return {
            "AA": self.average_accuracy(),
            "AF": self.average_forgetting(),
            "BWT": self.backward_transfer(),
        }
