"""
Continual learning evaluation metrics + sklearn-based per-task metrics.

CL metrics:
    - Average Accuracy (AA), Average Forgetting (AF), Backward Transfer (BWT)
    - Forward Transfer (FWT)
    - Learning Accuracy per task (diagonal of A)

Per-task supervised metrics:
    - Precision, Recall, F1
    - ROC AUC, PR AUC
    - Confusion matrix
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


# ──────────────────────────────────────────────────────────────────────────────
# Continual Learning metrics (work on the accuracy matrix)
# ──────────────────────────────────────────────────────────────────────────────

class CLMetrics:
    """Tracks accuracy matrix A[t][j] = accuracy on task j after training on task t."""

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        self.A: List[List[float]] = []

    def record(self, row: List[float]) -> None:
        assert len(row) == self.n_tasks, f"Expected {self.n_tasks} values, got {len(row)}"
        self.A.append(list(row))

    def average_accuracy(self) -> float:
        if not self.A:
            return 0.0
        return float(np.mean(self.A[-1]))

    def average_forgetting(self) -> float:
        if len(self.A) < 2:
            return 0.0
        T = self.n_tasks
        forgetting = []
        for j in range(T - 1):
            max_acc = max(self.A[t][j] for t in range(len(self.A)) if j < len(self.A[t]))
            final_acc = self.A[-1][j]
            forgetting.append(max_acc - final_acc)
        return float(np.mean(forgetting))

    def backward_transfer(self) -> float:
        if len(self.A) < 2:
            return 0.0
        T = self.n_tasks
        bwt = []
        for j in range(T - 1):
            if j < len(self.A):
                a_T_j = self.A[-1][j]
                a_j_j = self.A[j][j]
                bwt.append(a_T_j - a_j_j)
        return float(np.mean(bwt)) if bwt else 0.0

    def forward_transfer(self) -> float:
        """
        FWT = mean over tasks 2..T of (A[t-1][t] - random_baseline).
        Random baseline for binary tasks is 0.5.
        """
        if len(self.A) < 2:
            return 0.0
        random_baseline = 0.5
        fwt = []
        for t in range(1, len(self.A)):
            # Accuracy on task t before having trained it (i.e. after task t-1)
            if t < self.n_tasks:
                fwt.append(self.A[t - 1][t] - random_baseline)
        return float(np.mean(fwt)) if fwt else 0.0

    def learning_accuracy(self) -> List[float]:
        """Diagonal of A: peak accuracy reached on each task."""
        return [self.A[i][i] for i in range(len(self.A)) if i < len(self.A[i])]

    def matrix(self) -> np.ndarray:
        """Return the accuracy matrix as a numpy array (with zeros above diagonal)."""
        T = self.n_tasks
        M = np.zeros((T, T))
        for t in range(len(self.A)):
            for j in range(min(len(self.A[t]), T)):
                M[t, j] = self.A[t][j]
        return M

    def summary(self) -> Dict[str, float]:
        return {
            "AA":  self.average_accuracy(),
            "AF":  self.average_forgetting(),
            "BWT": self.backward_transfer(),
            "FWT": self.forward_transfer(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Per-task supervised metrics
# ──────────────────────────────────────────────────────────────────────────────

def supervised_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1, ROC AUC, PR AUC, accuracy.

    Args:
        y_true:  ground-truth labels (1-D, binary 0/1)
        y_pred:  predicted labels (1-D)
        y_score: predicted scores/probabilities for the positive class (optional)
    """
    out = {
        "accuracy":  accuracy_score(y_true, y_pred),
    }
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out["precision"] = float(p)
    out["recall"]    = float(r)
    out["f1"]        = float(f)

    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            out["pr_auc"]  = float(average_precision_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
            out["pr_auc"]  = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"]  = float("nan")

    return out


def confusion_data(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def roc_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(y_true, y_score)


def pr_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return precision_recall_curve(y_true, y_score)
