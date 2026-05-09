"""
Visualisation pipeline for QTCL experiments.

Generates per-run plots:
    - Loss / accuracy curves per epoch (per task and combined)
    - Accuracy matrix heatmap
    - Forgetting curve per task
    - Confusion matrix per task (after final training)
    - ROC and Precision-Recall curves per task
    - Probability histograms per task
    - Energy consumption summary

All plots are saved to the run's plots/ subdirectory.
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from metrics import (
    confusion_data, roc_curve_data, pr_curve_data,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig, path: str, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Training dynamics ─────────────────────────────────────────────────────────

def plot_learning_curves(training_log: List[Dict], plots_dir: str, run_id: str, dpi: int = 150) -> None:
    """One subplot per task: loss and accuracy curves over epochs."""
    if not training_log:
        return
    _ensure_dir(plots_dir)

    by_task: Dict[int, List[Dict]] = {}
    for entry in training_log:
        by_task.setdefault(entry["task_id"], []).append(entry)

    n_tasks = len(by_task)
    fig, axes = plt.subplots(2, n_tasks, figsize=(4 * n_tasks, 6), sharex=True)
    if n_tasks == 1:
        axes = axes.reshape(2, 1)

    for col, (tid, entries) in enumerate(sorted(by_task.items())):
        epochs = [e["epoch"] + 1 for e in entries]
        loss   = [e["loss"]      for e in entries]
        acc    = [e["acc"]       for e in entries]

        axes[0, col].plot(epochs, loss, "o-", color="tab:red")
        axes[0, col].set_title(f"Task {tid + 1} — loss")
        axes[0, col].set_ylabel("loss")
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(epochs, acc, "o-", color="tab:blue")
        axes[1, col].set_title(f"Task {tid + 1} — accuracy")
        axes[1, col].set_ylabel("accuracy")
        axes[1, col].set_xlabel("epoch")
        axes[1, col].set_ylim(0, 1.05)
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle(f"Learning curves — {run_id}", y=1.02)
    _save(fig, os.path.join(plots_dir, "learning_curves.png"), dpi)


def plot_loss_combined(training_log: List[Dict], plots_dir: str, run_id: str, dpi: int = 150) -> None:
    """Single chart with all tasks' loss curves concatenated."""
    if not training_log:
        return
    _ensure_dir(plots_dir)

    fig, (ax_l, ax_a) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Build a continuous x-axis (cumulative epoch index)
    losses = [e["loss"] for e in training_log]
    accs   = [e["acc"]  for e in training_log]
    x = np.arange(len(training_log))

    ax_l.plot(x, losses, color="tab:red")
    ax_a.plot(x, accs,   color="tab:blue")

    # Vertical lines marking task boundaries
    boundaries = []
    last_tid = -1
    for i, e in enumerate(training_log):
        if e["task_id"] != last_tid:
            boundaries.append((i, e["task_id"]))
            last_tid = e["task_id"]
    for b, tid in boundaries:
        ax_l.axvline(b, color="grey", linestyle="--", alpha=0.5)
        ax_a.axvline(b, color="grey", linestyle="--", alpha=0.5)
        ax_l.text(b, max(losses) * 0.95, f"T{tid+1}", fontsize=8)

    ax_l.set_ylabel("loss")
    ax_l.set_title(f"Training dynamics — {run_id}")
    ax_l.grid(True, alpha=0.3)

    ax_a.set_ylabel("accuracy")
    ax_a.set_xlabel("global epoch")
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(True, alpha=0.3)

    _save(fig, os.path.join(plots_dir, "loss_combined.png"), dpi)


# ── Accuracy matrix and forgetting ────────────────────────────────────────────

def plot_accuracy_matrix(matrix: np.ndarray, plots_dir: str, run_id: str, dpi: int = 150) -> None:
    """Heatmap of A[t][j]."""
    _ensure_dir(plots_dir)
    T = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    if HAS_SEABORN:
        sns.heatmap(matrix * 100, annot=True, fmt=".1f", cmap="Blues",
                    vmin=0, vmax=100, ax=ax,
                    cbar_kws={"label": "accuracy (%)"},
                    xticklabels=[f"T{j+1}" for j in range(T)],
                    yticklabels=[f"after T{t+1}" for t in range(T)])
    else:
        im = ax.imshow(matrix * 100, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(T)); ax.set_xticklabels([f"T{j+1}" for j in range(T)])
        ax.set_yticks(range(T)); ax.set_yticklabels([f"after T{t+1}" for t in range(T)])
        for t in range(T):
            for j in range(T):
                ax.text(j, t, f"{matrix[t,j]*100:.1f}", ha="center", va="center",
                        color="black" if matrix[t,j] < 0.5 else "white", fontsize=8)
        plt.colorbar(im, ax=ax, label="accuracy (%)")

    ax.set_title(f"Accuracy matrix — {run_id}")
    _save(fig, os.path.join(plots_dir, "accuracy_matrix.png"), dpi)


def plot_forgetting_curves(matrix: np.ndarray, plots_dir: str, run_id: str, dpi: int = 150) -> None:
    """One line per task: how its accuracy evolves over subsequent tasks."""
    _ensure_dir(plots_dir)
    T = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    for j in range(T):
        ys = [matrix[t, j] for t in range(j, T)]
        xs = list(range(j + 1, T + 1))
        ax.plot(xs, ys, "o-", label=f"Task {j + 1}")

    ax.set_xlabel("training step (after task t)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Forgetting curves — {run_id}")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(plots_dir, "forgetting_curves.png"), dpi)


# ── Per-task supervised plots ─────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, plots_dir: str, run_id: str,
                          task_id: int, dpi: int = 150) -> None:
    _ensure_dir(plots_dir)
    cm = confusion_data(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["0", "1"], yticklabels=["0", "1"])
    else:
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.colorbar(im, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion — Task {task_id + 1}")
    _save(fig, os.path.join(plots_dir, f"confusion_task{task_id+1}.png"), dpi)


def plot_roc(y_true, y_score, plots_dir: str, run_id: str, task_id: int, dpi: int = 150) -> None:
    if len(np.unique(y_true)) < 2:
        return
    _ensure_dir(plots_dir)
    fpr, tpr, _ = roc_curve_data(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="tab:orange", label="ROC")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC — Task {task_id + 1}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(plots_dir, f"roc_task{task_id+1}.png"), dpi)


def plot_pr(y_true, y_score, plots_dir: str, run_id: str, task_id: int, dpi: int = 150) -> None:
    if len(np.unique(y_true)) < 2:
        return
    _ensure_dir(plots_dir)
    p, r, _ = pr_curve_data(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(r, p, color="tab:green")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PR — Task {task_id + 1}")
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(plots_dir, f"pr_task{task_id+1}.png"), dpi)


def plot_probability_histogram(y_true, y_score, plots_dir: str, run_id: str,
                               task_id: int, dpi: int = 150) -> None:
    _ensure_dir(plots_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    bins = np.linspace(0, 1, 30)
    ax.hist(neg, bins=bins, alpha=0.5, label="class 0", color="tab:blue")
    ax.hist(pos, bins=bins, alpha=0.5, label="class 1", color="tab:red")
    ax.set_xlabel("predicted P(class=1)")
    ax.set_ylabel("count")
    ax.set_title(f"Probability histogram — Task {task_id + 1}")
    ax.legend()
    _save(fig, os.path.join(plots_dir, f"prob_hist_task{task_id+1}.png"), dpi)
