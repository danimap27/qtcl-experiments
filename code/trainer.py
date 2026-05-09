"""
QTCL Trainer — Quantum Transfer-Continual Learning.

Trains a backbone + classification head sequentially across T tasks
using EWC regularisation. Evaluates on all previous tasks after each task.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from data import SplitCIFAR10, SplitCIFAR100, get_task_loaders
from heads import get_head, count_trainable_params
from ewc import EWC
from metrics import CLMetrics

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)


# ── Backbone loading ──────────────────────────────────────────────────────────

def load_backbone(backbone_name: str, frozen: bool = True) -> Tuple[nn.Module, int]:
    """
    Load a pretrained backbone and return (model, feature_dim).

    Args:
        backbone_name: 'resnet18', 'mobilenetv2', or 'efficientnet_b0'.
        frozen:        If True, freeze all backbone parameters.

    Returns:
        (backbone_without_classifier, feature_dim)
    """
    if backbone_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()

    elif backbone_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()

    elif backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    if frozen:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    return model, feature_dim


# ── Full model wrapper ────────────────────────────────────────────────────────

class QTCLModel(nn.Module):
    """Backbone + multi-head wrapper for continual learning."""

    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.heads: nn.ModuleList = nn.ModuleList()
        self.current_task: int = 0

    def add_head(self, head: nn.Module) -> None:
        self.heads.append(head)
        self.current_task = len(self.heads) - 1

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        tid = task_id if task_id is not None else self.current_task
        with torch.set_grad_enabled(not self.backbone.training or len(list(self.backbone.parameters())) == 0):
            z = self.backbone(x)
        return self.heads[tid](z)

    def parameters_for_ewc(self):
        """Yield all trainable parameters (backbone if unfrozen + all heads)."""
        for p in self.parameters():
            if p.requires_grad:
                yield p


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: QTCLModel, loader, device: torch.device, task_id: int) -> float:
    """Return accuracy on a single task's validation loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, task_id=task_id)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# ── Training loop ─────────────────────────────────────────────────────────────

def train_task(
    model: QTCLModel,
    train_loader,
    ewc: EWC,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    task_id: int,
    run_id: str,
) -> List[Dict[str, float]]:
    """Train on a single task for `epochs` epochs. Returns training log."""
    criterion = nn.CrossEntropyLoss()
    training_log = []

    for epoch in range(epochs):
        model.train()
        model.backbone.eval()  # keep BN in eval if frozen

        epoch_loss = epoch_correct = epoch_total = 0

        it = tqdm(train_loader, desc=f"  [{run_id}] Task {task_id+1} Epoch {epoch+1}/{epochs}", leave=False) \
            if HAS_TQDM else train_loader

        t0 = time.time()
        for x, y in it:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x, task_id=task_id)
            loss = criterion(logits, y)

            if ewc.n_tasks_seen() > 0:
                loss = loss + ewc.penalty(model)

            loss.backward()
            optimizer.step()

            epoch_loss    += loss.item() * y.size(0)
            epoch_correct += (logits.argmax(1) == y).sum().item()
            epoch_total   += y.size(0)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(epoch_total, 1)
        avg_acc  = epoch_correct / max(epoch_total, 1)

        log_entry = {
            "task_id": task_id,
            "epoch":   epoch,
            "loss":    avg_loss,
            "acc":     avg_acc,
            "time_s":  epoch_time,
        }
        training_log.append(log_entry)
        logger.info(
            f"  [{run_id}] Task {task_id+1} Epoch {epoch+1}/{epochs} "
            f"loss={avg_loss:.4f} acc={avg_acc*100:.1f}% ({epoch_time:.1f}s)"
        )

    return training_log


# ── Main entry point ──────────────────────────────────────────────────────────

def train_and_evaluate(
    run_config,
    config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict], List[Dict]]:
    """
    Full QTCL training pipeline for one (dataset, backbone, head, seed) combination.

    Returns:
        result:       Summary dict with AA, AF, BWT, timing, param counts.
        predictions:  Per-task-per-sample predictions (empty list for now).
        training_log: Per-epoch training metrics.
    """
    if overrides is None:
        overrides = {}

    run_id   = run_config.run_id
    seed     = run_config.seed
    ds_name  = run_config.dataset
    bb_name  = run_config.backbone
    head_name = run_config.head

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info(f"[{run_id}] device={device} seed={seed}")

    # Dataset config
    ds_cfg = next(d for d in config["datasets"] if d["name"] == ds_name)
    n_tasks = overrides.get("n_tasks", ds_cfg["n_tasks"])

    train_cfg = config["training"]
    ewc_cfg   = config["ewc"]
    lambda_ewc = overrides.get("lambda", ewc_cfg["lambda"])

    # Head config
    head_cfg_list = [h for h in config["heads"] if h["name"] == head_name]
    if not head_cfg_list:
        raise ValueError(f"Head '{head_name}' not found in config")
    head_cfg = dict(head_cfg_list[0])
    if "n_qubits" in overrides:
        head_cfg["n_qubits"] = overrides["n_qubits"]
    if "depth" in overrides:
        head_cfg["depth"] = overrides["depth"]
    if "noise_channels" in overrides:
        head_cfg["noise_channels"] = overrides["noise_channels"]

    # Build dataset
    if ds_name == "split_cifar10":
        dataset = SplitCIFAR10(
            root=ds_cfg["root"],
            n_tasks=n_tasks,
            image_size=ds_cfg.get("image_size", 32),
            samples_per_task_train=ds_cfg.get("samples_per_task_train"),
            samples_per_task_val=ds_cfg.get("samples_per_task_val"),
        )
    elif ds_name == "split_cifar100":
        dataset = SplitCIFAR100(
            root=ds_cfg["root"],
            n_tasks=n_tasks,
            image_size=ds_cfg.get("image_size", 32),
            samples_per_task_train=ds_cfg.get("samples_per_task_train"),
            samples_per_task_val=ds_cfg.get("samples_per_task_val"),
        )
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    # Build backbone
    bb_cfg = next(b for b in config["backbones"] if b["name"] == bb_name)
    backbone, feature_dim = load_backbone(bb_name, frozen=bb_cfg.get("frozen", True))
    backbone = backbone.to(device)

    model = QTCLModel(backbone, feature_dim).to(device)
    ewc   = EWC(lambda_ewc)
    cl    = CLMetrics(n_tasks)

    # Keep validation loaders for all tasks
    val_loaders = []
    all_training_logs = []

    t_start = time.time()

    for task_id in range(n_tasks):
        logger.info(f"[{run_id}] === Task {task_id+1}/{n_tasks}: {dataset.task_description(task_id)} ===")

        train_loader, val_loader = get_task_loaders(
            dataset, task_id,
            batch_size=train_cfg["batch_size"],
            num_workers=2,
        )
        val_loaders.append(val_loader)

        # Add new head for this task
        head = get_head(head_cfg, feature_dim, n_classes=2).to(device)
        model.add_head(head)

        # Optimise only the new head (backbone frozen)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=overrides.get("lr", train_cfg["lr"]),
        )
        sched_cfg = train_cfg.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 3),
            gamma=sched_cfg.get("gamma", 0.9),
        ) if sched_cfg else None

        epochs = overrides.get("epochs_per_task", train_cfg["epochs_per_task"])
        task_log = train_task(
            model, train_loader, ewc, optimizer, scheduler,
            device, epochs, task_id, run_id,
        )
        all_training_logs.extend(task_log)

        # Update EWC with Fisher for this task
        ewc.update(model, train_loader, device, n_samples=ewc_cfg["fisher_samples"])

        # Evaluate on all tasks seen so far
        acc_row = []
        for j in range(n_tasks):
            if j <= task_id:
                acc = evaluate(model, val_loaders[j], device, task_id=j)
            else:
                acc = 0.0
            acc_row.append(acc)
        cl.record(acc_row)
        logger.info(f"[{run_id}] Task {task_id+1} eval: {[f'{a*100:.1f}' for a in acc_row]}")

    total_time = time.time() - t_start
    summary = cl.summary()

    result = {
        "run_id":      run_id,
        "dataset":     ds_name,
        "backbone":    bb_name,
        "head":        head_name,
        "seed":        seed,
        "n_tasks":     n_tasks,
        "lambda_ewc":  lambda_ewc,
        "n_qubits":    head_cfg.get("n_qubits", "N/A"),
        "depth":       head_cfg.get("depth", "N/A"),
        "noise":       head_cfg.get("noise", False),
        "AA":          summary["AA"],
        "AF":          summary["AF"],
        "BWT":         summary["BWT"],
        "train_time_s": total_time,
        "n_params_head": count_trainable_params(model.heads[-1]) if model.heads else 0,
        **{f"task_{j+1}_final_acc": cl.A[-1][j] for j in range(n_tasks)},
    }

    logger.info(
        f"[{run_id}] DONE — AA={summary['AA']*100:.2f}% "
        f"AF={summary['AF']*100:.2f}% BWT={summary['BWT']*100:.2f}% "
        f"time={total_time:.1f}s"
    )

    return result, [], all_training_logs
