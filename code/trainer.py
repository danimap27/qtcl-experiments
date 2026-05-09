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

from data import SplitCIFAR10, SplitMNIST, get_task_loaders
from heads import get_head, count_trainable_params
from cl_methods import get_cl_method, ReplayCL, DERPPCL
from metrics import CLMetrics, supervised_metrics

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

try:
    import visualization as viz
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

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
    """Backbone + single shared head for continual learning.

    All tasks share the same classification head. EWC consolidation protects
    the head parameters that are important for prior tasks. This is the
    canonical Split-MNIST / Split-CIFAR-10 setup where catastrophic
    forgetting is actually measurable.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.head: Optional[nn.Module] = None

    def set_head(self, head: nn.Module) -> None:
        self.head = head

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        # task_id is accepted for API compatibility but ignored — single shared head
        z = self.backbone(x)
        return self.head(z)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_full(model: QTCLModel, loader, device: torch.device
                  ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (accuracy, y_true, y_pred, y_score) using the shared head."""
    model.eval()
    all_true, all_pred, all_score = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_true .append(y.cpu().numpy())
            all_pred .append(preds.cpu().numpy())
            all_score.append(probs[:, 1].cpu().numpy())   # P(class=1)

    if not all_true:
        empty = np.array([], dtype=int)
        return 0.0, empty, empty, np.array([], dtype=float)

    y_true  = np.concatenate(all_true)
    y_pred  = np.concatenate(all_pred)
    y_score = np.concatenate(all_score)
    acc = float((y_true == y_pred).mean())
    return acc, y_true, y_pred, y_score


def evaluate(model: QTCLModel, loader, device: torch.device, task_id: int = 0) -> float:
    """Return accuracy on a loader using the (single) shared head."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# ── Training loop ─────────────────────────────────────────────────────────────

def train_task(
    model: QTCLModel,
    train_loader,
    cl_method,
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

            logits = model(x)
            loss = criterion(logits, y)

            # Vanilla replay: hard-label CE on replayed batch
            if isinstance(cl_method, ReplayCL):
                replay = cl_method.replay_batch(x.size(0))
                if replay is not None:
                    rx, ry = replay
                    rx, ry = rx.to(device), ry.to(device)
                    loss = loss + cl_method.lam * criterion(model(rx), ry)

            # DER++: MSE on stored logits + CE on stored labels (two independent batches)
            elif isinstance(cl_method, DERPPCL) and cl_method.n_tasks_seen() > 0:
                ba, bb = cl_method.replay_batches(x.size(0))
                if ba is not None:
                    xa, za = ba
                    xa, za = xa.to(device), za.to(device)
                    loss = loss + cl_method.alpha * nn.functional.mse_loss(model(xa), za)
                if bb is not None:
                    xb, yb = bb
                    xb, yb = xb.to(device), yb.to(device)
                    loss = loss + cl_method.beta * criterion(model(xb), yb)

            # Regularisation penalty (EWC, L2, SI)
            if cl_method.n_tasks_seen() > 0:
                loss = loss + cl_method.penalty(model)

            cl_method.pre_step(model, loss)
            loss.backward()
            optimizer.step()
            cl_method.post_step(model)

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
    common_ds_kwargs = dict(
        root=ds_cfg["root"],
        n_tasks=n_tasks,
        image_size=ds_cfg.get("image_size", 32),
        subset_fraction=ds_cfg.get("subset_fraction", 1.0),
        train_test_split=ds_cfg.get("train_test_split", 0.8),
        train_val_split=ds_cfg.get("train_val_split", 0.8),
        seed=seed,
    )
    if ds_name == "split_cifar10":
        dataset = SplitCIFAR10(**common_ds_kwargs)
    elif ds_name == "split_mnist":
        dataset = SplitMNIST(**common_ds_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    # Build backbone
    bb_cfg = next(b for b in config["backbones"] if b["name"] == bb_name)
    backbone, feature_dim = load_backbone(bb_name, frozen=bb_cfg.get("frozen", True))
    backbone = backbone.to(device)

    model = QTCLModel(backbone, feature_dim).to(device)

    # Single shared head across all tasks → catastrophic forgetting is measurable
    shared_head = get_head(head_cfg, feature_dim, n_classes=2).to(device)
    model.set_head(shared_head)

    # Single Adam optimiser persists across all tasks (state preserved between tasks)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=overrides.get("lr", train_cfg["lr"]),
    )

    # CL method (EWC by default, configurable via overrides or config)
    cl_method_name = overrides.get("cl_method", config.get("cl_method", "ewc"))
    cl_method = get_cl_method(cl_method_name, lam=lambda_ewc,
                              buffer_size=ewc_cfg.get("buffer_size", 200))
    cl = CLMetrics(n_tasks)

    # Per-run plots/predictions output directory
    plots_dir = os.path.join(config.get("output_dir", "./results"), "plots", run_id)
    os.makedirs(plots_dir, exist_ok=True)

    # CodeCarbon energy tracking
    tracker = None
    energy_kwh   = float("nan")
    co2_kg       = float("nan")
    if HAS_CODECARBON:
        try:
            tracker = EmissionsTracker(
                project_name=f"qtcl_{run_id}",
                output_dir=plots_dir,
                output_file="emissions.csv",
                log_level="error",
                save_to_file=True,
            )
            tracker.start()
        except Exception as e:
            logger.warning(f"[{run_id}] CodeCarbon init failed: {e}")
            tracker = None

    # Keep test loaders for all tasks (used for the accuracy matrix)
    test_loaders = []
    val_loaders  = []
    all_training_logs = []
    per_task_metrics: List[Dict[str, Any]] = []

    t_start = time.time()

    for task_id in range(n_tasks):
        logger.info(f"[{run_id}] === Task {task_id+1}/{n_tasks}: {dataset.task_description(task_id)} ===")

        train_loader, val_loader, test_loader = get_task_loaders(
            dataset, task_id,
            batch_size=train_cfg["batch_size"],
            num_workers=2,
        )
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

        # Per-task scheduler (resets at each task boundary)
        sched_cfg = train_cfg.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 3),
            gamma=sched_cfg.get("gamma", 0.9),
        ) if sched_cfg else None

        epochs = overrides.get("epochs_per_task", train_cfg["epochs_per_task"])
        task_log = train_task(
            model, train_loader, cl_method, optimizer, scheduler,
            device, epochs, task_id, run_id,
        )
        all_training_logs.extend(task_log)

        # Consolidate CL state for this task (Fisher / Omega / anchor / replay)
        cl_method.update(model, train_loader, device, n_samples=ewc_cfg["fisher_samples"])

        # Evaluate ALL tasks seen-so-far + future ones (zero-shot for FWT).
        # Future tasks need their loaders built lazily.
        acc_row = []
        for j in range(n_tasks):
            if j > task_id and j >= len(test_loaders):
                # Build the loader for a future task (only first time)
                _, _, future_test = get_task_loaders(
                    dataset, j, batch_size=train_cfg["batch_size"], num_workers=2
                )
                while len(test_loaders) <= j:
                    test_loaders.append(None)
                    val_loaders.append(None)
                test_loaders[j] = future_test

            acc, y_true, y_pred, y_score = evaluate_full(model, test_loaders[j], device)

            # Save per-task supervised metrics + plots only for the current task
            if j == task_id and HAS_VIZ:
                sm = supervised_metrics(y_true, y_pred, y_score)
                sm.update({"task_id": task_id, "after_task": task_id})
                per_task_metrics.append(sm)
                try:
                    viz.plot_confusion_matrix(y_true, y_pred, plots_dir, run_id, task_id)
                    viz.plot_roc(y_true, y_score, plots_dir, run_id, task_id)
                    viz.plot_pr(y_true, y_score, plots_dir, run_id, task_id)
                    viz.plot_probability_histogram(y_true, y_score, plots_dir, run_id, task_id)
                except Exception as e:
                    logger.warning(f"[{run_id}] Plot failed for task {task_id+1}: {e}")

            acc_row.append(acc)
        cl.record(acc_row)
        logger.info(f"[{run_id}] Task {task_id+1} test eval: {[f'{a*100:.1f}' for a in acc_row]}")

    total_time = time.time() - t_start

    # Stop energy tracker
    if tracker is not None:
        try:
            emissions = tracker.stop()
            co2_kg = float(emissions) if emissions is not None else float("nan")
            energy_kwh = float(getattr(tracker.final_emissions_data, "energy_consumed", float("nan")))
        except Exception as e:
            logger.warning(f"[{run_id}] CodeCarbon stop failed: {e}")

    # Generate summary plots after all tasks
    if HAS_VIZ:
        try:
            viz.plot_learning_curves(all_training_logs, plots_dir, run_id)
            viz.plot_loss_combined(all_training_logs, plots_dir, run_id)
            viz.plot_accuracy_matrix(cl.matrix(), plots_dir, run_id)
            viz.plot_forgetting_curves(cl.matrix(), plots_dir, run_id)
        except Exception as e:
            logger.warning(f"[{run_id}] Summary plot generation failed: {e}")

    summary = cl.summary()

    # Aggregate per-task metrics (final state only)
    if per_task_metrics:
        agg = {
            f"final_{k}_mean": float(np.mean([m[k] for m in per_task_metrics if not np.isnan(m.get(k, np.nan))]))
            for k in ("precision", "recall", "f1", "roc_auc", "pr_auc")
            if any(not np.isnan(m.get(k, np.nan)) for m in per_task_metrics)
        }
    else:
        agg = {}

    result = {
        "run_id":      run_id,
        "dataset":     ds_name,
        "backbone":    bb_name,
        "head":        head_name,
        "seed":        seed,
        "study":       getattr(run_config, "study", "main"),
        "n_tasks":     n_tasks,
        "cl_method":   cl_method_name,
        "lambda_ewc":  lambda_ewc,
        "ansatz":      head_cfg.get("ansatz", "circular"),
        "variant":     head_cfg.get("variant", "mlp"),
        "n_qubits":    head_cfg.get("n_qubits", "N/A"),
        "depth":       head_cfg.get("depth", "N/A"),
        "noise":       head_cfg.get("noise", False),
        "AA":          summary["AA"],
        "AF":          summary["AF"],
        "BWT":         summary["BWT"],
        "FWT":         summary.get("FWT", 0.0),
        "train_time_s": total_time,
        "energy_kwh":  energy_kwh,
        "co2_kg":      co2_kg,
        "n_params_head": count_trainable_params(model.head) if model.head is not None else 0,
        **agg,                                                          # final_precision_mean, final_f1_mean, ...
        **{f"task_{j+1}_final_acc": cl.A[-1][j] for j in range(n_tasks)},
    }

    logger.info(
        f"[{run_id}] DONE — AA={summary['AA']*100:.2f}% "
        f"AF={summary['AF']*100:.2f}% BWT={summary['BWT']*100:.2f}% "
        f"time={total_time:.1f}s"
    )

    return result, [], all_training_logs
