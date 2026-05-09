"""
Continual learning regularisers for QTCL.

Unified interface so the trainer can swap methods via config:
    naive   — no regularisation (catastrophic baseline)
    l2      — L2 distance to last anchor
    ewc     — Elastic Weight Consolidation (Fisher-weighted)
    si      — Synaptic Intelligence (Zenke et al., 2017)
    mas     — Memory Aware Synapses (Aljundi et al., 2018)
    replay  — Experience Replay (random buffer)

Every regulariser exposes:
    .penalty(model)            -> scalar tensor (loss term to add)
    .update(model, loader, dev) -> called after each task is finished
    .replay_batch(batch_size)  -> optional, for replay methods
    .pre_step(model, loss)     -> optional, for online methods (SI)
    .post_step(model)          -> optional, for online methods (SI)
"""

from typing import Dict, List, Optional, Tuple
import copy
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseCLMethod:
    """Default no-op behavior."""
    name = "base"

    def __init__(self, lam: float = 0.0):
        self.lam = lam

    def penalty(self, model: nn.Module) -> torch.Tensor:
        return torch.tensor(0.0)

    def update(self, model: nn.Module, loader: DataLoader, device: torch.device,
               n_samples: int = 200) -> None:
        pass

    def pre_step(self, model: nn.Module, loss: torch.Tensor) -> None:
        pass

    def post_step(self, model: nn.Module) -> None:
        pass

    def replay_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return None

    def n_tasks_seen(self) -> int:
        return 0


# ── Naive ─────────────────────────────────────────────────────────────────────

class NaiveCL(BaseCLMethod):
    """No regularisation. Baseline showing maximal forgetting."""
    name = "naive"


# ── L2 ────────────────────────────────────────────────────────────────────────

class L2CL(BaseCLMethod):
    """Penalise drift from last task's optimum (uniform importance)."""
    name = "l2"

    def __init__(self, lam: float = 1.0):
        super().__init__(lam)
        self._anchors: List[Dict[str, torch.Tensor]] = []

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self._anchors:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        params = dict(model.named_parameters())
        for anchor in self._anchors:
            for name, theta_star in anchor.items():
                if name in params:
                    p = params[name]
                    loss = loss + (p - theta_star.to(p.device)).pow(2).sum()
        return (self.lam / 2.0) * loss

    def update(self, model: nn.Module, loader: DataLoader, device: torch.device,
               n_samples: int = 200) -> None:
        self._anchors.append({
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        })

    def n_tasks_seen(self) -> int:
        return len(self._anchors)


# ── EWC ───────────────────────────────────────────────────────────────────────

class EWCCL(BaseCLMethod):
    """Elastic Weight Consolidation (Kirkpatrick et al. 2017)."""
    name = "ewc"

    def __init__(self, lam: float = 5000.0):
        super().__init__(lam)
        self._fisher: List[Dict[str, torch.Tensor]] = []
        self._optima: List[Dict[str, torch.Tensor]] = []

    def update(self, model: nn.Module, loader: DataLoader, device: torch.device,
               n_samples: int = 200) -> None:
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        crit = nn.CrossEntropyLoss()
        count = 0
        for x, y in loader:
            if count >= n_samples:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            crit(model(x), y).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2) * x.size(0)
            count += x.size(0)
        count = max(count, 1)
        for n in fisher:
            fisher[n] /= count
        self._fisher.append(fisher)
        self._optima.append({
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        })

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self._fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        params = dict(model.named_parameters())
        for fisher, optima in zip(self._fisher, self._optima):
            for n, f in fisher.items():
                if n in params:
                    p = params[n]
                    loss = loss + (f.to(p.device) * (p - optima[n].to(p.device)).pow(2)).sum()
        return (self.lam / 2.0) * loss

    def n_tasks_seen(self) -> int:
        return len(self._fisher)


# ── SI (Synaptic Intelligence) ────────────────────────────────────────────────

class SICL(BaseCLMethod):
    """
    Synaptic Intelligence (Zenke et al. 2017).
    Accumulates per-parameter importance online during training as the
    contribution of each weight to the loss decrease (path integral).
    """
    name = "si"

    def __init__(self, lam: float = 1.0, epsilon: float = 1e-3):
        super().__init__(lam)
        self.epsilon = epsilon
        self._omega: Dict[str, torch.Tensor]   = {}     # accumulated importance
        self._w:     Dict[str, torch.Tensor]   = {}     # online contribution
        self._prev:  Dict[str, torch.Tensor]   = {}     # last-step parameter
        self._anchor: Dict[str, torch.Tensor]  = {}     # task end optimum
        self._n_tasks = 0

    def _ensure_init(self, model: nn.Module) -> None:
        if self._omega:
            return
        for n, p in model.named_parameters():
            if p.requires_grad:
                self._omega[n]   = torch.zeros_like(p)
                self._w[n]       = torch.zeros_like(p)
                self._prev[n]    = p.detach().clone()
                self._anchor[n]  = p.detach().clone()

    def pre_step(self, model: nn.Module, loss: torch.Tensor) -> None:
        self._ensure_init(model)

    def post_step(self, model: nn.Module) -> None:
        self._ensure_init(model)
        for n, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            delta = p.detach() - self._prev[n]
            self._w[n] += -p.grad.detach() * delta   # contribution
            self._prev[n] = p.detach().clone()

    def update(self, model: nn.Module, loader: DataLoader, device: torch.device,
               n_samples: int = 200) -> None:
        self._ensure_init(model)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            theta_curr = p.detach().clone()
            change = (theta_curr - self._anchor[n]).pow(2) + self.epsilon
            self._omega[n] += torch.clamp(self._w[n] / change, min=0)
            self._w[n].zero_()
            self._anchor[n] = theta_curr
            self._prev[n]   = theta_curr.clone()
        self._n_tasks += 1

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if self._n_tasks == 0 or not self._omega:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self._omega:
                w = self._omega[n].to(p.device)
                a = self._anchor[n].to(p.device)
                loss = loss + (w * (p - a).pow(2)).sum()
        return self.lam * loss

    def n_tasks_seen(self) -> int:
        return self._n_tasks


# ── MAS (Memory Aware Synapses) ───────────────────────────────────────────────

class MASCL(BaseCLMethod):
    """
    Memory Aware Synapses (Aljundi et al. 2018).
    Importance = average squared L2 norm of the gradient of model output
    sensitivity. Computed unsupervised (no labels needed for importance).
    """
    name = "mas"

    def __init__(self, lam: float = 1.0):
        super().__init__(lam)
        self._omega: Dict[str, torch.Tensor]  = {}
        self._anchor: Dict[str, torch.Tensor] = {}
        self._n_tasks = 0

    def update(self, model: nn.Module, loader: DataLoader, device: torch.device,
               n_samples: int = 200) -> None:
        model.eval()
        omega = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        count = 0
        for x, _ in loader:
            if count >= n_samples:
                break
            x = x.to(device)
            model.zero_grad()
            out = model(x)
            # MAS sensitivity: || f(x) ||^2
            sensitivity = out.pow(2).sum()
            sensitivity.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    omega[n] += p.grad.detach().abs() * x.size(0)
            count += x.size(0)
        count = max(count, 1)
        for n in omega:
            omega[n] /= count
            if n in self._omega:
                self._omega[n] += omega[n]
            else:
                self._omega[n] = omega[n]
        self._anchor = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self._n_tasks += 1

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self._omega:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self._omega:
                w = self._omega[n].to(p.device)
                a = self._anchor[n].to(p.device)
                loss = loss + (w * (p - a).pow(2)).sum()
        return (self.lam / 2.0) * loss

    def n_tasks_seen(self) -> int:
        return self._n_tasks


# ── Experience Replay ─────────────────────────────────────────────────────────

class ReplayCL(BaseCLMethod):
    """
    Random reservoir sampling buffer of past samples.
    The trainer should call replay_batch() each step and add the replay loss.
    """
    name = "replay"

    def __init__(self, lam: float = 1.0, buffer_size: int = 200):
        super().__init__(lam)
        self.buffer_size = buffer_size
        self._buffer_x: List[torch.Tensor] = []
        self._buffer_y: List[torch.Tensor] = []
        self._seen = 0
        self._n_tasks = 0

    def store(self, x: torch.Tensor, y: torch.Tensor) -> None:
        for i in range(x.size(0)):
            if len(self._buffer_x) < self.buffer_size:
                self._buffer_x.append(x[i].detach().cpu())
                self._buffer_y.append(y[i].detach().cpu())
            else:
                idx = random.randint(0, self._seen)
                if idx < self.buffer_size:
                    self._buffer_x[idx] = x[i].detach().cpu()
                    self._buffer_y[idx] = y[i].detach().cpu()
            self._seen += 1

    def update(self, model: nn.Module, loader: DataLoader, device: torch.device,
               n_samples: int = 200) -> None:
        # Snapshot the last samples seen during training of this task
        count = 0
        for x, y in loader:
            self.store(x, y)
            count += x.size(0)
            if count >= n_samples:
                break
        self._n_tasks += 1

    def replay_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self._buffer_x:
            return None
        idxs = random.sample(range(len(self._buffer_x)), min(batch_size, len(self._buffer_x)))
        xb = torch.stack([self._buffer_x[i] for i in idxs])
        yb = torch.stack([self._buffer_y[i] for i in idxs])
        return xb, yb

    def n_tasks_seen(self) -> int:
        return self._n_tasks


# ── Factory ───────────────────────────────────────────────────────────────────

def get_cl_method(name: str, lam: float = 5000.0, **kwargs) -> BaseCLMethod:
    name = (name or "ewc").lower()
    if name == "naive":
        return NaiveCL(lam=lam)
    if name == "l2":
        return L2CL(lam=lam)
    if name == "ewc":
        return EWCCL(lam=lam)
    if name == "si":
        return SICL(lam=lam, epsilon=kwargs.get("epsilon", 1e-3))
    if name == "mas":
        return MASCL(lam=lam)
    if name == "replay":
        return ReplayCL(lam=lam, buffer_size=kwargs.get("buffer_size", 200))
    raise ValueError(f"Unknown cl_method: {name}")
