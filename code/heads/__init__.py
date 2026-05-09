from .classical_head import ClassicalHead
from .qiskit_head import QiskitHead

import torch.nn as nn
from typing import Dict, Any


def get_head(head_cfg: Dict[str, Any], feature_dim: int, n_classes: int = 2) -> nn.Module:
    """Instantiate a classification head from a config dict."""
    head_type = head_cfg.get("type", "classical")
    if head_type == "classical":
        return ClassicalHead(
            feature_dim=feature_dim,
            n_classes=n_classes,
            hidden_dim=head_cfg.get("hidden_dim", 128),
            dropout=head_cfg.get("dropout", 0.3),
        )
    elif head_type == "qiskit":
        return QiskitHead(
            feature_dim=feature_dim,
            n_classes=n_classes,
            n_qubits=head_cfg.get("n_qubits", 4),
            depth=head_cfg.get("depth", 2),
            backend=head_cfg.get("backend", "statevector"),
            noise=head_cfg.get("noise", False),
            shots=head_cfg.get("shots", None),
            noise_params=head_cfg.get("noise_params", None),
            noise_channels=head_cfg.get("noise_channels", None),
            gradient_method=head_cfg.get("gradient_method", "reverse"),
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["ClassicalHead", "QiskitHead", "get_head", "count_trainable_params"]
