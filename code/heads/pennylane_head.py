"""
PennyLane variational quantum circuit classification head.

Encodes classical features via angle embedding, applies StronglyEntanglingLayers,
and maps Pauli-Z expectation values to classification logits.
"""

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp


class PennyLaneHead(nn.Module):
    """
    Hybrid quantum-classical head using PennyLane VQC.

    Pipeline:
        1. Linear encoder: feature_dim -> n_qubits (with tanh activation)
        2. Angle embedding: R_Y(pi * x_i) on each qubit
        3. StronglyEntanglingLayers with `depth` layers
        4. Measure <Z_i> for i = 1..n_qubits
        5. Linear decoder: n_qubits -> n_classes
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int = 2,
        n_qubits: int = 4,
        depth: int = 2,
        backend: str = "default.qubit",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, n_qubits),
            nn.Tanh(),
        )

        dev = qml.device(backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs * torch.pi, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
        self.q_weights = nn.Parameter(torch.randn(*weight_shape) * 0.1)

        self.decoder = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        outputs = []
        for sample in z:
            meas = self.circuit(sample, self.q_weights)
            outputs.append(torch.stack(meas))
        q_out = torch.stack(outputs)
        return self.decoder(q_out)
