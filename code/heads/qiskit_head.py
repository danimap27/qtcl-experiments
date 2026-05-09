"""
Qiskit hybrid quantum-classical classification head.

Dual-path architecture: classical MLP branch + quantum VQC branch.
Both ideal and noisy backends use EstimatorQNN with single-qubit Pauli-Z
observables, which scales linearly with n_qubits (avoids the 2^n SamplerQNN
output explosion).

Ideal: qiskit.primitives.StatevectorEstimator (exact statevector).
Noisy: qiskit_aer.primitives.EstimatorV2 with IBM Heron r2 noise model.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List


def _build_vqc(n_qubits: int, depth: int, ansatz: str = "circular"):
    """
    Parameterized VQC with angle encoding and configurable ansatz.

    Args:
        ansatz: 'circular'    — R_Y rotations + circular CNOTs (default)
                'linear'      — R_Y rotations + linear CNOTs (no wrap-around)
                'no_ent'      — R_Y rotations only, no entanglement
                'real_amp'    — RealAmplitudes (R_Y + full entanglement)
                'efficient_su2'— EfficientSU2 (R_Y, R_Z + entanglement)
                'two_local'   — TwoLocal with R_X+R_Y rotation block
                're_upload'   — Data re-uploading (encoding repeated each layer)
    """
    from qiskit.circuit import QuantumCircuit, ParameterVector

    inputs = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)
    weights: List = []

    if ansatz in ("circular", "linear", "no_ent", "re_upload"):
        wp = ParameterVector("w", n_qubits * depth)
        weights = list(wp)

        # Initial encoding
        for i in range(n_qubits):
            qc.ry(inputs[i], i)

        idx = 0
        for _ in range(depth):
            if ansatz == "re_upload":
                for i in range(n_qubits):
                    qc.ry(inputs[i], i)
            for i in range(n_qubits):
                qc.ry(wp[idx], i)
                idx += 1
            if ansatz == "circular":
                for i in range(n_qubits - 1):
                    qc.cx(i, i + 1)
                if n_qubits > 1:
                    qc.cx(n_qubits - 1, 0)
            elif ansatz == "linear":
                for i in range(n_qubits - 1):
                    qc.cx(i, i + 1)
            elif ansatz == "re_upload":
                for i in range(n_qubits - 1):
                    qc.cx(i, i + 1)
            # no_ent: no CNOTs

        return qc, list(inputs), weights

    # Library ansätze
    from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal

    # Encoding first
    for i in range(n_qubits):
        qc.ry(inputs[i], i)

    if ansatz == "real_amp":
        ans = RealAmplitudes(n_qubits, reps=depth, entanglement="full")
    elif ansatz == "efficient_su2":
        ans = EfficientSU2(n_qubits, reps=depth, entanglement="circular")
    elif ansatz == "two_local":
        ans = TwoLocal(n_qubits, rotation_blocks=["rx", "ry"],
                       entanglement_blocks="cz", reps=depth, entanglement="linear")
    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")

    qc.compose(ans, qubits=range(n_qubits), inplace=True)
    weights = list(ans.parameters)
    return qc, list(inputs), weights


def _build_noise_model(noise_params: Dict[str, float], noise_channels: Optional[List[str]] = None):
    """IBM Heron r2-calibrated noise model."""
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError

    if noise_channels is None:
        noise_channels = ["amplitude_damping", "phase_damping", "depolarizing"]

    T1   = noise_params.get("T1_us", 250) * 1e3
    T2   = noise_params.get("T2_us", 150) * 1e3
    t1q  = noise_params.get("t1q_ns", 32)
    t2q  = noise_params.get("t2q_ns", 68)
    p1q  = noise_params.get("p1q", 0.0002)
    p2q  = noise_params.get("p2q", 0.005)
    p_ro = noise_params.get("readout_error", 0.012)

    nm = NoiseModel()

    if "amplitude_damping" in noise_channels or "phase_damping" in noise_channels:
        e1 = thermal_relaxation_error(T1, T2, t1q)
        e2_single = thermal_relaxation_error(T1, T2, t2q)
        e2 = e2_single.tensor(e2_single)
        nm.add_all_qubit_quantum_error(e1, ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x"])
        nm.add_all_qubit_quantum_error(e2, ["cx"])
    if "depolarizing" in noise_channels:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1q, 1),
                                       ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), ["cx"])

    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def _make_estimator(noise: bool, noise_params: Optional[Dict], noise_channels: Optional[List[str]],
                    shots: Optional[int]):
    """Return an Estimator primitive (V2)."""
    if not noise:
        from qiskit.primitives import StatevectorEstimator
        return StatevectorEstimator()

    # Noisy: Aer's native EstimatorV2 with noise_model in backend_options.
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
    nm = _build_noise_model(noise_params or {}, noise_channels)
    s = shots or 1024
    return AerEstimatorV2(
        options={
            "default_precision": 1.0 / (s ** 0.5),
            "backend_options": {"noise_model": nm},
            "run_options": {"shots": s},
        }
    )


class QiskitHead(nn.Module):
    """
    Dual-path hybrid head:
        Classical: Linear(d, 128) -> ReLU -> Linear(128, n_classes)
        Quantum:   Linear(d, n_q) -> tanh -> VQC -> Linear(n_q, n_classes)
        Output:    y_classical + alpha * y_quantum  (alpha learnable, init 0.5)
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int = 2,
        n_qubits: int = 4,
        depth: int = 2,
        backend: str = "statevector",
        noise: bool = False,
        shots: Optional[int] = None,
        noise_params: Optional[Dict[str, float]] = None,
        noise_channels: Optional[List[str]] = None,
        gradient_method: str = "reverse",
        ansatz: str = "circular",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.ansatz = ansatz

        self.classical_branch = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

        self.q_encoder = nn.Linear(feature_dim, n_qubits)

        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.connectors import TorchConnector
        from qiskit.quantum_info import SparsePauliOp

        qc, inputs_pv, weights_pv = _build_vqc(n_qubits, depth, ansatz=ansatz)

        # Single-qubit Pauli-Z observables (n_qubits outputs, NOT 2^n_qubits)
        observables = []
        for i in range(n_qubits):
            pauli_str = "I" * i + "Z" + "I" * (n_qubits - i - 1)
            observables.append(SparsePauliOp.from_list([(pauli_str[::-1], 1.0)]))

        estimator = _make_estimator(noise, noise_params, noise_channels, shots)

        qnn = EstimatorQNN(
            circuit=qc,
            estimator=estimator,
            observables=observables,
            input_params=list(inputs_pv),
            weight_params=list(weights_pv),
        )

        self.q_layer = TorchConnector(qnn)
        self.q_decoder = nn.Linear(n_qubits, n_classes)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_c = self.classical_branch(x)
        z = torch.tanh(self.q_encoder(x))
        q_out = self.q_layer(z)
        y_q = self.q_decoder(q_out)
        return y_c + self.alpha * y_q
