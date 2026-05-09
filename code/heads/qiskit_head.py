"""
Qiskit hybrid quantum-classical classification head.

Dual-path architecture: classical MLP branch + quantum VQC branch.
Supports both ideal (StatevectorEstimator) and noisy (AerSimulator) backends.
Noise model is calibrated to IBM Heron r2 device parameters.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List


def _build_vqc(n_qubits: int, depth: int):
    """Build parameterized VQC with angle encoding and circular entanglement."""
    from qiskit.circuit import QuantumCircuit, ParameterVector

    inputs  = ParameterVector("x", n_qubits)
    weights = ParameterVector("w", n_qubits * depth)

    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(inputs[i], i)

    idx = 0
    for _ in range(depth):
        for i in range(n_qubits):
            qc.ry(weights[idx], i)
            idx += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)

    return qc, inputs, weights


def _build_noise_model(noise_params: Dict[str, float], noise_channels: Optional[List[str]] = None):
    """
    Build Qiskit Aer noise model calibrated to IBM Heron r2 parameters.

    Args:
        noise_params:   Dict with keys T1_us, T2_us, t1q_ns, t2q_ns, p1q, p2q, readout_error.
        noise_channels: List of channels to activate (all activated if None).
                        Options: 'amplitude_damping', 'phase_damping', 'depolarizing'.
    """
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError

    if noise_channels is None:
        noise_channels = ["amplitude_damping", "phase_damping", "depolarizing"]

    T1  = noise_params.get("T1_us", 250) * 1e3    # ns
    T2  = noise_params.get("T2_us", 150) * 1e3    # ns
    t1q = noise_params.get("t1q_ns", 32)           # single-qubit gate time (ns)
    t2q = noise_params.get("t2q_ns", 68)           # two-qubit gate time (ns)
    p1q = noise_params.get("p1q", 0.0002)
    p2q = noise_params.get("p2q", 0.005)
    p_ro = noise_params.get("readout_error", 0.012)

    noise_model = NoiseModel()

    # Thermal relaxation on single-qubit gates
    if "amplitude_damping" in noise_channels or "phase_damping" in noise_channels:
        t1q_err = thermal_relaxation_error(T1, T2, t1q)
        noise_model.add_all_qubit_quantum_error(t1q_err, ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x"])

    # Thermal relaxation on two-qubit gates
    if "amplitude_damping" in noise_channels or "phase_damping" in noise_channels:
        t2q_err = thermal_relaxation_error(T1, T2, t2q).expand(2)
        noise_model.add_all_qubit_quantum_error(t2q_err, ["cx"])

    # Depolarizing error
    if "depolarizing" in noise_channels:
        dep1 = depolarizing_error(p1q, 1)
        dep2 = depolarizing_error(p2q, 2)
        noise_model.add_all_qubit_quantum_error(dep1, ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x"])
        noise_model.add_all_qubit_quantum_error(dep2, ["cx"])

    # Readout error
    ro_err = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    noise_model.add_all_qubit_readout_error(ro_err)

    return noise_model


class QiskitHead(nn.Module):
    """
    Dual-path hybrid quantum-classical head using Qiskit.

    Classical path: Linear(feature_dim, 128) -> ReLU -> Linear(128, n_classes)
    Quantum path:   Linear(feature_dim, n_qubits) -> tanh -> VQC -> Linear(n_qubits, n_classes)
    Output:         y_c + alpha * y_q   (alpha is a learnable scalar)

    Args:
        feature_dim:    Dimension of the input feature vector from the backbone.
        n_classes:      Number of output classes (2 for binary classification).
        n_qubits:       Number of qubits in the VQC.
        depth:          Number of variational layers in the VQC.
        backend:        'statevector' (ideal) or 'aer' (noisy simulation).
        noise:          Enable noise model (requires backend='aer').
        shots:          Number of measurement shots (None = exact statevector).
        noise_params:   Dict of noise parameters for the Heron r2 model.
        noise_channels: Subset of noise channels to activate.
        gradient_method:'reverse' (ideal) or 'spsa' (noisy).
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
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes

        # Classical branch
        self.classical_branch = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

        # Quantum encoder
        self.q_encoder = nn.Linear(feature_dim, n_qubits)

        # Build Qiskit QNN
        from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
        from qiskit_machine_learning.connectors import TorchConnector
        from qiskit.quantum_info import SparsePauliOp

        qc, inputs_pv, weights_pv = _build_vqc(n_qubits, depth)

        if backend == "statevector" and not noise:
            from qiskit.primitives import StatevectorEstimator
            estimator = StatevectorEstimator()
            observables = []
            for i in range(n_qubits):
                pauli_str = "I" * i + "Z" + "I" * (n_qubits - i - 1)
                observables.append(SparsePauliOp.from_list([(pauli_str[::-1], 1.0)]))
            qnn = EstimatorQNN(
                circuit=qc,
                estimator=estimator,
                observables=observables,
                input_params=list(inputs_pv),
                weight_params=list(weights_pv),
            )
        else:
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import SamplerV2 as AerSampler

            nm = _build_noise_model(noise_params or {}, noise_channels) if noise else None
            aer_backend = AerSimulator(noise_model=nm) if nm else AerSimulator()
            sampler = AerSampler.from_backend(aer_backend)

            qc_meas = qc.copy()
            qc_meas.measure_all()

            qnn = SamplerQNN(
                circuit=qc_meas,
                sampler=sampler,
                input_params=list(inputs_pv),
                weight_params=list(weights_pv),
                shots=shots or 1024,
            )

        self.q_layer = TorchConnector(qnn)
        self.q_decoder = nn.Linear(qnn.output_shape[-1], n_classes)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_c = self.classical_branch(x)
        z = torch.tanh(self.q_encoder(x))
        q_out = self.q_layer(z)
        y_q = self.q_decoder(q_out)
        return y_c + self.alpha * y_q
