# QTCL — Quantum Transfer-Continual Learning

Hybrid quantum-classical framework for mitigating catastrophic forgetting in sequential task learning. Combines ImageNet-pretrained backbones with a dual-path Qiskit classification head (classical MLP + variational quantum circuit) regularised by Elastic Weight Consolidation (EWC).

## Setup

```bash
conda create -n qtcl python=3.11
conda activate qtcl
pip install -r code/requirements.txt
```

## Structure

```
QTCL/
├── paper/
│   ├── main.tex          # Full paper (ACM Large format)
│   └── references.bib
└── code/
    ├── runner.py         # Experiment orchestrator
    ├── trainer.py        # CL training loop with EWC
    ├── ewc.py            # Fisher Information + EWC penalty
    ├── metrics.py        # AA, AF, BWT from accuracy matrix
    ├── generate_tables.py# CSV results → LaTeX tables
    ├── config.yaml       # Full experiment configuration
    ├── slurm_main.sh     # SLURM job array (main battery)
    ├── slurm_ablation.sh # SLURM job array (circuit ablation)
    ├── slurm_lambda.sh   # SLURM job array (λ sensitivity)
    ├── data/
    │   ├── split_cifar10.py   # 5 binary tasks from CIFAR-10
    │   └── split_cifar100.py  # 10 binary tasks from CIFAR-100 superclasses
    └── heads/
        ├── classical_head.py  # MLP baseline
        └── qiskit_head.py     # Dual-path: MLP branch + VQC branch
```

## Experiments

### Main battery (2 datasets × 3 backbones × 3 heads × 5 seeds = 90 runs)

```bash
cd code
python runner.py --config config.yaml --dry-run          # preview
python runner.py --config config.yaml --count            # count runs
python runner.py --config config.yaml                    # execute all
```

### Circuit ablation (n_qubits × depth)

```bash
python runner.py --config config.yaml --study ablation
```

### EWC λ sensitivity

```bash
python runner.py --config config.yaml --study lambda_sensitivity
```

### Noise decomposition (per channel)

```bash
python runner.py --config config.yaml --study noise_decomposition
```

### Filter runs

```bash
python runner.py --config config.yaml --dataset split_cifar10 --backbone resnet18 --head qk_ideal --seed 42
```

### Generate LaTeX tables from results

```bash
python generate_tables.py --results-dir ./results --tables-dir ./paper/tables
```

## Hercules (CICA cluster)

Update the path in the SLURM scripts (`/path/to/QTCL/code`), then:

```bash
# Export one command per run
python runner.py --config config.yaml --dry-run --export-commands > slurm_commands_main.txt

# Submit job array
sbatch slurm_main.sh
sbatch slurm_ablation.sh
sbatch slurm_lambda.sh
```

Results are written to `./results/NNN_<machine-id>/runs.csv` (resumable: already-completed runs are skipped automatically).

## Configuration

Key parameters in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seeds` | `[0, 42, 123, 456, 789]` | Random seeds for reproducibility |
| `ewc.lambda` | `5000` | EWC regularisation strength |
| `ewc.fisher_samples` | `200` | Samples for Fisher estimation |
| `training.epochs_per_task` | `10` | Training epochs per task |
| `heads[qk_ideal].n_qubits` | `4` | Qubits in VQC |
| `heads[qk_ideal].depth` | `2` | Variational layers |
| `heads[qk_noisy].shots` | `1024` | Measurement shots (noisy backend) |

## Models

**MLP** (`mlp`): Linear(d_f, 128) → ReLU → Dropout(0.3) → Linear(128, 2)

**QK-Ideal** (`qk_ideal`): Dual-path with `StatevectorEstimator` (exact, backprop gradients)

**QK-Noisy** (`qk_noisy`): Dual-path with `AerSimulator` + IBM Heron r2 noise model (SPSA gradients, 1024 shots)

All heads use EWC on all trainable parameters including quantum circuit weights.

## Datasets

**Split CIFAR-10**: 5 binary tasks — (airplane/automobile), (bird/cat), (deer/dog), (frog/horse), (ship/truck). ~10,000 training and ~2,000 test images per task.

**Split CIFAR-100**: 10 binary tasks from 20 CIFAR-100 superclasses paired sequentially. ~2,500 training and ~500 test images per task.

## Metrics

- **AA**: Average Accuracy over all tasks after full training
- **AF**: Average Forgetting — peak minus final accuracy per task
- **BWT**: Backward Transfer — signed accuracy change vs. task-completion baseline

All reported as mean ± std over 5 seeds.

## Citation

```bibtex
@article{MartinPerez2026qtcl,
  author  = {Martín-Pérez, D. and Rodríguez-Díaz, F. and Gutiérrez-Avilés, D. and Troncoso, A. and Martínez-Álvarez, F.},
  title   = {Quantum Transfer-Continual Learning: Mitigating Catastrophic Forgetting with Hybrid Quantum-Classical Circuits},
  year    = {2026}
}
```
