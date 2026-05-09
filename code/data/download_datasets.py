#!/usr/bin/env python3
"""
Pre-download MNIST and CIFAR-10 datasets to a local cache.

Run once before submitting SLURM jobs to avoid concurrent download races
across array tasks. After this script finishes, the SLURM jobs use
download=False and load from disk only.

Usage:
    python data/download_datasets.py                    # default root: ./data/datasets
    python data/download_datasets.py --root /path/to/datasets
"""

import argparse
import os
import sys

from torchvision import datasets


def download_all(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    print(f"[INFO] Target directory: {os.path.abspath(root)}\n")

    print("[1/4] Downloading MNIST train split...")
    datasets.MNIST(root=root, train=True,  download=True)

    print("[2/4] Downloading MNIST test split...")
    datasets.MNIST(root=root, train=False, download=True)

    print("[3/4] Downloading CIFAR-10 train split...")
    datasets.CIFAR10(root=root, train=True,  download=True)

    print("[4/4] Downloading CIFAR-10 test split...")
    datasets.CIFAR10(root=root, train=False, download=True)

    print("\n[OK] All datasets ready at:", os.path.abspath(root))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data/datasets",
                   help="Cache directory for the datasets")
    args = p.parse_args()
    try:
        download_all(args.root)
        return 0
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
