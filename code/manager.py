#!/usr/bin/env python3
"""
manager.py — QTCL Experiment Control Center & SLURM Monitor

Interactive menu to:
1. Refresh command lists from config.yaml.
2. Launch experiments in phases (Classical, Ideal, Noisy, Studies).
3. Monitor live progress by scanning results/ folders.
4. Generate LaTeX tables from aggregated CSV results.

Usage:
    python manager.py
"""

import os
import sys
import subprocess
import glob
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG      = "config.yaml"
RESULTS_DIR = "./results"

COMMAND_FILES = {
    "1": ("cmds_1_classical.txt", "Phase 1: Classical Baseline (MLP)"),
    "2": ("cmds_2_ideal.txt",     "Phase 2: Quantum Ideal (QK-Ideal)"),
    "3": ("cmds_3_noisy.txt",     "Phase 3: Quantum Noisy (QK-Noisy)"),
    "4": ("cmds_4_studies.txt",   "Phase 4: Studies (Ablation, Lambda, Scalability, Noise)"),
}

# 2 datasets × 3 backbones × 3 heads × 5 seeds = 90 main
# ablation: 2 heads × 15 configs × 2 seeds = 60
# lambda:   3 heads × 7 lambdas × 1 seed  = 21
# scale:    2 datasets × 1 backbone × 3 heads × 2 seeds = 12
# noise decomp: 1 dataset × 1 backbone × 3 channels × 1 seed = 3
EXPECTED_RUNS = 186


# ── Helpers ───────────────────────────────────────────────────────────────────

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print("=" * 70)
    print("        QTCL EXPERIMENT CONTROL CENTER")
    print("=" * 70)


def run_command(cmd: str, capture: bool = False):
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=True)
            return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if hasattr(e, "stderr") and e.stderr else ""
        print(f"\n[ERROR] Command failed: {cmd}")
        if stderr:
            print(f"  stderr: {stderr}")
        return None


def sbatch_available() -> bool:
    return subprocess.run("which sbatch", shell=True, capture_output=True).returncode == 0


def get_slurm_tasks(file_path: str) -> int:
    if not os.path.exists(file_path):
        return 0
    for enc in ("utf-8-sig", "utf-16", "latin1"):
        try:
            with open(file_path, encoding=enc) as f:
                return sum(1 for line in f if line.strip())
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 0


def progress_bar(done: int, total: int, width: int = 40) -> str:
    if total == 0:
        return f"[{'?' * width}] ?/?  ?%"
    pct = done / total
    filled = int(pct * width)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {done}/{total}  {pct*100:.1f}%"


def kbhit_nonblock() -> bool:
    if not HAS_TERMIOS:
        return False
    import select
    return select.select([sys.stdin], [], [], 0)[0] != []


# ── Results scanning ──────────────────────────────────────────────────────────

def scan_progress() -> Tuple[int, Dict[str, int], Optional[object]]:
    """
    Scan all runs.csv files in results/.
    Returns (total_completed, counts_by_head, dataframe_or_None).
    """
    csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "[0-9]*", "runs.csv")))
    if not csv_files:
        return 0, {}, None
    if HAS_PANDAS:
        try:
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["run_id"])
            counts = df["head"].value_counts().to_dict() if "head" in df.columns else {}
            return len(df), counts, df
        except Exception:
            pass
    return len(csv_files), {}, None


def collect_completed_ids() -> Set[str]:
    completed: Set[str] = set()
    for csv_path in glob.glob(os.path.join(RESULTS_DIR, "[0-9]*", "runs.csv")):
        try:
            if HAS_PANDAS:
                df = pd.read_csv(csv_path)
                if "run_id" in df.columns:
                    completed.update(df["run_id"].astype(str))
        except Exception:
            pass
    return completed


def parse_run_id_from_cmd(line: str) -> Optional[str]:
    """
    Reconstruct run_id from an exported command line.
    Main format: {dataset}_{backbone}_{head}_{seed}
    Ablation/lambda format varies — returned as-is based on flags.
    """
    parts = line.split()
    d = b = h = s = study = None
    for i, p in enumerate(parts):
        if p == "--dataset"  and i + 1 < len(parts): d = parts[i + 1]
        elif p == "--backbone" and i + 1 < len(parts): b = parts[i + 1]
        elif p == "--head"     and i + 1 < len(parts): h = parts[i + 1]
        elif p == "--seed"     and i + 1 < len(parts): s = parts[i + 1]
        elif p == "--study"    and i + 1 < len(parts): study = parts[i + 1]
    if d and b and h and s:
        if study in ("ablation", "lambda_sensitivity", "noise_decomposition", "scalability"):
            return None  # IDs for studies differ; skip
        return f"{d}_{b}_{h}_{s}"
    return None


def delete_run_results(run_ids: List[str]) -> None:
    if not HAS_PANDAS:
        print("  [WARN] pandas not available — cannot delete specific rows.")
        return
    ids_to_delete = set(str(r) for r in run_ids)
    for runs_csv in glob.glob(os.path.join(RESULTS_DIR, "[0-9]*", "runs.csv")):
        folder = str(Path(runs_csv).parent)
        try:
            df = pd.read_csv(runs_csv)
            if "run_id" not in df.columns or not df["run_id"].astype(str).isin(ids_to_delete).any():
                continue
            for csv_name in ("runs.csv", "predictions.csv", "training_log.csv"):
                p = os.path.join(folder, csv_name)
                if not os.path.exists(p):
                    continue
                sub = pd.read_csv(p)
                if "run_id" in sub.columns:
                    sub[~sub["run_id"].astype(str).isin(ids_to_delete)].to_csv(p, index=False)
            remaining = pd.read_csv(runs_csv)
            if remaining.empty:
                import shutil
                shutil.rmtree(folder)
                print(f"  Removed empty folder: {folder}")
        except Exception as e:
            print(f"  [WARN] {runs_csv}: {e}")


# ── Menu actions ──────────────────────────────────────────────────────────────

def refresh_commands():
    """Regenerate .txt command files from config.yaml."""
    print("\n[INFO] Refreshing command lists from config.yaml...\n")
    cmds = [
        # Phase 1
        (f"python runner.py --config {CONFIG} --head mlp "
         f"--dry-run --export-commands > cmds_1_classical.txt",
         "Phase 1: Classical"),
        # Phase 2
        (f"python runner.py --config {CONFIG} --head qk_ideal "
         f"--dry-run --export-commands > cmds_2_ideal.txt",
         "Phase 2: QK-Ideal"),
        # Phase 3
        (f"python runner.py --config {CONFIG} --head qk_noisy "
         f"--dry-run --export-commands > cmds_3_noisy.txt",
         "Phase 3: QK-Noisy"),
        # Phase 4: studies
        (f"python runner.py --config {CONFIG} --study ablation "
         f"--dry-run --export-commands > cmds_4_studies.txt",
         "Phase 4: Ablation"),
        (f"python runner.py --config {CONFIG} --study lambda_sensitivity "
         f"--dry-run --export-commands >> cmds_4_studies.txt",
         "Phase 4: Lambda"),
        (f"python runner.py --config {CONFIG} --study scalability "
         f"--dry-run --export-commands >> cmds_4_studies.txt",
         "Phase 4: Scalability"),
        (f"python runner.py --config {CONFIG} --study noise_decomposition "
         f"--dry-run --export-commands >> cmds_4_studies.txt",
         "Phase 4: Noise Decomp"),
    ]
    for cmd, label in cmds:
        print(f"  {label}...", end=" ", flush=True)
        print("[OK]" if run_command(cmd) else "[FAILED]")

    print("\n[VERIFY] Task counts:")
    for key, (path, name) in COMMAND_FILES.items():
        n = get_slurm_tasks(path)
        print(f"  [{key}] {name}: {n} tasks")

    input("\nEnter to return to menu...")


def check_completed(phase_key: Optional[str] = None, view_only: bool = False) -> Optional[str]:
    """
    Show completed / pending runs.
    view_only=True: show only (option C).
    view_only=False: ask what to do with already-completed runs.
    Returns 'skip_all', 'overwrite_all', or None (cancel).
    """
    completed_ids = collect_completed_ids()
    files = ([COMMAND_FILES[phase_key]] if phase_key and phase_key in COMMAND_FILES
             else list(COMMAND_FILES.values()))

    all_ids: List[str] = []
    for file_path, _ in files:
        if not os.path.exists(file_path):
            continue
        for enc in ("utf-8-sig", "utf-16", "latin1"):
            try:
                with open(file_path, encoding=enc) as f:
                    for line in f:
                        run_id = parse_run_id_from_cmd(line.strip())
                        if run_id:
                            all_ids.append(run_id)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

    if not all_ids:
        print("\n  No command files found. Run [R] first.")
        input("\nEnter to return...")
        return None

    done    = [r for r in all_ids if r in completed_ids]
    pending = [r for r in all_ids if r not in completed_ids]
    print(f"\n  Total: {len(all_ids)}  |  Done: {len(done)}  |  Pending: {len(pending)}")

    if view_only:
        if done:
            print(f"\n  Done ({len(done)}):")
            for r in done[:25]:
                print(f"    [x] {r}")
            if len(done) > 25:
                print(f"        ... and {len(done) - 25} more")
        if pending:
            print(f"\n  Pending ({len(pending)}):")
            for r in pending[:25]:
                print(f"    [ ] {r}")
            if len(pending) > 25:
                print(f"        ... and {len(pending) - 25} more")
        input("\nEnter to return...")
        return None

    if not done:
        return "skip_all"

    print(f"\n  {len(done)} run(s) already completed.")
    print(f"  [1] Keep results — run only {len(pending)} pending")
    print(f"  [2] Delete and re-run all {len(all_ids)} runs from scratch")
    print(f"  [C] Cancel")
    while True:
        ch = input("  Option: ").strip().upper()
        if ch == "1":
            return "skip_all"
        if ch == "2":
            print(f"\n  Deleting {len(done)} result(s)...")
            delete_run_results(done)
            print("  Done.")
            return "overwrite_all"
        if ch == "C":
            return None
        print("  Enter 1, 2, or C.")


def submit_phase(key: str, dependency_id: Optional[str] = None, overwrite: bool = False) -> Optional[str]:
    if key not in COMMAND_FILES:
        return None

    if not sbatch_available():
        print("\n[ERROR] sbatch not found. manager.py must run ON Hercules, not locally.")
        print("  Deploy with [D] or SSH to Hercules and run: python manager.py")
        return None

    file_path, name = COMMAND_FILES[key]
    n_tasks = get_slurm_tasks(file_path)
    if n_tasks == 0:
        print(f"\n[WARN] No tasks in {file_path}. Run [R] first.")
        return None

    dep_arg  = f"--dependency=afterok:{dependency_id}" if dependency_id else ""
    job_name = f"QTCL_{key}"

    # Build --export without trailing comma when EXTRA_ARGS is empty
    export_val = f"CMD_FILE={file_path}"
    if overwrite:
        export_val += ",EXTRA_ARGS=--overwrite"

    cmd = (
        f"sbatch --parsable --job-name='{job_name}' "
        f"--array=1-{n_tasks}%20 {dep_arg} "
        f"--export={export_val} slurm_generic.sh"
    )
    print(f"\n[SUBMIT] {name} ({n_tasks} tasks)...")
    job_id = run_command(cmd, capture=True)
    if job_id:
        print(f"[SUCCESS] Job ID: {job_id}")
    return job_id


def launch_full_pipeline(overwrite: bool = False):
    print("\n[PIPELINE] Submitting all phases with sequential dependencies...")
    j1 = submit_phase("1", overwrite=overwrite)
    j2 = submit_phase("2", j1, overwrite=overwrite)
    j3 = submit_phase("3", j2, overwrite=overwrite)
    j4 = submit_phase("4", j3, overwrite=overwrite)
    print(f"\n[OK] Submitted: {j1} → {j2} → {j3} → {j4}")
    input("\nEnter to return...")


def show_monitoring():
    """Live progress monitor. Refreshes every 2s. Press any key to exit."""
    old_settings = None
    if HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        while True:
            clear_screen()
            print_header()
            print()

            total, head_counts, df = scan_progress()
            print(f"  Overall  {progress_bar(total, EXPECTED_RUNS)}")
            print()

            # Per-head breakdown
            if head_counts:
                print("  Completed by head:")
                for head, count in sorted(head_counts.items()):
                    print(f"    {head:<20}: {count}")
                print()

            # Per-phase breakdown
            print("  Phase breakdown:")
            phase_heads = {
                "1": {"mlp"},
                "2": {"qk_ideal"},
                "3": {"qk_noisy"},
                "4": set(),  # studies: count from file
            }
            for key, (file_path, name) in COMMAND_FILES.items():
                n_total = get_slurm_tasks(file_path)
                n_done = 0
                if df is not None and HAS_PANDAS and "head" in df.columns:
                    allowed = phase_heads.get(key, set())
                    if allowed:
                        n_done = int(df[df["head"].isin(allowed)]["run_id"].nunique())
                    else:
                        # Fallback: count study runs
                        if "study" in df.columns:
                            n_done = int(df[df["study"].isin(
                                ["ablation", "lambda_sensitivity", "scalability", "noise_decomposition"]
                            )]["run_id"].nunique())
                pbar = progress_bar(n_done, n_total, width=22)
                print(f"    [{key}] {name:<44} {pbar}")
            print()

            # Summary stats if data available
            if df is not None and HAS_PANDAS and "AA" in df.columns:
                print("  Recent results (last 5):")
                recent = df.sort_values("run_id").tail(5)[["run_id", "AA", "AF", "BWT"]]
                for _, row in recent.iterrows():
                    aa  = f"{row['AA']*100:.1f}%" if pd.notna(row.get("AA")) else "?"
                    af  = f"{row['AF']*100:.1f}%" if pd.notna(row.get("AF")) else "?"
                    print(f"    {row['run_id']:<50} AA={aa} AF={af}")
                print()

            # SLURM queue
            squeue = run_command(
                "squeue -u $USER --format='%.10i %.9P %.30j %.8T %.10M' 2>/dev/null",
                capture=True,
            )
            if squeue:
                lines = squeue.splitlines()
                print(f"  Active SLURM jobs: {max(len(lines)-1, 0)}")
                for line in lines[:6]:
                    print(f"    {line}")
            else:
                print("  SLURM queue: not available (local mode?)")

            print()
            print("  " + "-" * 60)
            print("  [Press any key to return to menu]")

            if HAS_TERMIOS:
                if kbhit_nonblock():
                    sys.stdin.read(1)
                    break
            else:
                time.sleep(2)
                continue
            time.sleep(2)
    finally:
        if old_settings is not None and HAS_TERMIOS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print()


def deploy_to_hercules():
    """Rsync code to Hercules and print the command to run there."""
    print("\n[DEPLOY] Syncing code to Hercules...")
    hercules_user = input("  Hercules username [quantum-nas]: ").strip() or "quantum-nas"
    hercules_host = input("  Hercules host [hercules.cica.es]: ").strip() or "hercules.cica.es"
    remote_path   = input("  Remote path [~/QTCL/code]: ").strip() or "~/QTCL/code"

    local_path = os.path.dirname(os.path.abspath(__file__))
    rsync_cmd = (
        f"rsync -avz --exclude='results/' --exclude='__pycache__/' --exclude='*.pyc' "
        f"{local_path}/ {hercules_user}@{hercules_host}:{remote_path}/"
    )
    print(f"\n  Running: {rsync_cmd}")
    ok = run_command(rsync_cmd)
    if ok:
        print(f"\n[OK] Code deployed.")
        print(f"\n  SSH and run manager on Hercules:")
        print(f"    ssh {hercules_user}@{hercules_host}")
        print(f"    cd {remote_path}")
        print(f"    conda activate qtcl")
        print(f"    python manager.py")
    input("\nEnter to return...")


def generate_tables_action():
    print("\n[TABLES] Generating LaTeX tables from results/...")
    run_command("python generate_tables.py --results-dir ./results --tables-dir ./paper/tables")
    input("\nEnter to return...")


def show_summary():
    """Print a quick text summary of current results without entering monitor."""
    total, head_counts, df = scan_progress()
    print(f"\n  Total completed runs: {total} / {EXPECTED_RUNS}")
    if head_counts:
        print("  By head:")
        for head, count in sorted(head_counts.items()):
            print(f"    {head}: {count}")
    if df is not None and HAS_PANDAS and "AA" in df.columns:
        print("\n  Mean AA by head:")
        grp = df.groupby("head")["AA"].mean()
        for head, aa in grp.items():
            print(f"    {head:<20}: {aa*100:.2f}%")
    input("\nEnter to return...")


# ── Main menu ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs("logs",    exist_ok=True)
    os.makedirs("results", exist_ok=True)

    while True:
        clear_screen()
        print_header()

        total, _, _ = scan_progress()
        print(f"  Progress: {progress_bar(total, EXPECTED_RUNS, width=30)}")
        print()
        slurm_ok = sbatch_available()
        slurm_tag = "" if slurm_ok else "  [requires Hercules]"
        print("  [R] Refresh command files from config.yaml")
        print("  [D] Deploy code to Hercules (rsync + SSH instructions)")
        print("  ─────────────────────────────────────────")
        print(f"  [1] Submit Phase 1: Classical Baseline (MLP){slurm_tag}")
        print(f"  [2] Submit Phase 2: Quantum Ideal      (QK-Ideal){slurm_tag}")
        print(f"  [3] Submit Phase 3: Quantum Noisy      (QK-Noisy){slurm_tag}")
        print(f"  [4] Submit Phase 4: Studies            (Ablation / Lambda / Scalability / Noise){slurm_tag}")
        print(f"  [F] Submit FULL PIPELINE (1 → 2 → 3 → 4 with SLURM deps){slurm_tag}")
        print("  ─────────────────────────────────────────")
        print("  [M] Monitor live progress (refresh every 2s)")
        print("  [C] Check completed / pending runs")
        print("  [S] Quick summary")
        print("  [T] Generate LaTeX tables from results")
        print("  ─────────────────────────────────────────")
        print("  [X] Exit")
        print("-" * 70)

        choice = input("  Option: ").strip().upper()

        if choice == "R":
            refresh_commands()

        elif choice == "D":
            deploy_to_hercules()

        elif choice in ("1", "2", "3", "4"):
            mode = check_completed(phase_key=choice)
            if mode is not None:
                submit_phase(choice, overwrite=(mode == "overwrite_all"))
                input("\nEnter to return...")

        elif choice == "F":
            mode = check_completed()
            if mode is not None:
                launch_full_pipeline(overwrite=(mode == "overwrite_all"))

        elif choice == "M":
            show_monitoring()

        elif choice == "C":
            check_completed(view_only=True)

        elif choice == "S":
            show_summary()

        elif choice == "T":
            generate_tables_action()

        elif choice == "X":
            print("\nExiting. Good luck with the submission!\n")
            break

        else:
            print("\n  Invalid option.")
            time.sleep(1)


if __name__ == "__main__":
    main()
