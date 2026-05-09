"""
Generate LaTeX tables from experiment CSV results.

Reads runs.csv files from ./results/, aggregates over seeds (mean ± std),
and writes .tex table files to ./paper/tables/.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Optional


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Concatenate all runs.csv files found under results_dir."""
    dfs = []
    for csv_path in glob.glob(os.path.join(results_dir, "**", "runs.csv"), recursive=True):
        try:
            dfs.append(pd.read_csv(csv_path))
        except Exception as e:
            print(f"[WARNING] Could not read {csv_path}: {e}")
    if not dfs:
        raise FileNotFoundError(f"No runs.csv found under {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def fmt(mean: float, std: float, pct: bool = True, decimals: int = 2) -> str:
    """Format mean ± std as LaTeX string."""
    if pct:
        return f"${mean*100:.{decimals}f} \\pm {std*100:.{decimals}f}$"
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def table_main_results(df: pd.DataFrame, tables_dir: str) -> None:
    """
    Main results table: AA, AF, BWT per (dataset, backbone, head).
    Aggregated over seeds.
    """
    rows = []
    for (ds, bb, hd), grp in df[df["study"].isna() | (df["study"] == "main")].groupby(["dataset", "backbone", "head"]):
        rows.append({
            "Dataset": ds.replace("_", " "),
            "Backbone": bb,
            "Head": hd,
            "AA": fmt(grp["AA"].mean(), grp["AA"].std()),
            "AF": fmt(grp["AF"].mean(), grp["AF"].std()),
            "BWT": fmt(grp["BWT"].mean(), grp["BWT"].std()),
            "Time (s)": f"${grp['train_time_s'].mean():.0f} \\pm {grp['train_time_s'].std():.0f}$",
        })

    if not rows:
        print("[TABLE] No main results found.")
        return

    result_df = pd.DataFrame(rows)
    tex_lines = [
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "Dataset & Backbone & Head & AA (\\%) $\\uparrow$ & AF (\\%) $\\downarrow$ & BWT (\\%) \\\\",
        "\\midrule",
    ]
    prev_ds = None
    for _, row in result_df.iterrows():
        if row["Dataset"] != prev_ds:
            if prev_ds is not None:
                tex_lines.append("\\midrule")
            prev_ds = row["Dataset"]
        tex_lines.append(
            f"{row['Dataset']} & {row['Backbone']} & {row['Head']} & "
            f"{row['AA']} & {row['AF']} & {row['BWT']} \\\\"
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}"]

    _write_table(tables_dir, "main_results.tex", tex_lines)


def table_ablation(df: pd.DataFrame, tables_dir: str) -> None:
    """Ablation table: AA vs n_qubits × depth (averaged over seeds)."""
    ab = df[df.get("study", pd.Series()) == "ablation"] if "study" in df.columns else pd.DataFrame()
    if ab.empty:
        print("[TABLE] No ablation results found.")
        return

    for head in ab["head"].unique():
        sub = ab[ab["head"] == head]
        pivot_mean = sub.groupby(["n_qubits", "depth"])["AA"].mean().unstack("depth")
        pivot_std  = sub.groupby(["n_qubits", "depth"])["AA"].std().unstack("depth")

        depths = sorted(pivot_mean.columns)
        qubits = sorted(pivot_mean.index)

        tex_lines = [
            f"% Ablation: {head}",
            "\\begin{tabular}{l" + "c" * len(depths) + "}",
            "\\toprule",
            "Qubits & " + " & ".join(f"Depth {d}" for d in depths) + " \\\\",
            "\\midrule",
        ]
        for q in qubits:
            cells = []
            for d in depths:
                m = pivot_mean.loc[q, d] if (q in pivot_mean.index and d in pivot_mean.columns) else float("nan")
                s = pivot_std.loc[q, d]  if (q in pivot_std.index  and d in pivot_std.columns)  else 0.0
                cells.append(fmt(m, s) if not np.isnan(m) else "---")
            tex_lines.append(f"{q} & " + " & ".join(cells) + " \\\\")
        tex_lines += ["\\bottomrule", "\\end{tabular}"]

        _write_table(tables_dir, f"ablation_{head}.tex", tex_lines)


def table_lambda_sensitivity(df: pd.DataFrame, tables_dir: str) -> None:
    """Lambda sensitivity: AA vs lambda per head."""
    ls = df[df.get("study", pd.Series()) == "lambda_sensitivity"] if "study" in df.columns else pd.DataFrame()
    if ls.empty:
        print("[TABLE] No lambda_sensitivity results found.")
        return

    tex_lines = [
        "\\begin{tabular}{llc}",
        "\\toprule",
        "Head & $\\lambda$ & AA (\\%) $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for hd, grp in ls.groupby("head"):
        for lam, sub in grp.groupby("lambda_ewc"):
            tex_lines.append(f"{hd} & {lam:.0f} & {fmt(sub['AA'].mean(), sub['AA'].std())} \\\\")
        tex_lines.append("\\midrule")
    tex_lines = tex_lines[:-1]  # remove trailing midrule
    tex_lines += ["\\bottomrule", "\\end{tabular}"]

    _write_table(tables_dir, "lambda_sensitivity.tex", tex_lines)


def _write_table(tables_dir: str, filename: str, lines: list) -> None:
    os.makedirs(tables_dir, exist_ok=True)
    path = os.path.join(tables_dir, filename)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[TABLE] Written: {path}")


def generate_all_tables(results_dir: str, tables_dir: str) -> None:
    try:
        df = load_all_results(results_dir)
    except FileNotFoundError as e:
        print(f"[TABLE] {e}")
        return

    table_main_results(df, tables_dir)
    table_ablation(df, tables_dir)
    table_lambda_sensitivity(df, tables_dir)
    print("[TABLE] All tables generated.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--tables-dir",  default="./paper/tables")
    args = p.parse_args()
    generate_all_tables(args.results_dir, args.tables_dir)
