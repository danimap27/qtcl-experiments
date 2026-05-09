"""
Aggregate-level summary report for QTCL experiments.

Reads all runs.csv files from results/ and produces:
    - Aggregated LaTeX tables (mean ± std over seeds) for each study
    - Cross-method comparison plots (AA, AF, BWT vs head/backbone/dataset)
    - Ablation heatmaps (qubits × depth)
    - Energy efficiency comparison (kWh, CO2 vs accuracy)
    - Per-method radar charts (precision/recall/f1/AUC)
    - Lambda-sensitivity curves
    - Per-task forgetting profiles aggregated over seeds

Outputs go to:
    paper/tables/   — LaTeX tables
    paper/figures/  — PDF/PNG plots
"""

import os
import glob
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="whitegrid", context="paper")
except ImportError:
    HAS_SEABORN = False


# ── Loading ───────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> pd.DataFrame:
    """Concatenate all runs.csv files under results_dir."""
    paths = glob.glob(os.path.join(results_dir, "**", "runs.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No runs.csv found under {results_dir}")
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["run_id"])
    if "study" not in df.columns:
        df["study"] = "main"
    df["study"] = df["study"].fillna("main")
    return df


def fmt_pm(mean: float, std: float, pct: bool = True, dec: int = 2) -> str:
    if pd.isna(mean):
        return "---"
    if pct:
        return f"${mean*100:.{dec}f} \\pm {std*100:.{dec}f}$"
    return f"${mean:.{dec}f} \\pm {std:.{dec}f}$"


def write_tex(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[TABLE] {path}")


def save_fig(fig, path: str, dpi: int = 200) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG]   {path}")


# ── Tables ────────────────────────────────────────────────────────────────────

def table_main(df: pd.DataFrame, tables_dir: str) -> None:
    """Main results table grouped by dataset × backbone × head."""
    main = df[df["study"] == "main"]
    if main.empty:
        return

    rows = []
    for (ds, bb, hd), g in main.groupby(["dataset", "backbone", "head"]):
        rows.append({
            "Dataset":  ds.replace("_", " "),
            "Backbone": bb,
            "Head":     hd,
            "AA":       fmt_pm(g["AA"].mean(),  g["AA"].std()),
            "AF":       fmt_pm(g["AF"].mean(),  g["AF"].std()),
            "BWT":      fmt_pm(g["BWT"].mean(), g["BWT"].std()),
            "F1":       fmt_pm(g.get("final_f1_mean", pd.Series([np.nan])).mean(),
                               g.get("final_f1_mean", pd.Series([np.nan])).std()),
            "Time(s)":  f"${g['train_time_s'].mean():.0f} \\pm {g['train_time_s'].std():.0f}$",
            "Energy(Wh)": f"${g['energy_kwh'].mean()*1000:.1f}$" if "energy_kwh" in g and not g["energy_kwh"].isna().all() else "---",
        })
    out = pd.DataFrame(rows)
    lines = [
        "\\begin{tabular}{lll" + "c" * 5 + "}",
        "\\toprule",
        "Dataset & Backbone & Head & AA $\\uparrow$ & AF $\\downarrow$ & BWT & F1 & Time(s) \\\\",
        "\\midrule",
    ]
    prev_ds = None
    for _, r in out.iterrows():
        if prev_ds and r["Dataset"] != prev_ds:
            lines.append("\\midrule")
        lines.append(f"{r['Dataset']} & {r['Backbone']} & {r['Head']} & "
                     f"{r['AA']} & {r['AF']} & {r['BWT']} & {r['F1']} & {r['Time(s)']} \\\\")
        prev_ds = r["Dataset"]
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "main_results.tex"), lines)


def table_ablation(df: pd.DataFrame, tables_dir: str) -> None:
    ab = df[df["study"] == "ablation"]
    if ab.empty:
        return
    for (ds, hd), g in ab.groupby(["dataset", "head"]):
        pivot = g.groupby(["n_qubits", "depth"])["AA"].agg(["mean", "std"]).unstack("depth")
        depths = sorted(set(g["depth"]))
        qubits = sorted(set(g["n_qubits"]))

        lines = [
            f"% Ablation: {ds} / {hd}",
            "\\begin{tabular}{l" + "c" * len(depths) + "}",
            "\\toprule",
            "Qubits & " + " & ".join(f"$L = {d}$" for d in depths) + " \\\\",
            "\\midrule",
        ]
        for q in qubits:
            cells = []
            for d in depths:
                m = pivot[("mean", d)].get(q, np.nan) if ("mean", d) in pivot.columns else np.nan
                s = pivot[("std",  d)].get(q, 0.0)    if ("std",  d) in pivot.columns else 0.0
                cells.append(fmt_pm(m, s) if not np.isnan(m) else "---")
            lines.append(f"{q} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write_tex(os.path.join(tables_dir, f"ablation_{ds}_{hd}.tex"), lines)


def table_lambda(df: pd.DataFrame, tables_dir: str) -> None:
    ls = df[df["study"] == "lambda_sensitivity"]
    if ls.empty:
        return
    lines = [
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Head & $\\lambda$ & AA $\\uparrow$ & AF $\\downarrow$ \\\\",
        "\\midrule",
    ]
    prev_hd = None
    for hd, g in ls.groupby("head"):
        if prev_hd:
            lines.append("\\midrule")
        prev_hd = hd
        for lam, sub in g.groupby("lambda_ewc"):
            lines.append(
                f"{hd} & {lam:.0f} & "
                f"{fmt_pm(sub['AA'].mean(), sub['AA'].std())} & "
                f"{fmt_pm(sub['AF'].mean(), sub['AF'].std())} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "lambda_sensitivity.tex"), lines)


def table_energy(df: pd.DataFrame, tables_dir: str) -> None:
    if "energy_kwh" not in df.columns or df["energy_kwh"].isna().all():
        return
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Head & AA $\\uparrow$ & Energy (Wh) & CO$_2$ (g) \\\\",
        "\\midrule",
    ]
    main = df[df["study"] == "main"]
    for (ds, hd), g in main.groupby(["dataset", "head"]):
        e = g["energy_kwh"].mean() * 1000
        c = g["co2_kg"].mean() * 1000 if "co2_kg" in g else float("nan")
        lines.append(
            f"{ds.replace('_',' ')} & {hd} & "
            f"{fmt_pm(g['AA'].mean(), g['AA'].std())} & "
            f"${e:.1f}$ & ${c:.2f}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "energy.tex"), lines)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_method_comparison(df: pd.DataFrame, figures_dir: str) -> None:
    main = df[df["study"] == "main"]
    if main.empty:
        return
    metrics = ["AA", "AF", "BWT"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, m in zip(axes, metrics):
        if HAS_SEABORN:
            sns.barplot(data=main, x="head", y=m, hue="backbone", ax=ax,
                        errorbar="sd", palette="Set2")
        else:
            for bb, g in main.groupby("backbone"):
                ax.bar(g["head"], g[m], label=bb)
        ax.set_title(f"{m} (mean ± std)")
        ax.set_ylabel(m + " (%)" if m != "BWT" else "BWT (%)")
        if m in ("AA", "AF"):
            ax.set_ylim(0, 1.05)
        ax.legend(title="Backbone", fontsize=8)
        ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "method_comparison.pdf"))


def fig_ablation_heatmaps(df: pd.DataFrame, figures_dir: str) -> None:
    ab = df[df["study"] == "ablation"]
    if ab.empty:
        return
    for (ds, hd), g in ab.groupby(["dataset", "head"]):
        pivot = g.groupby(["n_qubits", "depth"])["AA"].mean().unstack("depth")
        fig, ax = plt.subplots(figsize=(5, 4))
        if HAS_SEABORN:
            sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="viridis", ax=ax,
                        cbar_kws={"label": "AA (%)"})
        else:
            im = ax.imshow(pivot.values * 100, cmap="viridis")
            plt.colorbar(im, ax=ax, label="AA (%)")
        ax.set_xlabel("circuit depth")
        ax.set_ylabel("number of qubits")
        ax.set_title(f"{hd} on {ds}")
        save_fig(fig, os.path.join(figures_dir, f"ablation_{ds}_{hd}.pdf"))


def fig_lambda_sensitivity(df: pd.DataFrame, figures_dir: str) -> None:
    ls = df[df["study"] == "lambda_sensitivity"]
    if ls.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for hd, g in ls.groupby("head"):
        agg = g.groupby("lambda_ewc")[["AA", "AF"]].agg(["mean", "std"])
        x = agg.index.values
        axes[0].errorbar(x, agg["AA"]["mean"], yerr=agg["AA"]["std"], label=hd, marker="o")
        axes[1].errorbar(x, agg["AF"]["mean"], yerr=agg["AF"]["std"], label=hd, marker="o")
    for ax, t in zip(axes, ["AA", "AF"]):
        ax.set_xscale("symlog")
        ax.set_xlabel(r"EWC $\lambda$")
        ax.set_ylabel(t)
        ax.set_title(f"{t} vs $\\lambda$")
        ax.legend()
        ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "lambda_sensitivity.pdf"))


def fig_energy_vs_accuracy(df: pd.DataFrame, figures_dir: str) -> None:
    if "energy_kwh" not in df.columns or df["energy_kwh"].isna().all():
        return
    main = df[df["study"] == "main"]
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"mlp": "o", "qk_ideal": "s", "qk_noisy": "^"}
    for hd, g in main.groupby("head"):
        ax.scatter(g["energy_kwh"] * 1000, g["AA"] * 100,
                   label=hd, marker=markers.get(hd, "x"), s=80, alpha=0.7)
    ax.set_xlabel("energy (Wh)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Energy efficiency: AA vs energy consumed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "energy_vs_accuracy.pdf"))


def fig_per_task_accuracy(df: pd.DataFrame, figures_dir: str) -> None:
    """Mean per-task final accuracy across methods."""
    main = df[df["study"] == "main"]
    if main.empty:
        return
    task_cols = [c for c in main.columns if c.startswith("task_") and c.endswith("_final_acc")]
    if not task_cols:
        return
    fig, axes = plt.subplots(1, len(main["dataset"].unique()), figsize=(6 * len(main["dataset"].unique()), 4.5),
                             squeeze=False)
    for ax, (ds, gds) in zip(axes[0], main.groupby("dataset")):
        for hd, g in gds.groupby("head"):
            ys = [g[c].mean() for c in task_cols]
            ax.plot(range(1, len(task_cols) + 1), ys, "o-", label=hd)
        ax.set_xlabel("task index")
        ax.set_ylabel("final accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-task final accuracy — {ds}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "per_task_accuracy.pdf"))


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all(results_dir: str, paper_dir: str) -> None:
    df = load_results(results_dir)
    print(f"[INFO] Loaded {len(df)} unique runs across {df['study'].nunique()} studies.")

    tables_dir  = os.path.join(paper_dir, "tables")
    figures_dir = os.path.join(paper_dir, "figures")

    table_main(df, tables_dir)
    table_ablation(df, tables_dir)
    table_lambda(df, tables_dir)
    table_energy(df, tables_dir)

    fig_method_comparison(df, figures_dir)
    fig_ablation_heatmaps(df, figures_dir)
    fig_lambda_sensitivity(df, figures_dir)
    fig_energy_vs_accuracy(df, figures_dir)
    fig_per_task_accuracy(df, figures_dir)

    print("\n[OK] Summary generation complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--paper-dir",   default="../paper")
    args = p.parse_args()
    generate_all(args.results_dir, args.paper_dir)
