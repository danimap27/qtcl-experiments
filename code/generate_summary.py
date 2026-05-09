"""
Aggregate-level summary report for QTCL experiments.

Reads all runs.csv files from results/ and produces:
    - LaTeX tables (mean ± std + 95% bootstrap CI) per study
    - Statistical comparison tables (Wilcoxon, Cohen's d, Friedman, p-values)
    - Cross-method comparison plots (AA, AF, BWT, F1, AUC)
    - Box plots / violin plots of metric distributions
    - Ablation heatmaps (qubits × depth) for AA, AF, time, energy
    - Lambda-sensitivity curves with confidence bands
    - Energy efficiency scatter (kWh/CO2 vs accuracy)
    - Per-task forgetting profiles aggregated over seeds
    - Statistical-significance heatmap (p-value matrix)

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

import statistical_tests as st


# ── Loading ───────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> pd.DataFrame:
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


def fmt_pm(mean, std, pct: bool = True, dec: int = 2) -> str:
    if pd.isna(mean):
        return "---"
    if pct:
        return f"${mean*100:.{dec}f} \\pm {std*100:.{dec}f}$"
    return f"${mean:.{dec}f} \\pm {std:.{dec}f}$"


def fmt_ci(values: np.ndarray, pct: bool = True, dec: int = 2) -> str:
    if len(values) == 0 or np.all(pd.isna(values)):
        return "---"
    m, lo, hi = st.bootstrap_ci(values)
    s = 100 if pct else 1
    return f"${m*s:.{dec}f}_{{[{lo*s:.{dec}f},\\,{hi*s:.{dec}f}]}}$"


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


def safe_mean(g, col):
    return g[col].mean() if col in g.columns and not g[col].isna().all() else np.nan


def safe_std(g, col):
    return g[col].std() if col in g.columns and not g[col].isna().all() else 0.0


# ── Tables ────────────────────────────────────────────────────────────────────

def table_main(df: pd.DataFrame, tables_dir: str) -> None:
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
            "F1":       fmt_pm(safe_mean(g, "final_f1_mean"), safe_std(g, "final_f1_mean")),
            "AUC":      fmt_pm(safe_mean(g, "final_roc_auc_mean"), safe_std(g, "final_roc_auc_mean")),
            "Time(s)":  f"${g['train_time_s'].mean():.0f} \\pm {g['train_time_s'].std():.0f}$",
        })
    out = pd.DataFrame(rows)

    lines = [
        "\\begin{tabular}{lll" + "c" * 6 + "}",
        "\\toprule",
        "Dataset & Backbone & Head & AA $\\uparrow$ & AF $\\downarrow$ & BWT & F1 & AUC & Time(s) \\\\",
        "\\midrule",
    ]
    prev_ds = None
    for _, r in out.iterrows():
        if prev_ds and r["Dataset"] != prev_ds:
            lines.append("\\midrule")
        lines.append(f"{r['Dataset']} & {r['Backbone']} & {r['Head']} & "
                     f"{r['AA']} & {r['AF']} & {r['BWT']} & {r['F1']} & {r['AUC']} & {r['Time(s)']} \\\\")
        prev_ds = r["Dataset"]
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "main_results.tex"), lines)


def table_ablation(df: pd.DataFrame, tables_dir: str) -> None:
    ab = df[df["study"] == "ablation"]
    if ab.empty:
        return
    for (ds, hd), g in ab.groupby(["dataset", "head"]):
        depths = sorted(g["depth"].dropna().unique())
        qubits = sorted(g["n_qubits"].dropna().unique())
        pivot_m = g.groupby(["n_qubits", "depth"])["AA"].mean()
        pivot_s = g.groupby(["n_qubits", "depth"])["AA"].std()

        lines = [
            f"% Ablation AA (\\%): {ds} / {hd}",
            "\\begin{tabular}{l" + "c" * len(depths) + "}",
            "\\toprule",
            "$n_q$ & " + " & ".join(f"$L = {int(d)}$" for d in depths) + " \\\\",
            "\\midrule",
        ]
        for q in qubits:
            cells = []
            for d in depths:
                key = (q, d)
                m = pivot_m.get(key, np.nan)
                s = pivot_s.get(key, 0.0)
                cells.append(fmt_pm(m, s) if not np.isnan(m) else "---")
            lines.append(f"{int(q)} & " + " & ".join(cells) + " \\\\")
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
    main = df[df["study"] == "main"]
    if main.empty:
        return
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Head & AA $\\uparrow$ & Energy (Wh) & CO$_2$ (g) \\\\",
        "\\midrule",
    ]
    for (ds, hd), g in main.groupby(["dataset", "head"]):
        e = safe_mean(g, "energy_kwh") * 1000 if not pd.isna(safe_mean(g, "energy_kwh")) else float("nan")
        c = safe_mean(g, "co2_kg") * 1000 if not pd.isna(safe_mean(g, "co2_kg")) else float("nan")
        e_str = f"${e:.2f}$" if not np.isnan(e) else "---"
        c_str = f"${c:.3f}$" if not np.isnan(c) else "---"
        lines.append(
            f"{ds.replace('_',' ')} & {hd} & "
            f"{fmt_pm(g['AA'].mean(), g['AA'].std())} & {e_str} & {c_str} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "energy.tex"), lines)


def table_statistical_tests(df: pd.DataFrame, tables_dir: str) -> None:
    """Pairwise Wilcoxon comparison between heads, per dataset/backbone."""
    main = df[df["study"] == "main"]
    if main.empty or main["head"].nunique() < 2:
        return

    lines = [
        "\\begin{tabular}{llllrrrrl}",
        "\\toprule",
        "Dataset & Backbone & Head A & Head B & $\\bar{A}$ & $\\bar{B}$ & $d$ & $p_{\\text{adj}}$ & sig. \\\\",
        "\\midrule",
    ]
    prev = None
    for (ds, bb), g in main.groupby(["dataset", "backbone"]):
        report = st.method_comparison_report(g, "head", "AA", paired=True)
        if report.empty:
            continue
        if prev and prev != (ds, bb):
            lines.append("\\midrule")
        prev = (ds, bb)
        for _, r in report.iterrows():
            sig = "yes" if r.get("significant_0.05", False) else "no"
            p_adj = r.get("p_value_adj", r["p_value"])
            lines.append(
                f"{ds.replace('_',' ')} & {bb} & {r['method_a']} & {r['method_b']} & "
                f"{r['mean_a']*100:.2f} & {r['mean_b']*100:.2f} & "
                f"{r['cohens_d']:.2f} & {p_adj:.4f} & {sig} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "statistical_tests.tex"), lines)


def table_friedman(df: pd.DataFrame, tables_dir: str) -> None:
    """Friedman test across heads per (dataset, backbone)."""
    main = df[df["study"] == "main"]
    if main.empty or main["head"].nunique() < 3:
        return
    lines = [
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Dataset & Backbone & Friedman $\\chi^2$ & $p$ & sig. \\\\",
        "\\midrule",
    ]
    for (ds, bb), g in main.groupby(["dataset", "backbone"]):
        samples = []
        for hd in sorted(g["head"].unique()):
            samples.append(g[g["head"] == hd].sort_values("seed")["AA"].values)
        n_min = min(len(s) for s in samples) if samples else 0
        if n_min < 2:
            continue
        samples = [s[:n_min] for s in samples]
        res = st.friedman(*samples)
        sig = "yes" if res["p_value"] < 0.05 else "no"
        lines.append(
            f"{ds.replace('_',' ')} & {bb} & "
            f"{res['statistic']:.3f} & {res['p_value']:.4f} & {sig} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "friedman.tex"), lines)


def table_bootstrap_ci(df: pd.DataFrame, tables_dir: str) -> None:
    """95% bootstrap CIs for AA/AF/BWT per main combination."""
    main = df[df["study"] == "main"]
    if main.empty:
        return
    lines = [
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "Dataset & Backbone & Head & AA $[95\\% \\text{CI}]$ & AF $[95\\% \\text{CI}]$ & BWT $[95\\% \\text{CI}]$ \\\\",
        "\\midrule",
    ]
    prev_ds = None
    for (ds, bb, hd), g in main.groupby(["dataset", "backbone", "head"]):
        if prev_ds and prev_ds != ds:
            lines.append("\\midrule")
        prev_ds = ds
        lines.append(
            f"{ds.replace('_',' ')} & {bb} & {hd} & "
            f"{fmt_ci(g['AA'].values)} & "
            f"{fmt_ci(g['AF'].values)} & "
            f"{fmt_ci(g['BWT'].values)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "bootstrap_ci.tex"), lines)


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
        ax.set_title(f"{m}")
        ax.set_ylabel(m + " (rate)")
        ax.legend(title="Backbone", fontsize=8)
        ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "method_comparison.pdf"))


def fig_boxplots(df: pd.DataFrame, figures_dir: str) -> None:
    main = df[df["study"] == "main"]
    if main.empty:
        return
    metrics = ["AA", "AF", "BWT"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, m in zip(axes, metrics):
        if HAS_SEABORN:
            sns.boxplot(data=main, x="head", y=m, hue="dataset", ax=ax, palette="Set2")
            sns.stripplot(data=main, x="head", y=m, hue="dataset", ax=ax,
                          dodge=True, alpha=0.5, palette="dark", legend=False)
        else:
            ax.boxplot([main[main["head"] == h][m].values for h in main["head"].unique()])
            ax.set_xticklabels(main["head"].unique())
        ax.set_title(f"{m} distribution across seeds")
        ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "boxplots.pdf"))


def fig_ablation_heatmaps(df: pd.DataFrame, figures_dir: str) -> None:
    ab = df[df["study"] == "ablation"]
    if ab.empty:
        return
    for (ds, hd), g in ab.groupby(["dataset", "head"]):
        for metric, label, cmap in [("AA", "AA (%)", "viridis"),
                                    ("AF", "AF (%)", "magma_r"),
                                    ("train_time_s", "Time (s)", "rocket")]:
            if metric not in g.columns:
                continue
            pivot = g.groupby(["n_qubits", "depth"])[metric].mean().unstack("depth")
            pivot = pivot.sort_index(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            data = pivot.values * (100 if metric in ("AA", "AF") else 1)
            if HAS_SEABORN:
                sns.heatmap(data, annot=True, fmt=".1f", cmap=cmap, ax=ax,
                            xticklabels=pivot.columns.astype(int),
                            yticklabels=pivot.index.astype(int),
                            cbar_kws={"label": label})
            else:
                im = ax.imshow(data, cmap=cmap)
                plt.colorbar(im, ax=ax, label=label)
            ax.set_xlabel("circuit depth")
            ax.set_ylabel("number of qubits")
            ax.set_title(f"{hd} — {ds} — {label}")
            save_fig(fig, os.path.join(figures_dir, f"ablation_{ds}_{hd}_{metric}.pdf"))


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
        ax.set_xscale("symlog", linthresh=1.0)
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
    if main.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"mlp": "o", "qk_ideal": "s", "qk_noisy": "^"}
    for hd, g in main.groupby("head"):
        ax.scatter(g["energy_kwh"] * 1000, g["AA"] * 100,
                   label=hd, marker=markers.get(hd, "x"), s=80, alpha=0.7)
    ax.set_xlabel("energy (Wh)")
    ax.set_ylabel("AA (%)")
    ax.set_title("Energy efficiency: AA vs energy consumed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "energy_vs_accuracy.pdf"))


def fig_per_task_accuracy(df: pd.DataFrame, figures_dir: str) -> None:
    main = df[df["study"] == "main"]
    if main.empty:
        return
    task_cols = [c for c in main.columns if c.startswith("task_") and c.endswith("_final_acc")]
    if not task_cols:
        return
    datasets = main["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        gds = main[main["dataset"] == ds]
        for hd, g in gds.groupby("head"):
            ys = [g[c].mean() for c in task_cols]
            errs = [g[c].std() for c in task_cols]
            ax.errorbar(range(1, len(task_cols) + 1), ys, yerr=errs,
                        marker="o", label=hd, capsize=3)
        ax.set_xlabel("task index")
        ax.set_ylabel("final accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-task final accuracy — {ds}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "per_task_accuracy.pdf"))


def table_cl_methods(df: pd.DataFrame, tables_dir: str) -> None:
    cm = df[df["study"] == "cl_methods"]
    if cm.empty:
        return
    lines = [
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "Dataset & Head & Method & AA $\\uparrow$ & AF $\\downarrow$ & BWT \\\\",
        "\\midrule",
    ]
    prev = None
    for (ds, hd), gh in cm.groupby(["dataset", "head"]):
        if prev and prev != (ds, hd):
            lines.append("\\midrule")
        prev = (ds, hd)
        for method, g in gh.groupby("cl_method"):
            lines.append(
                f"{ds.replace('_',' ')} & {hd} & {method} & "
                f"{fmt_pm(g['AA'].mean(), g['AA'].std())} & "
                f"{fmt_pm(g['AF'].mean(), g['AF'].std())} & "
                f"{fmt_pm(g['BWT'].mean(), g['BWT'].std())} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "cl_methods.tex"), lines)


def table_arch_variants(df: pd.DataFrame, tables_dir: str) -> None:
    av = df[df["study"] == "arch_variants"]
    if av.empty:
        return
    lines = [
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "Dataset & Backbone & Architecture & AA $\\uparrow$ & AF $\\downarrow$ & F1 \\\\",
        "\\midrule",
    ]
    prev = None
    for (ds, bb), gb in av.groupby(["dataset", "backbone"]):
        if prev and prev != (ds, bb):
            lines.append("\\midrule")
        prev = (ds, bb)
        for hd, g in gb.groupby("head"):
            lines.append(
                f"{ds.replace('_',' ')} & {bb} & {hd} & "
                f"{fmt_pm(g['AA'].mean(), g['AA'].std())} & "
                f"{fmt_pm(g['AF'].mean(), g['AF'].std())} & "
                f"{fmt_pm(safe_mean(g, 'final_f1_mean'), safe_std(g, 'final_f1_mean'))} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "arch_variants.tex"), lines)


def fig_cl_methods_bars(df: pd.DataFrame, figures_dir: str) -> None:
    cm = df[df["study"] == "cl_methods"]
    if cm.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, m in zip(axes, ["AA", "AF", "BWT"]):
        if HAS_SEABORN:
            sns.barplot(data=cm, x="cl_method", y=m, hue="head", ax=ax,
                        errorbar="sd", palette="Set1")
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "cl_methods_comparison.pdf"))


def fig_arch_variants_bars(df: pd.DataFrame, figures_dir: str) -> None:
    av = df[df["study"] == "arch_variants"]
    if av.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, m in zip(axes, ["AA", "AF"]):
        if HAS_SEABORN:
            sns.barplot(data=av, x="head", y=m, ax=ax, errorbar="sd", palette="viridis")
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "arch_variants_comparison.pdf"))


def fig_pvalue_heatmap(df: pd.DataFrame, figures_dir: str) -> None:
    """Holm-corrected p-value heatmap between heads (main study)."""
    main = df[df["study"] == "main"]
    if main.empty or main["head"].nunique() < 2:
        return
    pmat = st.pairwise_pvalue_matrix(main, "head", "AA", paired=True, correction="holm")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if HAS_SEABORN:
        sns.heatmap(pmat, annot=True, fmt=".3f", cmap="coolwarm", center=0.05,
                    vmin=0, vmax=1, ax=ax, cbar_kws={"label": "p (Holm-adj.)"})
    else:
        im = ax.imshow(pmat.values, cmap="coolwarm")
        plt.colorbar(im, ax=ax)
    ax.set_title("Pairwise Wilcoxon (AA, Holm-adj.)")
    save_fig(fig, os.path.join(figures_dir, "pvalue_heatmap.pdf"))


def fig_qubit_scaling(df: pd.DataFrame, figures_dir: str) -> None:
    """AA, AF, train time vs n_qubits at fixed depth."""
    ab = df[df["study"] == "ablation"]
    if ab.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics = [("AA", "AA"), ("AF", "AF"), ("train_time_s", "Time (s)")]
    for ax, (col, label) in zip(axes, metrics):
        if col not in ab.columns:
            continue
        for (ds, hd), g in ab.groupby(["dataset", "head"]):
            agg = g.groupby("n_qubits")[col].agg(["mean", "std"])
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                        marker="o", label=f"{ds}/{hd}", capsize=3)
        ax.set_xlabel("number of qubits")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs qubits")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "qubit_scaling.pdf"))


def fig_depth_scaling(df: pd.DataFrame, figures_dir: str) -> None:
    """AA vs depth at fixed n_qubits."""
    ab = df[df["study"] == "ablation"]
    if ab.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for (ds, hd), g in ab.groupby(["dataset", "head"]):
        agg = g.groupby("depth")["AA"].agg(["mean", "std"])
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                    marker="o", label=f"{ds}/{hd}", capsize=3)
    ax.set_xlabel("circuit depth")
    ax.set_ylabel("AA")
    ax.set_title("AA vs circuit depth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(figures_dir, "depth_scaling.pdf"))


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all(results_dir: str, paper_dir: str) -> None:
    df = load_results(results_dir)
    print(f"[INFO] {len(df)} unique runs loaded across {df['study'].nunique()} studies.")
    print(f"[INFO] Studies: {sorted(df['study'].unique())}")

    tables_dir  = os.path.join(paper_dir, "tables")
    figures_dir = os.path.join(paper_dir, "figures")

    # Tables
    table_main(df, tables_dir)
    table_ablation(df, tables_dir)
    table_lambda(df, tables_dir)
    table_energy(df, tables_dir)
    table_bootstrap_ci(df, tables_dir)
    table_statistical_tests(df, tables_dir)
    table_friedman(df, tables_dir)
    table_cl_methods(df, tables_dir)
    table_arch_variants(df, tables_dir)

    # Figures
    fig_method_comparison(df, figures_dir)
    fig_boxplots(df, figures_dir)
    fig_ablation_heatmaps(df, figures_dir)
    fig_qubit_scaling(df, figures_dir)
    fig_depth_scaling(df, figures_dir)
    fig_lambda_sensitivity(df, figures_dir)
    fig_energy_vs_accuracy(df, figures_dir)
    fig_per_task_accuracy(df, figures_dir)
    fig_pvalue_heatmap(df, figures_dir)
    fig_cl_methods_bars(df, figures_dir)
    fig_arch_variants_bars(df, figures_dir)

    print("\n[OK] Summary generation complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--paper-dir",   default="../paper")
    args = p.parse_args()
    generate_all(args.results_dir, args.paper_dir)
