"""Generates central publication-grade visualization plots across all completed experiments."""

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.helpers import ensure_dir, load_json
from utils.logger import logger


def plot_pareto_frontier(runs_dir: Path, output_dir: Path):
    """Plot 1: Context Compression Ratio vs. Model Embedding Parameter Overhead vs Val BPC."""
    data = []
    for r_dir in sorted(runs_dir.iterdir()):
        if not r_dir.is_dir():
            continue
        m_path = r_dir / "metrics.json"
        lm_path = r_dir / "lm_eval.json"
        if m_path.is_file() and lm_path.is_file():
            metrics = load_json(m_path)
            lm_eval = load_json(lm_path)
            v_size = metrics.get("vocab_size", 0)
            if v_size < 1000:
                continue
            name = metrics.get("model_name", r_dir.name)
            cr = metrics.get("overall", {}).get("compression_ratio", 0.0)
            bpc = lm_eval.get("val_bits_per_character", 0.0)
            params_m = (v_size * 384) / 1e6  # Millions of params in embedding table
            data.append({
                "name": name,
                "vocab": v_size,
                "cr": cr,
                "bpc": bpc,
                "params_m": params_m,
            })

    if not data:
        return

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    # Sort by vocab size
    data.sort(key=lambda x: x["vocab"])

    vocabs = [d["vocab"] // 1000 for d in data]
    crs = [d["cr"] for d in data]
    bpcs = [d["bpc"] for d in data]
    names = [d["name"] for d in data]

    color1 = "#1f77b4"
    ax1.set_xlabel("Vocabulary Size (k tokens)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Overall Compression Ratio (Bytes/Token)", color=color1, fontsize=11, fontweight="bold")
    line1 = ax1.plot(vocabs, crs, color=color1, marker="o", linewidth=2.5, markersize=8, label="Compression (B/T)")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("Validation Bits Per Character (BPC, lower is better)", color=color2, fontsize=11, fontweight="bold")
    line2 = ax2.plot(vocabs, bpcs, color=color2, marker="s", linewidth=2.5, markersize=8, linestyle="--", label="Val BPC")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Annotate points
    for d in data:
        vk = d["vocab"] // 1000
        ax1.annotate(f"{d['cr']:.2f} B/T", (vk, d["cr"]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
        ax2.annotate(f"{d['bpc']:.3f} BPC", (vk, d["bpc"]), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9, color=color2, fontweight="bold")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", frameon=True)

    plt.title("Pareto Frontier: Vocabulary Scaling vs Compression vs Validation BPC", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    out_file = output_dir / "pareto_scaling_frontier.png"
    plt.savefig(out_file, dpi=300)
    plt.savefig(out_file.with_suffix(".svg"))
    plt.close()
    logger.info(f"📊 Saved Pareto scaling frontier plot to {out_file.resolve()}")


def plot_multi_model_lm_curves(runs_dir: Path, output_dir: Path):
    """Plot 2: Validation BPC vs Training Steps across all variants."""
    histories = {}
    for r_dir in sorted(runs_dir.iterdir()):
        if not r_dir.is_dir():
            continue
        lm_path = r_dir / "lm_eval.json"
        if lm_path.is_file():
            lm_eval = load_json(lm_path)
            if "loss_history" in lm_eval and lm_eval["loss_history"]:
                name = lm_eval.get("model_name", r_dir.name)
                # Clean name
                clean_name = name.split("_exp_")[-1] if "_exp_" in name else name
                histories[clean_name] = lm_eval["loss_history"]

    if not histories:
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=300)

    palette = sns.color_palette("tab10", len(histories))
    for (name, hist), col in zip(histories.items(), palette):
        steps = [h["step"] for h in hist]
        bpcs = [h["bpc"] for h in hist]
        plt.plot(steps, bpcs, marker="o", linewidth=2.2, label=name, color=col)

    plt.title("Downstream MiniGPT Validation BPC Convergence (RTX 3050 AMP + SDPA)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Training Step", fontsize=11)
    plt.ylabel("Validation Bits Per Character (BPC)", fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_file = output_dir / "multi_model_lm_bpc_curves.png"
    plt.savefig(out_file, dpi=300)
    plt.savefig(out_file.with_suffix(".svg"))
    plt.close()
    logger.info(f"📊 Saved multi-model LM BPC curves to {out_file.resolve()}")


def plot_domain_radar_chart(runs_dir: Path, base_dir: Path, output_dir: Path):
    """Plot 3: Radar Comparison across 6 domains."""
    domains = ["prose", "technical", "scientific", "code", "numbers", "urls"]
    domain_labels = ["Prose", "Technical", "Scientific", "Code", "Numbers", "URLs"]

    models_to_plot = {}

    # 1. Look for reference gpt2
    ref_gpt2 = base_dir / "reference_gpt2.json"
    if ref_gpt2.is_file():
        m = load_json(ref_gpt2)
        models_to_plot["OpenAI GPT-2 (50k)"] = [m["domains"][d]["compression_ratio"] for d in domains]

    # 2. Gather our key models
    key_runs = {
        "exp_vocab_16k": "Our BPE (16k)",
        "general_purpose_bpe_32k": "Our BPE Baseline (32k)",
        "exp_digits_singledigit": "Our BPE Single-Digit (32k)",
        "exp_code_heavy_32k": "Our BPE Code-Heavy (32k)",
        "exp_vocab_64k": "Our BPE (64k)",
    }

    for r_dir in runs_dir.iterdir():
        if not r_dir.is_dir():
            continue
        m_path = r_dir / "metrics.json"
        if m_path.is_file():
            metrics = load_json(m_path)
            for k, label in key_runs.items():
                if k in r_dir.name and label not in models_to_plot:
                    models_to_plot[label] = [metrics["domains"][d]["compression_ratio"] for d in domains]

    if len(models_to_plot) < 2:
        return

    # Radar chart setup
    num_vars = len(domains)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for (label, vals), col in zip(models_to_plot.items(), palette):
        plot_vals = vals + vals[:1]
        ax.plot(angles, plot_vals, linewidth=2, label=label, color=col)
        ax.fill(angles, plot_vals, color=col, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), domain_labels, fontsize=11, fontweight="bold")
    ax.set_title("Multi-Domain Compression Ratio Radar Chart (Bytes/Token)", fontsize=13, fontweight="bold", pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True, fontsize=9)
    plt.tight_layout()

    out_file = output_dir / "domain_radar_chart.png"
    plt.savefig(out_file, dpi=300)
    plt.savefig(out_file.with_suffix(".svg"))
    plt.close()
    logger.info(f"📊 Saved domain radar comparison chart to {out_file.resolve()}")


def main():
    base_dir = Path("experiments")
    runs_dir = base_dir / "runs"
    output_dir = base_dir / "visuals"
    ensure_dir(output_dir)

    plot_pareto_frontier(runs_dir, output_dir)
    plot_multi_model_lm_curves(runs_dir, output_dir)
    plot_domain_radar_chart(runs_dir, base_dir, output_dir)


if __name__ == "__main__":
    main()
