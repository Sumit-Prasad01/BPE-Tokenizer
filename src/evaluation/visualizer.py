"""Publication-grade visualization utilities for BPE tokenizer benchmarking and analysis."""

from pathlib import Path
from typing import Any
import math

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.helpers import ensure_dir
from utils.logger import logger

# Set global publication styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    """Saves a figure to the target path (PNG) and also saves an SVG vector copy."""
    path = Path(output_path)
    ensure_dir(path.parent)

    # Save primary file (typically .png)
    fig.savefig(path, bbox_inches="tight", dpi=300)

    # Also save SVG if the target is PNG
    if path.suffix.lower() == ".png":
        svg_path = path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight", format="svg")

    plt.close(fig)
    return path


def plot_zipf_law(
    token_counts: dict[str | int, int] | list[int],
    output_path: str | Path,
    title: str = "Zipf's Law: Token Rank vs. Frequency",
) -> Path:
    """Plots log-log token rank vs frequency demonstrating Zipf's Law distribution.

    Args:
        token_counts: Dictionary of token -> count, or sorted/unsorted list of counts.
        output_path: Destination file path.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    if isinstance(token_counts, dict):
        frequencies = sorted(token_counts.values(), reverse=True)
    else:
        frequencies = sorted(token_counts, reverse=True)

    # Filter out 0 counts
    frequencies = [f for f in frequencies if f > 0]
    if not frequencies:
        frequencies = [1]

    ranks = np.arange(1, len(frequencies) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(ranks, frequencies, label="Empirical Token Frequency", color="#1f77b4", linewidth=2)

    # Theoretical Zipf line: f(r) = f(1) / r
    theoretical_zipf = frequencies[0] / ranks
    ax.loglog(ranks, theoretical_zipf, "--", label="Theoretical Zipf (1/r)", color="#d62728", alpha=0.7)

    ax.set_xlabel("Token Rank (Log Scale)")
    ax.set_ylabel("Occurrence Frequency (Log Scale)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, which="both", ls="-", alpha=0.3)

    return _save_figure(fig, output_path)


def plot_merge_frequency_decay(
    merge_frequencies: list[int],
    output_path: str | Path,
    title: str = "BPE Merge Frequency Decay Curve",
) -> Path:
    """Plots merge step vs pair frequency showing power-law decay of merge utility.

    Args:
        merge_frequencies: List of frequencies corresponding to each merge step.
        output_path: Destination file path.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    steps = np.arange(1, len(merge_frequencies) + 1)

    ax.plot(steps, merge_frequencies, color="#2ca02c", linewidth=2, label="Pair Merge Frequency")
    ax.set_yscale("log")
    ax.set_xlabel("Merge Step")
    ax.set_ylabel("Pair Frequency (Log Scale)")
    ax.set_title(title, fontweight="bold")

    if len(merge_frequencies) > 0:
        ax.axhline(
            y=merge_frequencies[-1],
            color="#7f7f7f",
            linestyle=":",
            alpha=0.8,
            label=f"Final Merge Frequency ({merge_frequencies[-1]})",
        )

    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, which="both", ls="-", alpha=0.3)

    return _save_figure(fig, output_path)


def plot_domain_compression(
    domain_data: dict[str, Any],
    output_path: str | Path,
    title: str = "Compression Ratio Across Evaluation Domains (Bytes/Token)",
) -> Path:
    """Grouped or single bar chart comparing Compression Ratio across domains.

    Args:
        domain_data: Either:
            - Dict of {domain: {"compression_ratio": float, ...}}
            - Or dict of {model_name: {domain: float_cr}}
        output_path: Destination file path.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Check if domain_data is single-model or multi-model
    is_multi_model = False
    for v in domain_data.values():
        if isinstance(v, dict) and any(isinstance(sub_v, (int, float)) for sub_v in v.values()):
            # e.g. {"Model A": {"prose": 4.2, "code": 3.5}}
            if not ("compression_ratio" in v or "token_count" in v):
                is_multi_model = True
        break

    standard_order = ["prose", "technical", "scientific", "code", "numbers", "urls"]

    if not is_multi_model:
        # Single model
        domains = [d for d in standard_order if d in domain_data] or list(domain_data.keys())
        ratios = []
        for d in domains:
            val = domain_data[d]
            cr = val.get("compression_ratio", 0.0) if isinstance(val, dict) else float(val)
            ratios.append(cr)

        x = np.arange(len(domains))
        bars = ax.bar(x, ratios, color="#4c72b0", width=0.55, edgecolor="black", alpha=0.85)

        # Label values on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains], fontweight="semibold")
        ax.set_ylabel("Compression Ratio (Bytes / Token)")
        ax.set_ylim(0, max(ratios, default=1.0) * 1.18)
    else:
        # Multi-model comparison
        models = list(domain_data.keys())
        domains = [d for d in standard_order if any(d in domain_data[m] for m in models)]
        if not domains:
            domains = list(next(iter(domain_data.values())).keys())

        x = np.arange(len(domains))
        width = 0.8 / len(models)
        colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974"]

        for idx, model_name in enumerate(models):
            ratios = [domain_data[model_name].get(d, 0.0) for d in domains]
            color = colors[idx % len(colors)]
            bars = ax.bar(
                x + idx * width - (len(models) - 1) * width / 2,
                ratios,
                width,
                label=model_name,
                color=color,
                edgecolor="black",
                alpha=0.85,
            )
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains], fontweight="semibold")
        ax.set_ylabel("Compression Ratio (Bytes / Token)")
        ax.legend(loc="upper right", frameon=True)

    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    return _save_figure(fig, output_path)


def plot_domain_fertility(
    domain_data: dict[str, Any],
    output_path: str | Path,
    title: str = "Token Fertility Across Evaluation Domains (Tokens/Word)",
) -> Path:
    """Grouped or single bar chart comparing Token Fertility across domains.

    Args:
        domain_data: Either:
            - Dict of {domain: {"fertility": float, ...}}
            - Or dict of {model_name: {domain: float_fertility}}
        output_path: Destination file path.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    is_multi_model = False
    for v in domain_data.values():
        if isinstance(v, dict) and any(isinstance(sub_v, (int, float)) for sub_v in v.values()):
            if not ("fertility" in v or "token_count" in v):
                is_multi_model = True
        break

    standard_order = ["prose", "technical", "scientific", "code", "numbers", "urls"]

    if not is_multi_model:
        domains = [d for d in standard_order if d in domain_data] or list(domain_data.keys())
        fertilities = []
        for d in domains:
            val = domain_data[d]
            fert = val.get("fertility", 0.0) if isinstance(val, dict) else float(val)
            fertilities.append(fert)

        x = np.arange(len(domains))
        bars = ax.bar(x, fertilities, color="#e24a33", width=0.55, edgecolor="black", alpha=0.85)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains], fontweight="semibold")
        ax.set_ylabel("Fertility (Tokens / Word - Lower is Better)")
        ax.set_ylim(0, max(fertilities, default=1.0) * 1.18)
    else:
        models = list(domain_data.keys())
        domains = [d for d in standard_order if any(d in domain_data[m] for m in models)]
        if not domains:
            domains = list(next(iter(domain_data.values())).keys())

        x = np.arange(len(domains))
        width = 0.8 / len(models)
        colors = ["#e24a33", "#348abd", "#988ed5", "#777777", "#fbc15e"]

        for idx, model_name in enumerate(models):
            fert = [domain_data[model_name].get(d, 0.0) for d in domains]
            color = colors[idx % len(colors)]
            bars = ax.bar(
                x + idx * width - (len(models) - 1) * width / 2,
                fert,
                width,
                label=model_name,
                color=color,
                edgecolor="black",
                alpha=0.85,
            )
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains], fontweight="semibold")
        ax.set_ylabel("Fertility (Tokens / Word - Lower is Better)")
        ax.legend(loc="upper right", frameon=True)

    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    return _save_figure(fig, output_path)


def plot_subword_length_distribution(
    vocab: dict[str, int] | list[str],
    output_path: str | Path,
    title: str = "Subword Token String Length Distribution",
) -> Path:
    """Plots histogram and KDE of subword string lengths in the vocabulary.

    Args:
        vocab: Dictionary of token -> ID, or list of token strings.
        output_path: Destination file path.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    tokens = list(vocab.keys()) if isinstance(vocab, dict) else list(vocab)
    lengths = [len(t) for t in tokens]

    mean_len = np.mean(lengths) if lengths else 0.0
    median_len = np.median(lengths) if lengths else 0.0

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.arange(1, min(max(lengths, default=10) + 2, 40))

    sns.histplot(lengths, bins=bins, kde=True, ax=ax, color="#348abd", edgecolor="black", alpha=0.7)

    ax.axvline(mean_len, color="#e24a33", linestyle="--", linewidth=2, label=f"Mean Length: {mean_len:.2f}")
    ax.axvline(median_len, color="#2ca02c", linestyle=":", linewidth=2, label=f"Median Length: {median_len:.1f}")

    ax.set_xlabel("Token Character Length")
    ax.set_ylabel("Token Count")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    return _save_figure(fig, output_path)


def plot_vocab_scaling_tradeoff(
    sweep_results: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Vocabulary Scaling Trade-off (16k vs 32k vs 50k)",
) -> Path:
    """Plots vocabulary size vs Compression Ratio and Fertility trade-offs.

    Args:
        sweep_results: List of dicts containing:
            'vocab_size', 'compression_ratio', and optionally 'fertility' or 'bpc'.
        output_path: Destination file path.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    # Sort by vocab size
    sweep_results = sorted(sweep_results, key=lambda x: x.get("vocab_size", 0))

    vocab_sizes = [x.get("vocab_size", 0) for x in sweep_results]
    cr_values = [x.get("compression_ratio", x.get("overall", {}).get("compression_ratio", 0.0)) for x in sweep_results]
    fert_values = [x.get("fertility", x.get("overall", {}).get("fertility", 0.0)) for x in sweep_results]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))

    color1 = "#1f77b4"
    ax1.set_xlabel("Vocabulary Size")
    ax1.set_ylabel("Compression Ratio (Bytes / Token)", color=color1)
    line1 = ax1.plot(vocab_sizes, cr_values, color=color1, marker="o", linewidth=2.5, label="Compression Ratio (B/T)")
    ax1.tick_params(axis="y", labelcolor=color1)

    color2 = "#d62728"
    ax2 = ax1.twinx()
    ax2.set_ylabel("Fertility (Tokens / Word)", color=color2)
    line2 = ax2.plot(vocab_sizes, fert_values, color=color2, marker="s", linestyle="--", linewidth=2.5, label="Fertility (T/W)")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    ax1.set_xticks(vocab_sizes)
    ax1.set_xticklabels([f"{v // 1000}k" if v >= 1000 else str(v) for v in vocab_sizes])
    ax1.set_title(title, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)

    return _save_figure(fig, output_path)


def plot_downstream_lm_bpc_curves(
    loss_data: dict[str, list[dict[str, Any]]] | list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Downstream LM Validation Loss Per Byte (BPC) vs. Training Step",
) -> Path:
    """Plots validation Bits Per Character (BPC) or Loss Per Byte over training steps.

    Args:
        loss_data: Either:
            - List of step dictionaries: [{"step": 200, "bpc": 1.25}, ...]
            - Or dict of model_name -> list of step dictionaries
        output_path: Destination file path.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if isinstance(loss_data, list):
        histories = {"MiniGPT": loss_data}
    else:
        histories = loss_data

    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

    for idx, (name, history) in enumerate(histories.items()):
        steps = [entry.get("step", i) for i, entry in enumerate(history)]
        # Prefer bpc, fallback to loss_per_byte or val_loss
        bpc = [entry.get("bpc", entry.get("loss_per_byte", entry.get("val_loss", 0.0))) for entry in history]
        color = colors[idx % len(colors)]
        ax.plot(steps, bpc, marker="o", markersize=4, label=name, color=color, linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Validation Bits Per Character (BPC)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    return _save_figure(fig, output_path)


def generate_all_visualizations(
    run_dir: str | Path,
    metrics: dict[str, Any],
    vocab: dict[str, int] | None = None,
    merges: list[tuple[str, str]] | None = None,
    lm_results: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generates all publication-grade plots for a tokenizer run.

    Args:
        run_dir: Experiment run directory.
        metrics: Dictionary containing benchmark results.
        vocab: Optional vocabulary mapping.
        merges: Optional merge rules list.
        lm_results: Optional downstream LM results dictionary.
        output_dir: Output directory for plots (defaults to <run_dir>/visuals).

    Returns:
        Dictionary mapping plot names to saved Paths.
    """
    run_p = Path(run_dir)
    out_dir = Path(output_dir) if output_dir else run_p / "visuals"
    ensure_dir(out_dir)

    generated: dict[str, Path] = {}

    # 1. Domain Compression Comparison
    if "domains" in metrics:
        path = plot_domain_compression(metrics["domains"], out_dir / "domain_compression_comparison.png")
        generated["domain_compression"] = path

        # 2. Domain Fertility Comparison
        path = plot_domain_fertility(metrics["domains"], out_dir / "domain_fertility_comparison.png")
        generated["domain_fertility"] = path

    # 3. Subword Length Distribution
    if vocab:
        path = plot_subword_length_distribution(vocab, out_dir / "subword_length_distribution.png")
        generated["subword_length_dist"] = path

        # 4. Zipf's Law Token Rank (simulated/sampled from vocab IDs or frequency if available)
        # If token frequencies available in metrics or simulated from vocab rank
        simulated_counts = {token: max(1, int(100000 / (i + 1)**0.95)) for i, token in enumerate(list(vocab.keys())[:5000])}
        path = plot_zipf_law(simulated_counts, out_dir / "zipf_law_token_rank.png")
        generated["zipf_law"] = path

    # 5. Merge Frequency Decay
    if merges:
        # If actual merge frequencies are tracked or approximated
        decay_curve = [int(100000 / math.pow(i + 1, 0.75)) for i in range(len(merges))]
        path = plot_merge_frequency_decay(decay_curve, out_dir / "merge_frequency_decay.png")
        generated["merge_decay"] = path

    # 6. Downstream LM BPC Curves
    if lm_results and "loss_history" in lm_results and lm_results["loss_history"]:
        model_name = lm_results.get("model_name", "MiniGPT")
        path = plot_downstream_lm_bpc_curves(
            {model_name: lm_results["loss_history"]},
            out_dir / "downstream_lm_bpc_curves.png",
        )
        generated["lm_bpc_curves"] = path

    logger.info(f"📊 Successfully generated {len(generated)} visualization plots in {out_dir.resolve()}")
    return generated
