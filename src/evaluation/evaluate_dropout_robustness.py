"""Track 4: Subword Regularization and BPE-Dropout Robustness Evaluation."""

import argparse
import random
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns

from src.tokenizer.tokenizer import Tokenizer
from utils.helpers import ensure_dir, save_json
from utils.logger import logger


def perturb_text(text: str, noise_rate: float, seed: int = 42) -> str:
    """Applies synthetic typographic and character perturbations.
    
    Noise includes:
    1. Adjacent character swaps (e.g. 'teh' for 'the')
    2. Leetspeak substitutions (e.g. 'e' -> '3', 'a' -> '4', 'o' -> '0')
    3. Dropped characters
    4. Random keyboard typo substitutions
    """
    if noise_rate <= 0.0:
        return text

    rng = random.Random(seed)
    leetspeak = {"e": "3", "a": "4", "o": "0", "i": "1", "s": "5", "t": "7"}
    keyboard_typos = {"a": "s", "s": "d", "e": "r", "r": "t", "t": "y", "o": "p", "n": "m"}

    chars = list(text)
    n = len(chars)
    i = 0

    while i < n:
        if chars[i].isspace():
            i += 1
            continue

        if rng.random() < noise_rate:
            op = rng.choice(["swap", "leet", "drop", "typo"])
            if op == "swap" and i + 1 < n and not chars[i + 1].isspace():
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                i += 2
                continue
            elif op == "leet" and chars[i].lower() in leetspeak:
                chars[i] = leetspeak[chars[i].lower()]
            elif op == "drop" and len(chars) > 10:
                chars[i] = ""
            elif op == "typo" and chars[i].lower() in keyboard_typos:
                chars[i] = keyboard_typos[chars[i].lower()]
        i += 1

    return "".join(chars)


def evaluate_robustness(
    tokenizer: Tokenizer,
    clean_text: str,
    noise_levels: list[float] | None = None,
    dropout_rates: list[float] | None = None,
    num_stochastic_samples: int = 5,
) -> dict[str, Any]:
    """Evaluates tokenizer compression and fertility across noise levels and BPE-dropout rates."""
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    if dropout_rates is None:
        dropout_rates = [0.0, 0.1, 0.2]

    results: dict[str, Any] = {
        "model_name": tokenizer.name,
        "noise_levels": noise_levels,
        "dropout_rates": dropout_rates,
        "experiments": [],
    }

    logger.info(f"🧪 Evaluating BPE-Dropout robustness on '{tokenizer.name}' across {len(noise_levels)} noise levels...")

    for noise in noise_levels:
        perturbed = perturb_text(clean_text, noise_rate=noise, seed=42)
        raw_bytes = len(perturbed.encode("utf-8"))
        word_count = len(perturbed.split())

        noise_entry: dict[str, Any] = {
            "noise_rate": noise,
            "raw_byte_count": raw_bytes,
            "word_count": word_count,
            "dropout_results": {},
        }

        for p in dropout_rates:
            token_counts = []
            segmentation_samples = []

            for s_idx in range(num_stochastic_samples if p > 0.0 else 1):
                tokens = tokenizer.encode(perturbed, p_dropout=p)
                token_counts.append(len(tokens))

                # Verify losslessness
                decoded = tokenizer.decode(tokens)
                assert decoded == perturbed, f"Losslessness failed for noise={noise}, p_dropout={p}!"

                # Track unique subwords
                if len(segmentation_samples) < 3:
                    subwords = tokenizer.tokenize(perturbed[:200], p_dropout=p)
                    segmentation_samples.append("/".join(subwords[:15]))

            avg_tokens = sum(token_counts) / len(token_counts)
            cr = round(raw_bytes / avg_tokens, 4) if avg_tokens > 0 else 0.0
            fertility = round(avg_tokens / word_count, 4) if word_count > 0 else 0.0

            noise_entry["dropout_results"][f"p_{p}"] = {
                "p_dropout": p,
                "avg_token_count": round(avg_tokens, 1),
                "compression_ratio": cr,
                "fertility": fertility,
                "is_lossless": True,
                "sample_segmentations": segmentation_samples,
            }
            logger.info(
                f"  Noise: {noise*100:4.1f}% | p_dropout: {p:.1f} | "
                f"CR: {cr:.2f} B/T | Fertility: {fertility:.2f} T/W | Lossless: 100%"
            )

        results["experiments"].append(noise_entry)

    return results


def plot_noise_degradation(results: dict[str, Any], output_path: Path) -> None:
    """Generates comparative noise degradation curve visualization."""
    ensure_dir(output_path.parent)
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    noise_pcts = [exp["noise_rate"] * 100 for exp in results["experiments"]]
    dropout_rates = results["dropout_rates"]

    colors = {0.0: "#1f77b4", 0.1: "#ff7f0e", 0.2: "#2ca02c"}
    labels = {0.0: "Deterministic (p=0.0)", 0.1: "BPE-Dropout (p=0.1)", 0.2: "BPE-Dropout (p=0.2)"}

    for p in dropout_rates:
        key = f"p_{p}"
        cr_vals = [exp["dropout_results"][key]["compression_ratio"] for exp in results["experiments"]]
        fert_vals = [exp["dropout_results"][key]["fertility"] for exp in results["experiments"]]

        ax1.plot(noise_pcts, cr_vals, marker="o", linewidth=2.2, label=labels.get(p, f"p={p}"), color=colors.get(p))
        ax2.plot(noise_pcts, fert_vals, marker="s", linewidth=2.2, label=labels.get(p, f"p={p}"), color=colors.get(p))

    ax1.set_title("Compression Ratio (Bytes/Token) vs Typo Noise", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Noise Perturbation Rate (%)", fontsize=10)
    ax1.set_ylabel("Compression Ratio (B/T, higher is better)", fontsize=10)
    ax1.legend(frameon=True)

    ax2.set_title("Token Fertility (Tokens/Word) vs Typo Noise", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Noise Perturbation Rate (%)", fontsize=10)
    ax2.set_ylabel("Token Fertility (T/W, lower is better)", fontsize=10)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".svg"))
    plt.close()
    logger.info(f"📊 Saved noise degradation plot to {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BPE-Dropout subword regularization under typo noise")
    parser.add_argument("--model", default="experiments/runs/20260913_023301_general_purpose_bpe_32k", help="Model directory")
    parser.add_argument("--corpus", default="data/processed/eval_corpus_heldout.txt", help="Evaluation text corpus")
    parser.add_argument("--output-json", default="experiments/bpe_dropout_robustness.json", help="Output JSON path")
    parser.add_argument("--output-plot", default="experiments/visuals/noise_degradation_curve.png", help="Output plot path")
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        # Fallback to any available 32k model
        runs = sorted(Path("experiments/runs").glob("*32k*"))
        if runs:
            model_dir = runs[0]
        else:
            raise FileNotFoundError(f"Model directory not found: {args.model}")

    tokenizer = Tokenizer.from_pretrained(model_dir)

    corpus_path = Path(args.corpus)
    if corpus_path.is_file():
        with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(200000)  # 200KB representative slice
    else:
        from src.evaluation.domain_slices import get_standard_domain_slices
        text = "\n\n".join(get_standard_domain_slices().values())

    results = evaluate_robustness(tokenizer, text)
    save_json(results, Path(args.output_json), indent=2)
    plot_noise_degradation(results, Path(args.output_plot))


if __name__ == "__main__":
    main()
