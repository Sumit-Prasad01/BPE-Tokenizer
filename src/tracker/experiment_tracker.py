"""Experiment tracker and central leaderboard generator for BPE Tokenizer benchmarks."""

import datetime
from pathlib import Path
from typing import Any

from src.tokenizer.serializer import TokenizerSerializer
from src.tokenizer.tokenizer import Tokenizer
from utils.helpers import ensure_dir, save_json, load_json, dump_yaml, format_number
from utils.logger import logger


class ExperimentTracker:
    """Tracks runs immutably, archives models and metrics, and updates central leaderboard."""

    def __init__(self, base_dir: str | Path = "experiments"):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.visuals_dir = self.base_dir / "visuals"
        self.leaderboard_json = self.base_dir / "leaderboard.json"
        self.leaderboard_md = self.base_dir / "leaderboard.md"

        ensure_dir(self.runs_dir)
        ensure_dir(self.visuals_dir)

    def init_run(self, exp_name: str, config_dict: dict[str, Any] | None = None) -> tuple[str, Path]:
        """Initializes a new immutable experiment run directory.

        Args:
            exp_name: Short identifier/name for the experiment.
            config_dict: Optional configuration dictionary to freeze.

        Returns:
            Tuple of (run_id, run_directory_path).
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_name = exp_name.strip().replace(" ", "_").replace("/", "_").lower()
        run_id = f"{timestamp}_{sanitized_name}"
        run_dir = self.runs_dir / run_id
        ensure_dir(run_dir)

        if config_dict:
            dump_yaml(config_dict, run_dir / "config.yaml")

        logger.info(f"📁 Initialized experiment run '{run_id}' at {run_dir.resolve()}")
        return run_id, run_dir

    def save_run_artifacts(
        self,
        run_id: str,
        tokenizer: Tokenizer | None = None,
        metrics: dict[str, Any] | None = None,
        lm_results: dict[str, Any] | None = None,
        config_dict: dict[str, Any] | None = None,
    ) -> Path:
        """Saves all run artifacts including serialized tokenizer, metrics, and report.

        Args:
            run_id: Experiment run ID.
            tokenizer: Trained Tokenizer instance.
            metrics: Benchmark evaluation dictionary.
            lm_results: Downstream LM results dictionary.
            config_dict: Configuration used for run.

        Returns:
            Path to run directory.
        """
        run_dir = self.runs_dir / run_id
        ensure_dir(run_dir)

        if config_dict and not (run_dir / "config.yaml").exists():
            dump_yaml(config_dict, run_dir / "config.yaml")

        if tokenizer:
            TokenizerSerializer.save_pretrained(tokenizer, run_dir)

        if metrics:
            save_json(metrics, run_dir / "metrics.json", indent=2)

        if lm_results:
            save_json(lm_results, run_dir / "lm_eval.json", indent=2)

        # Generate run report
        report_content = self.generate_run_report(run_id, metrics, lm_results)
        with open(run_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(report_content)

        # Update leaderboard
        self._record_to_leaderboard(run_id, metrics, lm_results)

        logger.info(f"💾 Run '{run_id}' artifacts saved successfully.")
        return run_dir

    def generate_run_report(
        self,
        run_id: str,
        metrics: dict[str, Any] | None = None,
        lm_results: dict[str, Any] | None = None,
    ) -> str:
        """Generates a comprehensive Markdown report for a specific run."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = metrics.get("model_name", run_id) if metrics else run_id
        vocab_size = metrics.get("vocab_size", "N/A") if metrics else "N/A"

        overall = metrics.get("overall", {}) if metrics else {}
        cr = overall.get("compression_ratio", "N/A")
        fertility = overall.get("fertility", "N/A")
        lossless = "✅ 100% Lossless" if overall.get("is_100_percent_lossless") else "❌ Drift detected"
        unk_rate = f"{overall.get('unk_rate_percent', 0.0):.2f}%"

        lines = [
            f"# Experiment Run Report: {model_name}",
            "",
            f"- **Run ID:** `{run_id}`",
            f"- **Timestamp:** {now}",
            f"- **Vocabulary Size:** {vocab_size}",
            f"- **Overall Compression Ratio:** {cr} Bytes/Token",
            f"- **Overall Fertility:** {fertility} Tokens/Word",
            f"- **UNK Rate:** {unk_rate}",
            f"- **Roundtrip Fidelity:** {lossless}",
            "",
            "## Multi-Domain Evaluation Breakdown",
            "",
            "| Domain | Bytes Evaluated | Tokens | Words | Compression Ratio (B/T) | Fertility (T/W) | Lossless |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]

        domains = metrics.get("domains", {}) if metrics else {}
        for domain, data in domains.items():
            lines.append(
                f"| **{domain.capitalize()}** | "
                f"{format_number(data.get('byte_count', 0))} | "
                f"{format_number(data.get('token_count', 0))} | "
                f"{format_number(data.get('word_count', 0))} | "
                f"{data.get('compression_ratio', 0.0):.2f} | "
                f"{data.get('fertility', 0.0):.2f} | "
                f"{'✅' if data.get('is_lossless') else '❌'} |"
            )

        if lm_results:
            lines.extend([
                "",
                "## Downstream Small LM Benchmark",
                "",
                f"- **Model Architecture:** MiniGPT (6-layer, 6-head, 384-embd)",
                f"- **Training Steps:** {lm_results.get('training_steps', 'N/A')}",
                f"- **Validation Loss Per Token:** {lm_results.get('val_loss_per_token', 'N/A')}",
                f"- **Validation Perplexity:** {lm_results.get('val_perplexity', 'N/A')}",
                f"- **Validation Loss Per Byte:** {lm_results.get('val_loss_per_byte', 'N/A')}",
                f"- **Validation Bits Per Character (BPC):** {lm_results.get('val_bits_per_character', 'N/A')}",
            ])

        lines.append("")
        return "\n".join(lines)

    def _record_to_leaderboard(
        self,
        run_id: str,
        metrics: dict[str, Any] | None,
        lm_results: dict[str, Any] | None,
    ) -> None:
        """Parses run artifacts and appends or updates the central leaderboard."""
        if not metrics:
            return

        leaderboard = self.get_leaderboard()

        overall = metrics.get("overall", {})
        domains = metrics.get("domains", {})

        def _get_cr(domain_key: str) -> float:
            d = domains.get(domain_key, {})
            return round(float(d.get("compression_ratio", 0.0)), 2)

        entry = {
            "run_id": run_id,
            "model_name": metrics.get("model_name", run_id),
            "vocab_size": metrics.get("vocab_size", 0),
            "total_tokens": overall.get("total_tokens_evaluated", 0),
            "overall_cr": round(float(overall.get("compression_ratio", 0.0)), 2),
            "fertility": round(float(overall.get("fertility", 0.0)), 2),
            "prose_cr": _get_cr("prose"),
            "code_cr": _get_cr("code"),
            "tech_cr": _get_cr("technical"),
            "science_cr": _get_cr("scientific"),
            "numbers_cr": _get_cr("numbers"),
            "urls_cr": _get_cr("urls"),
            "unk_rate": f"{overall.get('unk_rate_percent', 0.0):.2f}%",
            "lossless": "100%" if overall.get("is_100_percent_lossless", True) else "No",
            "val_bpc": round(float(lm_results.get("val_bits_per_character", 0.0)), 3) if lm_results else "-",
            "val_loss": round(float(lm_results.get("val_loss_per_token", 0.0)), 3) if lm_results else "-",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Update if exists, else append
        existing_idx = next((i for i, r in enumerate(leaderboard) if r.get("run_id") == run_id), None)
        if existing_idx is not None:
            leaderboard[existing_idx] = entry
        else:
            leaderboard.append(entry)

        # Sort by Overall CR descending
        leaderboard.sort(key=lambda x: x.get("overall_cr", 0.0), reverse=True)

        save_json(leaderboard, self.leaderboard_json, indent=2)
        self.render_leaderboard_table()

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Reads existing leaderboard from disk."""
        if self.leaderboard_json.is_file():
            try:
                return load_json(self.leaderboard_json)
            except Exception as e:
                logger.warning(f"Could not load leaderboard JSON: {e}")
        return []

    def render_leaderboard_table(self) -> str:
        """Renders and saves the central comparative Markdown leaderboard."""
        entries = self.get_leaderboard()

        header = (
            "# 🏆 General-Purpose Tokenizer Comparison Leaderboard\n\n"
            "Side-by-side performance benchmarking across all trained tokenizers and reference models.\n\n"
            "| Tokenizer Model | Vocab | Total Tokens | Overall CR (B/T) | Fertility (T/W) | Prose CR | Code CR | Tech CR | Science CR | Numbers CR | URLs CR | UNK Rate | Lossless | Val BPC |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        rows = []
        for e in entries:
            name = e.get("model_name", "Unknown")
            vocab = f"{e.get('vocab_size', 0) // 1000}k" if e.get("vocab_size", 0) >= 1000 else str(e.get("vocab_size", 0))
            tokens = format_number(e.get("total_tokens", 0))
            overall_cr = f"{e.get('overall_cr', 0.0):.2f}"
            fertility = f"{e.get('fertility', 0.0):.2f}"
            prose = f"{e.get('prose_cr', 0.0):.2f}"
            code = f"{e.get('code_cr', 0.0):.2f}"
            tech = f"{e.get('tech_cr', 0.0):.2f}"
            science = f"{e.get('science_cr', 0.0):.2f}"
            numbers = f"{e.get('numbers_cr', 0.0):.2f}"
            urls = f"{e.get('urls_cr', 0.0):.2f}"
            unk = e.get("unk_rate", "0.00%")
            lossless = e.get("lossless", "100%")
            bpc = str(e.get("val_bpc", "-"))

            rows.append(
                f"| **{name}** | {vocab} | {tokens} | {overall_cr} | {fertility} | "
                f"{prose} | {code} | {tech} | {science} | {numbers} | {urls} | "
                f"{unk} | {lossless} | {bpc} |"
            )

        markdown = header + "\n".join(rows) + "\n"

        with open(self.leaderboard_md, "w", encoding="utf-8") as f:
            f.write(markdown)

        logger.info(f"📊 Central Leaderboard updated at {self.leaderboard_md.resolve()}")
        return markdown

    def scan_and_rebuild_leaderboard(self) -> tuple[list[dict[str, Any]], str]:
        """Scans all run folders in experiments/runs and rebuilds the leaderboard."""
        runs = []
        if not self.runs_dir.exists():
            return [], ""

        for run_path in sorted(self.runs_dir.iterdir()):
            if run_path.is_dir():
                metrics_file = run_path / "metrics.json"
                lm_file = run_path / "lm_eval.json"
                if metrics_file.is_file():
                    try:
                        metrics = load_json(metrics_file)
                        lm_results = load_json(lm_file) if lm_file.is_file() else None
                        self._record_to_leaderboard(run_path.name, metrics, lm_results)
                    except Exception as exc:
                        logger.warning(f"Failed to parse run {run_path.name}: {exc}")

        return self.get_leaderboard(), self.render_leaderboard_table()
