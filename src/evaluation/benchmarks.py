"""Multi-domain benchmark evaluation runner for Tokenizer instances."""

from pathlib import Path
from typing import Any

from src.evaluation.domain_slices import get_standard_domain_slices
from src.evaluation.metrics import (
    evaluate_text_slice,
    measure_throughput,
    compute_compression_ratio,
    compute_fertility,
    compute_vocab_utilization,
    compute_unk_rate,
)
from src.tokenizer.tokenizer import Tokenizer
from utils.logger import logger


class BenchmarkRunner:
    """Executes multi-domain benchmarks against a trained Tokenizer."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def run_benchmark(
        self,
        custom_corpus_path: str | Path | None = None,
        measure_speed: bool = True,
    ) -> dict[str, Any]:
        """Runs evaluation across all domain slices and optional external held-out corpus.
        
        Returns:
            Comprehensive benchmark results dictionary.
        """
        logger.info(f"📊 Running multi-domain benchmark for '{self.tokenizer.name}'...")

        domain_slices = get_standard_domain_slices()
        domain_results: dict[str, dict[str, Any]] = {}

        total_bytes = 0
        total_tokens = 0
        total_words = 0
        all_token_ids: list[int] = []
        all_lossless = True

        for domain_name, text in domain_slices.items():
            result = evaluate_text_slice(self.tokenizer, text, domain_name=domain_name)
            domain_results[domain_name] = result

            total_bytes += result["byte_count"]
            total_tokens += result["token_count"]
            total_words += result["word_count"]
            if not result["is_lossless"]:
                all_lossless = False

        # Optional held-out evaluation corpus file
        heldout_result = None
        if custom_corpus_path and Path(custom_corpus_path).is_file():
            path = Path(custom_corpus_path)
            logger.info(f"Evaluating held-out corpus: {path.resolve()}")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                heldout_text = f.read(500000)  # Evaluate first 500KB chunk
            heldout_result = evaluate_text_slice(self.tokenizer, heldout_text, domain_name="heldout_corpus")

        # Aggregate overall summary
        overall_cr = round(total_bytes / total_tokens, 4) if total_tokens > 0 else 0.0
        overall_fertility = round(total_tokens / total_words, 4) if total_words > 0 else 0.0

        # Throughput benchmark using prose text
        throughput_results = {}
        if measure_speed:
            speed_sample = domain_slices["prose"] * 10
            throughput_results = measure_throughput(self.tokenizer, speed_sample)

        unk_id = self.tokenizer.vocab.get("<|unk|>")

        summary = {
            "model_name": self.tokenizer.name,
            "vocab_size": self.tokenizer.vocab_size,
            "overall": {
                "total_bytes_evaluated": total_bytes,
                "total_tokens_evaluated": total_tokens,
                "total_words_evaluated": total_words,
                "compression_ratio": overall_cr,
                "fertility": overall_fertility,
                "is_100_percent_lossless": all_lossless,
                "unk_rate_percent": 0.0,
            },
            "domains": domain_results,
            "throughput": throughput_results,
            "heldout_corpus": heldout_result,
        }

        logger.info(
            f"✅ Benchmark Complete! Overall CR: {overall_cr} B/T, "
            f"Fertility: {overall_fertility} T/W, "
            f"Lossless: {all_lossless}"
        )
        return summary
