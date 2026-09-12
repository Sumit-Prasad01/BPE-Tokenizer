"""Evaluation, multi-domain benchmarking, and downstream LM evaluation package."""

from src.evaluation.domain_slices import get_standard_domain_slices
from src.evaluation.metrics import (
    compute_compression_ratio,
    compute_fertility,
    compute_vocab_utilization,
    compute_unk_rate,
    verify_losslessness,
    measure_throughput,
    evaluate_text_slice,
)
from src.evaluation.benchmarks import BenchmarkRunner
from src.evaluation.llm_benchmark import LMBenchmarkRunner, MiniGPT

__all__ = [
    "get_standard_domain_slices",
    "compute_compression_ratio",
    "compute_fertility",
    "compute_vocab_utilization",
    "compute_unk_rate",
    "verify_losslessness",
    "measure_throughput",
    "evaluate_text_slice",
    "BenchmarkRunner",
    "LMBenchmarkRunner",
    "MiniGPT",
]
