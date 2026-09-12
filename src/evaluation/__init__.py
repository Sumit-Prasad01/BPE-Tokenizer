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
from src.evaluation.visualizer import (
    plot_zipf_law,
    plot_merge_frequency_decay,
    plot_domain_compression,
    plot_domain_fertility,
    plot_subword_length_distribution,
    plot_vocab_scaling_tradeoff,
    plot_downstream_lm_bpc_curves,
    generate_all_visualizations,
)

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
    "plot_zipf_law",
    "plot_merge_frequency_decay",
    "plot_domain_compression",
    "plot_domain_fertility",
    "plot_subword_length_distribution",
    "plot_vocab_scaling_tradeoff",
    "plot_downstream_lm_bpc_curves",
    "generate_all_visualizations",
]
