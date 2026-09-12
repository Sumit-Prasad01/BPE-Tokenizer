"""Unit tests for publication-grade visualization utilities."""

import pytest
from pathlib import Path

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


@pytest.fixture
def tmp_vis_dir(tmp_path: Path) -> Path:
    d = tmp_path / "visuals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_plot_zipf_law(tmp_vis_dir: Path):
    counts = {"the": 1000, "of": 800, "and": 600, "to": 400, "a": 300, "in": 200}
    out_file = tmp_vis_dir / "zipf.png"
    result = plot_zipf_law(counts, out_file)
    assert result.is_file()
    assert result.stat().st_size > 0
    # Also verify SVG copy was produced
    assert (tmp_vis_dir / "zipf.svg").is_file()


def test_plot_merge_decay(tmp_vis_dir: Path):
    decay = [1000, 500, 300, 200, 150, 100, 80, 50, 20, 10]
    out_file = tmp_vis_dir / "merge_decay.png"
    result = plot_merge_frequency_decay(decay, out_file)
    assert result.is_file()
    assert result.stat().st_size > 0


def test_plot_domain_compression_and_fertility(tmp_vis_dir: Path):
    domain_data = {
        "prose": {"compression_ratio": 4.25, "fertility": 1.15},
        "code": {"compression_ratio": 3.45, "fertility": 1.35},
        "technical": {"compression_ratio": 3.80, "fertility": 1.20},
        "scientific": {"compression_ratio": 3.20, "fertility": 1.45},
        "numbers": {"compression_ratio": 2.50, "fertility": 1.80},
        "urls": {"compression_ratio": 3.60, "fertility": 1.25},
    }

    cr_path = plot_domain_compression(domain_data, tmp_vis_dir / "cr.png")
    assert cr_path.is_file()

    fert_path = plot_domain_fertility(domain_data, tmp_vis_dir / "fert.png")
    assert fert_path.is_file()


def test_plot_multi_model_domain_comparison(tmp_vis_dir: Path):
    multi_data = {
        "Baseline A": {"prose": 3.8, "code": 2.8, "technical": 3.2},
        "Exp C": {"prose": 4.3, "code": 3.5, "technical": 3.8},
    }
    out = plot_domain_compression(multi_data, tmp_vis_dir / "multi_cr.png")
    assert out.is_file()


def test_plot_subword_length_distribution(tmp_vis_dir: Path):
    vocab = {"a": 0, "the": 1, "Ġhello": 2, "Ġinternationalization": 3, "Ġtransformer": 4}
    out = plot_subword_length_distribution(vocab, tmp_vis_dir / "subword_len.png")
    assert out.is_file()


def test_plot_vocab_scaling_tradeoff(tmp_vis_dir: Path):
    sweep_results = [
        {"vocab_size": 16000, "compression_ratio": 3.95, "fertility": 1.24},
        {"vocab_size": 32000, "compression_ratio": 4.32, "fertility": 1.12},
        {"vocab_size": 50000, "compression_ratio": 4.45, "fertility": 1.08},
    ]
    out = plot_vocab_scaling_tradeoff(sweep_results, tmp_vis_dir / "tradeoff.png")
    assert out.is_file()


def test_plot_downstream_lm_bpc_curves(tmp_vis_dir: Path):
    loss_data = [
        {"step": 100, "bpc": 1.45},
        {"step": 200, "bpc": 1.25},
        {"step": 300, "bpc": 1.10},
    ]
    out = plot_downstream_lm_bpc_curves(loss_data, tmp_vis_dir / "bpc.png")
    assert out.is_file()


def test_generate_all_visualizations(tmp_vis_dir: Path):
    metrics = {
        "model_name": "test_tok",
        "domains": {
            "prose": {"compression_ratio": 4.0, "fertility": 1.2},
            "code": {"compression_ratio": 3.2, "fertility": 1.4},
        },
    }
    vocab = {"a": 0, "b": 1, "ab": 2}
    merges = [("a", "b")]
    plots = generate_all_visualizations(
        run_dir=tmp_vis_dir,
        metrics=metrics,
        vocab=vocab,
        merges=merges,
        output_dir=tmp_vis_dir,
    )
    assert len(plots) >= 4
    assert "domain_compression" in plots
    assert "domain_fertility" in plots
    assert "subword_length_dist" in plots
