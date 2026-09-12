"""Unit tests for ExperimentTracker and Leaderboard management."""

from pathlib import Path
import pytest

from src.tokenizer.tokenizer import Tokenizer
from src.tracker.experiment_tracker import ExperimentTracker
from utils.helpers import load_json


@pytest.fixture
def test_tokenizer() -> Tokenizer:
    vocab = {chr(i): i for i in range(256)}
    vocab["<|endoftext|>"] = 256
    merges = []
    return Tokenizer(vocab=vocab, merges=merges, name="test_model_32k")


def test_experiment_tracker_lifecycle(tmp_path: Path, test_tokenizer: Tokenizer):
    tracker = ExperimentTracker(base_dir=tmp_path / "experiments")

    # 1. Init run
    run_id, run_dir = tracker.init_run("exp_test", config_dict={"seed": 42})
    assert run_dir.is_dir()
    assert (run_dir / "config.yaml").is_file()

    # 2. Save artifacts
    metrics = {
        "model_name": "test_model_32k",
        "vocab_size": 257,
        "overall": {
            "total_bytes_evaluated": 50000,
            "total_tokens_evaluated": 12000,
            "total_words_evaluated": 10000,
            "compression_ratio": 4.17,
            "fertility": 1.20,
            "is_100_percent_lossless": True,
            "unk_rate_percent": 0.0,
        },
        "domains": {
            "prose": {"compression_ratio": 4.25, "fertility": 1.15, "byte_count": 20000, "token_count": 4700, "word_count": 4000, "is_lossless": True},
            "code": {"compression_ratio": 3.45, "fertility": 1.35, "byte_count": 20000, "token_count": 5800, "word_count": 4300, "is_lossless": True},
        },
    }
    lm_results = {
        "training_steps": 100,
        "val_loss_per_token": 3.25,
        "val_perplexity": 25.8,
        "val_loss_per_byte": 0.78,
        "val_bits_per_character": 1.12,
    }

    saved_dir = tracker.save_run_artifacts(
        run_id=run_id,
        tokenizer=test_tokenizer,
        metrics=metrics,
        lm_results=lm_results,
    )
    assert (saved_dir / "vocab.json").is_file()
    assert (saved_dir / "merges.txt").is_file()
    assert (saved_dir / "metrics.json").is_file()
    assert (saved_dir / "report.md").is_file()

    # 3. Check Leaderboard
    leaderboard = tracker.get_leaderboard()
    assert len(leaderboard) == 1
    assert leaderboard[0]["model_name"] == "test_model_32k"
    assert leaderboard[0]["overall_cr"] == 4.17
    assert leaderboard[0]["val_bpc"] == 1.12

    # 4. Check Markdown table
    assert tracker.leaderboard_md.is_file()
    content = tracker.leaderboard_md.read_text(encoding="utf-8")
    assert "test_model_32k" in content
    assert "4.17" in content
