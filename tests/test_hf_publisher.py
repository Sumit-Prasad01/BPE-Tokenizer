"""Unit tests for Hugging Face Hub publisher and Model Card generator."""

from pathlib import Path
import pytest

from src.hub.hf_publisher import HFPublisher
from src.tokenizer.serializer import TokenizerSerializer
from src.tokenizer.tokenizer import Tokenizer


@pytest.fixture
def staged_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model_artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = {chr(i): i for i in range(256)}
    vocab["<|endoftext|>"] = 256
    merges = []
    tok = Tokenizer(vocab=vocab, merges=merges, name="gpt_bpe_test")
    TokenizerSerializer.save_pretrained(tok, model_dir)
    return model_dir


def test_generate_model_card():
    metrics = {
        "model_name": "GPT-BPE-32k-General",
        "vocab_size": 32000,
        "overall": {
            "compression_ratio": 4.35,
            "fertility": 1.12,
            "is_100_percent_lossless": True,
            "unk_rate_percent": 0.0,
        },
        "domains": {
            "prose": {"byte_count": 100000, "token_count": 23000, "compression_ratio": 4.35, "fertility": 1.12},
            "code": {"byte_count": 100000, "token_count": 28000, "compression_ratio": 3.57, "fertility": 1.34},
        },
    }

    card = HFPublisher.generate_model_card(metrics=metrics, repo_id="testuser/gpt-bpe-32k")
    assert "language:" in card
    assert "pipeline_tag: feature-extraction" in card
    assert "GPT-BPE-32k-General" in card
    assert "4.35" in card
    assert "AutoTokenizer.from_pretrained(\"testuser/gpt-bpe-32k\")" in card


def test_prepare_hub_package(staged_model_dir: Path, tmp_path: Path):
    dest = tmp_path / "staged_hf"
    result = HFPublisher.prepare_hub_package(
        model_dir=staged_model_dir,
        repo_id="testuser/sample-bpe",
        output_dir=dest,
    )
    assert result.is_dir()
    assert (result / "vocab.json").is_file()
    assert (result / "merges.txt").is_file()
    assert (result / "tokenizer.json").is_file()
    assert (result / "tokenizer_config.json").is_file()
    assert (result / "special_tokens_map.json").is_file()
    assert (result / "README.md").is_file()


def test_publish_to_hub_dry_run(staged_model_dir: Path):
    # Dry run should pass without HF token or network access
    success = HFPublisher.publish_to_hub(
        model_dir=staged_model_dir,
        repo_id="testuser/sample-bpe",
        dry_run=True,
    )
    assert success is True
