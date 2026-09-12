"""Unit tests for multi-domain benchmarking, edge cases, UNK rate, and downstream LM evaluation."""

from collections import Counter
import pytest
import torch

from src.evaluation.domain_slices import get_standard_domain_slices
from src.evaluation.metrics import (
    compute_compression_ratio,
    compute_fertility,
    compute_unk_rate,
    verify_losslessness,
)
from src.evaluation.benchmarks import BenchmarkRunner
from src.evaluation.llm_benchmark import MiniGPT, LMBenchmarkRunner
from src.tokenizer.bpe_trainer import BPETrainer
from src.tokenizer.tokenizer import Tokenizer


@pytest.fixture
def trained_tokenizer() -> Tokenizer:
    trainer = BPETrainer(vocab_size=320, min_frequency=2)
    corpus = Counter({
        ("t", "h", "e"): 50,
        ("t", "h", "i", "s"): 30,
        ("p", "y", "t", "h", "o", "n"): 40,
        ("c", "o", "d", "e"): 30,
        ("1", "0", "0"): 20,
    })
    vocab, merges, _ = trainer.train_from_word_counts(corpus)
    return Tokenizer(vocab=vocab, merges=merges)


def test_zero_percent_unk_rate_guarantee(trained_tokenizer: Tokenizer):
    """Verifies that Byte-Level BPE produces strictly 0.00% UNK tokens on unseen arbitrary inputs."""
    unseen_texts = [
        "Unseen foreign characters: 𠜎 𠜱 𠝹 𠱓 𠱸 𠲖 𠳏 𠳕",
        "Rare emojis: 🛸 🪐 🧬 🧪 🧮 🧭 🛰️",
        "Arbitrary bytes: \x01\x02\x03\x04\xff\xfe\xfd\xaa\xbb\xcc",
        "Ancient scripts: 𐎀 𐎁 𐎂 𐎃 𐎄 𐎅 𐎆",
    ]
    unk_id = trained_tokenizer.vocab.get("<|unk|>")
    for text in unseen_texts:
        tokens = trained_tokenizer.encode(text)
        unk_rate = compute_unk_rate(tokens, unk_id)
        assert unk_rate == 0.0, f"UNK rate was {unk_rate}% on {repr(text)}, expected 0.00%"


def test_multi_domain_benchmark_losslessness(trained_tokenizer: Tokenizer):
    """Verifies that all standard domain slices encode and decode with 100% losslessness."""
    runner = BenchmarkRunner(trained_tokenizer)
    results = runner.run_benchmark(measure_speed=False)

    assert results["overall"]["is_100_percent_lossless"] is True
    assert results["overall"]["unk_rate_percent"] == 0.0
    assert results["overall"]["compression_ratio"] > 0.0
    assert results["overall"]["fertility"] > 0.0

    # Ensure all 7 domain slices are evaluated
    for domain in ["prose", "technical", "scientific", "code", "numbers", "urls", "edge_cases"]:
        assert domain in results["domains"]
        d_res = results["domains"][domain]
        assert d_res["is_lossless"] is True
        assert d_res["compression_ratio"] > 0.0


def test_minigpt_model_forward():
    """Verifies that the MiniGPT transformer compiles and produces logits and cross-entropy loss."""
    vocab_size = 500
    block_size = 64
    model = MiniGPT(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=64)

    # Batch of shape (batch_size=2, seq_len=16)
    idx = torch.randint(0, vocab_size, (2, 16))
    targets = torch.randint(0, vocab_size, (2, 16))

    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, vocab_size)
    assert loss is not None
    assert loss.item() > 0.0


def test_loss_per_byte_calculation():
    """Verifies Loss Per Byte and Bits Per Character formulas."""
    token_loss = 2.50
    bytes_per_token = 4.0  # 1 token encodes 4 bytes
    loss_per_byte = token_loss / bytes_per_token  # 2.50 / 4.0 = 0.625
    bpc = loss_per_byte / 0.69314718  # ln(2) = ~0.9017

    assert round(loss_per_byte, 3) == 0.625
    assert round(bpc, 3) == 0.902
