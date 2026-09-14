"""Unit and integration tests for BPEInferenceEngine and Real-World Stress Suite."""

from pathlib import Path
import pytest
import torch

from src.inference.engine import BPEInferenceEngine, StreamingTextDecoder
from src.inference.real_world_tester import run_stress_suite


@pytest.fixture(scope="module")
def model_dir():
    root = Path(__file__).resolve().parent.parent
    run_dir = root / "experiments" / "runs" / "20260914_023959_exp_vocab_64k"
    if not run_dir.exists():
        runs = list((root / "experiments" / "runs").glob("*_exp_*"))
        run_dir = runs[0] if runs else (root / "experiments" / "runs" / "20260913_023301_general_purpose_bpe_32k")
    assert run_dir.exists()
    return run_dir


def test_inference_engine_pytorch_tensors(model_dir):
    """Verify PyTorch tensor formatting, padding, and attention mask generation."""
    engine = BPEInferenceEngine.from_pretrained(model_dir)

    texts = ["Short phrase.", "This is a significantly longer sentence to test padding."]
    out = engine.encode(texts, padding=True, pad_to_multiple_of=8, return_tensors="pt")

    assert isinstance(out["input_ids"], torch.Tensor)
    assert isinstance(out["attention_mask"], torch.Tensor)
    assert out["input_ids"].dtype == torch.long
    assert out["attention_mask"].dtype == torch.long

    batch_size, seq_len = out["input_ids"].shape
    assert batch_size == 2
    assert seq_len % 8 == 0  # pad_to_multiple_of=8 alignment

    # Verify attention mask matches pad tokens
    pad_id = engine.pad_token_id
    for b in range(batch_size):
        for s in range(seq_len):
            tok = out["input_ids"][b, s].item()
            mask_val = out["attention_mask"][b, s].item()
            if tok == pad_id:
                assert mask_val == 0
            else:
                assert mask_val == 1


def test_inference_engine_sliding_window_stride(model_dir):
    """Verify sliding-window chunking over long sequences."""
    engine = BPEInferenceEngine.from_pretrained(model_dir)
    long_text = "The quick brown fox jumps over the lazy dog. " * 12

    out = engine.encode(
        long_text,
        max_length=16,
        stride=4,
        padding="max_length",
        return_tensors="pt",
    )

    assert out["input_ids"].shape[0] > 1  # Multiple chunks created
    assert out["input_ids"].shape[1] == 16  # Fixed window length


def test_streaming_decoder_protection(model_dir):
    """Verify streaming decoder protects split multi-byte characters and emojis."""
    engine = BPEInferenceEngine.from_pretrained(model_dir)
    sample = "Streaming emoji test: 🌍🚀 and Hindi: नमस्ते"

    tokens = engine.encode(sample)
    streamer = engine.create_streamer()

    emitted_chunks = []
    for tok in tokens:
        chunk = streamer.feed(tok)
        emitted_chunks.append(chunk)

    emitted_chunks.append(streamer.flush())
    reconstructed = "".join(emitted_chunks)

    assert reconstructed == sample
    assert "\ufffd" not in reconstructed  # Zero replacement characters


def test_real_world_stress_suite_50_cases(model_dir):
    """Execute the full 55-case dirty data stress test suite and assert 100% pass rate."""
    _, summary = run_stress_suite(model_dir, verbose=False)

    assert summary["total_test_cases"] >= 50
    assert summary["failures_count"] == 0
    assert summary["lossless_pass_rate"] == "100.00%"
    assert summary["zero_unk_pass_rate"] == "100.00%"
    assert summary["streaming_pass_rate"] == "100.00%"
