"""Unit tests for the Tokenizer inference engine and serialization."""

from collections import Counter
from pathlib import Path
import pytest
from tokenizers import Tokenizer as HFTokenizer

from src.tokenizer.bpe_trainer import BPETrainer
from src.tokenizer.tokenizer import Tokenizer
from src.tokenizer.serializer import TokenizerSerializer


@pytest.fixture
def trained_tokenizer() -> Tokenizer:
    """Provides a small trained tokenizer fixture."""
    trainer = BPETrainer(vocab_size=300, min_frequency=2)
    corpus = Counter({
        ("t", "h", "e"): 50,
        ("t", "h", "i", "s"): 30,
        ("t", "h", "a", "t"): 20,
        ("i", "s"): 40,
        ("p", "y", "t", "h", "o", "n"): 35,
        ("c", "o", "d", "e"): 25,
    })
    vocab, merges, _ = trainer.train_from_word_counts(corpus)
    return Tokenizer(vocab=vocab, merges=merges)


def test_tokenizer_lossless_roundtrip(trained_tokenizer: Tokenizer):
    test_cases = [
        "Hello, World!",
        "The quick brown fox jumps over the lazy dog.",
        "def solve(x: int) -> float:\n    return x * 3.14159\n",
        r"\frac{\partial f}{\partial x} = \sum_{i=1}^n x_i^2",
        "Date: 2026-09-12, Float: 3.14159265, Hex: 0xDEADBEEF",
        "URL: https://huggingface.co/datasets?view=viewer#train",
        "Unicode & Emojis: 🚀 🤖 🔥 🐍 🌟, नमस्ते, 你好, مرحبا",
        "Whitespace test:   \t\n  multiple   spaces  and\n\nnewlines\n",
    ]

    for sample in test_cases:
        token_ids = trained_tokenizer.encode(sample)
        assert isinstance(token_ids, list)
        assert all(isinstance(t, int) for t in token_ids)
        
        decoded = trained_tokenizer.decode(token_ids)
        assert decoded == sample, f"Roundtrip failed for: {repr(sample)} != {repr(decoded)}"


def test_tokenizer_special_tokens(trained_tokenizer: Tokenizer):
    text = "Hello <|endoftext|> world"
    # When allowed_special is allowed
    ids = trained_tokenizer.encode(text, allowed_special="all")
    eos_id = trained_tokenizer.vocab["<|endoftext|>"]
    assert eos_id in ids
    decoded = trained_tokenizer.decode(ids)
    assert decoded == text

    # When special token is disallowed, should raise error
    with pytest.raises(Exception):
        trained_tokenizer.encode(text, allowed_special=None)


def test_tokenizer_batch_encode(trained_tokenizer: Tokenizer):
    texts = ["Hello", "def foo():\n    return 42"]
    batch_padded = trained_tokenizer.encode_batch(texts, padding=True)
    assert len(batch_padded[0]) == len(batch_padded[1]), "Batch padding should produce equal sequence lengths"


def test_tokenizer_serialization_and_hf_interop(trained_tokenizer: Tokenizer, tmp_path: Path):
    out_dir = tmp_path / "test_model"
    paths = TokenizerSerializer.save_pretrained(trained_tokenizer, out_dir)

    assert paths["vocab"].is_file()
    assert paths["merges"].is_file()
    assert paths["tokenizer_json"].is_file()
    assert paths["tokenizer_config"].is_file()
    assert paths["special_tokens_map"].is_file()

    # Test loading back via our serializer
    loaded_tokenizer = TokenizerSerializer.from_pretrained(out_dir)
    assert loaded_tokenizer.vocab_size == trained_tokenizer.vocab_size
    assert loaded_tokenizer.encode("Hello python") == trained_tokenizer.encode("Hello python")

    # Test Hugging Face tokenizers interoperability
    hf_tok = HFTokenizer.from_file(str(paths["tokenizer_json"]))
    assert hf_tok.get_vocab_size() == trained_tokenizer.vocab_size
