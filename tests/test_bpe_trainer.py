"""Unit tests for the BPE training engine."""

from collections import Counter
from src.tokenizer.bpe_trainer import BPETrainer


def test_bpe_trainer_synthetic_corpus():
    trainer = BPETrainer(vocab_size=265, min_frequency=2)
    
    # Synthetic corpus with clear frequent pair "th" and "the"
    word_counts = Counter({
        ("t", "h", "e"): 50,
        ("t", "h", "i", "s"): 30,
        ("t", "h", "a", "t"): 20,
        ("w", "h", "e", "n"): 10,
    })
    
    vocab, merges, stats = trainer.train_from_word_counts(word_counts)
    
    # The most frequent pair must be ('t', 'h') with 50+30+20 = 100 occurrences
    assert len(merges) > 0
    assert merges[0] == ("t", "h")
    assert "th" in vocab
    assert stats["merges_trained"] == len(merges)
    assert len(vocab) >= 256 + 5 + 1


def test_bpe_trainer_min_frequency_cutoff():
    # Only 1 occurrence of pair (x, y)
    trainer = BPETrainer(vocab_size=300, min_frequency=5)
    word_counts = Counter({
        ("x", "y", "z"): 1,
        ("a", "b", "c"): 2,
    })
    vocab, merges, stats = trainer.train_from_word_counts(word_counts)
    # No merges should be performed because max frequency is 2 (< min_frequency 5)
    assert len(merges) == 0
