"""Unit tests for the regex pre-tokenizer."""

from collections import Counter
from src.tokenizer.pre_tokenizer import RegexPreTokenizer


def test_regex_pre_tokenizer_splits():
    tokenizer = RegexPreTokenizer()
    text = "Don't say you'll leave! I'm 100% sure: pi=3.14."
    chunks = tokenizer.split_text(text)
    
    # Verify contractions and words
    assert "Don" in chunks
    assert "'t" in chunks
    assert " you" in chunks
    assert "'ll" in chunks
    assert " I" in chunks
    assert "'m" in chunks
    # Numbers should be grouped up to 3 digits and separated from whitespace
    assert "100" in chunks
    assert "%" in chunks
    assert " " in chunks


def test_regex_pre_tokenizer_byte_tuples():
    tokenizer = RegexPreTokenizer()
    text = "Hello world"
    tuples = tokenizer.pre_tokenize_to_byte_tuples(text)
    assert len(tuples) == 2
    # First chunk is "Hello"
    assert "".join(tuples[0]) == "Hello"
    # Second chunk is " world" with leading space mapped
    assert len(tuples[1]) == 6


def test_regex_pre_tokenizer_whitespace_preservation():
    tokenizer = RegexPreTokenizer()
    code = "def foo():\n    return 42\n"
    chunks = tokenizer.split_text(code)
    reconstructed = "".join(chunks)
    assert reconstructed == code, "Pre-tokenizer must preserve all code indentations and newlines"
