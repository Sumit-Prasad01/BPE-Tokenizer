"""Unit tests for text preprocessing and corpus acquisition utilities."""

import pytest
from src.config import PreprocessingConfig
from src.data.preprocessor import TextPreprocessor


def test_preprocessor_casing_preservation():
    preprocessor = TextPreprocessor()
    text = "def MyClassPascalCase(): return SNAKE_CASE_VARIABLE"
    cleaned = preprocessor.clean_text(text)
    assert cleaned == text, "Preprocessor must strictly preserve casing"


def test_preprocessor_whitespace_and_indentation():
    preprocessor = TextPreprocessor()
    code = "def foo():\n    if True:\n        return 42"
    cleaned = preprocessor.clean_text(code)
    assert cleaned == code, "Preprocessor must preserve indentation"


def test_preprocessor_control_characters():
    preprocessor = TextPreprocessor()
    dirty = "Hello\x00World\x07! Tab\there. Newline\nthere."
    expected = "HelloWorld! Tab\there. Newline\nthere."
    cleaned = preprocessor.clean_text(dirty)
    assert cleaned == expected, "Preprocessor must strip null/bell bytes but keep tabs and newlines"


def test_preprocessor_symbols_numbers_urls():
    preprocessor = TextPreprocessor()
    text = "Visit https://huggingface.co/datasets?q=fineweb with pi=3.14159 & x != y -> True"
    cleaned = preprocessor.clean_text(text)
    assert cleaned == text, "Preprocessor must preserve URLs, numbers, math, and symbols"


def test_preprocessor_min_length_filter():
    config = PreprocessingConfig(min_document_length_chars=30)
    preprocessor = TextPreprocessor(config)
    assert not preprocessor.is_valid_document("Short")
    assert preprocessor.is_valid_document("This is a valid long document that exceeds thirty characters.")
