"""Text cleaning and normalization pipeline that preserves casing, code indentation, symbols, numbers, and URLs."""

import re
from src.config import PreprocessingConfig


class TextPreprocessor:
    """Preprocesses text documents while preserving critical structural and linguistic characteristics."""

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        
        # Regex to strip null bytes and unwanted control characters, but PRESERVE tabs (\t), newlines (\n), and \r
        self._control_char_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
        # Regex to collapse excessive consecutive blank lines (>2) to a clean double newline
        self._excessive_newlines = re.compile(r"\n{3,}")

    def clean_text(self, text: str) -> str:
        """Cleans a single document string.
        
        Rules:
        - Never lowercases (preserves casing for proper nouns, acronyms, code identifiers).
        - Preserves whitespace and indentation (spaces, tabs, newlines).
        - Preserves punctuation, math operators, symbols, numbers, and URLs.
        - Strips null bytes and non-printable control characters.
        - Collapses 3+ consecutive newlines to 2.
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove null bytes and control chars
        if self.config.remove_null_bytes:
            text = self._control_char_pattern.sub("", text)

        # Normalize carriage returns
        if self.config.normalize_newlines:
            text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse excessive newlines
        text = self._excessive_newlines.sub("\n\n", text)

        return text.strip()

    def is_valid_document(self, text: str) -> bool:
        """Validates whether a document meets minimum quality criteria."""
        if not text:
            return False
        if len(text.strip()) < self.config.min_document_length_chars:
            return False
        return True
