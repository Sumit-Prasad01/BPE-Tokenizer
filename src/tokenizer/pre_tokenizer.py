"""Regex-based pre-tokenization for GPT-style models with byte-level conversion."""

from collections import Counter
from pathlib import Path
from typing import Iterator
import regex as re
from tqdm import tqdm

from src.tokenizer.byte_encoder import bytes_to_unicode, encode_bytes_to_string
from utils.logger import logger


DEFAULT_GPT4_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class RegexPreTokenizer:
    """Pre-tokenizes text using Unicode regular expressions and maps to byte tokens."""

    def __init__(self, pattern: str | None = None):
        self.pattern_str = pattern or DEFAULT_GPT4_REGEX
        self.compiled_regex = re.compile(self.pattern_str)
        self.byte_encoder = bytes_to_unicode()

    def split_text(self, text: str) -> list[str]:
        """Splits text into chunks according to the regex pattern."""
        return self.compiled_regex.findall(text)

    def pre_tokenize_to_byte_tuples(self, text: str) -> list[tuple[str, ...]]:
        """Splits text and converts each chunk into a tuple of mapped byte characters.
        
        Example:
            " Hello" -> ('Ġ', 'H', 'e', 'l', 'l', 'o')
        """
        chunks = self.split_text(text)
        tuples = []
        for chunk in chunks:
            raw_bytes = chunk.encode("utf-8")
            mapped_symbols = tuple(self.byte_encoder[b] for b in raw_bytes)
            tuples.append(mapped_symbols)
        return tuples

    def count_word_frequencies(
        self,
        file_path: str | Path,
        max_lines: int | None = None,
        batch_size: int = 20000,
    ) -> Counter[tuple[str, ...]]:
        """Streams a text file in batches, pre-tokenizes, and aggregates unique word frequencies.
        
        Returns:
            Counter mapping tuple of byte symbols to total occurrences.
        """
        path = Path(file_path)
        logger.info(f"🔍 Pre-tokenizing corpus from: {path.resolve()}")
        
        word_counts: Counter[tuple[str, ...]] = Counter()
        batch_lines: list[str] = []
        line_idx = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                batch_lines.append(line)
                line_idx += 1

                if max_lines and line_idx >= max_lines:
                    break

                if len(batch_lines) >= batch_size:
                    self._process_line_batch(batch_lines, word_counts)
                    batch_lines = []

            if batch_lines:
                self._process_line_batch(batch_lines, word_counts)

        logger.info(
            f"✨ Pre-tokenization complete: {line_idx:,} lines processed, "
            f"{len(word_counts):,} unique word patterns identified."
        )
        return word_counts

    def _process_line_batch(
        self,
        lines: list[str],
        counter: Counter[tuple[str, ...]],
    ) -> None:
        """Processes a batch of lines and updates word frequency counts."""
        full_text = "".join(lines)
        matches = self.compiled_regex.findall(full_text)
        for match in matches:
            raw_bytes = match.encode("utf-8")
            symbols = tuple(self.byte_encoder[b] for b in raw_bytes)
            counter[symbols] += 1
