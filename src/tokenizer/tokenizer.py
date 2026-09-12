"""Production-grade Tokenizer inference engine for Byte-Level BPE."""

from functools import lru_cache
from pathlib import Path
from typing import Any
import regex as re

from src.tokenizer.byte_encoder import bytes_to_unicode, unicode_to_bytes
from src.tokenizer.pre_tokenizer import RegexPreTokenizer, DEFAULT_GPT4_REGEX
from utils.custom_exception import TokenizerInferenceError
from utils.logger import logger


class Tokenizer:
    """Byte-Level BPE Tokenizer inference engine."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        special_tokens: list[str] | None = None,
        regex_pattern: str | None = None,
        name: str = "bpe_tokenizer",
    ):
        self.name = name
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens_list = special_tokens or [
            "<|endoftext|>",
            "<|pad|>",
            "<|unk|>",
            "<|bos|>",
            "<|eos|>",
        ]
        self.regex_pattern = regex_pattern or DEFAULT_GPT4_REGEX

        # Fast lookup tables
        self.decoder = {v: k for k, v in self.vocab.items()}
        self.bpe_ranks: dict[tuple[str, str], int] = {
            merge: i for i, merge in enumerate(self.merges)
        }

        # Byte bijection mappings
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = unicode_to_bytes()

        # Regex pre-tokenizer
        self.pre_tokenizer = RegexPreTokenizer(self.regex_pattern)

        # Special tokens setup
        self.special_tokens_set = set(self.special_tokens_list)
        self._build_special_tokens_regex()

        # Internal word BPE cache for speed
        self._cache: dict[str, tuple[str, ...]] = {}

    def _build_special_tokens_regex(self) -> None:
        """Builds a regex pattern to safely identify special tokens."""
        if self.special_tokens_list:
            escaped = [re.escape(tok) for tok in self.special_tokens_list]
            self.special_pattern = re.compile(f"({'|'.join(escaped)})")
        else:
            self.special_pattern = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int | None:
        return self.vocab.get("<|pad|>")

    @property
    def eos_token_id(self) -> int | None:
        return self.vocab.get("<|endoftext|>")

    def _get_pairs(self, word: tuple[str, ...]) -> set[tuple[str, str]]:
        """Returns the set of adjacent symbol pairs in a word tuple."""
        return set(zip(word, word[1:]))

    def _bpe(self, token_str: str) -> tuple[str, ...]:
        """Applies greedy BPE merge rules to a mapped byte-string chunk."""
        if token_str in self._cache:
            return self._cache[token_str]

        word: tuple[str, ...] = tuple(token_str)
        pairs = self._get_pairs(word)

        if not pairs:
            return (token_str,)

        while True:
            # Find candidate pairs present in our learned merge rules
            min_rank = float("inf")
            best_pair = None

            for pair in pairs:
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            # If no more pairs can be merged, break
            if best_pair is None:
                break

            first, second = best_pair
            new_word: list[str] = []
            i = 0
            n = len(word)

            while i < n:
                if i < n - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        self._cache[token_str] = word
        return word

    def encode(
        self,
        text: str,
        allowed_special: set[str] | str | None = None,
    ) -> list[int]:
        """Encodes raw text into a list of integer token IDs.
        
        Args:
            text: Input string to encode.
            allowed_special: 'all', None, or a set of allowed special token strings.
                             If a special token is observed and not allowed, raises error.
                             
        Returns:
            List of integer token IDs.
        """
        if not text:
            return []

        # Determine allowed special tokens
        if allowed_special == "all":
            allowed_set = self.special_tokens_set
        elif isinstance(allowed_special, (set, list)):
            allowed_set = set(allowed_special)
        else:
            allowed_set = set()

        token_ids: list[int] = []

        # Handle special tokens via splitting
        if self.special_pattern and allowed_set:
            parts = self.special_pattern.split(text)
            for part in parts:
                if not part:
                    continue
                if part in self.special_tokens_set:
                    if part in allowed_set:
                        token_ids.append(self.vocab[part])
                    else:
                        raise TokenizerInferenceError(
                            f"Encountered disallowed special token in text: '{part}'"
                        )
                else:
                    self._encode_ordinary_text(part, token_ids)
        else:
            # Check for disallowed special tokens if text contains them
            if self.special_pattern and not allowed_set:
                matched = self.special_pattern.search(text)
                if matched:
                    raise TokenizerInferenceError(
                        f"Encountered special token '{matched.group()}' but allowed_special is empty. "
                        "Set allowed_special='all' to allow special tokens."
                    )
            self._encode_ordinary_text(text, token_ids)

        return token_ids

    def _encode_ordinary_text(self, text: str, token_ids: list[int]) -> None:
        """Pre-tokenizes and encodes a non-special text segment."""
        # Find regex matches
        matches = self.pre_tokenizer.split_text(text)
        for match in matches:
            # Convert UTF-8 bytes to mapped Unicode string
            raw_bytes = match.encode("utf-8")
            mapped_str = "".join(self.byte_encoder[b] for b in raw_bytes)
            # Apply BPE merges
            bpe_subwords = self._bpe(mapped_str)
            for subword in bpe_subwords:
                token_ids.append(self.vocab[subword])

    def decode(self, tokens: list[int], errors: str = "replace") -> str:
        """Decodes a sequence of integer token IDs back to the original string.
        
        Guarantees 100% losslessness: decode(encode(text)) == text.
        """
        if not tokens:
            return ""

        result_chunks: list[str] = []
        byte_chars: list[int] = []

        for token_id in tokens:
            token_str = self.decoder.get(token_id)
            if token_str is None:
                raise TokenizerInferenceError(f"Unknown token ID: {token_id}")

            if token_str in self.special_tokens_set:
                if byte_chars:
                    result_chunks.append(bytes(byte_chars).decode("utf-8", errors=errors))
                    byte_chars = []
                result_chunks.append(token_str)
            else:
                for char in token_str:
                    byte_chars.append(self.byte_decoder[char])

        if byte_chars:
            result_chunks.append(bytes(byte_chars).decode("utf-8", errors=errors))

        return "".join(result_chunks)

    def tokenize(self, text: str, allowed_special: set[str] | str | None = None) -> list[str]:
        """Returns the list of subword strings for a given text."""
        ids = self.encode(text, allowed_special=allowed_special)
        return [self.decoder[i] for i in ids]

    def encode_batch(
        self,
        texts: list[str],
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
        pad_token_id: int | None = None,
        allowed_special: set[str] | str | None = None,
    ) -> list[list[int]]:
        """Encodes a batch of strings with optional padding and truncation."""
        batch_ids = [self.encode(text, allowed_special=allowed_special) for text in texts]

        if truncation and max_length is not None:
            batch_ids = [seq[:max_length] for seq in batch_ids]

        if padding:
            target_pad = pad_token_id if pad_token_id is not None else self.pad_token_id
            if target_pad is None:
                raise TokenizerInferenceError("Padding requested but no pad_token_id defined.")
            
            target_len = max_length if max_length is not None else max(len(s) for s in batch_ids)
            padded = []
            for seq in batch_ids:
                if len(seq) < target_len:
                    seq = seq + [target_pad] * (target_len - len(seq))
                padded.append(seq)
            return padded

        return batch_ids

    def decode_batch(self, batch_tokens: list[list[int]], errors: str = "replace") -> list[str]:
        """Decodes a batch of token ID sequences."""
        return [self.decode(tokens, errors=errors) for tokens in batch_tokens]

    def save_pretrained(self, output_dir: str | Path) -> dict[str, Path]:
        """Saves tokenizer artifacts to directory."""
        from src.tokenizer.serializer import TokenizerSerializer
        return TokenizerSerializer.save_pretrained(self, output_dir)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Tokenizer":
        """Loads tokenizer from saved directory."""
        from src.tokenizer.serializer import TokenizerSerializer
        return TokenizerSerializer.from_pretrained(model_dir)
