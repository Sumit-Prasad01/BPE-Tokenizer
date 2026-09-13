"""Serialization and deserialization for Byte-Level BPE tokenizers.

Supports:
1. Native GPT-2 format: vocab.json + merges.txt
2. Hugging Face standard format: tokenizer.json, tokenizer_config.json, special_tokens_map.json
"""

from pathlib import Path
from typing import Any
import json

from src.tokenizer.tokenizer import Tokenizer
from utils.custom_exception import SerializationError
from utils.helpers import ensure_dir, save_json, load_json
from utils.logger import logger


class TokenizerSerializer:
    """Handles saving and loading of trained BPE tokenizer models."""

    @staticmethod
    def save_pretrained(tokenizer: Tokenizer, output_dir: str | Path) -> dict[str, Path]:
        """Serializes tokenizer to disk in both native and Hugging Face compatible formats.
        
        Args:
            tokenizer: Tokenizer instance to serialize.
            output_dir: Destination directory.
            
        Returns:
            Dictionary of written file paths.
        """
        out_dir = Path(output_dir)
        ensure_dir(out_dir)

        logger.info(f"💾 Serializing tokenizer '{tokenizer.name}' to {out_dir.resolve()}...")

        paths: dict[str, Path] = {}

        # 1. Save vocab.json
        vocab_path = out_dir / "vocab.json"
        save_json(tokenizer.vocab, vocab_path, indent=2)
        paths["vocab"] = vocab_path

        # 2. Save merges.txt (standard GPT-2 format)
        merges_path = out_dir / "merges.txt"
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for a, b in tokenizer.merges:
                f.write(f"{a} {b}\n")
        paths["merges"] = merges_path

        # 3. Save special_tokens_map.json
        special_tokens_map_path = out_dir / "special_tokens_map.json"
        special_map = {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "bos_token": "<|bos|>",
        }
        save_json(special_map, special_tokens_map_path, indent=2)
        paths["special_tokens_map"] = special_tokens_map_path

        # 4. Save tokenizer_config.json
        config_path = out_dir / "tokenizer_config.json"
        tokenizer_config = {
            "tokenizer_class": "GPT2TokenizerFast",
            "model_type": "gpt2",
            "model_max_length": 2048,
            "eos_token": "<|endoftext|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "bos_token": "<|bos|>",
            "clean_up_tokenization_spaces": False,
            "name": tokenizer.name,
            "regex_pattern": getattr(tokenizer, "regex_pattern", None),
            "digit_mode": getattr(tokenizer, "digit_mode", "clustered"),
        }
        save_json(tokenizer_config, config_path, indent=2)
        paths["tokenizer_config"] = config_path

        # 5. Save Hugging Face tokenizer.json
        hf_tokenizer_path = out_dir / "tokenizer.json"
        hf_json = TokenizerSerializer._build_hf_tokenizer_json(tokenizer)
        save_json(hf_json, hf_tokenizer_path, indent=2)
        paths["tokenizer_json"] = hf_tokenizer_path

        logger.info(f"✅ Successfully exported all 5 tokenizer artifact files to {out_dir.resolve()}")
        return paths

    @staticmethod
    def _build_hf_tokenizer_json(tokenizer: Tokenizer) -> dict[str, Any]:
        """Constructs a standard Hugging Face Fast Tokenizer JSON schema."""
        added_tokens = []
        for token_str in tokenizer.special_tokens_list:
            if token_str in tokenizer.vocab:
                added_tokens.append({
                    "id": tokenizer.vocab[token_str],
                    "content": token_str,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                })

        merges_strings = [f"{a} {b}" for a, b in tokenizer.merges]

        return {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": None,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "vocab": tokenizer.vocab,
                "merges": merges_strings,
            },
        }

    @staticmethod
    def from_pretrained(model_dir: str | Path) -> Tokenizer:
        """Loads a Tokenizer from a saved directory.
        
        Reads vocab.json and merges.txt.
        """
        in_dir = Path(model_dir)
        vocab_path = in_dir / "vocab.json"
        merges_path = in_dir / "merges.txt"

        if not vocab_path.is_file():
            raise SerializationError(f"Cannot load tokenizer: missing {vocab_path.resolve()}")
        if not merges_path.is_file():
            raise SerializationError(f"Cannot load tokenizer: missing {merges_path.resolve()}")

        vocab = load_json(vocab_path)
        merges: list[tuple[str, str]] = []

        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        # Check for special tokens map
        special_map_path = in_dir / "special_tokens_map.json"
        special_tokens = None
        if special_map_path.is_file():
            s_map = load_json(special_map_path)
            special_tokens = list(s_map.values())

        # Check for tokenizer config (regex_pattern, digit_mode)
        regex_pattern = None
        digit_mode = "clustered"
        config_path = in_dir / "tokenizer_config.json"
        if config_path.is_file():
            cfg = load_json(config_path)
            regex_pattern = cfg.get("regex_pattern")
            digit_mode = cfg.get("digit_mode", "clustered")

        tokenizer = Tokenizer(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
            regex_pattern=regex_pattern,
            digit_mode=digit_mode,
            name=in_dir.name,
        )
        logger.info(
            f"📖 Loaded tokenizer from {in_dir.resolve()}: "
            f"Vocab size {tokenizer.vocab_size:,}, Merges {len(merges):,}"
        )
        return tokenizer
