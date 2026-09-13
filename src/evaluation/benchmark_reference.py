"""Standardized benchmark runner for industry reference tokenizers (GPT-2, GPT-4, LLaMA-3)."""

import argparse
import sys
from pathlib import Path
from typing import Any

from src.evaluation.benchmarks import BenchmarkRunner
from src.evaluation.domain_slices import get_standard_domain_slices
from utils.helpers import save_json
from utils.logger import logger


class GenericTokenizerWrapper:
    """Standardizes external tokenizers to the Tokenizer interface used by BenchmarkRunner."""

    def __init__(self, name: str, encode_fn, decode_fn, vocab_size: int, vocab_dict: dict | None = None):
        self.name = name
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn
        self._vocab_size = vocab_size
        self.vocab = vocab_dict or {}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, **kwargs) -> list[int]:
        return self._encode_fn(text)

    def decode(self, tokens: list[int], **kwargs) -> str:
        return self._decode_fn(tokens)


def load_hf_tokenizer(model_id: str) -> GenericTokenizerWrapper:
    """Loads a tokenizer using huggingface tokenizers library (fast rust backend)."""
    try:
        from tokenizers import Tokenizer as FastTokenizer
        tok = FastTokenizer.from_pretrained(model_id)
        vocab = tok.get_vocab()
        return GenericTokenizerWrapper(
            name=model_id,
            encode_fn=lambda t: tok.encode(t).ids,
            decode_fn=lambda ids: tok.decode(ids),
            vocab_size=tok.get_vocab_size(),
            vocab_dict=vocab,
        )
    except Exception as e:
        logger.error(f"Failed to load HuggingFace tokenizer '{model_id}': {e}")
        raise


def load_tiktoken_tokenizer(encoding_name: str = "cl100k_base") -> GenericTokenizerWrapper:
    """Loads an OpenAI tiktoken encoding."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return GenericTokenizerWrapper(
            name=f"tiktoken_{encoding_name}",
            encode_fn=lambda t: enc.encode(t, allowed_special="all"),
            decode_fn=lambda ids: enc.decode(ids),
            vocab_size=enc.n_vocab,
            vocab_dict={},
        )
    except ImportError:
        logger.warning(f"tiktoken is not installed. Cannot load '{encoding_name}'.")
        raise


def benchmark_tokenizer(wrapper: GenericTokenizerWrapper, custom_corpus_path: str | None = None) -> dict[str, Any]:
    """Runs standard BenchmarkRunner across all domains on external tokenizer."""
    runner = BenchmarkRunner(wrapper)  # type: ignore
    return runner.run_benchmark(custom_corpus_path=custom_corpus_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark reference tokenizers on standardized evaluation suite")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model ID or tiktoken encoding name")
    parser.add_argument("--backend", choices=["hf", "tiktoken"], default="hf", help="Backend library")
    parser.add_argument("--corpus", default="data/processed/eval_corpus_heldout.txt", help="Held-out eval corpus path")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.backend == "hf":
        tok_wrapper = load_hf_tokenizer(args.model)
    elif args.backend == "tiktoken":
        tok_wrapper = load_tiktoken_tokenizer(args.model)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    results = benchmark_tokenizer(tok_wrapper, custom_corpus_path=args.corpus)
    out_path = Path(args.output) if args.output else Path(f"experiments/reference_{tok_wrapper.name.replace('/', '_')}.json")
    save_json(results, out_path, indent=2)
    logger.info(f"Saved benchmark results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
