"""Evaluation metrics engine for tokenizer benchmarking."""

import time
from typing import Any
from src.tokenizer.tokenizer import Tokenizer
from utils.logger import logger


def compute_compression_ratio(raw_text: str, token_ids: list[int]) -> float:
    """Computes Compression Ratio = (Total UTF-8 Bytes) / (Number of Tokens).
    
    Higher is better (more bytes packed per token).
    """
    if not token_ids:
        return 0.0
    num_bytes = len(raw_text.encode("utf-8"))
    return round(num_bytes / len(token_ids), 4)


def compute_fertility(raw_text: str, token_ids: list[int]) -> float:
    """Computes Token Fertility = (Number of Tokens) / (Number of Whitespace Words).
    
    Lower is better (fewer subword fractures per linguistic word).
    """
    words = raw_text.split()
    if not words:
        return 0.0
    return round(len(token_ids) / len(words), 4)


def compute_vocab_utilization(token_ids: list[int], total_vocab_size: int) -> float:
    """Computes the percentage of vocabulary tokens activated on this evaluation text."""
    if total_vocab_size <= 0 or not token_ids:
        return 0.0
    unique_tokens = len(set(token_ids))
    return round((unique_tokens / total_vocab_size) * 100.0, 2)


def compute_unk_rate(token_ids: list[int], unk_token_id: int | None) -> float:
    """Computes percentage of unknown tokens. For Byte-Level BPE, this must be 0.00%."""
    if not token_ids or unk_token_id is None:
        return 0.0
    unk_count = sum(1 for t in token_ids if t == unk_token_id)
    return round((unk_count / len(token_ids)) * 100.0, 4)


def verify_losslessness(raw_text: str, decoded_text: str) -> dict[str, Any]:
    """Verifies 100% roundtrip fidelity: decode(encode(text)) == text."""
    is_exact_match = (raw_text == decoded_text)
    first_diff = None
    if not is_exact_match:
        min_len = min(len(raw_text), len(decoded_text))
        for idx in range(min_len):
            if raw_text[idx] != decoded_text[idx]:
                first_diff = idx
                break
        if first_diff is None:
            first_diff = min_len

    return {
        "is_lossless": is_exact_match,
        "raw_char_count": len(raw_text),
        "decoded_char_count": len(decoded_text),
        "raw_byte_count": len(raw_text.encode("utf-8")),
        "first_difference_index": first_diff,
    }


def measure_throughput(
    tokenizer: Tokenizer,
    text: str,
    repeats: int = 3,
) -> dict[str, float]:
    """Measures encoding and decoding throughput in MB/s and Tokens/s."""
    raw_bytes = len(text.encode("utf-8"))
    raw_mb = raw_bytes / (1024 * 1024)

    # Warmup
    tokens = tokenizer.encode(text)
    _ = tokenizer.decode(tokens)
    num_tokens = len(tokens)

    # Encode benchmark
    encode_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = tokenizer.encode(text)
        encode_times.append(time.perf_counter() - t0)
    avg_encode_time = sum(encode_times) / len(encode_times)

    # Decode benchmark
    decode_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = tokenizer.decode(tokens)
        decode_times.append(time.perf_counter() - t0)
    avg_decode_time = sum(decode_times) / len(decode_times)

    encode_mb_sec = raw_mb / avg_encode_time if avg_encode_time > 0 else 0.0
    encode_tokens_sec = num_tokens / avg_encode_time if avg_encode_time > 0 else 0.0
    decode_mb_sec = raw_mb / avg_decode_time if avg_decode_time > 0 else 0.0
    decode_tokens_sec = num_tokens / avg_decode_time if avg_decode_time > 0 else 0.0

    return {
        "encode_mb_per_sec": round(encode_mb_sec, 2),
        "encode_tokens_per_sec": round(encode_tokens_sec, 2),
        "decode_mb_per_sec": round(decode_mb_sec, 2),
        "decode_tokens_per_sec": round(decode_tokens_sec, 2),
    }


def evaluate_text_slice(
    tokenizer: Tokenizer,
    text: str,
    domain_name: str = "general",
) -> dict[str, Any]:
    """Evaluates all metrics for a given text slice."""
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    lossless_info = verify_losslessness(text, decoded)

    unk_id = tokenizer.vocab.get("<|unk|>")
    cr = compute_compression_ratio(text, tokens)
    fertility = compute_fertility(text, tokens)
    unk_rate = compute_unk_rate(tokens, unk_id)
    utilization = compute_vocab_utilization(tokens, tokenizer.vocab_size)

    return {
        "domain": domain_name,
        "byte_count": lossless_info["raw_byte_count"],
        "token_count": len(tokens),
        "word_count": len(text.split()),
        "compression_ratio": cr,
        "fertility": fertility,
        "unk_rate_percent": unk_rate,
        "vocab_utilization_percent": utilization,
        "is_lossless": lossless_info["is_lossless"],
    }
