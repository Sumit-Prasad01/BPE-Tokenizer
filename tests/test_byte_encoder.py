"""Unit tests for the reversible byte-to-unicode bijection."""

from src.tokenizer.byte_encoder import (
    bytes_to_unicode,
    unicode_to_bytes,
    encode_bytes_to_string,
    decode_string_to_bytes,
)


def test_byte_encoder_bijection_size():
    byte_map = bytes_to_unicode()
    inv_map = unicode_to_bytes()

    assert len(byte_map) == 256, "Must map exactly 256 bytes"
    assert len(inv_map) == 256, "Inverted mapping must have 256 entries"
    # All mapped characters must be unique
    assert len(set(byte_map.values())) == 256, "All 256 mapped characters must be distinct"


def test_byte_encoder_roundtrip_all_bytes():
    # Test all 256 individual bytes
    all_bytes = bytes(range(256))
    encoded = encode_bytes_to_string(all_bytes)
    recovered = decode_string_to_bytes(encoded)
    assert recovered == all_bytes, "All 256 raw bytes must be recovered losslessly"


def test_byte_encoder_roundtrip_utf8_text():
    samples = [
        "Hello, World!",
        "def compute_fibonacci(n: int) -> int: return 1 if n <= 2 else ...",
        "Unicode test: 🚀 🤖 🔥 🐍 🌟",
        "Multilingual: 你好世界, Bonjour le monde, नमस्ते दुनिया, مرحبا بالعالم",
        "Special chars: \t\n\r !@#$%^&*()_+-=[]{}|;:'\",.<>/?\\",
    ]
    for sample in samples:
        raw = sample.encode("utf-8")
        encoded_str = encode_bytes_to_string(raw)
        recovered_bytes = decode_string_to_bytes(encoded_str)
        assert recovered_bytes == raw
        assert recovered_bytes.decode("utf-8") == sample
