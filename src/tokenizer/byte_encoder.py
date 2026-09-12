"""Reversible byte-to-unicode bijection following the GPT-2 / GPT-4 standard.

Every byte in range(256) is mapped to a distinct, printable Unicode character.
This ensures that any arbitrary sequence of UTF-8 or raw bytes can be processed
as a regular Unicode string with regex pattern matching without control-character
mangling, and subsequently decoded with 100% lossless fidelity.
"""

from functools import lru_cache


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """Returns a dictionary mapping every byte (0..255) to a unique Unicode character.
    
    Standard printable ASCII and Latin-1 characters map to themselves.
    Non-printable and whitespace characters (e.g. 0-32, 127-160, 173) are shifted
    into the private/extended Unicode code points (starting at 256) so they remain
    distinct, printable, and regular-expression friendly.
    """
    # Printable ASCII and Latin-1 supplement ranges:
    # - 33 ('!') to 126 ('~')
    # - 161 ('¡') to 172 ('¬')
    # - 174 ('®') to 255 ('ÿ')
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # Map remaining 256 - len(bs) = 68 bytes to code points starting at 256 (0x0100)
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    characters = [chr(c) for c in cs]
    return dict(zip(bs, characters))


@lru_cache()
def unicode_to_bytes() -> dict[str, int]:
    """Returns the inverted mapping from unique Unicode characters back to raw byte values."""
    byte_map = bytes_to_unicode()
    return {v: k for k, v in byte_map.items()}


def encode_bytes_to_string(raw_bytes: bytes) -> str:
    """Converts raw bytes to mapped Unicode string."""
    byte_map = bytes_to_unicode()
    return "".join(byte_map[b] for b in raw_bytes)


def decode_string_to_bytes(encoded_str: str) -> bytes:
    """Converts a mapped Unicode string back to raw bytes."""
    inv_map = unicode_to_bytes()
    return bytes(inv_map[c] for c in encoded_str)
