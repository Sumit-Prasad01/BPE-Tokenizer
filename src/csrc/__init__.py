"""C++ High-Performance BPE acceleration module."""

try:
    from src.csrc.fast_bpe import BPEKernel
    HAS_FAST_BPE = True
except ImportError:
    HAS_FAST_BPE = False
    BPEKernel = None

__all__ = ["BPEKernel", "HAS_FAST_BPE"]
