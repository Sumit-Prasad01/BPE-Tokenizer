"""Integration tests for the Native C++ BPE Engine and standalone CLI."""

import subprocess
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def bpe_binaries():
    """Ensure the native C++ engine DLL and CLI binary exist."""
    project_root = Path(__file__).resolve().parent.parent
    exe_path = project_root / "bpe_engine.exe"
    dll_path = project_root / "src" / "csrc" / "bpe_engine.dll"

    if not exe_path.exists() or not dll_path.exists():
        from src.csrc.build import build_native_engine
        assert build_native_engine(), "Failed to build C++ native binaries"

    assert exe_path.exists()
    assert dll_path.exists()
    return exe_path, dll_path


@pytest.fixture(scope="module")
def model_paths():
    """Locate the verified 64k or 32k experiment run."""
    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "experiments" / "runs" / "20260914_023959_exp_vocab_64k"
    if not run_dir.exists():
        # Fallback to any run folder
        runs = list((project_root / "experiments" / "runs").glob("*_exp_*"))
        if runs:
            run_dir = runs[0]
        else:
            run_dir = project_root / "experiments" / "runs" / "20260913_023301_general_purpose_bpe_32k"

    vocab_path = run_dir / "vocab.json"
    merges_path = run_dir / "merges.txt"
    assert vocab_path.exists()
    assert merges_path.exists()
    return vocab_path, merges_path


def test_native_cpp_cli_help(bpe_binaries):
    """Test standalone --help output."""
    exe_path, _ = bpe_binaries
    res = subprocess.run([str(exe_path), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "High-Performance Native C++" in res.stdout


def test_native_cpp_encode_decode_roundtrip(bpe_binaries, model_paths):
    """Test C++ encoding, decoding, and 100% losslessness."""
    exe_path, _ = bpe_binaries
    vocab_path, merges_path = model_paths

    text = "Hello, World! Native C++ BPE is 100% lossless."
    cmd_enc = [
        str(exe_path),
        "--vocab", str(vocab_path),
        "--merges", str(merges_path),
        "--encode", text,
    ]
    res_enc = subprocess.run(cmd_enc, capture_output=True, text=True)
    assert res_enc.returncode == 0
    assert "Tokens (" in res_enc.stdout
    assert "Compression Ratio:" in res_enc.stdout

    # Extract tokens line: e.g. Tokens (12): [123, 456, 789]
    for line in res_enc.stdout.splitlines():
        if line.startswith("Tokens ("):
            tok_str = line.split("[")[1].split("]")[0]
            break

    # Now decode back
    cmd_dec = [
        str(exe_path),
        "--vocab", str(vocab_path),
        "--merges", str(merges_path),
        "--decode", tok_str,
    ]
    res_dec = subprocess.run(cmd_dec, capture_output=True, text=True)
    assert res_dec.returncode == 0
    assert f"Decoded ({len(text)} bytes): {text}" in res_dec.stdout


def test_native_cpp_streaming_decoder(bpe_binaries, model_paths):
    """Test token-by-token incremental streaming decoder."""
    exe_path, _ = bpe_binaries
    vocab_path, merges_path = model_paths

    stream_text = "Testing streaming tokens: 🚀 12345"
    cmd_stream = [
        str(exe_path),
        "--vocab", str(vocab_path),
        "--merges", str(merges_path),
        "--stream", stream_text,
    ]
    res_stream = subprocess.run(cmd_stream, capture_output=True, text=True)
    assert res_stream.returncode == 0
    assert "Stream completed successfully." in res_stream.stdout
