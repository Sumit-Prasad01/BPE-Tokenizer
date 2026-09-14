"""Automated build script for the Native C++ BPE Inference Engine.

Compiles:
1. `src/csrc/bpe_engine.dll` (C-ABI Shared Library for in-process foreign bindings)
2. `bpe_engine.exe` (Standalone native executable for zero-Python production serving)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_compiler() -> tuple[str, str]:
    """Detect available C++ compiler."""
    # Check g++ (MinGW / GCC)
    gpp = shutil.which("g++")
    if gpp:
        return "gcc", gpp

    # Check clang++
    clang = shutil.which("clang++")
    if clang:
        return "clang", clang

    # Check cl.exe (MSVC)
    cl = shutil.which("cl")
    if cl:
        return "msvc", cl

    raise RuntimeError("No C++ compiler found (checked g++, clang++, cl.exe). Please install MinGW or MSVC.")


def build_native_engine() -> bool:
    project_root = Path(__file__).resolve().parent.parent.parent
    csrc_dir = project_root / "src" / "csrc"

    engine_hpp = csrc_dir / "bpe_engine.hpp"
    engine_cpp = csrc_dir / "bpe_engine.cpp"
    cli_cpp = csrc_dir / "bpe_cli.cpp"
    dll_out = csrc_dir / "bpe_engine.dll"
    exe_out = project_root / "bpe_engine.exe"

    if not engine_hpp.exists() or not engine_cpp.exists():
        print(f"[build_engine] Error: Missing source files in {csrc_dir}")
        return False

    comp_type, comp_bin = find_compiler()
    print(f"[build_engine] Detected compiler: {comp_type} ({comp_bin})")

    # 1. Compile Shared Library (bpe_engine.dll)
    print(f"[build_engine] Compiling shared library: {dll_out.name} ...")
    if comp_type in ("gcc", "clang"):
        cmd_dll = [
            comp_bin,
            "-O3",
            "-shared",
            "-std=c++14",
            str(engine_cpp),
            "-o",
            str(dll_out),
            f"-Wl,--out-implib,{csrc_dir / 'libbpe_engine.a'}",
        ]
    else:  # msvc
        cmd_dll = [
            comp_bin,
            "/O2",
            "/LD",
            "/std:c++17",
            str(engine_cpp),
            f"/Fe:{dll_out}",
        ]

    res_dll = subprocess.run(cmd_dll, capture_output=True, text=True, cwd=str(project_root))
    if res_dll.returncode != 0:
        print(f"[build_engine] DLL compilation failed:\n{res_dll.stderr}")
        return False
    print(f"[build_engine] Successfully built {dll_out} ({dll_out.stat().st_size:,} bytes)")

    # 2. Compile Standalone CLI Executable (bpe_engine.exe)
    print(f"[build_engine] Compiling standalone executable: {exe_out.name} ...")
    if comp_type in ("gcc", "clang"):
        cmd_exe = [
            comp_bin,
            "-O3",
            "-std=c++14",
            str(cli_cpp),
            str(engine_cpp),
            f"-I{csrc_dir}",
            "-o",
            str(exe_out),
        ]
    else:  # msvc
        cmd_exe = [
            comp_bin,
            "/O2",
            "/std:c++17",
            str(cli_cpp),
            str(engine_cpp),
            f"/I{csrc_dir}",
            f"/Fe:{exe_out}",
        ]

    res_exe = subprocess.run(cmd_exe, capture_output=True, text=True, cwd=str(project_root))
    if res_exe.returncode != 0:
        print(f"[build_engine] Executable compilation failed:\n{res_exe.stderr}")
        return False
    print(f"[build_engine] Successfully built {exe_out} ({exe_out.stat().st_size:,} bytes)")

    # 3. Quick sanity check
    print("[build_engine] Running sanity test on bpe_engine.exe ...")
    test_run = subprocess.run(
        [str(exe_out), "--help"],
        capture_output=True,
        text=True,
        cwd=str(project_root)
    )
    if test_run.returncode == 0 and "High-Performance Native C++" in test_run.stdout:
        print("[build_engine] Sanity check passed! Engine is fully operational.")
        return True
    else:
        print(f"[build_engine] Sanity check failed:\n{test_run.stdout}\n{test_run.stderr}")
        return False


if __name__ == "__main__":
    success = build_native_engine()
    sys.exit(0 if success else 1)
