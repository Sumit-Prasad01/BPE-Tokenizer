import sys
from setuptools import setup, find_packages, Extension

compile_args = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"]

ext_modules = [
    Extension(
        "src.csrc.fast_bpe",
        sources=["src/csrc/bpe_kernel.cpp"],
        extra_compile_args=compile_args,
        language="c++",
    )
]

setup(
    name="bpe_tokenizer",
    version="0.1.0",
    description="A general-purpose, GPT-style Byte-Level BPE Tokenizer with experiment tracking and multi-domain benchmarking",
    author="LLM Engineering",
    packages=find_packages(include=["src*", "utils*"]),
    ext_modules=ext_modules,
    python_requires=">=3.10",
    install_requires=[
        "datasets>=2.18.0",
        "huggingface_hub>=0.21.0",
        "regex>=2023.12.25",
        "pyyaml>=6.0.1",
        "pydantic>=2.6.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "colorlog>=6.8.0",
        "rich>=13.7.0",
        "tqdm>=4.66.0",
        "tokenizers>=0.15.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
        ]
    },
)