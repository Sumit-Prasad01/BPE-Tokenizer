"""Unit and integration tests for the CLI commands in main.py."""

from pathlib import Path
import pytest

from main import build_parser, cmd_train, cmd_evaluate, cmd_visualize, cmd_compare_runs, cmd_push_to_hub
from src.tokenizer.serializer import TokenizerSerializer
from src.tokenizer.tokenizer import Tokenizer
from utils.helpers import dump_yaml


@pytest.fixture
def cli_sample_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    # 1. Synthetic training text
    corpus_file = tmp_path / "train.txt"
    corpus_file.write_text(
        "Hello world! This is a test corpus for BPE tokenization.\n"
        "Machine learning and natural language processing.\n" * 50,
        encoding="utf-8",
    )

    # 2. Tokenizer config
    config_file = tmp_path / "tok_config.yaml"
    cfg = {
        "tokenizer": {
            "name": "cli_test_bpe",
            "model_type": "byte_level_bpe",
            "vocab_size": 300,
            "min_frequency": 2,
            "byte_level": True,
            "regex_pattern": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            "special_tokens": {
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "bos_token": "<|bos|>",
            },
        }
    }
    dump_yaml(cfg, config_file)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return corpus_file, config_file, runs_dir


def test_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["prepare-data", "--config", "configs/dataset_mix.yaml"])
    assert args.command == "prepare-data"
    assert args.config == "configs/dataset_mix.yaml"

    args = parser.parse_args(["train", "--config", "configs/tokenizer_32k.yaml", "--corpus", "test.txt", "--vocab-size", "1000"])
    assert args.command == "train"
    assert args.vocab_size == 1000


def test_cli_train_and_evaluate_e2e(cli_sample_data: tuple[Path, Path, Path]):
    corpus_file, config_file, runs_dir = cli_sample_data
    parser = build_parser()

    # 1. Run train
    train_args = parser.parse_args([
        "train",
        "--config", str(config_file),
        "--corpus", str(corpus_file),
        "--vocab-size", "280",
        "--run-name", "cli_test_model",
    ])
    ret = cmd_train(train_args)
    assert ret == 0

    # Locate generated run dir
    exp_runs = Path("experiments/runs")
    created_runs = sorted(exp_runs.glob("*cli_test_model*"))
    assert len(created_runs) > 0
    latest_run = created_runs[-1]

    # 2. Run evaluate
    eval_args = parser.parse_args([
        "evaluate",
        "--model", str(latest_run),
    ])
    ret_eval = cmd_evaluate(eval_args)
    assert ret_eval == 0
    assert (latest_run / "metrics.json").is_file()

    # 3. Run visualize
    vis_args = parser.parse_args([
        "visualize",
        "--run", str(latest_run),
    ])
    ret_vis = cmd_visualize(vis_args)
    assert ret_vis == 0
    assert (latest_run / "visuals").is_dir()

    # 4. Run compare-runs
    comp_args = parser.parse_args([
        "compare-runs",
    ])
    ret_comp = cmd_compare_runs(comp_args)
    assert ret_comp == 0

    # 5. Run push-to-hub dry-run
    hub_args = parser.parse_args([
        "push-to-hub",
        "--model", str(latest_run),
        "--repo-id", "testuser/cli-test-bpe",
        "--dry-run",
    ])
    ret_hub = cmd_push_to_hub(hub_args)
    assert ret_hub == 0

    # 6. Run test-real-world
    from main import cmd_test_real_world, cmd_benchmark_inference
    stress_args = parser.parse_args([
        "test-real-world",
        "--model", str(latest_run),
    ])
    ret_stress = cmd_test_real_world(stress_args)
    assert ret_stress == 0

    # 7. Run benchmark-inference (python mode on train corpus)
    bench_args = parser.parse_args([
        "benchmark-inference",
        "--model", str(latest_run),
        "--corpus", str(corpus_file),
        "--mode", "python",
    ])
    ret_bench = cmd_benchmark_inference(bench_args)
    assert ret_bench == 0
