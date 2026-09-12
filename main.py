"""CLI Entrypoint for the General-Purpose GPT-Style Byte-Level BPE Tokenizer project."""

import os
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from src.config import (
    load_base_config,
    load_dataset_mix_config,
    load_tokenizer_config,
    load_lm_eval_config,
)
from src.data.corpus_builder import CorpusBuilder
from src.evaluation.benchmarks import BenchmarkRunner
from src.evaluation.llm_benchmark import LMBenchmarkRunner
from src.evaluation.visualizer import generate_all_visualizations, plot_vocab_scaling_tradeoff
from src.hub.hf_publisher import HFPublisher
from src.tokenizer.bpe_trainer import BPETrainer
from src.tokenizer.tokenizer import Tokenizer
from src.tracker.experiment_tracker import ExperimentTracker
from utils.custom_exception import TokenizerBaseException
from utils.helpers import load_json, load_yaml, save_json
from utils.logger import logger

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Download and assemble training and evaluation corpora."""
    logger.info(f"📥 Loading dataset mix config from '{args.config}'...")
    config = load_dataset_mix_config(args.config)

    builder = CorpusBuilder(config=config)
    result = builder.build_all()

    logger.info(
        f"✅ Corpora successfully created!\n"
        f"  - Train: {result['train_file']} ({result['train_size_bytes']:,} bytes)\n"
        f"  - Eval:  {result['eval_file']} ({result['eval_size_bytes']:,} bytes)"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train Byte-Level BPE Tokenizer from corpus."""
    logger.info(f"⚙️ Loading tokenizer config from '{args.config}'...")
    tok_config = load_tokenizer_config(args.config)

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        logger.error(f"Training corpus not found: {corpus_path.resolve()}")
        return 1

    vocab_size = args.vocab_size or tok_config.vocab_size
    min_frequency = args.min_frequency or tok_config.min_frequency

    logger.info(f"🚀 Training BPE Tokenizer (Vocab: {vocab_size:,}, Min Freq: {min_frequency})...")
    trainer = BPETrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=tok_config.special_tokens.all_special_tokens,
        regex_pattern=tok_config.regex_pattern,
    )

    vocab, merges, stats = trainer.train_from_file(corpus_path, max_lines=args.max_lines)

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=tok_config.special_tokens.all_special_tokens,
        regex_pattern=tok_config.regex_pattern,
        name=args.run_name or tok_config.name,
    )

    # Manage run artifacts via ExperimentTracker
    tracker = ExperimentTracker()
    run_name = args.run_name or tok_config.name
    run_id, run_dir = tracker.init_run(run_name, config_dict=tok_config.model_dump())

    # Immediately run benchmark evaluation
    logger.info("📊 Running automatic post-training benchmark evaluation...")
    benchmark_runner = BenchmarkRunner(tokenizer)
    metrics = benchmark_runner.run_benchmark(custom_corpus_path=args.eval_corpus)

    # Save artifacts
    tracker.save_run_artifacts(
        run_id=run_id,
        tokenizer=tokenizer,
        metrics=metrics,
        config_dict=tok_config.model_dump(),
    )

    # Generate visualizations
    try:
        generate_all_visualizations(
            run_dir=run_dir,
            metrics=metrics,
            vocab=vocab,
            merges=merges,
            output_dir=run_dir / "visuals",
        )
    except Exception as e:
        logger.warning(f"Could not generate visual plots: {e}")

    logger.info(f"🎉 Successfully trained and saved tokenizer to {run_dir.resolve()} (Run ID: {run_id})")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run comprehensive multi-domain evaluation benchmark."""
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        logger.error(f"Model directory not found: {model_dir.resolve()}")
        return 1

    tokenizer = Tokenizer.from_pretrained(model_dir)
    runner = BenchmarkRunner(tokenizer)
    metrics = runner.run_benchmark(custom_corpus_path=args.eval_corpus)

    out_file = Path(args.output) if args.output else model_dir / "metrics.json"
    save_json(metrics, out_file, indent=2)

    # Update leaderboard if run is inside an experiments directory
    tracker = ExperimentTracker()
    tracker.scan_and_rebuild_leaderboard()

    # Display results
    logger.info(f"✅ Evaluation complete. Metrics saved to {out_file.resolve()}")
    return 0


def cmd_run_lm_benchmark(args: argparse.Namespace) -> int:
    """Run downstream small language model training benchmark."""
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        logger.error(f"Model directory not found: {model_dir.resolve()}")
        return 1

    tokenizer = Tokenizer.from_pretrained(model_dir)
    lm_config = load_lm_eval_config(args.config)

    corpus_path = args.corpus or "data/processed/eval_corpus_heldout.txt"
    if not Path(corpus_path).is_file():
        logger.warning(f"Corpus file not found at {corpus_path}. Looking for training corpus...")
        corpus_path = "data/processed/train_corpus_250mb.txt"

    runner = LMBenchmarkRunner(tokenizer=tokenizer, config=lm_config, model_dir=model_dir)
    steps = args.steps or lm_config.training.max_steps
    results = runner.run_benchmark(corpus_path=corpus_path, steps=steps)

    out_file = Path(args.output) if args.output else model_dir / "lm_eval.json"
    save_json(results, out_file, indent=2)

    # Update report.md with LM evaluation metrics
    tracker = ExperimentTracker()
    metrics_file = model_dir / "metrics.json"
    metrics = load_json(metrics_file) if metrics_file.is_file() else None
    report_content = tracker.generate_run_report(model_dir.name, metrics=metrics, lm_results=results)
    with open(model_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    # Generate LM BPC curve plot if loss history exists
    try:
        from src.evaluation.visualizer import plot_downstream_lm_bpc_curves
        if "loss_history" in results and results["loss_history"]:
            vis_dir = model_dir / "visuals"
            vis_dir.mkdir(parents=True, exist_ok=True)
            plot_downstream_lm_bpc_curves(
                {tokenizer.name: results["loss_history"]},
                vis_dir / "downstream_lm_bpc_curves.png",
            )
    except Exception as e:
        logger.warning(f"Could not generate LM BPC curve plot: {e}")

    # Rebuild central leaderboard
    tracker.scan_and_rebuild_leaderboard()

    logger.info(
        f"✅ LM Benchmark Finished!\n"
        f"  - Validation Loss/Token: {results['val_loss_per_token']:.4f}\n"
        f"  - Validation Perplexity: {results['val_perplexity']:.2f}\n"
        f"  - Loss Per Byte:         {results['val_loss_per_byte']:.4f}\n"
        f"  - Bits Per Character:    {results['val_bits_per_character']:.4f} BPC"
    )
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """Generate all publication-grade visualization plots for a run."""
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        logger.error(f"Run directory not found: {run_dir.resolve()}")
        return 1

    metrics_file = run_dir / "metrics.json"
    vocab_file = run_dir / "vocab.json"
    merges_file = run_dir / "merges.txt"
    lm_file = run_dir / "lm_eval.json"

    if not metrics_file.is_file():
        logger.error(f"Metrics file not found: {metrics_file.resolve()}. Run evaluation first.")
        return 1

    metrics = load_json(metrics_file)
    vocab = load_json(vocab_file) if vocab_file.is_file() else None

    merges = []
    if merges_file.is_file():
        with open(merges_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    p = line.split(" ")
                    if len(p) == 2:
                        merges.append((p[0], p[1]))

    lm_results = load_json(lm_file) if lm_file.is_file() else None

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "visuals"
    plots = generate_all_visualizations(
        run_dir=run_dir,
        metrics=metrics,
        vocab=vocab,
        merges=merges,
        lm_results=lm_results,
        output_dir=out_dir,
    )

    logger.info(f"✅ Generated {len(plots)} visualization plots in {out_dir.resolve()}")
    return 0


def cmd_run_experiment(args: argparse.Namespace) -> int:
    """Execute an end-to-end experiment recipe."""
    recipe_path = Path(args.config)
    if not recipe_path.is_file():
        logger.error(f"Experiment recipe not found: {recipe_path.resolve()}")
        return 1

    recipe = load_yaml(recipe_path)
    tracker = ExperimentTracker()

    if "experiment" in recipe:
        exp_info = recipe["experiment"]
        exp_id = exp_info.get("id", recipe_path.stem)
        exp_name = exp_info.get("name", exp_id)
        vocab_size = exp_info.get("vocab_size", 32000)
        min_freq = exp_info.get("min_frequency", 2)

        logger.info(f"🧪 Executing Experiment Recipe: '{exp_name}' (Vocab: {vocab_size:,})...")

        # Determine corpus path
        corpus_path = Path(args.corpus or "data/processed/train_corpus_250mb.txt")
        if not corpus_path.is_file():
            # Try to build or check if dataset_mix_config provided
            mix_cfg = exp_info.get("dataset_mix_config")
            if mix_cfg and Path(mix_cfg).is_file():
                logger.info(f"Building dataset corpus using mix config '{mix_cfg}'...")
                mix_config = load_dataset_mix_config(mix_cfg)
                builder = CorpusBuilder(config=mix_config)
                built = builder.build_all()
                corpus_path = Path(built["train_file"])
            else:
                logger.error(f"Corpus not found at {corpus_path.resolve()} and no valid dataset config specified.")
                return 1

        # Train
        run_id, run_dir = tracker.init_run(exp_name=exp_id, config_dict=recipe)
        trainer = BPETrainer(vocab_size=vocab_size, min_frequency=min_freq)
        vocab, merges, stats = trainer.train_from_file(corpus_path)

        tokenizer = Tokenizer(vocab=vocab, merges=merges, name=exp_id)

        # Benchmark
        eval_path = "data/processed/eval_corpus_heldout.txt" if Path("data/processed/eval_corpus_heldout.txt").is_file() else None
        benchmark_runner = BenchmarkRunner(tokenizer)
        metrics = benchmark_runner.run_benchmark(custom_corpus_path=eval_path)

        # Save artifacts
        tracker.save_run_artifacts(
            run_id=run_id,
            tokenizer=tokenizer,
            metrics=metrics,
            config_dict=recipe,
        )

        # Visuals
        generate_all_visualizations(
            run_dir=run_dir,
            metrics=metrics,
            vocab=vocab,
            merges=merges,
            output_dir=run_dir / "visuals",
        )

        logger.info(f"🎉 Experiment '{exp_name}' finished successfully! (Run ID: {run_id})")
        return 0

    elif "experiment_sweep" in recipe:
        sweep_info = recipe["experiment_sweep"]
        sweep_name = sweep_info.get("name", "Vocabulary Sweep")
        vocab_sizes = sweep_info.get("vocab_sizes", [16000, 32000, 50000])
        min_freq = sweep_info.get("min_frequency", 2)

        logger.info(f"🧪 Executing Vocabulary Sweep: {vocab_sizes}...")
        corpus_path = Path(args.corpus or "data/processed/train_corpus_250mb.txt")
        if not corpus_path.is_file():
            logger.error(f"Corpus file not found: {corpus_path.resolve()}")
            return 1

        sweep_results = []
        for v_size in vocab_sizes:
            sub_id = f"sweep_{v_size // 1000}k"
            run_id, run_dir = tracker.init_run(exp_name=sub_id, config_dict=recipe)
            trainer = BPETrainer(vocab_size=v_size, min_frequency=min_freq)
            vocab, merges, stats = trainer.train_from_file(corpus_path)
            tokenizer = Tokenizer(vocab=vocab, merges=merges, name=sub_id)

            runner = BenchmarkRunner(tokenizer)
            metrics = runner.run_benchmark()

            tracker.save_run_artifacts(
                run_id=run_id,
                tokenizer=tokenizer,
                metrics=metrics,
                config_dict=recipe,
            )
            sweep_results.append({
                "vocab_size": v_size,
                "compression_ratio": metrics["overall"]["compression_ratio"],
                "fertility": metrics["overall"]["fertility"],
                "run_id": run_id,
            })

        # Generate sweep visualization
        tradeoff_path = tracker.visuals_dir / "vocab_scaling_tradeoff.png"
        plot_vocab_scaling_tradeoff(sweep_results, tradeoff_path)
        logger.info(f"📊 Saved vocabulary scaling trade-off plot to {tradeoff_path.resolve()}")
        return 0

    else:
        logger.error(f"Unknown recipe format in {recipe_path}")
        return 1


def cmd_compare_runs(args: argparse.Namespace) -> int:
    """Compare all runs on the central master leaderboard."""
    tracker = ExperimentTracker(base_dir=Path(args.runs_dir).parent if args.runs_dir else "experiments")
    runs, table_md = tracker.scan_and_rebuild_leaderboard()

    if console:
        table = Table(title="🏆 Tokenizer Comparison Leaderboard", show_header=True, header_style="bold magenta")
        table.add_column("Model Name", style="cyan")
        table.add_column("Vocab", justify="right")
        table.add_column("Overall CR", justify="right", style="green")
        table.add_column("Fertility", justify="right")
        table.add_column("Prose CR", justify="right")
        table.add_column("Code CR", justify="right")
        table.add_column("Val BPC", justify="right", style="yellow")
        table.add_column("Lossless", justify="center")

        for r in runs:
            v_str = f"{r.get('vocab_size', 0) // 1000}k" if r.get('vocab_size', 0) >= 1000 else str(r.get('vocab_size', 0))
            table.add_row(
                str(r.get("model_name", "")),
                v_str,
                f"{r.get('overall_cr', 0.0):.2f}",
                f"{r.get('fertility', 0.0):.2f}",
                f"{r.get('prose_cr', 0.0):.2f}",
                f"{r.get('code_cr', 0.0):.2f}",
                str(r.get("val_bpc", "-")),
                str(r.get("lossless", "100%")),
            )
        console.print(table)
    else:
        print(table_md)

    logger.info(f"✅ Leaderboard updated at {tracker.leaderboard_md.resolve()}")
    return 0


def cmd_push_to_hub(args: argparse.Namespace) -> int:
    """Push trained tokenizer to Hugging Face Hub."""
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        logger.error(f"Model directory not found: {model_dir.resolve()}")
        return 1

    metrics = None
    metrics_file = model_dir / "metrics.json"
    if metrics_file.is_file():
        metrics = load_json(metrics_file)

    success = HFPublisher.publish_to_hub(
        model_dir=model_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        dry_run=args.dry_run,
        metrics=metrics,
    )
    return 0 if success else 1


def build_parser() -> argparse.ArgumentParser:
    """Constructs the top-level argument parser and subcommands."""
    parser = argparse.ArgumentParser(
        description="General-Purpose GPT-Style Byte-Level BPE Tokenizer Toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    # 1. prepare-data
    p_data = subparsers.add_parser("prepare-data", help="Download and assemble streaming dataset corpus")
    p_data.add_argument("--config", default="configs/dataset_mix.yaml", help="Path to dataset mix YAML")
    p_data.add_argument("--raw-dir", default="data/raw", help="Raw chunk download cache directory")
    p_data.add_argument("--processed-dir", default="data/processed", help="Processed corpus destination directory")
    p_data.set_defaults(func=cmd_prepare_data)

    # 2. train
    p_train = subparsers.add_parser("train", help="Train BPE tokenizer on text corpus")
    p_train.add_argument("--config", default="configs/tokenizer_32k.yaml", help="Path to tokenizer config YAML")
    p_train.add_argument("--corpus", default="data/processed/train_corpus_250mb.txt", help="Path to training corpus")
    p_train.add_argument("--eval-corpus", default=None, help="Optional held-out evaluation corpus")
    p_train.add_argument("--vocab-size", type=int, default=None, help="Override vocabulary size")
    p_train.add_argument("--min-frequency", type=int, default=None, help="Override minimum pair frequency")
    p_train.add_argument("--max-lines", type=int, default=None, help="Optional limit on lines read for fast testing")
    p_train.add_argument("--run-name", default=None, help="Custom identifier for run")
    p_train.set_defaults(func=cmd_train)

    # 3. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Multi-domain benchmark evaluation")
    p_eval.add_argument("--model", required=True, help="Directory containing saved tokenizer artifacts")
    p_eval.add_argument("--eval-corpus", default=None, help="Optional held-out text corpus file")
    p_eval.add_argument("--output", default=None, help="Destination path for metrics JSON")
    p_eval.set_defaults(func=cmd_evaluate)

    # 4. run-lm-benchmark
    p_lm = subparsers.add_parser("run-lm-benchmark", help="Downstream small language model evaluation")
    p_lm.add_argument("--model", required=True, help="Directory containing saved tokenizer artifacts")
    p_lm.add_argument("--config", default="configs/lm_eval_config.yaml", help="Path to LM evaluation config YAML")
    p_lm.add_argument("--corpus", default=None, help="Path to text corpus for LM training")
    p_lm.add_argument("--steps", type=int, default=None, help="Override LM training steps")
    p_lm.add_argument("--output", default=None, help="Destination path for LM eval JSON")
    p_lm.set_defaults(func=cmd_run_lm_benchmark)

    # 5. visualize
    p_vis = subparsers.add_parser("visualize", help="Generate publication-grade visualization plots")
    p_vis.add_argument("--run", required=True, help="Path to experiment run directory")
    p_vis.add_argument("--output-dir", default=None, help="Destination directory for visuals")
    p_vis.set_defaults(func=cmd_visualize)

    # 6. run-experiment
    p_exp = subparsers.add_parser("run-experiment", help="Execute an end-to-end experiment recipe")
    p_exp.add_argument("--config", required=True, help="Path to experiment recipe YAML")
    p_exp.add_argument("--corpus", default=None, help="Optional explicit corpus path")
    p_exp.set_defaults(func=cmd_run_experiment)

    # 7. compare-runs
    p_comp = subparsers.add_parser("compare-runs", help="Rebuild and display central leaderboard")
    p_comp.add_argument("--runs-dir", default="experiments/runs", help="Directory containing run folders")
    p_comp.set_defaults(func=cmd_compare_runs)

    # 8. push-to-hub
    p_hub = subparsers.add_parser("push-to-hub", help="Package and push tokenizer to Hugging Face Hub")
    p_hub.add_argument("--model", required=True, help="Directory containing saved tokenizer artifacts")
    p_hub.add_argument("--repo-id", required=True, help="Target HF repository ID (username/model_name)")
    p_hub.add_argument("--token", default=None, help="Hugging Face write token")
    p_hub.add_argument("--private", action="store_true", help="Create private repository")
    p_hub.add_argument("--dry-run", action="store_true", help="Perform local staging and verification only")
    p_hub.set_defaults(func=cmd_push_to_hub)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return args.func(args)
    except TokenizerBaseException as e:
        logger.error(f"❌ Execution failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user.")
        return 130
    except Exception as e:
        logger.exception(f"💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
