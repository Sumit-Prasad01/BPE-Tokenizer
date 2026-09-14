"""Script to package and push a trained Byte-Level BPE Tokenizer to Hugging Face Hub.

Loads HF_TOKEN automatically from .env if present.

Usage Examples:
    # 1. Push by model name or experiment ID (HF_TOKEN loaded from .env):
    python push_to_hf.py --hf_username <username> --which_tokenizer_to_push exp_c_mixed_32k

    # 2. Push top-ranked model on the leaderboard:
    python push_to_hf.py --hf_username <username> --which_tokenizer_to_push best

    # 3. Push by direct directory path:
    python push_to_hf.py --hf_username <username> --which_tokenizer_to_push experiments/runs/20260913_015040_cli_test_model

    # 4. Dry-run locally to test packaging and generate model card without uploading:
    python push_to_hf.py --hf_username <username> --which_tokenizer_to_push exp_c_mixed_32k --dry_run
"""

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import get_token
from src.hub.hf_publisher import HFPublisher
from src.tokenizer.serializer import TokenizerSerializer
from src.tokenizer.tokenizer import Tokenizer
from utils.custom_exception import HubPublishError
from utils.helpers import load_json
from utils.logger import logger

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None


def load_env_file(env_path: str | Path | None = None) -> dict[str, str]:
    """Loads environment variables from a .env file without requiring external libraries.
    
    Searches current working directory, script directory, and parent directory.
    Handles UTF-8 BOM, whitespace around '=', 'export' keywords, quotes, and inline comments.
    """
    loaded: dict[str, str] = {}

    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(Path(env_path))
    candidates.extend([
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ])

    env_file: Path | None = None
    for cand in candidates:
        if cand.is_file():
            env_file = cand.resolve()
            break

    if env_file is None:
        return loaded

    try:
        # utf-8-sig automatically strips any UTF-8 BOM if present
        with open(env_file, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Strip leading 'export ' if present
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()

                    # Strip enclosing quotes
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1].strip()
                    elif " #" in val:
                        # Strip inline comment if not inside quotes
                        val = val.split(" #", 1)[0].strip()

                    if key:
                        # Populate into os.environ if missing or currently empty
                        if not os.environ.get(key):
                            os.environ[key] = val
                        loaded[key] = val

        hf_keys = ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        matched_key = next((k for k in hf_keys if k in loaded and loaded[k]), None)
        if matched_key:
            token_val = loaded[matched_key]
            masked = f"{token_val[:5]}...{token_val[-4:]}" if len(token_val) > 9 else "***"
            logger.info(f"🔑 Successfully loaded {matched_key} ({masked}) from {env_file}")

    except Exception as e:
        logger.warning(f"Failed to parse .env file ({env_file}): {e}")

    return loaded


# Automatically load .env at module import
_ENV_LOADED = load_env_file()


def get_hf_token(cli_token: str | None = None) -> str | None:
    """Resolves Hugging Face write token from CLI override, .env, environment, or hub cache."""
    # 1. Direct CLI argument override
    if cli_token and cli_token.strip():
        return cli_token.strip()

    # 2. Ensure .env is loaded
    load_env_file()

    # 3. Check environment variables
    for var_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env_val = os.environ.get(var_name)
        if env_val and env_val.strip():
            return env_val.strip()

    # 4. Check cached token in ~/.cache/huggingface/token
    try:
        cached = get_token()
        if cached and cached.strip():
            return cached.strip()
    except Exception:
        pass

    return None


def find_available_runs(runs_dir: Path) -> list[Path]:
    """Returns a list of directories containing tokenizer artifacts, checking runs_dir and logs/runs."""
    search_dirs = [runs_dir]
    alt_runs = Path(__file__).resolve().parent / "logs" / "runs"
    if alt_runs.is_dir() and alt_runs.resolve() != runs_dir.resolve():
        search_dirs.append(alt_runs)

    valid_runs: list[Path] = []
    seen_names: set[str] = set()
    for s_dir in search_dirs:
        if not s_dir.is_dir():
            continue
        for item in sorted(s_dir.iterdir()):
            if item.is_dir() and item.name not in seen_names:
                if (item / "vocab.json").is_file() and (item / "merges.txt").is_file():
                    valid_runs.append(item)
                    seen_names.add(item.name)
    return valid_runs


def resolve_tokenizer_directory(identifier: str, runs_dir: Path) -> Path:
    """Resolves which_tokenizer_to_push into a verified model directory Path."""
    # 1. Direct directory check
    direct_path = Path(identifier)
    if direct_path.is_dir() and (direct_path / "vocab.json").is_file():
        return direct_path.resolve()

    # 2. Path relative to runs_dir
    rel_path = runs_dir / identifier
    if rel_path.is_dir() and (rel_path / "vocab.json").is_file():
        return rel_path.resolve()

    available_runs = find_available_runs(runs_dir)

    # 3. Keyword "best": select top model from leaderboard
    if identifier.strip().lower() == "best":
        leaderboard_path = runs_dir.parent / "leaderboard.json"
        if leaderboard_path.is_file():
            try:
                entries = load_json(leaderboard_path)
                if entries:
                    best_run_id = entries[0].get("run_id")
                    candidate = runs_dir / best_run_id
                    if candidate.is_dir() and (candidate / "vocab.json").is_file():
                        logger.info(f"🏆 Selected top-ranked run from leaderboard: '{best_run_id}'")
                        return candidate.resolve()
            except Exception as exc:
                logger.warning(f"Could not read leaderboard for 'best': {exc}")

    # 4. Substring match across runs_dir
    matches = [r for r in available_runs if identifier.lower() in r.name.lower()]
    if len(matches) == 1:
        logger.info(f"🔍 Matched tokenizer run: '{matches[0].name}'")
        return matches[0].resolve()
    elif len(matches) > 1:
        names = [f"  - {m.name}" for m in matches]
        raise HubPublishError(
            f"Ambiguous tokenizer identifier '{identifier}'. Multiple matching runs found:\n" + "\n".join(names)
        )

    # 5. List available choices if none matched
    available_list = [f"  - {r.name}" for r in available_runs]
    if available_list:
        avail_str = f"Available trained tokenizers in {runs_dir}:\n" + "\n".join(available_list)
    else:
        avail_str = f"No trained tokenizers found in {runs_dir.resolve()}."

    raise HubPublishError(f"Could not find tokenizer matching '{identifier}'.\n{avail_str}")


def ensure_complete_artifacts(model_dir: Path) -> None:
    """Ensures HF fast tokenizer files exist in model_dir, generating them if needed."""
    required = ["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    missing = [f for f in required if not (model_dir / f).is_file()]

    if missing:
        logger.info(f"🔧 Generating missing Hugging Face format files ({missing}) in {model_dir.name}...")
        tok = Tokenizer.from_pretrained(model_dir)
        TokenizerSerializer.save_pretrained(tok, model_dir)


def main() -> int:
    # 1. Load .env file automatically before parsing arguments
    load_env_file(".env")

    parser = argparse.ArgumentParser(
        description="Publish a trained Byte-Level BPE Tokenizer to Hugging Face Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf_username",
        required=True,
        help="Your Hugging Face username or organization (e.g. 'username')",
    )
    parser.add_argument(
        "--which_tokenizer_to_push",
        required=True,
        help="Identifier of tokenizer to push: directory path, run name, or 'best'",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Optional HF write token override (by default reads HF_TOKEN from .env)",
    )
    parser.add_argument(
        "--repo_name",
        default=None,
        help="Custom repository name on Hugging Face (default: derived from tokenizer name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the destination repository private on Hugging Face Hub",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate packaging and generate Model Card locally without uploading to Hugging Face",
    )
    parser.add_argument(
        "--runs_dir",
        default="experiments/runs",
        help="Base directory containing experiment runs",
    )

    args = parser.parse_args()

    try:
        runs_dir = Path(args.runs_dir)

        # 2. Resolve which tokenizer directory to push
        model_dir = resolve_tokenizer_directory(args.which_tokenizer_to_push, runs_dir)
        logger.info(f"📦 Target tokenizer directory: {model_dir}")

        # 3. Ensure all 5 artifact files are present
        ensure_complete_artifacts(model_dir)

        # 4. Determine repository name and repo ID
        if args.repo_name:
            repo_name = args.repo_name.strip()
        else:
            raw_name = model_dir.name
            parts = raw_name.split("_", 2)
            if len(parts) >= 3 and len(parts[0]) == 8 and len(parts[1]) == 6 and parts[0].isdigit() and parts[1].isdigit():
                repo_name = parts[2]
            else:
                repo_name = raw_name

        repo_name = repo_name.replace("_", "-").lower()
        repo_id = f"{args.hf_username.strip()}/{repo_name}"

        # 5. Load benchmark metrics if available
        metrics_file = model_dir / "metrics.json"
        metrics = load_json(metrics_file) if metrics_file.is_file() else None

        # 6. Resolve authentication token (CLI flag -> .env -> environment -> cached token)
        token = get_hf_token(args.hf_token)

        if token:
            masked = f"{token[:5]}...{token[-4:]}" if len(token) > 9 else "***"
            logger.info(f"🔐 Authenticated for Hugging Face Hub ({masked})")
        elif not args.dry_run:
            logger.warning("No Hugging Face token found in .env, environment, or arguments.")
            try:
                entered = getpass.getpass("Enter your Hugging Face write token (or press Ctrl+C to cancel): ").strip()
                if entered:
                    token = entered
            except (KeyboardInterrupt, EOFError):
                print()
                logger.warning("Token input cancelled. Switching to --dry_run validation mode.")
                args.dry_run = True

        # 7. Publish or Dry-run staging
        success = HFPublisher.publish_to_hub(
            model_dir=model_dir,
            repo_id=repo_id,
            token=token,
            private=args.private,
            dry_run=args.dry_run,
            metrics=metrics,
        )

        if success:
            staging_path = model_dir / "hf_package"
            if args.dry_run:
                msg = (
                    f"✅ [bold green]Dry-Run Packaging Succeeded![/bold green]\n\n"
                    f"• Target Repo: [bold cyan]{repo_id}[/bold cyan]\n"
                    f"• Staging Directory: [cyan]{staging_path}[/cyan]\n"
                    f"• Model Card: [cyan]{staging_path / 'README.md'}[/cyan]\n\n"
                    f"To perform the real upload, place your HF_TOKEN in .env and run:\n"
                    f"[yellow]python push_to_hf.py --hf_username {args.hf_username} --which_tokenizer_to_push {args.which_tokenizer_to_push}[/yellow]"
                )
            else:
                msg = (
                    f"🎉 [bold green]Successfully Published to Hugging Face Hub![/bold green]\n\n"
                    f"• Repository URL: [bold blue underline]https://huggingface.co/{repo_id}[/bold blue underline]\n\n"
                    f"Usage in Python:\n"
                    f"```python\n"
                    f"from transformers import AutoTokenizer\n\n"
                    f"tokenizer = AutoTokenizer.from_pretrained('{repo_id}')\n"
                    f"tokens = tokenizer.encode('Hello world!')\n"
                    f"print(tokenizer.decode(tokens))\n"
                    f"```"
                )

            if console:
                console.print(Panel(msg, title="Hugging Face Hub Deployment", border_style="green"))
            else:
                print(msg)
            return 0
        else:
            logger.error("Failed to publish tokenizer to Hugging Face.")
            return 1

    except HubPublishError as e:
        logger.error(f"❌ {e}")
        return 1
    except Exception as e:
        logger.exception(f"💥 Unexpected error during publication: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
