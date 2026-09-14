"""Interactive Rich Terminal Playground and Tokenizer Visualizer REPL.

Visualizes subwords with alternating colored backgrounds, detailed token breakdown tables,
live compression/fertility telemetry, and dynamic BPE-Dropout experimentation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.inference.engine import BPEInferenceEngine
from src.tokenizer.byte_encoder import decode_string_to_bytes


# Color palette for alternating subwords
TOKEN_COLORS = [
    "bold black on bright_cyan",
    "bold white on dark_magenta",
    "bold black on bright_green",
    "bold white on bright_blue",
    "bold black on bright_yellow",
    "bold white on dark_red",
    "bold black on bright_white",
    "bold white on deep_sky_blue1",
    "bold black on bright_magenta",
    "bold white on chartreuse3",
]


def render_colored_subwords(
    token_ids: list[int],
    id_to_token: dict[int, str],
) -> Text:
    """Render tokens with alternating colored background blocks."""
    text_obj = Text()
    for idx, tok_id in enumerate(token_ids):
        color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
        token_str = id_to_token.get(tok_id, f"<unk_{tok_id}>")

        # Replace non-printable byte markers with visible representation
        disp_str = token_str.replace("Ġ", "·")
        disp_str = disp_str.replace("\n", "⏎\n")
        disp_str = disp_str.replace("\t", "⇥\t")

        text_obj.append(f" {disp_str} ", style=color)
        text_obj.append(" ", style="default")

    return text_obj


def launch_interactive_repl(
    model_dir: str | Path,
    p_dropout: float = 0.0,
) -> None:
    """Launch interactive terminal REPL."""
    console = Console()
    model_path = Path(model_dir)

    console.print(
        Panel(
            f"[bold cyan]⚡ High-Performance Byte-Level BPE Tokenizer Playground ⚡[/bold cyan]\n"
            f"[dim]Model Path:[/dim] [green]{model_path.resolve()}[/green]\n"
            f"[dim]Commands:[/dim] [yellow]/dropout <p>[/yellow] to change dropout | [yellow]/quit[/yellow] or [yellow]/exit[/yellow] to leave",
            border_style="bright_blue",
            box=box.ROUNDED,
        )
    )

    console.print("[dim]Loading tokenizer model and initializing engine...[/dim]")
    t0 = time.perf_counter()
    engine = BPEInferenceEngine.from_pretrained(model_path)
    t1 = time.perf_counter()

    console.print(
        f"[green]✔ Engine initialized in {(t1 - t0) * 1000:.1f} ms[/green] "
        f"(Vocab Size: [bold cyan]{len(engine.id_to_token):,}[/bold cyan] | "
        f"Backend: [bold yellow]{engine.active_backend}[/bold yellow])\n"
    )

    current_dropout = p_dropout

    while True:
        try:
            user_input = console.input("[bold magenta]tokenizer[/bold magenta][bold white]>[/bold white] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting playground. Goodbye![/yellow]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[yellow]Exiting playground. Goodbye![/yellow]")
            break

        if stripped.startswith("/dropout"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    new_p = float(parts[1])
                    if 0.0 <= new_p < 1.0:
                        current_dropout = new_p
                        console.print(f"[green]✔ BPE-Dropout set to p = {current_dropout:.2f}[/green]")
                    else:
                        console.print("[red]Error: p must be between 0.0 and 1.0[/red]")
                except ValueError:
                    console.print("[red]Error: invalid float for dropout[/red]")
            else:
                console.print(f"[cyan]Current BPE-Dropout: p = {current_dropout:.2f}[/cyan]")
            continue

        # Tokenize input
        t_start = time.perf_counter()
        tokens = engine.encode(user_input, p_dropout=current_dropout)
        t_end = time.perf_counter()
        latency_us = (t_end - t_start) * 1e6

        # Decode for roundtrip verification
        decoded_text = engine.decode(tokens)
        is_lossless = (decoded_text == user_input)

        # Calculate metrics
        num_chars = len(user_input)
        num_bytes = len(user_input.encode("utf-8"))
        num_tokens = len(tokens)
        words = user_input.split()
        num_words = len(words) if words else 1

        cr = num_bytes / max(num_tokens, 1)
        fertility = num_tokens / max(num_words, 1)

        # 1. Colored subword pill display
        subwords_render = render_colored_subwords(tokens, engine.id_to_token)
        console.print(
            Panel(
                subwords_render,
                title=f"[bold green]Tokens ({num_tokens})[/bold green]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        # 2. Telemetry dashboard
        lossless_str = "[bold green]100% Lossless ✔[/bold green]" if is_lossless else "[bold red]MISMATCH ✘[/bold red]"
        dropout_str = f" | [dim]Dropout p:[/dim] [yellow]{current_dropout:.2f}[/yellow]" if current_dropout > 0 else ""

        console.print(
            f"[dim]Bytes:[/dim] [cyan]{num_bytes}[/cyan] | "
            f"[dim]Tokens:[/dim] [cyan]{num_tokens}[/cyan] | "
            f"[dim]Compression Ratio:[/dim] [bold green]{cr:.2f} B/T[/bold green] | "
            f"[dim]Fertility:[/dim] [bold yellow]{fertility:.2f} T/W[/bold yellow] | "
            f"[dim]Latency:[/dim] [bold magenta]{latency_us:.1f} µs[/bold magenta] | "
            f"{lossless_str}{dropout_str}"
        )

        # 3. Detailed Token Breakdown Table (show first 30 tokens to keep screen readable)
        display_limit = 30
        table = Table(
            title="Detailed Token Breakdown",
            box=box.SIMPLE_HEAD,
            header_style="bold cyan",
            show_lines=False,
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Token ID", style="bold yellow", width=10, justify="right")
        table.add_column("Subword Text", style="bold white", width=25)
        table.add_column("Byte Hex", style="green", width=25)
        table.add_column("Length", style="magenta", width=8, justify="right")

        for idx, tok_id in enumerate(tokens[:display_limit]):
            tok_str = engine.id_to_token.get(tok_id, "")
            raw_bytes = decode_string_to_bytes(tok_str)
            hex_str = " ".join(f"{b:02X}" for b in raw_bytes)
            disp_tok = tok_str.replace("Ġ", "·").replace("\n", "\\n").replace("\t", "\\t")

            table.add_row(
                str(idx + 1),
                str(tok_id),
                f"'{disp_tok}'",
                hex_str,
                f"{len(raw_bytes)} B",
            )

        console.print(table)
        if len(tokens) > display_limit:
            console.print(f"[dim]... and {len(tokens) - display_limit} more tokens truncated for display.[/dim]")
        console.print()


if __name__ == "__main__":
    default_dir = PROJECT_ROOT / "experiments" / "runs" / "20260914_023959_exp_vocab_64k"
    if not default_dir.exists():
        runs = list((PROJECT_ROOT / "experiments" / "runs").glob("*_exp_*"))
        default_dir = runs[0] if runs else (PROJECT_ROOT / "experiments" / "runs" / "20260913_023301_general_purpose_bpe_32k")

    target = sys.argv[1] if len(sys.argv) > 1 else str(default_dir)
    launch_interactive_repl(target)
