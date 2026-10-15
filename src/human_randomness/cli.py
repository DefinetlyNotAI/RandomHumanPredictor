"""Rich-powered command-line interface and hidden-prediction gameplay."""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .evaluation import evaluate
from .model import OnlinePredictor, validate_number
from .storage import DEFAULT_HOME, load, save

console = Console()
err_console = Console(stderr=True)


def read_number(prompt: str = "Your number [0-100]") -> int | None:
    while True:
        try:
            raw = Prompt.ask(f"[bold cyan]{prompt}[/]", console=console).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        try:
            if not raw.isascii() or not raw.isdigit():
                raise ValueError
            return validate_number(int(raw))
        except ValueError:
            console.print("[yellow]Enter an integer from 0 to 100, or q to quit.[/]")


def header(profile: str, previous: int, revealed: bool) -> None:
    title = Text("HUMAN RANDOMNESS PREDICTOR", style="bold cyan")
    body = Text()
    body.append(f"Profile  {profile}\n", style="bold")
    body.append(f"Past rounds  {previous}\n")
    body.append("Mode  ")
    body.append("Visible predictions" if revealed else "Hidden predictions", style="yellow" if revealed else "green")
    body.append("\nChoose your number before typing it. Enter q to finish.", style="dim")
    console.print(Panel(body, title=title, border_style="cyan", expand=False))


def result_panel(actual: int, prediction, scored: int, hits: int, top5_hits: int) -> None:
    guess = prediction.choice
    matched = guess == actual
    top5 = actual in prediction.top[:5]
    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column(justify="right", style="bold")
    summary.add_row("Your choice", str(actual))
    summary.add_row("Model guess", f"[green]{guess}[/]" if matched else f"[red]{guess}[/]")
    summary.add_row("Top five", ", ".join(map(str, prediction.top[:5])))
    summary.add_row("Chance assigned to choice", f"{prediction.probabilities[actual]:.2%}")
    summary.add_row("Top-five result", "[green]Hit[/]" if top5 else "[dim]Miss[/]")
    status = "EXACT MATCH" if matched else "NOT AN EXACT MATCH"
    console.print(Panel(summary, title=f"[{'green' if matched else 'yellow'}]{status}[/]", border_style="green" if matched else "dim", expand=False))

    table = Table(title="Session performance", box=box.ROUNDED, expand=False)
    table.add_column("Metric")
    table.add_column("Your model", justify="right")
    table.add_column("Random baseline", justify="right", style="dim")
    table.add_row("Exact match", f"{hits / scored:.2%}", f"{1 / 101:.2%}")
    table.add_row("Top five", f"{top5_hits / scored:.2%}", f"{5 / 101:.2%}")
    table.add_row("Rounds", str(scored), "")
    console.print(table)


def show_commitment(digest: str, nonce: str, guess: int) -> None:
    # Keep proof out of the main view, but expose the full verification recipe.
    console.print(f"[dim]SHA-256 commitment:[/] [dim]{digest}[/]")
    console.print(f"[dim]Verification nonce:[/] [dim]{nonce}[/]")
    console.print(f"[dim]Verify: sha256('{nonce}:{guess}') = {digest}[/]")


def play(profile: str, home: Path, show_predictions: bool, show_proof: bool = False) -> None:
    data = load(profile, home)
    model = OnlinePredictor([r["actual"] for r in data["rounds"]])
    scored = sum("guess" in r for r in data["rounds"])
    hits = sum(r.get("guess") == r["actual"] for r in data["rounds"] if "guess" in r)
    top5_hits = sum(r["actual"] in r.get("top5", []) for r in data["rounds"] if "guess" in r)
    header(profile, len(model.history), show_predictions)
    try:
        while True:
            prediction = model.predict(top_k=10)
            nonce = secrets.token_hex(16)
            digest = hashlib.sha256(f"{nonce}:{prediction.choice}".encode()).hexdigest()
            console.rule(f"Round {len(model.history) + 1}", style="dim")
            if show_predictions:
                console.print(f"[bold yellow]Prediction:[/] {prediction.choice} [dim]· Top 5: {list(prediction.top[:5])}[/]")
            else:
                console.print("[green]Prediction locked.[/] [dim]Reveal follows your answer.[/]")
                if show_proof:
                    console.print(f"[dim]Commitment: {digest}[/]")
            actual = read_number()
            if actual is None:
                break
            data["rounds"].append({
                "actual": actual, "guess": prediction.choice,
                "top5": list(prediction.top[:5]), "nonce": nonce,
                "commitment": digest,
            })
            save(profile, data, home)
            model.learn(actual)
            scored += 1
            hits += prediction.choice == actual
            top5_hits += actual in prediction.top[:5]
            result_panel(actual, prediction, scored, hits, top5_hits)
            if show_proof:
                show_commitment(digest, nonce, prediction.choice)
            else:
                console.print("[dim]Cryptographic proof saved with this round. Use --show-proof to display it.[/]")
            console.print()
    except (EOFError, KeyboardInterrupt):
        pass
    console.print("[cyan]Session ended.[/] Your profile is saved.")


def print_stats(scores: dict) -> None:
    if not scores["rounds"]:
        console.print("[yellow]No observations yet.[/]")
        return
    table = Table(title="Chronological replay", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Result", justify="right")
    table.add_column("Uniform baseline", justify="right", style="dim")
    table.add_row("Rounds", str(scores["rounds"]), "")
    table.add_row("Exact match", f"{scores['top1']:.2%}", f"{1 / 101:.2%}")
    table.add_row("Top five", f"{scores['top5']:.2%}", f"{5 / 101:.2%}")
    table.add_row("Top ten", f"{scores['top10']:.2%}", f"{10 / 101:.2%}")
    table.add_row("Log loss (lower is better)", f"{scores['log_loss']:.4f}", f"{scores['baseline_log_loss']:.4f}")
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict a person's next self-selected number")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME, help="profile storage directory")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play", help="predict, reveal, and learn interactively")
    p.add_argument("--profile", default="default", help="user profile name")
    p.add_argument("--show-predictions", action="store_true", help="show guesses before answering")
    p.add_argument("--show-proof", action="store_true", help="print hash and nonce for each round")
    p = sub.add_parser("stats", help="replay history without lookahead")
    p.add_argument("--profile", default="default")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON without Rich formatting")
    p = sub.add_parser("import", help="import integer choices, one per line or comma-separated")
    p.add_argument("file", type=Path)
    p.add_argument("--profile", default="default")
    args = parser.parse_args(argv)
    try:
        if args.command == "play":
            play(args.profile, args.home, args.show_predictions, args.show_proof)
        elif args.command == "stats":
            data = load(args.profile, args.home)
            scores = evaluate([r["actual"] for r in data["rounds"]])
            scores.pop("results")
            if args.json:
                print(json.dumps(scores, indent=2))
            else:
                print_stats(scores)
        elif args.command == "import":
            data = load(args.profile, args.home)
            values = args.file.read_text(encoding="utf-8").replace(",", " ").split()
            imported = []
            for token in values:
                if not token.isascii() or not token.isdigit():
                    raise ValueError(f"Invalid value: {token!r}")
                imported.append(validate_number(int(token)))
            for value in imported:
                data["rounds"].append({"actual": value})
            save(args.profile, data, args.home)
            console.print(f"[green]Imported {len(imported)} choices[/] into [bold]{args.profile}[/].")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        err_console.print(f"[red]Error:[/] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
