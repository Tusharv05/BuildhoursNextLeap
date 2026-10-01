"""Ask questions and get answers, interactively.

This is the full pipeline: guardrails -> entity-aware retrieval -> Groq -> validation.
Retrieval only (no answer) is `scripts/search.py`; this one gives you the finished answer.

    .\\.venv\\Scripts\\python.exe scripts\\ask.py

Then type questions until you want out. `exit` quits, Ctrl-C works too.

    --debug     show the full trace: filter, hits, validators, repair, latency
    --timing    show latency for every question (the first is slow: model load)
    --no-color  plain output, for piping into a file
    --bare      print only the answer text, nothing else. The prompt moves to stderr, so
                `ask.py --bare > answers.txt` gives one clean answer per line.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mf_rag.config as config_mod
from mf_rag.answer import OFFICIAL_SOURCES as OFFICIAL
from mf_rag.llm import get_llm
from mf_rag.pipeline import query
from mf_rag.store import verify_collection

WIDTH = 78

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"


class Style:
    """Wraps text in an ANSI colour.

    `style(text, code)` returns the string, for embedding in a larger expression.
    `style.echo(text, code)` prints it. Both are needed: the REPL prompt has to *return*
    the coloured string for `input()`, while everything else only needs to print.
    """

    def __init__(self, enabled: bool):
        self.on = enabled

    def __call__(self, text: str, code: str = "") -> str:
        if not self.on or not code:
            return str(text)
        return f"{code}{text}{RESET}"

    def echo(self, text: str, code: str = "") -> None:
        print(self(text, code))


def wrap(text: str, indent: str = "  ", width: int = WIDTH) -> str:
    return textwrap.fill(
        " ".join(str(text).split()), width=width, initial_indent=indent, subsequent_indent=indent
    )


def show_debug(answer, style: Style) -> None:
    trace = answer.trace
    style.echo("  trace", CYAN)
    style.echo(f"    query_hash      {trace.query_hash[:16]}...", DIM)
    style.echo(f"    intent          {trace.intent}", DIM)
    style.echo(f"    pii_blocked     {trace.pii_blocked}", DIM)
    style.echo(f"    entity_detected {trace.entity_detected}", DIM)
    style.echo(f"    where_filter    {trace.where_filter}", DIM)
    style.echo(f"    chunks_used     {len(trace.chunks_used)}", DIM)
    style.echo(f"    selected_source {trace.selected_source_url}", DIM)
    style.echo(f"    validators_run  {trace.validators_run}", DIM)
    style.echo(f"    repair_used     {trace.repair_used}", DIM)
    style.echo(f"    latency_ms      {trace.latency_ms}", DIM)
    if trace.hits:
        style.echo("    hits", DIM)
        for hit in trace.hits:
            official = "yes" if hit.get("source_id") in OFFICIAL else "no"
            style.echo(
                f"      {hit.get('score', 0):.4f}  {hit.get('source_id')}  official={official}",
                DIM,
            )


def ask(question: str, args: argparse.Namespace, style: Style) -> None:
    started = time.perf_counter()
    try:
        answer = query(question)
    except Exception as exc:  # report it, do not die mid-session
        if args.bare:
            print(f"error: {exc}", file=sys.stderr)
            return
        print()
        style.echo(f"  ERROR {type(exc).__name__}: {exc}", RED)
        style.echo("  A wrong key or model id is the usual cause. Check with:", DIM)
        style.echo("    .\\.venv\\Scripts\\python.exe scripts\\check_llm.py --live", DIM)
        print()
        return
    elapsed = (time.perf_counter() - started) * 1000

    if args.bare:
        # The answer and nothing else. No header, no question echo, no source line, no
        # refusal banner, no freshness restated: the answer text already carries the URL
        # and the as-of date because the response contract requires both.
        print(answer.text.strip())
        return

    print()
    style.echo(f"Q  {question}", BOLD)
    style.echo("-" * WIDTH, DIM)

    if answer.is_refusal:
        style.echo("  [refused by a guardrail - working as intended]", YELLOW)
    elif answer.trace.repair_used:
        style.echo("  [model draft failed validation - used the safe quoted answer]", YELLOW)

    print()
    print(wrap(answer.text))

    if answer.source_url:
        print()
        style.echo(f"  source  {answer.source_url}", CYAN)
    if answer.fetched_at:
        style.echo(f"  as of   {answer.fetched_at}", DIM)
    if args.timing:
        style.echo(f"  took    {elapsed:.0f} ms", DIM)

    if args.debug:
        print()
        show_debug(answer, style)
    print()


SAMPLES = [
    "What is the exit load of HDFC Small Cap Fund?",
    "What is the ELSS lock-in period?",
    "Should I invest in HDFC Large Cap?     (refused on purpose)",
    "What is my PAN? ABCDE1234F              (blocked on purpose)",
    "What is HDFC Parag Paratotal returns?  (refused: wrong fund house)",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline questions interactively")
    parser.add_argument("question", nargs="*", help="ask one question and exit")
    parser.add_argument("--debug", action="store_true", help="print the full trace")
    parser.add_argument("--timing", action="store_true", help="print latency per question")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    parser.add_argument(
        "--bare", action="store_true", help="print only the answer text (prompt goes to stderr)"
    )
    args = parser.parse_args()

    style = Style(not args.no_color and sys.stdout.isatty())
    verified = verify_collection(config_mod.get_config())
    client, degraded = get_llm()

    if args.bare:
        if degraded:
            print(
                "warning: no LLM key, answers are quoted from the source page",
                file=sys.stderr,
            )
        if args.question:
            ask(" ".join(args.question), args, style)
            return 0
        # Interactive, but the prompt goes to stderr so stdout stays pure answers.
        # input(prompt) writes its prompt to stdout, so print it separately and call
        # input() with no argument, otherwise the prompt lands in the piped output.
        while True:
            try:
                print(style("you> ", CYAN) if style.on else "you> ", end="", file=sys.stderr)
                sys.stderr.flush()
                question = input().strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                return 0
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                return 0
            ask(question, args, style)

    print()
    style.echo("  HDFC Mutual Fund FAQ assistant", BOLD)
    style.echo(f"  {verified['count']} chunks | {verified['embedding_model']}", DIM)
    style.echo(f"  LLM: {type(client).__name__} ({getattr(client, 'model', 'n/a')})", DIM)
    if degraded:
        style.echo(
            "  DEGRADED MODE: no LLM key, so answers are quoted from the source page "
            "rather than written.",
            YELLOW,
        )
        style.echo("  Set LLM_API_KEY in .env for real answers.", DIM)
    style.echo("  Type a question, or 'exit'. Try:", DIM)
    for sample in SAMPLES:
        style.echo(f"    {sample}", DIM)
    print()

    if args.question:
        ask(" ".join(args.question), args, style)
        return 0

    while True:
        try:
            question = input(style("  you> ", CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            return 0
        ask(question, args, style)


if __name__ == "__main__":
    raise SystemExit(main())
