"""Interactive retrieval tester.

Two ways to use it:

    # one-shot
    .\\.venv\\Scripts\\python.exe scripts\\search.py "What is the ELSS lock-in period?"

    # REPL: type a question, get a ranked trace back. Repeat until you want out.
    .\\.venv\\Scripts\\python.exe scripts\\search.py

Useful flags while exploring:

    --no-entity-filter   turn the scheme/category `where` filter off, to see what the
                         embeddings alone would have ranked (this is how the Small Cap
                         / Large Cap mix-up was found)
    --min-score 0.30     try a different floor without editing config.yaml
    --top-k 10           show more than the 5 the config asks for
    --json               machine-readable, for piping into other tools

The point of this script is to make the *decisions* visible. If a result looks wrong,
the trace usually says why: a filter you did not expect, a fallback that fired, or a
floor that cut everything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mf_rag.config as cfg_mod
from mf_rag.generate import OFFICIAL_SOURCES
from mf_rag.retrieve import retrieve_with_trace
from mf_rag.store import verify_collection

MIN_WIDTH = 60


def _is_official(source_id: str) -> bool:
    return "yes" if source_id in OFFICIAL_SOURCES else "no"


def _wrap(text: str, width: int, indent: str) -> str:
    import textwrap

    return textwrap.fill(
        " ".join(text.split()), width=width, initial_indent=indent, subsequent_indent=indent
    )


def show(query: str, args: argparse.Namespace) -> dict:
    hits, trace = retrieve_with_trace(
        query,
        k=args.top_k,
        min_score=args.min_score,
        entity_filter=False if args.no_entity_filter else None,
    )

    if args.json:
        print(json.dumps({"query": query, "trace": trace}, indent=2, ensure_ascii=False))
        return trace

    filter_note = trace["where_filter_text"] or "none (unfiltered)"
    print(f"\n{'=' * MIN_WIDTH}")
    print(f"QUERY   {query}")
    print(f"{'-' * MIN_WIDTH}")
    print(f"entity        {trace['entity_detected']}")
    print(f"where filter  {filter_note}")
    if trace["where_filter"]:
        print(f"              {trace['where_filter']}")
    print(f"fallback      {'YES - filter matched nothing, retried unfiltered' if trace['fallback_unfiltered'] else 'no'}")
    print(f"min_score     {trace['min_score']}    top_k {trace['top_k']}")
    print(
        f"candidates    {trace['candidates_considered']} -> {trace['after_dedupe']} after "
        f"per-source dedupe"
    )

    if not hits:
        print()
        print(f"  NO HITS above the {trace['min_score']} floor.")
        print("  This is the honest NOT_IN_SOURCES path. Try --min-score 0.15 to see what")
        print("  was just below the line.")
        return trace

    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        print()
        print(f"  [{rank}] {hit.score:.4f}  {chunk.source_id}")
        heading = chunk.section_heading or "(no heading)"
        scheme = chunk.scheme or "no scheme"
        print(f"      {heading}")
        print(f"      scheme={scheme}  doc_type={chunk.doc_type}  official={_is_official(chunk.source_id)}")
        if chunk.fetched_at:
            print(f"      fetched_at={chunk.fetched_at}")
        print(f"      {chunk.source_url}")
        print()
        print(_wrap(chunk.text, MIN_WIDTH - 6, "      "))

    return trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive retrieval tester")
    parser.add_argument("query", nargs="*", help="question to search for")
    parser.add_argument("--top-k", type=int, default=None, help="override config top_k")
    parser.add_argument("--min-score", type=float, default=None, help="override config min_score")
    parser.add_argument(
        "--no-entity-filter", action="store_true", help="disable the scheme/category where filter"
    )
    parser.add_argument("--json", action="store_true", help="emit the raw trace as JSON")
    args = parser.parse_args()

    verified = verify_collection(cfg_mod.get_config())
    if args.json:
        return 0
    print(f"index: {verified['count']} chunks at {verified['persist_path']}")
    print(f"model: {verified['embedding_model']}")

    if args.query:
        show(" ".join(args.query), args)
        return 0

    print("\nType a question. Ctrl-C or 'exit' to quit.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            return 0
        show(query, args)


if __name__ == "__main__":
    raise SystemExit(main())
