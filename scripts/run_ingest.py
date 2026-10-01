"""RAG Stage Group A / Stage 1: Loading - CLI entry point.

Usage:
    python scripts/run_ingest.py --reset
    python scripts/run_ingest.py --only hdfc_elss_scheme_page
    python scripts/run_ingest.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mf_rag.config import get_config  # noqa: E402
from mf_rag.ingest import ingest_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 - Loading: fetch and clean the corpus")
    parser.add_argument("--reset", action="store_true", help="clear data/raw/ before fetching")
    parser.add_argument("--only", help="ingest a single source_id")
    parser.add_argument("--dry-run", action="store_true", help="list sources without fetching")
    args = parser.parse_args()

    cfg = get_config()
    print(f"Stage 1 - Loading  (config: chunking={cfg.chunking.strategy}, "
          f"delay={cfg.ingest.request_delay_sec}s, retries={cfg.ingest.max_retries})")
    if args.dry_run:
        print("dry run: no network calls\n")

    report = ingest_all(cfg, reset=args.reset, only=args.only, dry_run=args.dry_run)
    totals = report.totals

    print("\n" + "-" * 78)
    print(f"{'source_id':<34} {'status':<8} {'http':<6} {'chars':>8}")
    print("-" * 78)
    for result in report.sources:
        status = result.http_status or "-"
        print(f"{result.source_id:<34} {result.status:<8} {str(status):<6} {result.chars:>8}")
    print("-" * 78)
    print(
        f"total={totals['total']} ok={totals['ok']} skipped={totals['skipped']} "
        f"failed={totals['failed']} chars={totals['total_chars']}"
    )
    print(f"csv    -> {cfg.sources_csv}")
    print(f"report -> {cfg.outputs_dir / 'ingest_report.json'}")

    if totals["ok"] == 0:
        print("\nERROR: no source produced usable content; downstream stages cannot run.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
