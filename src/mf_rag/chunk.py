"""RAG Stage Group A / Stage 2: Chunking orchestrator.

Reads the Stage 1 output and writes data/chunks.jsonl plus data/chunk_stats.json.
Idempotent: the same corpus and strategy always produce byte-identical output.

CLI: python -m mf_rag.chunk --strategy section
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from mf_rag.chunking import get_chunker
from mf_rag.config import AppConfig, get_config
from mf_rag.ingest import load_source_docs
from mf_rag.models import Chunk, SourceDoc


def build_chunks(
    docs: Iterable[SourceDoc],
    cfg: AppConfig,
    strategy: str | None = None,
) -> tuple[list[Chunk], Any]:
    """Chunk every document. Returns (chunks, chunker) - the chunker carries
    per-strategy diagnostics such as orphan-row counts."""
    chunker = get_chunker(strategy, cfg.chunking)
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunker.split(doc))
    chunks.sort(key=lambda c: (c.source_id, c.chunk_index))
    return chunks, chunker


def chunk_stats(chunks: list[Chunk], chunker: Any, docs: list[SourceDoc]) -> dict[str, Any]:
    tokens = [c.token_count for c in chunks] or [0]
    per_source: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        bucket = per_source.setdefault(
            chunk.source_id,
            {"chunks": 0, "tokens": 0, "doc_type": chunk.doc_type, "scheme": chunk.scheme},
        )
        bucket["chunks"] += 1
        bucket["tokens"] += chunk.token_count
    return {
        "strategy": chunker.name,
        "documents": len(docs),
        "chunks": len(chunks),
        "avg_tokens": round(statistics.fmean(tokens), 1),
        "median_tokens": round(statistics.median(tokens), 1),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "orphan_table_rows": int(getattr(chunker, "orphan_rows", 0) or 0),
        "per_source": per_source,
        "total_source_chars": sum(len(d.text) for d in docs),
    }


def write_chunks(chunks: list[Chunk], cfg: AppConfig) -> Path:
    cfg.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True) + "\n")
    return cfg.chunks_path


def write_stats(stats: dict[str, Any], cfg: AppConfig) -> Path:
    path = cfg.data_dir / "chunk_stats.json"
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_chunks(cfg: AppConfig | None = None) -> list[Chunk]:
    cfg = cfg or get_config()
    if not cfg.chunks_path.is_file():
        raise FileNotFoundError(
            f"{cfg.chunks_path} not found - run 'python -m mf_rag.chunk' first (Phase 2)"
        )
    return [Chunk(**json.loads(line)) for line in cfg.chunks_path.read_text("utf-8").splitlines() if line]


def run(cfg: AppConfig | None = None, strategy: str | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    docs = load_source_docs(cfg)
    if not docs:
        raise RuntimeError(
            "No ingested sources found - run 'python scripts/run_ingest.py' first (Phase 1)"
        )
    chunks, chunker = build_chunks(docs, cfg, strategy)
    if not chunks:
        raise RuntimeError("Chunking produced no chunks - inspect data/raw/ for empty text")
    write_chunks(chunks, cfg)
    stats = chunk_stats(chunks, chunker, docs)
    write_stats(stats, cfg)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 - Chunking")
    parser.add_argument("--strategy", help="fixed | section | parent_child (default: config)")
    args = parser.parse_args(argv)

    cfg = get_config()
    stats = run(cfg, args.strategy)
    print(f"Stage 2 - Chunking (strategy={stats['strategy']})")
    print("-" * 70)
    print(f"documents   : {stats['documents']}")
    print(f"chunks      : {stats['chunks']}")
    print(f"tokens      : avg={stats['avg_tokens']} median={stats['median_tokens']} "
          f"min={stats['min_tokens']} max={stats['max_tokens']}")
    print(f"orphan rows : {stats['orphan_table_rows']}")
    print("-" * 70)
    for source_id, info in sorted(stats["per_source"].items(), key=lambda kv: -kv[1]["chunks"]):
        print(f"{source_id:<40} {info['chunks']:>4} chunks  {info['tokens']:>7} tokens")
    print("-" * 70)
    print(f"chunks -> {cfg.chunks_path}")
    print(f"stats  -> {cfg.data_dir / 'chunk_stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
