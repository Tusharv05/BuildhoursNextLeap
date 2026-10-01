"""Build the vector index: Stage 3 (embed) and Stage 4 (store).

Usage:
    python scripts/build_index.py --stage embed     # Stage 3 only
    python scripts/build_index.py --stage store     # Stage 4 only (needs embeddings)
    python scripts/build_index.py --stage all       # both, then verify
    python scripts/build_index.py --stage all --reset        # drop the collection first
    python scripts/build_index.py --stage all --reset-cache  # ignore the embedding cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mf_rag.chunk import load_chunks  # noqa: E402
from mf_rag.config import get_config  # noqa: E402
from mf_rag.embed_store import embed_chunks, embedding_cache_key, load_cache  # noqa: E402
from mf_rag.embedder import embed_model_name  # noqa: E402
from mf_rag.store import build_collection, verify_collection  # noqa: E402


def vectors_from_cache(chunks, cfg) -> dict[str, list[float]]:
    """Rebuild `{chunk_id: vector}` from the on-disk cache, without loading the model."""
    cache = load_cache(cfg.embeddings_path)
    model = embed_model_name(cfg)
    missing = [c.chunk_id for c in chunks if embedding_cache_key(model, c.embed_text) not in cache]
    if missing:
        raise SystemExit(
            f"{len(missing)} chunk(s) have no cached vector (e.g. {missing[:3]}). "
            f"Run 'python scripts/build_index.py --stage embed' first."
        )
    return {c.chunk_id: cache[embedding_cache_key(model, c.embed_text)] for c in chunks}


def _rule(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def run_embed(cfg, reset_cache: bool) -> tuple[dict, dict]:
    _rule("Stage 3 - Embedding")
    chunks = load_chunks()
    if not chunks:
        raise SystemExit(
            "data/chunks.jsonl is empty - run 'python -m mf_rag.chunk' (Phase 2) first."
        )
    started = time.time()
    vectors, stats = embed_chunks(chunks, cfg, reset_cache=reset_cache)
    print(f"documents   : {len({c.source_id for c in chunks})}")
    print(f"chunks      : {stats['total']}")
    print(f"model       : {stats['model']}")
    print(f"dimensions  : {stats['dim']}")
    print(
        f"embedded {stats['total']} chunks "
        f"(cache hits: {stats['cache_hits']}, misses: {stats['cache_misses']}, "
        f"hit ratio: {stats['cache_hit_ratio']:.1%})"
    )
    print(f"elapsed     : {time.time() - started:.1f}s")
    print(f"cache       : {stats['cache_path']}")
    print("-" * 72)
    return chunks, vectors


def run_store(cfg, chunks, vectors, reset: bool) -> dict:
    _rule("Stage 4 - Store Vector Data")
    written = build_collection(chunks, vectors, cfg, reset=reset)
    print(f"collection  : mf_faq")
    print(f"stored      : {written}/{len(chunks)} chunks")
    report = verify_collection(cfg, expected_count=len(chunks))
    print(f"count check : {report['count']}/{report['expected_count']} OK")
    print("metadata    : round-trip OK")
    print(f"filter check: where={{'category': 'ELSS'}} -> {report['elss_filter_hits']} docs OK")
    print(f"persist dir : {report['persist_path']}")
    print("-" * 72)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the vector index (Stages 3-4)")
    parser.add_argument("--stage", choices=("embed", "store", "all"), default="all")
    parser.add_argument("--reset", action="store_true", help="drop and recreate the collection")
    parser.add_argument("--reset-cache", action="store_true", help="ignore embeddings.jsonl")
    args = parser.parse_args()

    cfg = get_config()
    cfg.ensure_dirs()
    print(f"Stage 3-4 - Vector index build (stage={args.stage})\n")

    if args.stage in ("embed", "all"):
        chunks, vectors = run_embed(cfg, args.reset_cache)
    else:
        chunks = load_chunks()
        if not chunks:
            raise SystemExit("data/chunks.jsonl is empty - run 'python -m mf_rag.chunk' first.")
        vectors = vectors_from_cache(chunks, cfg)
        print(f"Loaded {len(vectors)} vectors from {cfg.embeddings_path} (no model calls)\n")

    if args.stage == "embed":
        _rule("Done")
        print("Next: python scripts/build_index.py --stage all")
        return 0

    if args.stage in ("store", "all"):
        run_store(cfg, chunks, vectors, reset=args.reset)

    _rule("Done")
    print("Index is persistent. Verify from a new process with:")
    print("  python -c \"from mf_rag.store import get_or_create_collection;"
          "print(get_or_create_collection().count())\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
