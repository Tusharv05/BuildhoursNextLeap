"""Dump the embedding index to a human-readable text file.

The .jsonl cache is fine for machines and terrible for reading. This writes a flat
.txt with the metadata, the source text and the full 384-dim vector for every chunk,
plus a nearest-neighbour section so the space can be eyeballed.

Usage:
    python scripts/inspect_embeddings.py                    # -> outputs/embeddings_dump.txt
    python scripts/inspect_embeddings.py --limit 5          # first 5 chunks only
    python scripts/inspect_embeddings.py --no-vectors       # metadata + text only
    python scripts/inspect_embeddings.py --query "exit load"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from mf_rag.chunk import load_chunks  # noqa: E402
from mf_rag.config import get_config  # noqa: E402
from mf_rag.embed_store import embedding_cache_key, load_cache  # noqa: E402
from mf_rag.embedder import EXPECTED_DIM, embed_model_name, embed_texts  # noqa: E402

PER_LINE = 8


def fmt_vector(vector: list[float]) -> str:
    """Wrap the vector into fixed-width rows so columns line up when read."""
    rows = []
    for start in range(0, len(vector), PER_LINE):
        rows.append("    " + " ".join(f"{v:>8.5f}" for v in vector[start : start + PER_LINE]))
    return "\n".join(rows)


def wrap(text: str, width: int = 96, indent: str = "    ") -> str:
    import textwrap

    return "\n".join(
        textwrap.wrap(text, width=width, subsequent_indent=indent) or [indent]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump embeddings to a readable txt file")
    parser.add_argument("--limit", type=int, default=0, help="only the first N chunks")
    parser.add_argument("--no-vectors", action="store_true", help="metadata and text only")
    parser.add_argument("--query", default=None, help="add a similarity table for this query")
    parser.add_argument("--out", default=None, help="output path")
    args = parser.parse_args()

    cfg = get_config()
    model = embed_model_name(cfg)
    cache = load_cache(cfg.embeddings_path)
    chunks = load_chunks()

    missing = [c for c in chunks if embedding_cache_key(model, c.embed_text) not in cache]
    if missing:
        print(
            f"ERROR: {len(missing)} chunk(s) have no vector. "
            f"Run 'python scripts/build_index.py --stage embed' first.",
            file=sys.stderr,
        )
        return 1

    vectors = {c.chunk_id: cache[embedding_cache_key(model, c.embed_text)] for c in chunks}
    matrix = np.array([vectors[c.chunk_id] for c in chunks], dtype=np.float32)

    out = Path(args.out) if args.out else cfg.outputs_dir / "embeddings_dump.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    norms = np.linalg.norm(matrix, axis=1)
    with out.open("w", encoding="utf-8") as fh:
        w = fh.write
        w("=" * 100 + "\n")
        w("EMBEDDING INDEX DUMP\n")
        w("=" * 100 + "\n")
        w(f"model              : {model}\n")
        w(f"dimensions         : {matrix.shape[1]} (expected {EXPECTED_DIM})\n")
        w(f"chunks             : {len(chunks)}\n")
        w(f"documents          : {len({c.source_id for c in chunks})}\n")
        w(f"normalised         : cfg.embedding.normalize = {cfg.embedding.normalize}\n")
        w(f"vector L2 norm     : min {norms.min():.6f}  max {norms.max():.6f}\n")
        w(f"cache key          : sha1(\"{{model}}:{{embed_text}}\")\n")
        w(f"chunks file        : {cfg.chunks_path}\n")
        w(f"cache file         : {cfg.embeddings_path}\n")
        w(
            f"cosine distance    : "
            f"{cfg.retrieval.distance} (Chroma hnsw:space=cosine)\n"
        )
        w("\n")
        w("-" * 100 + "\n")
        w("CONTENTS\n")
        w("-" * 100 + "\n")
        w(
            "  Part 1  every chunk: identity, source text, and the full 384-dim vector.\n"
            "  Part 2  nearest neighbours: for each chunk, the 3 most similar others.\n"
        )
        if args.query:
            w("  Part 3  similarity of a specific query against all chunks.\n")
        w("\n")
        w("=" * 100 + "\n")
        w("PART 1 - CHUNKS\n")
        w("=" * 100 + "\n\n")

        shown = chunks[: args.limit] if args.limit else chunks
        for position, chunk in enumerate(shown):
            w(f"[{position + 1}/{len(shown)}]\n")
            w(f"  chunk_id        : {chunk.chunk_id}\n")
            w(f"  cache_key       : {embedding_cache_key(model, chunk.embed_text)}\n")
            w(f"  source_id       : {chunk.source_id}\n")
            w(f"  source_url      : {chunk.source_url}\n")
            w(f"  scheme          : {chunk.scheme or '(none)'}\n")
            w(f"  category        : {chunk.category or '(none)'}\n")
            w(f"  doc_type        : {chunk.doc_type}\n")
            w(f"  section_heading : {chunk.section_heading or '(none)'}\n")
            w(f"  chunk_index     : {chunk.chunk_index}\n")
            w(f"  token_count     : {chunk.token_count}\n")
            w(f"  fetched_at      : {chunk.fetched_at}\n")
            w("  text            :\n")
            w(wrap(chunk.text.replace("\n", " | ")) + "\n")
            if not args.no_vectors:
                w("  embed_text      :\n")
                w(wrap(chunk.embed_text.replace("\n", " | ")) + "\n")
                w(f"  vector[{EXPECTED_DIM}]  :\n")
                w(fmt_vector(vectors[chunk.chunk_id]) + "\n")
            w("\n")

        if args.no_vectors:
            w("=" * 100 + "\n")
            w("(vectors omitted: --no-vectors)\n")
            print(f"wrote {out} ({len(shown)} chunks, vectors omitted)")
            return 0

        w("=" * 100 + "\n")
        w("PART 2 - NEAREST NEIGHBOURS\n")
        w("=" * 100 + "\n")
        w("Cosine similarity of each chunk against the rest of the corpus.\n")
        w("Chunks from the same source page should cluster near 1.0.\n\n")
        for position, chunk in enumerate(shown):
            sims = matrix @ matrix[position]
            order = np.argsort(-sims)
            neighbours = [i for i in order if i != position][:3]
            w(f"[{position + 1}] {chunk.source_id}  (chunk_index={chunk.chunk_index})\n")
            for rank, other in enumerate(neighbours, start=1):
                oc = chunks[other]
                w(f"    {rank}. cos={sims[other]:.4f}  {oc.source_id}  #{oc.chunk_index}\n")
                w(f"       {oc.text[:88].replace(chr(10), ' ')!r}\n")
            w("\n")

        if args.query:
            w("=" * 100 + "\n")
            w(f"PART 3 - QUERY SIMILARITY: {args.query!r}\n")
            w("=" * 100 + "\n\n")
            qv = embed_texts([args.query], cfg)[0]
            sims = matrix @ qv
            for rank, i in enumerate(np.argsort(-sims)[:10], start=1):
                c = chunks[i]
                w(f"  {rank:>2}. cos={sims[i]:.4f}  {c.source_id}  #{c.chunk_index}\n")
                w(f"      scheme : {c.scheme or '(none)'}\n")
                w(f"      text   : {c.text[:86].replace(chr(10), ' ')!r}\n")
            w("\n")

    size = out.stat().st_size
    print(f"wrote {out}")
    print(f"  {len(shown)} chunks, {matrix.shape[1]}-dim, {size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
