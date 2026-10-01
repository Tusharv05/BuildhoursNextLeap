"""Prove the Chroma store is on-disk and survives process exit.

Deliberately does NOT build anything: it only reads what is already in data/chroma.
Run: .\\.venv\\Scripts\\python.exe eval\\check_persistence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mf_rag.config as c
from mf_rag.retrieve import retrieve_with_trace
from mf_rag.store import get_or_create_collection, verify_collection

cfg = c.get_config()
col = get_or_create_collection(cfg)
verified = verify_collection(cfg)

print("collection name  :", col.name)
print("count on disk    :", col.count())
print("persist path     :", verified["persist_path"])
print("verify_collection:", verified)

meta = col.metadata or {}
print("embedding_model :", meta.get("embedding_model"), "(collection-level)")

for where in ({"category": "ELSS"}, {"scheme": "HDFC Small Cap Fund - Direct Growth"}):
    got = col.get(where=where, include=[])
    print(f"where {where} -> {len(got['ids'])} docs")

hits, trace = retrieve_with_trace("What is the exit load of HDFC Small Cap Fund?")
print()
print("retrieval served from the on-disk index:")
print("  entity   :", trace["entity_detected"], "|", trace["where_filter_text"])
print("  where    :", trace["where_filter"])
print("  fallback :", trace["fallback_unfiltered"])
print("  considered/deduped:", trace["candidates_considered"], "->", trace["after_dedupe"])
for hit in hits:
    print(f"  {hit.score:.4f}  {hit.chunk.source_id}  {hit.chunk.section_heading[:50]}")

# The source_id set is the thing that makes the citation check meaningful, so
# confirm the persisted metadata is complete enough to validate against.
docs = col.get(include=["metadatas"])
missing = [d for d in docs["metadatas"] if not d.get("source_url") or not d.get("fetched_at")]
print()
print(f"records missing source_url/fetched_at: {len(missing)} of {len(docs['metadatas'])}")
