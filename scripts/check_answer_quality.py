"""End-to-end check against the real provider. Never prints the key.

Run: .\\.venv\\Scripts\\python.exe scripts\\check_answer_quality.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mf_rag.llm import get_llm
from mf_rag.pipeline import query

QUESTIONS = [
    "What is the exit load of HDFC Small Cap Fund?",
    "What is the ELSS lock-in period?",
    "What is the expense ratio of HDFC Flexi Cap Fund?",
]

client, degraded = get_llm()
print(f"client={type(client).__name__}  degraded={degraded}")
print(f"model={getattr(client, 'model', 'n/a')}")
print()

for question in QUESTIONS:
    started = time.perf_counter()
    try:
        answer = query(question)
    except Exception as exc:
        print(f"Q: {question}\n  RAISED {type(exc).__name__}: {exc}\n")
        continue
    elapsed = (time.perf_counter() - started) * 1000

    text = answer.text or ""
    print(f"Q: {question}")
    print(f"  latency      {elapsed:.0f} ms")
    print(f"  source_url   {answer.source_url}")
    print(f"  fetched_at   {answer.fetched_at}")
    print(f"  chars        {len(text)}")
    print(f"  text         {text[:400]!r}")
    if not text.strip():
        print("  !! EMPTY ANSWER - the model returned no content (reasoning budget?)")
    print()
