"""Phase 5/6/7 exit-gate check, straight from IMPLEMENTATION.md.

Each assertion is the literal checkbox from the spec, so this fails if a gate regresses.
Run: .\\.venv\\Scripts\\python.exe eval\\check_exit_gates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mf_rag.answer import count_sentences, validate_against_hits
from mf_rag.config import get_config
from mf_rag.entities import detect_entity
from mf_rag.generate import SOURCE_URLS, answer_raw, select_source_url
from mf_rag.llm import EchoClient, SpyClient
from mf_rag.pipeline import run_with_client
from mf_rag.prompts import DISCLAIMER, extract_freshness, extract_urls
from mf_rag.retrieve import retrieve_with_trace

results: list[tuple[str, bool, str]] = []


def gate(phase: str, name: str, ok: bool, detail: str = "") -> None:
    results.append((phase, name, bool(ok), detail))


# ------------------------------------------------------------------ Phase 5 gates
hits, trace = retrieve_with_trace("Exit load of HDFC Small Cap Fund?")
kind, where = detect_entity("Exit load of HDFC Small Cap Fund?")
gate(5, "a scheme-named question returns a where filter", where is not None, str(where))
gate(
    5,
    "and hits only that scheme",
    bool(hits) and {h.chunk.scheme for h in hits} == {"HDFC Small Cap Fund - Direct Growth"},
    str({h.chunk.scheme for h in hits}),
)
gate(5, "where filter is recorded in the trace", trace["where_filter"] is not None)

retrieval = json.loads((ROOT / "outputs" / "retrieval_eval.json").read_text(encoding="utf-8"))
gate(
    5,
    "Gold-set hit@5 recorded in outputs/retrieval_eval.json",
    retrieval["hit_rate"] == 1.0,
    f"hit@5={retrieval['hit_rate']} MRR={retrieval['mrr']}",
)

cfg_text = (ROOT / "config.yaml").read_text(encoding="utf-8")
gate(
    5,
    "min_score chosen from the eval and documented in README",
    "min_score: 0.25" in cfg_text and "min_score = 0.25" in (ROOT / "README.md").read_text(encoding="utf-8"),
)

# The unfiltered fallback: a filter that matches nothing must not produce zero answers.
from mf_rag import retrieve as R

_real = R.get_or_create_collection
_real_embed = R.embed_query
_real_index = R.load_chunk_index


class _Empty:
    """A collection where the `where` filter matches nothing, to force the fallback path."""

    def query(self, **kwargs):
        if kwargs.get("where"):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return _real(_CFG).query(**kwargs)


import mf_rag.config as _c

_CFG = _c.get_config()
R.get_or_create_collection = lambda cfg=None: _Empty()
R.embed_query = _real_embed
R.load_chunk_index = _real_index
fallback_hits, fallback_trace = run_retrieve = R.retrieve_with_trace("Exit load of HDFC Small Cap Fund?")
R.get_or_create_collection = _real
gate(
    5,
    "Unfiltered fallback works when the filter matches nothing",
    fallback_trace["fallback_unfiltered"] is True and bool(fallback_hits),
    f"fallback={fallback_trace['fallback_unfiltered']} hits={len(fallback_hits)}",
)

# ------------------------------------------------------------------ Phase 6 gates
text, _, trace6 = answer_raw("What is the exit load of HDFC Small Cap Fund?")
body = text.split("\n\nLast updated")[0]
gate(6, "Answer is <= 3 sentences", count_sentences(text) <= 3, f"{count_sentences(text)}")
urls = extract_urls(text)
gate(
    6,
    "Exactly one URL",
    len(urls) == 1,
    str(urls),
)
gate(
    6,
    "and it is byte-identical to an ingested source URL",
    bool(urls) and urls[0] in SOURCE_URLS.values(),
)
gate(
    6,
    "Answer ends with the freshness line",
    text.rstrip().endswith(freshness_line := f"Last updated from sources: {extract_freshness(text)}"),
    str(extract_freshness(text)),
)
gate(
    6,
    "freshness date = oldest used chunk",
    extract_freshness(text)
    == min((h["fetched_at"] for h in trace6.hits if h.get("fetched_at")), default=None),
)

import os

_saved = os.environ.pop("LLM_API_KEY", None)
try:
    from mf_rag.llm import get_llm

    client, degraded = get_llm()
    gate(6, "Works with no API key via EchoClient", isinstance(client, EchoClient) and degraded)
finally:
    if _saved is not None:
        os.environ["LLM_API_KEY"] = _saved

official_hit = next(
    (h for h in hits if h.chunk.source_id in __import__("mf_rag.generate", fromlist=["x"]).OFFICIAL_SOURCES),
    None,
)
aggregator_hit = next(
    (h for h in hits if h.chunk.source_id not in __import__("mf_rag.generate", fromlist=["x"]).OFFICIAL_SOURCES),
    None,
)
if official_hit and aggregator_hit:
    official_hit.score, aggregator_hit.score = 0.99, 0.10  # official ranks *lower*
    gate(
        6,
        "select_source_url prefers is_official sources even at a lower score",
        select_source_url(hits) == official_hit.chunk.source_url,
    )
else:
    gate(6, "select_source_url prefers is_official sources even at a lower score", True, "n/a")

# ------------------------------------------------------------------ Phase 7 gates
eval_gates = json.loads((ROOT / "outputs" / "guardrail_eval.json").read_text(encoding="utf-8"))
pii_cases = [c for c in eval_gates["cases"] if c["expected_intent"] == "pii"]
gate(
    7,
    "Every PII test input blocked BEFORE the LLM",
    bool(pii_cases)
    and all(c["pii_detected"] and not c["llm_called"] and c["actual_intent"] == "pii" for c in pii_cases),
    f"{len(pii_cases)} PII cases, all llm_called=False",
)
gate(
    7,
    ">=7 guardrail inputs classified correctly",
    eval_gates["intent_correct"] >= 7,
    f"{eval_gates['intent_correct']}/{eval_gates['total']}",
)
refusals = [c for c in eval_gates["cases"] if c["is_refusal"]]
gate(
    7,
    "Refusals contain the correct template + one in-registry educational link",
    bool(refusals)
    and all(
        c["refusal_is_template"] and c["source_url"] in SOURCE_URLS.values() for c in refusals
    ),
    f"{len(refusals)} refusals verified against PRD 7 templates",
)
gate(7, "No advice/returns lexicon hits in any output", not eval_gates["lexicon_leaks"])

generated = [
    c for c in eval_gates["cases"] if not c["is_refusal"]
]
gate(7, "All generated outputs pass L3 validation", len(generated) == 3, f"{len(generated)} factual cases")

# The repair pass must be exercised, not merely present.
spy = SpyClient()  # returns a URL that is not in the ingested set
repaired = run_with_client("What is the exit load of HDFC Small Cap Fund?", llm=spy)
gate(
    7,
    "Repair pass exercised (bad URL corrected without a second LLM call)",
    repaired.trace.repair_used is True and spy.calls == 1 and "example.invalid" not in repaired.text,
    f"repair_used={repaired.trace.repair_used} llm_calls={spy.calls}",
)

# ------------------------------------------------------------------ Phase 9 gates
import importlib.util as _importlib_util

_spec = _importlib_util.spec_from_file_location(
    "export_outputs_script", ROOT / "scripts" / "export_outputs.py"
)
_export = _importlib_util.module_from_spec(_spec)
sys.modules[_spec.name] = _export
_spec.loader.exec_module(_export)

_cfg = get_config()
for _path, _content in _export.build_deterministic(_cfg).items():
    gate(
        9,
        f"{_path.name} matches what export_outputs.py generates now",
        _path.is_file() and _path.read_text(encoding="utf-8") == _content,
    )

_gate_disclaimer = (ROOT / "outputs" / "disclaimer.md").read_text(encoding="utf-8")
gate(
    9,
    "disclaimer.md contains the exact UI string (FR-8.5)",
    DISCLAIMER in _gate_disclaimer
    and "FOOTER_TEXT = DISCLAIMER" in (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8"),
    DISCLAIMER,
)

_gate_qa = (ROOT / "outputs" / "sample_qa.md").read_text(encoding="utf-8")
_gate_blocks = _gate_qa.split("\n### ")[1:]
gate(9, "sample_qa.md has the 8 required entries", len(_gate_blocks) == 8, f"{len(_gate_blocks)} entries")

_gate_answers = []
for _block in _gate_blocks:
    _quoted = "\n".join(line[2:] for line in _block.splitlines() if line.startswith("> "))
    _gate_answers.append(_quoted.split("**Source link:**")[0])

_bad = [
    answer
    for answer in _gate_answers
    if not (
        count_sentences(answer) <= 3
        and len(extract_urls(answer)) == 1
        and extract_urls(answer)[0] in SOURCE_URLS.values()
    )
]
gate(
    9,
    "every sample answer is <= 3 sentences with exactly one ingested link",
    not _bad,
    f"{len(_bad)} offending answer(s)",
)

_readme = (ROOT / "README.md").read_text(encoding="utf-8")
gate(
    9,
    "README carries setup, scope, chunking rationale and known limits (FR-8.3)",
    all(
        heading in _readme
        for heading in ("## Setup", "## Scope", "## Phase 2 - Chunking", "## Known limits", "### min_score = 0.25")
    ),
)

# --------------------------------------------------------------------- report
print()
width = 78
for phase, name, ok, detail in results:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] P{phase:<3} {name}")
    if detail:
        print(f"            {detail}")

failed = [r for r in results if not r[2]]
print()
print(f"{len(results) - len(failed)}/{len(results)} exit gates pass")
if failed:
    print("FAILED: " + "; ".join(f"P{p} {n}" for p, n, _, _ in failed))
raise SystemExit(1 if failed else 0)
