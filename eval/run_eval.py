"""Retrieval + guardrail evaluation harness.

    python eval/run_eval.py --mode retrieval
    python eval/run_eval.py --mode guardrails

Retrieval mode answers one question per gold entry: did a chunk that can actually answer
it reach the top-k? The unit of judgement is *answer-bearing*, so an answer-bearing
`source_id` (optionally narrowed by `scheme`/`doc_type`) inside the ranked hit list counts
as a hit. Rank is recorded so we can report MRR rather than a bare hit rate.

Guardrail mode runs `eval/questions_guardrail.json` through the real pipeline with a spy
LLM client, and asserts three things: intent classification matches expectation, the
advice/returns lexicons have zero hits in any emitted text, and no PII value ever reached
the client.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mf_rag.config import get_config  # noqa: E402
from mf_rag.retrieve import retrieve_with_trace  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
GUARDRAIL_PATH = Path(__file__).resolve().parent / "questions_guardrail.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing eval file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _answer_bearing(hit, gold: dict[str, Any]) -> bool:
    """Could this chunk answer the question? Source-level, filtered by the gold tags."""
    from mf_rag.sources import SOURCES

    source = next((s for s in SOURCES if s["source_id"] == hit.chunk.source_id), None)
    if source is None:
        return False
    if gold.get("scheme") and source["scheme"] != gold["scheme"]:
        return False
    if gold.get("doc_type") and source["doc_type"] != gold["doc_type"]:
        return False
    return True


def run_retrieval(top_k: int, min_score: float | None, entity_filter: bool) -> dict[str, Any]:
    data = _load(QUESTIONS_PATH)
    questions = data["questions"]
    per_question: list[dict[str, Any]] = []

    for gold in questions:
        started = time.perf_counter()
        hits, trace = retrieve_with_trace(
            gold["question"], k=top_k, min_score=min_score, entity_filter=entity_filter
        )
        elapsed = (time.perf_counter() - started) * 1000

        ranked = [h for h in hits if _answer_bearing(h, gold)]
        rank = next((i + 1 for i, h in enumerate(hits) if _answer_bearing(h, gold)), None)
        per_question.append(
            {
                "id": gold["id"],
                "question": gold["question"],
                "pass": rank is not None,
                "rank": rank,
                "hits_returned": len(hits),
                "top_score": trace["top_score"],
                "where_filter": trace["where_filter"],
                "fallback_unfiltered": trace["fallback_unfiltered"],
                "latency_ms": round(elapsed, 1),
                "expected_keywords": gold.get("expected_keywords", []),
                "top_chunk": (
                    {
                        "source_id": hits[0].chunk.source_id,
                        "score": round(hits[0].score, 4),
                        "section_heading": hits[0].chunk.section_heading,
                    }
                    if hits
                    else None
                ),
            }
        )
        mark = "PASS" if rank is not None else "FAIL"
        print(
            f"  [{mark}] {gold['id']:<34} rank={str(rank):<5} "
            f"top={trace['top_score']} n={len(hits)} {elapsed:6.0f}ms"
        )

    passed = sum(1 for row in per_question if row["pass"])
    reciprocals = [1.0 / row["rank"] for row in per_question if row["rank"]]
    summary = {
        "mode": "retrieval",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "top_k": top_k,
        "min_score": min_score if min_score is not None else get_config().retrieval.min_score,
        "entity_filter": entity_filter,
        "total": len(per_question),
        "passed": passed,
        "hit_rate": round(passed / len(per_question), 4),
        "hit@1": round(sum(1 for r in per_question if r["rank"] == 1) / len(per_question), 4),
        "mrr": round(sum(reciprocals) / len(per_question), 4),
        "max_latency_ms": round(max((r["latency_ms"] for r in per_question), default=0), 1),
        "questions": per_question,
    }
    print(
        f"\n  hit@{top_k} = {summary['hit_rate']:.3f} ({passed}/{len(per_question)})  "
        f"hit@1 = {summary['hit@1']:.3f}  MRR = {summary['mrr']:.3f}  "
        f"slowest = {summary['max_latency_ms']:.0f}ms"
    )
    return summary


def run_guardrails() -> dict[str, Any]:
    import re

    from mf_rag.answer import _lexicon_hits, ADVICE_LEXICON, RETURNS_LEXICON
    from mf_rag.guardrails import detect_pii
    from mf_rag.llm import SpyClient
    from mf_rag.pipeline import run_with_client

    data = _load(GUARDRAIL_PATH)
    cases = data["cases"]
    per_case: list[dict[str, Any]] = []
    pii_leaks: list[str] = []

    for case in cases:
        text = case["input"]
        expected = case["expected_intent"]
        spy = SpyClient()
        answer = run_with_client(text, llm=spy)

        # The contract is the pipeline's final intent, not a direct classify_intent() call:
        # L1 blocks PII before L2 ever runs, so a PII input is never classified.
        actual = answer.intent
        pii_names = detect_pii(text)

        if answer.is_refusal:
            # A refusal is a system-authored constant, not model prose. Check it is exactly
            # a known template; lexicon-scanning it would just re-find the word "recommend"
            # inside "I can't ... recommend a scheme".
            template_ok = _is_known_refusal(answer.text, answer.source_url or "")
            advice_hits: list[str] = []
            returns_hits: list[str] = []
        else:
            template_ok = True
            advice_hits = _lexicon_hits(ADVICE_LEXICON, answer.text)
            returns_hits = _lexicon_hits(RETURNS_LEXICON, answer.text)

        if spy.called and _pii_echoed(text, spy.last_prompt or ""):
            pii_leaks.append(case["id"])

        ok = actual == expected and template_ok and not advice_hits and not returns_hits
        per_case.append(
            {
                "id": case["id"],
                "input": _redact(text, case),
                "expected_intent": expected,
                "actual_intent": actual,
                "intent_ok": actual == expected,
                "pii_detected": pii_names,
                "llm_called": spy.called,
                "is_refusal": answer.is_refusal,
                "refusal_is_template": template_ok,
                "advice_lexicon_hits": advice_hits,
                "returns_lexicon_hits": returns_hits,
                "source_url": answer.source_url,
                "pass": ok,
            }
        )
        mark = "PASS" if ok else "FAIL"
        note = ""
        if not template_ok:
            note = "  <-- NOT A KNOWN TEMPLATE"
        elif advice_hits or returns_hits:
            note = "  <-- LEXICON LEAK"
        print(
            f"  [{mark}] {case['id']:<28} {expected:<14} -> {actual:<14} "
            f"llm={'yes' if spy.called else 'no ':<3} pii={','.join(pii_names) or '-':<22}{note}"
        )

    passed = sum(1 for row in per_case if row["pass"])
    intent_ok = sum(1 for row in per_case if row["intent_ok"])
    summary = {
        "mode": "guardrails",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(per_case),
        "passed": passed,
        "pass_rate": round(passed / len(per_case), 4),
        "intent_correct": intent_ok,
        "intent_accuracy": round(intent_ok / len(per_case), 4),
        "lexicon_leaks": [r["id"] for r in per_case if r["advice_lexicon_hits"] or r["returns_lexicon_hits"]],
        "pii_leaks": pii_leaks,
        "pii_cases": [r["id"] for r in per_case if r["expected_intent"] == "pii"],
        "llm_called_on_pii_cases": [
            r["id"] for r in per_case if r["expected_intent"] == "pii" and r["llm_called"]
        ],
        "cases": per_case,
    }
    print(
        f"\n  pass rate = {summary['pass_rate']:.3f} ({passed}/{len(per_case)})  "
        f"intent accuracy = {summary['intent_accuracy']:.3f}  "
        f"lexicon leaks = {len(summary['lexicon_leaks'])}  "
        f"PII reaching LLM = {len(pii_leaks)}"
    )
    if summary["llm_called_on_pii_cases"]:
        print(f"  FAIL: LLM was called on PII input: {summary['llm_called_on_pii_cases']}")
    return summary


def _is_known_refusal(text: str, url: str) -> bool:
    """True when `text` is one of the PRD 7 templates with an in-registry link."""
    from mf_rag.guardrails import REFUSAL_TEMPLATES
    from mf_rag.generate import SOURCE_URLS

    if url and url not in SOURCE_URLS.values():
        return False
    return any(
        text == template.format(link=url) for template in REFUSAL_TEMPLATES.values()
    ) or any(text == template.format(link=url) for template in _extra_templates())


def _extra_templates() -> list[str]:
    from mf_rag.prompts import PII_WARNING

    return [PII_WARNING]


def _pii_echoed(question: str, prompt: str) -> bool:
    """True if a PII-shaped value from the question appears in what the LLM was shown."""
    import re

    for value in re.findall(r"[A-Z]{5}\d{4}[A-Z]|\b[2-9]\d{11}\b|\b\d{10}\b|[\w.+-]+@[\w-]+\.[\w.]+", question):
        if value in prompt:
            return True
    return False


def _lexicon_hits(patterns, text: str) -> list[str]:
    from mf_rag.answer import _lexicon_hits as _impl

    return _impl(patterns, text)


def _redact(text: str, case: dict[str, Any]) -> str:
    import re

    if not case.get("pii_types"):
        return text
    return re.sub(r"[A-Z]{5}\d{4}[A-Z]|\d{10,12}|[\w.]+@[\w.]+", "<redacted>", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retrieval", "guardrails", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--no-entity-filter", action="store_true")
    args = parser.parse_args()

    config = get_config()
    results: dict[str, Any] = {}

    if args.mode in ("retrieval", "all"):
        print(f"\nRETRIEVAL EVAL  (top_k={args.top_k or config.retrieval.top_k}, "
              f"min_score={args.min_score if args.min_score is not None else config.retrieval.min_score}, "
              f"entity_filter={not args.no_entity_filter})")
        summary = run_retrieval(
            top_k=args.top_k or config.retrieval.top_k,
            min_score=args.min_score,
            entity_filter=not args.no_entity_filter,
        )
        path = config.outputs_dir / "retrieval_eval.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        results["retrieval"] = path
        print(f"  wrote {path}")

    if args.mode in ("guardrails", "all"):
        print("\nGUARDRAIL EVAL")
        summary = run_guardrails()
        path = config.outputs_dir / "guardrail_eval.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        results["guardrails"] = path
        print(f"  wrote {path}")

    for mode, path in results.items():
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        if s.get("hit_rate") is not None and s["hit_rate"] < 1.0:
            print(f"  note: {mode} hit rate is {s['hit_rate']} (< 1.0)")
        if s.get("intent_accuracy") is not None and s["intent_accuracy"] < 1.0:
            print(f"  note: {mode} intent accuracy is {s['intent_accuracy']} (< 1.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
