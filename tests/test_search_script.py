"""Smoke tests for scripts/search.py.

The script is a thin CLI over retrieve_with_trace(), so these check the two things that
can break silently: that it stays runnable, and that the trace it prints actually
contains the fields the output claims to show.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "search.py"

from conftest import needs_model

TRACE_FIELDS_SHOWN = (
    "entity_detected",
    "where_filter",
    "where_filter_text",
    "fallback_unfiltered",
    "min_score",
    "top_k",
    "candidates_considered",
    "after_dedupe",
    "hits",
)


def _load_search_module():
    spec = importlib.util.spec_from_file_location("search_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_runs_without_touching_the_index():
    """--help must not require the model or the built index."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "--no-entity-filter" in proc.stdout


def test_module_import_is_side_effect_free():
    """Importing must not build, embed or print, so it stays cheap to test."""
    module = _load_search_module()
    assert callable(module.main)
    assert callable(module.show)
    assert callable(module._is_official)


def test_is_official_reads_the_real_registry():
    module = _load_search_module()
    assert module._is_official("hdfcfund_small_cap_official") == "yes"
    assert module._is_official("hdfc_small_cap_scheme_page") == "no"


@needs_model
def test_json_mode_emits_every_field_the_human_output_prints(capsys):
    """If the trace grows a field, the printer and this list must change together."""
    module = _load_search_module()
    args = module.argparse.Namespace(
        query=[], top_k=None, min_score=None, no_entity_filter=False, json=True
    )
    trace = module.show("What is the exit load of HDFC Small Cap Fund?", args)
    for field in TRACE_FIELDS_SHOWN:
        assert field in trace, f"trace is missing {field!r}, which show() prints"
    assert json.loads(capsys.readouterr().out)["trace"] == trace


@needs_model
def test_no_entity_filter_is_reachable_and_actually_disables_it():
    module = _load_search_module()
    args = module.argparse.Namespace(
        query=[], top_k=3, min_score=None, no_entity_filter=True, json=False
    )
    trace = module.show("Exit load of HDFC Small Cap Fund?", args)
    assert trace["entity_detected"] is None
    assert trace["where_filter"] is None
    assert trace["fallback_unfiltered"] is False


@pytest.mark.parametrize("bad", [["--min-score", "not-a-number"]])
def test_bad_flags_fail_loudly(bad):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *bad], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode != 0
