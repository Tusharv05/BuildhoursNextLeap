"""Shared pytest fixtures. Phase 0 tests run with no network and no models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mf_rag.config import load_config  # noqa: E402
from mf_rag.models import Chunk, SourceDoc  # noqa: E402


def _model_available() -> bool:
    """Can the embedding model actually be loaded right now?"""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    from mf_rag.embedder import _load_model

    try:
        _load_model(load_config(PROJECT_ROOT / "config.yaml").embedding.model)
    except Exception:
        return False
    return True


def needs_model(func):
    """Tag a test as needing the embedding model, and skip it when unavailable.

    Composed as a function rather than `mark & mark` because MarkDecorator does not
    support `&`. The `model` marker also lets `pytest -m "not model"` deselect these
    explicitly, which is how CI runs an offline subset.
    """
    func = pytest.mark.model(func)
    return pytest.mark.skipif(
        not _model_available(), reason="sentence-transformers model unavailable (offline?)"
    )(func)


@pytest.fixture(scope="session", autouse=True)
def _no_ambient_llm_credentials():
    """Make the suite hermetic against the developer's own .env.

    `config.load_config` calls `load_dotenv(PROJECT_ROOT / ".env")`, so as soon as
    someone puts a real key in .env, `get_llm()` stops returning EchoClient and returns a
    live OpenAICompatClient. Any test that calls `answer_raw()` without injecting a client
    would then make a real, billable HTTP request -- and its result would depend on the
    network and on model nondeterminism.

    That is not hypothetical: three tests in test_generate.py did exactly this and failed
    with `LLMError: ... HTTPError` the moment a key was present.

    Clearing the variables once is NOT enough, because `load_config` re-runs
    `load_dotenv` on every call (config.py:167), which would put the key straight back.
    So the loader itself is stubbed out for the session, and the variables are cleared on
    top of that. The suite then cannot reach a provider no matter what is in .env, which
    is what llm.py's docstring already claims ("EchoClient is the only client the test
    suite uses"). Tests that need a key set it themselves via monkeypatch, which still
    works because monkeypatch runs after this fixture.
    """
    import mf_rag.config as config_mod

    names = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    saved = {name: os.environ.pop(name, None) for name in names}
    real_loader = config_mod.load_dotenv
    config_mod.load_dotenv = lambda *args, **kwargs: False
    try:
        yield
    finally:
        config_mod.load_dotenv = real_loader
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_config(PROJECT_ROOT / "config.yaml")


@pytest.fixture
def sample_source_doc() -> SourceDoc:
    return SourceDoc(
        source_id="hdfc_elss_scheme_page",
        source_url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        scheme="HDFC ELSS Tax Saver Fund - Direct Plan Growth",
        category="ELSS",
        doc_type="scheme_page",
        title="HDFC ELSS Tax Saver Fund - Direct Plan Growth",
        text="Lock-in period: 3 years. Minimum SIP: Rs 500. Exit load: Nil.",
        fetched_at="2026-09-27",
        is_official=False,
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="abc123",
        source_id="hdfc_elss_scheme_page",
        source_url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        scheme="HDFC ELSS Tax Saver Fund - Direct Plan Growth",
        category="ELSS",
        doc_type="scheme_page",
        section_heading="Lock-in period",
        chunk_index=0,
        text="Lock-in period: 3 years.",
        embed_text="HDFC ELSS Tax Saver Fund - Direct Plan Growth — scheme_page — Lock-in period\n\nLock-in period: 3 years.",
        token_count=14,
        fetched_at="2026-09-27",
    )
