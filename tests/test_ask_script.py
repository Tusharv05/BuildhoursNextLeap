"""Smoke tests for scripts/ask.py.

ask.py is a thin CLI over pipeline.query(), so the things worth pinning are the ones that
fail silently rather than loudly.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "ask.py"


def _load():
    spec = importlib.util.spec_from_file_location("ask_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_runs_without_touching_the_network():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "--debug" in proc.stdout


def test_import_is_side_effect_free():
    module = _load()
    assert callable(module.main)
    assert callable(module.ask)


def test_style_call_returns_and_echo_prints(capsys):
    """Regression: Style.__call__ only *returns* a string, it does not print.

    The first version of this script used `style(text, code)` as a display statement, which
    formatted the string and threw it away. Every header, prompt label, source line and
    debug line vanished and the tool looked like it had printed only the answer body. The
    distinction is invisible until you check that echo() actually emits.
    """
    module = _load()
    style = module.Style(enabled=False)

    returned = style("hello", module.BOLD)
    assert returned == "hello"
    assert capsys.readouterr().out == "", "__call__ must not print"

    style.echo("hello", module.BOLD)
    assert capsys.readouterr().out == "hello\n"


def test_disabled_style_emits_no_ansi_codes():
    module = _load()
    style = module.Style(enabled=False)
    assert "\033" not in style("x", module.RED)
    assert "\033" in module.Style(enabled=True)("x", module.RED)


def test_wrap_collapses_whitespace_and_indents():
    module = _load()
    out = module.wrap("a\n\n  b   c", indent="  ", width=20)
    assert out.startswith("  a b c")
    assert module.Style(False)("", "") == ""


@pytest.mark.parametrize("bad", [["--nope"]])
def test_unknown_flag_fails_loudly(bad):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *bad], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode != 0


def test_bare_is_a_recognised_flag():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120
    )
    assert "--bare" in proc.stdout


def test_bare_prompt_goes_to_stderr_not_stdout():
    """The REPL prompt must not contaminate piped stdout.

    Regression: `input(prompt)` writes the prompt to stdout, so `--bare` piped a `you> `
    into the middle of the answer text. `ask.py --bare > answers.txt` is supposed to give
    one clean answer per question, which is impossible if the prompt is on stdout.

    Piping "exit" means the loop quits before any question, so this stays fast and never
    calls the LLM.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--bare"],
        input="exit\n",
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "you>" not in proc.stdout, f"prompt leaked into stdout: {proc.stdout!r}"
    assert proc.stdout.strip() == "", f"expected empty stdout, got {proc.stdout!r}"
    assert "you>" in proc.stderr, "prompt should still be visible on the terminal"
