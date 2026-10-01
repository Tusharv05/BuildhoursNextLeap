"""Check that .env is present, sane, and actually reaches the provider.

Prints the key's shape, never the key itself. Run this after editing .env:

    .\\.venv\\Scripts\\python.exe scripts\\check_llm.py

With --live it makes one real API call (costs a fraction of a cent). Without it, the
script only reports local configuration, so it is safe to run offline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ENV_PATH = ROOT / ".env"
EXAMPLE_PATH = ROOT / ".env.example"


def mask(value: str) -> str:
    """Show enough to identify a key, never enough to use one."""
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return f"{'*' * len(value)} ({len(value)} chars)"
    return f"{value[:4]}...{value[-2:]} ({len(value)} chars)"


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader, so this works even if dotenv misbehaves."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify .env / LLM configuration")
    parser.add_argument(
        "--live", action="store_true", help="make one real API call to confirm the key works"
    )
    args = parser.parse_args()

    import mf_rag.config as config_mod
    from mf_rag.llm import LLMError, OpenAICompatClient, describe_client, get_llm

    # get_config() is what triggers load_dotenv, so call it before reading os.environ.
    cfg = config_mod.get_config()

    print("=" * 66)
    print(f".env present      : {ENV_PATH.is_file()}")
    print(f".env.example      : {EXAMPLE_PATH.is_file()}")
    print(f"git-ignored       : {'.env' in (ROOT / '.gitignore').read_text(encoding='utf-8')}")
    print(f"provider (config) : {cfg.generation.provider}")
    print(f"model (config)     : {cfg.generation.model}")
    print(f"temperature        : {cfg.generation.temperature}")
    print(f"max_tokens         : {cfg.generation.max_tokens}")

    on_disk = parse_env_file(ENV_PATH)
    print()
    print("--- what the code will actually use ---")
    print(f"LLM_API_KEY  {mask(os.getenv('LLM_API_KEY') or '')}")
    print(f"LLM_BASE_URL {os.getenv('LLM_BASE_URL') or '(blank -> default OpenAI endpoint)'}")
    print(f"LLM_MODEL    {os.getenv('LLM_MODEL') or f'(blank -> config: {cfg.generation.model})'}")

    # Only a *non-empty* key in the file that never reached the environment is a problem.
    # A blank LLM_API_KEY= is the intended starting state, not a load failure.
    if on_disk.get("LLM_API_KEY") and not os.getenv("LLM_API_KEY"):
        print()
        print("  ! .env holds a non-empty LLM_API_KEY but it is not in the environment, so")
        print("    load_dotenv did not run for this process. Check config.py:167 is reached.")

    client, degraded = get_llm()
    print()
    print(f"client           : {describe_client(client)}")
    print(f"degraded mode     : {degraded}")

    if degraded:
        print()
        print("Running in DEGRADED mode (extractive answers, no LLM). To fix, one of:")
        print("  1. LLM_API_KEY is blank or missing in .env")
        print("  2. provider is 'echo' in config.yaml (set it to openai_compatible)")
        if not args.live:
            print()
            print("Nothing else to check offline. Re-run with --live once a key is set.")
        return 0

    if not args.live:
        print()
        print("Configuration looks complete. Re-run with --live to confirm the key works:")
        print("  .\\.venv\\Scripts\\python.exe scripts\\check_llm.py --live")
        return 0

    print()
    print("--- live call ---")
    try:
        reply = client.complete(
            [
                {"role": "system", "content": "Reply with the single word: ready"},
                {"role": "user", "content": "ping"},
            ],
            max_tokens=10,
        )
    except LLMError as exc:
        print(f"FAILED: {exc}")
        print()
        print("Common causes:")
        print("  - key is wrong, or has not been activated yet")
        print("  - LLM_BASE_URL points at the wrong path for that provider")
        print("  - the provider needs a different model id than config.yaml names")
        print("  - no credit on the account")
        return 1

    print(f"OK, provider replied: {reply.strip()[:80]!r}")
    print()
    print("Now try a real question end to end:")
    print("  .\\.venv\\Scripts\\python.exe -c \"from mf_rag.pipeline import query; a=query('"
          "What is the exit load of HDFC Small Cap Fund?'); print(a.text)\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
