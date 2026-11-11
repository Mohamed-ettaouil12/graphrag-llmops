"""Environment verification for LLM ops.

Idempotent: exits 0 only if ALL checks pass.
Budget: 1 real Mistral call + 1 Langfuse write + 1 read.
Never prints secret values.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
GITIGNORE_PATH = ROOT / ".gitignore"

# ── Load .env ─────────────────────────────────────────────────────────
load_dotenv(dotenv_path=ENV_PATH, override=False)

# Read env vars
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


# ── Helpers ───────────────────────────────────────────────────────────

def _env_present() -> bool:
    """Return True if all four required env vars are non-empty."""
    return all(
        v for v in [
            MISTRAL_API_KEY,
            LANGFUSE_PUBLIC_KEY,
            LANGFUSE_SECRET_KEY,
            LANGFUSE_HOST,
        ]
    )


def _git_ignores_env() -> bool:
    """Return True if .gitignore (or git) ignores .env."""
    try:
        with open(GITIGNORE_PATH) as f:
            return ".env" in f.read()
    except FileNotFoundError:
        return False


def _scan_git_history() -> list[str]:
    """Scan git log -p for API key patterns; return list of match lines (redacted)."""
    matches: list[str] = []
    try:
        result = os.popen("git log -p --all").read()
        # Patterns we never want in history
        patterns = [
            r"MISTRAL_API_KEY\s*=\s*[^\s]+",
            r"LANGFUSE_SECRET_KEY\s*=\s*[^\s]+",
            r"MISTRAL_API_KEY\s*=",
            r"LANGFUSE_PUBLIC_KEY\s*=",
        ]
        for pat in patterns:
            for m in re.finditer(pat, result, re.MULTILINE):
                # Record the line but REDACT the value
                line = m.group(0)
                # Replace anything after the = sign with ***REDACTED***
                stripped = re.sub(r"=.*$", "= ***REDACTED***", line)
                matches.append(stripped)
    except Exception:
        pass
    return matches


# ── Check 1: .env exists and has keys ─────────────────────────────────

def check_env_exists() -> tuple[bool, str]:
    if not ENV_PATH.exists():
        return False, ".env file missing at repo root"
    if not _env_present():
        missing = [k for k, v in zip(
            ["MISTRAL_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"],
            [MISTRAL_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST],
        ) if not v
        ]
        return False, f".env missing keys: {', '.join(missing)}"
    return True, ".env present with all required keys"


# ── Check 2: .env is git-ignored ──────────────────────────────────────

def check_env_git_ignored() -> tuple[bool, str]:
    if not _git_ignores_env():
        return False, ".env is NOT git-ignored — add '.env' to .gitignore"
    return True, ".env is git-ignored"


# ── Check 3: Mistral key valid (one real call) ────────────────────────

def check_mistral_key() -> tuple[bool, str]:
    """ONE call to mistral/mistral-small-latest ('ping')."""
    import litellm

    # Minimal ping — one token prompt, expect one completion token
    try:
        response = litellm.completion(
            model="mistral/mistral-small-latest",
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
        )
        # Extract model and usage; NEVER print key material
        model = getattr(response, "model", "unknown")
        usage = getattr(response, "usage", None)
        if usage:
            pt = getattr(usage, "prompt_tokens", 0)
            ct = getattr(usage, "completion_tokens", 0)
            total = getattr(usage, "total_tokens", pt + ct)
        else:
            pt = ct = total = 0

        info = (
            f"Mistral model: {model}; "
            f"tokens: prompt={pt}, completion={ct}, total={total}"
        )
        return True, info
    except Exception as exc:
        return False, f"Mistral call failed: {exc}"


# ── Check 4: Langfuse keys valid (write + read) ──────────────────────

def check_langfuse_keys() -> tuple[bool, str]:
    """Ingest one test trace, fetch it back through Langfuse API, assert visible."""
    from langfuse import Langfuse

    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        return False, "Langfuse keys not configured in .env"

    try:
        lf = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )

        # Write one test trace
        trace = lf.start_trace(
            name="check-env:test-trace",
            input="Environment verification ping",
        )

        # Add a simple message
        trace.update(
            output="check-env: Langfuse key ingestion test",
            metadata={"check": "env"},
        )

        # Fetch the trace back
        fetched = lf.get_traces(limit=1)
        if not fetched:
            return False, "Langfuse: could not fetch any traces after write"

        fetched_trace = fetched[0]
        trace_id_match = fetched_trace.trace_id == trace.trace_id
        has_output = fetched_trace.output is not None

        if trace_id_match and has_output:
            return True, "Langfuse: write + read successful, trace visible in UI"
        else:
            reasons = []
            if not trace_id_match:
                reasons.append("trace_id mismatch")
            if not has_output:
                reasons.append("no output on fetched trace")
            return False, f"Langfuse mismatch: {', '.join(reasons)}"

    except Exception as exc:
        return False, f"Langfuse operation failed: {exc}"


# ── Check 5: Git history secret scan ──────────────────────────────────

def check_git_history() -> tuple[bool, str]:
    """Scan git log -p for API key patterns; report PASS/FAIL without printing values."""
    matches = _scan_git_history()
    if not matches:
        return True, "Git history: no API key patterns found — PASS"
    # We have matches — FAIL but don't print the values
    count = len(matches)
    return False, f"Git history: {count} secret pattern(s) found in git history — FAIL (run 'git filter-repo' or rewrite history)"


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    results: dict[str, tuple[bool, str]] = {
        "env_exists": check_env_exists(),
        "env_git_ignored": check_env_git_ignored(),
        "mistral_key": check_mistral_key(),
        "langfuse_keys": check_langfuse_keys(),
        "git_history": check_git_history(),
    }

    all_pass = all(r[0] for r in results.values())

    # Print summary (no secret values)
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)

    order = [
        "env_exists",
        "env_git_ignored",
        "mistral_key",
        "langfuse_keys",
        "git_history",
    ]
    for key in order:
        ok, msg = results[key]
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {key}: {msg}")

    print("=" * 60)
    if all_pass:
        print("RESULT: All checks PASSED — environment ready")
    else:
        print("RESULT: Some checks FAILED — see above")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())