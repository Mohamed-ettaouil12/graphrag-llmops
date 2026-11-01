"""P01 — Scaffolding tests for the LLMOps layer (no GPU, no Neo4j required)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_llmops_modules_importable():
    import importlib

    for name in ("adapter", "router", "cache"):
        assert importlib.import_module(f"llmops.{name}") is not None


def test_core_dirs_have_gitkeep():
    for name in ("eval", "ops", "prompts", "reports"):
        assert (REPO_ROOT / name / ".gitkeep").exists(), f"missing {name}/.gitkeep"


def test_env_example_has_required_keys():
    lines = (REPO_ROOT / ".env.example").read_text().splitlines()
    required = ("MISTRAL_API_KEY=", "LANGFUSE_PUBLIC_KEY=", "LANGFUSE_SECRET_KEY=", "LANGFUSE_HOST=")
    for key in required:
        assert any(line.startswith(key) for line in lines), f"{key} missing in .env.example"


def test_makefile_targets():
    targets = ("install:", "test:", "lint:", "eval:", "bench:")
    content = (REPO_ROOT / "Makefile").read_text()
    for t in targets:
        assert t in content, f"target {t} missing in Makefile"


def test_git_ignores_env():
    if shutil.which("git") is None:
        import pytest
        pytest.skip("git not available")
    env_file = REPO_ROOT / ".env"
    existed = env_file.exists()
    if not existed:
        env_file.write_text("MISTRAL_API_KEY=dummy\n")
    try:
        code = subprocess.run(
            ["git", "check-ignore", ".env"], capture_output=True, check=False, cwd=REPO_ROOT
        ).returncode
        assert code == 0, "git check-ignore .env must succeed"
    finally:
        if not existed:
            env_file.unlink(missing_ok=True)


def test_git_ignores_scaffold_contents_but_keeps_gitkeep():
    if shutil.which("git") is None:
        import pytest
        pytest.skip("git not available")
    probe = REPO_ROOT / "ops" / "_probe_tmp.txt"
    if probe.exists():
        os.remove(probe)
    probe.write_text("tmp")
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "ops/_probe_tmp.txt"], capture_output=True, check=False, cwd=REPO_ROOT
        ).returncode == 0
        keep = subprocess.run(
            ["git", "check-ignore", "ops/.gitkeep"], capture_output=True, check=False, cwd=REPO_ROOT
        ).returncode != 0
        assert ignored, "ops/ contents must be gitignored"
        assert keep, "ops/.gitkeep must remain tracked"
    finally:
        if probe.exists():
            os.remove(probe)


def test_pipeline_state_importable_without_side_effects():
    import importlib

    state_mod = importlib.import_module("src.orchestrator.state")
    for req in ("PipelineState", "PipelineConfig", "Triplet"):
        assert hasattr(state_mod, req)
