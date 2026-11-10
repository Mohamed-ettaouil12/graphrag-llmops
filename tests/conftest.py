"""Conftest for all tests — adds repo root to sys.path so llmops.* is importable."""
import sys
from pathlib import Path

# Add repo root (parent of tests/) to sys.path
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))