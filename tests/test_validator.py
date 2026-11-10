"""Unit tests for the dataset validator — offline, deterministic."""

from pathlib import Path

# Ensure repo root is on the path
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))

from scripts.validate_dataset import ValidationError, load_rows


def test_load_rows():
    """Test that load_rows can read the golden dataset."""
    dataset_path = _REPO_ROOT / "eval" / "datasets" / "golden_v1.jsonl"
    rows = load_rows(dataset_path)
    assert len(rows) >= 50


def test_load_rows_raises_on_missing():
    """Test that load_rows raises ValidationError on missing dataset."""
    try:
        load_rows(_REPO_ROOT / "nonexistent.jsonl")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass


def test_validate_row_schema():
    """Test that validate_row checks the required keys."""
    from scripts.validate_dataset import validate_row
    valid_row = {
        "id": "test-001",
        "question": "test question?",
        "expected_answer": "test answer",
        "expected_entities": ["entity1"],
        "expected_triples": [["e1", "r", "e2"]],
        "hops": 1,
        "mode": "local",
        "tags": ["tag1"],
    }
    try:
        validate_row(valid_row, 1)
        assert True
    except ValidationError:
        assert False, "Valid row should not raise"


def test_validate_row_missing_keys():
    """Test that validate_row raises on missing keys."""
    from scripts.validate_dataset import validate_row
    invalid_row = {
        "id": "test-001",
        # missing many required keys
    }
    try:
        validate_row(invalid_row, 1)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass


def test_distribution_counts():
    """Test that distribution counts match expected minimums."""
    from scripts.validate_dataset import (
        MIN_DISTRIBUTION,
        MIN_TOTAL,
        distribution,
        load_rows,
    )
    dataset_path = _REPO_ROOT / "eval" / "datasets" / "golden_v1.jsonl"
    rows = load_rows(dataset_path)
    counts = distribution(rows)
    assert len(rows) >= MIN_TOTAL
    for key, minimum in MIN_DISTRIBUTION.items():
        assert counts.get(key, 0) >= minimum, f"{key}: {counts.get(key, 0)} < {minimum}"


def test_check_dataset():
    """Test that check_dataset validates the golden dataset successfully."""
    from scripts.validate_dataset import check_dataset
    dataset_path = _REPO_ROOT / "eval" / "datasets" / "golden_v1.jsonl"
    counts = check_dataset(dataset_path)
    assert len(counts) > 0