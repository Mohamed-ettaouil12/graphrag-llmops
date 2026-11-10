"""Unit tests for llmops pricing math — offline, deterministic."""

from llmops.pricing import (
    MISTRAL_LARGE_INPUT_COST_PER_MILLION,
    MISTRAL_LARGE_OUTPUT_COST_PER_MILLION,
    MISTRAL_SMALL_INPUT_COST_PER_MILLION,
    MISTRAL_SMALL_OUTPUT_COST_PER_MILLION,
    QWEN_1_5B_4BIT_COST_PER_MILLION,
    TYPICAL_COST_JUDGE_LARGE,
    TYPICAL_COST_JUDGE_SMALL,
    TYPICAL_COST_LOCAL,
    compute_cost_judge,
    compute_cost_local,
)


def test_pricing_constants_exist():
    """Verify all pricing constants are defined and positive."""
    assert MISTRAL_LARGE_INPUT_COST_PER_MILLION > 0
    assert MISTRAL_LARGE_OUTPUT_COST_PER_MILLION > 0
    assert MISTRAL_SMALL_INPUT_COST_PER_MILLION > 0
    assert MISTRAL_SMALL_OUTPUT_COST_PER_MILLION > 0
    assert QWEN_1_5B_4BIT_COST_PER_MILLION == 0.0  # local model, no API cost


def test_compute_cost_judge_mistral_large():
    """Test compute_cost_judge with mistral-large model."""
    cost = compute_cost_judge(1_000_000, 1_000_000, model="mistral-large")
    expected = MISTRAL_LARGE_INPUT_COST_PER_MILLION + MISTRAL_LARGE_OUTPUT_COST_PER_MILLION
    # Use rounding to avoid floating point issues
    assert round(cost, 6) == round(expected, 6)


def test_compute_cost_judge_mistral_small():
    """Test compute_cost_judge with mistral-small model."""
    cost = compute_cost_judge(1_000_000, 1_000_000, model="mistral-small")
    expected = MISTRAL_SMALL_INPUT_COST_PER_MILLION + MISTRAL_SMALL_OUTPUT_COST_PER_MILLION
    assert round(cost, 6) == round(expected, 6)


def test_compute_cost_judge_small_inputs():
    """Test compute_cost_judge with small token counts."""
    cost = compute_cost_judge(100, 50, model="mistral-small")
    expected = 0.30 * (100 / 1_000_000) + 0.90 * (50 / 1_000_000)
    assert round(cost, 6) == round(expected, 6)


def test_compute_cost_local():
    """Test compute_cost_local for local Qwen 1.5B 4-bit inference."""
    cost = compute_cost_local(1000)
    assert cost == 0.0


def test_typical_costs_are_nonnegative():
    """Verify typical cost constants are non-negative (or zero for local)."""
    assert TYPICAL_COST_JUDGE_LARGE >= 0
    assert TYPICAL_COST_JUDGE_SMALL >= 0
    assert TYPICAL_COST_LOCAL == 0.0  # local, no API cost


def test_cost_model_is_deterministic():
    """Verify that cost computation is deterministic (same inputs -> same output)."""
    cost1 = compute_cost_judge(500, 30, model="mistral-large")
    cost2 = compute_cost_judge(500, 30, model="mistral-large")
    assert cost1 == cost2