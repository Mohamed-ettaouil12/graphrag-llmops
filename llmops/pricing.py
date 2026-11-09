"""Pricing table for LLM cost tracking.

All prices are per 1M tokens. Use these to compute approximate cost from
token counts collected during benchmarking or RAGAS judging.

Reference: Mistral API pricing (2026-11-09).
"""

# Mistral large inference (judge model)
MISTRAL_LARGE_INPUT_COST_PER_MILLION = 2.00  # USD per 1M prompt tokens
MISTRAL_LARGE_OUTPUT_COST_PER_MILLION = 6.00  # USD per 1M completion tokens

# Mistral small (cheaper judge, for graph metrics)
MISTRAL_SMALL_INPUT_COST_PER_MILLION = 0.30  # USD per 1M prompt tokens
MISTRAL_SMALL_OUTPUT_COST_PER_MILLION = 0.90  # USD per 1M completion tokens

# Qwen2.5-1.5B-Instruct 4-bit (local model, no API cost since it runs locally)
# Cost is captured as infrastructure cost only (electricity, etc.) — $0.00 per token
QWEN_1_5B_4BIT_COST_PER_MILLION = 0.00


def compute_cost_judge(prompt_tokens: int, completion_tokens: int, model: str = "mistral-large") -> float:
    """Compute cost for a single judge call via litellm/Mistral API.

    Args:
        prompt_tokens: number of prompt tokens
        completion_tokens: number of completion tokens
        model: model name, either "mistral-large" or "mistral-small"

    Returns:
        cost in USD
    """
    if model == "mistral-small":
        input_cost = (prompt_tokens / 1_000_000) * MISTRAL_SMALL_INPUT_COST_PER_MILLION
        output_cost = (completion_tokens / 1_000_000) * MISTRAL_SMALL_OUTPUT_COST_PER_MILLION
    else:
        input_cost = (prompt_tokens / 1_000_000) * MISTRAL_LARGE_INPUT_COST_PER_MILLION
        output_cost = (completion_tokens / 1_000_000) * MISTRAL_LARGE_OUTPUT_COST_PER_MILLION
    return round(input_cost + output_cost, 6)


def compute_cost_local(new_tokens: int) -> float:
    """Compute local infrastructure cost for a Qwen 1.5B 4-bit generation.

    Since the model runs locally (4-bit QLoRA on RTX 3050 Ti), there is no
    API cost. We only account for approximate electricity/infrastructure cost.

    Args:
        new_tokens: number of new tokens generated

    Returns:
        approximate cost in USD (typically <<$0.01 per query)
    """
    # Very rough estimate: ~0.001 USD per query for local 4-bit inference
    # on consumer GPU. This is deliberately near-zero since the model is
    # already deployed and the cost is fixed infrastructure.
    return round(new_tokens / 1_000_000 * QWEN_1_5B_4BIT_COST_PER_MILLION, 6)


# Convenience: cost per typical query (prompt + response)
TYPICAL_PROMPT_TOKENS = 500   # prompt + triplets context
TYPICAL_RESPONSE_TOKENS = 30  # answer + sources

TYPICAL_COST_JUDGE_LARGE = compute_cost_judge(TYPICAL_PROMPT_TOKENS, TYPICAL_RESPONSE_TOKENS, "mistral-large")
TYPICAL_COST_JUDGE_SMALL = compute_cost_judge(TYPICAL_PROMPT_TOKENS, TYPICAL_RESPONSE_TOKENS, "mistral-small")
TYPICAL_COST_LOCAL = compute_cost_local(TYPICAL_RESPONSE_TOKENS)

print(f"Typical judge (mistral-large) cost: ${TYPICAL_COST_JUDGE_LARGE:.6f} per query")
print(f"Typical judge (mistral-small) cost: ${TYPICAL_COST_JUDGE_SMALL:.6f} per query")
print(f"Typical local cost: ${TYPICAL_COST_LOCAL:.6f} per query")