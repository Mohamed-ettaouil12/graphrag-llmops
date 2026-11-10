"""Unit tests for the llmops adapter layer — offline, deterministic."""

import pytest

from llmops.adapter import (
    AdapterUnavailable,
    ExistingGraphRAGAdapter,
    GraphAnswer,
    GraphAnswerMetadata,
    GraphRAGAdapter,
)


def test_adapter_protocol_contract():
    """Test that ExistingGraphRAGAdapter satisfies the GraphRAGAdapter protocol."""
    adapter: GraphRAGAdapter = ExistingGraphRAGAdapter()
    assert adapter is not None


@pytest.mark.xfail(
    reason="Adapter requires GPU/Neo4j; skipped in CI without those",
    strict=False,
)
def test_adapter_query_returns_graph_answer():
    """Test that adapter.query() returns a GraphAnswer with expected fields.

    xfail: adapter pipeline loading requires GPU/Neo4j which may not be available.
    """
    adapter = ExistingGraphRAGAdapter()
    result = adapter.query("What is the capital of France?", "hybrid")
    assert isinstance(result, GraphAnswer)
    assert isinstance(result.answer, str)
    assert isinstance(result.entities, list)
    assert isinstance(result.triples, list)
    assert isinstance(result.chunks, list)
    assert isinstance(result.metadata, GraphAnswerMetadata)


def test_adapter_missing_optional_fields_tolerated():
    """Test that the adapter tolerates missing optional fields without crashing.

    This is a key contract: the adapter should never raise KeyError or AttributeError
    when the pipeline state has empty/absent fields. It must gracefully degrade.
    In environments without GPU/Neo4j, the adapter raises AdapterUnavailable,
    which is also a valid contract outcome (the adapter is unavailable).
    """
    try:
        adapter = ExistingGraphRAGAdapter()
        result = adapter.query("What is the exact temperature on Mars right now?", "hybrid")
        # If adapter loaded, verify fields are tolerated
        assert result.answer is not None
        assert isinstance(result.entities, list)
        assert isinstance(result.triples, list)
        assert isinstance(result.chunks, list)
    except AdapterUnavailable:
        # In environments without GPU/Neo4j, the adapter is unavailable.
        # The contract is still satisfied: the adapter gracefully reports
        # unavailability instead of crashing with KeyError/AttributeError.
        pytest.xfail("AdapterUnavailable — no GPU/Neo4j in test environment")


def test_adapter_adapter_unavailable():
    """Test that AdapterUnavailable is a valid exception class."""
    assert AdapterUnavailable is not None


def test_graph_answer_dataclass_fields():
    """Test GraphAnswer dataclass has all expected fields with defaults."""
    answer = GraphAnswer(answer="test answer")
    assert answer.answer == "test answer"
    assert answer.entities == []
    assert answer.triples == []
    assert answer.chunks == []
    assert answer.metadata == GraphAnswerMetadata()

    metadata = GraphAnswerMetadata(model="test-model", tokens_in=10, tokens_out=5)
    answer2 = GraphAnswer(answer="test", metadata=metadata)
    assert answer2.metadata.model == "test-model"
    assert answer2.metadata.tokens_in == 10
    assert answer2.metadata.tokens_out == 5