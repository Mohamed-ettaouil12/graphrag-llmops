"""P02 — intégration app/api.py avec FakeAdapter (aucun LLM réel lancé)."""
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from llmops.adapter import (
    AdapterUnavailable,
    ExistingGraphRAGAdapter,
    GraphAnswer,
    GraphAnswerMetadata,
)

QUESTION = "Which city is the capital of the country where the Eiffel Tower is located?"


class FakeAdapter:
    """Adapter factice : réponse canned, trace des appels."""

    def __init__(self, latency_ms: float = 2.7):
        self.latency_ms = latency_ms
        self.calls: list[tuple[str, str]] = []

    def query(self, question: str, mode: str = "hybrid") -> GraphAnswer:
        self.calls.append((question, mode))
        return GraphAnswer(
            answer="Paris",
            entities=["eiffel tower", "paris"],
            triples=[("eiffel tower", "location", "paris")],
            chunks=[{"doc": "Paris", "page": None, "text": "Paris is the capital of France.", "score": 0.9}],
            metadata=GraphAnswerMetadata(
                mode=mode,
                latency_ms=self.latency_ms,
                tokens_in=42,
                tokens_out=3,
                model="fake-model",
            ),
        )


class UnavailableAdapter:
    def query(self, question: str, mode: str = "hybrid") -> GraphAnswer:
        raise AdapterUnavailable("no gpu/models on this box")


class BrokenAdapter:
    def query(self, question: str, mode: str = "hybrid") -> GraphAnswer:
        raise ValueError("boom")


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture
def client(fake_adapter) -> TestClient:
    return TestClient(create_app(fake_adapter))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["adapter"] == "FakeAdapter"


def test_query_happy_path(client, fake_adapter):
    response = client.post("/query", json={"question": QUESTION})
    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == "Paris"
    assert data["entities"] == ["eiffel tower", "paris"]
    assert data["triples"] == [{"h": "eiffel tower", "r": "location", "t": "paris"}]
    assert len(data["chunks"]) == 1
    chunk = data["chunks"][0]
    assert chunk["doc"] == "Paris" and chunk["page"] is None and chunk["text"]

    metadata = data["metadata"]
    assert metadata["mode"] == "hybrid"
    assert metadata["model"] == "fake-model"
    assert metadata["latency_ms"] == pytest.approx(2.7)
    assert metadata["tokens_in"] == 42 and metadata["tokens_out"] == 3
    assert uuid.UUID(metadata["trace_id"]).version == 4

    assert fake_adapter.calls == [(QUESTION, "hybrid")]


def test_query_mode_passthrough(client, fake_adapter):
    response = client.post("/query", json={"question": QUESTION, "mode": "global"})
    assert response.status_code == 200
    assert fake_adapter.calls == [(QUESTION, "global")]
    assert response.json()["metadata"]["mode"] == "global"


def test_query_validation_error(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_adapter_unavailable():
    client = TestClient(create_app(UnavailableAdapter()))
    response = client.post("/query", json={"question": QUESTION})
    assert response.status_code == 503
    assert "adapter unavailable" in response.json()["detail"]


def test_query_internal_error():
    client = TestClient(create_app(BrokenAdapter()))
    response = client.post("/query", json={"question": QUESTION}, )
    assert response.status_code == 500
    assert "query failed" in response.json()["detail"]
    assert "boom" in response.json()["detail"]


def test_query_latency_fallback_when_adapter_reports_zero():
    client = TestClient(create_app(FakeAdapter(latency_ms=0.0)))
    response = client.post("/query", json={"question": QUESTION})
    assert response.status_code == 200
    assert response.json()["metadata"]["latency_ms"] > 0


def test_existing_adapter_lazy_loading():
    adapter = ExistingGraphRAGAdapter()
    assert adapter.graph is None
