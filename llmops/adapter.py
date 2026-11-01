"""Adapter LLMOps autour du pipeline GraphRAG legacy qa-multisaut.

Réf. docs/audit_graphrag.md (P01) : le pipeline est appelé tel quel via les
surfaces existantes `src.orchestrator.graph.create_pipeline` et
`src.orchestrator.state.PipelineState/PipelineConfig`. Aucune logique du
pipeline n'est modifiée : l'adapter ne fait que traduire l'état final du
LangGraph en `GraphAnswer` normalisé, exploitable par les couches LLMOps
(routing, cache, tracing) sans dépendre des internals.

Un seul écart assumé par rapport à `run_pipeline` (graph.py:91) : le graphe
compilé est construit une seule fois, paresseusement, au premier `query()`
— exactement comme le fait `src/api/main.py:29` (lifespan). `run_pipeline`
reconstruisait le graphe (et rechargeait les modèles) à chaque appel.

Le mode (`hybrid` par défaut) n'a pas d'équivalent dans le legacy : il est
accepté et transmis dans `metadata.mode` sans changer le comportement.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.orchestrator.state import PipelineConfig, PipelineState

logger = logging.getLogger(__name__)

MODEL_LABEL = "Qwen/Qwen2.5-1.5B-Instruct (local, 4-bit NF4)"


@dataclass
class GraphAnswerMetadata:
    """Métadonnées d'exécution associées à une réponse GraphRAG."""

    mode: str = "hybrid"
    latency_ms: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphAnswer:
    """Réponse normalisée, agnostique du backend (Neo4j, NetworkX...)."""

    answer: str
    entities: list[str] = field(default_factory=list)
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    metadata: GraphAnswerMetadata = field(default_factory=GraphAnswerMetadata)


class GraphRAGAdapter(Protocol):
    """Contrat stable exposé par tout backend GraphRAG."""

    def query(self, question: str, mode: str = "hybrid") -> GraphAnswer:
        """Pose une question ; mode = hybrid|local|global (backend,str)."""
        ...


class AdapterUnavailable(RuntimeError):
    """Pipeline impossible à charger ou à exécuter (GPU/Neo4j/données)."""


def _estimate_tokens(text: str) -> int | None:
    """Estimation char/4 (aucun tokenizer partagé avec le legacy)."""
    if not text:
        return None
    return max(1, len(text) // 4)


class ExistingGraphRAGAdapter:
    """Enveloppe read-only du pipeline qa-multisaut.

    Entraîne un warning (champs laissés vides) quand l'état legacy ne
    contient pas d'entités ou de triplets, sans jamais restructurer le
    pipeline.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config if config is not None else PipelineConfig()
        self.graph = None

    def _ensure_graph(self):
        if self.graph is not None:
            return self.graph
        from src.orchestrator.graph import create_pipeline

        try:
            self.graph = create_pipeline(self.config)
        except Exception as exc:
            logger.error("pipeline load failed: %s", exc)
            raise AdapterUnavailable(f"pipeline load failed: {exc}") from exc
        return self.graph

    def query(self, question: str, mode: str = "hybrid") -> GraphAnswer:
        """Exécute le pipeline legacy et normalise l'état final."""
        start = time.perf_counter()
        graph = self._ensure_graph()

        initial_state = PipelineState(
            question=question, config=self.config.model_dump()
        )
        try:
            result = graph.invoke(initial_state)
        except Exception as exc:
            logger.error("pipeline query failed: %s", exc)
            raise AdapterUnavailable(f"pipeline query failed: {exc}") from exc
        if isinstance(result, dict):
            result = PipelineState(**result)

        latency_ms = (time.perf_counter() - start) * 1000.0

        entities = list(result.question_entities)
        if not entities:
            logger.warning(
                "pipeline returned no entities; field left empty (legacy behavior)"
            )

        triples = [(t.h, t.r, t.t) for t in result.topk_triplets]
        chunks = [
            {
                "doc": t.source_title or "",
                "page": None,
                "text": t.sentence or f"({t.h}, {t.r}, {t.t})",
                "score": t.score,
            }
            for t in result.topk_triplets
        ]
        if not triples:
            logger.warning(
                "pipeline returned no top-k triplets; triples/chunks left empty"
            )

        raw_response = result.generation_log.get("raw_response", "")
        metadata = GraphAnswerMetadata(
            mode=mode,
            latency_ms=round(latency_ms, 3),
            tokens_in=_estimate_tokens(result.prompt),
            tokens_out=_estimate_tokens(raw_response or result.answer),
            model=MODEL_LABEL,
            extras={
                "abstained": result.abstained,
                "abstention_reason": result.abstention_reason,
                "retrieval_method": result.retrieval_method,
                "note": (
                    "tokens estimés (chars/4) ; llm_path ignoré par le legacy "
                    "et le mode non interprété (audit G9, pipeline inchangé)"
                ),
            },
        )
        return GraphAnswer(
            answer=result.answer,
            entities=entities,
            triples=triples,
            chunks=chunks,
            metadata=metadata,
        )
