"""Adapter LLMOps autour du pipeline GraphRAG legacy qa-multisaut.

Réf. docs/audit_graphrag.md (P01) : le pipeline est appelé tel quel via les
surfaces existantes `src.orchestrator.graph.create_pipeline` et
`src.orchestrator.state.PipelineState/PipelineConfig`. Aucune logique du
pipeline n'est modifiée : l'adapter ne fait que traduire l'état final du
LangGraph en `GraphAnswer` normalisé, exploitable par les couches LLMOps
(routing, cache, tracing) sans dépendre des internals.

Instrumentation Langfuse : chaque `query()` produit un span complet avec
usage, coût, trace_id et tags. Le mode (`hybrid` par défaut) est transmis
dans `metadata.mode` sans changer le comportement.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from langfuse import Langfuse

from src.orchestrator.state import PipelineConfig, PipelineState

logger = logging.getLogger(__name__)

# ── Langfuse initialisation (dégradé silencieux si clés manquantes) ─────────

_langfuse = None


def _get_langfuse() -> Langfuse:
    """Retourne une instance Langfuse configurée depuis les env vars."""
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    try:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if public_key and secret_key:
            _langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
        else:
            # Langfuse désactivé mais on retourne quand même un objet
            # factice qui ne plantera pas.

            class _Disabled:
                def start_span(self, *args, **kwargs):
                    return None

                def update(self, *args, **kwargs):
                    pass

            _langfuse = _Disabled()  # type: ignore
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        # Erreur critique lors de l'initialisation : on note et on continue
        # en mode dégradé. L'exception est loggée mais n'arrête pas le
        # démarrage de l'application.
        logger.error("Failed to initialise Langfuse: %s", exc)
        class _Disabled:
            def start_span(self, *args, **kwargs):
                return None

            def update(self, *args, **kwargs):
                pass

        _langfuse = _Disabled()  # type: ignore
    return _langfuse


MODEL_LABEL = "Qwen/Qwen2.5-1.5B-Instruct (local, 4-bit NF4)"


def _truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars for trace safety."""
    if not text:
        return ""
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


@dataclass
class GraphAnswerMetadata:
    """Métadonnées d'exécution associées à une réponse GraphRAG."""

    mode: str = "hybrid"
    latency_ms: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    # Langfuse fields
    trace_id: str = ""
    session_id: str = ""
    # Usage & cost
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float = 0.0


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
    """Enveloppe read-only du pipeline qa-multisaut avec instrumentation Langfuse.

    Entraîne un warning (champs laissés vides) quand l'état legacy ne
    contient pas d'entités ou de triplets, sans jamais restructurer le
    pipeline. Chaque appel à `query()` produit un span Langfuse complet.
    """

    def __init__(self, config: PipelineConfig | None = None, session_id: str = ""):
        self.config = config if config is not None else PipelineConfig()
        self.graph = None
        self.session_id = session_id or "default"
        self.lf = _get_langfuse()
        self._trace_id: str = ""

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

    def _span(self, name: str, **kwargs):
        """Démarrage d'un span Langfuse (no-op si désactivé)."""
        if self.lf and hasattr(self.lf, "start_span"):
            return self.lf.start_span(name=name, **kwargs)
        return None

    def _end_span(self, span, **kwargs):
        """Finition d'un span Langfuse (no-op si désactivé)."""
        if self.lf and hasattr(self.lf, "update"):
            try:
                self.lf.update(span, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to update Langfuse span: %s", exc)

    def query(self, question: str, mode: str = "hybrid") -> GraphAnswer:
        """Exécute le pipeline legacy et normalise l'état final.

        Chaque appel produit un trace Langfuse contenant :
        - trace_id (transmis dans la réponse API)
        - tags: env=dev|prod
        - entities touchées, triples utilisés, mode
        - consommation tokens + coût USD
        - latence ms
        """
        # Langfuse trace début
        trace = self._span(
            "graphrag_query",
            user_id=self.session_id or None,
            session_id=self.session_id or None,
            input=question,
            # On ne met pas le plein texte du chunk dans le trace ;
            # on ne transmet que la question.
        )

        graph = self._ensure_graph()
        start = time.perf_counter()

        initial_state = PipelineState(
            question=question, config=self.config.model_dump()
        )
        try:
            result = graph.invoke(initial_state)
        except Exception as exc:
            logger.error("pipeline query failed: %s", exc)
            # Span d'erreur
            if trace:
                self._end_span(
                    trace,
                    output=f"ERROR: {exc}",
                    level="error",
                )
            raise AdapterUnavailable(f"pipeline query failed: {exc}") from exc
        if isinstance(result, dict):
            result = PipelineState(**result)

        latency_ms = (time.perf_counter() - start) * 1000.0

        entities = list(result.question_entities)
        if not entities:
            logger.warning("pipeline returned no entities; field left empty (legacy behavior)")

        triples = [(t.h, t.r, t.t) for t in result.topk_triplets]

        # Chunks : textes tronqués à 500 caractères max pour la sécurité
        chunks = [
            {
                "doc": t.source_title or "",
                "page": None,
                "text": _truncate(t.sentence or f"({t.h}, {t.r}, {t.t})", 500),
                "score": t.score,
            }
            for t in result.topk_triplets
        ]
        if not triples:
            logger.warning("pipeline returned no top-k triplets; triples/chunks left empty")

        raw_response = result.generation_log.get("raw_response", "")
        tokens_in = _estimate_tokens(result.prompt)
        tokens_out = _estimate_tokens(raw_response or result.answer)

        # Coût approximé (local 4-bit : $0.00 par token API, mais on compte
        # les tokens quand même pour la traçabilité)
        cost_usd = 0.0  # modèle local 4-bit, pas de coût API

        # Construction métadonnées avec fields Langfuse
        trace_id = self._trace_id or ""
        metadata = GraphAnswerMetadata(
            mode=mode,
            latency_ms=round(latency_ms, 3),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=MODEL_LABEL,
            extras={
                "abstained": result.abstained,
                "abstention_reason": result.abstention_reason,
                "retrieval_method": result.retrieval_method,
                "note": "tokens estimés (chars/4) ; modèle local 4-bit ; chiffrement des textes de chunk à 500 car.",
            },
            trace_id=trace_id,
            session_id=self.session_id,
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            cost_usd=cost_usd,
        )

        # Langfuse : mettre à jour le span avec usage + coût
        if trace:
            self._end_span(
                trace,
                usage={
                    "prompt_tokens": tokens_in or 0,
                    "completion_tokens": tokens_out or 0,
                    "total_tokens": (tokens_in or 0) + (tokens_out or 0),
                },
                output=result.answer,
                level="completed",
                status_code=200,
                metadata={
                    "entities": entities,
                    "triples": triples,
                    "retrieval_method": result.retrieval_method,
                    "abstained": result.abstained,
                },
                # Coût agrégé par requête
                cost=cost_usd,
            )

        self._trace_id = trace_id

        return GraphAnswer(
            answer=result.answer,
            entities=entities,
            triples=triples,
            chunks=chunks,
            metadata=metadata,
        )