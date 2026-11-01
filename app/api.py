"""Passerelle HTTP stable au-dessus de llmops/adapter.py (RF-05, RNF-01).

Lancement : uvicorn app.api:app --host 0.0.0.0 --port 8000
Endpoints : POST /query {question, mode?} -> preuve + métriques ; GET /health.
Les modèles ne sont pas chargés au démarrage (lazy, premier /query) ; ici,
une machine sans GPU/données renvoie 503 sans casser le service.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from llmops.adapter import (
    AdapterUnavailable,
    ExistingGraphRAGAdapter,
    GraphAnswer,
    GraphRAGAdapter,
)

API_VERSION = "2.0.0"
_SAFE_LOG_KEYS = ("trace_id", "mode", "latency_ms", "question_chars", "adapter", "status")


def _json_logger(name: str) -> logging.Logger:
    """Logger applicatif en JSON (jamais de secret, jamais de prompt brut)."""
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            for key in _SAFE_LOG_KEYS:
                if hasattr(record, key):
                    payload[key] = getattr(record, key)
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload, ensure_ascii=False)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg


logger = _json_logger("app.api")


class QueryIn(BaseModel):
    """Requête utilisateur."""

    question: str = Field(min_length=1, max_length=1000)
    mode: str = Field(default="hybrid", min_length=1, max_length=32)


class TripleDto(BaseModel):
    """Triplet de preuve (head, relation, tail)."""

    h: str
    r: str
    t: str


class ChunkDto(BaseModel):
    """Extrait cité (doc/page ; le legacy n'expose pas la page -> null)."""

    doc: str = ""
    page: int | None = None
    text: str = ""
    score: float | None = None


class MetaDto(BaseModel):
    """Métadonnées d'une /query ; extras du pipeline « bypassé »."""

    model_config = ConfigDict(extra="allow")

    mode: str = "hybrid"
    latency_ms: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str = ""
    trace_id: str = ""


class QueryOut(BaseModel):
    answer: str
    entities: list[str] = Field(default_factory=list)
    triples: list[TripleDto] = Field(default_factory=list)
    chunks: list[ChunkDto] = Field(default_factory=list)
    metadata: MetaDto
    abstained: bool = False


def _to_query_out(result: GraphAnswer, trace_id: str, fallback_latency_ms: float) -> QueryOut:
    """GraphAnswer (dataclass) -> schémas Pydantic de l'API."""
    latency_ms = result.metadata.latency_ms
    if latency_ms <= 0.0:
        latency_ms = fallback_latency_ms
    extras = dict(result.metadata.extras)
    meta = MetaDto(
        mode=result.metadata.mode,
        latency_ms=round(latency_ms, 3),
        tokens_in=result.metadata.tokens_in,
        tokens_out=result.metadata.tokens_out,
        model=result.metadata.model,
        trace_id=trace_id,
        **extras,
    )
    return QueryOut(
        answer=result.answer,
        entities=list(result.entities),
        triples=[TripleDto(h=h, r=r, t=t) for h, r, t in result.triples],
        chunks=[ChunkDto(**chunk) for chunk in result.chunks],
        metadata=meta,
        abstained=bool(extras.get("abstained", False)),
    )


def create_app(adapter: GraphRAGAdapter | None = None) -> FastAPI:
    """Application FastAPI paramétrée par un GraphRAGAdapter injectable."""
    adapter = adapter if adapter is not None else ExistingGraphRAGAdapter()
    adapter_name = type(adapter).__name__

    gateway = FastAPI(
        title="GraphRAG LLMOps Gateway",
        description="Passerelle /query stable au-dessus du pipeline legacy (qa-multisaut)",
        version=API_VERSION,
    )

    @gateway.get("/")
    def root():
        return {
            "name": "GraphRAG LLMOps Gateway",
            "version": API_VERSION,
            "endpoints": {"health": "GET /health", "query": "POST /query"},
        }

    @gateway.get("/health")
    def health():
        return {
            "status": "ok",
            "version": API_VERSION,
            "adapter": adapter_name,
            "pipeline": "qa-multisaut legacy (create_pipeline/PipelineState, non modifié)",
        }

    @gateway.post("/query", response_model=QueryOut)
    def query(payload: QueryIn):
        trace_id = str(uuid.uuid4())
        logger.info(
            "query received",
            extra={
                "trace_id": trace_id,
                "mode": payload.mode,
                "question_chars": len(payload.question),
                "adapter": adapter_name,
            },
        )
        start = time.perf_counter()
        try:
            result = adapter.query(payload.question.strip(), payload.mode)
        except AdapterUnavailable as exc:
            logger.warning(
                "adapter unavailable",
                extra={"trace_id": trace_id, "mode": payload.mode, "adapter": adapter_name},
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"adapter unavailable: {exc}",
            ) from exc
        except Exception as exc:
            logger.exception(
                "query failed",
                extra={"trace_id": trace_id, "mode": payload.mode, "adapter": adapter_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"query failed: {exc}",
            ) from exc

        wall_ms = (time.perf_counter() - start) * 1000.0
        out = _to_query_out(result, trace_id, wall_ms)
        logger.info(
            "query done",
            extra={
                "trace_id": trace_id,
                "mode": payload.mode,
                "latency_ms": out.metadata.latency_ms,
                "adapter": adapter_name,
                "status": "abstained" if out.abstained else "answered",
            },
        )
        return out

    return gateway


app = create_app()
