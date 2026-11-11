"""FastAPI application for QA multi-hop."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llmops.adapter import ExistingGraphRAGAdapter, GraphAnswer
from src.orchestrator.graph import create_pipeline
from src.orchestrator.state import PipelineConfig, PipelineState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None
pipeline_config = None

# Langfuse instance (peut être désactivée si pas de clés)
lf: ExistingGraphRAGAdapter = None  # reused as carrier of trace state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup."""
    global pipeline, pipeline_config, lf
    logger.info("Starting up... Loading pipeline...")
    
    pipeline_config = PipelineConfig()
    try:
        pipeline = create_pipeline(pipeline_config)
        logger.info("Pipeline loaded successfully")
    except Exception:  # nosec BLE001 (startup degradation : failure non critique)
        logger.exception("Failed to load pipeline")
        pipeline = None
    
    # Init Langfuse adapter (désactivé si pas de clés)
    from llmops.adapter import _get_langfuse
    lf = _get_langfuse()
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="QA Multi-Hop API",
    description="Multi-hop Question Answering with KG + GNN + LLM",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    """Request for /ask endpoint."""
    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")


class TripletOut(BaseModel):
    """Output triplet format."""
    h: str
    r: str
    t: str
    source_title: str = ""
    sentence: str = ""
    label: int = 0
    score: float = 0.0


class AskResponse(BaseModel):
    """Response for /ask endpoint."""
    answer: str
    sources: list[TripletOut] = Field(default_factory=list)
    latency_ms: float
    config: dict
    abstained: bool = False
    abstention_reason: str = ""
    format_ok: bool = True
    trace_id: str = ""
    session_id: str = "default"


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str
    version: str
    model_loaded: bool
    config: dict


def _state_to_response(state: PipelineState, latency_ms: float, trace_id: str, session_id: str) -> AskResponse:
    """Convert pipeline state to API response."""
    # Convert triplets to output format
    sources = []
    for i, t in enumerate(state.topk_triplets):
        sources.append(TripletOut(
            h=t.h, r=t.r, t=t.t,
            source_title=t.source_title,
            sentence=t.sentence,
            label=t.label,
            score=t.score
        ))
    
    # Check format
    format_ok = "Sources:" in state.answer or len(sources) == 0
    
    return AskResponse(
        answer=state.answer,
        sources=sources,
        latency_ms=latency_ms,
        config=state.config,
        abstained=state.abstained,
        abstention_reason=state.abstention_reason,
        format_ok=format_ok,
        trace_id=trace_id,
        session_id=session_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if pipeline is not None else "degraded",
        version="1.0.0",
        model_loaded=pipeline is not None,
        config=pipeline_config.model_dump() if pipeline_config else {},
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Answer a question using the multi-hop pipeline."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    start = time.time()
    
    try:
        # Run pipeline via the adapter (instrumented)
        adapter = ExistingGraphRAGAdapter(session_id="api-session")
        result: GraphAnswer = adapter.query(request.question.strip(), "hybrid")
        
        latency_ms = (time.time() - start) * 1000
        
        return _state_to_response(
            result.metadata,  # PipelineState has been merged into GraphAnswerMetadata
            latency_ms,
            trace_id=result.metadata.trace_id,
            session_id=result.metadata.session_id,
        )
    
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Internal error: {e!s}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "QA Multi-Hop API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "ask": "POST /ask",
            "docs": "GET /docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)