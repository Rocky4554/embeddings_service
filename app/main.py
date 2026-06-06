"""name.ai embedding + reranking microservice (FastAPI).

Endpoints:
  GET  /health        → liveness + which models are loaded
  POST /embed         → { texts: [...] }              → { embeddings: [[...384]], dim, model }
  POST /embed/single  → { text: "..." }               → { embedding: [...384], dim, model }
  POST /rerank        → { query, texts: [...] }       → { scores: [...], model }

Self-hosted drop-in for the HuggingFace Inference API so name.ai search runs
with no per-call credits and far lower latency. Auth via the X-API-Key header
when EMBED_SERVICE_API_KEY is set.
"""

import logging
import time

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app import config, models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("embed-service")

app = FastAPI(
    title="nameai-embed-service",
    version="1.1.0",
    description="Self-hosted embedding + reranking service. See /docs for interactive API explorer.",
)


@app.on_event("startup")
def _startup() -> None:
    t0 = time.time()
    models.load_models()
    log.info("startup complete in %.1fs", time.time() - t0)


def _auth(x_api_key: str | None) -> None:
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")


# ── Request / Response schemas ────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list, description="Batch of texts to embed (max 256)")


class SingleEmbedRequest(BaseModel):
    text: str = Field(description="Single text to embed")


class RerankRequest(BaseModel):
    query: str = Field(description="Search query")
    texts: list[str] = Field(default_factory=list, description="Candidate texts to score against the query (max 100)")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Service liveness + loaded models")
def health():
    return {
        "status": "ok" if models.is_ready() else "loading",
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "embed_dim": config.EMBED_DIM,
    }


@app.post("/embed", summary="Batch embedding — embed multiple texts at once")
def embed(req: EmbedRequest, x_api_key: str | None = Header(default=None)):
    """
    Embed a batch of texts. Returns one vector per input text.

    - **texts**: list of strings (max 256 per request)
    - **Returns**: `embeddings` — list of float arrays, one per input text
    """
    _auth(x_api_key)
    if not req.texts:
        return {"embeddings": [], "dim": config.EMBED_DIM, "model": config.EMBED_MODEL}
    if len(req.texts) > config.MAX_EMBED_BATCH:
        raise HTTPException(status_code=413, detail=f"batch too large (max {config.MAX_EMBED_BATCH})")
    t0 = time.time()
    vectors = models.embed(req.texts)
    log.info("embed n=%d ms=%d", len(req.texts), int((time.time() - t0) * 1000))
    return {"embeddings": vectors, "dim": config.EMBED_DIM, "model": config.EMBED_MODEL}


@app.post("/embed/single", summary="Single embedding — embed one text, get one vector back")
def embed_single(req: SingleEmbedRequest, x_api_key: str | None = Header(default=None)):
    """
    Embed a single text. Simpler interface for apps that embed one item at a time.

    - **text**: the string to embed
    - **Returns**: `embedding` — a single float array (length = dim)
    """
    _auth(x_api_key)
    t0 = time.time()
    vectors = models.embed([req.text])
    log.info("embed/single ms=%d", int((time.time() - t0) * 1000))
    return {"embedding": vectors[0], "dim": config.EMBED_DIM, "model": config.EMBED_MODEL}


@app.post("/rerank", summary="Rerank — score candidate texts against a query")
def rerank(req: RerankRequest, x_api_key: str | None = Header(default=None)):
    """
    Score each candidate text for relevance to the query using a cross-encoder.

    - **query**: the search query string
    - **texts**: list of candidate texts (max 100)
    - **Returns**: `scores` — one float (0–1) per input text, in the same order as input.
      Higher = more relevant. Use these scores to sort or filter your candidates.
    """
    _auth(x_api_key)
    if not req.texts:
        return {"scores": [], "model": config.RERANK_MODEL}
    if len(req.texts) > config.MAX_RERANK_DOCS:
        raise HTTPException(status_code=413, detail=f"too many docs (max {config.MAX_RERANK_DOCS})")
    t0 = time.time()
    scores = models.rerank(req.query, req.texts)
    log.info("rerank n=%d ms=%d", len(req.texts), int((time.time() - t0) * 1000))
    return {"scores": scores, "model": config.RERANK_MODEL}
