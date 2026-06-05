"""name.ai embedding + reranking microservice (FastAPI).

Endpoints:
  GET  /health   → liveness + which models are loaded
  POST /embed    → { texts: [...] }              → { embeddings: [[...384]], dim, model }
  POST /rerank   → { query, texts: [...] }       → { scores: [...], model }

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

app = FastAPI(title="nameai-embed-service", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    t0 = time.time()
    models.load_models()
    log.info("startup complete in %.1fs", time.time() - t0)


def _auth(x_api_key: str | None) -> None:
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)


class RerankRequest(BaseModel):
    query: str
    texts: list[str] = Field(default_factory=list)


@app.get("/health")
def health():
    return {
        "status": "ok" if models.is_ready() else "loading",
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "embed_dim": config.EMBED_DIM,
    }


@app.post("/embed")
def embed(req: EmbedRequest, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    if not req.texts:
        return {"embeddings": [], "dim": config.EMBED_DIM, "model": config.EMBED_MODEL}
    if len(req.texts) > config.MAX_EMBED_BATCH:
        raise HTTPException(status_code=413, detail=f"batch too large (max {config.MAX_EMBED_BATCH})")
    t0 = time.time()
    vectors = models.embed(req.texts)
    log.info("embed n=%d ms=%d", len(req.texts), int((time.time() - t0) * 1000))
    return {"embeddings": vectors, "dim": config.EMBED_DIM, "model": config.EMBED_MODEL}


@app.post("/rerank")
def rerank(req: RerankRequest, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    if not req.texts:
        return {"scores": [], "model": config.RERANK_MODEL}
    if len(req.texts) > config.MAX_RERANK_DOCS:
        raise HTTPException(status_code=413, detail=f"too many docs (max {config.MAX_RERANK_DOCS})")
    t0 = time.time()
    scores = models.rerank(req.query, req.texts)
    log.info("rerank n=%d ms=%d", len(req.texts), int((time.time() - t0) * 1000))
    return {"scores": scores, "model": config.RERANK_MODEL}
