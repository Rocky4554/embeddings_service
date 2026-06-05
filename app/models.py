"""Model loading + inference.

The two models are loaded ONCE (at startup) and held in module-level singletons,
so every request reuses the warm, in-memory model. CPU inference via PyTorch.
"""

import logging

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch.nn import Identity

from app import config

log = logging.getLogger("embed-service")

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def load_models() -> None:
    """Load both models into memory. Called once on startup."""
    global _embedder, _reranker
    log.info("loading embed model: %s", config.EMBED_MODEL)
    _embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    log.info("loading rerank model: %s", config.RERANK_MODEL)
    _reranker = CrossEncoder(config.RERANK_MODEL, device="cpu", max_length=config.RERANK_MAX_LENGTH)
    # Warm up with a tiny forward pass so the first real request isn't slow.
    _embedder.encode(["warmup"], convert_to_numpy=True)
    _reranker.predict([["warmup", "warmup"]], convert_to_numpy=True)
    log.info("models ready")


def is_ready() -> bool:
    return _embedder is not None and _reranker is not None


def embed(texts: list[str]) -> list[list[float]]:
    """Return a 384-dim embedding per input text."""
    vecs = _embedder.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=config.EMBED_NORMALIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32).tolist()


def rerank(query: str, texts: list[str]) -> list[float]:
    """Return one relevance score per (query, text) pair.

    We force raw logits (Identity activation) then apply our own sigmoid so the
    output is a deterministic 0–1 score that matches the HF text-ranking API.
    """
    pairs = [[query, t] for t in texts]
    scores = _reranker.predict(
        pairs,
        batch_size=config.RERANK_BATCH_SIZE,
        activation_fct=Identity(),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if config.RERANK_SIGMOID:
        scores = 1.0 / (1.0 + np.exp(-scores))
    return scores.tolist()
