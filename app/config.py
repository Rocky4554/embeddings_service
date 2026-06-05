"""Runtime configuration for the name.ai embedding + reranking service.

All knobs are environment variables so the same image runs in dev / on the VM
without code changes. Defaults match name.ai's existing pipeline so this service
is a drop-in replacement for the HuggingFace Inference API.
"""

import os


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# Models — keep EMBED_MODEL identical to the one that built the existing corpus
# (all-MiniLM-L6-v2, 384-dim) so new vectors are compatible with the stored ones.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# Shared-secret auth. If set, callers must send `X-API-Key: <this>`. Leave empty
# only on a fully private network.
API_KEY = os.getenv("EMBED_SERVICE_API_KEY", "")

# Embedding L2-normalization. OFF by default to mirror the HF feature-extraction
# output that built the corpus. (Cosine search is scale-invariant either way.)
EMBED_NORMALIZE = _bool("EMBED_NORMALIZE", False)

# Reranker: apply sigmoid to logits so scores are 0–1, matching the HF
# text-ranking API the app's relevance cutoff was tuned against.
RERANK_SIGMOID = _bool("RERANK_SIGMOID", True)

# Safety limits.
MAX_EMBED_BATCH = int(os.getenv("MAX_EMBED_BATCH", "256"))
MAX_RERANK_DOCS = int(os.getenv("MAX_RERANK_DOCS", "100"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "32"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))

EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
