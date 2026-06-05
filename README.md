# name.ai embed-service

Self-hosted **embedding + reranking** microservice for name.ai partner search.
Runs the same models the app already uses — **all-MiniLM-L6-v2** (embeddings) and
**BAAI/bge-reranker-base** (reranking) — locally, so search has **no per-call
credits** and **much lower latency** than the HuggingFace Inference API.

> Drop-in replacement: same embedding model = vectors compatible with the existing
> `domain_embeddings` corpus; reranker output is sigmoid 0–1 = same scale the app's
> relevance cutoff was tuned against.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{ status, embed_model, rerank_model, embed_dim }` |
| POST | `/embed` | `{ "texts": ["..."] }` | `{ "embeddings": [[...384]], "dim": 384, "model" }` |
| POST | `/rerank` | `{ "query": "...", "texts": ["..."] }` | `{ "scores": [0.94, ...], "model" }` |

Auth: when `EMBED_SERVICE_API_KEY` is set, send header `X-API-Key: <secret>`.

## Run with Docker (recommended)

```bash
cp .env.example .env          # set a strong EMBED_SERVICE_API_KEY
docker compose build          # ~5–10 min first time (downloads + bakes the models)
docker compose up -d
python smoke_test.py          # API_KEY=<secret> python smoke_test.py
```

First build downloads ~370 MB of model weights and bakes them into the image, so
the container starts offline and answers immediately after a ~30–60s warm-up.

## Run locally (dev, no Docker)

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on Linux
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Deploy on the VM

1. Copy this folder to the VM (or `git clone`).
2. `cp .env.example .env` and set `EMBED_SERVICE_API_KEY`.
3. `docker compose up -d --build`.
4. Keep port **8000 on the private network / firewalled** (or behind a reverse
   proxy with TLS). The name.ai app reaches it over the internal network.

**Hardware:** CPU-only. ~2 vCPU / 4 GB RAM is comfortable (both models ≈ 0.5–1 GB
resident). No GPU. Scale throughput by running more replicas behind a load
balancer, not more uvicorn workers (each worker = another full copy in RAM).

## Wire into name.ai

Set in the name.ai app env:

```
EMBED_SERVICE_URL=http://<vm-host>:8000
EMBED_SERVICE_API_KEY=<same secret>
```

Then `lib/server/partner-api/search/embed-client.js` and `rerank-client.js` call
this service instead of HuggingFace (with the HF API kept as an automatic
fallback). See the app-side change that accompanies this service.

## Config (env)

| Var | Default | Notes |
|---|---|---|
| `EMBED_SERVICE_API_KEY` | _(none)_ | shared secret; required header when set |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | must match the corpus model |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | swap to `bge-reranker-v2-m3` for higher quality |
| `EMBED_NORMALIZE` | `false` | mirrors HF feature-extraction output |
| `RERANK_SIGMOID` | `true` | 0–1 scores matching the HF text-ranking API |
| `OMP_NUM_THREADS` | `4` | CPU threads for inference |
