# nameai-embed-service

Self-hosted **embedding + reranking** microservice. Runs two models on CPU:

| Model | Purpose | Default |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Text → 384-dim vector | Embedding |
| `BAAI/bge-reranker-v2-m3` | Query + text → relevance score 0–1 | Reranking |

Both are swappable via env vars — no code changes needed.

**Live URL (VPS3):** `https://embedding.vps3.auctionhacker.com`

---

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Liveness check — returns loaded model names |
| `POST` | `/embed` | Embed a **batch** of texts → one vector per text |
| `POST` | `/embed/single` | Embed a **single** text → one vector |
| `POST` | `/rerank` | Score candidate texts against a query → relevance scores |

Interactive docs (Swagger UI): `https://embedding.vps3.auctionhacker.com/docs`

Auth: every request needs the header `X-API-Key: <secret>`.

---

## `/embed` — Batch Embedding

Embed multiple texts at once. Returns one 384-float vector per input text.

**Request**
```json
POST /embed
X-API-Key: <secret>
Content-Type: application/json

{
  "texts": ["consulting firm for startups", "child custody lawyer"]
}
```

**Response**
```json
{
  "embeddings": [
    [0.021, -0.045, 0.112, ...],  // 384 floats for text[0]
    [0.033,  0.071, 0.009, ...]   // 384 floats for text[1]
  ],
  "dim": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**Limits:** max 256 texts per request.

### curl
```bash
curl -s https://embedding.vps3.auctionhacker.com/embed \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["consulting firm", "legal services"]}' | jq .
```

### Python
```python
import requests

BASE = "https://embedding.vps3.auctionhacker.com"
KEY  = "YOUR_API_KEY"

resp = requests.post(
    f"{BASE}/embed",
    headers={"X-API-Key": KEY},
    json={"texts": ["consulting firm", "legal services"]},
)
data = resp.json()
print(data["dim"])           # 384
print(data["embeddings"][0]) # vector for "consulting firm"
```

### JavaScript / Node.js
```js
const BASE = "https://embedding.vps3.auctionhacker.com";
const KEY  = "YOUR_API_KEY";

const res = await fetch(`${BASE}/embed`, {
  method: "POST",
  headers: { "X-API-Key": KEY, "Content-Type": "application/json" },
  body: JSON.stringify({ texts: ["consulting firm", "legal services"] }),
});
const data = await res.json();
console.log(data.embeddings[0]); // vector for "consulting firm"
```

---

## `/embed/single` — Single Embedding

Simpler interface when you only need to embed one text at a time.

**Request**
```json
POST /embed/single
X-API-Key: <secret>
Content-Type: application/json

{
  "text": "weight loss clinic"
}
```

**Response**
```json
{
  "embedding": [0.021, -0.045, 0.112, ...],
  "dim": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### curl
```bash
curl -s https://embedding.vps3.auctionhacker.com/embed/single \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "weight loss clinic"}' | jq .embedding | head -c 100
```

### Python
```python
resp = requests.post(
    f"{BASE}/embed/single",
    headers={"X-API-Key": KEY},
    json={"text": "weight loss clinic"},
)
vector = resp.json()["embedding"]  # list of 384 floats
```

### JavaScript
```js
const res = await fetch(`${BASE}/embed/single`, {
  method: "POST",
  headers: { "X-API-Key": KEY, "Content-Type": "application/json" },
  body: JSON.stringify({ text: "weight loss clinic" }),
});
const { embedding } = await res.json(); // 384-float array
```

---

## `/rerank` — Reranking

Score a list of candidate texts for relevance to a query. Returns a score (0–1) per text — higher = more relevant. Use this to sort or filter search results.

**Request**
```json
POST /rerank
X-API-Key: <secret>
Content-Type: application/json

{
  "query": "child custody lawyer",
  "texts": [
    "custodylawyer.com — domain for family law firms and divorce attorneys",
    "woodenpuzzle.com — handcrafted wooden puzzles for children",
    "paralegalschool.com — online paralegal training and certification"
  ]
}
```

**Response**
```json
{
  "scores": [0.9987, 0.0001, 0.0018],
  "model": "BAAI/bge-reranker-v2-m3"
}
```

Scores are in the **same order as the input texts**. `custodylawyer.com` scored 0.9987 (highly relevant), `woodenpuzzle.com` scored 0.0001 (not relevant), `paralegalschool.com` scored 0.0018 (slightly related).

**Limits:** max 100 texts per request.

### curl
```bash
curl -s https://embedding.vps3.auctionhacker.com/rerank \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "weight loss",
    "texts": ["quickweightloss.com — diet and fitness programs", "bank.ai — financial services"]
  }' | jq .scores
```

### Python
```python
resp = requests.post(
    f"{BASE}/rerank",
    headers={"X-API-Key": KEY},
    json={
        "query": "weight loss",
        "texts": [
            "quickweightloss.com — diet and fitness programs",
            "bank.ai — financial services",
        ],
    },
)
scores = resp.json()["scores"]
# scores[0] → relevance of text[0] to "weight loss"
# scores[1] → relevance of text[1] to "weight loss"

# Sort candidates by score descending
candidates = ["quickweightloss.com", "bank.ai"]
ranked = sorted(zip(scores, candidates), reverse=True)
print(ranked)  # [( 0.84, 'quickweightloss.com'), (0.001, 'bank.ai')]
```

### JavaScript
```js
const res = await fetch(`${BASE}/rerank`, {
  method: "POST",
  headers: { "X-API-Key": KEY, "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "weight loss",
    texts: [
      "quickweightloss.com — diet and fitness programs",
      "bank.ai — financial services",
    ],
  }),
});
const { scores } = await res.json();
// scores[i] corresponds to texts[i]
const candidates = ["quickweightloss.com", "bank.ai"];
const ranked = candidates
  .map((name, i) => ({ name, score: scores[i] }))
  .sort((a, b) => b.score - a.score);
console.log(ranked);
```

---

## `/health` — Liveness Check

```bash
curl https://embedding.vps3.auctionhacker.com/health
```
```json
{
  "status": "ok",
  "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
  "rerank_model": "BAAI/bge-reranker-v2-m3",
  "embed_dim": 384
}
```

---

## Environment Variables (full reference)

Change models or tune performance by setting these in Coolify (or `.env`). No code changes needed.

### Embedding

| Variable | Default | Notes |
|---|---|---|
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | ⚠️ Must match the model that built your vector corpus. Changing this = all stored vectors become incompatible. |
| `EMBED_DIM` | `384` | Must match the model's output dimension. |
| `EMBED_NORMALIZE` | `false` | Set `true` to L2-normalize vectors before returning. |
| `MAX_EMBED_BATCH` | `256` | Max texts per `/embed` request. |
| `EMBED_BATCH_SIZE` | `64` | Internal inference batch size. |

### Reranking

| Variable | Default | Notes |
|---|---|---|
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Swap to any HuggingFace cross-encoder. No corpus re-build needed. |
| `RERANK_SIGMOID` | `true` | Converts raw logits to 0–1 scores. Keep `true`. |
| `RERANK_MAX_LENGTH` | `512` | Max tokens per (query, text) pair. |
| `MAX_RERANK_DOCS` | `100` | Max texts per `/rerank` request. |
| `RERANK_BATCH_SIZE` | `32` | Internal inference batch size. |

### Auth & Performance

| Variable | Default | Notes |
|---|---|---|
| `EMBED_SERVICE_API_KEY` | _(none)_ | Shared secret. Required header: `X-API-Key`. |
| `OMP_NUM_THREADS` | `4` | CPU threads for PyTorch. Match to your vCPU count. |

### Recommended reranker models (all CPU-compatible)

| Model | Size | Speed | Quality |
|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | ~568MB | Fast | ✅ Best for CPU |
| `BAAI/bge-reranker-base` | ~278MB | Fastest | Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~120MB | Very fast | Good |
| `jinaai/jina-reranker-v2-base-multilingual` | ~280MB | Fast | Good |
| `Qwen3-Reranker-4B` | ~8GB | ❌ Too slow | Best (GPU only) |

---

## Run locally (dev)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env            # set EMBED_SERVICE_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

## Run with Docker

```bash
cp .env.example .env            # set EMBED_SERVICE_API_KEY
docker compose build            # downloads + bakes models (~5-10 min first time)
docker compose up -d
python smoke_test.py            # verify it works
```

## Deploy on Coolify (VPS3)

1. Set env vars in Coolify (copy from `.env.example`, fill in `EMBED_SERVICE_API_KEY`).
2. Set `RERANK_MODEL=BAAI/bge-reranker-v2-m3`.
3. Trigger redeploy — models download automatically on first start.
4. Hit `/health` to confirm both models loaded.
