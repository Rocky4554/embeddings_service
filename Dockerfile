FROM python:3.11-slim

# Models cache here; baked into the image so the container is self-contained.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /srv

# 1) CPU-only PyTorch first (avoids the ~2GB CUDA build).
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# 2) The rest of the deps.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) App code.
COPY app/ ./app/

# 4) Pre-download the models at BUILD time → no runtime download, works offline.
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG RERANK_MODEL=BAAI/bge-reranker-base
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}'); CrossEncoder('${RERANK_MODEL}'); print('models cached')"

EXPOSE 8000

# One worker keeps a single in-memory copy of each model (lowest RAM). Scale by
# running more containers behind a load balancer, not more workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
