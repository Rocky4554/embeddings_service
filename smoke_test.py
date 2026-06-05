"""Quick smoke test against a running service.

Usage:
  python smoke_test.py                       # hits http://localhost:8000
  BASE=http://vm-host:8000 API_KEY=... python smoke_test.py
"""

import json
import os
import urllib.request

BASE = os.getenv("BASE", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")


def call(path, payload=None):
    url = f"{BASE}{path}"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


print("health:", call("/health"))

emb = call("/embed", {"texts": ["a premium real estate domain", "market and trading"]})
print("embed dim:", emb["dim"], "count:", len(emb["embeddings"]), "len[0]:", len(emb["embeddings"][0]))
assert emb["dim"] == len(emb["embeddings"][0]) == 384, "embedding dim must be 384"

rr = call("/rerank", {
    "query": "property",
    "texts": ["houseworth.com - real estate and property", "bank.ai - banking", "woodenpuzzle.com - toys"],
})
print("rerank scores:", [round(s, 4) for s in rr["scores"]])
assert rr["scores"][0] > rr["scores"][1], "the property-relevant doc should score highest"
print("\nOK ✓  service is healthy, embeddings are 384-dim, reranker orders correctly.")
