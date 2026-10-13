# Intelligent AI Caching API

FastAPI implementation for the AI-caching assignment. It exposes `POST /`, `GET /analytics`, `POST /cache/invalidate`, and `GET /health`.

## Strategies implemented

1. **Input normalization**: Unicode NFKC, case-folding, trimming, and collapsing whitespace make `Hello`, `hello`, and whitespace variants share an exact cache key.
2. **Exact cache**: SHA-256 key includes both application and normalized query, to prevent cross-application answer mixing.
3. **Vector similarity cache**: cosine similarity is checked against the configurable threshold (default `0.95`). It stores each query vector with the cached answer. Uses learned `BAAI/bge-small-en-v1.5` embeddings via FastEmbed by default; if model initialization fails, it falls back to a lightweight local feature vectorizer. Set `SEMANTIC_BACKEND=local-hash` to skip model downloads.
4. **LRU eviction**: cache is capped by `MAX_CACHE_SIZE` (default 1,500); least-recently-used entry is removed first.
5. **TTL expiration**: entries expire after `CACHE_TTL_SECONDS` (default 86,400 = 24 hours). Reads refresh the TTL.
6. **Analytics**: hit/miss rates, exact/semantic hits, estimated token/cost savings, latency, approximate cache memory, and a low-hit-rate alert.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive testing. The **actual endpoint URL** to submit is the service root, e.g. `https://YOUR-SERVICE.onrender.com/` (not `/docs`). Analytics URL: `https://YOUR-SERVICE.onrender.com/analytics`.

## Test requests

First request should miss and populate the cache:

```bash
curl -X POST http://127.0.0.1:8000/ -H "Content-Type: application/json" -d '{"query":"Hello","application":"code review assistant"}'
```

Second request uses normalized exact caching:

```bash
curl -X POST http://127.0.0.1:8000/ -H "Content-Type: application/json" -d '{"query":"  HELLO  ","application":"code review assistant"}'
```

Then inspect metrics:

```bash
curl http://127.0.0.1:8000/analytics
```

Invalidate one exact query:

```bash
curl -X POST http://127.0.0.1:8000/cache/invalidate -H "Content-Type: application/json" -d '{"query":"Hello","application":"code review assistant"}'
```

Clear all cached entries: `POST /cache/invalidate` with `{}`.

## Enable learned embeddings / real LLM responses

- The Render Blueprint sets `SEMANTIC_BACKEND=fastembed`, using `BAAI/bge-small-en-v1.5` via FastEmbed. Model download/inference can add cold-start time and memory use. Set `SEMANTIC_BACKEND=local-hash` to skip model downloads; the API also falls back automatically if FastEmbed fails to initialize.
- Add `OPENAI_API_KEY` to the hosting environment to use OpenAI for cache misses. Without that key, the endpoint returns an explicitly labeled demo code-review response so caching and analytics can still be tested. Do not commit API keys to Git.
- You may set `OPENAI_MODEL`, `CACHE_TTL_SECONDS`, `MAX_CACHE_SIZE`, `SIMILARITY_THRESHOLD`, `AVG_TOKENS_PER_REQUEST`, and `MODEL_COST_PER_MILLION_TOKENS` as environment variables.

## Deploy on Render

1. Create a new GitHub repository and upload `app.py`, `requirements.txt`, and `render.yaml` to its root.
2. In Render, choose **New + → Blueprint** and select the repository (or create a Web Service with build command `pip install -r requirements.txt` and start command `uvicorn app:app --host 0.0.0.0 --port $PORT`).
3. Wait for deploy to finish. Open `https://YOUR-SERVICE.onrender.com/health`, then test `POST https://YOUR-SERVICE.onrender.com/` and `GET https://YOUR-SERVICE.onrender.com/analytics`.
4. Submit the root endpoint URL `https://YOUR-SERVICE.onrender.com/` in the assignment form. Do **not** submit the `/docs` URL.

## Important limits

Cache and analytics are in-memory and reset when the web process restarts; multiple server instances will not share a cache. This is suitable for a small assignment demo. Use Redis or persistent storage for production. Semantic caching can return an old answer for a highly similar but materially different request, so a stricter threshold and/or app-specific partitioning is important for code review.
