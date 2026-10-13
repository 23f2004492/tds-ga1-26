"""Intelligent caching API for a code-review assistant.

POST /               Query endpoint
GET  /analytics      Cache metrics
POST /cache/invalidate  Invalidate one query or clear the cache

For semantic embeddings, set SEMANTIC_BACKEND=fastembed and install fastembed.
If the model cannot be loaded, the app safely falls back to local hashed text vectors.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_NAME = "Intelligent AI Cache"
TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 hours
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1500"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.95"))
AVG_TOKENS_PER_REQUEST = int(os.getenv("AVG_TOKENS_PER_REQUEST", "2000"))
MODEL_COST_PER_MILLION_TOKENS = float(os.getenv("MODEL_COST_PER_MILLION_TOKENS", "1.20"))
SEMANTIC_BACKEND_SETTING = os.getenv("SEMANTIC_BACKEND", "fastembed").strip().lower()

app = FastAPI(title=APP_NAME, version="1.0.0", description="Exact and semantic response caching with LRU, TTL, and analytics.")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20000)
    application: str = Field(default="code review assistant", min_length=1, max_length=200)


class InvalidateRequest(BaseModel):
    query: Optional[str] = None
    application: Optional[str] = None


@dataclass
class CacheEntry:
    cache_key: str
    normalized_query: str
    normalized_application: str
    answer: str
    embedding: list[float]
    created_at: float
    last_accessed: float
    expires_at: float
    estimated_bytes: int


# OrderedDict order is oldest to newest access; moving a hit to the end implements LRU.
cache: OrderedDict[str, CacheEntry] = OrderedDict()
lock = RLock()

# Request analytics. Counters describe this running process; an in-memory cache resets on restart.
metrics: dict[str, Any] = {
    "totalRequests": 0,
    "cacheHits": 0,
    "cacheMisses": 0,
    "exactHits": 0,
    "semanticHits": 0,
    "llmCalls": 0,
    "errors": 0,
    "latencyTotalMs": 0.0,
    "latencyLastMs": 0.0,
    "savedTokens": 0,
    "cacheInvalidations": 0,
}

_embedder = None
_embedding_backend = "local-hash"
_embedding_init_error: Optional[str] = None


def normalize_text(value: str) -> str:
    """Normalize case, Unicode, and whitespace so equivalent inputs share a cache key."""
    value = unicodedata.normalize("NFKC", value)
    value = value.casefold().strip()
    return re.sub(r"\s+", " ", value)


def make_cache_key(application: str, normalized_query: str) -> str:
    # Include application so answers from different applications are never mixed.
    raw = f"{normalize_text(application)}\0{normalized_query}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hashed_embedding(text: str, dims: int = 512) -> list[float]:
    """Small local fallback vectorizer (word and character features; no network/API needed).

    This is a lexical vector representation, not a learned language-model embedding. For
    stronger paraphrase matching use SEMANTIC_BACKEND=fastembed.
    """
    text = normalize_text(text)
    tokens = re.findall(r"[a-z0-9_]+|[^\w\s]", text)
    features: dict[int, float] = {}

    def add_feature(feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "little") % dims
        features[index] = features.get(index, 0.0) + weight

    stop_words = {"the", "a", "an", "is", "are", "to", "of", "in", "on", "for", "and", "or", "please", "can", "you", "me", "how", "what", "this", "that", "it", "with", "do", "does", "i", "my", "your", "from", "at", "as", "be", "by", "using", "use"}
    useful = [t for t in tokens if t not in stop_words]
    if not useful:
        useful = tokens
    for token in useful:
        add_feature("w:" + token, 1.0)
        padded = f"^{token}$"
        for n in (3, 4):
            for i in range(max(0, len(padded) - n + 1)):
                add_feature(f"c{n}:" + padded[i:i+n], 0.18)
    for left, right in zip(useful, useful[1:]):
        add_feature("b:" + left + "_" + right, 0.45)

    norm = math.sqrt(sum(v * v for v in features.values())) or 1.0
    result = [0.0] * dims
    for index, value in features.items():
        result[index] = value / norm
    return result


def _get_fastembed():
    global _embedder, _embedding_backend, _embedding_init_error
    if _embedder is not None:
        return _embedder
    if SEMANTIC_BACKEND_SETTING != "fastembed":
        return None
    try:
        from fastembed import TextEmbedding  # optional runtime dependency
        _embedder = TextEmbedding(model_name=os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))
        _embedding_backend = "fastembed:BAAI/bge-small-en-v1.5"
        return _embedder
    except Exception as exc:  # fallback keeps the API available if a model download fails
        _embedding_init_error = f"{type(exc).__name__}: {exc}"
        _embedding_backend = "local-hash (fastembed fallback)"
        return None


def embed_text(text: str) -> list[float]:
    model = _get_fastembed()
    if model is not None:
        try:
            vector = next(model.embed([text]))
            # Model vectors are normalized for reliable cosine similarity.
            values = [float(x) for x in vector]
            norm = math.sqrt(sum(x * x for x in values)) or 1.0
            return [x / norm for x in values]
        except Exception as exc:
            global _embedding_backend, _embedding_init_error
            _embedding_init_error = f"{type(exc).__name__}: {exc}"
            _embedding_backend = "local-hash (fastembed fallback)"
    return _hashed_embedding(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    # Vectors are unit-normalized, so dot product equals cosine similarity.
    return sum(x * y for x, y in zip(a, b))


def estimated_entry_size(entry: CacheEntry) -> int:
    return len(entry.cache_key) + len(entry.normalized_query.encode("utf-8")) + len(entry.answer.encode("utf-8")) + 8 * len(entry.embedding) + 256


def purge_expired(now: Optional[float] = None) -> int:
    now = now if now is not None else time.time()
    expired_keys = [key for key, entry in cache.items() if entry.expires_at <= now]
    for key in expired_keys:
        cache.pop(key, None)
    return len(expired_keys)


def find_semantic_match(app_norm: str, embedding: list[float]) -> tuple[Optional[CacheEntry], float]:
    best_entry: Optional[CacheEntry] = None
    best_similarity = -1.0
    # At the configured 1,500 entries this bounded in-memory scan is small and simple.
    for entry in cache.values():
        if entry.normalized_application != app_norm:
            continue
        score = cosine_similarity(embedding, entry.embedding)
        if score > best_similarity:
            best_similarity, best_entry = score, entry
    if best_entry is not None and best_similarity > SIMILARITY_THRESHOLD:
        return best_entry, best_similarity
    return None, best_similarity


async def generate_answer(query: str, application: str) -> str:
    """Use OpenAI when configured; otherwise return a transparent demo code-review result."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a concise code review assistant. Identify likely bugs, security risks, edge cases, and actionable improvements. If the user did not include code, ask them to provide it."},
                    {"role": "user", "content": f"Application: {application}\n\nRequest: {query}"},
                ],
            )
            content = response.choices[0].message.content
            return content.strip() if content else "No answer was returned by the model."
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"LLM provider request failed: {type(exc).__name__}") from exc

    # Demo fallback lets the caching behaviour be tested without a paid API key.
    return (
        "Demo code-review response (no OPENAI_API_KEY configured).\n"
        f"Request received: {query}\n\n"
        "Review checklist: check correctness and edge cases; validate input; handle exceptions; "
        "avoid exposing secrets; verify authorization and data access; add focused tests. "
        "Configure OPENAI_API_KEY to enable real model-generated reviews."
    )


@app.get("/")
async def health_check() -> dict[str, str]:
    return {"service": APP_NAME, "status": "ok", "queryEndpoint": "POST /", "analyticsEndpoint": "GET /analytics"}


@app.post("/")
async def query_endpoint(body: QueryRequest) -> dict[str, Any]:
    started = time.perf_counter()
    query_norm = normalize_text(body.query)
    app_norm = normalize_text(body.application)
    if not query_norm:
        raise HTTPException(status_code=422, detail="query must contain non-whitespace text")

    key = make_cache_key(app_norm, query_norm)
    now = time.time()
    query_embedding: Optional[list[float]] = None

    with lock:
        purge_expired(now)
        metrics["totalRequests"] += 1
        exact_entry = cache.get(key)
        if exact_entry is not None:
            exact_entry.last_accessed = now
            # Refresh TTL on access (sliding 24h TTL).
            exact_entry.expires_at = now + TTL_SECONDS
            cache.move_to_end(key)
            metrics["cacheHits"] += 1
            metrics["exactHits"] += 1
            metrics["savedTokens"] += AVG_TOKENS_PER_REQUEST
            elapsed = (time.perf_counter() - started) * 1000
            metrics["latencyLastMs"] = elapsed
            metrics["latencyTotalMs"] += elapsed
            return {
                "answer": exact_entry.answer,
                "cached": True,
                "latency": round(elapsed, 3),
                "cacheKey": key,
                "cacheType": "exact",
                "similarity": 1.0,
            }

    # Embed outside the lock to avoid blocking simultaneous request bookkeeping.
    query_embedding = embed_text(query_norm)

    with lock:
        # A concurrent request may have populated the exact key while this request embedded.
        purge_expired()
        exact_entry = cache.get(key)
        if exact_entry is not None:
            exact_entry.last_accessed = time.time()
            exact_entry.expires_at = time.time() + TTL_SECONDS
            cache.move_to_end(key)
            metrics["cacheHits"] += 1
            metrics["exactHits"] += 1
            metrics["savedTokens"] += AVG_TOKENS_PER_REQUEST
            elapsed = (time.perf_counter() - started) * 1000
            metrics["latencyLastMs"] = elapsed
            metrics["latencyTotalMs"] += elapsed
            return {"answer": exact_entry.answer, "cached": True, "latency": round(elapsed, 3), "cacheKey": key, "cacheType": "exact", "similarity": 1.0}

        semantic_entry, similarity = find_semantic_match(app_norm, query_embedding)
        if semantic_entry is not None:
            semantic_entry.last_accessed = time.time()
            semantic_entry.expires_at = time.time() + TTL_SECONDS
            cache.move_to_end(semantic_entry.cache_key)
            metrics["cacheHits"] += 1
            metrics["semanticHits"] += 1
            metrics["savedTokens"] += AVG_TOKENS_PER_REQUEST
            elapsed = (time.perf_counter() - started) * 1000
            metrics["latencyLastMs"] = elapsed
            metrics["latencyTotalMs"] += elapsed
            return {
                "answer": semantic_entry.answer,
                "cached": True,
                "latency": round(elapsed, 3),
                "cacheKey": semantic_entry.cache_key,
                "cacheType": "semantic",
                "similarity": round(similarity, 5),
            }
        metrics["cacheMisses"] += 1
        metrics["llmCalls"] += 1

    try:
        answer = await generate_answer(body.query.strip(), body.application.strip())
    except HTTPException:
        with lock:
            metrics["errors"] += 1
            # Treat failed provider calls as misses, but they are not inserted into cache.
        raise
    finished = time.time()
    with lock:
        entry = CacheEntry(
            cache_key=key,
            normalized_query=query_norm,
            normalized_application=app_norm,
            answer=answer,
            embedding=query_embedding,
            created_at=finished,
            last_accessed=finished,
            expires_at=finished + TTL_SECONDS,
            estimated_bytes=0,
        )
        entry.estimated_bytes = estimated_entry_size(entry)
        cache[key] = entry
        cache.move_to_end(key)
        while len(cache) > MAX_CACHE_SIZE:
            cache.popitem(last=False)  # LRU eviction
        elapsed = (time.perf_counter() - started) * 1000
        metrics["latencyLastMs"] = elapsed
        metrics["latencyTotalMs"] += elapsed

    return {
        "answer": answer,
        "cached": False,
        "latency": round(elapsed, 3),
        "cacheKey": key,
        "cacheType": "miss",
    }


@app.get("/analytics")
async def analytics() -> dict[str, Any]:
    with lock:
        purge_expired()
        total = metrics["totalRequests"]
        hits = metrics["cacheHits"]
        misses = metrics["cacheMisses"]
        hit_rate = hits / total if total else 0.0
        saved_tokens = metrics["savedTokens"]
        savings = saved_tokens * MODEL_COST_PER_MILLION_TOKENS / 1_000_000
        baseline_cost = total * AVG_TOKENS_PER_REQUEST * MODEL_COST_PER_MILLION_TOKENS / 1_000_000
        actual_model_cost = misses * AVG_TOKENS_PER_REQUEST * MODEL_COST_PER_MILLION_TOKENS / 1_000_000
        memory_bytes = sum(entry.estimated_bytes for entry in cache.values())
        average_latency = metrics["latencyTotalMs"] / total if total else 0.0
        alerts = []
        if total >= 100 and hit_rate < 0.10:
            alerts.append("Cache hit rate is below 10%; review normalization, TTL, and repeated-query patterns.")
        return {
            "hitRate": round(hit_rate, 4),
            "missRate": round(misses / total, 4) if total else 0.0,
            "totalRequests": total,
            "cacheHits": hits,
            "cacheMisses": misses,
            "exactHits": metrics["exactHits"],
            "semanticHits": metrics["semanticHits"],
            "llmCalls": metrics["llmCalls"],
            "errors": metrics["errors"],
            "cacheSize": len(cache),
            "maxCacheSize": MAX_CACHE_SIZE,
            "memoryBytesApprox": memory_bytes,
            "memoryKBApprox": round(memory_bytes / 1024, 2),
            "ttlSeconds": TTL_SECONDS,
            "similarityThreshold": SIMILARITY_THRESHOLD,
            "embeddingBackend": _embedding_backend,
            "embeddingWarning": _embedding_init_error,
            "averageLatencyMs": round(average_latency, 3),
            "lastLatencyMs": round(metrics["latencyLastMs"], 3),
            "avgTokensPerRequestAssumption": AVG_TOKENS_PER_REQUEST,
            "modelCostPerMillionTokens": MODEL_COST_PER_MILLION_TOKENS,
            "baselineCost": round(baseline_cost, 4),
            "estimatedActualModelCost": round(actual_model_cost, 4),
            "costSavings": round(savings, 4),
            "savingsPercent": round((savings / baseline_cost) * 100, 2) if baseline_cost else 0.0,
            "strategies": ["input normalization", "exact match caching", "semantic cosine similarity", "LRU eviction", "TTL expiration", "cost and latency analytics"],
            "alerts": alerts,
            "note": "Metrics and cache are in-memory for this process; restarting the service resets them. Costs assume 2,000 tokens per request and exclude embedding/API overhead.",
        }


@app.post("/cache/invalidate")
async def invalidate_cache(body: InvalidateRequest) -> dict[str, Any]:
    with lock:
        if not body.query:
            removed = len(cache)
            cache.clear()
            metrics["cacheInvalidations"] += removed
            return {"invalidated": removed, "scope": "all", "message": "Cache cleared."}

        app_norm = normalize_text(body.application or "code review assistant")
        query_norm = normalize_text(body.query)
        key = make_cache_key(app_norm, query_norm)
        removed = 1 if cache.pop(key, None) is not None else 0
        metrics["cacheInvalidations"] += removed
        return {"invalidated": removed, "scope": "exact query", "cacheKey": key}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": APP_NAME, "cacheEntries": len(cache), "embeddingBackend": _embedding_backend}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
