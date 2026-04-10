# LLM-gateway
A cost-aware LLM gateway that routes queries based on complexity and reduces latency and API cost using semantic caching.

Achieves ~100x latency improvement and ~35% cost savings by avoiding redundant LLM calls.
Built with FastAPI, Redis, PostgreSQL, and Groq.

---

## Why I built this

Most apps just send every request to the most powerful model available. That works, but it is expensive and slower than it needs to be. A simple question does not need the same LLM as a complex one.

This project sits in front of LLM calls and makes that decision automatically. The goal is simple: cut cost, reduce latency, and avoid calling an LLM when the answer already exists in cache.

---

## What makes this different

- Semantic caching using embeddings instead of exact string matching
- Cosine similarity to find near-duplicate questions
- Complexity-based routing so each query goes to a model that fits the task
- Cost-aware design that tries to keep expensive calls to a minimum

---

## What it does

- Checks if a similar query was already answered using a semantic cache in Redis
- If not, scores the query complexity from 1–10
- Routes to the cheapest model that can handle it
- Falls back to a stronger model if the primary one fails
- Logs every request to PostgreSQL — model used, tokens, cost, latency
- Runs a background evaluation loop to check if routing decisions are actually good
- Uses similarity scoring internally to decide whether a cached answer is close enough

---

## How it works

```text
User Query
   → Classifier (score 1–10)
   → Router (select model)
   → Cache check (embeddings + cosine similarity)
   → HIT → return cached response
   → MISS → call LLM → store → return
```

---

## Verified Results

These are real numbers from running the system locally, not estimates.

**Analytics after 7 requests:**

```json
{
  "total_requests": 7,
  "avg_latency_ms": 1582.85,
  "total_cost_usd": 0.0044,
  "cache_hit_rate": 42.86,
  "cost_saved_usd": 0.0024,
  "cost_saved_percent": 35.55,
  "total_tokens": 3395
}
```

**Benchmark — model distribution:**

```json
{
  "total_requests": 7,
  "model_distribution": {
    "llama-3.1-8b-instant": {
      "count": 4,
      "percentage": 57.14
    },
    "openai/gpt-oss-120b": {
      "count": 3,
      "percentage": 42.86
    }
  },
  "cache_hit_rate": 42.86,
  "cost_saved_percent": 35.55
}
```

---

## Routing in action

**Simple query — cache miss (first time):**

Query: `"what is api"`

```json
{
  "model_used": "llama-3.1-8b-instant",
  "tier": "simple",
  "complexity_score": 2,
  "cache_hit": false,
  "latency_ms": 1758.29,
  "tokens_used": 626,
  "cost_usd": 0.0000313,
  "similarity_score": 0
}
```

**Semantic cache hit — similar but not identical query:**

Query: `"define api"` — routed to cache because similarity score 0.951 exceeded the simple tier threshold of 0.90

```json
{
  "model_used": "llama-3.1-8b-instant",
  "tier": "simple",
  "complexity_score": 2,
  "cache_hit": true,
  "latency_ms": 24.89,
  "tokens_used": 0,
  "cost_usd": 0,
  "similarity_score": 0.951
}
```

Latency dropped from 1758ms to 24ms. Cost dropped to zero.

**Complex query — routed to stronger model:**

Query: `"design a distributed rate limiter system"`

```json
{
  "model_used": "openai/gpt-oss-120b",
  "tier": "complex",
  "complexity_score": 8,
  "cache_hit": false,
  "latency_ms": 2969.25,
  "tokens_used": 1077,
  "cost_usd": 0.002154,
  "similarity_score": 0.037
}
```

**Complex query — semantic cache hit:**

Query: `"how would you build a distributed rate limiting system"` — different wording, same intent.Similarity score (0.876) was below the complex tier threshold (0.95), so the request was sent to the LLM just below threshold, so it called the LLM. But on the second attempt with a closer phrasing, similarity hit 1.0 and returned from cache instantly.

```json
{
  "model_used": "openai/gpt-oss-120b",
  "tier": "complex",
  "complexity_score": 8,
  "cache_hit": true,
  "latency_ms": 17.09,
  "tokens_used": 0,
  "cost_usd": 0,
  "similarity_score": 1
}
```
---
## Performance Summary

- Latency reduced from ~1750ms → ~20ms using semantic caching (~100x improvement)
- Cache hit rate: ~42%
- Cost reduced by ~35% by avoiding repeated LLM calls
- Zero-token responses for cached queries
- Semantic matching allows reuse across differently phrased queries

---
## Demo

### Cache Miss vs Cache Hit
- First request: ~1758ms
- Cached request: ~17ms
![Cache Hit vs Miss](./assets/hit_vs_miss.png)

### Semantic Match
- "what is api" → "define api"
- similarity_score: 0.951 → cache hit
![Semantic Match](./assets/similarity_score.png)

### Analytics
- Cache hit rate: ~42%
- Cost saved: ~35%
![Analytics](./assets/analytics.png)
![Benchmark](./assets/analytics_benchmark.png)
---
## Stack

- Python 3.12
- FastAPI + Uvicorn
- Groq API (LLaMA 3 models)
- Redis — semantic cache with cosine similarity
- PostgreSQL — request logs and analytics
- Docker Compose — runs FastAPI, Redis, and PostgreSQL together

---

## Getting started

You need Docker and a Groq API key.

```bash
git clone https://github.com/Akhilesh0605/llm-gateway.git
cd llm-gateway
```

Create a `.env` file and fill it in:

```env
GROQ_API_KEY=your_key_here
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/llmgateway
DAILY_BUDGET_USD=10.0
```

Start the app through Docker
```bash
docker-compose up --build
```
App runs on `http://localhost:8000`.

---

## Sending a query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is a binary search tree"}'
```

---

## Endpoints

`POST /query` — send a query, get a response

`GET /analytics` — total requests, cost saved, cache hit rate

`GET /analytics/benchmark` — routing accuracy, model distribution, evaluation loop results

`GET /health` — check if Redis and Postgres are reachable

---

## How routing works

Every query gets a complexity score from 1 to 10 based on token count, structure, and keywords. That score maps to a model:

| Score | Model |
|---|---|
| 1–3 | llama-3.1-8b-instant |
| 4–6 | llama-3.3-70b-versatile |
| 7–10 | openai/gpt-oss-120b |

If the routed model fails, it tries the next one up. If the daily budget is exceeded, everything routes to the cheapest model until midnight.

---

## Semantic cache

Queries are converted to embeddings using `sentence-transformers`. On each new request, the embedding is compared against everything in Redis using cosine similarity. If the score is above the threshold, the cached response is returned — usually in under 50ms.

Thresholds are per complexity tier because a near-match on a simple factual question is fine, but a near-match on a complex reasoning task might not be close enough.

| Tier | Threshold | TTL |
|---|---|---|
| Simple | 0.90 | 2 hours |
| Medium | 0.92 | 1 hour |
| Complex | 0.95 | 30 minutes |

---
## Architecture
    ┌──────────────┐
    │   User Query │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Classifier   │  → complexity score (1–10)
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Router       │  → selects model
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Cache (Redis)│  → embeddings + cosine similarity
    └──────┬───────┘
     HIT   ↓   MISS
    ┌──────────────┐
    │ Return       │
    │ Cached       │
    └──────────────┘
           ↓
    ┌──────────────┐
    │ LLM Call     │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Store Cache  │
    └──────────────┘

---

## Evaluation loop

About 10% of requests — and anything that scores right on a routing boundary (3 or 6) — get sent to both the routed model and the strongest model. The outputs are compared with cosine similarity and logged.

Over time this shows whether the routing thresholds are actually working, and where the cheap model starts to fall short.

---

## Project structure

```
llm-gateway/
├── app/
│   ├── main.py        # FastAPI app, query flow, and endpoints
│   ├── config.py      # Environment settings and model thresholds
│   ├── models.py      # Request/response schemas and routing types
│   ├── classifier.py  # Complexity scoring logic
│   ├── router.py      # Model selection and budget checks
│   ├── cache.py       # Semantic cache lookup and storage
│   ├── analytics.py   # Request logging and analytics queries
│   ├── llm_client.py  # LLM API calls and fallback handling
│   └── evaluation.py  # Shadow evaluation and agreement scoring
├── docker-compose.yaml
├── init_db.py
├── requirements.txt
└── README.md
```

---
## Key Features

- Semantic caching using embeddings (not exact match)
- Dynamic model routing based on query complexity
- Cost-aware LLM usage with budget control
- Automatic fallback to stronger models
- Background evaluation loop for routing accuracy
- Real-time analytics (latency, cost, cache hits)

---

## Future Improvements

- Add a vector database such as FAISS or Pinecone for larger cache lookups
- Build a simple frontend UI for testing and viewing results
- Add rate limiting to control abuse and keep costs predictable
- Support multi-provider routing beyond the current Groq setup

---

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key |
| `DAILY_BUDGET_USD` | 10.0 | Hard cap on daily spend |
| `SIMILARITY_THRESHOLD_SIMPLE` | 0.90 | Cache match strictness for simple queries |
| `SIMILARITY_THRESHOLD_MEDIUM` | 0.92 | Cache match strictness for medium queries |
| `SIMILARITY_THRESHOLD_COMPLEX` | 0.95 | Cache match strictness for complex queries |
| `EVALUATION_LOOP_RATE` | 0.10 | How often to run shadow evaluation |
| `CACHE_TTL_SIMPLE` | 7200 | Cache lifetime in seconds |
| `CACHE_TTL_MEDIUM` | 3600 | Cache lifetime in seconds |
| `CACHE_TTL_COMPLEX` | 1800 | Cache lifetime in seconds |

---

## Notes

This project focuses on optimizing LLM usage in real-world scenarios by combining routing, caching, and cost-awareness into a single system. All metrics shown in this README are from actual test runs, not simulated data.