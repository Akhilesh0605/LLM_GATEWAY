# llm-gateway

A backend service that routes LLM queries to the right model based on complexity, then uses semantic caching to avoid unnecessary repeat calls.

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

## Example Response

```json
{
  "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "response": "A binary search tree is...",
  "model_used": "llama-3.1-8b-instant",
  "tier": "simple",
  "complexity_score": 2,
  "cache_hit": false,
  "latency_ms": 310,
  "tokens_used": 198,
  "cost_usd": 0.000041,
  "similarity_score": 0.92
}
```

`similarity_score` is optional and mostly useful for debugging cache behavior.

---

## Example Metrics

These are easy to update after a run:

- Cache hit rate: 68%
- Cost saved: 41%
- Average latency: 184 ms

These numbers depend on workload, but even a small cache hit rate significantly reduces cost and improves response time.
---

## Performance Insight

- Cache miss: triggers an LLM call (higher latency and cost)
- Cache hit: returns instantly from Redis (low latency, zero cost)
---

## Stack

- Python 3.12
- FastAPI + Uvicorn
- Groq API (LLaMA 3 models)
- Redis — semantic cache with cosine similarity
- PostgreSQL — request logs and analytics
- Docker (Redis only) + Local services
  - Redis runs via Docker
  - FastAPI runs locally
  - PostgreSQL runs locally

---

## Getting started

You need Docker and a Groq API key. That's it.

```bash
git clone https://github.com/Akhilesh0605/llm-gateway.git
cd llm-gateway
```

Create a `.env` file and fill it in:

```env
GROQ_API_KEY=your_key_here
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/llmgateway
DAILY_BUDGET_USD=10.0
```

Then:

# Start Redis using Docker
```bash
docker run -d -p 6379:6379 redis:7
```
# Run FastAPI locally
```bash
uvicorn app.main:app --reload
```
App runs on `http://localhost:8000`.

---

## Sending a query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is a binary search tree"}'
```

Response:

```json
{
  "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "response": "A binary search tree is...",
  "model_used": "llama-3.1-8b-instant",
  "complexity_score": 2,
  "cache_hit": false,
  "latency_ms": 310,
  "tokens_used": 198,
  "cost_usd": 0.000041,
  "similarity_score": 0.92
}
```

If you expose headers, they can look like this:

```
X-Model-Used: llama-3.1-8b-instant
X-Cache-Status: MISS
X-Latency: 310ms
X-Request-ID: f47ac10b-58cc-4372-a567-0e02b2c3d479
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
| 7–10 |openai/gpt-oss-120b |

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

## System Architecture

```text
Client
  ↓
FastAPI
  ↓
Redis (cache)
  ↓
PostgreSQL (logs)
  ↓
LLM APIs
```

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
├── docker-compose.yaml # Local stack for app, Redis, and PostgreSQL
├── init_db.py          # Creates the database tables
├── requirements.txt    # Python dependencies
└── README.md
```

Main files in `app/` do the heavy lifting, while the root files are just for setup and running the project.

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

This project focuses on optimizing LLM usage in real-world scenarios by combining routing, caching, and cost-awareness into a single system.

