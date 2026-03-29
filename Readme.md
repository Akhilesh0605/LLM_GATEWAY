# llm-gateway

A backend service that routes LLM queries to the right model based on complexity — so you're not paying GPT-4 prices for questions a smaller model handles just as well.

Built with FastAPI, Redis, PostgreSQL, and Groq.

---

## Why I built this

Most apps just send every request to the most powerful (and expensive) model available. That works, but it's wasteful. A question like "what is a REST API" doesn't need the same model as "design a rate limiter for a distributed system."

This project sits in front of your LLM calls and makes that decision automatically. It also caches responses using embeddings — so similar questions don't hit the API at all.

---

## What it does

- Checks if a similar query was already answered (semantic cache via Redis)
- If not, scores the query complexity from 1–10
- Routes to the cheapest model that can handle it
- Falls back to a stronger model if the primary one fails
- Logs every request to PostgreSQL — model used, tokens, cost, latency
- Runs a background evaluation loop to check if routing decisions are actually good

---

## Stack

- Python 3.12
- FastAPI + Uvicorn
- Groq API (LLaMA 3 models)
- Redis — semantic cache with cosine similarity
- PostgreSQL — request logs and analytics
- Docker Compose — runs everything together

---

## Getting started

You need Docker and a Groq API key. That's it.

```bash
git clone https://github.com/Akhilesh0605/llm-gateway.git
cd llm-gateway
cp .env.example .env
```

Fill in `.env`:

```env
GROQ_API_KEY=your_key_here
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/llmgateway
DAILY_BUDGET_USD=10.0
```

Then:

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
  "cost_usd": 0.000041
}
```

Response headers tell you what happened:

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
| 1–3 | model-1|
| 4–6 | model-2 |
| 7–10 | model-3 |

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

## Evaluation loop

About 10% of requests — and anything that scores right on a routing boundary (3 or 6) — get sent to both the routed model and the strongest model. The outputs are compared with cosine similarity and logged.

Over time this shows whether the routing thresholds are actually working, and where the cheap model starts to fall short.

---

## Project structure

```
llm-gateway/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── classifier.py
│   ├── router.py
│   ├── cache.py
│   ├── analytics.py
│   └── llm_client.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

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

## License

MIT