# LLM Gateway

A multi-tenant, multi-provider **BYOK** gateway that sits in front of LLM calls and makes them cheaper and faster.

> Reduce LLM latency by up to 32x and cut API spend by 35%–90% through neural routing and semantic caching.

Built with **FastAPI**, **ONNX Runtime**, **Redis**, **PostgreSQL**, and **Docker**.

---

## Why

Most apps send every request to the most powerful (and most expensive) model. A "what is 2+2?" question does not need the same model as "design a distributed rate limiter". LLM Gateway decides automatically:

- **Semantic cache** — return a cached answer for near-duplicate questions at $0 and single-digit ms.
- **Neural routing** — classify query complexity (1–10) in ~8 ms and route to the cheapest model that can handle it.
- **BYOK** — each tenant brings their own provider keys, encrypted at rest.
- **Fallback chain** — if the chosen model fails, retry down the tiers before giving up.

---

## Features

- Multi-provider support: **Groq, OpenAI, Google Gemini, Anthropic, Together AI**
- Semantic cache per tenant using embeddings + cosine similarity (per-tier thresholds)
- Neural complexity classifier (`all-MiniLM-L6-v2` + a small MLP), served via **ONNX Runtime** (no PyTorch at runtime)
- BYOK provider keys, **Fernet-encrypted at rest**, decrypted only in memory
- Per-tenant daily budget enforcement (over budget forces the cheapest tier)
- Cascading model fallback (Complex → Medium → Simple → 502)
- Background shadow evaluation (boundary scores + 10% sample) for routing quality
- Real-time analytics: cost, tokens, latency, cache hit rate, model distribution
- Built-in web dashboard (chat playground, analytics, BYOK settings)
- Dockerized, non-root container, production security headers

---

## How it works

```text
User Query
   → Auth (X-Gateway-Key)
   → Neural Classifier        → complexity score 1–10
   → Semantic Cache (Redis)   → HIT? return cached ($0, ~20 ms)
   → Router                   → user's configured model for the tier
   → LLM Provider (BYOK key)  → with cascading fallback
   → Store in cache + log to PostgreSQL + shadow-evaluate
```

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Classification | ONNX Runtime + `all-MiniLM-L6-v2` (384-d) + NumPy MLP (384→64→1) |
| Cache | Redis (cosine similarity over per-tenant embeddings) |
| Database | PostgreSQL (async SQLAlchemy + asyncpg) |
| Security | Fernet (AES) key encryption, SHA-256 gateway keys |
| Packaging | Docker (multi-stage), Docker Compose for local dev |

---

## Quick start (local, Docker)

You need Docker. No provider key is required globally — keys are added per tenant at runtime.

```bash
git clone https://github.com/<you>/llm-gateway.git
cd llm-gateway

cp .env.example .env
# then set ENCRYPTION_KEY (generate below)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

docker compose up --build
```

App: `http://localhost:8000` (dashboard at `/`, OpenAPI docs at `/docs`).

Local `.env` values used by `docker-compose.yaml`:

```env
REDIS_SERVER_LINK=redis://redis:6379
POSTGRESQL_LINK=postgresql+asyncpg://user:password@postgres:5432/llmgateway
ENCRYPTION_KEY=your-stable-fernet-key
```

---

## Using the API

### 1. Register and get a gateway key

```bash
curl -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name": "Akhil", "email": "you@example.com"}'
# → { "gateway_api_key": "gw_live_..." }  (shown once)
```

All tenant endpoints require the key as a header: `X-Gateway-Key: gw_live_...`

### 2. Add a provider key (BYOK)

```bash
curl -X POST http://localhost:8000/v1/providers/add \
  -H "Content-Type: application/json" -H "X-Gateway-Key: gw_live_..." \
  -d '{"provider": "groq", "api_key": "gsk_...", "label": "My Groq Key"}'
```

### 3. Assign models to complexity tiers

```bash
curl -X PUT http://localhost:8000/v1/tiers/config \
  -H "Content-Type: application/json" -H "X-Gateway-Key: gw_live_..." \
  -d '{
    "simple_provider": "groq",  "simple_model": "openai/gpt-oss-20b",
    "medium_provider": "groq",  "medium_model": "qwen/qwen3.8-27b",
    "complex_provider": "groq", "complex_model": "openai/gpt-oss-120b"
  }'
```

### 4. Send a query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" -H "X-Gateway-Key: gw_live_..." \
  -d '{"query": "What is a binary search tree?"}'
```

---

## API reference

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/` | no | Web dashboard |
| `POST` | `/v1/auth/register` | no | Register a tenant, returns a gateway key (once) |
| `GET` | `/v1/providers` | no | List supported providers |
| `GET` | `/v1/providers/{provider}/models` | no | Model catalog for a provider |
| `POST` | `/v1/providers/add` | yes | Add/update an encrypted provider key |
| `PUT` | `/v1/tiers/config` | yes | Set provider+model per tier |
| `GET` | `/v1/tiers/config` | yes | Read current tier config |
| `POST` | `/query` | yes | Classify → cache → route → respond |
| `GET` | `/analytics` | yes | Aggregated cost/latency/cache metrics |
| `GET` | `/analytics/benchmark` | yes | Model distribution + cache stats |
| `GET` | `/health` | no | Liveness + Redis status |

`/query` response shape:

```json
{
  "request_id": "uuid",
  "response": "…",
  "provider": "groq",
  "model_used": "openai/gpt-oss-20b",
  "tier": "simple",
  "complexity_score": 2,
  "cache_hit": false,
  "latency_ms": 912.4,
  "tokens_used": 431,
  "cost_usd": 0.0000806,
  "similarity_score": 0.0
}
```

---

## Routing & classification

Every query is embedded once with `all-MiniLM-L6-v2`; a 384→64→1 MLP scores complexity from 1 to 10:

| Score | Tier | Typical queries |
|---|---|---|
| 1–3 | `simple` | greetings, arithmetic, lookups |
| 4–6 | `medium` | code generation, comparisons, summarization |
| 7–10 | `complex` | system design, distributed systems, deep reasoning |

The tier maps to **your** configured provider/model. If a call fails, the gateway falls back down the chain (Complex → Medium → Simple). If the daily budget is exceeded, everything routes to the Simple tier.

The classifier runs locally via **ONNX Runtime** — no network, no LLM, deterministic.

---

## Semantic cache

- Per-tenant embeddings stored in Redis; matched by cosine similarity.
- Thresholds are stricter for harder queries (a near-match on a factual question is safe; on a reasoning task it is not).

| Tier | Similarity threshold | TTL |
|---|---|---|
| Simple | 0.90 | 2 h |
| Medium | 0.92 | 1 h |
| Complex | 0.95 | 30 min |

- Cache hits return in ~20 ms at **$0.00** cost and log zero tokens.

---

## Verified performance

Real measurements from local runs:

| Metric | Value |
|---|---|
| Classifier latency (ONNX, avg) | **~7.6 ms** |
| Cache-hit response | **~20 ms** |
| Cache speedup vs LLM call | **up to ~32x** |
| Cost saved (example workload) | **~35–90%** |
| Runtime memory (peak RSS) | **~269 MB** |
| Docker image size | **~928 MB** |
| E2E suite | **8/8 passing** |

> The classifier pairs a pre-trained `all-MiniLM-L6-v2` encoder with a small MLP trained on a labeled query corpus (see `train_classifier.py`). Training data is not committed; the pipeline scripts are.

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `REDIS_SERVER_LINK` | yes | — | Redis URL (`rediss://…` for TLS) |
| `POSTGRESQL_LINK` | yes | — | Async Postgres URL (`asyncpg`) |
| `ENCRYPTION_KEY` | yes | — | Fernet key for BYOK encryption (**must stay constant**) |
| `SIMILARITY_THRESHOLD_SIMPLE` | no | 0.90 | Cache strictness (simple) |
| `SIMILARITY_THRESHOLD_MEDIUM` | no | 0.92 | Cache strictness (medium) |
| `SIMILARITY_THRESHOLD_COMPLEX` | no | 0.95 | Cache strictness (complex) |
| `CACHE_TTL_SIMPLE` | no | 7200 | Cache TTL seconds (simple) |
| `CACHE_TTL_MEDIUM` | no | 3600 | Cache TTL seconds (medium) |
| `CACHE_TTL_COMPLEX` | no | 1800 | Cache TTL seconds (complex) |
| `DAILY_BUDGET_USD` | no | 10.0 | Default daily budget for new tenants |
| `EVALUATION_LOOP_RATE` | no | 0.10 | Share of requests sent to shadow evaluation |

> ⚠️ `ENCRYPTION_KEY` must not change after data exists, or stored provider keys become undecryptable.

---

## Deployment (free tier)

The app runs as a single stateless container; Postgres and Redis are external managed services.

| Component | Service | Notes |
|---|---|---|
| App | Render (Docker) | Free 512 MB; keep-awake via uptime ping |
| Postgres | Neon | Use `?ssl=require` (asyncpg, **not** `sslmode`) |
| Redis | Upstash | Use the `rediss://` TLS endpoint |

Render environment variables:

```env
POSTGRESQL_LINK=postgresql+asyncpg://<user>:<pass>@<neon-host>/<db>?ssl=require
REDIS_SERVER_LINK=rediss://default:<pass>@<upstash-host>:6379
ENCRYPTION_KEY=<stable-fernet-key>
```

Notes:
- Do **not** set `PORT` — the container binds to the platform-provided `PORT` automatically.
- Health check path: `/health`.
- Free services sleep; a 5-minute uptime monitor on `/health` keeps Render awake within its monthly quota.

---

## Project structure

```text
llm-gateway/
├── app/
│   ├── main.py            # FastAPI app, endpoints, query pipeline
│   ├── config.py          # Environment settings
│   ├── models.py          # SQLAlchemy models + Pydantic schemas
│   ├── auth.py            # X-Gateway-Key auth dependency
│   ├── security.py        # Key generation + Fernet encryption
│   ├── classifier.py      # ONNX embedder + NumPy MLP (complexity scoring)
│   ├── router.py          # Tier → model resolution + fallback + budget
│   ├── cache.py           # Semantic cache (Redis + cosine similarity)
│   ├── llm_client.py      # Multi-provider async client + model registry
│   ├── analytics.py       # Request logging + aggregation
│   ├── evaluation.py      # Background shadow evaluation
│   ├── classifier_mlp.pth # Trained MLP weights (converted at build time)
│   └── templates/
│       └── dashboard.html # Single-file dashboard (Tailwind + Chart.js)
├── scripts/
│   └── export_onnx.py     # Build-time ONNX export (runs in builder stage)
├── test_e2e.py            # End-to-end API test suite (8 checks)
├── benchmark.py           # Latency/cost benchmark across all tiers
├── Dockerfile             # Multi-stage build (torch only in builder)
├── docker-compose.yaml    # Local app + Redis + Postgres
├── requirements.txt       # Runtime dependencies
├── requirements-train.txt # Training-only dependencies (torch, sentence-transformers)
├── .env.example           # Environment template
└── Readme.md
```

---

## Testing

```bash
# With the app running on http://localhost:8000
GROQ_API_KEY=gsk_... python test_e2e.py   # 8 automated checks
GROQ_API_KEY=gsk_... python benchmark.py # latency/cost benchmark
```

---

## Security

- Provider keys are **Fernet-encrypted at rest**; decrypted only in-process.
- Gateway keys are stored **hashed**; the raw key is shown only once at registration.
- Container runs as a **non-root** user.
- Baseline security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, HSTS over TLS).
- Provider errors are logged server-side; API responses do not leak upstream internals.
- `.env` and generated model artifacts are git-ignored.

---

## Retraining the classifier

```bash
pip install -r requirements-train.txt
python train_classifier.py     # produces app/classifier_mlp.pth
```

The Docker build converts the `.pth` to `.npz` and exports the embedder to ONNX automatically — no PyTorch is shipped in the runtime image.

---

## Roadmap

- Vector index (FAISS/pgvector) for larger caches
- Per-tenant rate limiting
- More granular cost dashboards
- Streaming responses

---

## License

Add a license (e.g., MIT) before publishing if you intend others to reuse this.
