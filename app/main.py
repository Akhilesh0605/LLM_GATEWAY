from fastapi import FastAPI, HTTPException, BackgroundTasks
from uuid import uuid4
import time
import random
import redis.asyncio as redis

from app.models import QueryRequest, QueryResponse
from app.classifier import classify
from app.router import route
from app.cache import check_cache, store_in_cache
from app.llm_client import call_llm
from app.analytics import log_request, get_analytics, get_benchmark
from app.evaluation import run_evaluation
from app.config import get_settings

app = FastAPI()
settings = get_settings()

# Redis client
redis_client = redis.from_url(settings.REDIS_SERVER_LINK)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    request_id = str(uuid4())
    query = request.query

    start_time = time.perf_counter()

    try:
        # Step 1: classify
        complexity_score, tier = classify(query)

        # Step 2: route
        decision = await route(complexity_score, tier, redis_client)
        model = decision.selected_model

        # Step 3: cache check
        cached_response,similarity_score = await check_cache(query, tier, redis_client)

        if cached_response:
            latency = (time.perf_counter() - start_time) * 1000

            await log_request(
                request_id=request_id,
                query=query,
                model_used=model,
                complexity_score=complexity_score,
                tier=tier,
                cache_hit=True,
                latency_ms=latency,
                tokens_used=0,
                cost_usd=0.0
            )


            return QueryResponse(
                request_id=request_id,
                response=cached_response,
                model_used=model,
                tier=tier,
                complexity_score=complexity_score,
                cache_hit=True,
                latency_ms=latency,
                tokens_used=0,
                cost_usd=0.0,
                similarity_score=round(similarity_score,3)
            )

        # Step 4: call LLM
        response_text, tokens_used, cost_usd, latency_llm = await call_llm(
            model=model,
            query=query,
            fallback_chain=[]
        )

        # Step 5: store cache
        await store_in_cache(query, response_text, tier, redis_client)

        total_latency = (time.perf_counter() - start_time) * 1000

        # Step 6: log request
        await log_request(
            request_id=request_id,
            query=query,
            model_used=model,
            complexity_score=complexity_score,
            tier=tier,
            cache_hit=False,
            latency_ms=total_latency,
            tokens_used=tokens_used,
            cost_usd=cost_usd
        )

        # Step 7: evaluation trigger
        if decision.is_boundary or random.random() < settings.EVALUATION_LOOP_RATE:
            background_tasks.add_task(
                run_evaluation,
                request_id,
                query,
                response_text,
                model,
                complexity_score,
                "auto"
            )

        return QueryResponse(
            request_id=request_id,
            response=response_text,
            model_used=model,
            tier=tier,
            complexity_score=complexity_score,
            cache_hit=False,
            latency_ms=total_latency,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            similarity_score=round(similarity_score,3)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics")
async def analytics():
    return await get_analytics()


@app.get("/analytics/benchmark")
async def benchmark():
    return await get_benchmark()


@app.get("/health")
async def health():
    try:
        await redis_client.ping()
        return {"status": "healthy"}
    except Exception:
        raise HTTPException(status_code=500, detail="Redis not reachable")