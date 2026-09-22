# app/analytics.py
"""
Asynchronous Analytics Logging and Aggregation Engine.
Saves gateway telemetry to Postgres and generates real-time savings metrics.
"""

from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import get_settings
from app.models import Base, RequestLogDB, EvaluationLogDB

settings = get_settings()
engine = create_async_engine(settings.POSTGRESQL_LINK)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Initializes and builds all physical database tables if they do not exist."""
    from app.models import UserDB, UserProviderKeyDB, UserTierConfigDB
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def log_request(
    request_id: str,
    query: str,
    provider: str,
    model_used: str,
    complexity_score: float,
    tier: str,
    cache_hit: bool,
    latency_ms: float,
    tokens_used: int,
    cost_usd: float,
    user_id: int | None = None,
) -> None:
    """
    Asynchronously logs a complete gateway request transaction to PostgreSQL.
    """
    log_entry = RequestLogDB(
        request_id=request_id,
        user_id=user_id,
        query=query,
        provider=provider,
        model_used=model_used,
        complexity_score=complexity_score,
        tier=tier,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
    )

    async with async_session() as session:
        async with session.begin():
            session.add(log_entry)


async def log_evaluation(
    request_id: str,
    query: str,
    routed_model: str,
    complexity_score: float,
    agreement_score: float,
    triggered_by: str,
) -> None:
    """
    Asynchronously logs background shadow evaluation outputs.
    """
    evaluation_log = EvaluationLogDB(
        request_id=request_id,
        query=query,
        routed_model=routed_model,
        complexity_score=complexity_score,
        agreement_score=agreement_score,
        triggered_by=triggered_by,
    )

    async with async_session() as session:
        async with session.begin():
            session.add(evaluation_log)


async def get_analytics(user_id: int | None = None) -> dict:
    """
    Calculates aggregated analytics and cost savings metrics for a tenant.
    Baseline assumption: Standard gpt-4 cost baseline of $2.00 per 1M tokens.
    """
    where_clause = "WHERE user_id = :user_id" if user_id is not None else ""
    sql_query = text(f"""
        SELECT
            COUNT(*) as total_requests,
            AVG(latency_ms) as avg_latency_ms,
            SUM(cost_usd) as total_cost_usd,
            SUM(CASE WHEN cache_hit = true THEN 1 ELSE 0 END) as cache_hits,
            SUM(tokens_used) as total_tokens
        FROM request_logs
        {where_clause}
    """)

    async with async_session() as session:
        params = {"user_id": user_id} if user_id is not None else {}
        result = await session.execute(sql_query, params)
        row = result.fetchone()

    total_requests = row[0] or 0
    avg_latency_ms = row[1] or 0.0
    total_cost_usd = row[2] or 0.0
    cache_hits     = row[3] or 0
    total_tokens   = row[4] or 0

    cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0.0
    
    # Calculate savings against a standard $2.00 per 1M token benchmark (e.g. GPT-4o-mini equivalent)
    baseline_cost = (total_tokens / 1_000_000) * 2.00
    cost_saved_usd = max(0.0, baseline_cost - total_cost_usd)
    cost_saved_percent = (cost_saved_usd / baseline_cost * 100) if baseline_cost > 0 else 0.0

    return {
        "total_requests": total_requests,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "total_cost_usd": round(total_cost_usd, 4),
        "cache_hit_rate": round(cache_hit_rate, 2),
        "cost_saved_usd": round(cost_saved_usd, 4),
        "cost_saved_percent": round(cost_saved_percent, 2),
        "total_tokens": total_tokens,
    }


async def get_benchmark(user_id: int | None = None) -> dict:
    """
    Computes model usage percentages and cost performance benchmarks.
    """
    where_clause = "WHERE user_id = :user_id" if user_id is not None else ""
    model_distribution_query = text(f"""
        SELECT model_used, COUNT(*) as count
        FROM request_logs
        {where_clause}
        GROUP BY model_used
    """)

    async with async_session() as session:
        params = {"user_id": user_id} if user_id is not None else {}
        result = await session.execute(model_distribution_query, params)
        rows = result.fetchall()

    model_usage = {}
    total = 0

    for row in rows:
        model_name = row[0]
        count = row[1]
        model_usage[model_name] = count
        total += count

    model_distribution = {}
    for model_name, count in model_usage.items():
        percentage = (count / total * 100) if total > 0 else 0.0
        model_distribution[model_name] = {
            "count": count,
            "percentage": round(percentage, 2),
        }

    analytics = await get_analytics(user_id=user_id)
    cache_hit_rate = analytics["cache_hit_rate"]
    cost_saved_percent = analytics["cost_saved_percent"]

    return {
        "total_requests": total,
        "model_distribution": model_distribution,
        "cache_hit_rate": cache_hit_rate,
        "cost_saved_percent": cost_saved_percent,
    }