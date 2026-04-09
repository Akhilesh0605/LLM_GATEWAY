from app.models import ComplexityTier,RouteDecision
from app.config import get_settings
import redis.asyncio as redis
from datetime import date

settings=get_settings()

async def route(complexity_score:int,tier:ComplexityTier,redis_client:redis.Redis) -> RouteDecision :
    today=date.today().isoformat()
    budget_key=f"budget:{today}"

    raw_spend = await redis_client.get(budget_key)
    current_spend=float(raw_spend) if raw_spend else 0.0

    if current_spend>= settings.DAILY_BUDGET_USD:
        return RouteDecision(
            complexity_score=complexity_score,
            tier=tier,
            selected_model=settings.MODEL_SIMPLE,
            budget_exceeded=True
        )

    is_boundary=complexity_score in (3,6)

    model_map={
        ComplexityTier.SIMPLE:settings.MODEL_SIMPLE,
        ComplexityTier.MEDIUM:settings.MODEL_MEDIUM,
        ComplexityTier.COMPLEX:settings.MODEL_COMPLEX
    }

    selected_model=model_map[tier]

    return RouteDecision(
        complexity_score=complexity_score,
        tier=tier,
        selected_model=selected_model,
        budget_exceeded=False,
        is_boundary=is_boundary
    )