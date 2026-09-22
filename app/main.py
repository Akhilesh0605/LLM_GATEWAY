# app/main.py
"""
Production Multi-Tenant LLM Gateway API.
Handles onboarding, BYOK provider key management, dynamic model tiering,
Redis semantic caching, neural complexity classification, and routing.
"""

import uuid
from datetime import date
from typing import List
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.config import get_settings
from app.models import (
    UserDB,
    UserProviderKeyDB,
    UserTierConfigDB,
    ProviderEnum,
    UserRegisterRequest,
    UserRegisterResponse,
    ProviderKeyRequest,
    ProviderKeyResponse,
    TierConfigRequest,
    TierConfigResponse,
    QueryRequest,
    QueryResponse,
)
from app.security import generate_gateway_key, encrypt_api_key
from app.auth import get_current_user, get_db
from app.classifier import get_classifier
from app.router import route_and_execute
from app.cache import check_cache, store_in_cache
from app.llm_client import get_provider_models, get_all_providers
from app.analytics import log_request, get_analytics, get_benchmark, init_db
from app.evaluation import run_evaluation

settings = get_settings()
redis_client = None
classifier_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, classifier_service

    # 1. Initialize DB tables
    await init_db()

    # 2. Connect to Redis
    redis_client = redis.from_url(settings.REDIS_SERVER_LINK, decode_responses=False)

    # 3. Load Neural Classifier singleton (<2ms CPU inference)
    classifier_service = get_classifier()

    yield

    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="Multi-Tenant Cost-Aware LLM Gateway",
    description="Provider-agnostic AI proxy with semantic caching, neural routing, and daily budget controls.",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================================
# 1. ONBOARDING & PROFILE ENDPOINTS
# ============================================================================

@app.post("/v1/auth/register", response_model=UserRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: UserRegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Step 1 of Onboarding: Register a user profile with name and email.
    Generates and returns the SHA-256 backed X-Gateway-Key (shown ONCE).
    """
    # Check if email is already registered
    existing_user = await db.execute(select(UserDB).filter(UserDB.email == request.email))
    if existing_user.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists.",
        )

    raw_gateway_key, hashed_gateway_key = generate_gateway_key()

    new_user = UserDB(
        name=request.name,
        email=request.email,
        hashed_gateway_key=hashed_gateway_key,
        daily_budget_usd=settings.DAILY_BUDGET_USD,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return UserRegisterResponse(
        name=new_user.name,
        email=new_user.email,
        gateway_api_key=raw_gateway_key,
    )


# ============================================================================
# 2. PROVIDER KEYS & MODEL CATALOG ENDPOINTS
# ============================================================================

@app.get("/v1/providers", response_model=List[str])
async def list_providers():
    """Returns all supported AI providers (groq, openai, gemini, anthropic, together)."""
    return get_all_providers()


@app.get("/v1/providers/{provider}/models", response_model=List[str])
async def list_models_for_provider(provider: str):
    """
    Returns the curated model catalog for a provider (used for frontend dropdowns).
    Ensures model-provider compatibility.
    """
    try:
        return get_provider_models(provider)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.post("/v1/providers/add", response_model=ProviderKeyResponse)
async def add_provider_key(
    request: ProviderKeyRequest,
    user: UserDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Step 2 of Onboarding: Add or update a third-party API key for a provider.
    The key is encrypted with Fernet AES-128 before DB persistence.
    """
    encrypted_key = encrypt_api_key(request.api_key)

    # Check if key for this provider already exists for the user
    stmt = select(UserProviderKeyDB).filter(
        UserProviderKeyDB.user_id == user.id,
        UserProviderKeyDB.provider == request.provider,
    )
    existing_key = (await db.execute(stmt)).scalars().first()

    if existing_key:
        existing_key.encrypted_api_key = encrypted_key
        existing_key.label = request.label
        existing_key.is_active = True
    else:
        new_key = UserProviderKeyDB(
            user_id=user.id,
            provider=request.provider,
            label=request.label,
            encrypted_api_key=encrypted_key,
            is_active=True,
        )
        db.add(new_key)

    await db.commit()

    return ProviderKeyResponse(
        provider=request.provider.value,
        label=request.label,
        message=f"API key for {request.provider.value} saved and encrypted successfully.",
    )


# ============================================================================
# 3. TIER MODEL CONFIGURATION ENDPOINTS
# ============================================================================

@app.put("/v1/tiers/config", response_model=TierConfigResponse)
async def set_tier_config(
    request: TierConfigRequest,
    user: UserDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Step 3 of Onboarding: Assign provider + model for each complexity tier.
    Users pick models from the provider's dropdown list.
    """
    stmt = select(UserTierConfigDB).filter(UserTierConfigDB.user_id == user.id)
    config = (await db.execute(stmt)).scalars().first()

    if not config:
        config = UserTierConfigDB(
            user_id=user.id,
            simple_provider=request.simple_provider,
            simple_model=request.simple_model,
            medium_provider=request.medium_provider,
            medium_model=request.medium_model,
            complex_provider=request.complex_provider,
            complex_model=request.complex_model,
        )
        db.add(config)
    else:
        config.simple_provider = request.simple_provider
        config.simple_model = request.simple_model
        config.medium_provider = request.medium_provider
        config.medium_model = request.medium_model
        config.complex_provider = request.complex_provider
        config.complex_model = request.complex_model

    await db.commit()

    return TierConfigResponse(
        message="Tier configuration updated successfully.",
        config={
            "simple": f"{request.simple_provider.value} / {request.simple_model}",
            "medium": f"{request.medium_provider.value} / {request.medium_model}",
            "complex": f"{request.complex_provider.value} / {request.complex_model}",
        },
    )


@app.get("/v1/tiers/config")
async def get_user_tier_config(
    user: UserDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieves the current user's model configuration across tiers."""
    stmt = select(UserTierConfigDB).filter(UserTierConfigDB.user_id == user.id)
    config = (await db.execute(stmt)).scalars().first()

    if not config:
        return {"configured": False, "message": "No tier configuration found. Please configure tiers."}

    return {
        "configured": True,
        "simple": {"provider": config.simple_provider.value, "model": config.simple_model},
        "medium": {"provider": config.medium_provider.value, "model": config.medium_model},
        "complex": {"provider": config.complex_provider.value, "model": config.complex_model},
    }


# ============================================================================
# 4. MAIN RUNTIME QUERY ENDPOINT
# ============================================================================

@app.post("/query", response_model=QueryResponse)
async def handle_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    user: UserDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Main Gateway Query Pipeline:
    1. Neural Complexity Classification (<0.5ms) -> score (1-10) + tier
    2. Tenant-Isolated Semantic Cache Check in Redis -> return in ~15ms on HIT
    3. Dynamic Routing & In-Memory Key Decryption -> call provider with fallback chain
    4. Async Cache Invalidation/Storage, Budget Tracking, and DB Logging
    """
    request_id = str(uuid.uuid4())

    # Step 1: Neural classification
    complexity_score, tier = classifier_service.classify(request.query)

    # Step 2: Check Redis Semantic Cache
    cached_response, similarity_score = await check_cache(
        query=request.query,
        tier=tier,
        redis_client=redis_client,
        user_id=user.id,
    )

    # Step 3 (Cache HIT): Return cached answer ($0 cost, ~15ms)
    if cached_response is not None:
        background_tasks.add_task(
            log_request,
            request_id=request_id,
            query=request.query,
            provider="cache",
            model_used="semantic-cache",
            complexity_score=complexity_score,
            tier=tier.value,
            cache_hit=True,
            latency_ms=0.0,
            tokens_used=0,
            cost_usd=0.0,
            user_id=user.id,
        )

        return QueryResponse(
            request_id=request_id,
            response=cached_response,
            provider="cache",
            model_used="semantic-cache",
            tier=tier,
            complexity_score=complexity_score,
            cache_hit=True,
            latency_ms=0.0,
            tokens_used=0,
            cost_usd=0.0,
            similarity_score=round(similarity_score, 3),
        )

    # Step 4 (Cache MISS): Dynamic routing, in-memory decryption, and LLM call with fallback
    result = await route_and_execute(
        db=db,
        user=user,
        redis_client=redis_client,
        query=request.query,
        complexity_score=complexity_score,
        tier=tier,
    )

    # Step 5: Update tenant daily spend in Redis
    today_str = date.today().isoformat()
    budget_key = f"budget:{user.id}:{today_str}"
    if result["cost_usd"] > 0:
        await redis_client.incrbyfloat(budget_key, result["cost_usd"])

    # Step 6: Store in Redis semantic cache
    await store_in_cache(
        query=request.query,
        response=result["response"],
        tier=result["tier"],
        redis_client=redis_client,
        user_id=user.id,
    )

    # Step 7: Async DB request logging
    background_tasks.add_task(
        log_request,
        request_id=request_id,
        query=request.query,
        provider=result["provider"],
        model_used=result["model_used"],
        complexity_score=complexity_score,
        tier=result["tier"].value,
        cache_hit=False,
        latency_ms=result["latency_ms"],
        tokens_used=result["tokens_used"],
        cost_usd=result["cost_usd"],
        user_id=user.id,
    )

    # Step 8: Trigger background shadow evaluation for boundary cases or sampling
    background_tasks.add_task(
        run_evaluation,
        request_id=request_id,
        query=request.query,
        routed_model=result["model_used"],
        complexity_score=complexity_score,
        is_boundary=result["is_boundary"],
    )

    return QueryResponse(
        request_id=request_id,
        response=result["response"],
        provider=result["provider"],
        model_used=result["model_used"],
        tier=result["tier"],
        complexity_score=complexity_score,
        cache_hit=False,
        latency_ms=float(result["latency_ms"]),
        tokens_used=result["tokens_used"],
        cost_usd=result["cost_usd"],
        similarity_score=round(similarity_score, 3) if similarity_score else 0.0,
    )


# ============================================================================
# 5. ANALYTICS & HEALTH ENDPOINTS
# ============================================================================

@app.get("/analytics")
async def analytics(
    user: UserDB = Depends(get_current_user),
):
    return await get_analytics(user_id=user.id)


@app.get("/analytics/benchmark")
async def benchmark(
    user: UserDB = Depends(get_current_user),
):
    return await get_benchmark(user_id=user.id)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "redis_connected": redis_client is not None}