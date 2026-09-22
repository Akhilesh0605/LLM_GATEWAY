# app/router.py
"""
Dynamic Multi-Provider Gateway Router with Resilient Fallback Chain.
Maps complexity tier to user-configured models, enforces per-tenant daily budget in Redis,
decrypts provider keys in-memory, and handles cascading fallbacks on 5xx errors.
"""

import logging
from datetime import date
from typing import Dict, Any, Tuple, Optional
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models import (
    UserDB,
    UserProviderKeyDB,
    UserTierConfigDB,
    ComplexityTier,
    ProviderEnum,
)
from app.security import decrypt_api_key
from app.llm_client import llm_client

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER: IN-MEMORY KEY & TIER RESOLUTION
# ============================================================================

async def _get_user_tier_target(
    db: AsyncSession,
    user_id: int,
    tier: ComplexityTier,
) -> Tuple[str, str]:
    """
    Fetches the configured (provider, model) for the given tier from DB.
    """
    result = await db.execute(
        select(UserTierConfigDB).filter(UserTierConfigDB.user_id == user_id)
    )
    tier_config = result.scalars().first()

    if not tier_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User tier configuration not found. Please configure your models at /v1/tiers/config first.",
        )

    if tier == ComplexityTier.SIMPLE:
        return tier_config.simple_provider.value, tier_config.simple_model
    elif tier == ComplexityTier.MEDIUM:
        return tier_config.medium_provider.value, tier_config.medium_model
    else:
        return tier_config.complex_provider.value, tier_config.complex_model


async def _get_decrypted_provider_key(
    db: AsyncSession,
    user_id: int,
    provider: str,
) -> str:
    """
    Retrieves and decrypts the user's API key for the specified provider.
    Decryption is performed strictly in-memory.
    """
    result = await db.execute(
        select(UserProviderKeyDB).filter(
            UserProviderKeyDB.user_id == user_id,
            UserProviderKeyDB.provider == ProviderEnum(provider),
            UserProviderKeyDB.is_active == True,
        )
    )
    key_entry = result.scalars().first()

    if not key_entry:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No active API key found for provider '{provider}'. Please add it at /v1/providers/add.",
        )

    try:
        return decrypt_api_key(key_entry.encrypted_api_key)
    except Exception as e:
        logger.error(f"Decryption failure for user {user_id}, provider {provider}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to decrypt provider API key.",
        )


async def _check_budget_exceeded(
    redis_client,
    user_id: int,
    budget_limit: float,
) -> bool:
    """
    Checks per-tenant daily spend in Redis: budget:{user_id}:{YYYY-MM-DD}.
    """
    if not redis_client:
        return False

    today_str = date.today().isoformat()
    key = f"budget:{user_id}:{today_str}"

    try:
        spend = await redis_client.get(key)
        if spend:
            return float(spend) >= budget_limit
    except Exception as e:
        logger.warning(f"Redis budget check failed: {e}")
    return False


# ============================================================================
# MAIN ROUTING & DISPATCH FUNCTION
# ============================================================================

async def route_and_execute(
    db: AsyncSession,
    user: UserDB,
    redis_client,
    query: str,
    complexity_score: int,
    tier: ComplexityTier,
) -> Dict[str, Any]:
    """
    Complete routing pipeline:
    1. Budget Enforcement -> Force SIMPLE if limit breached.
    2. Model Resolution -> Fetch user's model mapping for the tier.
    3. Key Decryption -> In-memory decryption of provider credentials.
    4. Execution with Cascading Fallback:
       - COMPLEX fails -> fallback to MEDIUM
       - MEDIUM fails  -> fallback to SIMPLE
       - SIMPLE fails  -> 502 Bad Gateway
    """
    is_boundary = complexity_score in (3, 7)

    # 1. Budget Gate
    budget_exceeded = await _check_budget_exceeded(
        redis_client=redis_client,
        user_id=user.id,
        budget_limit=user.daily_budget_usd,
    )

    effective_tier = ComplexityTier.SIMPLE if budget_exceeded else tier
    if budget_exceeded:
        logger.warning(f"Daily budget exceeded for user {user.id}. Routing forced to SIMPLE tier.")

    # 2. Build Cascading Execution Chain
    execution_chain = []

    # Primary target
    provider, model = await _get_user_tier_target(db, user.id, effective_tier)
    execution_chain.append((effective_tier, provider, model))

    # Add fallbacks if starting at higher tiers
    if effective_tier == ComplexityTier.COMPLEX:
        med_p, med_m = await _get_user_tier_target(db, user.id, ComplexityTier.MEDIUM)
        execution_chain.append((ComplexityTier.MEDIUM, med_p, med_m))

    if effective_tier in (ComplexityTier.COMPLEX, ComplexityTier.MEDIUM):
        simp_p, simp_m = await _get_user_tier_target(db, user.id, ComplexityTier.SIMPLE)
        execution_chain.append((ComplexityTier.SIMPLE, simp_p, simp_m))

    # 3. Execute Down the Chain
    messages = [{"role": "user", "content": query}]
    last_exception = None
    fallback_occurred = False

    for attempt_idx, (tier_target, target_provider, target_model) in enumerate(execution_chain):
        try:
            if attempt_idx > 0:
                logger.warning(f"Executing fallback attempt #{attempt_idx} -> {target_provider}/{target_model}")
                fallback_occurred = True

            # Decrypt key for target provider
            decrypted_key = await _get_decrypted_provider_key(db, user.id, target_provider)

            # Call LLM
            result = await llm_client.chat(
                provider=target_provider,
                model=target_model,
                api_key=decrypted_key,
                messages=messages,
            )

            # Success -> Return response with full routing metadata
            return {
                "response": result["response"],
                "provider": target_provider,
                "model_used": target_model,
                "tier": tier_target,
                "tokens_used": result["tokens_used"],
                "cost_usd": result["cost_usd"],
                "latency_ms": result["latency_ms"],
                "is_boundary": is_boundary,
                "budget_exceeded": budget_exceeded,
                "fallback_occurred": fallback_occurred,
            }

        except Exception as e:
            logger.error(f"Provider {target_provider} ({target_model}) call failed: {e}")
            last_exception = e
            continue

    # 4. If all fallbacks failed
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"All configured LLM providers failed. Last error: {str(last_exception)}",
    )