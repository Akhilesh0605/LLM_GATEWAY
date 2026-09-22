# app/evaluation.py
"""
Asynchronous Multi-Tenant Shadow Evaluation Engine.
Generates baseline teacher outputs from the user's configured COMPLEX model
and calculates semantic agreement scores to evaluate routing precision.
"""

import random
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.config import get_settings
from app.analytics import async_session, log_evaluation
from app.models import UserTierConfigDB, UserProviderKeyDB, ProviderEnum, ComplexityTier
from app.security import decrypt_api_key
from app.llm_client import llm_client
from app.cache import cosine_similarity, get_embedding

logger = logging.getLogger(__name__)
settings = get_settings()


async def _resolve_complex_target(db: AsyncSession, user_id: int) -> tuple[str, str, str]:
    """
    Returns (provider, model, decrypted_key) configured for the user's COMPLEX tier.
    """
    # 1. Fetch user's COMPLEX tier configuration
    config_result = await db.execute(
        select(UserTierConfigDB).filter(UserTierConfigDB.user_id == user_id)
    )
    config = config_result.scalars().first()
    
    if not config:
        raise ValueError("User has not configured their COMPLEX tier.")
        
    provider = config.complex_provider.value
    model = config.complex_model
    
    # 2. Retrieve corresponding provider API key
    key_result = await db.execute(
        select(UserProviderKeyDB).filter(
            UserProviderKeyDB.user_id == user_id,
            UserProviderKeyDB.provider == ProviderEnum(provider),
            UserProviderKeyDB.is_active == True
        )
    )
    key_entry = key_result.scalars().first()
    
    if not key_entry:
        raise ValueError(f"No active key registered for COMPLEX provider: {provider}")
        
    decrypted_key = decrypt_api_key(key_entry.encrypted_api_key)
    return provider, model, decrypted_key


async def run_evaluation(
    request_id: str,
    user_id: int,
    query: str,
    routed_model: str,
    complexity_score: float,
    is_boundary: bool = False,
) -> None:
    """
    Shadow Evaluation:
    Evaluates 100% of boundary routing scores (3, 7) and a 10% random sample
    against the user's configured COMPLEX teacher model to verify semantic alignment.
    """
    should_evaluate = is_boundary or (random.random() < settings.EVALUATION_LOOP_RATE)
    if not should_evaluate:
        return

    triggered_by = "boundary_score" if is_boundary else "random_sample"
    messages = [{"role": "user", "content": query}]

    # Run DB lookups inside an isolated async session to stay background-safe
    async with async_session() as db:
        try:
            teacher_provider, teacher_model, teacher_key = await _resolve_complex_target(db, user_id)

            # If routed model is already the COMPLEX teacher model, semantic agreement is 1.0
            if routed_model == teacher_model:
                await log_evaluation(
                    request_id=request_id,
                    query=query,
                    routed_model=routed_model,
                    complexity_score=complexity_score,
                    agreement_score=1.0,
                    triggered_by=triggered_by,
                )
                return

            # Call Teacher Model (Complex)
            teacher_result = await llm_client.chat(
                provider=teacher_provider,
                model=teacher_model,
                api_key=teacher_key,
                messages=messages,
            )
            teacher_response = teacher_result["response"]

            # Generate Routed Response output again (or parse existing - here we recalculate safely)
            # Fetch routed model's provider
            routed_provider_result = await db.execute(
                select(UserTierConfigDB).filter(UserTierConfigDB.user_id == user_id)
            )
            config = routed_provider_result.scalars().first()
            
            # Determine which provider owned the routed model to resolve the correct key
            routed_provider = None
            if routed_model == config.simple_model:
                routed_provider = config.simple_provider.value
            elif routed_model == config.medium_model:
                routed_provider = config.medium_provider.value
            else:
                routed_provider = config.complex_provider.value

            routed_key_result = await db.execute(
                select(UserProviderKeyDB).filter(
                    UserProviderKeyDB.user_id == user_id,
                    UserProviderKeyDB.provider == ProviderEnum(routed_provider),
                    UserProviderKeyDB.is_active == True
                )
            )
            routed_key_entry = routed_key_result.scalars().first()
            routed_key = decrypt_api_key(routed_key_entry.encrypted_api_key)

            routed_result = await llm_client.chat(
                provider=routed_provider,
                model=routed_model,
                api_key=routed_key,
                messages=messages,
            )
            routed_response = routed_result["response"]

            # Calculate semantic cosine similarity between routed response and teacher response
            routed_emb = get_embedding(routed_response)
            teacher_emb = get_embedding(teacher_response)
            agreement_score = float(cosine_similarity(routed_emb, teacher_emb))

            # Log evaluation metrics to Postgres
            await log_evaluation(
                request_id=request_id,
                query=query,
                routed_model=routed_model,
                complexity_score=complexity_score,
                agreement_score=round(agreement_score, 3),
                triggered_by=triggered_by,
            )

        except Exception as e:
            # Shadow tasks fail silently to ensure client requests are never blocked
            logger.warning(f"Shadow evaluation bypassed for request {request_id}: {e}")