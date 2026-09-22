# app/cache.py
"""
Tenant-Isolated Semantic Cache Engine.
Uses cosine similarity over pre-computed MiniLM vector embeddings.
Avoids duplicate transformer allocation in memory.
"""

import json
import numpy as np
import redis.asyncio as redis
from typing import Tuple, List, Optional, Union

from app.config import get_settings
from app.models import ComplexityTier

settings = get_settings()

threshold_map = {
    ComplexityTier.SIMPLE: settings.SIMILARITY_THRESHOLD_SIMPLE,
    ComplexityTier.MEDIUM: settings.SIMILARITY_THRESHOLD_MEDIUM,
    ComplexityTier.COMPLEX: settings.SIMILARITY_THRESHOLD_COMPLEX,
}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_embedding(text: str) -> np.ndarray:
    """
    Helper to access the singleton classifier's embedder to prevent loading
    duplicate 400MB SentenceTransformer models into RAM.
    """
    from app.classifier import get_classifier
    return get_classifier().encode(text)


async def check_cache(
    query: str,
    tier: ComplexityTier,
    redis_client: redis.Redis,
    user_id: int,
    query_embedding: Optional[Union[np.ndarray, List[float]]] = None,
) -> Tuple[Optional[str], float]:
    """
    Checks semantic cache for matching responses from the same user.
    Reuses the classification stage embedding if available to bypass transformer inference.
    """
    if query_embedding is None:
        emb_arr = get_embedding(query)
    elif isinstance(query_embedding, list):
        emb_arr = np.array(query_embedding, dtype=np.float32)
    else:
        emb_arr = query_embedding

    # Scoped strictly to the specific tenant in Redis
    emb_key = f"user:{user_id}:cache:embeddings"
    resp_key = f"user:{user_id}:cache:responses"

    raw_embeddings = await redis_client.get(emb_key)
    raw_responses = await redis_client.get(resp_key)

    if not raw_embeddings or not raw_responses:
        return None, 0.0

    try:
        embeddings = json.loads(raw_embeddings)
        responses = json.loads(raw_responses)
    except Exception:
        return None, 0.0

    best_score = 0.0
    best_response = None
    threshold = threshold_map[tier]

    for i, emb in enumerate(embeddings):
        emb_np = np.array(emb, dtype=np.float32)
        score = cosine_similarity(emb_arr, emb_np)

        if score > best_score:
            best_score = score
            best_response = responses[i]

    # Return match only if similarity meets the threshold defined for that tier
    if best_score >= threshold:
        return best_response, best_score

    return None, best_score


async def store_in_cache(
    query: str,
    response: str,
    tier: ComplexityTier,
    redis_client: redis.Redis,
    user_id: int,
    query_embedding: Optional[Union[np.ndarray, List[float]]] = None,
) -> None:
    """
    Saves a query, its embedding, and its response to the user's Redis cache block.
    """
    if query_embedding is None:
        emb_arr = get_embedding(query)
    elif isinstance(query_embedding, list):
        emb_arr = np.array(query_embedding, dtype=np.float32)
    else:
        emb_arr = query_embedding

    emb_key = f"user:{user_id}:cache:embeddings"
    resp_key = f"user:{user_id}:cache:responses"

    raw_embeddings = await redis_client.get(emb_key)
    raw_responses = await redis_client.get(resp_key)

    if raw_embeddings and raw_responses:
        try:
            embeddings = json.loads(raw_embeddings)
            responses = json.loads(raw_responses)
        except Exception:
            embeddings = []
            responses = []
    else:
        embeddings = []
        responses = []

    embeddings.append(emb_arr.tolist())
    responses.append(response)

    await redis_client.set(emb_key, json.dumps(embeddings))
    await redis_client.set(resp_key, json.dumps(responses))

    # Set cache expiry based on query complexity (shorter TTL for complex models)
    ttl_map = {
        ComplexityTier.SIMPLE: settings.CACHE_TTL_SIMPLE,
        ComplexityTier.MEDIUM: settings.CACHE_TTL_MEDIUM,
        ComplexityTier.COMPLEX: settings.CACHE_TTL_COMPLEX,
    }
    ttl = ttl_map[tier]
    await redis_client.expire(emb_key, ttl)
    await redis_client.expire(resp_key, ttl)