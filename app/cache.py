from app.models import ComplexityTier
from app.config import get_settings
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis
import numpy as np
import json 

model=SentenceTransformer("all-MiniLM-L6-v2")

settings = get_settings()
threshold_map = {
    ComplexityTier.SIMPLE: settings.SIMILARITY_THRESHOLD_SIMPLE,
    ComplexityTier.MEDIUM: settings.SIMILARITY_THRESHOLD_MEDIUM,
    ComplexityTier.COMPLEX: settings.SIMILARITY_THRESHOLD_COMPLEX,
}


def cosine_similarity(a:np.ndarray,b:np.ndarray)-> float:
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_embedding(text:str):
    return model.encode(text)

async def check_cache(query:str,tier:ComplexityTier,redis_client)->str|None :
    query_embedding=get_embedding(query)

    raw_embeddings=await redis_client.get("cache:embeddings")
    raw_responses=await redis_client.get("cache:responses")

    if not raw_embeddings or not raw_responses :
        return None,0.0
    
    embeddings=json.loads(raw_embeddings)
    embeddings = [np.array(e) for e in embeddings]
    responses=json.loads(raw_responses)

    best_score = 0.0
    best_response = None
    threshold = threshold_map[tier]
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)

        if score > best_score:
            best_score = score
            best_response = responses[i]

    # Check threshold
    if best_score >= threshold:
        return best_response,best_score

    return None,best_score


async def store_in_cache(query:str,response:str,tier:ComplexityTier,redis_client)->None:
    query_embedding=get_embedding(query)

    embeddings=await redis_client.get("cache:embeddings")
    responses=await redis_client.get("cache:responses")

    if embeddings:
        embeddings=json.loads(embeddings)
        responses=json.loads(responses)
    else:
        embeddings=[]
        responses=[]

    embeddings.append(query_embedding.tolist())
    responses.append(response)

    await redis_client.set("cache:embeddings",json.dumps(embeddings))
    await redis_client.set("cache:responses",json.dumps(responses))

    ttl_map = {
    ComplexityTier.SIMPLE: settings.CACHE_TTL_SIMPLE,
    ComplexityTier.MEDIUM: settings.CACHE_TTL_MEDIUM,
    ComplexityTier.COMPLEX: settings.CACHE_TTL_COMPLEX,
    }
    ttl = ttl_map[tier]
    await redis_client.expire("cache:embeddings", ttl)
    await redis_client.expire("cache:responses", ttl)