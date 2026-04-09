from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings
import time

settings=get_settings()
client=AsyncGroq(api_key=settings.GROQ_API_KEY)

COST_PER_MILLION_TOKENS = {
    settings.MODEL_SIMPLE: 0.05,
    settings.MODEL_MEDIUM: 0.59,
    settings.MODEL_COMPLEX: 2.00,
}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1,min=1,max=4)
)
async def _call_groq(model:str,query:str):
    response = await client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": query}],
    max_tokens=1000,
    )
    return response

#returns responce,tokens used,costusd,latency in milliseconds
async def call_llm(
        model:str,
        query:str,
        fallback_chain:list[str]
    ) -> tuple[str,int,float,float]:
    models_to_try=[model]+fallback_chain
    for current_model in models_to_try:
        try:
            start=time.perf_counter()
            response=await _call_groq(current_model,query)
            latency_ms = (time.perf_counter() - start) * 1000

            response_text=response.choices[0].message.content
            tokens_used=response.usage.total_tokens

            cost_usd=(tokens_used/1_000_000) * COST_PER_MILLION_TOKENS.get(current_model,0.0)

            return response_text, tokens_used, cost_usd, latency_ms
    
        except Exception:
            continue
    raise RuntimeError("All models in fallback chain failed")