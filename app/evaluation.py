from app.cache import get_embedding, cosine_similarity
from app.llm_client import call_llm
from app.analytics import log_evaluation
from app.config import get_settings

settings = get_settings()

async def run_evaluation(
    request_id: str,
    query: str,
    routed_response: str,
    routed_model: str,
    complexity_score: int,
    triggered_by: str,
) -> None:

    try:
        strong_response, _, _, _ = await call_llm(
            model=settings.MODEL_COMPLEX,
            query=query,
            fallback_chain=[]
        )
    except Exception:
        return

    if not strong_response:
        return


    routed_embedding = get_embedding(routed_response)
    strong_embedding = get_embedding(strong_response)


    agreement_score = cosine_similarity(routed_embedding, strong_embedding)

    await log_evaluation(
        request_id=request_id,
        query=query,
        routed_model=routed_model,
        complexity_score=complexity_score,
        agreement_score=agreement_score,
        triggered_by=triggered_by
    )