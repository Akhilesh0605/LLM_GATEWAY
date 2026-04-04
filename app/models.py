from pydantic import BaseModel,Field
from enum import Enum
from uuid import UUID

class ComplexityTier(str,Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class RouteDecision(BaseModel):
    complexity_score: int = Field(ge=1, le=10)
    tier: ComplexityTier
    selected_model: str
    budget_exceeded: bool = False

class QueryRequest(BaseModel):
    query : str=Field(min_length=1,max_length=5000)


class QueryResponse(BaseModel):
    request_id:UUID
    response:str
    model_used:str
    tier:ComplexityTier
    complexity_score:int
    cache_hit:bool
    latency_ms:float=Field(ge=0)
    tokens_used:int = Field(ge=0)
    cost_usd:float = Field(ge=0)

     