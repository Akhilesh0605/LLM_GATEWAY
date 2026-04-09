from pydantic_settings import BaseSettings,SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    GROQ_API_KEY:str #groq api
    REDIS_SERVER_LINK:str #redis server
    POSTGRESQL_LINK:str #postgreslink
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    MODEL_SIMPLE: str = "llama-3.1-8b-instant"  #models names to route
    MODEL_MEDIUM: str = "llama-3.3-70b-versatile"
    MODEL_COMPLEX: str = "openai/gpt-oss-120b"
  
    SIMILARITY_THRESHOLD_SIMPLE: float = 0.90  #threshold to route
    SIMILARITY_THRESHOLD_MEDIUM: float = 0.92
    SIMILARITY_THRESHOLD_COMPLEX: float = 0.95

    CACHE_TTL_SIMPLE: int = 7200 # TTL - time to live keep the queries based on complexity
    CACHE_TTL_MEDIUM: int = 3600
    CACHE_TTL_COMPLEX: int = 1800


    DAILY_BUDGET_USD: float = 10.0 #budeget threshold


    EVALUATION_LOOP_RATE: float = 0.10 



@lru_cache
def get_settings() -> Settings:
    return Settings()