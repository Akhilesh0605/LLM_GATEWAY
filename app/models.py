# app/models.py
"""
Database Models (Async SQLAlchemy) & Pydantic Schemas.
Single source of truth for all data shapes in the gateway.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Enum as SAEnum
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ============================================================================
# ENUMS
# ============================================================================

class ProviderEnum(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"


class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


# ============================================================================
# SQLALCHEMY ORM MODELS
# ============================================================================

class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_gateway_key = Column(String(64), unique=True, index=True, nullable=False)
    daily_budget_usd = Column(Float, default=10.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    provider_keys = relationship("UserProviderKeyDB", back_populates="user", cascade="all, delete-orphan")
    tier_config = relationship("UserTierConfigDB", back_populates="user", uselist=False, cascade="all, delete-orphan")
    logs = relationship("RequestLogDB", back_populates="user")


class UserProviderKeyDB(Base):
    """
    1-to-many: User can store API keys for multiple providers.
    Keys are Fernet-encrypted at rest, decrypted in-memory only during requests.
    """
    __tablename__ = "user_provider_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider = Column(SAEnum(ProviderEnum), nullable=False)
    label = Column(String(50), nullable=False)  # e.g., "My Groq Key", "Work OpenAI"
    encrypted_api_key = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserDB", back_populates="provider_keys")


class UserTierConfigDB(Base):
    """
    1-to-1: User's model assignment per complexity tier.
    Users pick provider + model from dropdown for each tier.
    """
    __tablename__ = "user_tier_configs"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)

    simple_provider = Column(SAEnum(ProviderEnum), nullable=False)
    simple_model = Column(String(100), nullable=False)

    medium_provider = Column(SAEnum(ProviderEnum), nullable=False)
    medium_model = Column(String(100), nullable=False)

    complex_provider = Column(SAEnum(ProviderEnum), nullable=False)
    complex_model = Column(String(100), nullable=False)

    user = relationship("UserDB", back_populates="tier_config")


class RequestLogDB(Base):
    __tablename__ = "request_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    query = Column(String, nullable=False)
    provider = Column(String(20), nullable=False)
    model_used = Column(String(100), nullable=False)
    tier = Column(String(10), nullable=False)
    complexity_score = Column(Float, nullable=False)
    cache_hit = Column(Boolean, nullable=False)
    latency_ms = Column(Float, nullable=False)
    tokens_used = Column(Integer, nullable=False)
    cost_usd = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("UserDB", back_populates="logs")


class EvaluationLogDB(Base):
    __tablename__ = "evaluation_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String, nullable=False)
    query = Column(String, nullable=False)
    routed_model = Column(String, nullable=False)
    complexity_score = Column(Float, nullable=False)
    agreement_score = Column(Float, nullable=False)
    triggered_by = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ============================================================================
# PYDANTIC SCHEMAS (API Request/Response Validation)
# ============================================================================

# --- Auth ---

class UserRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr


class UserRegisterResponse(BaseModel):
    name: str
    email: str
    gateway_api_key: str
    message: str = "Store this gateway_api_key securely. It will not be shown again."


# --- Provider Keys ---

class ProviderKeyRequest(BaseModel):
    provider: ProviderEnum
    api_key: str = Field(..., min_length=10)
    label: str = Field(default="default", max_length=50)


class ProviderKeyResponse(BaseModel):
    provider: str
    label: str
    message: str


# --- Tier Config ---

class TierConfigRequest(BaseModel):
    simple_provider: ProviderEnum
    simple_model: str
    medium_provider: ProviderEnum
    medium_model: str
    complex_provider: ProviderEnum
    complex_model: str


class TierConfigResponse(BaseModel):
    message: str
    config: dict


# --- Query ---

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)


class QueryResponse(BaseModel):
    request_id: str
    response: str
    provider: str
    model_used: str
    tier: ComplexityTier
    complexity_score: int
    cache_hit: bool
    latency_ms: float = Field(ge=0)
    tokens_used: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    similarity_score: float | None = None


# --- Routing ---

class RouteDecision(BaseModel):
    complexity_score: int = Field(ge=1, le=10)
    tier: ComplexityTier
    provider: str
    selected_model: str
    budget_exceeded: bool = False
    is_boundary: bool = False