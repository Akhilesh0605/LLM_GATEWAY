# app/auth.py
"""
FastAPI Authentication Middleware & Database Session Dependency.
Validates SHA-256 gateway keys for tenant isolation.
"""

from fastapi import Header, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.analytics import async_session
from app.models import UserDB
from app.security import hash_gateway_key


async def get_db():
    """FastAPI dependency yielding async SQLAlchemy database sessions."""
    async with async_session() as session:
        yield session


async def get_current_user(
    x_gateway_key: str = Header(..., alias="X-Gateway-Key"),
    db: AsyncSession = Depends(get_db),
) -> UserDB:
    """
    Validates X-Gateway-Key header and retrieves the authenticated user row.
    Returns: UserDB instance (tenant context).
    
    Raises:
        401 UNAUTHORIZED: If header is missing or key hash does not match.
    """
    if not x_gateway_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing 'X-Gateway-Key' header",
        )

    hashed_key = hash_gateway_key(x_gateway_key)
    result = await db.execute(select(UserDB).filter(UserDB.hashed_gateway_key == hashed_key))
    user = result.scalars().first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Gateway API Key",
        )

    return user