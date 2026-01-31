"""JWT token handling."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from pydantic import BaseModel

from codeverify_api.config import settings


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str  # User ID
    exp: datetime
    iat: datetime
    github_id: int
    username: str


class TokenData(BaseModel):
    """Decoded token data."""

    user_id: UUID
    github_id: int
    username: str


def create_access_token(
    user_id: UUID,
    github_id: int,
    username: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a new JWT access token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=settings.JWT_EXPIRATION_HOURS)

    payload = {
        "sub": str(user_id),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "github_id": github_id,
        "username": username,
    }

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenData | None:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return TokenData(
            user_id=UUID(payload["sub"]),
            github_id=payload["github_id"],
            username=payload["username"],
        )
    except JWTError:
        return None


async def get_current_user(token: str) -> TokenData | None:
    """Get current user from token."""
    return decode_access_token(token)
