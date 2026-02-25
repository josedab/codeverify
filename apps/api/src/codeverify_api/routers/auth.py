"""Authentication endpoints."""

import secrets
from typing import Annotated, Any
from urllib.parse import urlparse
from uuid import UUID

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.auth.dependencies import TokenData, get_current_user
from codeverify_api.auth.github import GitHubOAuth
from codeverify_api.auth.jwt import create_access_token
from codeverify_api.config import settings
from codeverify_api.db import User, get_db
from codeverify_api.utils.encryption import encrypt_token

router = APIRouter()
logger = structlog.get_logger()

OAUTH_STATE_TTL_SECONDS = 600  # 10-minute expiry for OAuth states
OAUTH_STATE_PREFIX = "oauth_state:"


def _get_redis() -> aioredis.Redis:
    """Get a Redis client for OAuth state storage."""
    return aioredis.from_url(settings.REDIS_URL, decode_responses=True)


async def _store_oauth_state(state: str, redirect_uri: str) -> None:
    """Store OAuth state in Redis with TTL."""
    client = _get_redis()
    try:
        await client.set(f"{OAUTH_STATE_PREFIX}{state}", redirect_uri, ex=OAUTH_STATE_TTL_SECONDS)
    finally:
        await client.aclose()


async def _pop_oauth_state(state: str) -> str | None:
    """Retrieve and delete OAuth state from Redis (atomic pop)."""
    client = _get_redis()
    try:
        key = f"{OAUTH_STATE_PREFIX}{state}"
        pipe = client.pipeline()
        pipe.get(key)
        pipe.delete(key)
        results = await pipe.execute()
        return results[0]
    finally:
        await client.aclose()


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict[str, Any]


class UserResponse(BaseModel):
    """User response model."""

    id: UUID
    github_id: int
    username: str
    email: str | None
    avatar_url: str | None


def _validate_redirect_uri(redirect_uri: str) -> None:
    """Validate redirect_uri against allowed origins to prevent open redirects."""
    parsed = urlparse(redirect_uri)
    allowed_origins = settings.CORS_ORIGINS
    redirect_origin = f"{parsed.scheme}://{parsed.netloc}"
    if redirect_origin not in allowed_origins:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="redirect_uri is not in the list of allowed origins",
        )


@router.get("/login")
async def login(
    redirect_uri: str = Query(..., description="URL to redirect after authentication"),
) -> RedirectResponse:
    """Initiate GitHub OAuth login flow."""
    _validate_redirect_uri(redirect_uri)
    state = secrets.token_urlsafe(32)
    await _store_oauth_state(state, redirect_uri)

    github = GitHubOAuth()
    callback_url = f"{settings.API_HOST}:{settings.API_PORT}/api/v1/auth/callback"
    if settings.ENVIRONMENT != "development":
        callback_url = "https://api.codeverify.dev/api/v1/auth/callback"

    authorize_url = github.get_authorize_url(state=state, redirect_uri=callback_url)

    return RedirectResponse(url=authorize_url)


@router.get("/callback")
async def oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    """Handle GitHub OAuth callback."""
    # Verify state (atomic get-and-delete from Redis)
    redirect_uri = await _pop_oauth_state(state)
    if redirect_uri is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid state parameter",
        )

    github = GitHubOAuth()

    # Exchange code for token
    token_data = await github.exchange_code(code)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to exchange code for token",
        )

    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No access token in response",
        )

    # Get user info
    github_user = await github.get_user(access_token)
    if github_user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to get user info",
        )

    # Get primary email
    email = await github.get_primary_email(access_token)

    # Find or create user
    result = await db.execute(select(User).where(User.github_id == github_user["id"]))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            github_id=github_user["id"],
            username=github_user["login"],
            email=email,
            avatar_url=github_user.get("avatar_url"),
            access_token_encrypted=encrypt_token(access_token),
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)
        logger.info("Created new user", user_id=str(user.id), username=user.username)
    else:
        user.username = github_user["login"]
        user.email = email
        user.avatar_url = github_user.get("avatar_url")
        user.access_token_encrypted = encrypt_token(access_token)
        logger.info("Updated existing user", user_id=str(user.id), username=user.username)

    # Create JWT
    jwt_token = create_access_token(
        user_id=user.id,
        github_id=user.github_id,
        username=user.username,
    )

    # Redirect with token
    separator = "&" if "?" in redirect_uri else "?"
    return RedirectResponse(url=f"{redirect_uri}{separator}token={jwt_token}")


@router.post("/token", response_model=TokenResponse)
async def exchange_token(
    code: str,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Exchange GitHub OAuth code for JWT token (for API clients)."""
    github = GitHubOAuth()

    # Exchange code for token
    token_data = await github.exchange_code(code)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to exchange code for token",
        )

    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No access token in response",
        )

    # Get user info
    github_user = await github.get_user(access_token)
    if github_user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to get user info",
        )

    email = await github.get_primary_email(access_token)

    # Find or create user
    result = await db.execute(select(User).where(User.github_id == github_user["id"]))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            github_id=github_user["id"],
            username=github_user["login"],
            email=email,
            avatar_url=github_user.get("avatar_url"),
            access_token_encrypted=encrypt_token(access_token),
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)

    jwt_token = create_access_token(
        user_id=user.id,
        github_id=user.github_id,
        username=user.username,
    )

    return TokenResponse(
        access_token=jwt_token,
        expires_in=settings.JWT_EXPIRATION_HOURS * 3600,
        user={
            "id": str(user.id),
            "github_id": user.github_id,
            "username": user.username,
            "email": user.email,
            "avatar_url": user.avatar_url,
        },
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[TokenData, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Get current authenticated user info."""
    result = await db.execute(select(User).where(User.id == current_user.user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse(
        id=user.id,
        github_id=user.github_id,
        username=user.username,
        email=user.email,
        avatar_url=user.avatar_url,
    )


@router.post("/logout")
async def logout() -> dict[str, str]:
    """Logout (client should discard token)."""
    return {"message": "Logged out successfully"}
