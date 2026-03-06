"""Unit tests for auth/dependencies.py — FastAPI auth guards."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from codeverify_api.auth.dependencies import (
    get_current_user,
    get_current_user_optional,
    require_admin,
)
from codeverify_api.auth.jwt import TokenData, create_access_token

TEST_SECRET = "test-secret-key-for-unit-tests"
TEST_ALGORITHM = "HS256"


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch):
    monkeypatch.setattr("codeverify_api.auth.jwt.settings.SECRET_KEY", TEST_SECRET)
    monkeypatch.setattr("codeverify_api.auth.jwt.settings.JWT_ALGORITHM", TEST_ALGORITHM)
    monkeypatch.setattr("codeverify_api.auth.jwt.settings.JWT_EXPIRATION_HOURS", 24)


def _make_credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _valid_token() -> str:
    return create_access_token(user_id=uuid4(), github_id=42, username="testuser")


class TestGetCurrentUserOptional:
    """Test the optional auth dependency."""

    async def test_returns_none_when_no_credentials(self):
        result = await get_current_user_optional(credentials=None)
        assert result is None

    async def test_returns_token_data_for_valid_credentials(self):
        creds = _make_credentials(_valid_token())
        result = await get_current_user_optional(credentials=creds)
        assert isinstance(result, TokenData)
        assert result.username == "testuser"

    async def test_returns_none_for_invalid_token(self):
        creds = _make_credentials("invalid-jwt")
        result = await get_current_user_optional(credentials=creds)
        assert result is None


class TestGetCurrentUser:
    """Test the required auth dependency (401 paths)."""

    async def test_raises_401_when_no_credentials(self):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=None)
        assert exc_info.value.status_code == 401
        assert "Not authenticated" in exc_info.value.detail

    async def test_raises_401_for_invalid_token(self):
        creds = _make_credentials("bad-token")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=creds)
        assert exc_info.value.status_code == 401
        assert "Invalid or expired token" in exc_info.value.detail

    async def test_raises_401_with_www_authenticate_header(self):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=None)
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}

    async def test_returns_token_data_for_valid_token(self):
        creds = _make_credentials(_valid_token())
        result = await get_current_user(credentials=creds)
        assert isinstance(result, TokenData)
        assert result.github_id == 42


class TestRequireAdmin:
    """Test the admin guard dependency (403 path)."""

    async def test_raises_403_for_regular_user(self):
        user = TokenData(user_id=uuid4(), github_id=1, username="regular")
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(user=user)
        assert exc_info.value.status_code == 403
        assert "Admin privileges required" in exc_info.value.detail

    async def test_passes_for_is_admin_flag(self):
        user = MagicMock()
        user.is_admin = True
        user.roles = []
        result = await require_admin(user=user)
        assert result is user

    async def test_passes_for_admin_role(self):
        user = MagicMock()
        user.is_admin = False
        user.roles = ["admin"]
        result = await require_admin(user=user)
        assert result is user

    async def test_passes_for_org_admin_role(self):
        user = MagicMock()
        user.is_admin = False
        user.roles = ["org_admin"]
        result = await require_admin(user=user)
        assert result is user
