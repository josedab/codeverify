"""Unit tests for auth/jwt.py — token creation and decoding."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from jose import jwt

from codeverify_api.auth.jwt import (
    TokenData,
    TokenPayload,
    create_access_token,
    decode_access_token,
    get_current_user,
)

TEST_SECRET = "test-secret-key-for-unit-tests"
TEST_ALGORITHM = "HS256"


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch):
    """Provide deterministic settings for all JWT tests."""
    monkeypatch.setattr("codeverify_api.auth.jwt.settings.SECRET_KEY", TEST_SECRET)
    monkeypatch.setattr("codeverify_api.auth.jwt.settings.JWT_ALGORITHM", TEST_ALGORITHM)
    monkeypatch.setattr("codeverify_api.auth.jwt.settings.JWT_EXPIRATION_HOURS", 24)


class TestCreateAccessToken:
    """Test create_access_token."""

    def test_creates_valid_jwt(self):
        user_id = uuid4()
        token = create_access_token(user_id=user_id, github_id=12345, username="testuser")
        assert isinstance(token, str)
        # Decode raw to inspect payload
        payload = jwt.decode(token, TEST_SECRET, algorithms=[TEST_ALGORITHM])
        assert payload["sub"] == str(user_id)
        assert payload["github_id"] == 12345
        assert payload["username"] == "testuser"
        assert "exp" in payload
        assert "iat" in payload

    def test_default_expiration(self):
        token = create_access_token(user_id=uuid4(), github_id=1, username="u")
        payload = jwt.decode(token, TEST_SECRET, algorithms=[TEST_ALGORITHM])
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        delta = exp - iat
        # Default is 24 hours; allow small clock skew
        assert timedelta(hours=23, minutes=59) <= delta <= timedelta(hours=24, minutes=1)

    def test_custom_expiration(self):
        token = create_access_token(
            user_id=uuid4(),
            github_id=1,
            username="u",
            expires_delta=timedelta(minutes=30),
        )
        payload = jwt.decode(token, TEST_SECRET, algorithms=[TEST_ALGORITHM])
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        delta = exp - iat
        assert timedelta(minutes=29) <= delta <= timedelta(minutes=31)


class TestDecodeAccessToken:
    """Test decode_access_token."""

    def test_decodes_valid_token(self):
        user_id = uuid4()
        token = create_access_token(user_id=user_id, github_id=42, username="alice")
        result = decode_access_token(token)
        assert isinstance(result, TokenData)
        assert result.user_id == user_id
        assert result.github_id == 42
        assert result.username == "alice"

    def test_returns_none_for_expired_token(self):
        token = create_access_token(
            user_id=uuid4(),
            github_id=1,
            username="u",
            expires_delta=timedelta(seconds=-1),
        )
        assert decode_access_token(token) is None

    def test_returns_none_for_invalid_token(self):
        assert decode_access_token("not-a-valid-jwt") is None

    def test_returns_none_for_wrong_secret(self):
        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "github_id": 1,
            "username": "u",
        }
        token = jwt.encode(payload, "wrong-secret", algorithm=TEST_ALGORITHM)
        assert decode_access_token(token) is None

    def test_raises_on_missing_fields(self):
        """Source code doesn't catch KeyError — missing fields raise."""
        payload = {
            "sub": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            # Missing github_id and username
        }
        token = jwt.encode(payload, TEST_SECRET, algorithm=TEST_ALGORITHM)
        with pytest.raises(KeyError):
            decode_access_token(token)

    def test_raises_on_invalid_uuid_sub(self):
        """Source code doesn't catch ValueError — bad UUID raises."""
        payload = {
            "sub": "not-a-uuid",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "github_id": 1,
            "username": "u",
        }
        token = jwt.encode(payload, TEST_SECRET, algorithm=TEST_ALGORITHM)
        with pytest.raises(ValueError):
            decode_access_token(token)


class TestGetCurrentUser:
    """Test the async get_current_user helper."""

    async def test_returns_token_data_for_valid_token(self):
        user_id = uuid4()
        token = create_access_token(user_id=user_id, github_id=7, username="bob")
        result = await get_current_user(token)
        assert result is not None
        assert result.user_id == user_id

    async def test_returns_none_for_invalid_token(self):
        result = await get_current_user("garbage")
        assert result is None


class TestTokenPayloadModel:
    """Test TokenPayload pydantic model."""

    def test_valid_payload(self):
        now = datetime.now(timezone.utc)
        payload = TokenPayload(
            sub="abc123",
            exp=now + timedelta(hours=1),
            iat=now,
            github_id=1,
            username="u",
        )
        assert payload.sub == "abc123"

    def test_rejects_missing_fields(self):
        with pytest.raises(Exception):
            TokenPayload(sub="abc123")  # type: ignore[call-arg]
