"""Integration tests for auth router — /auth/login, /auth/callback, /auth/me."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from codeverify_api.auth.jwt import create_access_token

TEST_SECRET = "test-secret-key-for-integration"
TEST_ALGORITHM = "HS256"


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch):
    monkeypatch.setattr("codeverify_api.auth.jwt.settings.SECRET_KEY", TEST_SECRET)
    monkeypatch.setattr("codeverify_api.auth.jwt.settings.JWT_ALGORITHM", TEST_ALGORITHM)
    monkeypatch.setattr("codeverify_api.auth.jwt.settings.JWT_EXPIRATION_HOURS", 24)


def _create_test_app():
    """Create a minimal FastAPI app with the auth router for testing."""
    from codeverify_api.routers.auth import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/auth")
    return app


@pytest.fixture
def app():
    return _create_test_app()


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestLoginEndpoint:
    """Test GET /api/v1/auth/login."""

    def test_login_redirects_to_github(self, client, monkeypatch):
        monkeypatch.setattr(
            "codeverify_api.routers.auth.settings.CORS_ORIGINS",
            ["http://localhost:3000"],
        )
        monkeypatch.setattr("codeverify_api.routers.auth.settings.GITHUB_CLIENT_ID", "test-client")
        monkeypatch.setattr(
            "codeverify_api.routers.auth.settings.GITHUB_CLIENT_SECRET", "test-secret"
        )
        monkeypatch.setattr("codeverify_api.routers.auth.settings.ENVIRONMENT", "development")
        monkeypatch.setattr("codeverify_api.routers.auth.settings.API_HOST", "http://localhost")
        monkeypatch.setattr("codeverify_api.routers.auth.settings.API_PORT", 8000)

        with patch("codeverify_api.routers.auth._store_oauth_state", new_callable=AsyncMock):
            response = client.get(
                "/api/v1/auth/login",
                params={"redirect_uri": "http://localhost:3000/callback"},
                follow_redirects=False,
            )

        assert response.status_code == 307
        assert "github.com/login/oauth/authorize" in response.headers["location"]

    def test_login_rejects_invalid_redirect_uri(self, client, monkeypatch):
        monkeypatch.setattr(
            "codeverify_api.routers.auth.settings.CORS_ORIGINS",
            ["http://localhost:3000"],
        )

        response = client.get(
            "/api/v1/auth/login",
            params={"redirect_uri": "https://evil.com/steal"},
        )

        assert response.status_code == 400
        assert "not in the list of allowed origins" in response.json()["detail"]

    def test_login_requires_redirect_uri(self, client):
        response = client.get("/api/v1/auth/login")
        assert response.status_code == 422


class TestCallbackEndpoint:
    """Test GET /api/v1/auth/callback."""

    def _mock_db(self, client):
        """Override db dependency for callback tests."""
        from codeverify_api.db import get_db

        mock_session = AsyncMock()

        async def mock_get_db():
            yield mock_session

        client.app.dependency_overrides[get_db] = mock_get_db
        return mock_session

    def test_callback_rejects_invalid_state(self, client):
        self._mock_db(client)
        with patch(
            "codeverify_api.routers.auth._pop_oauth_state",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.get(
                "/api/v1/auth/callback",
                params={"code": "test-code", "state": "invalid-state"},
            )

        assert response.status_code == 400
        assert "Invalid state" in response.json()["detail"]
        client.app.dependency_overrides.clear()

    def test_callback_rejects_failed_code_exchange(self, client, monkeypatch):
        self._mock_db(client)
        monkeypatch.setattr("codeverify_api.routers.auth.settings.GITHUB_CLIENT_ID", "test")
        monkeypatch.setattr("codeverify_api.routers.auth.settings.GITHUB_CLIENT_SECRET", "test")

        with (
            patch(
                "codeverify_api.routers.auth._pop_oauth_state",
                new_callable=AsyncMock,
                return_value="http://localhost:3000",
            ),
            patch(
                "codeverify_api.routers.auth.GitHubOAuth.exchange_code",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            response = client.get(
                "/api/v1/auth/callback",
                params={"code": "bad-code", "state": "valid-state"},
            )

        assert response.status_code == 400
        assert "Failed to exchange" in response.json()["detail"]
        client.app.dependency_overrides.clear()

    def test_callback_requires_code_and_state(self, client):
        self._mock_db(client)
        response = client.get("/api/v1/auth/callback")
        assert response.status_code == 422
        client.app.dependency_overrides.clear()


class TestMeEndpoint:
    """Test GET /api/v1/auth/me."""

    def test_me_returns_401_without_token(self, client):
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401

    def test_me_returns_401_with_invalid_token(self, client):
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid-jwt-token"},
        )
        assert response.status_code == 401

    def test_me_returns_user_with_valid_token(self, client):
        user_id = uuid4()
        token = create_access_token(user_id=user_id, github_id=42, username="testuser")

        mock_user = MagicMock()
        mock_user.id = user_id
        mock_user.github_id = 42
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.avatar_url = "https://example.com/avatar.png"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        async def mock_get_db():
            yield mock_session

        from codeverify_api.db import get_db

        client.app.dependency_overrides[get_db] = mock_get_db

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["github_id"] == 42

        # Clean up
        client.app.dependency_overrides.clear()

    def test_me_returns_404_for_deleted_user(self, client):
        user_id = uuid4()
        token = create_access_token(user_id=user_id, github_id=99, username="ghost")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        async def mock_get_db():
            yield mock_session

        from codeverify_api.db import get_db

        client.app.dependency_overrides[get_db] = mock_get_db

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 404

        client.app.dependency_overrides.clear()


class TestLogoutEndpoint:
    """Test POST /api/v1/auth/logout."""

    def test_logout_returns_success(self, client):
        response = client.post("/api/v1/auth/logout")
        assert response.status_code == 200
        assert response.json()["message"] == "Logged out successfully"
