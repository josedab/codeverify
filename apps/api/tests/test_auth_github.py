"""Unit tests for auth/github.py — GitHub OAuth client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from codeverify_api.auth.github import GitHubOAuth


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch):
    monkeypatch.setattr("codeverify_api.auth.github.settings.GITHUB_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(
        "codeverify_api.auth.github.settings.GITHUB_CLIENT_SECRET", "test-client-secret"
    )


@pytest.fixture
def oauth():
    return GitHubOAuth()


class TestGetAuthorizeUrl:
    """Test get_authorize_url."""

    def test_builds_correct_url(self, oauth):
        url = oauth.get_authorize_url(state="abc123", redirect_uri="http://localhost/callback")
        assert "https://github.com/login/oauth/authorize?" in url
        assert "client_id=test-client-id" in url
        assert "state=abc123" in url
        assert "redirect_uri=http" in url
        assert "scope=read%3Auser+user%3Aemail+read%3Aorg" in url

    def test_includes_all_required_params(self, oauth):
        url = oauth.get_authorize_url(state="s", redirect_uri="http://example.com")
        for param in ["client_id", "redirect_uri", "scope", "state"]:
            assert param in url


class TestExchangeCode:
    """Test exchange_code — mocked HTTP."""

    async def test_returns_token_data_on_success(self, oauth):
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "gho_xxx", "token_type": "bearer"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("codeverify_api.auth.github.httpx.AsyncClient", return_value=mock_client):
            result = await oauth.exchange_code("test-code")

        assert result == {"access_token": "gho_xxx", "token_type": "bearer"}
        mock_client.post.assert_called_once()

    async def test_returns_none_on_github_error_response(self, oauth):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": "bad_verification_code",
            "error_description": "The code passed is incorrect",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("codeverify_api.auth.github.httpx.AsyncClient", return_value=mock_client):
            result = await oauth.exchange_code("bad-code")

        assert result is None

    async def test_returns_none_on_http_error(self, oauth):
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPError("Connection failed")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("codeverify_api.auth.github.httpx.AsyncClient", return_value=mock_client):
            result = await oauth.exchange_code("code")

        assert result is None


class TestGetUser:
    """Test get_user — mocked HTTP."""

    async def test_returns_user_data_on_success(self, oauth):
        user_data = {"id": 12345, "login": "octocat", "avatar_url": "https://example.com/a.png"}
        mock_response = MagicMock()
        mock_response.json.return_value = user_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("codeverify_api.auth.github.httpx.AsyncClient", return_value=mock_client):
            result = await oauth.get_user("gho_token")

        assert result == user_data

    async def test_returns_none_on_http_error(self, oauth):
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("401 Unauthorized")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("codeverify_api.auth.github.httpx.AsyncClient", return_value=mock_client):
            result = await oauth.get_user("bad-token")

        assert result is None


class TestGetUserEmails:
    """Test get_user_emails — mocked HTTP."""

    async def test_returns_emails_on_success(self, oauth):
        emails = [
            {"email": "user@example.com", "primary": True, "verified": True},
            {"email": "alt@example.com", "primary": False, "verified": True},
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = emails
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("codeverify_api.auth.github.httpx.AsyncClient", return_value=mock_client):
            result = await oauth.get_user_emails("token")

        assert len(result) == 2
        assert result[0]["email"] == "user@example.com"

    async def test_returns_empty_list_on_http_error(self, oauth):
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("error")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("codeverify_api.auth.github.httpx.AsyncClient", return_value=mock_client):
            result = await oauth.get_user_emails("token")

        assert result == []


class TestGetPrimaryEmail:
    """Test get_primary_email."""

    async def test_returns_primary_verified_email(self, oauth):
        emails = [
            {"email": "alt@example.com", "primary": False, "verified": True},
            {"email": "primary@example.com", "primary": True, "verified": True},
        ]
        with patch.object(oauth, "get_user_emails", new_callable=AsyncMock, return_value=emails):
            result = await oauth.get_primary_email("token")
        assert result == "primary@example.com"

    async def test_returns_none_when_no_primary(self, oauth):
        emails = [{"email": "alt@example.com", "primary": False, "verified": True}]
        with patch.object(oauth, "get_user_emails", new_callable=AsyncMock, return_value=emails):
            result = await oauth.get_primary_email("token")
        assert result is None

    async def test_returns_none_when_primary_unverified(self, oauth):
        emails = [{"email": "user@example.com", "primary": True, "verified": False}]
        with patch.object(oauth, "get_user_emails", new_callable=AsyncMock, return_value=emails):
            result = await oauth.get_primary_email("token")
        assert result is None

    async def test_returns_none_when_no_emails(self, oauth):
        with patch.object(oauth, "get_user_emails", new_callable=AsyncMock, return_value=[]):
            result = await oauth.get_primary_email("token")
        assert result is None


class TestGetUserOrgs:
    """Test get_user_orgs."""

    async def test_returns_orgs_on_success(self, oauth):
        orgs = [{"login": "myorg", "id": 99}]
        mock_response = MagicMock()
        mock_response.json.return_value = orgs
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("codeverify_api.auth.github.httpx.AsyncClient", return_value=mock_client):
            result = await oauth.get_user_orgs("token")

        assert result == orgs

    async def test_returns_empty_list_on_http_error(self, oauth):
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("error")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("codeverify_api.auth.github.httpx.AsyncClient", return_value=mock_client):
            result = await oauth.get_user_orgs("token")

        assert result == []
