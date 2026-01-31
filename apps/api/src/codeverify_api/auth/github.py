"""GitHub OAuth integration."""

from typing import Any
from urllib.parse import urlencode

import httpx
import structlog

from codeverify_api.config import settings

logger = structlog.get_logger()


class GitHubOAuth:
    """GitHub OAuth client."""

    AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
    TOKEN_URL = "https://github.com/login/oauth/access_token"
    USER_URL = "https://api.github.com/user"
    USER_EMAILS_URL = "https://api.github.com/user/emails"

    def __init__(self) -> None:
        """Initialize GitHub OAuth client."""
        self.client_id = settings.GITHUB_CLIENT_ID
        self.client_secret = settings.GITHUB_CLIENT_SECRET

    def get_authorize_url(self, state: str, redirect_uri: str) -> str:
        """Get GitHub OAuth authorization URL."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": "read:user user:email read:org",
            "state": state,
        }
        return f"{self.AUTHORIZE_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> dict[str, Any] | None:
        """Exchange authorization code for access token."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.TOKEN_URL,
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "code": code,
                    },
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    logger.error("GitHub OAuth error", error=data.get("error_description"))
                    return None

                return data
            except httpx.HTTPError as e:
                logger.error("GitHub OAuth exchange failed", error=str(e))
                return None

    async def get_user(self, access_token: str) -> dict[str, Any] | None:
        """Get GitHub user information."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.USER_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json",
                    },
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error("Failed to get GitHub user", error=str(e))
                return None

    async def get_user_emails(self, access_token: str) -> list[dict[str, Any]]:
        """Get GitHub user emails."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.USER_EMAILS_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json",
                    },
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error("Failed to get GitHub user emails", error=str(e))
                return []

    async def get_primary_email(self, access_token: str) -> str | None:
        """Get user's primary email."""
        emails = await self.get_user_emails(access_token)
        for email in emails:
            if email.get("primary") and email.get("verified"):
                return email.get("email")
        return None

    async def get_user_orgs(self, access_token: str) -> list[dict[str, Any]]:
        """Get user's organizations."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "https://api.github.com/user/orgs",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json",
                    },
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error("Failed to get GitHub user orgs", error=str(e))
                return []
