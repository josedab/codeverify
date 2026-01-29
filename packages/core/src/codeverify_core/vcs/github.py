"""GitHub VCS client implementation."""

import hashlib
import hmac
import time
from datetime import datetime
from typing import Any

import structlog

from codeverify_core.vcs.base import (
    CheckConclusion,
    CheckRun,
    CheckRunAnnotation,
    CheckStatus,
    PullRequest,
    PullRequestComment,
    PullRequestFile,
    Repository,
    User,
    VCSClient,
    VCSConfig,
)

logger = structlog.get_logger()


class GitHubAppAuthenticator:
    """Handles GitHub App authentication with installation tokens."""

    def __init__(
        self,
        app_id: str,
        private_key: str,
        installation_id: int,
        base_url: str = "https://api.github.com",
    ) -> None:
        """Initialize the authenticator."""
        self.app_id = app_id
        self.private_key = private_key
        self.installation_id = installation_id
        self.base_url = base_url
        self._token: str | None = None
        self._token_expires_at: float = 0

    def _create_jwt(self) -> str:
        """Create a JWT for GitHub App authentication."""
        try:
            import jwt
        except ImportError:
            raise ImportError("PyJWT is required for GitHub App authentication")

        now = int(time.time())
        payload = {
            "iat": now - 60,  # Issued 60 seconds ago to handle clock drift
            "exp": now + (10 * 60),  # Expires in 10 minutes
            "iss": self.app_id,
        }
        return jwt.encode(payload, self.private_key, algorithm="RS256")

    async def get_installation_token(self) -> str:
        """Get or refresh installation access token."""
        # Return cached token if still valid (with 60s buffer)
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token

        import httpx

        app_jwt = self._create_jwt()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/app/installations/{self.installation_id}/access_tokens",
                headers={
                    "Authorization": f"Bearer {app_jwt}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            response.raise_for_status()
            data = response.json()

        self._token = data["token"]
        # Parse expiration or default to 1 hour
        expires_at = data.get("expires_at")
        if expires_at:
            self._token_expires_at = datetime.fromisoformat(
                expires_at.replace("Z", "+00:00")
            ).timestamp()
        else:
            self._token_expires_at = time.time() + 3600

        logger.debug(
            "Obtained GitHub App installation token",
            installation_id=self.installation_id,
        )
        return self._token


class GitHubClient(VCSClient):
    """GitHub VCS client using REST API.
    
    Supports both token-based and GitHub App authentication.
    """

    def __init__(
        self,
        config: VCSConfig,
        authenticator: GitHubAppAuthenticator | None = None,
    ) -> None:
        """Initialize GitHub client.
        
        Args:
            config: VCS configuration with token or app credentials
            authenticator: Optional pre-configured GitHub App authenticator
        """
        super().__init__(config)
        self._client: Any = None
        self.base_url = config.base_url or "https://api.github.com"
        self._authenticator = authenticator

        # Create authenticator from config if app credentials provided
        if not self._authenticator and config.app_id and config.private_key:
            # Installation ID must be set later or passed in config
            self._pending_app_auth = True
        else:
            self._pending_app_auth = False

    @classmethod
    def from_installation(
        cls,
        installation_id: int,
        app_id: str,
        private_key: str,
        base_url: str | None = None,
    ) -> "GitHubClient":
        """Create a client authenticated as a GitHub App installation.
        
        Args:
            installation_id: GitHub App installation ID
            app_id: GitHub App ID
            private_key: GitHub App private key (PEM format)
            base_url: Optional custom API base URL
            
        Returns:
            Configured GitHubClient instance
        """
        config = VCSConfig(
            provider="github",
            base_url=base_url,
            app_id=app_id,
            private_key=private_key,
        )
        authenticator = GitHubAppAuthenticator(
            app_id=app_id,
            private_key=private_key,
            installation_id=installation_id,
            base_url=base_url or "https://api.github.com",
        )
        return cls(config, authenticator)

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "github"

    async def _get_auth_token(self) -> str | None:
        """Get the authentication token (static or from App)."""
        if self._authenticator:
            return await self._authenticator.get_installation_token()
        return self.config.token

    def _get_client(self) -> Any:
        """Get or create HTTP client (without auth header - set per-request)."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    async def _get_headers(self) -> dict[str, str]:
        """Get request headers with current auth token."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = await self._get_auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Any:
        """Make an authenticated request."""
        client = self._get_client()
        headers = await self._get_headers()
        
        # Merge with any headers passed in kwargs
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        
        response = await client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response

    def _parse_user(self, data: dict[str, Any]) -> User:
        """Parse GitHub user data."""
        return User(
            id=data["id"],
            username=data["login"],
            display_name=data.get("name"),
            email=data.get("email"),
            avatar_url=data.get("avatar_url"),
        )

    def _parse_repository(self, data: dict[str, Any]) -> Repository:
        """Parse GitHub repository data."""
        return Repository(
            id=data["id"],
            name=data["name"],
            full_name=data["full_name"],
            owner=data["owner"]["login"],
            description=data.get("description"),
            default_branch=data.get("default_branch", "main"),
            private=data.get("private", False),
            clone_url=data.get("clone_url"),
            html_url=data.get("html_url"),
        )

    def _parse_pull_request(self, data: dict[str, Any]) -> PullRequest:
        """Parse GitHub pull request data."""
        return PullRequest(
            id=data["id"],
            number=data["number"],
            title=data["title"],
            body=data.get("body"),
            state=data["state"],
            head_sha=data["head"]["sha"],
            head_ref=data["head"]["ref"],
            base_sha=data["base"]["sha"],
            base_ref=data["base"]["ref"],
            author=self._parse_user(data["user"]),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
            merged_at=(
                datetime.fromisoformat(data["merged_at"].replace("Z", "+00:00"))
                if data.get("merged_at")
                else None
            ),
            html_url=data.get("html_url"),
            diff_url=data.get("diff_url"),
            labels=[label["name"] for label in data.get("labels", [])],
        )

    async def get_repository(self, repo_full_name: str) -> Repository:
        """Get repository information."""
        response = await self._request("GET", f"/repos/{repo_full_name}")
        return self._parse_repository(response.json())

    async def get_file_content(
        self,
        repo_full_name: str,
        path: str,
        ref: str | None = None,
    ) -> str:
        """Get file content from repository."""
        import base64

        url = f"/repos/{repo_full_name}/contents/{path}"
        params = {"ref": ref} if ref else {}

        response = await self._request("GET", url, params=params)

        data = response.json()
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8")
        return data.get("content", "")

    async def list_files(
        self,
        repo_full_name: str,
        path: str = "",
        ref: str | None = None,
    ) -> list[str]:
        """List files in a directory."""
        url = f"/repos/{repo_full_name}/contents/{path}"
        params = {"ref": ref} if ref else {}

        response = await self._request("GET", url, params=params)

        data = response.json()
        if isinstance(data, list):
            return [item["path"] for item in data if item["type"] == "file"]
        return []

    async def get_pull_request(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> PullRequest:
        """Get pull request details."""
        response = await self._request("GET", f"/repos/{repo_full_name}/pulls/{pr_number}")
        return self._parse_pull_request(response.json())

    async def get_pull_request_files(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> list[PullRequestFile]:
        """Get files changed in a pull request."""
        response = await self._request("GET", f"/repos/{repo_full_name}/pulls/{pr_number}/files")

        files = []
        for file_data in response.json():
            files.append(
                PullRequestFile(
                    filename=file_data["filename"],
                    status=file_data["status"],
                    additions=file_data.get("additions", 0),
                    deletions=file_data.get("deletions", 0),
                    changes=file_data.get("changes", 0),
                    patch=file_data.get("patch"),
                    previous_filename=file_data.get("previous_filename"),
                    blob_url=file_data.get("blob_url"),
                )
            )
        return files

    async def get_pull_request_diff(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> str:
        """Get the diff for a pull request."""
        response = await self._request(
            "GET",
            f"/repos/{repo_full_name}/pulls/{pr_number}",
            headers={"Accept": "application/vnd.github.diff"},
        )
        return response.text

    async def create_pull_request_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str,
    ) -> PullRequestComment:
        """Create a comment on a pull request."""
        response = await self._request(
            "POST",
            f"/repos/{repo_full_name}/issues/{pr_number}/comments",
            json={"body": body},
        )

        data = response.json()
        return PullRequestComment(
            id=data["id"],
            body=data["body"],
            author=self._parse_user(data["user"]),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
        )

    async def create_review_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str,
        commit_sha: str,
        path: str,
        line: int,
        side: str = "RIGHT",
    ) -> PullRequestComment:
        """Create an inline review comment."""
        response = await self._request(
            "POST",
            f"/repos/{repo_full_name}/pulls/{pr_number}/comments",
            json={
                "body": body,
                "commit_id": commit_sha,
                "path": path,
                "line": line,
                "side": side,
            },
        )

        data = response.json()
        return PullRequestComment(
            id=data["id"],
            body=data["body"],
            author=self._parse_user(data["user"]),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            path=data.get("path"),
            line=data.get("line"),
        )

    async def update_comment(
        self,
        repo_full_name: str,
        comment_id: int,
        body: str,
    ) -> PullRequestComment:
        """Update an existing comment."""
        response = await self._request(
            "PATCH",
            f"/repos/{repo_full_name}/issues/comments/{comment_id}",
            json={"body": body},
        )

        data = response.json()
        return PullRequestComment(
            id=data["id"],
            body=data["body"],
            author=self._parse_user(data["user"]),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
        )

    async def delete_comment(
        self,
        repo_full_name: str,
        comment_id: int,
    ) -> bool:
        """Delete a comment."""
        response = await self._request(
            "DELETE",
            f"/repos/{repo_full_name}/issues/comments/{comment_id}",
        )
        return response.status_code == 204

    async def create_check_run(
        self,
        repo_full_name: str,
        head_sha: str,
        check_run: CheckRun,
    ) -> CheckRun:
        """Create a check run."""
        payload: dict[str, Any] = {
            "name": check_run.name,
            "head_sha": head_sha,
            "status": check_run.status.value,
        }

        if check_run.external_id:
            payload["external_id"] = check_run.external_id
        if check_run.details_url:
            payload["details_url"] = check_run.details_id
        if check_run.started_at:
            payload["started_at"] = check_run.started_at.isoformat()

        if check_run.status == CheckStatus.COMPLETED:
            payload["conclusion"] = check_run.conclusion.value if check_run.conclusion else "neutral"
            if check_run.completed_at:
                payload["completed_at"] = check_run.completed_at.isoformat()

        if check_run.title or check_run.summary:
            payload["output"] = {}
            if check_run.title:
                payload["output"]["title"] = check_run.title
            if check_run.summary:
                payload["output"]["summary"] = check_run.summary
            if check_run.text:
                payload["output"]["text"] = check_run.text
            if check_run.annotations:
                payload["output"]["annotations"] = [
                    {
                        "path": a.path,
                        "start_line": a.start_line,
                        "end_line": a.end_line,
                        "annotation_level": a.annotation_level,
                        "message": a.message,
                        "title": a.title,
                        "raw_details": a.raw_details,
                    }
                    for a in check_run.annotations
                ]

        response = await self._request(
            "POST",
            f"/repos/{repo_full_name}/check-runs",
            json=payload,
        )

        data = response.json()
        check_run.id = data["id"]
        return check_run

    async def update_check_run(
        self,
        repo_full_name: str,
        check_run_id: int,
        check_run: CheckRun,
    ) -> CheckRun:
        """Update an existing check run."""
        payload: dict[str, Any] = {
            "status": check_run.status.value,
        }

        if check_run.status == CheckStatus.COMPLETED:
            payload["conclusion"] = check_run.conclusion.value if check_run.conclusion else "neutral"
            if check_run.completed_at:
                payload["completed_at"] = check_run.completed_at.isoformat()

        if check_run.title or check_run.summary:
            payload["output"] = {}
            if check_run.title:
                payload["output"]["title"] = check_run.title
            if check_run.summary:
                payload["output"]["summary"] = check_run.summary
            if check_run.text:
                payload["output"]["text"] = check_run.text
            if check_run.annotations:
                payload["output"]["annotations"] = [
                    {
                        "path": a.path,
                        "start_line": a.start_line,
                        "end_line": a.end_line,
                        "annotation_level": a.annotation_level,
                        "message": a.message,
                        "title": a.title,
                    }
                    for a in check_run.annotations
                ]

        response = await self._request(
            "PATCH",
            f"/repos/{repo_full_name}/check-runs/{check_run_id}",
            json=payload,
        )

        check_run.id = check_run_id
        return check_run

    async def create_commit_status(
        self,
        repo_full_name: str,
        sha: str,
        state: str,
        context: str,
        description: str | None = None,
        target_url: str | None = None,
    ) -> dict[str, Any]:
        """Create a commit status."""
        payload: dict[str, Any] = {
            "state": state,
            "context": context,
        }
        if description:
            payload["description"] = description
        if target_url:
            payload["target_url"] = target_url

        response = await self._request(
            "POST",
            f"/repos/{repo_full_name}/statuses/{sha}",
            json=payload,
        )
        return response.json()

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify GitHub webhook signature."""
        if not self.config.webhook_secret:
            logger.warning("No webhook secret configured")
            return False

        expected = "sha256=" + hmac.new(
            self.config.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse GitHub webhook event into normalized format."""
        event_type = headers.get("X-GitHub-Event", "unknown")

        normalized = {
            "provider": "github",
            "event_type": event_type,
            "delivery_id": headers.get("X-GitHub-Delivery"),
            "raw_payload": payload,
        }

        if event_type == "pull_request":
            pr_data = payload.get("pull_request", {})
            normalized.update(
                {
                    "action": payload.get("action"),
                    "repo_full_name": payload["repository"]["full_name"],
                    "repo_id": payload["repository"]["id"],
                    "pr_number": payload["number"],
                    "pr_title": pr_data.get("title"),
                    "head_sha": pr_data.get("head", {}).get("sha"),
                    "base_sha": pr_data.get("base", {}).get("sha"),
                    "sender": payload["sender"]["login"],
                }
            )
        elif event_type == "push":
            normalized.update(
                {
                    "repo_full_name": payload["repository"]["full_name"],
                    "repo_id": payload["repository"]["id"],
                    "ref": payload.get("ref"),
                    "before": payload.get("before"),
                    "after": payload.get("after"),
                    "sender": payload["sender"]["login"],
                }
            )

        return normalized
