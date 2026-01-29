"""Bitbucket VCS client implementation."""

import hashlib
import hmac
from datetime import datetime
from typing import Any

import structlog

from codeverify_core.vcs.base import (
    CheckConclusion,
    CheckRun,
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


class BitbucketClient(VCSClient):
    """Bitbucket Cloud VCS client using REST API 2.0."""

    def __init__(self, config: VCSConfig) -> None:
        """Initialize Bitbucket client."""
        super().__init__(config)
        self._client: Any = None
        self.base_url = config.base_url or "https://api.bitbucket.org/2.0"

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "bitbucket"

    def _get_client(self) -> Any:
        """Get or create HTTP client."""
        if self._client is None:
            import httpx

            headers = {
                "Content-Type": "application/json",
            }

            # Bitbucket uses OAuth2 Bearer token or App passwords
            auth = None
            if self.config.token:
                if ":" in self.config.token:
                    # App password format: username:app_password
                    username, password = self.config.token.split(":", 1)
                    auth = httpx.BasicAuth(username, password)
                else:
                    # OAuth2 Bearer token
                    headers["Authorization"] = f"Bearer {self.config.token}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                auth=auth,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def _parse_user(self, data: dict[str, Any]) -> User:
        """Parse Bitbucket user/account data."""
        return User(
            id=data.get("uuid", data.get("account_id", "")),
            username=data.get("nickname", data.get("username", "")),
            display_name=data.get("display_name"),
            avatar_url=data.get("links", {}).get("avatar", {}).get("href"),
        )

    def _parse_repository(self, data: dict[str, Any]) -> Repository:
        """Parse Bitbucket repository data."""
        return Repository(
            id=data.get("uuid", ""),
            name=data["name"],
            full_name=data["full_name"],
            owner=data["owner"]["nickname"] if data.get("owner") else data["full_name"].split("/")[0],
            description=data.get("description"),
            default_branch=data.get("mainbranch", {}).get("name", "main"),
            private=data.get("is_private", False),
            clone_url=next(
                (l["href"] for l in data.get("links", {}).get("clone", []) if l["name"] == "https"),
                None,
            ),
            html_url=data.get("links", {}).get("html", {}).get("href"),
        )

    def _parse_pull_request(self, data: dict[str, Any]) -> PullRequest:
        """Parse Bitbucket pull request data."""
        return PullRequest(
            id=data["id"],
            number=data["id"],  # Bitbucket uses id as the number
            title=data["title"],
            body=data.get("description"),
            state=self._map_pr_state(data["state"]),
            head_sha=data["source"]["commit"]["hash"],
            head_ref=data["source"]["branch"]["name"],
            base_sha=data["destination"]["commit"]["hash"],
            base_ref=data["destination"]["branch"]["name"],
            author=self._parse_user(data["author"]),
            created_at=datetime.fromisoformat(data["created_on"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_on"].replace("Z", "+00:00")),
            merged_at=None,  # Bitbucket doesn't include this in PR data
            html_url=data.get("links", {}).get("html", {}).get("href"),
        )

    def _map_pr_state(self, bb_state: str) -> str:
        """Map Bitbucket PR state to normalized state."""
        state_map = {
            "OPEN": "open",
            "MERGED": "merged",
            "DECLINED": "closed",
            "SUPERSEDED": "closed",
        }
        return state_map.get(bb_state, bb_state.lower())

    async def get_repository(self, repo_full_name: str) -> Repository:
        """Get repository information."""
        client = self._get_client()
        response = await client.get(f"/repositories/{repo_full_name}")
        response.raise_for_status()
        return self._parse_repository(response.json())

    async def get_file_content(
        self,
        repo_full_name: str,
        path: str,
        ref: str | None = None,
    ) -> str:
        """Get file content from repository."""
        client = self._get_client()
        commit = ref or "HEAD"
        response = await client.get(
            f"/repositories/{repo_full_name}/src/{commit}/{path}"
        )
        response.raise_for_status()
        return response.text

    async def list_files(
        self,
        repo_full_name: str,
        path: str = "",
        ref: str | None = None,
    ) -> list[str]:
        """List files in a directory."""
        client = self._get_client()
        commit = ref or "HEAD"

        url = f"/repositories/{repo_full_name}/src/{commit}/{path}"
        response = await client.get(url)
        response.raise_for_status()

        files = []
        data = response.json()
        for item in data.get("values", []):
            if item["type"] == "commit_file":
                files.append(item["path"])

        return files

    async def get_pull_request(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> PullRequest:
        """Get pull request details."""
        client = self._get_client()
        response = await client.get(
            f"/repositories/{repo_full_name}/pullrequests/{pr_number}"
        )
        response.raise_for_status()
        return self._parse_pull_request(response.json())

    async def get_pull_request_files(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> list[PullRequestFile]:
        """Get files changed in a pull request."""
        client = self._get_client()
        response = await client.get(
            f"/repositories/{repo_full_name}/pullrequests/{pr_number}/diffstat"
        )
        response.raise_for_status()

        files = []
        for item in response.json().get("values", []):
            status_map = {
                "added": "added",
                "removed": "removed",
                "modified": "modified",
                "renamed": "renamed",
            }
            status = status_map.get(item.get("status", "modified"), "modified")

            files.append(
                PullRequestFile(
                    filename=item.get("new", {}).get("path", item.get("old", {}).get("path", "")),
                    status=status,
                    additions=item.get("lines_added", 0),
                    deletions=item.get("lines_removed", 0),
                    previous_filename=item.get("old", {}).get("path") if status == "renamed" else None,
                )
            )
        return files

    async def get_pull_request_diff(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> str:
        """Get the diff for a pull request."""
        client = self._get_client()
        response = await client.get(
            f"/repositories/{repo_full_name}/pullrequests/{pr_number}/diff"
        )
        response.raise_for_status()
        return response.text

    async def create_pull_request_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str,
    ) -> PullRequestComment:
        """Create a comment on a pull request."""
        client = self._get_client()
        response = await client.post(
            f"/repositories/{repo_full_name}/pullrequests/{pr_number}/comments",
            json={"content": {"raw": body}},
        )
        response.raise_for_status()

        data = response.json()
        return PullRequestComment(
            id=data["id"],
            body=data["content"]["raw"],
            author=self._parse_user(data["user"]),
            created_at=datetime.fromisoformat(data["created_on"].replace("Z", "+00:00")),
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
        """Create an inline comment on a pull request."""
        client = self._get_client()

        payload = {
            "content": {"raw": body},
            "inline": {
                "path": path,
                "to": line,  # Bitbucket uses 'to' for the line number
            },
        }

        response = await client.post(
            f"/repositories/{repo_full_name}/pullrequests/{pr_number}/comments",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        return PullRequestComment(
            id=data["id"],
            body=data["content"]["raw"],
            author=self._parse_user(data["user"]),
            created_at=datetime.fromisoformat(data["created_on"].replace("Z", "+00:00")),
            path=path,
            line=line,
        )

    async def update_comment(
        self,
        repo_full_name: str,
        comment_id: int,
        body: str,
    ) -> PullRequestComment:
        """Update an existing comment."""
        # Bitbucket requires PR ID for comment updates
        raise NotImplementedError(
            "Bitbucket comment updates require PR ID. Use specific update method."
        )

    async def delete_comment(
        self,
        repo_full_name: str,
        comment_id: int,
    ) -> bool:
        """Delete a comment."""
        # Bitbucket requires PR ID for comment deletions
        raise NotImplementedError(
            "Bitbucket comment deletions require PR ID. Use specific delete method."
        )

    async def create_check_run(
        self,
        repo_full_name: str,
        head_sha: str,
        check_run: CheckRun,
    ) -> CheckRun:
        """Create a build status (Bitbucket equivalent of check run)."""
        state_map = {
            CheckStatus.QUEUED: "INPROGRESS",
            CheckStatus.IN_PROGRESS: "INPROGRESS",
            CheckStatus.COMPLETED: "SUCCESSFUL" if check_run.conclusion == CheckConclusion.SUCCESS else "FAILED",
        }

        await self.create_commit_status(
            repo_full_name=repo_full_name,
            sha=head_sha,
            state=state_map.get(check_run.status, "INPROGRESS"),
            context=check_run.name,
            description=check_run.summary or check_run.title,
            target_url=check_run.details_url,
        )

        return check_run

    async def update_check_run(
        self,
        repo_full_name: str,
        check_run_id: int,
        check_run: CheckRun,
    ) -> CheckRun:
        """Update build status."""
        logger.warning(
            "Bitbucket doesn't support check run updates, create new status",
            check_run_id=check_run_id,
        )
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
        """Create a build status."""
        client = self._get_client()

        # Bitbucket requires a unique key
        key = f"codeverify-{context}".replace(" ", "-")[:40]

        payload: dict[str, Any] = {
            "state": state,
            "key": key,
            "name": context,
        }
        if description:
            payload["description"] = description[:255]  # Bitbucket limit
        if target_url:
            payload["url"] = target_url

        response = await client.post(
            f"/repositories/{repo_full_name}/commit/{sha}/statuses/build",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify Bitbucket webhook signature."""
        if not self.config.webhook_secret:
            logger.warning("No webhook secret configured")
            return False

        # Bitbucket uses HMAC-SHA256
        expected = hmac.new(
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
        """Parse Bitbucket webhook event into normalized format."""
        event_type = headers.get("X-Event-Key", "unknown")

        normalized = {
            "provider": "bitbucket",
            "event_type": event_type,
            "hook_uuid": headers.get("X-Hook-UUID"),
            "raw_payload": payload,
        }

        if event_type.startswith("pullrequest:"):
            pr_data = payload.get("pullrequest", {})
            repo = payload.get("repository", {})
            normalized.update(
                {
                    "action": event_type.split(":")[1],
                    "repo_full_name": repo.get("full_name"),
                    "repo_id": repo.get("uuid"),
                    "pr_number": pr_data.get("id"),
                    "pr_title": pr_data.get("title"),
                    "head_sha": pr_data.get("source", {}).get("commit", {}).get("hash"),
                    "base_sha": pr_data.get("destination", {}).get("commit", {}).get("hash"),
                    "sender": payload.get("actor", {}).get("nickname"),
                }
            )
        elif event_type.startswith("repo:push"):
            repo = payload.get("repository", {})
            push = payload.get("push", {})
            changes = push.get("changes", [{}])[0] if push.get("changes") else {}
            normalized.update(
                {
                    "repo_full_name": repo.get("full_name"),
                    "repo_id": repo.get("uuid"),
                    "ref": changes.get("new", {}).get("name"),
                    "before": changes.get("old", {}).get("target", {}).get("hash"),
                    "after": changes.get("new", {}).get("target", {}).get("hash"),
                    "sender": payload.get("actor", {}).get("nickname"),
                }
            )

        return normalized
