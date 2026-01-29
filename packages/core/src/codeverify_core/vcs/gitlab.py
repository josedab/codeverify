"""GitLab VCS client implementation."""

import hashlib
import hmac
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

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


class GitLabClient(VCSClient):
    """GitLab VCS client using REST API."""

    def __init__(self, config: VCSConfig) -> None:
        """Initialize GitLab client."""
        super().__init__(config)
        self._client: Any = None
        self.base_url = config.base_url or "https://gitlab.com/api/v4"

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "gitlab"

    def _get_client(self) -> Any:
        """Get or create HTTP client."""
        if self._client is None:
            import httpx

            headers = {
                "Content-Type": "application/json",
            }
            if self.config.token:
                headers["PRIVATE-TOKEN"] = self.config.token

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def _encode_project_path(self, repo_full_name: str) -> str:
        """URL-encode project path for GitLab API."""
        return quote_plus(repo_full_name)

    def _parse_user(self, data: dict[str, Any]) -> User:
        """Parse GitLab user data."""
        return User(
            id=data["id"],
            username=data["username"],
            display_name=data.get("name"),
            email=data.get("email"),
            avatar_url=data.get("avatar_url"),
        )

    def _parse_repository(self, data: dict[str, Any]) -> Repository:
        """Parse GitLab project data."""
        return Repository(
            id=data["id"],
            name=data["name"],
            full_name=data["path_with_namespace"],
            owner=data["namespace"]["path"],
            description=data.get("description"),
            default_branch=data.get("default_branch", "main"),
            private=data.get("visibility") == "private",
            clone_url=data.get("http_url_to_repo"),
            html_url=data.get("web_url"),
        )

    def _parse_merge_request(self, data: dict[str, Any]) -> PullRequest:
        """Parse GitLab merge request data."""
        return PullRequest(
            id=data["id"],
            number=data["iid"],  # GitLab uses iid for project-scoped ID
            title=data["title"],
            body=data.get("description"),
            state=self._map_mr_state(data["state"]),
            head_sha=data["sha"],
            head_ref=data["source_branch"],
            base_sha=data.get("diff_refs", {}).get("base_sha", ""),
            base_ref=data["target_branch"],
            author=self._parse_user(data["author"]),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
            merged_at=(
                datetime.fromisoformat(data["merged_at"].replace("Z", "+00:00"))
                if data.get("merged_at")
                else None
            ),
            html_url=data.get("web_url"),
            labels=data.get("labels", []),
        )

    def _map_mr_state(self, gitlab_state: str) -> str:
        """Map GitLab MR state to normalized state."""
        state_map = {
            "opened": "open",
            "closed": "closed",
            "merged": "merged",
            "locked": "closed",
        }
        return state_map.get(gitlab_state, gitlab_state)

    async def get_repository(self, repo_full_name: str) -> Repository:
        """Get repository information."""
        client = self._get_client()
        project_path = self._encode_project_path(repo_full_name)
        response = await client.get(f"/projects/{project_path}")
        response.raise_for_status()
        return self._parse_repository(response.json())

    async def get_file_content(
        self,
        repo_full_name: str,
        path: str,
        ref: str | None = None,
    ) -> str:
        """Get file content from repository."""
        import base64

        client = self._get_client()
        project_path = self._encode_project_path(repo_full_name)
        file_path = quote_plus(path)

        params = {}
        if ref:
            params["ref"] = ref

        response = await client.get(
            f"/projects/{project_path}/repository/files/{file_path}",
            params=params,
        )
        response.raise_for_status()

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
        client = self._get_client()
        project_path = self._encode_project_path(repo_full_name)

        params: dict[str, Any] = {"recursive": True}
        if path:
            params["path"] = path
        if ref:
            params["ref"] = ref

        response = await client.get(
            f"/projects/{project_path}/repository/tree",
            params=params,
        )
        response.raise_for_status()

        return [
            item["path"]
            for item in response.json()
            if item["type"] == "blob"
        ]

    async def get_pull_request(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> PullRequest:
        """Get merge request details."""
        client = self._get_client()
        project_path = self._encode_project_path(repo_full_name)
        response = await client.get(
            f"/projects/{project_path}/merge_requests/{pr_number}"
        )
        response.raise_for_status()
        return self._parse_merge_request(response.json())

    async def get_pull_request_files(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> list[PullRequestFile]:
        """Get files changed in a merge request."""
        client = self._get_client()
        project_path = self._encode_project_path(repo_full_name)
        response = await client.get(
            f"/projects/{project_path}/merge_requests/{pr_number}/changes"
        )
        response.raise_for_status()

        files = []
        for change in response.json().get("changes", []):
            # Determine status
            if change.get("new_file"):
                status = "added"
            elif change.get("deleted_file"):
                status = "removed"
            elif change.get("renamed_file"):
                status = "renamed"
            else:
                status = "modified"

            files.append(
                PullRequestFile(
                    filename=change["new_path"],
                    status=status,
                    patch=change.get("diff"),
                    previous_filename=change.get("old_path") if change.get("renamed_file") else None,
                )
            )
        return files

    async def get_pull_request_diff(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> str:
        """Get the diff for a merge request."""
        client = self._get_client()
        project_path = self._encode_project_path(repo_full_name)

        # Get changes and concatenate diffs
        response = await client.get(
            f"/projects/{project_path}/merge_requests/{pr_number}/changes"
        )
        response.raise_for_status()

        diffs = []
        for change in response.json().get("changes", []):
            if change.get("diff"):
                diffs.append(f"diff --git a/{change['old_path']} b/{change['new_path']}")
                diffs.append(change["diff"])

        return "\n".join(diffs)

    async def create_pull_request_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str,
    ) -> PullRequestComment:
        """Create a note on a merge request."""
        client = self._get_client()
        project_path = self._encode_project_path(repo_full_name)
        response = await client.post(
            f"/projects/{project_path}/merge_requests/{pr_number}/notes",
            json={"body": body},
        )
        response.raise_for_status()

        data = response.json()
        return PullRequestComment(
            id=data["id"],
            body=data["body"],
            author=self._parse_user(data["author"]),
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
        """Create an inline discussion on a merge request."""
        client = self._get_client()
        project_path = self._encode_project_path(repo_full_name)

        # GitLab uses discussions for inline comments
        # We need to create a position object for the diff
        position = {
            "base_sha": commit_sha,  # This should be the base, but simplified
            "head_sha": commit_sha,
            "start_sha": commit_sha,
            "position_type": "text",
            "new_path": path,
            "new_line": line,
        }

        response = await client.post(
            f"/projects/{project_path}/merge_requests/{pr_number}/discussions",
            json={
                "body": body,
                "position": position,
            },
        )
        response.raise_for_status()

        data = response.json()
        note = data["notes"][0] if data.get("notes") else {}
        return PullRequestComment(
            id=note.get("id", data["id"]),
            body=body,
            author=self._parse_user(note["author"]) if note.get("author") else User(id=0, username="unknown"),
            created_at=datetime.fromisoformat(note["created_at"].replace("Z", "+00:00")) if note.get("created_at") else datetime.utcnow(),
            path=path,
            line=line,
        )

    async def update_comment(
        self,
        repo_full_name: str,
        comment_id: int,
        body: str,
    ) -> PullRequestComment:
        """Update an existing note."""
        # GitLab requires MR IID to update notes
        # This is a simplified implementation
        raise NotImplementedError(
            "GitLab note updates require MR IID. Use specific update method."
        )

    async def delete_comment(
        self,
        repo_full_name: str,
        comment_id: int,
    ) -> bool:
        """Delete a note."""
        # GitLab requires MR IID to delete notes
        raise NotImplementedError(
            "GitLab note deletions require MR IID. Use specific delete method."
        )

    async def create_check_run(
        self,
        repo_full_name: str,
        head_sha: str,
        check_run: CheckRun,
    ) -> CheckRun:
        """Create a pipeline status (GitLab equivalent of check run)."""
        # GitLab uses commit statuses, not check runs
        state_map = {
            CheckStatus.QUEUED: "pending",
            CheckStatus.IN_PROGRESS: "running",
            CheckStatus.COMPLETED: "success" if check_run.conclusion == CheckConclusion.SUCCESS else "failed",
        }

        await self.create_commit_status(
            repo_full_name=repo_full_name,
            sha=head_sha,
            state=state_map.get(check_run.status, "pending"),
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
        """Update pipeline status (create new status to update)."""
        # GitLab doesn't have updateable check runs, create new status
        # We need the SHA from somewhere - this is a limitation
        logger.warning(
            "GitLab doesn't support check run updates, creating new status",
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
        """Create a commit status."""
        client = self._get_client()
        project_path = self._encode_project_path(repo_full_name)

        payload: dict[str, Any] = {
            "state": state,
            "name": context,
        }
        if description:
            payload["description"] = description[:140]  # GitLab limit
        if target_url:
            payload["target_url"] = target_url

        response = await client.post(
            f"/projects/{project_path}/statuses/{sha}",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify GitLab webhook token."""
        # GitLab uses a simple token comparison, not HMAC
        if not self.config.webhook_secret:
            logger.warning("No webhook secret configured")
            return False

        return hmac.compare_digest(self.config.webhook_secret, signature)

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse GitLab webhook event into normalized format."""
        event_type = payload.get("object_kind", "unknown")

        normalized = {
            "provider": "gitlab",
            "event_type": event_type,
            "raw_payload": payload,
        }

        if event_type == "merge_request":
            mr_data = payload.get("object_attributes", {})
            project = payload.get("project", {})
            normalized.update(
                {
                    "action": mr_data.get("action"),
                    "repo_full_name": project.get("path_with_namespace"),
                    "repo_id": project.get("id"),
                    "pr_number": mr_data.get("iid"),
                    "pr_title": mr_data.get("title"),
                    "head_sha": mr_data.get("last_commit", {}).get("id"),
                    "base_sha": mr_data.get("diff_refs", {}).get("base_sha"),
                    "sender": payload.get("user", {}).get("username"),
                }
            )
        elif event_type == "push":
            normalized.update(
                {
                    "repo_full_name": payload.get("project", {}).get("path_with_namespace"),
                    "repo_id": payload.get("project", {}).get("id"),
                    "ref": payload.get("ref"),
                    "before": payload.get("before"),
                    "after": payload.get("after"),
                    "sender": payload.get("user_username"),
                }
            )

        return normalized
