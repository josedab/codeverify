"""GitHub API client for posting results."""

import time
from typing import Any

import httpx
import jwt
import structlog

from codeverify_api.config import settings

logger = structlog.get_logger()


class GitHubClient:
    """GitHub API client for CodeVerify operations."""

    BASE_URL = "https://api.github.com"

    def __init__(self, installation_id: int | None = None) -> None:
        """Initialize GitHub client."""
        self.installation_id = installation_id
        self._installation_token: str | None = None
        self._token_expires_at: int = 0

    def _create_app_jwt(self) -> str:
        """Create JWT for GitHub App authentication."""
        now = int(time.time())
        payload = {
            "iat": now - 60,  # Issued 60 seconds ago
            "exp": now + (10 * 60),  # Expires in 10 minutes
            "iss": settings.GITHUB_APP_ID,
        }

        # Read private key
        try:
            with open(settings.GITHUB_APP_PRIVATE_KEY_PATH, "r") as f:
                private_key = f.read()
        except FileNotFoundError:
            # Try using the key directly (for env var storage)
            private_key = settings.GITHUB_APP_PRIVATE_KEY_PATH

        return jwt.encode(payload, private_key, algorithm="RS256")

    async def _get_installation_token(self) -> str:
        """Get installation access token."""
        if self._installation_token and time.time() < self._token_expires_at - 60:
            return self._installation_token

        if not self.installation_id:
            raise ValueError("Installation ID is required")

        app_jwt = self._create_app_jwt()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/app/installations/{self.installation_id}/access_tokens",
                headers={
                    "Authorization": f"Bearer {app_jwt}",
                    "Accept": "application/vnd.github+json",
                },
            )
            response.raise_for_status()
            data = response.json()

        self._installation_token = data["token"]
        # Token expires in 1 hour, we'll refresh a bit earlier
        self._token_expires_at = int(time.time()) + 3600

        return self._installation_token

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make authenticated request to GitHub API."""
        token = await self._get_installation_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                f"{self.BASE_URL}{endpoint}",
                headers=headers,
                **kwargs,
            )
            response.raise_for_status()
            return response.json() if response.content else {}

    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Get PR diff."""
        token = await self._get_installation_token()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{pr_number}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github.diff",
                },
            )
            response.raise_for_status()
            return response.text

    async def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """Get list of files changed in PR."""
        return await self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/files",
        )

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str:
        """Get file content at a specific ref."""
        import base64

        data = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{path}",
            params={"ref": ref},
        )

        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8")
        return data.get("content", "")

    async def create_check_run(
        self,
        owner: str,
        repo: str,
        name: str,
        head_sha: str,
        status: str = "queued",
        details_url: str | None = None,
    ) -> dict[str, Any]:
        """Create a check run."""
        data: dict[str, Any] = {
            "name": name,
            "head_sha": head_sha,
            "status": status,
        }

        if details_url:
            data["details_url"] = details_url

        return await self._request(
            "POST",
            f"/repos/{owner}/{repo}/check-runs",
            json=data,
        )

    async def update_check_run(
        self,
        owner: str,
        repo: str,
        check_run_id: int,
        status: str | None = None,
        conclusion: str | None = None,
        title: str | None = None,
        summary: str | None = None,
        text: str | None = None,
        annotations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Update a check run."""
        data: dict[str, Any] = {}

        if status:
            data["status"] = status
        if conclusion:
            data["conclusion"] = conclusion

        output: dict[str, Any] = {}
        if title:
            output["title"] = title
        if summary:
            output["summary"] = summary
        if text:
            output["text"] = text
        if annotations:
            output["annotations"] = annotations[:50]  # Max 50 per request

        if output:
            data["output"] = output

        return await self._request(
            "PATCH",
            f"/repos/{owner}/{repo}/check-runs/{check_run_id}",
            json=data,
        )

    async def create_pr_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
    ) -> dict[str, Any]:
        """Create a comment on a PR."""
        return await self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )

    async def create_pr_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        body: str,
        event: str = "COMMENT",  # APPROVE, REQUEST_CHANGES, COMMENT
        comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a PR review with optional line comments."""
        data: dict[str, Any] = {
            "commit_id": commit_sha,
            "body": body,
            "event": event,
        }

        if comments:
            data["comments"] = comments

        return await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
            json=data,
        )

    async def create_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        path: str,
        line: int,
        body: str,
        side: str = "RIGHT",
    ) -> dict[str, Any]:
        """Create a single review comment on a specific line."""
        return await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            json={
                "commit_id": commit_sha,
                "path": path,
                "line": line,
                "side": side,
                "body": body,
            },
        )

    async def create_suggested_change(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        path: str,
        line: int,
        suggestion: str,
        description: str = "",
        start_line: int | None = None,
    ) -> dict[str, Any]:
        """Create a suggested change comment (one-click apply).
        
        Uses GitHub's suggestion syntax which renders as a one-click "Apply" button.
        For multi-line suggestions, provide start_line.
        """
        body = f"{description}\n\n```suggestion\n{suggestion}\n```"

        data: dict[str, Any] = {
            "commit_id": commit_sha,
            "path": path,
            "line": line,
            "side": "RIGHT",
            "body": body,
        }
        
        # Multi-line suggestion
        if start_line and start_line != line:
            data["start_line"] = start_line
            data["start_side"] = "RIGHT"

        return await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            json=data,
        )

    async def create_review_with_suggestions(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        findings: list[dict[str, Any]],
        summary_body: str,
        event: str = "COMMENT",
    ) -> dict[str, Any]:
        """Create a PR review with multiple inline suggestions.
        
        This is the one-click fix feature - creates suggestions that users
        can apply directly from the GitHub UI.
        """
        comments = []
        
        for finding in findings:
            if not finding.get("fix_suggestion"):
                continue
                
            file_path = finding.get("file_path", "")
            line_start = finding.get("line_start", 1)
            line_end = finding.get("line_end") or line_start
            fix = finding.get("fix_suggestion", "")
            title = finding.get("title", "Issue")
            description = finding.get("description", "")
            severity = finding.get("severity", "medium")
            confidence = finding.get("confidence", 0)
            
            # Build comment body with suggestion
            severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
            emoji = severity_emoji.get(severity, "⚪")
            
            body_parts = [
                f"{emoji} **{title}**",
                "",
                description,
                "",
                f"*Confidence: {int(confidence * 100)}%*",
                "",
                "**Suggested fix** (click 'Apply suggestion' to apply):",
                "",
                "```suggestion",
                fix,
                "```",
            ]
            
            comment: dict[str, Any] = {
                "path": file_path,
                "line": line_end,
                "body": "\n".join(body_parts),
            }
            
            # Multi-line change
            if line_start != line_end:
                comment["start_line"] = line_start
            
            comments.append(comment)
        
        # Create the review with all comments
        data: dict[str, Any] = {
            "commit_id": commit_sha,
            "body": summary_body,
            "event": event,
        }
        
        if comments:
            data["comments"] = comments[:50]  # GitHub limit
        
        return await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
            json=data,
        )

    async def create_inline_annotations(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        findings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create individual inline comments for findings without suggestions."""
        results = []
        
        for finding in findings:
            # Skip if already has fix suggestion (handled by review)
            if finding.get("fix_suggestion"):
                continue
                
            file_path = finding.get("file_path", "")
            line = finding.get("line_start", 1)
            title = finding.get("title", "Issue")
            description = finding.get("description", "")
            severity = finding.get("severity", "medium")
            verification_type = finding.get("verification_type", "ai")
            
            severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
            emoji = severity_emoji.get(severity, "⚪")
            
            verification_badge = "🔬 Formally Verified" if verification_type == "formal" else "🤖 AI Detected"
            
            body = f"{emoji} **{title}**\n\n{description}\n\n_{verification_badge}_"
            
            try:
                result = await self.create_review_comment(
                    owner=owner,
                    repo=repo,
                    pr_number=pr_number,
                    commit_sha=commit_sha,
                    path=file_path,
                    line=line,
                    body=body,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to create inline comment: {e}")
        
        return results


def format_check_annotations(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format findings as GitHub check annotations."""
    annotations = []

    level_map = {
        "critical": "failure",
        "high": "failure",
        "medium": "warning",
        "low": "notice",
        "info": "notice",
    }

    for finding in findings:
        annotation = {
            "path": finding.get("file_path", ""),
            "start_line": finding.get("line_start", 1),
            "end_line": finding.get("line_end") or finding.get("line_start", 1),
            "annotation_level": level_map.get(finding.get("severity", "low"), "notice"),
            "title": finding.get("title", "Issue found"),
            "message": finding.get("description", ""),
        }
        annotations.append(annotation)

    return annotations


def format_pr_comment(
    summary: dict[str, Any],
    findings: list[dict[str, Any]],
    details_url: str | None = None,
) -> str:
    """Format analysis results as a PR comment."""
    total = summary.get("total_issues", 0)
    critical = summary.get("critical", 0)
    high = summary.get("high", 0)
    medium = summary.get("medium", 0)
    low = summary.get("low", 0)

    passed = summary.get("pass", total == 0)

    lines = [
        "## 🔍 CodeVerify Analysis",
        "",
        "### Summary",
        "",
        "| Total | Critical | High | Medium | Low |",
        "|:---:|:---:|:---:|:---:|:---:|",
        f"| {total} | {critical} | {high} | {medium} | {low} |",
        "",
    ]

    if passed:
        lines.append("✅ **Status: Passed** - No critical issues found")
    else:
        lines.append("❌ **Status: Issues Found** - Please review the findings below")

    if findings:
        lines.extend(["", "### Findings", ""])

        severity_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🔵",
        }

        for finding in findings[:10]:  # Limit to 10 in comment
            emoji = severity_emoji.get(finding.get("severity", "low"), "⚪")
            title = finding.get("title", "Issue")
            file_path = finding.get("file_path", "")
            line = finding.get("line_start", "")
            description = finding.get("description", "")

            lines.extend([
                f"<details>",
                f"<summary>{emoji} <b>{title}</b> ({file_path}:{line})</summary>",
                "",
                description,
                "",
            ])

            if fix := finding.get("fix_suggestion"):
                lines.extend([
                    "**Suggested fix:**",
                    "```suggestion",
                    fix,
                    "```",
                    "",
                ])

            lines.extend(["</details>", ""])

        if len(findings) > 10:
            lines.append(f"*...and {len(findings) - 10} more findings*")

    lines.extend([
        "",
        "---",
        "*Powered by [CodeVerify](https://codeverify.dev) - AI-powered code review with formal verification*",
    ])

    if details_url:
        lines.insert(-1, f"[View full report]({details_url})")

    return "\n".join(lines)
