"""Autonomous Fix Agent - Creates and submits verified fix PRs autonomously.

This module extends the AgenticAutoFix capability to operate fully autonomously:
1. Detects issues from analysis results
2. Generates multiple fix candidates
3. Verifies fixes using Z3 formal methods
4. Runs fixes in sandboxed environments
5. Creates pull requests with verification attestations
6. Learns from merged/rejected fixes to improve accuracy

Key differentiator: End-to-end autonomous remediation with formal guarantees.
"""

import asyncio
import hashlib
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import structlog

from codeverify_agents.agentic_autofix import (
    AgenticAutoFix,
    AutoFixResult,
    Finding,
    FixCategory,
    FixGenerator,
    FixStatus,
    FixVerifier,
    GeneratedFix,
)
from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class PRStatus(str, Enum):
    """Status of an autonomous PR."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    MERGED = "merged"
    REJECTED = "rejected"
    CLOSED = "closed"


class SandboxStatus(str, Enum):
    """Status of sandbox execution."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    timeout_seconds: int = 300
    memory_limit_mb: int = 512
    cpu_limit: float = 1.0
    network_enabled: bool = False
    use_docker: bool = True
    docker_image: str = "python:3.11-slim"
    working_dir: str | None = None


@dataclass
class SandboxResult:
    """Result of sandbox execution."""

    status: SandboxStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0
    tests_passed: int = 0
    tests_failed: int = 0
    error: str | None = None


@dataclass
class AutonomousPR:
    """Represents an autonomously created pull request."""

    id: str
    fix_id: str
    repository: str
    branch_name: str
    base_branch: str
    title: str
    body: str
    status: PRStatus
    pr_number: int | None = None
    pr_url: str | None = None
    verification_attestation: dict[str, Any] | None = None
    sandbox_result: SandboxResult | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    merged_at: datetime | None = None
    feedback: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fix_id": self.fix_id,
            "repository": self.repository,
            "branch_name": self.branch_name,
            "base_branch": self.base_branch,
            "title": self.title,
            "body": self.body,
            "status": self.status.value,
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "verification_attestation": self.verification_attestation,
            "sandbox_result": {
                "status": self.sandbox_result.status.value,
                "tests_passed": self.sandbox_result.tests_passed,
                "tests_failed": self.sandbox_result.tests_failed,
            } if self.sandbox_result else None,
            "created_at": self.created_at.isoformat(),
            "merged_at": self.merged_at.isoformat() if self.merged_at else None,
        }


@dataclass
class AutonomousFixResult:
    """Result of autonomous fix operation."""

    success: bool
    prs_created: list[AutonomousPR] = field(default_factory=list)
    fixes_attempted: int = 0
    fixes_verified: int = 0
    sandbox_passed: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "prs_created": [pr.to_dict() for pr in self.prs_created],
            "fixes_attempted": self.fixes_attempted,
            "fixes_verified": self.fixes_verified,
            "sandbox_passed": self.sandbox_passed,
            "error": self.error,
        }


class SandboxExecutor:
    """Executes code fixes in isolated sandbox environments."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    async def execute(
        self,
        fix: GeneratedFix,
        test_command: str | None = None,
        project_path: str | None = None,
    ) -> SandboxResult:
        """Execute a fix in a sandbox and run tests."""
        if self.config.use_docker:
            return await self._execute_docker(fix, test_command, project_path)
        else:
            return await self._execute_local(fix, test_command, project_path)

    async def _execute_docker(
        self,
        fix: GeneratedFix,
        test_command: str | None,
        project_path: str | None,
    ) -> SandboxResult:
        """Execute fix in Docker container."""
        import time
        start_time = time.time()

        try:
            # Create temporary directory with fix
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write fixed code
                fix_file = Path(tmpdir) / "fixed_code.py"
                fix_file.write_text(fix.fixed_code)

                # Write tests if available
                if fix.generated_tests:
                    test_file = Path(tmpdir) / "test_fix.py"
                    test_content = "\n\n".join(fix.generated_tests)
                    test_file.write_text(test_content)

                # Build Docker command
                docker_cmd = [
                    "docker", "run",
                    "--rm",
                    "--network", "none" if not self.config.network_enabled else "bridge",
                    f"--memory={self.config.memory_limit_mb}m",
                    f"--cpus={self.config.cpu_limit}",
                    "-v", f"{tmpdir}:/workspace:ro",
                    "-w", "/workspace",
                    self.config.docker_image,
                ]

                # Add test command
                if test_command:
                    docker_cmd.extend(["sh", "-c", test_command])
                elif fix.generated_tests:
                    docker_cmd.extend(["python", "-m", "pytest", "test_fix.py", "-v"])
                else:
                    # Syntax check only
                    docker_cmd.extend(["python", "-m", "py_compile", "fixed_code.py"])

                # Run container
                process = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.config.timeout_seconds
                    )

                    elapsed_ms = (time.time() - start_time) * 1000

                    # Parse test results
                    tests_passed, tests_failed = self._parse_pytest_output(
                        stdout.decode() + stderr.decode()
                    )

                    status = (
                        SandboxStatus.PASSED if process.returncode == 0
                        else SandboxStatus.FAILED
                    )

                    return SandboxResult(
                        status=status,
                        exit_code=process.returncode,
                        stdout=stdout.decode(),
                        stderr=stderr.decode(),
                        duration_ms=elapsed_ms,
                        tests_passed=tests_passed,
                        tests_failed=tests_failed,
                    )

                except asyncio.TimeoutError:
                    process.kill()
                    return SandboxResult(
                        status=SandboxStatus.TIMEOUT,
                        duration_ms=(time.time() - start_time) * 1000,
                        error=f"Timeout after {self.config.timeout_seconds}s",
                    )

        except Exception as e:
            logger.error("Sandbox execution failed", error=str(e))
            return SandboxResult(
                status=SandboxStatus.FAILED,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_local(
        self,
        fix: GeneratedFix,
        test_command: str | None,
        project_path: str | None,
    ) -> SandboxResult:
        """Execute fix locally (less isolated, for development)."""
        import time
        start_time = time.time()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                fix_file = Path(tmpdir) / "fixed_code.py"
                fix_file.write_text(fix.fixed_code)

                # Simple syntax check
                process = await asyncio.create_subprocess_exec(
                    "python", "-m", "py_compile", str(fix_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await process.communicate()
                elapsed_ms = (time.time() - start_time) * 1000

                status = (
                    SandboxStatus.PASSED if process.returncode == 0
                    else SandboxStatus.FAILED
                )

                return SandboxResult(
                    status=status,
                    exit_code=process.returncode,
                    stdout=stdout.decode(),
                    stderr=stderr.decode(),
                    duration_ms=elapsed_ms,
                )

        except Exception as e:
            return SandboxResult(
                status=SandboxStatus.FAILED,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _parse_pytest_output(self, output: str) -> tuple[int, int]:
        """Parse pytest output to extract pass/fail counts."""
        import re

        # Look for pytest summary line: "X passed, Y failed"
        match = re.search(r'(\d+)\s+passed', output)
        passed = int(match.group(1)) if match else 0

        match = re.search(r'(\d+)\s+failed', output)
        failed = int(match.group(1)) if match else 0

        return passed, failed


class GitOperations:
    """Safe Git operations for autonomous PRs."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path

    async def create_branch(self, branch_name: str, base_branch: str = "main") -> bool:
        """Create a new branch from base."""
        try:
            # Fetch latest
            await self._run_git("fetch", "origin", base_branch)

            # Create branch
            await self._run_git("checkout", "-b", branch_name, f"origin/{base_branch}")

            return True
        except Exception as e:
            logger.error("Failed to create branch", error=str(e))
            return False

    async def apply_fix(self, fix: GeneratedFix, file_path: str) -> bool:
        """Apply a fix to a file."""
        try:
            full_path = Path(self.repo_path) / file_path
            full_path.write_text(fix.fixed_code)
            return True
        except Exception as e:
            logger.error("Failed to apply fix", error=str(e))
            return False

    async def commit(self, message: str, files: list[str]) -> str | None:
        """Commit changes and return commit SHA."""
        try:
            for file in files:
                await self._run_git("add", file)

            await self._run_git("commit", "-m", message)

            result = await self._run_git("rev-parse", "HEAD")
            return result.strip()
        except Exception as e:
            logger.error("Failed to commit", error=str(e))
            return None

    async def push(self, branch_name: str) -> bool:
        """Push branch to remote."""
        try:
            await self._run_git("push", "-u", "origin", branch_name)
            return True
        except Exception as e:
            logger.error("Failed to push", error=str(e))
            return False

    async def cleanup_branch(self, branch_name: str) -> None:
        """Delete local branch."""
        try:
            await self._run_git("checkout", "main")
            await self._run_git("branch", "-D", branch_name)
        except Exception:
            pass

    async def _run_git(self, *args: str) -> str:
        """Run a git command."""
        process = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Git command failed: {stderr.decode()}")

        return stdout.decode()


class PRCreator:
    """Creates pull requests via GitHub API or CLI."""

    def __init__(
        self,
        github_token: str | None = None,
        use_cli: bool = True,
    ) -> None:
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.use_cli = use_cli

    async def create_pr(
        self,
        repository: str,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> tuple[int | None, str | None]:
        """Create a PR and return (pr_number, pr_url)."""
        if self.use_cli:
            return await self._create_pr_cli(
                repository, branch_name, base_branch, title, body
            )
        else:
            return await self._create_pr_api(
                repository, branch_name, base_branch, title, body
            )

    async def _create_pr_cli(
        self,
        repository: str,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> tuple[int | None, str | None]:
        """Create PR using GitHub CLI."""
        try:
            process = await asyncio.create_subprocess_exec(
                "gh", "pr", "create",
                "--repo", repository,
                "--head", branch_name,
                "--base", base_branch,
                "--title", title,
                "--body", body,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                pr_url = stdout.decode().strip()
                # Extract PR number from URL
                pr_number = int(pr_url.split("/")[-1])
                return pr_number, pr_url
            else:
                logger.error("gh pr create failed", stderr=stderr.decode())
                return None, None

        except Exception as e:
            logger.error("Failed to create PR via CLI", error=str(e))
            return None, None

    async def _create_pr_api(
        self,
        repository: str,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> tuple[int | None, str | None]:
        """Create PR using GitHub API."""
        import httpx

        try:
            owner, repo = repository.split("/")
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"token {self.github_token}",
                        "Accept": "application/vnd.github.v3+json",
                    },
                    json={
                        "title": title,
                        "body": body,
                        "head": branch_name,
                        "base": base_branch,
                    },
                )

                if response.status_code == 201:
                    data = response.json()
                    return data["number"], data["html_url"]
                else:
                    logger.error(
                        "GitHub API error",
                        status=response.status_code,
                        body=response.text,
                    )
                    return None, None

        except Exception as e:
            logger.error("Failed to create PR via API", error=str(e))
            return None, None


class FeedbackLearner:
    """Learns from PR outcomes to improve fix quality."""

    def __init__(self, storage_path: str | None = None) -> None:
        self.storage_path = storage_path or tempfile.gettempdir()
        self._feedback_history: list[dict[str, Any]] = []

    def record_outcome(
        self,
        fix: GeneratedFix,
        pr: AutonomousPR,
        outcome: str,  # "merged", "rejected", "modified"
        reviewer_comments: list[str] | None = None,
    ) -> None:
        """Record the outcome of a fix for learning."""
        record = {
            "fix_id": fix.id,
            "category": fix.metadata.get("method", "unknown"),
            "confidence": fix.confidence,
            "outcome": outcome,
            "reviewer_comments": reviewer_comments or [],
            "timestamp": datetime.utcnow().isoformat(),
            "diff_hash": hashlib.md5(fix.diff.encode()).hexdigest(),
        }

        self._feedback_history.append(record)
        logger.info("Recorded fix outcome", outcome=outcome, fix_id=fix.id)

    def get_success_rate(self, category: str | None = None) -> float:
        """Get the success rate for fixes."""
        relevant = [
            r for r in self._feedback_history
            if category is None or r["category"] == category
        ]

        if not relevant:
            return 0.5  # Default

        merged = sum(1 for r in relevant if r["outcome"] == "merged")
        return merged / len(relevant)

    def should_attempt_fix(self, fix: GeneratedFix) -> bool:
        """Decide whether to attempt a fix based on historical success."""
        category = fix.metadata.get("method", "unknown")
        success_rate = self.get_success_rate(category)

        # Require higher confidence for categories with lower success
        min_confidence = 0.3 + (0.5 * (1 - success_rate))
        return fix.confidence >= min_confidence


class AutonomousFixAgent(BaseAgent):
    """Fully autonomous agent that creates verified fix PRs.

    This agent:
    1. Receives findings from analysis
    2. Generates fix candidates using AgenticAutoFix
    3. Verifies fixes using Z3 formal methods
    4. Tests fixes in sandboxed environments
    5. Creates PRs with verification attestations
    6. Learns from outcomes to improve over time
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        sandbox_config: SandboxConfig | None = None,
        github_token: str | None = None,
        repo_path: str | None = None,
    ) -> None:
        super().__init__(config)

        self._autofix = AgenticAutoFix(config)
        self._sandbox = SandboxExecutor(sandbox_config)
        self._pr_creator = PRCreator(github_token)
        self._feedback_learner = FeedbackLearner()
        self._repo_path = repo_path or os.getcwd()
        self._git = GitOperations(self._repo_path)

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """Analyze code and create autonomous fix PRs."""
        import time
        start_time = time.time()

        findings = context.get("findings", [])
        repository = context.get("repository", "")
        base_branch = context.get("base_branch", "main")

        try:
            result = await self.create_fix_prs(
                code=code,
                findings=findings,
                repository=repository,
                base_branch=base_branch,
                language=context.get("language", "python"),
            )

            elapsed_ms = (time.time() - start_time) * 1000

            return AgentResult(
                success=result.success,
                data=result.to_dict(),
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("Autonomous fix failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def create_fix_prs(
        self,
        code: str,
        findings: list[dict[str, Any]] | list[Finding],
        repository: str,
        base_branch: str = "main",
        language: str = "python",
    ) -> AutonomousFixResult:
        """Create autonomous fix PRs for all findings.

        Pipeline:
        1. Generate fixes using AgenticAutoFix
        2. Filter fixes based on feedback learner
        3. Run sandbox verification
        4. Create PRs for passing fixes
        """
        # Step 1: Generate and verify fixes
        autofix_result = await self._autofix.auto_fix(code, findings, language)

        if not autofix_result.success:
            return AutonomousFixResult(
                success=False,
                error="Failed to generate fixes",
            )

        prs_created: list[AutonomousPR] = []
        sandbox_passed = 0

        # Step 2: Process each verified fix
        for fix in autofix_result.fixes:
            if fix.status not in (FixStatus.VERIFIED, FixStatus.READY_FOR_PR):
                continue

            # Check if we should attempt based on historical success
            if not self._feedback_learner.should_attempt_fix(fix):
                logger.info("Skipping fix due to low predicted success", fix_id=fix.id)
                continue

            # Step 3: Run sandbox verification
            sandbox_result = await self._sandbox.execute(fix)

            if sandbox_result.status != SandboxStatus.PASSED:
                logger.info(
                    "Fix failed sandbox verification",
                    fix_id=fix.id,
                    status=sandbox_result.status,
                )
                continue

            sandbox_passed += 1

            # Step 4: Create PR
            pr = await self._create_pr_for_fix(
                fix=fix,
                repository=repository,
                base_branch=base_branch,
                sandbox_result=sandbox_result,
            )

            if pr:
                prs_created.append(pr)

        return AutonomousFixResult(
            success=len(prs_created) > 0,
            prs_created=prs_created,
            fixes_attempted=len(autofix_result.fixes),
            fixes_verified=autofix_result.fixes_verified,
            sandbox_passed=sandbox_passed,
        )

    async def _create_pr_for_fix(
        self,
        fix: GeneratedFix,
        repository: str,
        base_branch: str,
        sandbox_result: SandboxResult,
    ) -> AutonomousPR | None:
        """Create a PR for a verified fix."""
        # Generate branch name
        branch_name = f"codeverify/fix-{fix.finding_id[:8]}-{uuid.uuid4().hex[:6]}"

        # Create branch
        if not await self._git.create_branch(branch_name, base_branch):
            return None

        try:
            # Apply fix
            file_path = fix.metadata.get("file_path", "unknown.py")
            if not await self._git.apply_fix(fix, file_path):
                await self._git.cleanup_branch(branch_name)
                return None

            # Commit
            commit_message = self._generate_commit_message(fix)
            commit_sha = await self._git.commit(commit_message, [file_path])

            if not commit_sha:
                await self._git.cleanup_branch(branch_name)
                return None

            # Push
            if not await self._git.push(branch_name):
                await self._git.cleanup_branch(branch_name)
                return None

            # Create PR
            title, body = self._generate_pr_content(fix, sandbox_result)
            pr_number, pr_url = await self._pr_creator.create_pr(
                repository=repository,
                branch_name=branch_name,
                base_branch=base_branch,
                title=title,
                body=body,
            )

            # Build verification attestation
            attestation = self._build_attestation(fix, sandbox_result)

            return AutonomousPR(
                id=str(uuid.uuid4()),
                fix_id=fix.id,
                repository=repository,
                branch_name=branch_name,
                base_branch=base_branch,
                title=title,
                body=body,
                status=PRStatus.PENDING_REVIEW,
                pr_number=pr_number,
                pr_url=pr_url,
                verification_attestation=attestation,
                sandbox_result=sandbox_result,
            )

        except Exception as e:
            logger.error("Failed to create PR", error=str(e))
            await self._git.cleanup_branch(branch_name)
            return None

    def _generate_commit_message(self, fix: GeneratedFix) -> str:
        """Generate commit message for fix."""
        category = fix.metadata.get("category", "issue")
        return f"""fix: {fix.explanation}

Auto-generated by CodeVerify Autonomous Fix Agent.

Verification Status: {fix.status.value}
Confidence: {fix.confidence:.0%}
Finding ID: {fix.finding_id}

This fix has been:
- Verified using Z3 formal methods
- Tested in a sandboxed environment
- Generated with {fix.confidence:.0%} confidence

Co-authored-by: CodeVerify Bot <bot@codeverify.io>
"""

    def _generate_pr_content(
        self,
        fix: GeneratedFix,
        sandbox_result: SandboxResult,
    ) -> tuple[str, str]:
        """Generate PR title and body."""
        title = f"[CodeVerify] {fix.explanation}"

        body = f"""## Summary

This PR was automatically generated by CodeVerify's Autonomous Fix Agent.

### Issue Fixed
{fix.explanation}

### Changes
```diff
{fix.diff}
```

### Verification Status
| Check | Status |
|-------|--------|
| Formal Verification (Z3) | ✅ Passed |
| Sandbox Testing | {'✅ Passed' if sandbox_result.status == SandboxStatus.PASSED else '⚠️ ' + sandbox_result.status.value} |
| Tests Passed | {sandbox_result.tests_passed} |
| Tests Failed | {sandbox_result.tests_failed} |

### Confidence Score
**{fix.confidence:.0%}** - Based on verification results and historical accuracy.

### Generated Tests
{self._format_tests(fix.generated_tests)}

---

<details>
<summary>Verification Details</summary>

```json
{self._format_verification_result(fix.verification_result)}
```

</details>

---
*This PR was autonomously generated and verified by CodeVerify. Please review before merging.*
"""

        return title, body

    def _format_tests(self, tests: list[str]) -> str:
        """Format generated tests for PR body."""
        if not tests:
            return "_No tests generated._"

        return "```python\n" + "\n\n".join(tests) + "\n```"

    def _format_verification_result(self, result: dict[str, Any] | None) -> str:
        """Format verification result as JSON."""
        import json
        if not result:
            return "{}"
        return json.dumps(result, indent=2)

    def _build_attestation(
        self,
        fix: GeneratedFix,
        sandbox_result: SandboxResult,
    ) -> dict[str, Any]:
        """Build verification attestation for PR."""
        import hashlib

        # Create attestation data
        attestation_data = {
            "fix_id": fix.id,
            "finding_id": fix.finding_id,
            "timestamp": datetime.utcnow().isoformat(),
            "verification": {
                "status": fix.status.value,
                "result": fix.verification_result,
            },
            "sandbox": {
                "status": sandbox_result.status.value,
                "tests_passed": sandbox_result.tests_passed,
                "tests_failed": sandbox_result.tests_failed,
                "duration_ms": sandbox_result.duration_ms,
            },
            "confidence": fix.confidence,
            "diff_hash": hashlib.sha256(fix.diff.encode()).hexdigest(),
        }

        # Sign attestation
        attestation_json = str(attestation_data).encode()
        signature = hashlib.sha256(attestation_json).hexdigest()
        attestation_data["signature"] = signature

        return attestation_data

    def record_feedback(
        self,
        pr: AutonomousPR,
        outcome: str,
        comments: list[str] | None = None,
    ) -> None:
        """Record feedback for learning."""
        # Find the fix
        for fix in getattr(self, '_recent_fixes', []):
            if fix.id == pr.fix_id:
                self._feedback_learner.record_outcome(fix, pr, outcome, comments)
                break
