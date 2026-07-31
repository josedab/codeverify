"""Integration tests for next-gen features."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest


class TestTrustScoreIntegration:
    """Integration tests for trust score feature."""

    @pytest.fixture
    def sample_code(self):
        return """
def calculate_discount(price: float, discount_percent: float) -> float:
    '''Calculate discounted price.'''
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Invalid discount percentage")
    return price * (1 - discount_percent / 100)
"""

    @pytest.mark.asyncio
    async def test_trust_score_end_to_end(self, sample_code):
        """Test complete trust score flow."""
        from codeverify_agents import AgentResult, TrustScoreAgent

        agent = TrustScoreAgent()
        result = await agent.analyze(
            sample_code,
            {"file_path": "discounts.py", "language": "python"},
        )

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert 0 <= result.data["score"] <= 100
        assert result.data["risk_level"] in ["low", "medium", "high", "critical"]
        assert set(result.data["factors"]) == {
            "complexity_score",
            "pattern_confidence",
            "historical_accuracy",
            "verification_coverage",
            "code_quality_signals",
            "ai_detection_confidence",
        }

    @pytest.mark.asyncio
    async def test_trust_score_with_risky_code(self):
        """Trust score detects risky patterns."""
        from codeverify_agents import TrustScoreAgent

        risky_code = """
import os
def run(cmd):
    os.system(cmd)  # Security risk
    eval(cmd)       # Another risk
"""
        agent = TrustScoreAgent()
        result = await agent.analyze(
            risky_code,
            {"file_path": "runner.py", "language": "python"},
        )
        safe_result = await agent.analyze(
            "def run(value: str) -> str:\n    return value.strip()\n",
            {"file_path": "safe_runner.py", "language": "python"},
        )

        assert result.success is True
        assert result.data["risk_level"] == "high"
        assert (
            result.data["factors"]["pattern_confidence"]
            < safe_result.data["factors"]["pattern_confidence"]
        )


class TestVCSIntegration:
    """Integration tests for VCS abstraction."""

    @pytest.mark.asyncio
    async def test_github_client_mock_api(self):
        """Test GitHub client with mocked API."""
        from unittest.mock import MagicMock

        from codeverify_core.vcs import GitHubClient, VCSConfig

        client = GitHubClient(VCSConfig(provider="github", token="test-token"))
        response = MagicMock()
        response.json.return_value = {
            "id": 100,
            "number": 1,
            "title": "Test PR",
            "body": "Description",
            "state": "open",
            "head": {"ref": "feature", "sha": "abc123"},
            "base": {"ref": "main", "sha": "def456"},
            "user": {"id": 7, "login": "user"},
            "labels": [{"name": "feature"}],
            "html_url": "https://github.com/test/repo/pull/1",
            "diff_url": "https://github.com/test/repo/pull/1.diff",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

        with patch.object(
            client,
            "_request",
            new=AsyncMock(return_value=response),
        ) as mock_request:
            pr = await client.get_pull_request("test/repo", 1)

        assert pr.number == 1
        assert pr.title == "Test PR"
        assert pr.author.username == "user"
        assert pr.labels == ["feature"]
        mock_request.assert_awaited_once_with("GET", "/repos/test/repo/pulls/1")

    def test_vcs_factory_creates_correct_client(self):
        """Factory creates appropriate client for URL."""
        from codeverify_core.vcs import (
            BitbucketClient,
            GitHubClient,
            GitLabClient,
            create_vcs_client,
        )

        github = create_vcs_client(url="https://github.com/owner/repo", token="t")
        assert isinstance(github, GitHubClient)

        gitlab = create_vcs_client(url="https://gitlab.com/owner/repo", token="t")
        assert isinstance(gitlab, GitLabClient)

        bitbucket = create_vcs_client(url="https://bitbucket.org/owner/repo", token="t")
        assert isinstance(bitbucket, BitbucketClient)


class TestRulesIntegration:
    """Integration tests for custom rules."""

    def test_rule_evaluation_end_to_end(self):
        """Test complete rule evaluation flow."""
        from codeverify_core.rules import RuleBuilder, RuleEvaluator, RuleSeverity

        rule = (
            RuleBuilder()
            .name("No Print")
            .description("Disallow print statements")
            .severity(RuleSeverity.LOW)
            .pattern(r"print\s*\(")
            .action("Use logger instead of print")
            .for_languages("python")
            .build()
        )

        code = """
def hello():
    print("Hello")
    print("World")
    logger.info("Better")
"""

        evaluator = RuleEvaluator([rule])
        violations = evaluator.evaluate(code, "greetings.py", "python")

        assert len(violations) == 2
        assert all(v["rule_id"] == str(rule.id) for v in violations)
        assert {v["line"] for v in violations} == {3, 4}

    def test_builtin_rules_all_valid(self):
        """All builtin rules can be loaded and have required fields."""
        from codeverify_core.rules import get_builtin_rules

        rules = get_builtin_rules()

        assert len(rules) > 0
        assert "no-print" in rules
        for rule_name, rule in rules.items():
            assert rule_name
            assert rule.id, "Rule must have id"
            assert rule.name, "Rule must have name"
            assert rule.severity.value in ["critical", "high", "medium", "low", "info"]


class TestDebuggerIntegration:
    """Integration tests for verification debugger."""

    @pytest.mark.asyncio
    async def test_debugger_trace_simple_function(self):
        """Debugger traces simple function."""
        from codeverify_verifier import VerificationDebugger

        code = """
def add(a: int, b: int) -> int:
    return a + b
"""

        debugger = VerificationDebugger()
        result = await debugger.trace(code)

        assert "steps" in result
        assert "result" in result

    @pytest.mark.asyncio
    async def test_debugger_session_management(self):
        """Debugger manages sessions correctly."""
        from codeverify_verifier import VerificationDebugger

        debugger = VerificationDebugger()

        session1 = debugger.create_session()
        session2 = debugger.create_session()

        assert session1.session_id != session2.session_id


class TestDiffSummarizerIntegration:
    """Integration tests for diff summarizer."""

    @pytest.fixture
    def sample_diff(self):
        return """
diff --git a/src/auth.py b/src/auth.py
index abc123..def456 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,12 @@ def authenticate(user, password):
     if not user:
         return False
+
+    # Add rate limiting
+    if is_rate_limited(user):
+        raise RateLimitError("Too many attempts")
+
     return check_password(user, password)
"""

    @pytest.mark.asyncio
    async def test_diff_summarizer_generates_description(self, sample_diff):
        """Diff summarizer generates PR description."""
        from codeverify_agents import DiffSummarizerAgent

        agent = DiffSummarizerAgent()
        expected_summary = {
            "summary": "Adds authentication rate limiting.",
            "change_type": "security",
            "files_changed": ["src/auth.py"],
        }
        with patch.object(
            agent,
            "_call_llm",
            new=AsyncMock(return_value={"content": json.dumps(expected_summary), "tokens": 24}),
        ):
            result = await agent.analyze(
                sample_diff,
                {
                    "pr_number": 42,
                    "base_branch": "main",
                },
            )

        assert result.success is True
        assert result.data == expected_summary
        assert result.tokens_used == 24


class TestNotificationsIntegration:
    """Integration tests for notifications."""

    def test_slack_formatter_creates_valid_blocks(self):
        """Slack formatter creates valid Block Kit blocks."""
        from codeverify_core.notifications import AnalysisNotification, SlackFormatter

        formatter = SlackFormatter()
        notification = AnalysisNotification(
            repo_full_name="owner/repo",
            pr_number=42,
            pr_title="Harden authentication",
            pr_url="https://github.com/owner/repo/pull/42",
            status="failed",
            total_findings=3,
            critical_findings=1,
            high_findings=1,
            findings_url="https://codeverify.dev/analyses/42",
            author="octocat",
            analyzed_at=datetime.now(UTC),
        )
        message = formatter.format_analysis(notification)

        attachment = message["attachments"][0]
        assert attachment["color"] == "#ff0000"
        assert attachment["blocks"][0]["text"]["text"] == "🚨 CodeVerify Analysis Complete"
        assert attachment["blocks"][-1]["elements"][0]["url"] == notification.findings_url

    def test_teams_formatter_creates_valid_card(self):
        """Teams formatter creates valid MessageCard."""
        from codeverify_core.notifications import AnalysisNotification, TeamsFormatter

        formatter = TeamsFormatter()
        notification = AnalysisNotification(
            repo_full_name="owner/repo",
            pr_number=42,
            pr_title="Harden authentication",
            pr_url="https://github.com/owner/repo/pull/42",
            status="failed",
            total_findings=3,
            critical_findings=1,
            high_findings=1,
            findings_url="https://codeverify.dev/analyses/42",
            author="octocat",
            analyzed_at=datetime.now(UTC),
        )
        message = formatter.format_analysis(notification)

        assert message["@type"] == "MessageCard"
        assert message["themeColor"] == "FF0000"
        assert message["potentialAction"][0]["targets"][0]["uri"] == notification.findings_url


class TestScanningIntegration:
    """Integration tests for codebase scanning."""

    def test_scan_configuration_validation(self):
        """Scan configuration validates correctly."""
        from codeverify_core.scanning import ScanConfiguration

        config = ScanConfiguration(
            repo_full_name="owner/repo",
            branch="main",
            include_patterns=["**/*.py"],
            exclude_patterns=["**/test/**"],
        )

        assert config.repo_full_name == "owner/repo"
        assert config.branch == "main"
        assert config.include_security is True


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def api_client(self, monkeypatch: pytest.MonkeyPatch):
        """Create test API client."""
        from fastapi.testclient import TestClient

        monkeypatch.setenv("CORS_ORIGINS", '["http://test"]')
        monkeypatch.setenv("ENVIRONMENT", "development")
        from codeverify_api.main import app

        with TestClient(app, base_url="http://test") as client:
            yield client

    def test_trust_score_endpoint(self, api_client):
        """Trust score API endpoint works."""
        response = api_client.post(
            "/api/v1/trust-score",
            json={
                "code": "def test() -> None:\n    pass",
                "file_path": "test.py",
                "language": "python",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["score"] <= 100
        assert data["risk_level"] in {"low", "medium", "high", "critical"}
        assert len(data["code_hash"]) == 64

    def test_rules_endpoint(self, api_client):
        """Rules API endpoint works."""
        response = api_client.get("/api/v1/rules/templates")

        assert response.status_code == 200
        templates = response.json()["templates"]
        assert "no-print" in templates
        assert templates["no-print"]["severity"] == "low"

    def test_scan_trigger_endpoint(self, api_client):
        """Scan trigger endpoint works."""
        response = api_client.post(
            "/api/v1/scans/trigger",
            json={"repo_full_name": "owner/repo", "branch": "main"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["repo_full_name"] == "owner/repo"
        assert data["status"] == "queued"
        assert data["scan_type"] == "full"


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_cli_help(self):
        """CLI shows help."""
        from click.testing import CliRunner
        from codeverify_cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "CodeVerify" in result.output

    def test_cli_trust_score_command_exists(self):
        """Trust score command exists."""
        from click.testing import CliRunner
        from codeverify_cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["trust-score", "--help"])

        assert result.exit_code == 0
        assert "trust score" in result.output.lower()

    def test_cli_rules_command_exists(self):
        """Rules command exists."""
        from click.testing import CliRunner
        from codeverify_cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["rules", "--help"])

        assert result.exit_code == 0

    def test_cli_scan_command_exists(self):
        """Scan command exists."""
        from click.testing import CliRunner
        from codeverify_cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["scan", "--help"])

        assert result.exit_code == 0

    def test_cli_debug_command_exists(self):
        """Debug command exists."""
        from click.testing import CliRunner
        from codeverify_cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["debug", "--help"])

        assert result.exit_code == 0

    def test_cli_languages_command(self):
        """CLI serializes the core language registry."""
        from click.testing import CliRunner
        from codeverify_cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["languages", "--format", "json"])

        assert result.exit_code == 0
        languages = json.loads(result.output)
        assert ".py" in languages["python"]["extensions"]
        assert languages["typescript"]["generics"] is True
