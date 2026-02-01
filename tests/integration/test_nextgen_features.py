"""Integration tests for next-gen features."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio


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
        from codeverify_agents import TrustScoreAgent
        
        agent = TrustScoreAgent()
        result = await agent.analyze(sample_code)
        
        assert result is not None
        assert 0 <= result.score <= 100
        assert result.risk_level in ["low", "medium", "high", "critical"]
        assert result.factors is not None

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
        result = await agent.analyze(risky_code)
        
        # Risky code should have lower score
        assert result.risk_level in ["medium", "high", "critical"]


class TestVCSIntegration:
    """Integration tests for VCS abstraction."""

    @pytest.mark.asyncio
    async def test_github_client_mock_api(self):
        """Test GitHub client with mocked API."""
        from codeverify_core.vcs import GitHubClient
        
        client = GitHubClient(
            owner="test",
            repo="repo",
            token="test-token"
        )
        
        # Mock the HTTP request
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {
                "number": 1,
                "title": "Test PR",
                "body": "Description",
                "state": "open",
                "head": {"ref": "feature", "sha": "abc123"},
                "base": {"ref": "main", "sha": "def456"},
                "user": {"login": "user"},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
            
            pr = await client.get_pull_request(1)
            
            assert pr.number == 1
            assert pr.title == "Test PR"

    def test_vcs_factory_creates_correct_client(self):
        """Factory creates appropriate client for URL."""
        from codeverify_core.vcs import create_vcs_client
        from codeverify_core.vcs import GitHubClient, GitLabClient, BitbucketClient
        
        github = create_vcs_client("https://github.com/owner/repo", token="t")
        assert isinstance(github, GitHubClient)
        
        gitlab = create_vcs_client("https://gitlab.com/owner/repo", token="t")
        assert isinstance(gitlab, GitLabClient)
        
        bitbucket = create_vcs_client("https://bitbucket.org/owner/repo", token="t")
        assert isinstance(bitbucket, BitbucketClient)


class TestRulesIntegration:
    """Integration tests for custom rules."""

    def test_rule_evaluation_end_to_end(self):
        """Test complete rule evaluation flow."""
        from codeverify_core.rules import CustomRule, RuleType, RuleEvaluator
        
        rule = CustomRule(
            id="test-no-print",
            name="No Print",
            description="Disallow print statements",
            type=RuleType.PATTERN,
            pattern=r"print\s*\(",
            severity="warning",
            message="Use logger instead of print",
        )
        
        code = """
def hello():
    print("Hello")
    print("World")
    logger.info("Better")
"""
        
        evaluator = RuleEvaluator()
        violations = evaluator.evaluate(rule, code)
        
        assert len(violations) == 2
        assert all(v.rule_id == "test-no-print" for v in violations)

    def test_builtin_rules_all_valid(self):
        """All builtin rules can be loaded and have required fields."""
        from codeverify_core.rules import get_builtin_rules
        
        rules = get_builtin_rules()
        
        assert len(rules) > 0
        for rule in rules:
            assert rule.id, "Rule must have id"
            assert rule.name, "Rule must have name"
            assert rule.severity in ["error", "warning", "info", "critical"]


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
        result = await agent.analyze(sample_diff, {
            "pr_number": 42,
            "base_branch": "main",
        })
        
        assert result is not None
        # Should have some form of summary


class TestNotificationsIntegration:
    """Integration tests for notifications."""

    def test_slack_formatter_creates_valid_blocks(self):
        """Slack formatter creates valid Block Kit blocks."""
        from codeverify_core.notifications import SlackFormatter
        
        formatter = SlackFormatter()
        
        message = formatter.format_analysis_complete(
            repo="owner/repo",
            pr_number=42,
            findings_count=3,
            passed=False,
        )
        
        assert "blocks" in message
        assert isinstance(message["blocks"], list)

    def test_teams_formatter_creates_valid_card(self):
        """Teams formatter creates valid MessageCard."""
        from codeverify_core.notifications import TeamsFormatter
        
        formatter = TeamsFormatter()
        
        message = formatter.format_analysis_complete(
            repo="owner/repo",
            pr_number=42,
            findings_count=3,
            passed=False,
        )
        
        assert "@type" in message
        assert message["@type"] == "MessageCard"


class TestScanningIntegration:
    """Integration tests for codebase scanning."""

    def test_scan_configuration_validation(self):
        """Scan configuration validates correctly."""
        from codeverify_core.scanning import ScanConfiguration
        
        config = ScanConfiguration(
            repository="owner/repo",
            branch="main",
            include_patterns=["**/*.py"],
            exclude_patterns=["**/test/**"],
        )
        
        assert config.repository == "owner/repo"
        assert config.branch == "main"


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def api_client(self):
        """Create test API client."""
        from fastapi.testclient import TestClient
        from codeverify_api.main import app
        return TestClient(app)

    def test_trust_score_endpoint(self, api_client):
        """Trust score API endpoint works."""
        response = api_client.post(
            "/api/v1/trust-score/analyze",
            json={"code": "def test(): pass", "language": "python"}
        )
        
        assert response.status_code in [200, 201, 422]

    def test_rules_endpoint(self, api_client):
        """Rules API endpoint works."""
        response = api_client.get("/api/v1/rules")
        
        assert response.status_code == 200

    def test_scan_trigger_endpoint(self, api_client):
        """Scan trigger endpoint works."""
        response = api_client.post(
            "/api/v1/scans",
            json={"repository": "owner/repo", "branch": "main"}
        )
        
        assert response.status_code in [200, 201, 202]


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

    def test_cli_list_rules_command(self):
        """List-rules command shows rules."""
        from click.testing import CliRunner
        from codeverify_cli.main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["list-rules"])
        
        assert result.exit_code == 0
