"""Tests for Plugin Marketplace & Agent SDK modules."""

import pytest

from codeverify_core.agent_sdk import (
    AgentCapability,
    AgentCategory,
    AgentLanguage,
    AgentManifest,
    AgentPackage,
    AgentLifecycle,
    AnalysisContext,
    AnalysisResult,
    BaseAgent,
    Finding,
    SeverityLevel,
    agent,
)
from codeverify_core.agent_runtime import (
    AgentLoadError,
    AgentSandbox,
    IsolatedAgentRunner,
    ResourceLimitExceeded,
    SandboxConfig,
    SandboxError,
    SecurityViolation,
    run_agent,
)


class TestAgentCapability:
    def test_capabilities_exist(self):
        assert AgentCapability.ANALYZE is not None
        assert AgentCapability.FIX is not None
        assert AgentCapability.REPORT is not None


class TestAgentCategory:
    def test_categories_exist(self):
        assert AgentCategory.SECURITY is not None
        assert AgentCategory.QUALITY is not None
        assert AgentCategory.PERFORMANCE is not None


class TestAgentLanguage:
    def test_languages_exist(self):
        assert AgentLanguage.PYTHON is not None
        assert AgentLanguage.TYPESCRIPT is not None


class TestSeverityLevel:
    def test_levels_exist(self):
        assert SeverityLevel.CRITICAL is not None
        assert SeverityLevel.HIGH is not None
        assert SeverityLevel.MEDIUM is not None
        assert SeverityLevel.LOW is not None


class TestFinding:
    def test_creation(self):
        finding = Finding(
            title="SQL Injection",
            description="Unsanitized input",
            severity=SeverityLevel.CRITICAL,
            file_path="app.py",
            line_start=42,
        )
        assert finding.title == "SQL Injection"
        assert finding.severity == SeverityLevel.CRITICAL


class TestAnalysisContext:
    def test_creation(self):
        ctx = AnalysisContext(
            repo_full_name="owner/repo",
            file_paths=["src/main.py"],
        )
        assert ctx.repo_full_name == "owner/repo"
        assert len(ctx.file_paths) == 1


class TestAnalysisResult:
    def test_creation(self):
        result = AnalysisResult(
            agent_id="test-agent",
            agent_version="1.0.0",
            findings=[],
            metadata={"duration_ms": 100},
        )
        assert len(result.findings) == 0
        assert result.agent_id == "test-agent"


class TestAgentManifest:
    def test_creation(self):
        manifest = AgentManifest(
            name="my-agent",
            version="1.0.0",
            description="A test agent",
            author="tester",
            capabilities=[AgentCapability.ANALYZE],
            category=AgentCategory.SECURITY,
            languages=[AgentLanguage.PYTHON],
        )
        assert manifest.name == "my-agent"
        assert manifest.version == "1.0.0"


class TestAgentDecorator:
    def test_decorator_exists(self):
        assert callable(agent)


# --- Agent Runtime Tests ---

class TestSandboxConfig:
    def test_defaults(self):
        config = SandboxConfig()
        assert config is not None


class TestAgentSandbox:
    def test_creation(self):
        sandbox = AgentSandbox()
        assert sandbox is not None


class TestIsolatedAgentRunner:
    def test_creation(self):
        runner = IsolatedAgentRunner()
        assert runner is not None


class TestRunAgent:
    def test_function_exists(self):
        assert callable(run_agent)


class TestExceptions:
    def test_sandbox_error(self):
        with pytest.raises(SandboxError):
            raise SandboxError("sandbox failure")

    def test_agent_load_error(self):
        with pytest.raises(AgentLoadError):
            raise AgentLoadError("load failure")

    def test_resource_limit(self):
        with pytest.raises(ResourceLimitExceeded):
            raise ResourceLimitExceeded("memory exceeded")

    def test_security_violation(self):
        with pytest.raises(SecurityViolation):
            raise SecurityViolation("unauthorized access")
