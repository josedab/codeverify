"""Tests for the core AI agent interfaces."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from codeverify_agents import (
    AgentConfig,
    AgentResult,
    SecurityAgent,
    SemanticAgent,
    SynthesisAgent,
)
from codeverify_agents.base import BaseAgent
from codeverify_agents.factory import MockLLMClientProvider


class _ConcreteAgent(BaseAgent):
    """Minimal concrete agent used to exercise BaseAgent behavior."""

    async def analyze(self, code: str, context: dict) -> AgentResult:
        return AgentResult(success=True, data={"code": code, "context": context})


class TestBaseAgent:
    """Tests for base agent functionality."""

    def test_agent_initialization(self):
        """Agent stores the supplied AgentConfig."""
        config = AgentConfig(provider="anthropic", anthropic_model="claude-test")
        agent = _ConcreteAgent(config)

        assert agent.config is config
        assert agent.config.provider == "anthropic"
        assert agent.config.anthropic_model == "claude-test"

    def test_agent_default_provider(self):
        """Agent uses the documented default configuration."""
        agent = _ConcreteAgent()

        assert agent.config.provider == "openai"
        assert agent.config.openai_model == "gpt-4-turbo-preview"

    @pytest.mark.asyncio
    async def test_call_llm_openai(self):
        """Agent calls an injected OpenAI client without network access."""
        agent = _ConcreteAgent(AgentConfig(provider="openai", openai_api_key="test-key"))
        agent._llm_provider = MockLLMClientProvider(openai_response="Test response")

        response = await agent._call_llm("System prompt", "User prompt", json_mode=True)

        assert response["content"] == "Test response"
        assert response["tokens"] == 100
        assert response["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_call_llm_anthropic(self):
        """Agent calls an injected Anthropic client without network access."""
        agent = _ConcreteAgent(AgentConfig(provider="anthropic", anthropic_api_key="test-key"))
        agent._llm_provider = MockLLMClientProvider(anthropic_response="Test response")

        response = await agent._call_llm("System prompt", "User prompt")

        assert response["content"] == "Test response"
        assert response["tokens"] == 100
        assert response["latency_ms"] >= 0


class TestSemanticAgent:
    """Tests for the current semantic analysis agent."""

    @pytest.fixture
    def agent(self):
        return SemanticAgent()

    def test_agent_type(self, agent):
        """SemanticAgent is a BaseAgent with semantic defaults."""
        assert isinstance(agent, BaseAgent)
        assert agent.config.provider == "openai"
        assert agent.config.openai_model == "gpt-4-turbo-preview"

    @pytest.mark.asyncio
    async def test_analyze_function(self, agent):
        """SemanticAgent returns an AgentResult containing parsed analysis."""
        code = """
def calculate_total(items, tax_rate):
    subtotal = sum(item.price for item in items)
    return subtotal * (1 + tax_rate)
"""
        payload = {
            "summary": "Calculate total price with tax",
            "functions": [
                {
                    "name": "calculate_total",
                    "purpose": "Calculate a taxed total",
                    "preconditions": ["items is iterable", "tax_rate >= 0"],
                    "postconditions": ["return value >= 0"],
                    "assumptions": [],
                    "edge_cases": ["empty items list"],
                    "concerns": [],
                }
            ],
            "behavioral_changes": [],
            "verification_hints": ["check tax_rate"],
        }
        response = {
            "content": json.dumps(payload),
            "tokens": 37,
            "latency_ms": 1.5,
        }

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=response)) as mock_llm:
            result = await agent.analyze(code, {"file_path": "pricing.py"})

        assert result.success is True
        assert result.data["summary"] == "Calculate total price with tax"
        assert result.data["functions"][0]["preconditions"] == [
            "items is iterable",
            "tax_rate >= 0",
        ]
        assert result.tokens_used == 37
        assert mock_llm.await_args.kwargs["json_mode"] is True
        assert "`pricing.py`" in mock_llm.await_args.kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_handles_malformed_response(self, agent):
        """Malformed JSON is preserved in the standard parse fallback."""
        response = {"content": "Not valid JSON", "tokens": 4, "latency_ms": 0.5}

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=response)):
            result = await agent.analyze("def foo(): pass", {})

        assert result.success is True
        assert result.data == {"raw_response": "Not valid JSON"}
        assert result.tokens_used == 4


class TestSecurityAgent:
    """Tests for the current security analysis agent."""

    @pytest.fixture
    def agent(self):
        return SecurityAgent()

    def test_agent_type(self, agent):
        """SecurityAgent is a BaseAgent with security defaults."""
        assert isinstance(agent, BaseAgent)
        assert agent.config.provider == "anthropic"
        assert agent.config.anthropic_model == "claude-3-sonnet-20240229"

    @pytest.mark.asyncio
    async def test_detect_sql_injection(self, agent):
        """SecurityAgent returns structured vulnerabilities from the LLM."""
        code = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
"""
        payload = {
            "vulnerabilities": [
                {
                    "id": "vuln-1",
                    "severity": "critical",
                    "category": "injection",
                    "cwe_id": "CWE-89",
                    "title": "SQL injection",
                    "description": "User input is interpolated into SQL",
                    "location": {"file": "db.py", "line": 3},
                    "fix_suggestion": "Use a parameterized query",
                    "confidence": 0.98,
                }
            ],
            "secrets_detected": [],
            "security_score": 20,
            "summary": "Critical SQL injection found",
        }
        response = {"content": json.dumps(payload), "tokens": 29, "latency_ms": 1.0}

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=response)):
            result = await agent.analyze(code, {"file_path": "db.py"})

        assert result.success is True
        assert result.data["vulnerabilities"][0]["cwe_id"] == "CWE-89"
        assert result.data["vulnerabilities"][0]["fix_suggestion"] == ("Use a parameterized query")
        assert result.tokens_used == 29

    @pytest.mark.asyncio
    async def test_detect_secret_exposure(self, agent):
        """The public pattern scanner reports concrete secret types and lines."""
        code = """
API_KEY = "sk-abcdef12345678901234567890"
password = "super_secret_password"
"""

        secrets = await agent.scan_for_secrets(code)

        assert [(secret["type"], secret["line"]) for secret in secrets] == [
            ("api_key", 2),
            ("openai_key", 2),
            ("password", 3),
        ]
        assert all(secret["severity"] == "high" for secret in secrets)


class TestSynthesisAgent:
    """Tests for synthesis through the current BaseAgent interface."""

    @pytest.fixture
    def agent(self):
        return SynthesisAgent()

    def test_agent_type(self, agent):
        """SynthesisAgent is a BaseAgent with synthesis defaults."""
        assert isinstance(agent, BaseAgent)
        assert agent.config.provider == "openai"
        assert agent.config.openai_model == "gpt-4-turbo-preview"

    @pytest.mark.asyncio
    async def test_consolidate_findings(self, agent):
        """SynthesisAgent exposes consolidated findings in AgentResult.data."""
        payload = {
            "summary": {
                "total_issues": 2,
                "critical": 1,
                "high": 1,
                "medium": 0,
                "low": 0,
                "pass": False,
                "recommendation": "Fix critical issues before merging",
            },
            "findings": [
                {"id": "f1", "title": "SQL injection", "severity": "critical"},
                {"id": "f2", "title": "Integer overflow", "severity": "high"},
            ],
            "github_comment": "Two issues found",
        }
        response = {"content": json.dumps(payload), "tokens": 51, "latency_ms": 2.0}
        context = {
            "semantic_results": {"issues": [{"title": "Missing null check"}]},
            "verification_results": {"violations": [{"title": "Integer overflow"}]},
            "security_results": {"vulnerabilities": [{"title": "SQL injection"}]},
        }

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=response)):
            result = await agent.analyze("def query(): pass", context)

        assert result.success is True
        assert [finding["id"] for finding in result.data["findings"]] == ["f1", "f2"]
        assert result.data["summary"]["recommendation"] == ("Fix critical issues before merging")
        assert result.tokens_used == 51

    @pytest.mark.asyncio
    async def test_deduplicates_findings(self, agent):
        """The synthesized response represents duplicate source findings once."""
        payload = {
            "summary": {"total_issues": 1, "pass": False},
            "findings": [{"id": "f1", "title": "Null check", "severity": "medium"}],
        }
        response = {"content": json.dumps(payload), "tokens": 12, "latency_ms": 1.0}
        context = {
            "semantic_results": {"issues": [{"title": "Null check", "line": 42}]},
            "verification_results": {"violations": [{"title": "Null check", "line": 42}]},
        }

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=response)) as mock_llm:
            result = await agent.analyze("value = item.name", context)

        assert result.success is True
        assert result.data["findings"] == [
            {"id": "f1", "title": "Null check", "severity": "medium"}
        ]
        prompt = mock_llm.await_args.kwargs["user_prompt"]
        assert "## Semantic Analysis Results" in prompt
        assert "## Formal Verification Results" in prompt

    @pytest.mark.asyncio
    async def test_generates_summary(self, agent):
        """A passing synthesis preserves the structured summary schema."""
        payload = {
            "summary": {
                "total_issues": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "pass": True,
                "recommendation": "Ready to merge",
            },
            "findings": [],
            "github_comment": "No issues found",
        }
        response = {"content": json.dumps(payload), "tokens": 8, "latency_ms": 0.5}

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=response)):
            result = await agent.analyze(
                "def safe(): return True",
                {"semantic_results": {"issues": []}},
            )

        assert result.success is True
        assert result.data["summary"]["pass"] is True
        assert result.data["summary"]["total_issues"] == 0
        assert result.data["findings"] == []
