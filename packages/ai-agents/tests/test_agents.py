"""Tests for AI agents."""
import pytest
from unittest.mock import Mock, AsyncMock, patch

from codeverify_agents.base import BaseAgent
from codeverify_agents.semantic import SemanticAnalysisAgent
from codeverify_agents.security import SecurityAnalysisAgent
from codeverify_agents.synthesis import SynthesisAgent


class TestBaseAgent:
    """Tests for base agent functionality."""
    
    def test_agent_initialization(self):
        """Agent initializes with provider settings."""
        agent = BaseAgent(provider="openai", model="gpt-4")
        assert agent.provider == "openai"
        assert agent.model == "gpt-4"
    
    def test_agent_default_provider(self):
        """Agent uses default provider if not specified."""
        agent = BaseAgent()
        assert agent.provider in ["openai", "anthropic"]
    
    @pytest.mark.asyncio
    async def test_call_llm_openai(self):
        """Agent can call OpenAI API."""
        agent = BaseAgent(provider="openai", model="gpt-4")
        
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=Mock(
                    choices=[Mock(message=Mock(content="Test response"))]
                )
            )
            mock_openai.return_value = mock_client
            
            response = await agent._call_llm("Test prompt")
            
            assert response == "Test response"
    
    @pytest.mark.asyncio
    async def test_call_llm_anthropic(self):
        """Agent can call Anthropic API."""
        agent = BaseAgent(provider="anthropic", model="claude-3-sonnet")
        
        with patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create = AsyncMock(
                return_value=Mock(
                    content=[Mock(text="Test response")]
                )
            )
            mock_anthropic.return_value = mock_client
            
            response = await agent._call_llm("Test prompt")
            
            assert response == "Test response"


class TestSemanticAnalysisAgent:
    """Tests for semantic analysis agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a semantic analysis agent."""
        return SemanticAnalysisAgent()
    
    def test_agent_type(self, agent):
        """Agent has correct type."""
        assert agent.agent_type == "semantic"
    
    @pytest.mark.asyncio
    async def test_analyze_function(self, agent):
        """Agent can analyze a function."""
        code = """
def calculate_total(items, tax_rate):
    subtotal = sum(item.price for item in items)
    return subtotal * (1 + tax_rate)
"""
        
        with patch.object(agent, "_call_llm") as mock_llm:
            mock_llm.return_value = """
{
    "intent": "Calculate total price with tax",
    "preconditions": ["items is iterable", "tax_rate >= 0"],
    "postconditions": ["return value >= 0"],
    "edge_cases": ["empty items list", "negative tax_rate"],
    "issues": []
}
"""
            result = await agent.analyze(code, {"file_path": "pricing.py"})
            
            assert "intent" in result
            assert "preconditions" in result
    
    @pytest.mark.asyncio
    async def test_handles_malformed_response(self, agent):
        """Agent handles malformed LLM response gracefully."""
        with patch.object(agent, "_call_llm") as mock_llm:
            mock_llm.return_value = "Not valid JSON"
            
            result = await agent.analyze("def foo(): pass", {})
            
            # Should return empty or error result, not crash
            assert isinstance(result, dict)


class TestSecurityAnalysisAgent:
    """Tests for security analysis agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a security analysis agent."""
        return SecurityAnalysisAgent()
    
    def test_agent_type(self, agent):
        """Agent has correct type."""
        assert agent.agent_type == "security"
    
    @pytest.mark.asyncio
    async def test_detect_sql_injection(self, agent):
        """Agent detects SQL injection vulnerability."""
        code = '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
'''
        
        with patch.object(agent, "_call_llm") as mock_llm:
            mock_llm.return_value = """
{
    "vulnerabilities": [
        {
            "type": "sql_injection",
            "severity": "critical",
            "description": "User input used directly in SQL query",
            "line": 3,
            "fix": "Use parameterized queries"
        }
    ]
}
"""
            result = await agent.analyze(code, {"file_path": "db.py"})
            
            assert "vulnerabilities" in result
            assert len(result["vulnerabilities"]) > 0
            assert result["vulnerabilities"][0]["type"] == "sql_injection"
    
    @pytest.mark.asyncio
    async def test_detect_secret_exposure(self, agent):
        """Agent detects hardcoded secrets."""
        code = '''
API_KEY = "sk-proj-abcdef123456"
password = "super_secret_password"
'''
        
        with patch.object(agent, "_call_llm") as mock_llm:
            mock_llm.return_value = """
{
    "vulnerabilities": [
        {
            "type": "hardcoded_secret",
            "severity": "high",
            "description": "API key exposed in source code",
            "line": 1
        }
    ]
}
"""
            result = await agent.analyze(code, {})
            
            assert "vulnerabilities" in result
            assert any(v["type"] == "hardcoded_secret" for v in result["vulnerabilities"])


class TestSynthesisAgent:
    """Tests for synthesis agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a synthesis agent."""
        return SynthesisAgent()
    
    def test_agent_type(self, agent):
        """Agent has correct type."""
        assert agent.agent_type == "synthesis"
    
    @pytest.mark.asyncio
    async def test_consolidate_findings(self, agent):
        """Agent consolidates findings from multiple sources."""
        findings = {
            "semantic": {
                "issues": [
                    {"title": "Missing null check", "severity": "medium"}
                ]
            },
            "security": {
                "vulnerabilities": [
                    {"type": "sql_injection", "severity": "critical"}
                ]
            },
            "formal": {
                "violations": [
                    {"check": "integer_overflow", "severity": "high"}
                ]
            }
        }
        
        with patch.object(agent, "_call_llm") as mock_llm:
            mock_llm.return_value = """
{
    "consolidated_findings": [
        {
            "title": "SQL Injection Vulnerability",
            "severity": "critical",
            "category": "security",
            "confidence": 0.95
        },
        {
            "title": "Potential Integer Overflow",
            "severity": "high",
            "category": "logic_error",
            "confidence": 0.88
        }
    ],
    "summary": "2 critical issues found",
    "recommendation": "Fix SQL injection immediately"
}
"""
            result = await agent.synthesize(findings)
            
            assert "consolidated_findings" in result
            assert len(result["consolidated_findings"]) >= 1
    
    @pytest.mark.asyncio
    async def test_deduplicates_findings(self, agent):
        """Agent removes duplicate findings."""
        findings = {
            "semantic": {
                "issues": [{"title": "Null check", "line": 42}]
            },
            "formal": {
                "violations": [{"title": "Null check", "line": 42}]  # Same issue
            }
        }
        
        with patch.object(agent, "_call_llm") as mock_llm:
            mock_llm.return_value = """
{
    "consolidated_findings": [
        {"title": "Null check issue", "line": 42, "confidence": 0.9}
    ]
}
"""
            result = await agent.synthesize(findings)
            
            # Should deduplicate
            assert len(result.get("consolidated_findings", [])) == 1
    
    @pytest.mark.asyncio
    async def test_generates_summary(self, agent):
        """Agent generates human-readable summary."""
        findings = {"semantic": {"issues": []}}
        
        with patch.object(agent, "_call_llm") as mock_llm:
            mock_llm.return_value = """
{
    "consolidated_findings": [],
    "summary": "No issues found. Code looks good!",
    "pass": true
}
"""
            result = await agent.synthesize(findings)
            
            assert "summary" in result
            assert result.get("pass") is True
