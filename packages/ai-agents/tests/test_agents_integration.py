"""Integration tests for AI agents using MockLLMClientProvider.

These tests verify that agents correctly process LLM responses and produce
expected outputs without requiring actual API calls.
"""

import asyncio
import json
from unittest.mock import patch, MagicMock

import pytest

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent
from codeverify_agents.factory import (
    AgentFactory,
    MockLLMClientProvider,
    set_llm_provider,
    reset_llm_provider,
)
from codeverify_agents.semantic import SemanticAgent
from codeverify_agents.security import SecurityAgent


class TestSemanticAgentIntegration:
    """Integration tests for SemanticAgent."""
    
    def setup_method(self):
        """Set up mock provider before each test."""
        reset_llm_provider()
    
    def teardown_method(self):
        """Reset provider after each test."""
        reset_llm_provider()
    
    @pytest.mark.asyncio
    async def test_analyze_simple_function(self):
        """Test semantic analysis of a simple function."""
        # Mock response simulating GPT-4 output
        mock_response = json.dumps({
            "summary": "A function that adds two numbers",
            "functions": [
                {
                    "name": "add",
                    "purpose": "Adds two integers and returns the result",
                    "preconditions": ["a and b must be integers"],
                    "postconditions": ["returns the sum of a and b"],
                    "assumptions": ["inputs are valid integers"],
                    "edge_cases": ["integer overflow for very large numbers"],
                    "concerns": []
                }
            ],
            "behavioral_changes": [],
            "verification_hints": ["check for integer overflow"]
        })
        
        provider = MockLLMClientProvider(openai_response=mock_response)
        set_llm_provider(provider)
        
        config = AgentConfig(
            openai_api_key="test-key",
            provider="openai",
        )
        agent = SemanticAgent(config)
        agent._llm_provider = provider
        
        code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        context = {
            "file_path": "math_utils.py",
            "language": "python",
        }
        
        result = await agent.analyze(code, context)
        
        assert result.success is True
        assert "functions" in result.data
        assert len(result.data["functions"]) == 1
        assert result.data["functions"][0]["name"] == "add"
        assert "integer overflow" in result.data["functions"][0]["edge_cases"][0]
    
    @pytest.mark.asyncio
    async def test_analyze_with_diff(self):
        """Test semantic analysis with git diff context."""
        mock_response = json.dumps({
            "summary": "Modified validation logic",
            "functions": [
                {
                    "name": "validate_email",
                    "purpose": "Validates email format",
                    "preconditions": ["email must be a string"],
                    "postconditions": ["returns True if valid, False otherwise"],
                    "assumptions": [],
                    "edge_cases": ["empty string", "unicode characters"],
                    "concerns": ["regex may be too permissive"]
                }
            ],
            "behavioral_changes": [
                "Now allows plus signs in email addresses",
                "Changed return type from string to boolean"
            ],
            "verification_hints": ["verify email format regex"]
        })
        
        provider = MockLLMClientProvider(openai_response=mock_response)
        set_llm_provider(provider)
        
        config = AgentConfig(openai_api_key="test-key", provider="openai")
        agent = SemanticAgent(config)
        agent._llm_provider = provider
        
        code = """
def validate_email(email: str) -> bool:
    import re
    pattern = r'^[a-zA-Z0-9+._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
"""
        diff = """
- pattern = r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
+ pattern = r'^[a-zA-Z0-9+._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
"""
        context = {
            "file_path": "validators.py",
            "language": "python",
            "diff": diff,
        }
        
        result = await agent.analyze(code, context)
        
        assert result.success is True
        assert len(result.data["behavioral_changes"]) >= 1
        assert any("plus" in change.lower() for change in result.data["behavioral_changes"])
    
    @pytest.mark.asyncio
    async def test_extract_verification_conditions(self):
        """Test extraction of formal verification conditions."""
        mock_response = json.dumps({
            "conditions": [
                {
                    "id": "vc_1",
                    "type": "overflow_check",
                    "description": "Check if multiplication can overflow",
                    "expression": "a * b",
                    "variables": {
                        "a": {"type": "int32", "range": [0, 1000000]},
                        "b": {"type": "int32", "range": [0, 1000000]}
                    },
                    "location": {"line": 5, "column": 12}
                },
                {
                    "id": "vc_2",
                    "type": "division_check",
                    "description": "Check for division by zero",
                    "expression": "total / count",
                    "variables": {
                        "count": {"type": "int32", "range": None}
                    },
                    "location": {"line": 8, "column": 10}
                }
            ]
        })
        
        provider = MockLLMClientProvider(openai_response=mock_response)
        set_llm_provider(provider)
        
        config = AgentConfig(openai_api_key="test-key", provider="openai")
        agent = SemanticAgent(config)
        agent._llm_provider = provider
        
        code = """
def calculate_average(items: list[int]) -> float:
    total = 0
    for item in items:
        total += item * 2  # Potential overflow
    count = len(items)
    return total / count  # Potential division by zero
"""
        
        result = await agent.extract_verification_conditions(code, "python")
        
        assert "conditions" in result
        assert len(result["conditions"]) == 2
        assert result["conditions"][0]["type"] == "overflow_check"
        assert result["conditions"][1]["type"] == "division_check"


class TestSecurityAgentIntegration:
    """Integration tests for SecurityAgent."""
    
    def setup_method(self):
        reset_llm_provider()
    
    def teardown_method(self):
        reset_llm_provider()
    
    @pytest.mark.asyncio
    async def test_detect_sql_injection(self):
        """Test detection of SQL injection vulnerability."""
        mock_response = json.dumps({
            "vulnerabilities": [
                {
                    "id": "vuln_1",
                    "severity": "critical",
                    "category": "injection",
                    "cwe_id": "CWE-89",
                    "title": "SQL Injection vulnerability",
                    "description": "User input is directly concatenated into SQL query without sanitization",
                    "location": {"file": "db.py", "line": 5, "column": 12},
                    "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
                    "fix_suggestion": "Use parameterized queries instead of string concatenation",
                    "fix_code": "cursor.execute(\"SELECT * FROM users WHERE id = ?\", (user_id,))",
                    "confidence": 0.95
                }
            ],
            "secrets_detected": [],
            "security_score": 25,
            "summary": "Critical SQL injection vulnerability found"
        })
        
        # SecurityAgent uses Anthropic by default
        provider = MockLLMClientProvider(anthropic_response=mock_response)
        set_llm_provider(provider)
        
        config = AgentConfig(
            anthropic_api_key="test-key",
            provider="anthropic",
        )
        agent = SecurityAgent(config)
        agent._llm_provider = provider
        
        code = """
def get_user(user_id: str):
    import sqlite3
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
"""
        context = {
            "file_path": "db.py",
            "language": "python",
        }
        
        result = await agent.analyze(code, context)
        
        assert result.success is True
        assert len(result.data["vulnerabilities"]) == 1
        vuln = result.data["vulnerabilities"][0]
        assert vuln["severity"] == "critical"
        assert vuln["cwe_id"] == "CWE-89"
        assert "injection" in vuln["category"].lower()
    
    @pytest.mark.asyncio
    async def test_detect_multiple_vulnerabilities(self):
        """Test detection of multiple vulnerabilities."""
        mock_response = json.dumps({
            "vulnerabilities": [
                {
                    "id": "vuln_1",
                    "severity": "high",
                    "category": "injection",
                    "cwe_id": "CWE-78",
                    "title": "Command Injection",
                    "description": "User input passed to shell command",
                    "location": {"file": "utils.py", "line": 3},
                    "code_snippet": "os.system(f'ls {path}')",
                    "fix_suggestion": "Use subprocess with shell=False",
                    "confidence": 0.9
                },
                {
                    "id": "vuln_2",
                    "severity": "medium",
                    "category": "path_traversal",
                    "cwe_id": "CWE-22",
                    "title": "Path Traversal",
                    "description": "File path not validated for traversal",
                    "location": {"file": "utils.py", "line": 8},
                    "code_snippet": "open(user_path, 'r')",
                    "fix_suggestion": "Validate and sanitize file paths",
                    "confidence": 0.85
                }
            ],
            "secrets_detected": [],
            "security_score": 45,
            "summary": "Found 2 security issues"
        })
        
        provider = MockLLMClientProvider(anthropic_response=mock_response)
        set_llm_provider(provider)
        
        config = AgentConfig(anthropic_api_key="test-key", provider="anthropic")
        agent = SecurityAgent(config)
        agent._llm_provider = provider
        
        code = """
import os

def list_directory(path: str):
    os.system(f'ls {path}')

def read_file(user_path: str):
    with open(user_path, 'r') as f:
        return f.read()
"""
        context = {"file_path": "utils.py", "language": "python"}
        
        result = await agent.analyze(code, context)
        
        assert result.success is True
        assert len(result.data["vulnerabilities"]) == 2
        severities = [v["severity"] for v in result.data["vulnerabilities"]]
        assert "high" in severities
        assert "medium" in severities
    
    @pytest.mark.asyncio
    async def test_detect_hardcoded_secrets(self):
        """Test detection of hardcoded secrets."""
        mock_response = json.dumps({
            "vulnerabilities": [
                {
                    "id": "vuln_1",
                    "severity": "critical",
                    "category": "secrets",
                    "cwe_id": "CWE-798",
                    "title": "Hardcoded API Key",
                    "description": "API key is hardcoded in source code",
                    "location": {"file": "config.py", "line": 2},
                    "code_snippet": "API_KEY = 'sk-1234567890abcdef'",
                    "fix_suggestion": "Use environment variables for secrets",
                    "confidence": 0.99
                }
            ],
            "secrets_detected": [
                {
                    "type": "openai_key",
                    "location": {"line": 2},
                    "pattern": "sk-..."
                }
            ],
            "security_score": 30,
            "summary": "Hardcoded secret detected"
        })
        
        provider = MockLLMClientProvider(anthropic_response=mock_response)
        set_llm_provider(provider)
        
        config = AgentConfig(anthropic_api_key="test-key", provider="anthropic")
        agent = SecurityAgent(config)
        agent._llm_provider = provider
        
        code = """
# Configuration
API_KEY = 'sk-1234567890abcdef1234567890abcdef'
DB_PASSWORD = 'super_secret_password'
"""
        context = {"file_path": "config.py", "language": "python"}
        
        result = await agent.analyze(code, context)
        
        assert result.success is True
        assert len(result.data["secrets_detected"]) >= 1
    
    def test_scan_for_secrets_pattern_matching(self):
        """Test local pattern-based secret scanning (no LLM)."""
        config = AgentConfig(anthropic_api_key="test-key", provider="anthropic")
        agent = SecurityAgent(config)
        
        code = """
# Various secrets
OPENAI_KEY = 'sk-abcdefghijklmnopqrstuvwxyz123456'
GITHUB_TOKEN = 'ghp_abcdefghijklmnopqrstuvwxyz1234567890'
AWS_ACCESS_KEY = 'AKIAIOSFODNN7EXAMPLE'
PASSWORD = 'my_password_123'
"""
        
        # This method doesn't use LLM, so we can test directly
        secrets = asyncio.get_event_loop().run_until_complete(
            agent.scan_for_secrets(code)
        )
        
        assert len(secrets) >= 2  # At least OpenAI and GitHub tokens
        secret_types = [s["type"] for s in secrets]
        assert "openai_key" in secret_types
        assert "github_token" in secret_types
    
    @pytest.mark.asyncio
    async def test_ai_generated_code_flag(self):
        """Test that AI-generated flag is passed in context."""
        mock_response = json.dumps({
            "vulnerabilities": [],
            "secrets_detected": [],
            "security_score": 85,
            "summary": "No issues found"
        })
        
        provider = MockLLMClientProvider(anthropic_response=mock_response)
        set_llm_provider(provider)
        
        config = AgentConfig(anthropic_api_key="test-key", provider="anthropic")
        agent = SecurityAgent(config)
        agent._llm_provider = provider
        
        code = "def safe_function(): pass"
        context = {
            "file_path": "test.py",
            "language": "python",
            "is_ai_generated": True,  # Flag for AI-generated code
        }
        
        result = await agent.analyze(code, context)
        
        assert result.success is True
        # The agent should handle the flag without errors


class TestAgentFactoryIntegration:
    """Integration tests for AgentFactory."""
    
    def setup_method(self):
        reset_llm_provider()
    
    def teardown_method(self):
        reset_llm_provider()
    
    def test_create_agents_with_mock_provider(self):
        """Test creating agents with mock provider."""
        mock_response = json.dumps({"result": "test"})
        provider = MockLLMClientProvider(
            openai_response=mock_response,
            anthropic_response=mock_response,
        )
        
        factory = AgentFactory(llm_provider=provider)
        
        semantic = factory.create_semantic_agent()
        security = factory.create_security_agent()
        
        assert semantic is not None
        assert security is not None
        assert semantic._llm_provider is provider
        assert security._llm_provider is provider
    
    @pytest.mark.asyncio
    async def test_factory_agents_work_with_mocks(self):
        """Test that factory-created agents work with mock provider."""
        mock_response = json.dumps({
            "summary": "Test analysis",
            "functions": [],
            "behavioral_changes": [],
            "verification_hints": []
        })
        
        provider = MockLLMClientProvider(openai_response=mock_response)
        factory = AgentFactory(llm_provider=provider)
        
        agent = factory.create_semantic_agent()
        
        result = await agent.analyze("def test(): pass", {
            "file_path": "test.py",
            "language": "python"
        })
        
        assert result.success is True
        assert result.data["summary"] == "Test analysis"


class TestAgentErrorHandling:
    """Test error handling in agents."""
    
    def setup_method(self):
        reset_llm_provider()
    
    def teardown_method(self):
        reset_llm_provider()
    
    @pytest.mark.asyncio
    async def test_handles_malformed_json_response(self):
        """Test that agents handle malformed JSON gracefully."""
        # Malformed JSON that can't be parsed
        provider = MockLLMClientProvider(openai_response="not valid json {{{")
        set_llm_provider(provider)
        
        config = AgentConfig(openai_api_key="test-key", provider="openai")
        agent = SemanticAgent(config)
        agent._llm_provider = provider
        
        result = await agent.analyze("def test(): pass", {
            "file_path": "test.py",
            "language": "python"
        })
        
        # Should still succeed but with empty/default data
        assert result.success is True
        # The agent should have handled the parse error
    
    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        """Test that agents handle empty responses."""
        provider = MockLLMClientProvider(openai_response="")
        set_llm_provider(provider)
        
        config = AgentConfig(openai_api_key="test-key", provider="openai")
        agent = SemanticAgent(config)
        agent._llm_provider = provider
        
        result = await agent.analyze("def test(): pass", {
            "file_path": "test.py",
            "language": "python"
        })
        
        # Should handle empty response gracefully
        assert result.success is True


class TestCrossAgentWorkflow:
    """Test workflows involving multiple agents."""
    
    def setup_method(self):
        reset_llm_provider()
    
    def teardown_method(self):
        reset_llm_provider()
    
    @pytest.mark.asyncio
    async def test_semantic_then_security_analysis(self):
        """Test running semantic analysis followed by security analysis."""
        semantic_response = json.dumps({
            "summary": "Database query function",
            "functions": [{
                "name": "query_db",
                "purpose": "Executes database queries",
                "preconditions": ["query must be valid SQL"],
                "postconditions": ["returns query results"],
                "assumptions": ["database connection is valid"],
                "edge_cases": ["empty results"],
                "concerns": ["SQL injection risk"]
            }],
            "behavioral_changes": [],
            "verification_hints": ["validate SQL query input"]
        })
        
        security_response = json.dumps({
            "vulnerabilities": [{
                "id": "vuln_1",
                "severity": "critical",
                "category": "injection",
                "cwe_id": "CWE-89",
                "title": "SQL Injection",
                "description": "Unvalidated input in SQL query",
                "location": {"line": 3},
                "confidence": 0.95
            }],
            "secrets_detected": [],
            "security_score": 30,
            "summary": "Critical vulnerability found"
        })
        
        # Create provider that returns different responses for different clients
        class DualMockProvider(MockLLMClientProvider):
            def __init__(self, openai_resp, anthropic_resp):
                super().__init__(openai_resp, anthropic_resp)
        
        provider = DualMockProvider(semantic_response, security_response)
        
        factory = AgentFactory(llm_provider=provider)
        
        code = """
def query_db(user_input: str):
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return db.execute(query)
"""
        context = {"file_path": "db_utils.py", "language": "python"}
        
        # Run semantic analysis (uses OpenAI)
        semantic_agent = factory.create_semantic_agent()
        semantic_result = await semantic_agent.analyze(code, context)
        
        assert semantic_result.success is True
        assert "SQL injection risk" in semantic_result.data["functions"][0]["concerns"][0]
        
        # Run security analysis (uses Anthropic)
        security_agent = factory.create_security_agent()
        security_result = await security_agent.analyze(code, context)
        
        assert security_result.success is True
        assert len(security_result.data["vulnerabilities"]) == 1
        assert security_result.data["vulnerabilities"][0]["cwe_id"] == "CWE-89"
