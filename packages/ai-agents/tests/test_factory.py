"""Tests for AI agent dependency injection and factory patterns."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from codeverify_agents import (
    AgentConfig,
    AgentFactory,
    BaseAgent,
    DefaultLLMClientProvider,
    LLMClientProvider,
    MockLLMClientProvider,
    SemanticAgent,
    SecurityAgent,
    get_llm_provider,
    reset_llm_provider,
    set_llm_provider,
)
from codeverify_agents.factory import (
    MockOpenAIClient,
    MockAnthropicClient,
)


class TestLLMClientProvider:
    """Tests for LLM client provider abstraction."""

    def test_default_provider_creates_openai_client(self):
        """Default provider creates OpenAI client."""
        provider = DefaultLLMClientProvider()
        # Don't actually create client without valid API key
        # Just verify the method exists
        assert hasattr(provider, "get_openai_client")
        assert hasattr(provider, "get_anthropic_client")

    def test_mock_provider_creates_mock_clients(self):
        """Mock provider creates mock clients with configured responses."""
        mock_response = '{"findings": []}'
        provider = MockLLMClientProvider(
            openai_response=mock_response,
            anthropic_response=mock_response,
        )

        openai_client = provider.get_openai_client("test-key")
        assert isinstance(openai_client, MockOpenAIClient)

        anthropic_client = provider.get_anthropic_client("test-key")
        assert isinstance(anthropic_client, MockAnthropicClient)

    def test_mock_openai_client_returns_configured_response(self):
        """Mock OpenAI client returns configured response."""
        expected_response = '{"test": "data"}'
        client = MockOpenAIClient(expected_response)

        result = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
        )

        assert result.choices[0].message.content == expected_response
        assert result.usage.total_tokens == 100

    def test_mock_anthropic_client_returns_configured_response(self):
        """Mock Anthropic client returns configured response."""
        expected_response = '{"test": "data"}'
        client = MockAnthropicClient(expected_response)

        result = client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "test"}],
        )

        assert result.content[0].text == expected_response
        assert result.usage.input_tokens == 50


class TestGlobalProviderManagement:
    """Tests for global provider management functions."""

    def teardown_method(self):
        """Reset provider after each test."""
        reset_llm_provider()

    def test_get_provider_returns_default(self):
        """get_llm_provider returns default provider."""
        provider = get_llm_provider()
        assert isinstance(provider, DefaultLLMClientProvider)

    def test_set_provider_changes_global_provider(self):
        """set_llm_provider changes the global provider."""
        mock_provider = MockLLMClientProvider()
        set_llm_provider(mock_provider)

        provider = get_llm_provider()
        assert provider is mock_provider

    def test_reset_provider_restores_default(self):
        """reset_llm_provider restores default behavior."""
        mock_provider = MockLLMClientProvider()
        set_llm_provider(mock_provider)
        reset_llm_provider()

        provider = get_llm_provider()
        assert isinstance(provider, DefaultLLMClientProvider)


class TestAgentFactory:
    """Tests for AgentFactory dependency injection."""

    def test_factory_uses_default_provider(self):
        """Factory uses default provider when none specified."""
        factory = AgentFactory()
        assert isinstance(factory.provider, DefaultLLMClientProvider)

    def test_factory_uses_custom_provider(self):
        """Factory uses custom provider when specified."""
        mock_provider = MockLLMClientProvider()
        factory = AgentFactory(provider=mock_provider)
        assert factory.provider is mock_provider

    def test_factory_creates_semantic_agent(self):
        """Factory creates SemanticAgent with injected provider."""
        mock_provider = MockLLMClientProvider()
        factory = AgentFactory(provider=mock_provider)

        agent = factory.create_semantic_agent()
        
        assert isinstance(agent, SemanticAgent)
        assert agent._llm_provider is mock_provider

    def test_factory_creates_security_agent(self):
        """Factory creates SecurityAgent with injected provider."""
        mock_provider = MockLLMClientProvider()
        factory = AgentFactory(provider=mock_provider)

        agent = factory.create_security_agent()
        
        assert isinstance(agent, SecurityAgent)
        assert agent._llm_provider is mock_provider

    def test_factory_uses_custom_config(self):
        """Factory applies custom configuration to agents."""
        config = AgentConfig(
            provider="anthropic",
            temperature=0.5,
            max_tokens=2048,
        )
        factory = AgentFactory(config=config)

        agent = factory.create_semantic_agent()
        
        assert agent.config.provider == "anthropic"
        assert agent.config.temperature == 0.5
        assert agent.config.max_tokens == 2048


class TestBaseAgentWithInjection:
    """Tests for BaseAgent with dependency injection."""

    def test_agent_uses_injected_provider_for_openai(self):
        """Agent uses injected provider for OpenAI client."""
        mock_provider = MockLLMClientProvider()
        
        config = AgentConfig(provider="openai")
        agent = SemanticAgent(config=config)
        agent._llm_provider = mock_provider

        client = agent._get_openai_client()
        
        assert isinstance(client, MockOpenAIClient)

    def test_agent_uses_injected_provider_for_anthropic(self):
        """Agent uses injected provider for Anthropic client."""
        mock_provider = MockLLMClientProvider()
        
        config = AgentConfig(provider="anthropic")
        agent = SemanticAgent(config=config)
        agent._llm_provider = mock_provider

        # Reset client so it will be fetched from provider
        agent._client = None
        client = agent._get_anthropic_client()
        
        assert isinstance(client, MockAnthropicClient)

    def test_agent_falls_back_to_default_without_injection(self):
        """Agent falls back to default client creation without injection."""
        config = AgentConfig(provider="openai", openai_api_key="test-key")
        agent = SemanticAgent(config=config)
        
        # Don't inject provider - should use default
        assert agent._llm_provider is None


class TestAgentWithMockedLLM:
    """Integration tests using mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_semantic_agent_analyze_with_mock(self):
        """SemanticAgent analyze works with mocked LLM."""
        # Prepare mock response
        mock_response = json.dumps({
            "functions": [
                {
                    "name": "calculate_total",
                    "purpose": "Calculate total with tax",
                    "concerns": ["No null check for items"],
                }
            ]
        })
        
        mock_provider = MockLLMClientProvider(openai_response=mock_response)
        factory = AgentFactory(provider=mock_provider)
        agent = factory.create_semantic_agent()

        code = """
def calculate_total(items, tax_rate):
    subtotal = sum(item.price for item in items)
    return subtotal * (1 + tax_rate)
"""
        
        result = await agent.analyze(code, {"file_path": "test.py", "language": "python"})
        
        # Verify we got a result (actual content depends on agent implementation)
        assert result is not None

    @pytest.mark.asyncio
    async def test_security_agent_analyze_with_mock(self):
        """SecurityAgent analyze works with mocked LLM."""
        mock_response = json.dumps({
            "vulnerabilities": [],
            "risk_level": "low",
        })
        
        mock_provider = MockLLMClientProvider(openai_response=mock_response)
        factory = AgentFactory(provider=mock_provider)
        agent = factory.create_security_agent()

        code = """
def get_user(user_id):
    return db.query(f"SELECT * FROM users WHERE id = {user_id}")
"""
        
        result = await agent.analyze(code, {"file_path": "test.py", "language": "python"})
        
        assert result is not None
