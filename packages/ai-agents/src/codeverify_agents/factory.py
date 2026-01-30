"""Dependency injection and factory patterns for AI agents.

This module provides factories for creating AI agent instances and
LLM client providers, enabling easier testing and configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Type, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


class LLMClientProvider(ABC):
    """Abstract provider for LLM clients.
    
    Implement this interface to provide custom LLM clients,
    enabling dependency injection and easier testing.
    """

    @abstractmethod
    def get_openai_client(self, api_key: str) -> Any:
        """Get an OpenAI client instance.
        
        Args:
            api_key: OpenAI API key
            
        Returns:
            Configured OpenAI client
        """
        pass

    @abstractmethod
    def get_anthropic_client(self, api_key: str) -> Any:
        """Get an Anthropic client instance.
        
        Args:
            api_key: Anthropic API key
            
        Returns:
            Configured Anthropic client
        """
        pass


class DefaultLLMClientProvider(LLMClientProvider):
    """Default LLM client provider using official SDKs."""

    def get_openai_client(self, api_key: str) -> Any:
        """Get OpenAI client using official SDK."""
        from openai import OpenAI
        return OpenAI(api_key=api_key)

    def get_anthropic_client(self, api_key: str) -> Any:
        """Get Anthropic client using official SDK."""
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)


class MockLLMClientProvider(LLMClientProvider):
    """Mock LLM client provider for testing.
    
    Returns mock clients that can be configured with
    predetermined responses.
    """

    def __init__(
        self,
        openai_response: str = '{"result": "mock"}',
        anthropic_response: str = '{"result": "mock"}',
    ) -> None:
        """Initialize with mock responses.
        
        Args:
            openai_response: Response content for OpenAI mock
            anthropic_response: Response content for Anthropic mock
        """
        self._openai_response = openai_response
        self._anthropic_response = anthropic_response

    def get_openai_client(self, api_key: str) -> Any:
        """Get mock OpenAI client."""
        return MockOpenAIClient(self._openai_response)

    def get_anthropic_client(self, api_key: str) -> Any:
        """Get mock Anthropic client."""
        return MockAnthropicClient(self._anthropic_response)


@dataclass
class MockChatCompletion:
    """Mock OpenAI chat completion response."""
    content: str
    
    @property
    def choices(self):
        return [type("Choice", (), {"message": type("Message", (), {"content": self.content})()})]
    
    @property
    def usage(self):
        return type("Usage", (), {"total_tokens": 100})()


class MockOpenAIClient:
    """Mock OpenAI client for testing."""

    def __init__(self, response: str) -> None:
        self._response = response

    @property
    def chat(self):
        return type("Chat", (), {"completions": MockCompletions(self._response)})()


class MockCompletions:
    """Mock completions endpoint."""

    def __init__(self, response: str) -> None:
        self._response = response

    def create(self, **kwargs) -> MockChatCompletion:
        return MockChatCompletion(content=self._response)


@dataclass
class MockAnthropicResponse:
    """Mock Anthropic response."""
    content: list
    
    @property
    def usage(self):
        return type("Usage", (), {"input_tokens": 50, "output_tokens": 50})()


class MockAnthropicClient:
    """Mock Anthropic client for testing."""

    def __init__(self, response: str) -> None:
        self._response = response

    @property
    def messages(self):
        return MockMessages(self._response)


class MockMessages:
    """Mock messages endpoint."""

    def __init__(self, response: str) -> None:
        self._response = response

    def create(self, **kwargs) -> MockAnthropicResponse:
        return MockAnthropicResponse(
            content=[type("Content", (), {"text": self._response})()]
        )


# Global provider instance (can be replaced for testing)
_provider: LLMClientProvider | None = None


def get_llm_provider() -> LLMClientProvider:
    """Get the current LLM client provider.
    
    Returns:
        Current LLMClientProvider instance
    """
    global _provider
    if _provider is None:
        _provider = DefaultLLMClientProvider()
    return _provider


def set_llm_provider(provider: LLMClientProvider) -> None:
    """Set the LLM client provider (useful for testing).
    
    Args:
        provider: LLMClientProvider instance to use
    """
    global _provider
    _provider = provider


def reset_llm_provider() -> None:
    """Reset to default LLM client provider."""
    global _provider
    _provider = None


class AgentFactory:
    """Factory for creating agent instances with dependency injection.
    
    Example:
        >>> factory = AgentFactory()
        >>> semantic = factory.create_semantic_agent()
        >>> 
        >>> # For testing
        >>> factory = AgentFactory(provider=MockLLMClientProvider())
        >>> testable_agent = factory.create_semantic_agent()
    """

    def __init__(
        self,
        provider: LLMClientProvider | None = None,
        config: "AgentConfig | None" = None,
    ) -> None:
        """Initialize the factory.
        
        Args:
            provider: LLM client provider (uses default if None)
            config: Default agent configuration
        """
        self._provider = provider
        self._default_config = config

    @property
    def provider(self) -> LLMClientProvider:
        """Get the LLM provider."""
        return self._provider or get_llm_provider()

    def create_semantic_agent(self, config: "AgentConfig | None" = None) -> "SemanticAgent":
        """Create a SemanticAgent instance.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Configured SemanticAgent
        """
        from codeverify_agents.semantic import SemanticAgent
        return self._create_agent(SemanticAgent, config)

    def create_security_agent(self, config: "AgentConfig | None" = None) -> "SecurityAgent":
        """Create a SecurityAgent instance.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Configured SecurityAgent
        """
        from codeverify_agents.security import SecurityAgent
        return self._create_agent(SecurityAgent, config)

    def create_synthesis_agent(self, config: "AgentConfig | None" = None) -> "SynthesisAgent":
        """Create a SynthesisAgent instance.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Configured SynthesisAgent
        """
        from codeverify_agents.synthesis import SynthesisAgent
        return self._create_agent(SynthesisAgent, config)

    def create_trust_score_agent(self, config: "AgentConfig | None" = None) -> "TrustScoreAgent":
        """Create a TrustScoreAgent instance.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Configured TrustScoreAgent
        """
        from codeverify_agents.trust_score import TrustScoreAgent
        return self._create_agent(TrustScoreAgent, config)

    def _create_agent(
        self,
        agent_class: Type[T],
        config: "AgentConfig | None" = None,
    ) -> T:
        """Create an agent instance with the configured provider.
        
        Args:
            agent_class: Agent class to instantiate
            config: Optional configuration
            
        Returns:
            Configured agent instance
        """
        from codeverify_agents.base import AgentConfig
        
        effective_config = config or self._default_config or AgentConfig()
        agent = agent_class(config=effective_config)
        
        # Inject the provider's client factory
        agent._llm_provider = self.provider
        
        return agent


# Type aliases for clarity
SemanticAgent = Any  # Forward reference
SecurityAgent = Any
SynthesisAgent = Any
TrustScoreAgent = Any
AgentConfig = Any
