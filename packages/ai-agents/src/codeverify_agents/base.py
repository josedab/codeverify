"""Base agent class and configuration."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

from codeverify_agents.retry import RetryConfig, async_retry, DEFAULT_LLM_RETRY_CONFIG

logger = structlog.get_logger()


@dataclass
class CodeContext:
    """Context for code analysis, reducing parameter proliferation."""

    code: str
    file_path: str = "unknown"
    language: str = "python"
    diff: str | None = None
    related_code: str | None = None
    framework: str | None = None
    is_ai_generated: bool = False
    author: str | None = None
    verification_results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, code: str, context: dict[str, Any]) -> "CodeContext":
        """Create CodeContext from legacy dict-based context."""
        return cls(
            code=code,
            file_path=context.get("file_path", "unknown"),
            language=context.get("language", "python"),
            diff=context.get("diff"),
            related_code=context.get("related_code"),
            framework=context.get("framework"),
            is_ai_generated=context.get("is_ai_generated", False),
            author=context.get("author"),
            verification_results=context.get("verification_results", {}),
            metadata={k: v for k, v in context.items() if k not in {
                "file_path", "language", "diff", "related_code", "framework",
                "is_ai_generated", "author", "verification_results"
            }},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert back to dict for backward compatibility."""
        result = {
            "file_path": self.file_path,
            "language": self.language,
            "is_ai_generated": self.is_ai_generated,
        }
        if self.diff:
            result["diff"] = self.diff
        if self.related_code:
            result["related_code"] = self.related_code
        if self.framework:
            result["framework"] = self.framework
        if self.author:
            result["author"] = self.author
        if self.verification_results:
            result["verification_results"] = self.verification_results
        result.update(self.metadata)
        return result


@dataclass
class ParsedResponse:
    """Result of parsing an LLM JSON response."""

    data: dict[str, Any]
    parse_error: bool = False
    raw_content: str | None = None


@dataclass
class AgentConfig:
    """Configuration for AI agents."""

    # Provider selection: "openai", "anthropic", or "copilot"
    provider: str = "openai"

    # Model selection
    openai_model: str = "gpt-4-turbo-preview"
    anthropic_model: str = "claude-3-sonnet-20240229"

    # API keys (loaded from environment if not provided)
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Request settings
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        """Load API keys from environment if not provided."""
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")


@dataclass
class AgentResult:
    """Result from an agent analysis."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    tokens_used: int = 0
    latency_ms: float = 0


class BaseAgent(ABC):
    """Base class for all AI agents.
    
    Supports dependency injection of LLM client providers for testing.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize agent with configuration."""
        self.config = config or AgentConfig()
        self._client: Any = None
        self._llm_provider: Any = None  # Injected by AgentFactory

    def _get_openai_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            # Use injected provider if available
            if self._llm_provider is not None:
                self._client = self._llm_provider.get_openai_client(
                    self.config.openai_api_key
                )
            else:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.config.openai_api_key)
        return self._client

    def _get_anthropic_client(self) -> Any:
        """Get or create Anthropic client."""
        if self._client is None:
            # Use injected provider if available
            if self._llm_provider is not None:
                self._client = self._llm_provider.get_anthropic_client(
                    self.config.anthropic_api_key
                )
            else:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    async def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Make a call to OpenAI API."""
        import time

        client = self._get_openai_client()
        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"} if json_mode else None,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0

            logger.info(
                "OpenAI API call completed",
                model=self.config.openai_model,
                tokens=tokens,
                latency_ms=elapsed_ms,
            )

            return {
                "content": content,
                "tokens": tokens,
                "latency_ms": elapsed_ms,
            }

        except Exception as e:
            logger.error("OpenAI API call failed", error=str(e))
            raise

    async def _call_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Make a call to Anthropic API."""
        import time

        client = self._get_anthropic_client()
        start_time = time.time()

        try:
            response = client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            elapsed_ms = (time.time() - start_time) * 1000
            content = response.content[0].text if response.content else ""
            tokens = response.usage.input_tokens + response.usage.output_tokens

            logger.info(
                "Anthropic API call completed",
                model=self.config.anthropic_model,
                tokens=tokens,
                latency_ms=elapsed_ms,
            )

            return {
                "content": content,
                "tokens": tokens,
                "latency_ms": elapsed_ms,
            }

        except Exception as e:
            logger.error("Anthropic API call failed", error=str(e))
            raise

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Call the configured LLM provider with retry logic.
        
        Uses exponential backoff for transient failures.
        """
        return await self._call_llm_with_retry(system_prompt, user_prompt, json_mode)

    @async_retry(config=DEFAULT_LLM_RETRY_CONFIG)
    async def _call_llm_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Internal method with retry decorator applied."""
        if self.config.provider == "anthropic":
            return await self._call_anthropic(system_prompt, user_prompt)
        else:
            return await self._call_openai(system_prompt, user_prompt, json_mode)

    @abstractmethod
    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Analyze code and return results.

        Args:
            code: The code to analyze
            context: Additional context (file path, language, diff, etc.)

        Returns:
            AgentResult with analysis findings
        """
        pass

    def _parse_json_response(
        self,
        response: dict[str, Any],
        fallback_key: str = "raw_response",
    ) -> ParsedResponse:
        """
        Parse JSON content from LLM response with consistent error handling.

        Args:
            response: The response dict from _call_llm containing 'content'
            fallback_key: Key to use when storing raw content on parse failure

        Returns:
            ParsedResponse with parsed data or fallback
        """
        content = response.get("content", "")
        try:
            data = json.loads(content)
            return ParsedResponse(data=data, parse_error=False)
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse JSON response",
                error=str(e),
                content_preview=content[:200] if content else "",
            )
            return ParsedResponse(
                data={fallback_key: content},
                parse_error=True,
                raw_content=content,
            )

    def _build_code_block(
        self,
        code: str,
        language: str,
        label: str | None = None,
        max_length: int | None = None,
    ) -> str:
        """Build a markdown code block for prompts."""
        if max_length and len(code) > max_length:
            code = code[:max_length] + "\n... (truncated)"
        
        lines = []
        if label:
            lines.append(f"{label}:")
            lines.append("")
        lines.append(f"```{language}")
        lines.append(code)
        lines.append("```")
        return "\n".join(lines)
