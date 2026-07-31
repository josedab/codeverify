"""Pluggable Language Adapter Framework.

Provides a unified protocol for adding new language support to CodeVerify.
Each adapter implements parsing, Z3 constraint generation, and agent prompt
templates via a standard interface. Ships with adapters for Python, TypeScript,
Go, Java, and Rust.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

from codeverify_core.language_support import (
    AdvancedLanguageAnalyzer,
    LanguageParser,
    SupportedLanguage,
    Z3ConstraintGenerator,
)

logger = structlog.get_logger()


@dataclass
class AdapterResult:
    """Result from a language adapter analysis."""

    functions: list[dict[str, Any]] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    constraints: list[dict[str, str]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    prompt_context: str = ""


class LanguageAdapter(ABC):
    """Protocol for pluggable language support.

    Subclass this to add a new language to CodeVerify. Each adapter must implement:
    - language: which SupportedLanguage it handles
    - parse(): extract functions and imports
    - generate_constraints(): produce Z3 SMT-LIB assertions
    - agent_prompt_template(): return a prompt template for LLM analysis
    - analyze(): run language-specific static checks
    """

    @property
    @abstractmethod
    def language(self) -> SupportedLanguage:
        """The language this adapter handles."""

    @abstractmethod
    def parse(self, code: str) -> dict[str, Any]:
        """Parse code and return functions and imports."""

    @abstractmethod
    def generate_constraints(self, code: str) -> list[dict[str, str]]:
        """Generate Z3 SMT-LIB constraints for the given code."""

    @abstractmethod
    def agent_prompt_template(self, code: str, context: str = "") -> str:
        """Return a prompt template for LLM-based analysis."""

    @abstractmethod
    def analyze(self, code: str) -> list[dict[str, Any]]:
        """Run language-specific static analysis."""

    def full_analysis(self, code: str, context: str = "") -> AdapterResult:
        """Run all analysis steps and return a consolidated result."""
        parsed = self.parse(code)
        constraints = self.generate_constraints(code)
        findings = self.analyze(code)
        prompt = self.agent_prompt_template(code, context)
        return AdapterResult(
            functions=parsed.get("functions", []),
            imports=parsed.get("imports", []),
            constraints=constraints,
            findings=findings,
            prompt_context=prompt,
        )


class _BaseLanguageAdapter(LanguageAdapter):
    """Common base with shared parser and constraint generator."""

    def __init__(self) -> None:
        self._parser = LanguageParser()
        self._constraint_gen = Z3ConstraintGenerator()
        self._analyzer = AdvancedLanguageAnalyzer()

    def parse(self, code: str) -> dict[str, Any]:
        return {
            "functions": self._parser.parse_functions(code, self.language),
            "imports": self._parser.parse_imports(code, self.language),
        }

    def generate_constraints(self, code: str) -> list[dict[str, str]]:
        return self._analyzer.generate_constraints(code, self.language)

    def analyze(self, code: str) -> list[dict[str, Any]]:
        return self._analyzer.analyze(code, self.language)


class PythonAdapter(_BaseLanguageAdapter):
    """Adapter for Python code analysis."""

    @property
    def language(self) -> SupportedLanguage:
        return SupportedLanguage.PYTHON

    def agent_prompt_template(self, code: str, context: str = "") -> str:
        return (
            "You are analyzing Python code. Focus on:\n"
            "- Type safety (especially Optional/None handling)\n"
            "- Mutable default arguments\n"
            "- Exception handling completeness\n"
            "- Async/await correctness\n\n"
            f"Context: {context}\n\nCode:\n```python\n{code}\n```"
        )


class TypeScriptAdapter(_BaseLanguageAdapter):
    """Adapter for TypeScript code analysis."""

    @property
    def language(self) -> SupportedLanguage:
        return SupportedLanguage.TYPESCRIPT

    def agent_prompt_template(self, code: str, context: str = "") -> str:
        return (
            "You are analyzing TypeScript code. Focus on:\n"
            "- Strict null checks and type narrowing\n"
            "- 'any' type usage\n"
            "- Promise/async error handling\n"
            "- Type assertion safety\n\n"
            f"Context: {context}\n\nCode:\n```typescript\n{code}\n```"
        )


class GoAdapter(_BaseLanguageAdapter):
    """Adapter for Go code analysis."""

    @property
    def language(self) -> SupportedLanguage:
        return SupportedLanguage.GO

    def agent_prompt_template(self, code: str, context: str = "") -> str:
        return (
            "You are analyzing Go code. Focus on:\n"
            "- Error return value handling (never ignore errors)\n"
            "- Nil pointer dereferences\n"
            "- Goroutine leaks and synchronization\n"
            "- Context propagation\n"
            "- Defer correctness in loops\n\n"
            f"Context: {context}\n\nCode:\n```go\n{code}\n```"
        )


class JavaAdapter(_BaseLanguageAdapter):
    """Adapter for Java code analysis."""

    @property
    def language(self) -> SupportedLanguage:
        return SupportedLanguage.JAVA

    def agent_prompt_template(self, code: str, context: str = "") -> str:
        return (
            "You are analyzing Java code. Focus on:\n"
            "- NullPointerException prevention\n"
            "- Resource leaks (use try-with-resources)\n"
            "- Checked exception handling\n"
            "- Thread safety and synchronization\n"
            "- Optional usage patterns\n\n"
            f"Context: {context}\n\nCode:\n```java\n{code}\n```"
        )


class RustAdapter(_BaseLanguageAdapter):
    """Adapter for Rust code analysis."""

    @property
    def language(self) -> SupportedLanguage:
        return SupportedLanguage.RUST

    def agent_prompt_template(self, code: str, context: str = "") -> str:
        return (
            "You are analyzing Rust code. Focus on:\n"
            "- Unsafe block safety invariants\n"
            "- Error handling (prefer ? over .unwrap())\n"
            "- Ownership and borrowing patterns\n"
            "- Mutex poisoning and deadlock risks\n"
            "- Lifetime correctness\n\n"
            f"Context: {context}\n\nCode:\n```rust\n{code}\n```"
        )


# =============================================================================
# Adapter Registry
# =============================================================================


class LanguageAdapterRegistry:
    """Registry for language adapters. Supports dynamic registration."""

    def __init__(self) -> None:
        self._adapters: dict[SupportedLanguage, LanguageAdapter] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        for adapter_cls in (PythonAdapter, TypeScriptAdapter, GoAdapter, JavaAdapter, RustAdapter):
            adapter = adapter_cls()
            self._adapters[adapter.language] = adapter

    def register(self, adapter: LanguageAdapter) -> None:
        """Register a custom language adapter."""
        self._adapters[adapter.language] = adapter
        logger.info("language_adapter_registered", language=adapter.language.value)

    def get(self, language: SupportedLanguage) -> LanguageAdapter | None:
        """Get the adapter for a language."""
        return self._adapters.get(language)

    def get_all(self) -> dict[SupportedLanguage, LanguageAdapter]:
        """Return all registered adapters."""
        return dict(self._adapters)

    def supported_languages(self) -> list[SupportedLanguage]:
        """Return list of languages with registered adapters."""
        return list(self._adapters.keys())

    def analyze_file(
        self, code: str, language: SupportedLanguage, context: str = ""
    ) -> AdapterResult | None:
        """Analyze code using the appropriate adapter."""
        adapter = self.get(language)
        if adapter is None:
            return None
        return adapter.full_analysis(code, context)


# Singleton
_adapter_registry: LanguageAdapterRegistry | None = None


def get_adapter_registry() -> LanguageAdapterRegistry:
    """Get the global language adapter registry."""
    global _adapter_registry
    if _adapter_registry is None:
        _adapter_registry = LanguageAdapterRegistry()
    return _adapter_registry


def reset_adapter_registry() -> None:
    """Reset the global adapter registry (for testing)."""
    global _adapter_registry
    _adapter_registry = None
