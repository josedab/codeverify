"""Multi-Language Support for CodeVerify.

Provides language-aware analysis capabilities for formal verification
across multiple programming languages (Python, TypeScript, Go, Java):
- Language detection and feature registry
- Function and import extraction using regex-based parsing
- Language-specific verification rule sets with pre-registered rules
- SMT-LIB constraint generation for null, bounds, overflow, and error checks
- Singleton-based language rule registry with get/reset accessors

This module enables CodeVerify to apply the correct verification strategies
based on the target language's type system and idioms.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# =============================================================================
# Enums
# =============================================================================


class SupportedLanguage(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"


# Backward-compatible alias used by existing imports
Language = SupportedLanguage


class LanguageFeature(str, Enum):
    """Verification-relevant language features."""

    NULL_SAFETY = "null_safety"
    BOUNDS_CHECK = "bounds_check"
    OVERFLOW = "overflow"
    DIVISION_BY_ZERO = "division_by_zero"
    CONCURRENCY = "concurrency"
    MEMORY_SAFETY = "memory_safety"
    ERROR_HANDLING = "error_handling"
    TYPE_SAFETY = "type_safety"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class LanguageConfig:
    """Describes the characteristics and capabilities of a language."""

    language: SupportedLanguage
    file_extensions: list[str]
    comment_styles: dict[str, str]
    type_system: str
    standard_lib_models: list[str] = field(default_factory=list)

    # Legacy compatibility fields
    null_type: str = "null"
    integer_types: list[str] = field(default_factory=list)
    supports_generics: bool = True
    supports_null_safety: bool = False


# Backward-compatible alias
LanguageFeatures = LanguageConfig


@dataclass
class LanguageRule:
    """A language-specific verification rule."""

    id: str
    language: SupportedLanguage
    pattern: str
    message: str
    severity: str
    category: str
    fix_template: str | None = None


# =============================================================================
# LANGUAGE_REGISTRY (backward-compatible global)
# =============================================================================


LANGUAGE_REGISTRY: dict[SupportedLanguage, LanguageConfig] = {
    SupportedLanguage.PYTHON: LanguageConfig(
        language=SupportedLanguage.PYTHON,
        file_extensions=[".py", ".pyi", ".pyw"],
        comment_styles={"line": "#", "block_start": '"""', "block_end": '"""'},
        type_system="dynamic",
        standard_lib_models=["ast", "typing", "dataclasses"],
        null_type="None",
        integer_types=["int"],
        supports_generics=True,
        supports_null_safety=False,
    ),
    SupportedLanguage.TYPESCRIPT: LanguageConfig(
        language=SupportedLanguage.TYPESCRIPT,
        file_extensions=[".ts", ".tsx", ".mts", ".cts"],
        comment_styles={"line": "//", "block_start": "/*", "block_end": "*/"},
        type_system="static",
        standard_lib_models=["Promise", "Array", "Map", "Set"],
        null_type="null",
        integer_types=["number"],
        supports_generics=True,
        supports_null_safety=True,
    ),
    SupportedLanguage.GO: LanguageConfig(
        language=SupportedLanguage.GO,
        file_extensions=[".go"],
        comment_styles={"line": "//", "block_start": "/*", "block_end": "*/"},
        type_system="static",
        standard_lib_models=["fmt", "errors", "sync", "context", "io"],
        null_type="nil",
        integer_types=[
            "int", "int8", "int16", "int32", "int64",
            "uint", "uint8", "uint16", "uint32", "uint64",
        ],
        supports_generics=True,
        supports_null_safety=False,
    ),
    SupportedLanguage.JAVA: LanguageConfig(
        language=SupportedLanguage.JAVA,
        file_extensions=[".java"],
        comment_styles={"line": "//", "block_start": "/*", "block_end": "*/"},
        type_system="static",
        standard_lib_models=["java.util", "java.io", "java.lang", "java.util.concurrent"],
        null_type="null",
        integer_types=["byte", "short", "int", "long", "Byte", "Short", "Integer", "Long"],
        supports_generics=True,
        supports_null_safety=False,
    ),
}


# =============================================================================
# Language Parser
# =============================================================================


class LanguageParser:
    """Regex-based parser for extracting functions and imports."""

    _FUNC_PATTERNS: dict[SupportedLanguage, re.Pattern[str]] = {
        SupportedLanguage.PYTHON: re.compile(
            r"^[ \t]*(?:async\s+)?def\s+(?P<name>\w+)"
            r"\s*\((?P<params>[^)]*)\)"
            r"(?:\s*->\s*(?P<ret>[^:]+))?\s*:",
            re.MULTILINE,
        ),
        SupportedLanguage.TYPESCRIPT: re.compile(
            r"^[ \t]*(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)"
            r"\s*(?:<[^>]*>)?\s*\((?P<params>[^)]*)\)"
            r"(?:\s*:\s*(?P<ret>[^{]+))?\s*\{",
            re.MULTILINE,
        ),
        SupportedLanguage.GO: re.compile(
            r"^func\s+(?:\([^)]*\)\s+)?(?P<name>\w+)"
            r"\s*\((?P<params>[^)]*)\)"
            r"(?:\s*(?:\([^)]*\)|(?P<ret>[^\s{]+)))?\s*\{",
            re.MULTILINE,
        ),
        SupportedLanguage.JAVA: re.compile(
            r"^[ \t]*(?:(?:public|private|protected)\s+)?"
            r"(?:(?:static|final|abstract|synchronized)\s+)*"
            r"(?:<[^>]*>\s+)?"
            r"(?P<ret>\w[\w<>\[\],\s?]*?)\s+(?P<name>\w+)"
            r"\s*\((?P<params>[^)]*)\)"
            r"(?:\s*throws\s+[\w,\s]+)?\s*\{",
            re.MULTILINE,
        ),
    }

    _IMPORT_PATTERNS: dict[SupportedLanguage, re.Pattern[str]] = {
        SupportedLanguage.PYTHON: re.compile(
            r"^(?:from\s+(\S+)\s+)?import\s+(.+)$", re.MULTILINE
        ),
        SupportedLanguage.TYPESCRIPT: re.compile(
            r"^import\s+.*?from\s+['\"]([^'\"]+)['\"]", re.MULTILINE
        ),
        SupportedLanguage.GO: re.compile(
            r'^\s*"([^"]+)"', re.MULTILINE
        ),
        SupportedLanguage.JAVA: re.compile(
            r"^import\s+(?:static\s+)?([^;]+);", re.MULTILINE
        ),
    }

    _EXT_MAP: dict[str, SupportedLanguage] = {}
    for _lang, _cfg in LANGUAGE_REGISTRY.items():
        for _ext in _cfg.file_extensions:
            _EXT_MAP[_ext] = _lang

    def parse_functions(
        self, code: str, language: SupportedLanguage
    ) -> list[dict[str, Any]]:
        """Extract function signatures from *code*.

        Returns a list of dicts with keys ``name``, ``params``, and
        ``return_type``.
        """
        pattern = self._FUNC_PATTERNS.get(language)
        if pattern is None:
            return []

        results: list[dict[str, Any]] = []
        for m in pattern.finditer(code):
            results.append({
                "name": m.group("name"),
                "params": m.group("params").strip(),
                "return_type": m.group("ret").strip() if m.group("ret") else None,
            })
        return results

    def parse_imports(
        self, code: str, language: SupportedLanguage
    ) -> list[str]:
        """Extract imported module / package names from *code*."""
        pattern = self._IMPORT_PATTERNS.get(language)
        if pattern is None:
            return []

        imports: list[str] = []
        for m in pattern.finditer(code):
            # Use the last non-None group
            for g in reversed(m.groups()):
                if g is not None:
                    imports.append(g.strip())
                    break
        return imports

    def detect_language(self, file_path: str) -> SupportedLanguage | None:
        """Detect the language of *file_path* by extension."""
        _, ext = os.path.splitext(file_path)
        return self._EXT_MAP.get(ext.lower())


# Module-level convenience (backward-compatible)
_EXTENSION_MAP: dict[str, SupportedLanguage] = LanguageParser._EXT_MAP


def detect_language(file_path: str) -> SupportedLanguage | None:  # noqa: F811
    """Detect the programming language of a file based on its extension."""
    _, ext = os.path.splitext(file_path)
    return _EXTENSION_MAP.get(ext.lower())


def get_language_features(language: SupportedLanguage) -> LanguageConfig:
    """Retrieve the ``LanguageConfig`` for a supported language."""
    if language not in LANGUAGE_REGISTRY:
        raise KeyError(
            f"Language {language.value!r} is not in the registry. "
            f"Available: {[l.value for l in LANGUAGE_REGISTRY]}"
        )
    return LANGUAGE_REGISTRY[language]


# =============================================================================
# Z3 Constraint Generator
# =============================================================================


class Z3ConstraintGenerator:
    """Generates SMT-LIB assertions for common verification checks."""

    def generate_null_check(
        self, var_name: str, language: SupportedLanguage
    ) -> str:
        """Return an SMT-LIB assertion that *var_name* is not null/nil."""
        null_val = LANGUAGE_REGISTRY[language].null_type if language in LANGUAGE_REGISTRY else "null"
        return (
            f"; null check for {var_name} ({language.value})\n"
            f"(declare-const {var_name} Int)\n"
            f"(declare-const {var_name}_is_null Bool)\n"
            f"(assert (not {var_name}_is_null))\n"
            f"; {var_name} != {null_val}"
        )

    def generate_bounds_check(
        self, array_var: str, index_var: str, language: SupportedLanguage
    ) -> str:
        """Return an SMT-LIB assertion for array bounds safety."""
        return (
            f"; bounds check for {array_var}[{index_var}] ({language.value})\n"
            f"(declare-const {array_var}_len Int)\n"
            f"(declare-const {index_var} Int)\n"
            f"(assert (>= {index_var} 0))\n"
            f"(assert (< {index_var} {array_var}_len))\n"
            f"(assert (> {array_var}_len 0))\n"
            f"(check-sat)"
        )

    def generate_overflow_check(
        self, expr: str, bit_width: int, language: SupportedLanguage
    ) -> str:
        """Return an SMT-LIB assertion that *expr* fits in *bit_width* bits."""
        if bit_width <= 0:
            bit_width = 32
        max_val = (1 << (bit_width - 1)) - 1
        min_val = -(1 << (bit_width - 1))
        return (
            f"; overflow check for {expr} ({language.value}, {bit_width}-bit)\n"
            f"(declare-const {expr} Int)\n"
            f"(assert (>= {expr} {min_val}))\n"
            f"(assert (<= {expr} {max_val}))\n"
            f"(check-sat)"
        )

    def generate_error_handling_check(
        self, code: str, language: SupportedLanguage
    ) -> str:
        """Return an SMT-LIB comment block for error-handling verification.

        For Go: checks that returned errors are inspected.
        For Java: checks that exceptions are handled.
        """
        if language == SupportedLanguage.GO:
            has_err_check = bool(re.search(r"\bif\s+err\s*!=\s*nil\b", code))
            has_err_return = bool(re.search(r"\breturn\b.*\berr\b", code))
            status = "handled" if (has_err_check or has_err_return) else "UNHANDLED"
            return (
                f"; Go error handling check\n"
                f"(declare-const err_checked Bool)\n"
                f"(assert (= err_checked {'true' if has_err_check else 'false'}))\n"
                f"; status: {status}"
            )

        if language == SupportedLanguage.JAVA:
            has_try = bool(re.search(r"\btry\s*\{", code))
            has_catch = bool(re.search(r"\bcatch\s*\(", code))
            status = "handled" if (has_try and has_catch) else "UNHANDLED"
            return (
                f"; Java exception handling check\n"
                f"(declare-const exceptions_handled Bool)\n"
                f"(assert (= exceptions_handled {'true' if has_catch else 'false'}))\n"
                f"; status: {status}"
            )

        return f"; error handling check not specialised for {language.value}"


# =============================================================================
# Language Rule Registry
# =============================================================================


# Pre-defined rules for Go
_GO_RULES: list[LanguageRule] = [
    LanguageRule(
        id="go_error_ignored",
        language=SupportedLanguage.GO,
        pattern=r"\b\w+,\s*_\s*:?=\s*\w+\(",
        message="Error return value is ignored; check or explicitly discard with a comment.",
        severity="warning",
        category="error_handling",
        fix_template="if err != nil { return err }",
    ),
    LanguageRule(
        id="go_panic_in_lib",
        language=SupportedLanguage.GO,
        pattern=r"\bpanic\s*\(",
        message="Avoid panic() in library code; return an error instead.",
        severity="error",
        category="error_handling",
        fix_template="return fmt.Errorf(\"unexpected: %w\", err)",
    ),
    LanguageRule(
        id="go_unchecked_type_assert",
        language=SupportedLanguage.GO,
        pattern=r"\b\w+\s*:=\s*\w+\.\(\w+\)",
        message="Unchecked type assertion may panic; use the two-value form.",
        severity="warning",
        category="type_safety",
        fix_template="val, ok := x.(T); if !ok { ... }",
    ),
    LanguageRule(
        id="go_defer_in_loop",
        language=SupportedLanguage.GO,
        pattern=r"\bfor\b[^{]*\{[^}]*\bdefer\b",
        message="defer inside a loop may cause resource leaks; move to a helper function.",
        severity="warning",
        category="memory_safety",
    ),
    LanguageRule(
        id="go_goroutine_leak",
        language=SupportedLanguage.GO,
        pattern=r"\bgo\s+func\s*\([^)]*\)\s*\{[^}]*\}(?!\s*\()",
        message="Goroutine may leak; ensure it can be cancelled via context or channel.",
        severity="info",
        category="concurrency",
    ),
]

# Pre-defined rules for Java
_JAVA_RULES: list[LanguageRule] = [
    LanguageRule(
        id="java_empty_catch",
        language=SupportedLanguage.JAVA,
        pattern=r"\bcatch\s*\([^)]+\)\s*\{\s*\}",
        message="Empty catch block silently swallows exceptions.",
        severity="error",
        category="error_handling",
        fix_template="catch (Exception e) { log.error(\"Unexpected\", e); }",
    ),
    LanguageRule(
        id="java_raw_type",
        language=SupportedLanguage.JAVA,
        pattern=r"\bList\s+\w+|Map\s+\w+|Set\s+\w+",
        message="Use of raw type; prefer parameterised generics (e.g. List<String>).",
        severity="warning",
        category="type_safety",
        fix_template="List<String>",
    ),
    LanguageRule(
        id="java_null_deref",
        language=SupportedLanguage.JAVA,
        pattern=r"\b\w+\.(?:get|toString|hashCode)\s*\(",
        message="Potential null dereference; add a null check or use Optional.",
        severity="warning",
        category="null_safety",
        fix_template="Objects.requireNonNull(obj)",
    ),
    LanguageRule(
        id="java_resource_leak",
        language=SupportedLanguage.JAVA,
        pattern=r"\bnew\s+(?:FileInputStream|BufferedReader|Connection)\s*\(",
        message="Resource may leak; use try-with-resources.",
        severity="error",
        category="memory_safety",
        fix_template="try (var res = new Resource()) { ... }",
    ),
    LanguageRule(
        id="java_synchronized_on_non_final",
        language=SupportedLanguage.JAVA,
        pattern=r"\bsynchronized\s*\(\s*(?!this\b|\.class\b)\w+\s*\)",
        message="Synchronizing on a non-final field is error-prone; use a dedicated lock.",
        severity="warning",
        category="concurrency",
    ),
]


class LanguageRuleRegistry:
    """Registry of language configs and verification rules.

    Pre-loads Go and Java rules at construction time.
    """

    def __init__(self) -> None:
        """Initialise with pre-registered language rules."""
        self._configs: dict[SupportedLanguage, LanguageConfig] = {}
        self._rules: dict[SupportedLanguage, list[LanguageRule]] = {}

        # Pre-register all languages from the global registry
        for lang, cfg in LANGUAGE_REGISTRY.items():
            self.register_language(cfg)

        # Pre-register rules
        for rule in _GO_RULES:
            self._rules.setdefault(rule.language, []).append(rule)
        for rule in _JAVA_RULES:
            self._rules.setdefault(rule.language, []).append(rule)

    def register_language(self, config: LanguageConfig) -> None:
        """Register (or replace) a language configuration."""
        self._configs[config.language] = config
        self._rules.setdefault(config.language, [])

    def get_rules(self, language: SupportedLanguage) -> list[LanguageRule]:
        """Return all rules for *language*."""
        return list(self._rules.get(language, []))

    def get_config(self, language: SupportedLanguage) -> LanguageConfig | None:
        """Return the config for *language*, or ``None``."""
        return self._configs.get(language)

    def add_rule(self, rule: LanguageRule) -> None:
        """Add a rule to the registry."""
        self._rules.setdefault(rule.language, []).append(rule)

    def get_all_languages(self) -> list[SupportedLanguage]:
        """Return all registered languages."""
        return list(self._configs.keys())


# =============================================================================
# Module Singletons
# =============================================================================


_language_registry: LanguageRuleRegistry | None = None


def get_language_registry() -> LanguageRuleRegistry:
    """Get the global language rule registry instance."""
    global _language_registry
    if _language_registry is None:
        _language_registry = LanguageRuleRegistry()
    return _language_registry


def reset_language_registry() -> None:
    """Reset the global language rule registry (mainly for testing)."""
    global _language_registry
    _language_registry = None
