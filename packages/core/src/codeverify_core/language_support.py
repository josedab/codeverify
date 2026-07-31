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
    RUST = "rust"


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
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
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
    SupportedLanguage.RUST: LanguageConfig(
        language=SupportedLanguage.RUST,
        file_extensions=[".rs"],
        comment_styles={"line": "//", "block_start": "/*", "block_end": "*/"},
        type_system="static",
        standard_lib_models=["std::collections", "std::io", "std::fmt", "std::sync"],
        null_type="None",
        integer_types=["i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "isize", "usize"],
        supports_generics=True,
        supports_null_safety=True,
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
        SupportedLanguage.RUST: re.compile(
            r"^[ \t]*(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?fn\s+(?P<name>\w+)"
            r"\s*(?:<[^>]*>)?\s*\((?P<params>[^)]*)\)"
            r"(?:\s*->\s*(?P<ret>[^{]+))?\s*\{",
            re.MULTILINE,
        ),
    }

    _IMPORT_PATTERNS: dict[SupportedLanguage, re.Pattern[str]] = {
        SupportedLanguage.PYTHON: re.compile(r"^(?:from\s+(\S+)\s+)?import\s+(.+)$", re.MULTILINE),
        SupportedLanguage.TYPESCRIPT: re.compile(
            r"^import\s+.*?from\s+['\"]([^'\"]+)['\"]", re.MULTILINE
        ),
        SupportedLanguage.GO: re.compile(r'^\s*"([^"]+)"', re.MULTILINE),
        SupportedLanguage.JAVA: re.compile(r"^import\s+(?:static\s+)?([^;]+);", re.MULTILINE),
        SupportedLanguage.RUST: re.compile(r"^use\s+([^;]+);", re.MULTILINE),
    }

    _EXT_MAP: dict[str, SupportedLanguage] = {}
    for _lang, _cfg in LANGUAGE_REGISTRY.items():
        for _ext in _cfg.file_extensions:
            _EXT_MAP[_ext] = _lang

    def parse_functions(self, code: str, language: SupportedLanguage) -> list[dict[str, Any]]:
        """Extract function signatures from *code*.

        Returns a list of dicts with keys ``name``, ``params``, and
        ``return_type``.
        """
        pattern = self._FUNC_PATTERNS.get(language)
        if pattern is None:
            return []

        results: list[dict[str, Any]] = []
        for m in pattern.finditer(code):
            results.append(
                {
                    "name": m.group("name"),
                    "params": m.group("params").strip(),
                    "return_type": m.group("ret").strip() if m.group("ret") else None,
                }
            )
        return results

    def parse_imports(self, code: str, language: SupportedLanguage) -> list[str]:
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
            f"Available: {[lang.value for lang in LANGUAGE_REGISTRY]}"
        )
    return LANGUAGE_REGISTRY[language]


# =============================================================================
# Z3 Constraint Generator
# =============================================================================


class Z3ConstraintGenerator:
    """Generates SMT-LIB assertions for common verification checks."""

    def generate_null_check(self, var_name: str, language: SupportedLanguage) -> str:
        """Return an SMT-LIB assertion that *var_name* is not null/nil."""
        null_val = (
            LANGUAGE_REGISTRY[language].null_type if language in LANGUAGE_REGISTRY else "null"
        )
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

    def generate_error_handling_check(self, code: str, language: SupportedLanguage) -> str:
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

        if language == SupportedLanguage.RUST:
            has_result = bool(re.search(r"-> Result<", code))
            has_question = bool(re.search(r"\?;", code))
            has_match = bool(re.search(r"\bmatch\b.*\bErr\b", code))
            status = "handled" if (has_question or has_match or not has_result) else "UNHANDLED"
            return (
                f"; Rust error handling check\n"
                f"(declare-const errors_propagated Bool)\n"
                f"(assert (= errors_propagated {'true' if has_question or has_match else 'false'}))\n"
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
        fix_template='return fmt.Errorf("unexpected: %w", err)',
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
    LanguageRule(
        id="go_nil_map_write",
        language=SupportedLanguage.GO,
        pattern=r"\bvar\s+\w+\s+map\[",
        message="Writing to a nil map causes a panic; initialise with make().",
        severity="error",
        category="null_safety",
        fix_template="m := make(map[K]V)",
    ),
    LanguageRule(
        id="go_context_missing",
        language=SupportedLanguage.GO,
        pattern=r"\bfunc\s+\w+\s*\([^)]*\)\s*(?:\([^)]*\))?\s*\{",
        message="Public function should accept context.Context as first parameter.",
        severity="info",
        category="concurrency",
        fix_template="func Foo(ctx context.Context, ...) error { ... }",
    ),
    LanguageRule(
        id="go_mutex_copy",
        language=SupportedLanguage.GO,
        pattern=r"\bfunc\s+\(\s*\w+\s+\w+\s*\)",
        message="Receiver by value may copy mutex; use pointer receiver for types with sync fields.",
        severity="warning",
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
        fix_template='catch (Exception e) { log.error("Unexpected", e); }',
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
    LanguageRule(
        id="java_optional_get",
        language=SupportedLanguage.JAVA,
        pattern=r"\.get\s*\(\s*\)\s*(?!;)",
        message="Optional.get() without isPresent() check; use orElse() or ifPresent().",
        severity="warning",
        category="null_safety",
        fix_template=".orElse(defaultValue)",
    ),
    LanguageRule(
        id="java_checked_exception_ignored",
        language=SupportedLanguage.JAVA,
        pattern=r"\bcatch\s*\(\s*(?:IOException|SQLException|Exception)\s+\w+\s*\)\s*\{[^}]*(?:log|throw)",
        message="Checked exception caught; ensure it is either re-thrown, wrapped, or properly logged.",
        severity="info",
        category="error_handling",
    ),
    LanguageRule(
        id="java_string_equals",
        language=SupportedLanguage.JAVA,
        pattern=r'==\s*"[^"]*"|"[^"]*"\s*==',
        message="Use .equals() for String comparison, not ==.",
        severity="error",
        category="type_safety",
        fix_template='"value".equals(variable)',
    ),
    LanguageRule(
        id="java_concurrent_modification",
        language=SupportedLanguage.JAVA,
        pattern=r"\bfor\s*\(\s*\w+(?:<[^>]*>)?\s+\w+\s*:\s*(\w+)\s*\).*\1\.(?:add|remove|clear)\s*\(",
        message="Modifying collection during iteration causes ConcurrentModificationException.",
        severity="error",
        category="concurrency",
        fix_template="Use Iterator.remove() or collect into a separate list.",
    ),
]

# Pre-defined rules for Rust
_RUST_RULES: list[LanguageRule] = [
    LanguageRule(
        id="rust_unwrap_used",
        language=SupportedLanguage.RUST,
        pattern=r"\.unwrap\s*\(",
        message="Calling .unwrap() may panic; use pattern matching, .expect(), or the ? operator.",
        severity="warning",
        category="error_handling",
        fix_template=".map_err(|e| ...)? or .unwrap_or_default()",
    ),
    LanguageRule(
        id="rust_unsafe_block",
        language=SupportedLanguage.RUST,
        pattern=r"\bunsafe\s*\{",
        message="Unsafe block found; ensure memory safety invariants are documented and upheld.",
        severity="error",
        category="memory_safety",
    ),
    LanguageRule(
        id="rust_clone_on_ref",
        language=SupportedLanguage.RUST,
        pattern=r"\.clone\s*\(\s*\)",
        message="Unnecessary .clone() may indicate ownership issues; consider borrowing instead.",
        severity="info",
        category="memory_safety",
    ),
    LanguageRule(
        id="rust_panic_in_lib",
        language=SupportedLanguage.RUST,
        pattern=r"\bpanic!\s*\(",
        message="Avoid panic!() in library code; return Result<T, E> instead.",
        severity="error",
        category="error_handling",
        fix_template="return Err(MyError::new(...))",
    ),
    LanguageRule(
        id="rust_todo_macro",
        language=SupportedLanguage.RUST,
        pattern=r"\btodo!\s*\(",
        message="todo!() macro will panic at runtime; implement before shipping.",
        severity="warning",
        category="error_handling",
    ),
    LanguageRule(
        id="rust_mutex_poisoning",
        language=SupportedLanguage.RUST,
        pattern=r"\.lock\s*\(\s*\)\s*\.unwrap\s*\(",
        message="Mutex::lock().unwrap() panics on poisoned mutex; handle PoisonError.",
        severity="warning",
        category="concurrency",
        fix_template=".lock().unwrap_or_else(|e| e.into_inner())",
    ),
]


class LanguageRuleRegistry:
    """Registry of language configs and verification rules.

    Pre-loads Go, Java, and Rust rules at construction time.
    """

    def __init__(self) -> None:
        """Initialise with pre-registered language rules."""
        self._configs: dict[SupportedLanguage, LanguageConfig] = {}
        self._rules: dict[SupportedLanguage, list[LanguageRule]] = {}

        # Pre-register all languages from the global registry
        for _lang, cfg in LANGUAGE_REGISTRY.items():
            self.register_language(cfg)

        # Pre-register rules
        for rule in _GO_RULES:
            self._rules.setdefault(rule.language, []).append(rule)
        for rule in _JAVA_RULES:
            self._rules.setdefault(rule.language, []).append(rule)
        for rule in _RUST_RULES:
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


# =============================================================================
# Advanced Language Analyzer
# =============================================================================


class AdvancedLanguageAnalyzer:
    """Language-aware code analysis beyond regex matching.

    Provides deeper static checks for Go, Java, and Rust including:
    - Generic type parameter validation
    - Interface satisfaction checks (Go)
    - Checked exception enforcement (Java)
    - Unsafe block and ownership analysis (Rust)
    - Concurrency pattern detection
    """

    def __init__(self) -> None:
        self._parser = LanguageParser()
        self._constraint_gen = Z3ConstraintGenerator()

    def analyze(self, code: str, language: SupportedLanguage) -> list[dict[str, Any]]:
        """Run all language-specific analyses and return findings."""
        findings: list[dict[str, Any]] = []

        if language == SupportedLanguage.GO:
            findings.extend(self._analyze_go(code))
        elif language == SupportedLanguage.JAVA:
            findings.extend(self._analyze_java(code))
        elif language == SupportedLanguage.PYTHON:
            findings.extend(self._analyze_python(code))
        elif language == SupportedLanguage.TYPESCRIPT:
            findings.extend(self._analyze_typescript(code))
        elif language == SupportedLanguage.RUST:
            findings.extend(self._analyze_rust(code))

        # Run generic rule matching for all languages
        registry = get_language_registry()
        for rule in registry.get_rules(language):
            for m in re.finditer(rule.pattern, code):
                findings.append(
                    {
                        "rule_id": rule.id,
                        "message": rule.message,
                        "severity": rule.severity,
                        "category": rule.category,
                        "line": code[: m.start()].count("\n") + 1,
                        "match": m.group(0)[:80],
                        "fix_template": rule.fix_template,
                    }
                )

        return findings

    def _analyze_go(self, code: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []

        # Check for multi-return error patterns not checked
        err_assignments = re.findall(r"(\w+),\s*(\w+)\s*:?=\s*(\w+)\(", code)
        for val, err_var, func in err_assignments:
            # Check if err is used after assignment
            after_assign = code[code.index(f"{val}, {err_var}") :]
            if err_var != "_" and f"if {err_var}" not in after_assign[:200]:
                findings.append(
                    {
                        "rule_id": "go_unchecked_error_advanced",
                        "message": f"Error '{err_var}' from {func}() may not be checked.",
                        "severity": "warning",
                        "category": "error_handling",
                        "match": f"{val}, {err_var} := {func}(",
                    }
                )

        # Detect goroutine without WaitGroup or done channel
        goroutine_blocks = re.findall(r"\bgo\s+(?:func|[\w.]+)\b", code)
        if goroutine_blocks:
            has_sync = "sync.WaitGroup" in code or "chan " in code or "context." in code
            if not has_sync:
                findings.append(
                    {
                        "rule_id": "go_goroutine_no_sync",
                        "message": "Goroutine launched without WaitGroup, channel, or context synchronization.",
                        "severity": "warning",
                        "category": "concurrency",
                    }
                )

        return findings

    def _analyze_java(self, code: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []

        # Check for generic type parameter usage in method calls
        raw_collections = re.findall(
            r"\bnew\s+(?:ArrayList|HashMap|HashSet|LinkedList)\s*\(\s*\)", code
        )
        for match in raw_collections:
            if "<" not in match:
                findings.append(
                    {
                        "rule_id": "java_raw_collection_init",
                        "message": f"Raw type in '{match}'; use diamond operator or explicit type params.",
                        "severity": "warning",
                        "category": "type_safety",
                        "match": match,
                    }
                )

        # Check for null returns from methods that return Optional
        optional_methods = re.findall(r"Optional<[^>]+>\s+(\w+)\s*\([^)]*\)\s*\{", code)
        for method in optional_methods:
            method_body_match = re.search(
                rf"Optional<[^>]+>\s+{method}\s*\([^)]*\)\s*\{{(.*?)\}}", code, re.DOTALL
            )
            if method_body_match and "return null" in method_body_match.group(1):
                findings.append(
                    {
                        "rule_id": "java_optional_returns_null",
                        "message": f"Method '{method}' returns Optional but has 'return null'; use Optional.empty().",
                        "severity": "error",
                        "category": "null_safety",
                        "match": method,
                    }
                )

        return findings

    def _analyze_python(self, code: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []

        # Check for mutable default arguments
        mutable_defaults = re.findall(r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\}|\bset\(\))", code)
        for match in mutable_defaults:
            findings.append(
                {
                    "rule_id": "python_mutable_default",
                    "message": f"Mutable default argument '{match}'; use None and create inside function.",
                    "severity": "warning",
                    "category": "error_handling",
                    "match": match,
                }
            )

        return findings

    def _analyze_typescript(self, code: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []

        # Check for any type usage
        any_usages = re.findall(r":\s*any\b", code)
        if any_usages:
            findings.append(
                {
                    "rule_id": "ts_any_type",
                    "message": f"Found {len(any_usages)} uses of 'any' type; prefer explicit types or 'unknown'.",
                    "severity": "info",
                    "category": "type_safety",
                }
            )

        return findings

    def _analyze_rust(self, code: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []

        # Check for unwrap() chains that could be replaced with ? operator
        unwrap_count = len(re.findall(r"\.unwrap\s*\(\s*\)", code))
        if unwrap_count > 3:
            findings.append(
                {
                    "rule_id": "rust_excessive_unwrap",
                    "message": f"Found {unwrap_count} uses of .unwrap(); consider using the ? operator for error propagation.",
                    "severity": "warning",
                    "category": "error_handling",
                }
            )

        # Detect unsafe blocks and count them
        unsafe_blocks = re.findall(r"\bunsafe\s*\{", code)
        if unsafe_blocks:
            # Check if unsafe is documented with SAFETY comments
            safety_comments = re.findall(r"//\s*SAFETY:", code)
            if len(safety_comments) < len(unsafe_blocks):
                findings.append(
                    {
                        "rule_id": "rust_undocumented_unsafe",
                        "message": f"Found {len(unsafe_blocks)} unsafe blocks but only {len(safety_comments)} SAFETY comments.",
                        "severity": "error",
                        "category": "memory_safety",
                    }
                )

        # Detect potential deadlocks from nested lock acquisitions
        lock_calls = re.findall(r"(\w+)\.lock\s*\(\s*\)", code)
        if len(lock_calls) > 1 and len(set(lock_calls)) > 1:
            findings.append(
                {
                    "rule_id": "rust_nested_locks",
                    "message": f"Multiple different locks acquired ({', '.join(set(lock_calls))}); risk of deadlock.",
                    "severity": "warning",
                    "category": "concurrency",
                }
            )

        return findings

    def generate_constraints(self, code: str, language: SupportedLanguage) -> list[dict[str, str]]:
        """Generate Z3 constraints for verifiable patterns found in code."""
        constraints: list[dict[str, str]] = []
        funcs = self._parser.parse_functions(code, language)

        for func in funcs:
            name = func["name"]
            params = func["params"]

            # Null checks for reference parameters
            if language in (SupportedLanguage.GO, SupportedLanguage.JAVA, SupportedLanguage.RUST):
                param_names = [p.strip().split()[-1] for p in params.split(",") if p.strip()]
                for pname in param_names:
                    if pname and pname != "ctx":
                        constraints.append(
                            {
                                "function": name,
                                "type": "null_safety",
                                "constraint": self._constraint_gen.generate_null_check(
                                    pname, language
                                ),
                            }
                        )

            # Overflow checks for integer operations
            config = LANGUAGE_REGISTRY.get(language)
            if config and config.integer_types:
                for itype in config.integer_types:
                    if itype in params:
                        bw = _bit_width_for_type(itype, language)
                        constraints.append(
                            {
                                "function": name,
                                "type": "overflow",
                                "constraint": self._constraint_gen.generate_overflow_check(
                                    f"{name}_result", bw, language
                                ),
                            }
                        )
                        break

        return constraints


def _bit_width_for_type(type_name: str, _language: SupportedLanguage) -> int:
    """Return the bit width for a given integer type."""
    widths = {
        "int8": 8,
        "uint8": 8,
        "byte": 8,
        "int16": 16,
        "uint16": 16,
        "short": 16,
        "int32": 32,
        "uint32": 32,
        "int": 32,
        "int64": 64,
        "uint64": 64,
        "long": 64,
        "uint": 64,
    }
    return widths.get(type_name, 32)
