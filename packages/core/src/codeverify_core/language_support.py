"""Multi-Language Support for CodeVerify.

Provides language-aware analysis capabilities for formal verification
across multiple programming languages:
- Language detection and feature registry
- Function extraction using regex-based parsing
- Language-specific verification rule sets (null checks, overflow, etc.)
- Rich data for Go (nil, integer types, goroutines) and Java (null, overflow, generics)

This module enables CodeVerify to apply the correct verification strategies
based on the target language's type system and idioms.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# Language Enum
# =============================================================================


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"  # reserved for future support


# =============================================================================
# Language Features
# =============================================================================


@dataclass
class LanguageFeatures:
    """Describes the characteristics and capabilities of a language.

    Captures file extensions, type system traits, null semantics, and
    comment syntax so that downstream analysis can adapt automatically.
    """

    language: Language
    file_extensions: list[str]
    null_type: str
    integer_types: list[str]
    supports_generics: bool
    supports_null_safety: bool
    comment_styles: dict[str, str]

    # Optional extras
    has_pointer_arithmetic: bool = False
    has_goroutines: bool = False
    has_checked_exceptions: bool = False
    is_garbage_collected: bool = True
    default_visibility: str = "public"


# =============================================================================
# LANGUAGE_REGISTRY
# =============================================================================


LANGUAGE_REGISTRY: dict[Language, LanguageFeatures] = {
    Language.PYTHON: LanguageFeatures(
        language=Language.PYTHON,
        file_extensions=[".py", ".pyi", ".pyw"],
        null_type="None",
        integer_types=["int"],
        supports_generics=True,
        supports_null_safety=False,
        comment_styles={
            "line": "#",
            "block_start": '"""',
            "block_end": '"""',
        },
        default_visibility="public",
    ),
    Language.TYPESCRIPT: LanguageFeatures(
        language=Language.TYPESCRIPT,
        file_extensions=[".ts", ".tsx", ".mts", ".cts"],
        null_type="null",
        integer_types=["number"],
        supports_generics=True,
        supports_null_safety=True,
        comment_styles={
            "line": "//",
            "block_start": "/*",
            "block_end": "*/",
        },
    ),
    Language.JAVASCRIPT: LanguageFeatures(
        language=Language.JAVASCRIPT,
        file_extensions=[".js", ".jsx", ".mjs", ".cjs"],
        null_type="null",
        integer_types=["number"],
        supports_generics=False,
        supports_null_safety=False,
        comment_styles={
            "line": "//",
            "block_start": "/*",
            "block_end": "*/",
        },
    ),
    Language.GO: LanguageFeatures(
        language=Language.GO,
        file_extensions=[".go"],
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
        comment_styles={
            "line": "//",
            "block_start": "/*",
            "block_end": "*/",
        },
        has_pointer_arithmetic=False,
        has_goroutines=True,
        is_garbage_collected=True,
        default_visibility="unexported",
    ),
    Language.JAVA: LanguageFeatures(
        language=Language.JAVA,
        file_extensions=[".java"],
        null_type="null",
        integer_types=[
            "byte",
            "short",
            "int",
            "long",
            "Byte",
            "Short",
            "Integer",
            "Long",
        ],
        supports_generics=True,
        supports_null_safety=False,
        comment_styles={
            "line": "//",
            "block_start": "/*",
            "block_end": "*/",
        },
        has_checked_exceptions=True,
        default_visibility="package-private",
    ),
    Language.RUST: LanguageFeatures(
        language=Language.RUST,
        file_extensions=[".rs"],
        null_type="None",
        integer_types=[
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "isize",
            "usize",
        ],
        supports_generics=True,
        supports_null_safety=True,
        comment_styles={
            "line": "//",
            "block_start": "/*",
            "block_end": "*/",
        },
        has_pointer_arithmetic=True,
        is_garbage_collected=False,
        default_visibility="private",
    ),
}


# =============================================================================
# Function Extraction
# =============================================================================


@dataclass
class FunctionInfo:
    """Information about an extracted function definition."""

    name: str
    params: list[dict[str, str | None]]
    return_type: str | None
    body: str
    start_line: int
    end_line: int
    is_async: bool = False

    def param_names(self) -> list[str]:
        """Return just the parameter names."""
        return [p["name"] for p in self.params if p.get("name")]


class FunctionExtractor:
    """Regex-based function extraction for multiple languages.

    Extracts function signatures, bodies, and metadata from source code.
    Each language has its own extraction regex and body-boundary logic.
    """

    # -- Python ----------------------------------------------------------------

    _PYTHON_FUNC_RE = re.compile(
        r"^(?P<indent>[ \t]*)(?P<async>async\s+)?def\s+(?P<name>\w+)"
        r"\s*\((?P<params>[^)]*)\)"
        r"(?:\s*->\s*(?P<ret>[^:]+))?\s*:",
        re.MULTILINE,
    )

    # -- TypeScript / JavaScript -----------------------------------------------

    _TS_FUNC_RE = re.compile(
        r"^[ \t]*(?:export\s+)?(?P<async>async\s+)?function\s+(?P<name>\w+)"
        r"\s*(?:<[^>]*>)?"
        r"\s*\((?P<params>[^)]*)\)"
        r"(?:\s*:\s*(?P<ret>[^{]+))?\s*\{",
        re.MULTILINE,
    )

    # -- Go --------------------------------------------------------------------

    _GO_FUNC_RE = re.compile(
        r"^func\s+(?:\([^)]*\)\s+)?(?P<name>\w+)"
        r"\s*\((?P<params>[^)]*)\)"
        r"(?:\s*(?:\([^)]*\)|(?P<ret>[^\s{]+)))?\s*\{",
        re.MULTILINE,
    )

    # -- Java ------------------------------------------------------------------

    _JAVA_FUNC_RE = re.compile(
        r"^[ \t]*(?:(?:public|private|protected)\s+)?"
        r"(?:(?:static|final|abstract|synchronized)\s+)*"
        r"(?:<[^>]*>\s+)?"
        r"(?P<ret>\w[\w<>\[\],\s?]*?)\s+(?P<name>\w+)"
        r"\s*\((?P<params>[^)]*)\)"
        r"(?:\s*throws\s+[\w,\s]+)?\s*\{",
        re.MULTILINE,
    )

    # --------------------------------------------------------------------------

    def extract_functions(
        self,
        code: str,
        language: Language,
    ) -> list[FunctionInfo]:
        """Extract all functions from *code* written in *language*.

        Returns a list of ``FunctionInfo`` instances sorted by ``start_line``.
        """
        handlers: dict[Language, Any] = {
            Language.PYTHON: self._extract_python,
            Language.TYPESCRIPT: self._extract_typescript,
            Language.JAVASCRIPT: self._extract_typescript,
            Language.GO: self._extract_go,
            Language.JAVA: self._extract_java,
        }

        handler = handlers.get(language)
        if handler is None:
            logger.warning(
                "function_extraction_unsupported",
                language=language.value,
            )
            return []

        functions = handler(code)
        logger.debug(
            "functions_extracted",
            language=language.value,
            count=len(functions),
        )
        return functions

    # -- Python extraction -----------------------------------------------------

    def _extract_python(self, code: str) -> list[FunctionInfo]:
        lines = code.splitlines()
        results: list[FunctionInfo] = []

        for m in self._PYTHON_FUNC_RE.finditer(code):
            start_line = code[: m.start()].count("\n") + 1
            indent_len = len(m.group("indent"))
            body_start = start_line  # body starts on the next line

            # Walk forward to find the end of the indented block
            end_line = body_start
            for idx in range(body_start, len(lines)):
                stripped = lines[idx].strip()
                if not stripped or stripped.startswith("#"):
                    end_line = idx + 1
                    continue
                current_indent = len(lines[idx]) - len(lines[idx].lstrip())
                if current_indent <= indent_len and idx > body_start - 1:
                    break
                end_line = idx + 1

            body = "\n".join(lines[body_start:end_line])

            results.append(FunctionInfo(
                name=m.group("name"),
                params=self._parse_python_params(m.group("params")),
                return_type=m.group("ret").strip() if m.group("ret") else None,
                body=body,
                start_line=start_line,
                end_line=end_line,
                is_async=m.group("async") is not None,
            ))

        return results

    @staticmethod
    def _parse_python_params(raw: str) -> list[dict[str, str | None]]:
        params: list[dict[str, str | None]] = []
        for part in raw.split(","):
            part = part.strip()
            if not part or part in ("*", "/"):
                continue
            part = part.lstrip("*")
            name_type = part.split(":", 1)
            name = name_type[0].split("=")[0].strip()
            type_hint = name_type[1].split("=")[0].strip() if len(name_type) > 1 else None
            if name:
                params.append({"name": name, "type": type_hint})
        return params

    # -- TypeScript / JavaScript extraction ------------------------------------

    def _extract_typescript(self, code: str) -> list[FunctionInfo]:
        lines = code.splitlines()
        results: list[FunctionInfo] = []

        for m in self._TS_FUNC_RE.finditer(code):
            start_line = code[: m.start()].count("\n") + 1
            end_line = self._find_brace_end(lines, start_line - 1)

            body = "\n".join(lines[start_line:end_line])

            results.append(FunctionInfo(
                name=m.group("name"),
                params=self._parse_ts_params(m.group("params")),
                return_type=m.group("ret").strip() if m.group("ret") else None,
                body=body,
                start_line=start_line,
                end_line=end_line,
                is_async=m.group("async") is not None,
            ))

        return results

    @staticmethod
    def _parse_ts_params(raw: str) -> list[dict[str, str | None]]:
        params: list[dict[str, str | None]] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            name_type = part.split(":", 1)
            name = name_type[0].replace("?", "").strip()
            type_hint = name_type[1].strip() if len(name_type) > 1 else None
            if name:
                params.append({"name": name, "type": type_hint})
        return params

    # -- Go extraction ---------------------------------------------------------

    def _extract_go(self, code: str) -> list[FunctionInfo]:
        lines = code.splitlines()
        results: list[FunctionInfo] = []

        for m in self._GO_FUNC_RE.finditer(code):
            start_line = code[: m.start()].count("\n") + 1
            end_line = self._find_brace_end(lines, start_line - 1)

            body = "\n".join(lines[start_line:end_line])

            results.append(FunctionInfo(
                name=m.group("name"),
                params=self._parse_go_params(m.group("params")),
                return_type=m.group("ret").strip() if m.group("ret") else None,
                body=body,
                start_line=start_line,
                end_line=end_line,
                is_async=False,
            ))

        return results

    @staticmethod
    def _parse_go_params(raw: str) -> list[dict[str, str | None]]:
        params: list[dict[str, str | None]] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            tokens = part.split()
            if len(tokens) >= 2:
                params.append({"name": tokens[0], "type": " ".join(tokens[1:])})
            elif len(tokens) == 1:
                # Type-only parameter (common in Go for grouped params)
                params.append({"name": tokens[0], "type": None})
        return params

    # -- Java extraction -------------------------------------------------------

    def _extract_java(self, code: str) -> list[FunctionInfo]:
        lines = code.splitlines()
        results: list[FunctionInfo] = []

        for m in self._JAVA_FUNC_RE.finditer(code):
            start_line = code[: m.start()].count("\n") + 1
            end_line = self._find_brace_end(lines, start_line - 1)

            body = "\n".join(lines[start_line:end_line])

            results.append(FunctionInfo(
                name=m.group("name"),
                params=self._parse_java_params(m.group("params")),
                return_type=m.group("ret").strip() if m.group("ret") else None,
                body=body,
                start_line=start_line,
                end_line=end_line,
                is_async=False,
            ))

        return results

    @staticmethod
    def _parse_java_params(raw: str) -> list[dict[str, str | None]]:
        params: list[dict[str, str | None]] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            # Handle annotations like @NotNull String name
            part = re.sub(r"@\w+\s*", "", part).strip()
            tokens = part.rsplit(None, 1)
            if len(tokens) == 2:
                params.append({"name": tokens[1], "type": tokens[0]})
            elif len(tokens) == 1:
                params.append({"name": tokens[0], "type": None})
        return params

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _find_brace_end(lines: list[str], start_idx: int) -> int:
        """Find the line index of the closing brace that matches the first
        opening brace found at or after *start_idx*."""
        depth = 0
        found_open = False

        for idx in range(start_idx, len(lines)):
            for ch in lines[idx]:
                if ch == "{":
                    depth += 1
                    found_open = True
                elif ch == "}":
                    depth -= 1
                    if found_open and depth == 0:
                        return idx + 1  # 1-based inclusive end

        return len(lines)


# =============================================================================
# Integer Type Info
# =============================================================================


@dataclass
class IntegerTypeInfo:
    """Metadata for an integer type including its overflow boundaries."""

    type_name: str
    bit_width: int
    signed: bool
    min_val: int
    max_val: int


# =============================================================================
# Language-Specific Verification Rules
# =============================================================================


class LanguageSpecificVerificationRules:
    """Provides language-aware patterns and type data for verification.

    Methods return regex patterns, integer boundary info, and idiomatic
    error-handling patterns that the verification engine can use directly.
    """

    # -- Null / nil check patterns -------------------------------------------

    _NULL_CHECK_PATTERNS: dict[Language, list[str]] = {
        Language.PYTHON: [
            r"\bis\s+None\b",
            r"\bis\s+not\s+None\b",
            r"\bif\s+\w+\s*:",
            r"\bif\s+not\s+\w+\s*:",
            r"Optional\[",
        ],
        Language.TYPESCRIPT: [
            r"!==?\s*null\b",
            r"===?\s*null\b",
            r"!==?\s*undefined\b",
            r"===?\s*undefined\b",
            r"\?\.",
            r"\?\?",
            r"\w+\s*!\.",
        ],
        Language.JAVASCRIPT: [
            r"!==?\s*null\b",
            r"===?\s*null\b",
            r"!==?\s*undefined\b",
            r"===?\s*undefined\b",
            r"\?\.",
            r"\?\?",
            r"typeof\s+\w+\s*!==?\s*['\"]undefined['\"]",
        ],
        Language.GO: [
            r"\b\w+\s*!=\s*nil\b",
            r"\b\w+\s*==\s*nil\b",
            r"\bif\s+err\s*!=\s*nil\b",
            r"\bif\s+\w+\s*==\s*nil\b",
            r"\bif\s+\w+\s*!=\s*nil\b",
        ],
        Language.JAVA: [
            r"\b\w+\s*!=\s*null\b",
            r"\b\w+\s*==\s*null\b",
            r"Objects\.requireNonNull\(",
            r"Optional\.",
            r"@NonNull\b",
            r"@Nullable\b",
            r"@NotNull\b",
        ],
    }

    # -- Integer overflow types ----------------------------------------------

    _OVERFLOW_TYPES: dict[Language, list[IntegerTypeInfo]] = {
        Language.PYTHON: [
            # Python integers have arbitrary precision; no overflow risk.
        ],
        Language.TYPESCRIPT: [
            IntegerTypeInfo("number", 64, True, -(2**53) + 1, 2**53 - 1),
        ],
        Language.JAVASCRIPT: [
            IntegerTypeInfo("number", 64, True, -(2**53) + 1, 2**53 - 1),
        ],
        Language.GO: [
            IntegerTypeInfo("int8", 8, True, -128, 127),
            IntegerTypeInfo("int16", 16, True, -32_768, 32_767),
            IntegerTypeInfo("int32", 32, True, -2_147_483_648, 2_147_483_647),
            IntegerTypeInfo("int64", 64, True, -(2**63), 2**63 - 1),
            IntegerTypeInfo("int", 64, True, -(2**63), 2**63 - 1),
            IntegerTypeInfo("uint8", 8, False, 0, 255),
            IntegerTypeInfo("uint16", 16, False, 0, 65_535),
            IntegerTypeInfo("uint32", 32, False, 0, 4_294_967_295),
            IntegerTypeInfo("uint64", 64, False, 0, 2**64 - 1),
            IntegerTypeInfo("uint", 64, False, 0, 2**64 - 1),
        ],
        Language.JAVA: [
            IntegerTypeInfo("byte", 8, True, -128, 127),
            IntegerTypeInfo("Byte", 8, True, -128, 127),
            IntegerTypeInfo("short", 16, True, -32_768, 32_767),
            IntegerTypeInfo("Short", 16, True, -32_768, 32_767),
            IntegerTypeInfo("int", 32, True, -2_147_483_648, 2_147_483_647),
            IntegerTypeInfo("Integer", 32, True, -2_147_483_648, 2_147_483_647),
            IntegerTypeInfo("long", 64, True, -(2**63), 2**63 - 1),
            IntegerTypeInfo("Long", 64, True, -(2**63), 2**63 - 1),
        ],
        Language.RUST: [
            IntegerTypeInfo("i8", 8, True, -128, 127),
            IntegerTypeInfo("i16", 16, True, -32_768, 32_767),
            IntegerTypeInfo("i32", 32, True, -2_147_483_648, 2_147_483_647),
            IntegerTypeInfo("i64", 64, True, -(2**63), 2**63 - 1),
            IntegerTypeInfo("i128", 128, True, -(2**127), 2**127 - 1),
            IntegerTypeInfo("u8", 8, False, 0, 255),
            IntegerTypeInfo("u16", 16, False, 0, 65_535),
            IntegerTypeInfo("u32", 32, False, 0, 4_294_967_295),
            IntegerTypeInfo("u64", 64, False, 0, 2**64 - 1),
            IntegerTypeInfo("u128", 128, False, 0, 2**128 - 1),
            IntegerTypeInfo("isize", 64, True, -(2**63), 2**63 - 1),
            IntegerTypeInfo("usize", 64, False, 0, 2**64 - 1),
        ],
    }

    # -- Array / slice access patterns ---------------------------------------

    _ARRAY_ACCESS_PATTERNS: dict[Language, list[str]] = {
        Language.PYTHON: [
            r"\w+\[[^]]+\]",
            r"\w+\[\s*-?\d+\s*\]",
            r"\w+\[\s*\w+\s*:\s*\w*\s*\]",
        ],
        Language.TYPESCRIPT: [
            r"\w+\[[^]]+\]",
            r"\w+\.at\(\s*-?\d+\s*\)",
        ],
        Language.JAVASCRIPT: [
            r"\w+\[[^]]+\]",
            r"\w+\.at\(\s*-?\d+\s*\)",
        ],
        Language.GO: [
            r"\w+\[[^]]+\]",
            r"\w+\[\s*\w+\s*:\s*\w*\s*\]",
            r"len\(\w+\)",
            r"cap\(\w+\)",
            r"append\(",
        ],
        Language.JAVA: [
            r"\w+\[[^]]+\]",
            r"\w+\.get\(\s*\d+\s*\)",
            r"\w+\.set\(\s*\d+\s*,",
            r"\w+\.size\(\)",
            r"\w+\.length\b",
        ],
    }

    # -- Error handling patterns ---------------------------------------------

    _ERROR_HANDLING_PATTERNS: dict[Language, list[str]] = {
        Language.PYTHON: [
            r"\btry\s*:",
            r"\bexcept\b",
            r"\bexcept\s+\w+",
            r"\braise\b",
            r"\bfinally\s*:",
            r"\bwith\b",
        ],
        Language.TYPESCRIPT: [
            r"\btry\s*\{",
            r"\bcatch\s*\(",
            r"\bfinally\s*\{",
            r"\bthrow\b",
            r"\.catch\(",
            r"\.then\(",
            r"\bPromise\.reject\(",
        ],
        Language.JAVASCRIPT: [
            r"\btry\s*\{",
            r"\bcatch\s*\(",
            r"\bfinally\s*\{",
            r"\bthrow\b",
            r"\.catch\(",
            r"\.then\(",
            r"\bPromise\.reject\(",
        ],
        Language.GO: [
            r"\bif\s+err\s*!=\s*nil\b",
            r"\breturn\b.*\berr\b",
            r"\berrors\.New\(",
            r"\bfmt\.Errorf\(",
            r"\berrors\.Is\(",
            r"\berrors\.As\(",
            r"\berrors\.Wrap\(",
            r"\bdefer\b",
            r"\brecover\(\)",
            r"\bpanic\(",
        ],
        Language.JAVA: [
            r"\btry\s*\{",
            r"\bcatch\s*\(",
            r"\bfinally\s*\{",
            r"\bthrow\s+new\b",
            r"\bthrows\b",
            r"\btry\s*\(\s*\w+",
            r"\.orElseThrow\(",
            r"@SuppressWarnings",
        ],
    }

    # -- Go concurrency patterns ---------------------------------------------

    _GO_CONCURRENCY_PATTERNS: list[str] = [
        r"\bgo\s+\w+",
        r"\bgo\s+func\(",
        r"\bchan\b",
        r"\b<-\s*\w+",
        r"\b\w+\s*<-",
        r"\bselect\s*\{",
        r"\bsync\.Mutex\b",
        r"\bsync\.RWMutex\b",
        r"\bsync\.WaitGroup\b",
        r"\bsync\.Once\b",
        r"\bcontext\.Context\b",
        r"\bcontext\.WithCancel\(",
        r"\bcontext\.WithTimeout\(",
    ]

    # -- public API ----------------------------------------------------------

    def get_null_check_patterns(self, language: Language) -> list[str]:
        """Return regex patterns that detect null/nil checks in *language*."""
        patterns = self._NULL_CHECK_PATTERNS.get(language, [])
        logger.debug(
            "null_check_patterns",
            language=language.value,
            count=len(patterns),
        )
        return patterns

    def get_overflow_risk_types(self, language: Language) -> list[IntegerTypeInfo]:
        """Return integer types with overflow boundaries for *language*."""
        types = self._OVERFLOW_TYPES.get(language, [])
        logger.debug(
            "overflow_risk_types",
            language=language.value,
            count=len(types),
        )
        return types

    def get_array_access_patterns(self, language: Language) -> list[str]:
        """Return regex patterns that detect array/slice accesses."""
        return self._ARRAY_ACCESS_PATTERNS.get(language, [])

    def get_error_handling_patterns(self, language: Language) -> list[str]:
        """Return regex patterns for idiomatic error handling."""
        return self._ERROR_HANDLING_PATTERNS.get(language, [])

    def get_concurrency_patterns(self, language: Language) -> list[str]:
        """Return regex patterns for concurrency primitives.

        Currently provides rich data for Go; returns an empty list for
        languages without specialized concurrency pattern sets.
        """
        if language == Language.GO:
            return list(self._GO_CONCURRENCY_PATTERNS)
        return []

    def has_overflow_risk(self, language: Language) -> bool:
        """Return True when the language has fixed-width integer types."""
        return len(self._OVERFLOW_TYPES.get(language, [])) > 0

    def get_type_info(
        self,
        language: Language,
        type_name: str,
    ) -> IntegerTypeInfo | None:
        """Look up an ``IntegerTypeInfo`` by language and type name."""
        for info in self._OVERFLOW_TYPES.get(language, []):
            if info.type_name == type_name:
                return info
        return None


# =============================================================================
# Language Detection
# =============================================================================

# Extension to Language mapping built from the registry
_EXTENSION_MAP: dict[str, Language] = {}
for _lang, _features in LANGUAGE_REGISTRY.items():
    for _ext in _features.file_extensions:
        _EXTENSION_MAP[_ext] = _lang


def detect_language(file_path: str) -> Language | None:
    """Detect the programming language of a file based on its extension.

    Args:
        file_path: Absolute or relative path to the source file.

    Returns:
        The detected ``Language`` or ``None`` if the extension is not
        recognised.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    language = _EXTENSION_MAP.get(ext)
    if language is not None:
        logger.debug(
            "language_detected",
            file_path=file_path,
            language=language.value,
        )
    else:
        logger.debug(
            "language_detection_failed",
            file_path=file_path,
            extension=ext,
        )
    return language


# =============================================================================
# Feature Accessor
# =============================================================================


def get_language_features(language: Language) -> LanguageFeatures:
    """Retrieve the ``LanguageFeatures`` for a supported language.

    Args:
        language: The target language.

    Returns:
        The corresponding ``LanguageFeatures`` instance.

    Raises:
        KeyError: If *language* is not present in the registry (e.g.
            ``Language.RUST`` before its data is finalised).
    """
    if language not in LANGUAGE_REGISTRY:
        raise KeyError(
            f"Language {language.value!r} is not in the registry. "
            f"Available: {[l.value for l in LANGUAGE_REGISTRY]}"
        )
    return LANGUAGE_REGISTRY[language]
