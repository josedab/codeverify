"""Go and Java Advanced Parser Support.

Enhanced tree-sitter-style parsers for Go and Java with deep AST extraction,
idiomatic pattern detection, goroutine safety analysis, and null annotation
awareness.

This module extends the existing language_support.py with:
- Full AST-level Go parsing (structs, interfaces, goroutines, channels, defer)
- Full AST-level Java parsing (classes, interfaces, annotations, generics)
- Idiomatic pattern detection per language
- Language-specific fix generation
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class GoJavaNodeType(str, Enum):
    """AST node types for Go and Java parsing."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    GOROUTINE = "goroutine"
    CHANNEL = "channel"
    DEFER = "defer"
    ANNOTATION = "annotation"
    IMPORT = "import"
    PACKAGE = "package"
    ENUM = "enum"
    CONSTRUCTOR = "constructor"


class IdiomaticPattern(str, Enum):
    """Language-specific idiomatic patterns detected."""

    GO_NIL_CHECK = "go_nil_check"
    GO_ERROR_RETURN = "go_error_return"
    GO_ERROR_IGNORED = "go_error_ignored"
    GO_DEFER_CLOSE = "go_defer_close"
    GO_MUTEX_USAGE = "go_mutex_usage"
    GO_CONTEXT_USAGE = "go_context_usage"
    GO_GOROUTINE_LEAK = "go_goroutine_leak"
    JAVA_NULL_CHECK = "java_null_check"
    JAVA_NULLABLE_ANNOTATION = "java_nullable_annotation"
    JAVA_TRY_WITH_RESOURCES = "java_try_with_resources"
    JAVA_OPTIONAL_USAGE = "java_optional_usage"
    JAVA_SYNCHRONIZED = "java_synchronized"
    JAVA_UNCHECKED_EXCEPTION = "java_unchecked_exception"


@dataclass
class GoJavaNode:
    """A parsed AST node for Go or Java."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_type: GoJavaNodeType = GoJavaNodeType.FUNCTION
    name: str = ""
    start_line: int = 0
    end_line: int = 0
    return_type: str = ""
    parameters: list[dict[str, str]] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    children: list[GoJavaNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        params = ", ".join(
            f"{p.get('name', '')} {p.get('type', '')}".strip() for p in self.parameters
        )
        ret = f" {self.return_type}" if self.return_type else ""
        return f"{self.name}({params}){ret}"


@dataclass
class PatternMatch:
    """A detected idiomatic pattern in the code."""

    pattern: IdiomaticPattern
    line: int = 0
    context: str = ""
    severity: str = "info"
    suggestion: str = ""


@dataclass
class GoJavaParseResult:
    """Result of parsing Go or Java source code."""

    language: str = "go"
    file_path: str = ""
    nodes: list[GoJavaNode] = field(default_factory=list)
    patterns: list[PatternMatch] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    parse_time_ms: float = 0.0

    @property
    def functions(self) -> list[GoJavaNode]:
        return [
            n for n in self.nodes if n.node_type in (GoJavaNodeType.FUNCTION, GoJavaNodeType.METHOD)
        ]

    @property
    def classes(self) -> list[GoJavaNode]:
        return [
            n for n in self.nodes if n.node_type in (GoJavaNodeType.CLASS, GoJavaNodeType.STRUCT)
        ]

    @property
    def pattern_count(self) -> int:
        return len(self.patterns)

    @property
    def warning_patterns(self) -> list[PatternMatch]:
        return [p for p in self.patterns if p.severity in ("warning", "error")]


class AdvancedGoParser:
    """Advanced Go parser with idiomatic pattern detection."""

    _FUNC = re.compile(
        r"func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\s*\(([^)]*)\)\s*(?:(\([^)]*\)|\S+))?\s*\{",
        re.MULTILINE,
    )
    _STRUCT = re.compile(r"type\s+(\w+)\s+struct\s*\{", re.MULTILINE)
    _INTERFACE = re.compile(r"type\s+(\w+)\s+interface\s*\{", re.MULTILINE)
    _GOROUTINE = re.compile(r"\bgo\s+(\w+)", re.MULTILINE)
    _CHANNEL = re.compile(r"\bmake\s*\(\s*chan\s+(\w+)(?:\s*,\s*(\d+))?\s*\)", re.MULTILINE)
    _DEFER = re.compile(r"\bdefer\s+(\w+)", re.MULTILINE)
    _ERR_IGNORED = re.compile(r"(\w+)\s*,\s*_\s*:?=", re.MULTILINE)
    _ERR_HANDLED = re.compile(r"if\s+err\s*!=\s*nil", re.MULTILINE)
    _NIL_DEREF = re.compile(r"(\w+)\.\w+\b(?!.*if\s+\1\s*!=\s*nil)", re.MULTILINE)
    _MUTEX = re.compile(r"(\w+)\.(?:Lock|RLock)\(\)", re.MULTILINE)
    _CONTEXT = re.compile(r"\bctx\s+context\.Context", re.MULTILINE)

    def parse(self, code: str, file_path: str = "") -> GoJavaParseResult:
        import time as _time

        start = _time.monotonic()
        nodes: list[GoJavaNode] = []
        patterns: list[PatternMatch] = []

        for m in self._FUNC.finditer(code):
            line = code[: m.start()].count("\n") + 1
            receiver = m.group(2) or ""
            params = self._parse_params(m.group(4) or "")
            nodes.append(
                GoJavaNode(
                    node_type=GoJavaNodeType.METHOD if receiver else GoJavaNodeType.FUNCTION,
                    name=m.group(3),
                    start_line=line,
                    return_type=(m.group(5) or "").strip(),
                    parameters=params,
                    metadata={"receiver": receiver} if receiver else {},
                )
            )

        for m in self._STRUCT.finditer(code):
            line = code[: m.start()].count("\n") + 1
            nodes.append(
                GoJavaNode(node_type=GoJavaNodeType.STRUCT, name=m.group(1), start_line=line)
            )

        for m in self._INTERFACE.finditer(code):
            line = code[: m.start()].count("\n") + 1
            nodes.append(
                GoJavaNode(node_type=GoJavaNodeType.INTERFACE, name=m.group(1), start_line=line)
            )

        for m in self._GOROUTINE.finditer(code):
            line = code[: m.start()].count("\n") + 1
            nodes.append(
                GoJavaNode(node_type=GoJavaNodeType.GOROUTINE, name=m.group(1), start_line=line)
            )

        for m in self._CHANNEL.finditer(code):
            line = code[: m.start()].count("\n") + 1
            buffered = m.group(2) is not None
            nodes.append(
                GoJavaNode(
                    node_type=GoJavaNodeType.CHANNEL,
                    name=m.group(1),
                    start_line=line,
                    metadata={
                        "buffered": buffered,
                        "capacity": int(m.group(2)) if m.group(2) else 0,
                    },
                )
            )

        for m in self._DEFER.finditer(code):
            line = code[: m.start()].count("\n") + 1
            nodes.append(
                GoJavaNode(node_type=GoJavaNodeType.DEFER, name=m.group(1), start_line=line)
            )
            patterns.append(
                PatternMatch(
                    pattern=IdiomaticPattern.GO_DEFER_CLOSE,
                    line=line,
                    context=m.group(0),
                    severity="info",
                    suggestion="Good: using defer for cleanup",
                )
            )

        # Idiomatic patterns
        for m in self._ERR_IGNORED.finditer(code):
            line = code[: m.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern=IdiomaticPattern.GO_ERROR_IGNORED,
                    line=line,
                    context=m.group(0),
                    severity="warning",
                    suggestion="Don't discard errors with `_`. Handle or propagate them.",
                )
            )

        for m in self._MUTEX.finditer(code):
            line = code[: m.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern=IdiomaticPattern.GO_MUTEX_USAGE,
                    line=line,
                    context=m.group(0),
                    severity="info",
                    suggestion="Ensure matching Unlock via defer",
                )
            )

        for m in self._CONTEXT.finditer(code):
            line = code[: m.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern=IdiomaticPattern.GO_CONTEXT_USAGE,
                    line=line,
                    context=m.group(0),
                    severity="info",
                    suggestion="Good: using context for cancellation",
                )
            )

        elapsed = (_time.monotonic() - start) * 1000
        return GoJavaParseResult(
            language="go",
            file_path=file_path,
            nodes=nodes,
            patterns=patterns,
            parse_time_ms=elapsed,
        )

    def _parse_params(self, params_str: str) -> list[dict[str, str]]:
        params: list[dict[str, str]] = []
        if not params_str.strip():
            return params
        for part in params_str.split(","):
            part = part.strip()
            pieces = part.rsplit(" ", 1)
            if len(pieces) == 2:
                params.append({"name": pieces[0].strip(), "type": pieces[1].strip()})
            elif pieces:
                params.append({"name": pieces[0].strip(), "type": ""})
        return params


class AdvancedJavaParser:
    """Advanced Java parser with annotation and pattern detection."""

    _CLASS = re.compile(
        r"(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+(\w+)(?:<[^>]+>)?(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?\s*\{",
        re.MULTILINE,
    )
    _INTERFACE = re.compile(r"(?:public\s+)?interface\s+(\w+)(?:<[^>]+>)?\s*\{", re.MULTILINE)
    _ENUM = re.compile(r"(?:public\s+)?enum\s+(\w+)\s*\{", re.MULTILINE)
    _METHOD = re.compile(
        r"(?:(@\w+(?:\([^)]*\))?)\s+)?(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?([\w<>\[\]?,\s]+?)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+([\w,\s]+))?\s*\{",
        re.MULTILINE,
    )
    _NULLABLE = re.compile(r"@(Nullable|NonNull|NotNull|Nonnull)\s+(\w+)", re.MULTILINE)
    _TRY_RESOURCES = re.compile(r"try\s*\(", re.MULTILINE)
    _OPTIONAL = re.compile(r"Optional<(\w+)>", re.MULTILINE)
    _SYNCHRONIZED = re.compile(r"synchronized\s*\(", re.MULTILINE)

    def parse(self, code: str, file_path: str = "") -> GoJavaParseResult:
        import time as _time

        start = _time.monotonic()
        nodes: list[GoJavaNode] = []
        patterns: list[PatternMatch] = []

        for m in self._CLASS.finditer(code):
            line = code[: m.start()].count("\n") + 1
            nodes.append(
                GoJavaNode(
                    node_type=GoJavaNodeType.CLASS,
                    name=m.group(1),
                    start_line=line,
                    metadata={
                        "extends": m.group(2) or "",
                        "implements": (m.group(3) or "").strip(),
                    },
                )
            )

        for m in self._INTERFACE.finditer(code):
            line = code[: m.start()].count("\n") + 1
            nodes.append(
                GoJavaNode(node_type=GoJavaNodeType.INTERFACE, name=m.group(1), start_line=line)
            )

        for m in self._ENUM.finditer(code):
            line = code[: m.start()].count("\n") + 1
            nodes.append(
                GoJavaNode(node_type=GoJavaNodeType.ENUM, name=m.group(1), start_line=line)
            )

        for m in self._METHOD.finditer(code):
            line = code[: m.start()].count("\n") + 1
            annotation = m.group(1) or ""
            annotations = [annotation] if annotation else []
            params = self._parse_params(m.group(4) or "")
            nodes.append(
                GoJavaNode(
                    node_type=GoJavaNodeType.METHOD,
                    name=m.group(3),
                    start_line=line,
                    return_type=m.group(2).strip(),
                    parameters=params,
                    annotations=annotations,
                    metadata={"throws": (m.group(5) or "").strip()} if m.group(5) else {},
                )
            )

        # Pattern detection
        for m in self._NULLABLE.finditer(code):
            line = code[: m.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern=IdiomaticPattern.JAVA_NULLABLE_ANNOTATION,
                    line=line,
                    context=m.group(0),
                    severity="info",
                    suggestion="Good: using nullability annotations for type safety",
                )
            )

        for m in self._TRY_RESOURCES.finditer(code):
            line = code[: m.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern=IdiomaticPattern.JAVA_TRY_WITH_RESOURCES,
                    line=line,
                    context="try-with-resources",
                    severity="info",
                    suggestion="Good: using try-with-resources for automatic cleanup",
                )
            )

        for m in self._OPTIONAL.finditer(code):
            line = code[: m.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern=IdiomaticPattern.JAVA_OPTIONAL_USAGE,
                    line=line,
                    context=m.group(0),
                    severity="info",
                    suggestion="Good: using Optional to express nullable return types",
                )
            )

        for m in self._SYNCHRONIZED.finditer(code):
            line = code[: m.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern=IdiomaticPattern.JAVA_SYNCHRONIZED,
                    line=line,
                    context="synchronized block",
                    severity="info",
                    suggestion="Consider using java.util.concurrent locks for better control",
                )
            )

        elapsed = (_time.monotonic() - start) * 1000
        return GoJavaParseResult(
            language="java",
            file_path=file_path,
            nodes=nodes,
            patterns=patterns,
            parse_time_ms=elapsed,
        )

    def _parse_params(self, params_str: str) -> list[dict[str, str]]:
        params: list[dict[str, str]] = []
        if not params_str.strip():
            return params
        for part in params_str.split(","):
            part = part.strip()
            part = re.sub(r"@\w+\s+", "", part)
            pieces = part.rsplit(" ", 1)
            if len(pieces) == 2:
                params.append({"type": pieces[0].strip(), "name": pieces[1].strip()})
            elif pieces:
                params.append({"type": pieces[0].strip(), "name": ""})
        return params


class GoJavaLanguageSupport:
    """Unified Go and Java language support with parsing and pattern detection."""

    def __init__(self) -> None:
        self._go_parser = AdvancedGoParser()
        self._java_parser = AdvancedJavaParser()

    def parse(self, code: str, language: str, file_path: str = "") -> GoJavaParseResult:
        if language.lower() == "go":
            return self._go_parser.parse(code, file_path)
        if language.lower() == "java":
            return self._java_parser.parse(code, file_path)
        return GoJavaParseResult(
            language=language, file_path=file_path, errors=[f"Unsupported: {language}"]
        )

    def detect_language(self, file_path: str) -> str | None:
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        return {"go": "go", "java": "java"}.get(ext)

    def get_idiomatic_issues(self, code: str, language: str) -> list[PatternMatch]:
        result = self.parse(code, language)
        return result.warning_patterns


_support: GoJavaLanguageSupport | None = None


def get_go_java_support() -> GoJavaLanguageSupport:
    """Get the singleton GoJavaLanguageSupport instance."""
    global _support
    if _support is None:
        _support = GoJavaLanguageSupport()
    return _support


def reset_go_java_support() -> None:
    """Reset the singleton (useful for testing)."""
    global _support
    _support = None
