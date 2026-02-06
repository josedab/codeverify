"""CodeVerify Language Server - LSP server with streaming verification.

Provides progressive, multi-stage verification of source code:
    1. Pattern Match  (<100ms) - regex-based detection of common issues
    2. AI Analysis    (<2s)    - LLM-based semantic analysis
    3. Formal Verification (<5s) - Z3 SMT solver proof
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class VerificationStage(str, Enum):
    """Stage within the progressive verification pipeline."""

    pattern_match = "pattern_match"
    static_analysis = "static_analysis"
    ai_analysis = "ai_analysis"
    formal_verification = "formal_verification"


class DiagnosticSeverity(int, Enum):
    """LSP-compatible diagnostic severity levels (1-based per spec)."""

    error = 1
    warning = 2
    information = 3
    hint = 4


@dataclass
class StreamingDiagnostic:
    """A single diagnostic emitted during streaming verification.

    Fields follow LSP range conventions (zero-based line/character).
    *is_final* is ``True`` when this diagnostic will not be superseded
    by a later verification stage.
    """

    file_uri: str
    line: int
    character: int
    end_line: int
    end_character: int
    severity: DiagnosticSeverity
    message: str
    source: str = "codeverify"
    code: str | None = None
    stage: VerificationStage = VerificationStage.pattern_match
    is_final: bool = False

    def to_lsp_dict(self) -> dict[str, Any]:
        """Serialize to an LSP-compatible Diagnostic object."""
        diag: dict[str, Any] = {
            "range": {
                "start": {"line": self.line, "character": self.character},
                "end": {"line": self.end_line, "character": self.end_character},
            },
            "severity": self.severity.value,
            "message": self.message,
            "source": self.source,
        }
        if self.code is not None:
            diag["code"] = self.code
        return diag


# -- Language-specific pattern rules ----------------------------------------

_PatternRule = tuple[str, str, str, DiagnosticSeverity]

PYTHON_PATTERNS: list[_PatternRule] = [
    (
        r"except\s*:",
        "CV001",
        "Bare except clause catches all exceptions including KeyboardInterrupt and SystemExit",
        DiagnosticSeverity.warning,
    ),
    (
        r"def\s+\w+\s*\([^)]*(?::\s*(?:list|dict|set)\s*=\s*(?:\[\]|\{\}))",
        "CV002",
        "Mutable default argument: default mutable values are shared across calls",
        DiagnosticSeverity.warning,
    ),
    (
        r"['\"][^'\"]*\{[^}]+\}[^'\"]*['\"](?!\s*\.\s*format)",
        "CV003",
        "String contains braces but is not an f-string; prefix with 'f' if interpolation is intended",
        DiagnosticSeverity.hint,
    ),
    (
        r"==\s*None|None\s*==",
        "CV004",
        "Use 'is None' instead of '== None' for identity comparison",
        DiagnosticSeverity.information,
    ),
]

TYPESCRIPT_PATTERNS: list[_PatternRule] = [
    (
        r":\s*any\b",
        "CV100",
        "Avoid using 'any' type; prefer a specific type or 'unknown'",
        DiagnosticSeverity.warning,
    ),
    (
        r"\w+\s*!\s*\.",
        "CV101",
        "Non-null assertion operator (!) bypasses type safety; use proper null checks",
        DiagnosticSeverity.warning,
    ),
    (
        r"[^=!]==[^=]",
        "CV102",
        "Use strict equality (===) instead of loose equality (==)",
        DiagnosticSeverity.information,
    ),
]

GO_PATTERNS: list[_PatternRule] = [
    (
        r"\w+,\s*_\s*(?::=|=)\s*\w+\(",
        "CV200",
        "Error return value is discarded; handle or explicitly ignore with a comment",
        DiagnosticSeverity.warning,
    ),
    (
        r"if\s+\w+\s*!=\s*nil\s*\{[^}]*\}\s*\n\s*\w+\.\w+",
        "CV201",
        "Potential nil pointer dereference after nil check; verify control flow",
        DiagnosticSeverity.error,
    ),
]

JAVA_PATTERNS: list[_PatternRule] = [
    (
        r"\b(?:List|Map|Set|Collection|ArrayList|HashMap|HashSet)\s*[^<\s]",
        "CV300",
        "Raw type usage; add generic type parameter for type safety",
        DiagnosticSeverity.warning,
    ),
    (
        r"catch\s*\([^)]+\)\s*\{\s*\}",
        "CV301",
        "Empty catch block silently swallows exceptions",
        DiagnosticSeverity.warning,
    ),
]

_LANGUAGE_PATTERNS: dict[str, list[_PatternRule]] = {
    "python": PYTHON_PATTERNS,
    "typescript": TYPESCRIPT_PATTERNS,
    "javascript": TYPESCRIPT_PATTERNS,
    "go": GO_PATTERNS,
    "java": JAVA_PATTERNS,
}


class ProgressiveVerificationPipeline:
    """Three-stage verification pipeline that yields diagnostics progressively.

    Cancellation is supported per-document via :meth:`cancel`.
    """

    def __init__(self) -> None:
        self._active_tasks: dict[str, bool] = {}

    async def verify_document(
        self, uri: str, content: str, language: str,
    ) -> AsyncGenerator[list[StreamingDiagnostic], None]:
        """Run the full pipeline, yielding a diagnostic list per stage."""
        self._active_tasks[uri] = True
        logger.info("verification.started", uri=uri, language=language)

        try:
            # Stage 1 - Pattern Match (<100ms)
            pattern_diags = self._run_pattern_stage(uri, content, language)
            if not self._active_tasks.get(uri, False):
                return
            if pattern_diags:
                yield pattern_diags

            # Stage 2 - AI Analysis (<2s)
            ai_diags = await self._run_ai_stage(uri, content, language)
            if not self._active_tasks.get(uri, False):
                return
            if ai_diags:
                yield ai_diags

            # Stage 3 - Formal Verification (<5s)
            formal_diags = await self._run_formal_stage(uri, content, language)
            if not self._active_tasks.get(uri, False):
                return
            if formal_diags:
                yield formal_diags

            logger.info("verification.completed", uri=uri)
        finally:
            self._active_tasks.pop(uri, None)

    def cancel(self, uri: str) -> None:
        """Cancel any in-progress verification for *uri*."""
        if uri in self._active_tasks:
            self._active_tasks[uri] = False
            logger.info("verification.cancelled", uri=uri)

    # -- internal stages ---------------------------------------------------

    def _run_pattern_stage(
        self, uri: str, content: str, language: str,
    ) -> list[StreamingDiagnostic]:
        """Stage 1: regex-based pattern matching (synchronous, <100ms)."""
        start = time.monotonic()
        diagnostics: list[StreamingDiagnostic] = []
        rules = _LANGUAGE_PATTERNS.get(language, [])
        lines = content.splitlines()

        for pattern, code, message, severity in rules:
            for line_idx, line_text in enumerate(lines):
                for match in re.finditer(pattern, line_text):
                    diagnostics.append(StreamingDiagnostic(
                        file_uri=uri,
                        line=line_idx,
                        character=match.start(),
                        end_line=line_idx,
                        end_character=match.end(),
                        severity=severity,
                        message=message,
                        code=code,
                        stage=VerificationStage.pattern_match,
                        is_final=False,
                    ))

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug("stage.pattern_match.done", uri=uri, count=len(diagnostics),
                      elapsed_ms=round(elapsed_ms, 2))
        return diagnostics

    async def _run_ai_stage(
        self, uri: str, content: str, language: str,
    ) -> list[StreamingDiagnostic]:
        """Stage 2: LLM-based semantic analysis (async, <2s).

        The current implementation is a stub that simulates latency and
        returns heuristic-based diagnostics for demonstration purposes.
        """
        start = time.monotonic()
        diagnostics: list[StreamingDiagnostic] = []
        await asyncio.sleep(0.05)

        lines = content.splitlines()
        func_start: int | None = None
        func_name: str = ""

        for idx, line in enumerate(lines):
            func_match = re.match(r"^(\s*)(?:async\s+)?def\s+(\w+)", line)
            if func_match:
                if func_start is not None and (idx - func_start) > 50:
                    diagnostics.append(StreamingDiagnostic(
                        file_uri=uri,
                        line=func_start,
                        character=0,
                        end_line=idx - 1,
                        end_character=len(lines[idx - 1]) if idx > 0 else 0,
                        severity=DiagnosticSeverity.information,
                        message=(
                            f"Function '{func_name}' is {idx - func_start} lines long; "
                            "consider refactoring into smaller units"
                        ),
                        code="CV-AI-001",
                        stage=VerificationStage.ai_analysis,
                        is_final=False,
                    ))
                func_start = idx
                func_name = func_match.group(2)

        # Check final function
        if func_start is not None and (len(lines) - func_start) > 50:
            diagnostics.append(StreamingDiagnostic(
                file_uri=uri,
                line=func_start,
                character=0,
                end_line=len(lines) - 1,
                end_character=len(lines[-1]) if lines else 0,
                severity=DiagnosticSeverity.information,
                message=(
                    f"Function '{func_name}' is {len(lines) - func_start} lines long; "
                    "consider refactoring into smaller units"
                ),
                code="CV-AI-001",
                stage=VerificationStage.ai_analysis,
                is_final=False,
            ))

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug("stage.ai_analysis.done", uri=uri, count=len(diagnostics),
                      elapsed_ms=round(elapsed_ms, 2))
        return diagnostics

    async def _run_formal_stage(
        self, uri: str, content: str, language: str,
    ) -> list[StreamingDiagnostic]:
        """Stage 3: Z3 SMT solver proof (async, <5s).

        The current implementation is a stub that simulates latency and
        emits diagnostics for trivially detectable logical issues.
        """
        start = time.monotonic()
        diagnostics: list[StreamingDiagnostic] = []
        await asyncio.sleep(0.1)

        lines = content.splitlines()
        for idx, line in enumerate(lines):
            # Detect infinite loops (while True without break)
            if re.search(r"\bwhile\s+True\b", line):
                has_break = False
                search_end = min(idx + 20, len(lines))
                for lookahead in range(idx + 1, search_end):
                    if re.search(r"\bbreak\b", lines[lookahead]):
                        has_break = True
                        break
                if not has_break:
                    diagnostics.append(StreamingDiagnostic(
                        file_uri=uri,
                        line=idx, character=0,
                        end_line=idx, end_character=len(line),
                        severity=DiagnosticSeverity.error,
                        message=(
                            "Potential non-termination: 'while True' loop without "
                            "reachable break statement in the next 20 lines"
                        ),
                        code="CV-FV-001",
                        stage=VerificationStage.formal_verification,
                        is_final=True,
                    ))

            # Detect division by zero risk
            div_match = re.search(r"/\s*(\w+)", line)
            if div_match and div_match.group(1) == "0":
                diagnostics.append(StreamingDiagnostic(
                    file_uri=uri,
                    line=idx, character=div_match.start(),
                    end_line=idx, end_character=div_match.end(),
                    severity=DiagnosticSeverity.error,
                    message="Division by zero",
                    code="CV-FV-002",
                    stage=VerificationStage.formal_verification,
                    is_final=True,
                ))

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug("stage.formal_verification.done", uri=uri,
                      count=len(diagnostics), elapsed_ms=round(elapsed_ms, 2))
        return diagnostics


class DebounceManager:
    """Debounces rapid document edits so the verification pipeline is not
    triggered on every keystroke.
    """

    def __init__(self, debounce_ms: int = 500) -> None:
        self.debounce_ms: int = debounce_ms
        self._timers: dict[str, float] = {}

    def should_process(self, uri: str) -> bool:
        """Return True if the debounce window has elapsed for *uri*."""
        last_edit = self._timers.get(uri)
        if last_edit is None:
            return True
        elapsed_ms = (time.monotonic() - last_edit) * 1000
        return elapsed_ms >= self.debounce_ms

    def reset(self, uri: str) -> None:
        """Record the current time as the latest edit for *uri*."""
        self._timers[uri] = time.monotonic()

    def remove(self, uri: str) -> None:
        """Remove the timer for *uri* (e.g. when a document is closed)."""
        self._timers.pop(uri, None)


class CodeVerifyLanguageServer:
    """Simulated Language Server Protocol server for CodeVerify.

    Wires together :class:`ProgressiveVerificationPipeline` and
    :class:`DebounceManager` behind LSP-style handler methods.  Does not
    depend on ``pygls`` so it can be unit-tested independently.
    """

    def __init__(self, debounce_ms: int = 500) -> None:
        self._pipeline = ProgressiveVerificationPipeline()
        self._debouncer = DebounceManager(debounce_ms=debounce_ms)
        self._open_documents: dict[str, str] = {}
        self._diagnostics: dict[str, list[StreamingDiagnostic]] = {}
        self._document_languages: dict[str, str] = {}
        self._initialized: bool = False
        self._shutdown_requested: bool = False

    # -- LSP lifecycle handlers --------------------------------------------

    def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle the ``initialize`` request, returning server capabilities."""
        self._initialized = True
        logger.info(
            "server.initialize",
            client_name=params.get("clientInfo", {}).get("name", "unknown"),
        )
        return {
            "capabilities": {
                "textDocumentSync": {
                    "openClose": True,
                    "change": 1,  # Full document sync
                    "save": {"includeText": True},
                },
                "diagnosticProvider": {
                    "interFileDependencies": False,
                    "workspaceDiagnostics": False,
                },
                "codeActionProvider": True,
            },
            "serverInfo": {"name": "codeverify-lsp", "version": "0.1.0"},
        }

    async def handle_did_open(self, uri: str, text: str, language: str) -> None:
        """Handle ``textDocument/didOpen`` -- store content and verify."""
        self._open_documents[uri] = text
        self._document_languages[uri] = language
        self._diagnostics[uri] = []
        logger.info("document.opened", uri=uri, language=language)
        await self._run_verification(uri)

    async def handle_did_change(self, uri: str, text: str) -> None:
        """Handle ``textDocument/didChange`` -- update, cancel, and re-verify."""
        self._open_documents[uri] = text
        self._pipeline.cancel(uri)
        self._debouncer.reset(uri)
        if self._debouncer.should_process(uri):
            await self._run_verification(uri)

    def handle_did_close(self, uri: str) -> None:
        """Handle ``textDocument/didClose`` -- clean up all document state."""
        self._pipeline.cancel(uri)
        self._open_documents.pop(uri, None)
        self._document_languages.pop(uri, None)
        self._diagnostics.pop(uri, None)
        self._debouncer.remove(uri)
        logger.info("document.closed", uri=uri)

    def handle_shutdown(self) -> None:
        """Handle ``shutdown`` -- cancel all verifications."""
        self._shutdown_requested = True
        for uri in list(self._open_documents):
            self._pipeline.cancel(uri)
        logger.info("server.shutdown")

    # -- diagnostics -------------------------------------------------------

    def _publish_diagnostics(
        self, uri: str, diagnostics: list[StreamingDiagnostic],
    ) -> None:
        """Store diagnostics for *uri* (simulates publishDiagnostics)."""
        if uri not in self._diagnostics:
            self._diagnostics[uri] = []
        self._diagnostics[uri].extend(diagnostics)
        logger.debug("diagnostics.published", uri=uri, count=len(diagnostics),
                      total=len(self._diagnostics[uri]))

    def get_diagnostics(self, uri: str) -> list[StreamingDiagnostic]:
        """Return all diagnostics currently stored for *uri*."""
        return list(self._diagnostics.get(uri, []))

    # -- internal helpers --------------------------------------------------

    async def _run_verification(self, uri: str) -> None:
        """Run the full pipeline and publish results incrementally."""
        content = self._open_documents.get(uri)
        if content is None:
            return
        language = self._document_languages.get(uri, "")
        self._diagnostics[uri] = []
        async for batch in self._pipeline.verify_document(uri, content, language):
            self._publish_diagnostics(uri, batch)

    def _detect_language(self, uri: str) -> str:
        """Infer language from a document URI based on file extension."""
        extension_map: dict[str, str] = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".java": "java",
        }
        for ext, lang in extension_map.items():
            if uri.endswith(ext):
                return lang
        return ""
