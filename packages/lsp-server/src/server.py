"""CodeVerify LSP Server - Universal IDE Protocol Implementation.

This module implements the Language Server Protocol for CodeVerify, enabling
single integration point for all IDEs (VS Code, JetBrains, Neovim, Emacs, etc.).

Key features:
1. Standard LSP diagnostics for verification findings
2. Code actions for quick fixes
3. Hover information with verification details
4. Custom commands for verification operations
5. Streaming diagnostics for real-time verification
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

# LSP library imports
try:
    from lsprotocol import types as lsp
    from pygls.server import LanguageServer
    from pygls.workspace import TextDocument
except ImportError:
    raise ImportError(
        "LSP dependencies not installed. Install with: pip install pygls lsprotocol"
    )

import structlog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()


# Server configuration
@dataclass
class ServerConfig:
    """Configuration for the LSP server."""

    api_endpoint: str = "http://localhost:8000"
    api_key: str = ""
    enable_real_time: bool = True
    real_time_delay_ms: int = 500
    enable_formal_verification: bool = True
    enable_ai_analysis: bool = True
    max_file_size_kb: int = 500
    supported_languages: list[str] = field(
        default_factory=lambda: ["python", "typescript", "javascript", "go", "java", "rust"]
    )


# Custom CodeVerify language server
class CodeVerifyLanguageServer(LanguageServer):
    """CodeVerify Language Server implementation."""

    def __init__(self, config: ServerConfig | None = None):
        super().__init__(
            name="codeverify-lsp",
            version="0.1.0",
        )

        self.config = config or ServerConfig()
        self._verification_cache: dict[str, dict[str, Any]] = {}
        self._pending_verifications: dict[str, asyncio.Task] = {}
        self._findings_by_uri: dict[str, list[dict[str, Any]]] = {}

        # Register handlers
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all LSP handlers."""

        @self.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
        async def did_open(params: lsp.DidOpenTextDocumentParams) -> None:
            """Handle document open."""
            await self._analyze_document(params.text_document.uri)

        @self.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
        async def did_save(params: lsp.DidSaveTextDocumentParams) -> None:
            """Handle document save."""
            await self._analyze_document(params.text_document.uri)

        @self.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
        async def did_change(params: lsp.DidChangeTextDocumentParams) -> None:
            """Handle document change - trigger real-time verification."""
            if self.config.enable_real_time:
                await self._schedule_analysis(params.text_document.uri)

        @self.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
        def did_close(params: lsp.DidCloseTextDocumentParams) -> None:
            """Handle document close."""
            uri = params.text_document.uri
            # Clean up
            self._findings_by_uri.pop(uri, None)
            self._verification_cache.pop(uri, None)

        @self.feature(lsp.TEXT_DOCUMENT_CODE_ACTION)
        async def code_action(
            params: lsp.CodeActionParams,
        ) -> list[lsp.CodeAction] | None:
            """Provide code actions for findings."""
            return await self._get_code_actions(params)

        @self.feature(lsp.TEXT_DOCUMENT_HOVER)
        async def hover(params: lsp.HoverParams) -> lsp.Hover | None:
            """Provide hover information."""
            return await self._get_hover(params)

        @self.command("codeverify.analyze")
        async def analyze_command(uri: str) -> dict[str, Any]:
            """Command to trigger analysis."""
            await self._analyze_document(uri, force=True)
            return {"success": True}

        @self.command("codeverify.getTrustScore")
        async def trust_score_command(uri: str) -> dict[str, Any]:
            """Command to get trust score."""
            return await self._get_trust_score(uri)

        @self.command("codeverify.getFindings")
        async def findings_command(uri: str) -> list[dict[str, Any]]:
            """Command to get all findings."""
            return self._findings_by_uri.get(uri, [])

        @self.command("codeverify.applyFix")
        async def apply_fix_command(
            uri: str,
            finding_id: str,
            fix_code: str,
        ) -> dict[str, Any]:
            """Command to apply a fix."""
            return await self._apply_fix(uri, finding_id, fix_code)

        @self.command("codeverify.dismissFinding")
        async def dismiss_finding_command(
            uri: str,
            finding_id: str,
            reason: str,
        ) -> dict[str, Any]:
            """Command to dismiss a finding."""
            return await self._dismiss_finding(uri, finding_id, reason)

        @self.command("codeverify.debugVerification")
        async def debug_command(uri: str, line: int) -> dict[str, Any]:
            """Command to debug verification at line."""
            return await self._debug_verification(uri, line)

    async def _schedule_analysis(self, uri: str) -> None:
        """Schedule analysis with debouncing."""
        # Cancel pending analysis for this URI
        if uri in self._pending_verifications:
            self._pending_verifications[uri].cancel()

        # Schedule new analysis
        async def delayed_analysis():
            await asyncio.sleep(self.config.real_time_delay_ms / 1000)
            await self._analyze_document(uri)

        task = asyncio.create_task(delayed_analysis())
        self._pending_verifications[uri] = task

    async def _analyze_document(self, uri: str, force: bool = False) -> None:
        """Analyze a document and publish diagnostics."""
        # Get document
        document = self.workspace.get_text_document(uri)
        if not document:
            return

        # Check file size
        if len(document.source) > self.config.max_file_size_kb * 1024:
            logger.warning("File too large for analysis", uri=uri)
            return

        # Check language
        language = self._get_language(uri)
        if language not in self.config.supported_languages:
            return

        try:
            # Analyze
            findings = await self._run_analysis(document.source, language, uri)
            self._findings_by_uri[uri] = findings

            # Convert to diagnostics
            diagnostics = self._findings_to_diagnostics(findings)

            # Publish
            self.publish_diagnostics(uri, diagnostics)

            logger.info(
                "Analysis complete",
                uri=uri,
                findings=len(findings),
            )

        except Exception as e:
            logger.error("Analysis failed", uri=uri, error=str(e))

    async def _run_analysis(
        self,
        code: str,
        language: str,
        uri: str,
    ) -> list[dict[str, Any]]:
        """Run verification analysis on code."""
        # Try API first
        if self.config.api_endpoint and self.config.api_key:
            try:
                return await self._api_analysis(code, language, uri)
            except Exception as e:
                logger.warning("API analysis failed, using local", error=str(e))

        # Fall back to local analysis
        return await self._local_analysis(code, language)

    async def _api_analysis(
        self,
        code: str,
        language: str,
        uri: str,
    ) -> list[dict[str, Any]]:
        """Run analysis via API."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.config.api_endpoint}/api/v1/analyze",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "code": code,
                    "language": language,
                    "file_path": self._uri_to_path(uri),
                    "options": {
                        "formal_verification": self.config.enable_formal_verification,
                        "ai_analysis": self.config.enable_ai_analysis,
                    },
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                return response.json().get("findings", [])
            else:
                raise Exception(f"API error: {response.status_code}")

    async def _local_analysis(
        self,
        code: str,
        language: str,
    ) -> list[dict[str, Any]]:
        """Run local analysis (pattern-based)."""
        findings = []

        lines = code.split("\n")

        for i, line in enumerate(lines):
            # Null safety checks
            if language == "python":
                if ".method" in line and "is not None" not in line and "if " not in line:
                    findings.append(self._create_finding(
                        id=str(uuid.uuid4()),
                        title="Potential None dereference",
                        description="This code may raise AttributeError if the object is None",
                        severity="medium",
                        category="null_safety",
                        line_start=i + 1,
                        line_end=i + 1,
                    ))

            elif language in ("typescript", "javascript"):
                if "." in line and "?." not in line and "&&" not in line:
                    # Skip common safe patterns
                    if not any(s in line for s in ["console.", "Math.", "JSON.", "Object."]):
                        findings.append(self._create_finding(
                            id=str(uuid.uuid4()),
                            title="Potential null/undefined dereference",
                            description="Consider using optional chaining (?.) to prevent runtime errors",
                            severity="low",
                            category="null_safety",
                            line_start=i + 1,
                            line_end=i + 1,
                        ))

            # Array bounds checks
            if "[" in line and "]" in line and "len" not in line.lower() and "length" not in line.lower():
                findings.append(self._create_finding(
                    id=str(uuid.uuid4()),
                    title="Array access without bounds check",
                    description="Consider adding bounds checking to prevent IndexError",
                    severity="low",
                    category="array_bounds",
                    line_start=i + 1,
                    line_end=i + 1,
                ))

            # SQL injection
            if "execute" in line.lower() and ("+" in line or "%" in line or "format" in line):
                if "SELECT" in line.upper() or "INSERT" in line.upper() or "UPDATE" in line.upper():
                    findings.append(self._create_finding(
                        id=str(uuid.uuid4()),
                        title="Potential SQL injection",
                        description="Use parameterized queries instead of string concatenation",
                        severity="critical",
                        category="security",
                        line_start=i + 1,
                        line_end=i + 1,
                    ))

        return findings

    def _create_finding(
        self,
        id: str,
        title: str,
        description: str,
        severity: str,
        category: str,
        line_start: int,
        line_end: int,
    ) -> dict[str, Any]:
        """Create a finding dict."""
        return {
            "id": id,
            "title": title,
            "description": description,
            "severity": severity,
            "category": category,
            "line_start": line_start,
            "line_end": line_end,
            "source": "codeverify",
        }

    def _findings_to_diagnostics(
        self,
        findings: list[dict[str, Any]],
    ) -> list[lsp.Diagnostic]:
        """Convert findings to LSP diagnostics."""
        diagnostics = []

        for finding in findings:
            severity = self._severity_to_lsp(finding.get("severity", "medium"))

            diagnostic = lsp.Diagnostic(
                range=lsp.Range(
                    start=lsp.Position(
                        line=finding.get("line_start", 1) - 1,
                        character=0,
                    ),
                    end=lsp.Position(
                        line=finding.get("line_end", finding.get("line_start", 1)) - 1,
                        character=1000,  # End of line
                    ),
                ),
                message=f"{finding.get('title', 'Issue')}: {finding.get('description', '')}",
                severity=severity,
                source="CodeVerify",
                code=finding.get("category", "unknown"),
                data={"finding_id": finding.get("id")},
            )

            diagnostics.append(diagnostic)

        return diagnostics

    def _severity_to_lsp(self, severity: str) -> lsp.DiagnosticSeverity:
        """Convert CodeVerify severity to LSP severity."""
        mapping = {
            "critical": lsp.DiagnosticSeverity.Error,
            "high": lsp.DiagnosticSeverity.Error,
            "medium": lsp.DiagnosticSeverity.Warning,
            "low": lsp.DiagnosticSeverity.Information,
            "info": lsp.DiagnosticSeverity.Hint,
        }
        return mapping.get(severity.lower(), lsp.DiagnosticSeverity.Warning)

    async def _get_code_actions(
        self,
        params: lsp.CodeActionParams,
    ) -> list[lsp.CodeAction] | None:
        """Get code actions for the given range."""
        uri = params.text_document.uri
        findings = self._findings_by_uri.get(uri, [])

        actions = []

        # Find findings in range
        start_line = params.range.start.line + 1
        end_line = params.range.end.line + 1

        for finding in findings:
            finding_start = finding.get("line_start", 0)
            finding_end = finding.get("line_end", finding_start)

            if finding_start <= end_line and finding_end >= start_line:
                # Add fix action
                fix = self._get_fix_for_finding(finding, params.text_document.uri)
                if fix:
                    actions.append(fix)

                # Add dismiss action
                dismiss = lsp.CodeAction(
                    title=f"Dismiss: {finding.get('title', 'Issue')}",
                    kind=lsp.CodeActionKind.QuickFix,
                    command=lsp.Command(
                        title="Dismiss Finding",
                        command="codeverify.dismissFinding",
                        arguments=[uri, finding.get("id"), "Dismissed by user"],
                    ),
                )
                actions.append(dismiss)

                # Add explain action
                explain = lsp.CodeAction(
                    title=f"Explain: {finding.get('title', 'Issue')}",
                    kind=lsp.CodeActionKind.Empty,
                    command=lsp.Command(
                        title="Explain Finding",
                        command="codeverify.explainFinding",
                        arguments=[uri, finding.get("id")],
                    ),
                )
                actions.append(explain)

        return actions if actions else None

    def _get_fix_for_finding(
        self,
        finding: dict[str, Any],
        uri: str,
    ) -> lsp.CodeAction | None:
        """Generate a fix for a finding."""
        document = self.workspace.get_text_document(uri)
        if not document:
            return None

        category = finding.get("category", "")
        line_num = finding.get("line_start", 1) - 1
        lines = document.source.split("\n")

        if line_num >= len(lines):
            return None

        line = lines[line_num]
        indent = len(line) - len(line.lstrip())
        indent_str = " " * indent

        # Generate fix based on category
        if category == "null_safety":
            language = self._get_language(uri)
            if language == "python":
                # Add None check
                fix_line = f"{indent_str}if obj is not None:\n{indent_str}    {line.strip()}"
                new_text = fix_line
            elif language in ("typescript", "javascript"):
                # Add optional chaining
                new_text = line.replace(".", "?.")
            else:
                return None

        elif category == "array_bounds":
            language = self._get_language(uri)
            if language == "python":
                new_text = line  # Would need AST analysis for proper fix
            else:
                return None

        elif category == "security":
            # SQL injection fix
            if "execute" in line.lower():
                # Simplified - would need proper parsing
                new_text = line.replace('+ ', '? ').replace('" + ', '", (').replace("' + ", "', (")
            else:
                return None

        else:
            return None

        # Create text edit
        edit = lsp.TextEdit(
            range=lsp.Range(
                start=lsp.Position(line=line_num, character=0),
                end=lsp.Position(line=line_num, character=len(line)),
            ),
            new_text=new_text,
        )

        return lsp.CodeAction(
            title=f"Fix: {finding.get('title', 'Issue')}",
            kind=lsp.CodeActionKind.QuickFix,
            edit=lsp.WorkspaceEdit(
                changes={uri: [edit]},
            ),
            is_preferred=True,
        )

    async def _get_hover(self, params: lsp.HoverParams) -> lsp.Hover | None:
        """Get hover information at position."""
        uri = params.text_document.uri
        line = params.position.line + 1
        findings = self._findings_by_uri.get(uri, [])

        # Find findings at this line
        relevant_findings = [
            f for f in findings
            if f.get("line_start", 0) <= line <= f.get("line_end", f.get("line_start", 0))
        ]

        if not relevant_findings:
            return None

        # Build hover content
        contents = []
        for finding in relevant_findings:
            severity = finding.get("severity", "medium").upper()
            title = finding.get("title", "Issue")
            description = finding.get("description", "")
            category = finding.get("category", "unknown")

            content = f"""**CodeVerify [{severity}]** {title}

{description}

*Category: {category}*
"""
            contents.append(content)

        return lsp.Hover(
            contents=lsp.MarkupContent(
                kind=lsp.MarkupKind.Markdown,
                value="\n---\n".join(contents),
            ),
        )

    async def _get_trust_score(self, uri: str) -> dict[str, Any]:
        """Get trust score for a document."""
        document = self.workspace.get_text_document(uri)
        if not document:
            return {"error": "Document not found"}

        # Simple local trust score calculation
        code = document.source
        findings = self._findings_by_uri.get(uri, [])

        # Base score
        score = 100

        # Deduct for findings
        for finding in findings:
            severity = finding.get("severity", "medium")
            if severity == "critical":
                score -= 20
            elif severity == "high":
                score -= 10
            elif severity == "medium":
                score -= 5
            else:
                score -= 2

        score = max(0, score)

        # Determine risk level
        if score >= 80:
            risk_level = "low"
        elif score >= 60:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "score": score,
            "risk_level": risk_level,
            "findings_count": len(findings),
            "uri": uri,
        }

    async def _apply_fix(
        self,
        uri: str,
        finding_id: str,
        fix_code: str,
    ) -> dict[str, Any]:
        """Apply a fix to the document."""
        # Find the finding
        findings = self._findings_by_uri.get(uri, [])
        finding = next((f for f in findings if f.get("id") == finding_id), None)

        if not finding:
            return {"success": False, "error": "Finding not found"}

        # Apply via workspace edit
        line_start = finding.get("line_start", 1) - 1
        line_end = finding.get("line_end", line_start + 1) - 1

        edit = lsp.WorkspaceEdit(
            changes={
                uri: [
                    lsp.TextEdit(
                        range=lsp.Range(
                            start=lsp.Position(line=line_start, character=0),
                            end=lsp.Position(line=line_end + 1, character=0),
                        ),
                        new_text=fix_code + "\n",
                    )
                ]
            }
        )

        self.apply_edit(edit)

        # Remove finding from cache
        self._findings_by_uri[uri] = [f for f in findings if f.get("id") != finding_id]

        return {"success": True}

    async def _dismiss_finding(
        self,
        uri: str,
        finding_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Dismiss a finding."""
        findings = self._findings_by_uri.get(uri, [])
        self._findings_by_uri[uri] = [f for f in findings if f.get("id") != finding_id]

        # Re-publish diagnostics
        diagnostics = self._findings_to_diagnostics(self._findings_by_uri[uri])
        self.publish_diagnostics(uri, diagnostics)

        logger.info("Finding dismissed", finding_id=finding_id, reason=reason)

        return {"success": True}

    async def _debug_verification(
        self,
        uri: str,
        line: int,
    ) -> dict[str, Any]:
        """Get debug information for verification at line."""
        findings = self._findings_by_uri.get(uri, [])

        relevant = [
            f for f in findings
            if f.get("line_start", 0) <= line <= f.get("line_end", f.get("line_start", 0))
        ]

        return {
            "line": line,
            "findings": relevant,
            "verification_status": "complete" if not relevant else "issues_found",
        }

    def _get_language(self, uri: str) -> str:
        """Get language ID from URI."""
        path = self._uri_to_path(uri)
        ext = Path(path).suffix.lower()

        ext_to_lang = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".java": "java",
            ".rs": "rust",
        }

        return ext_to_lang.get(ext, "unknown")

    def _uri_to_path(self, uri: str) -> str:
        """Convert URI to file path."""
        parsed = urlparse(uri)
        return unquote(parsed.path)


def main():
    """Main entry point for the LSP server."""
    # Parse config from environment
    config = ServerConfig(
        api_endpoint=os.environ.get("CODEVERIFY_API_ENDPOINT", "http://localhost:8000"),
        api_key=os.environ.get("CODEVERIFY_API_KEY", ""),
        enable_real_time=os.environ.get("CODEVERIFY_REAL_TIME", "true").lower() == "true",
        real_time_delay_ms=int(os.environ.get("CODEVERIFY_REAL_TIME_DELAY", "500")),
    )

    # Create and start server
    server = CodeVerifyLanguageServer(config)

    logger.info("Starting CodeVerify LSP server")

    # Run via stdio (standard for LSP)
    server.start_io()


if __name__ == "__main__":
    main()
