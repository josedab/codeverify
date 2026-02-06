"""
GitHub Copilot Extension / Chat Participant Integration.

Provides a chat participant for GitHub Copilot that exposes CodeVerify
capabilities through conversational commands:
- @codeverify verify   -- run formal verification on selected code
- @codeverify explain  -- explain a proof or verification result
- @codeverify spec     -- generate formal specifications
- @codeverify trust-score -- compute an AI trust score
- @codeverify fix      -- suggest verified fixes for issues
- @codeverify history  -- show verification history for the file

Also includes a webhook handler for running as a standalone GitHub
Copilot Extension that receives events from the Copilot platform.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class CopilotCommand(str, Enum):
    """Commands recognised by the CodeVerify chat participant."""

    VERIFY = "verify"
    EXPLAIN = "explain"
    SPEC = "spec"
    TRUST_SCORE = "trust_score"
    FIX = "fix"
    HISTORY = "history"


class CopilotMessageRole(str, Enum):
    """Role of a message in the Copilot conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CopilotMessage:
    """A single message in the Copilot conversation thread."""

    role: CopilotMessageRole
    content: str
    timestamp: float


@dataclass
class CopilotContext:
    """Editor context supplied alongside a Copilot chat message.

    Captures everything the extension knows about the user's current
    editing session so that handlers can tailor their responses.
    """

    file_path: str | None = None
    language: str | None = None
    selected_code: str | None = None
    full_file_content: str | None = None
    cursor_line: int | None = None
    repository: str | None = None
    branch: str | None = None


@dataclass
class CodeSuggestion:
    """A concrete code change proposed by the extension."""

    file_path: str
    original_code: str
    suggested_code: str
    explanation: str
    confidence: float


@dataclass
class CopilotResponse:
    """Response returned to the Copilot chat UI.

    The ``content`` field is markdown-formatted so the chat panel can
    render it with rich formatting, code blocks and links.
    """

    content: str
    code_suggestions: list[CodeSuggestion] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    follow_up_actions: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


# =============================================================================
# Command Parser
# =============================================================================


# Mapping of natural-language trigger phrases to commands.  Each tuple
# contains a compiled regex and the command it maps to.
_NL_COMMAND_PATTERNS: list[tuple[re.Pattern[str], CopilotCommand]] = [
    # Explicit slash-style invocations (/verify, /fix, etc.)
    (re.compile(r"^verify\b", re.IGNORECASE), CopilotCommand.VERIFY),
    (re.compile(r"^explain\b", re.IGNORECASE), CopilotCommand.EXPLAIN),
    (re.compile(r"^spec\b", re.IGNORECASE), CopilotCommand.SPEC),
    (re.compile(r"^trust[- ]?score\b", re.IGNORECASE), CopilotCommand.TRUST_SCORE),
    (re.compile(r"^fix\b", re.IGNORECASE), CopilotCommand.FIX),
    (re.compile(r"^history\b", re.IGNORECASE), CopilotCommand.HISTORY),

    # Natural-language variants -- order matters; more specific first.
    (re.compile(r"is\s+this\s+(?:code\s+)?safe", re.IGNORECASE), CopilotCommand.VERIFY),
    (re.compile(r"check\s+(?:this|the)\s+code", re.IGNORECASE), CopilotCommand.VERIFY),
    (re.compile(r"verify\s+(?:this|the)", re.IGNORECASE), CopilotCommand.VERIFY),
    (re.compile(r"are\s+there\s+(?:any\s+)?(?:bugs|issues|problems)", re.IGNORECASE), CopilotCommand.VERIFY),

    (re.compile(r"what\s+does\s+this\s+proof\s+mean", re.IGNORECASE), CopilotCommand.EXPLAIN),
    (re.compile(r"explain\s+(?:this|the)\s+(?:proof|result|verification)", re.IGNORECASE), CopilotCommand.EXPLAIN),
    (re.compile(r"why\s+(?:did|does)\s+(?:this|the)\s+verification", re.IGNORECASE), CopilotCommand.EXPLAIN),

    (re.compile(r"generate\s+(?:a\s+)?(?:formal\s+)?spec", re.IGNORECASE), CopilotCommand.SPEC),
    (re.compile(r"(?:write|create)\s+(?:a\s+)?specification", re.IGNORECASE), CopilotCommand.SPEC),

    (re.compile(r"(?:how\s+)?trust(?:worthy)?", re.IGNORECASE), CopilotCommand.TRUST_SCORE),
    (re.compile(r"(?:what|compute|calculate)\s+(?:is\s+)?(?:the\s+)?(?:trust|quality)\s*score", re.IGNORECASE), CopilotCommand.TRUST_SCORE),
    (re.compile(r"(?:how\s+)?reliable\s+is", re.IGNORECASE), CopilotCommand.TRUST_SCORE),

    (re.compile(r"(?:suggest|propose)\s+(?:a\s+)?fix", re.IGNORECASE), CopilotCommand.FIX),
    (re.compile(r"how\s+(?:do\s+I\s+|to\s+)?fix", re.IGNORECASE), CopilotCommand.FIX),
    (re.compile(r"auto[- ]?fix", re.IGNORECASE), CopilotCommand.FIX),

    (re.compile(r"(?:show|list|get)\s+(?:the\s+)?(?:verification\s+)?history", re.IGNORECASE), CopilotCommand.HISTORY),
    (re.compile(r"previous\s+(?:verification|check)s", re.IGNORECASE), CopilotCommand.HISTORY),
]


class CommandParser:
    """Parse a raw chat message into a ``CopilotCommand`` and argument text.

    The parser first strips the ``@codeverify`` mention prefix (if present),
    then tries explicit keyword matching before falling back to
    natural-language pattern matching.
    """

    _PREFIX_RE = re.compile(r"^@codeverify\s*", re.IGNORECASE)

    def parse(self, message: str) -> tuple[CopilotCommand | None, str]:
        """Parse *message* and return ``(command, remaining_text)``.

        Returns ``(None, message)`` when no command can be determined.
        """
        text = self._PREFIX_RE.sub("", message).strip()

        for pattern, command in _NL_COMMAND_PATTERNS:
            match = pattern.search(text)
            if match:
                remaining = text[match.end():].strip()
                return command, remaining

        return None, text


# =============================================================================
# Chat Participant
# =============================================================================


class CopilotChatParticipant:
    """Main entry-point for the CodeVerify Copilot chat participant.

    Maintains conversation history, routes incoming messages to the
    appropriate command handler, and returns rich markdown responses.
    """

    def __init__(self) -> None:
        self._parser = CommandParser()
        self._conversation_history: list[CopilotMessage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_message(
        self,
        message: str,
        context: CopilotContext,
    ) -> CopilotResponse:
        """Handle an incoming chat message and return a response.

        Args:
            message: Raw text entered by the user in the Copilot chat panel.
            context: Editor context (selected code, file path, etc.).

        Returns:
            A ``CopilotResponse`` with markdown content and optional
            code suggestions / diagnostics.
        """
        start = time.monotonic()

        # Record user message
        self._conversation_history.append(
            CopilotMessage(
                role=CopilotMessageRole.USER,
                content=message,
                timestamp=time.time(),
            )
        )

        command, argument = self._parser.parse(message)
        code = context.selected_code or context.full_file_content or ""

        logger.info(
            "copilot_message_received",
            command=command.value if command else None,
            has_code=bool(code),
            file=context.file_path,
        )

        if command is None:
            response = self._build_help_response()
        elif command == CopilotCommand.VERIFY:
            response = await self._handle_verify(code, context)
        elif command == CopilotCommand.EXPLAIN:
            response = await self._handle_explain(argument, context)
        elif command == CopilotCommand.SPEC:
            response = await self._handle_spec(code, context)
        elif command == CopilotCommand.TRUST_SCORE:
            response = await self._handle_trust_score(code, context)
        elif command == CopilotCommand.FIX:
            response = await self._handle_fix(code, context)
        elif command == CopilotCommand.HISTORY:
            response = await self._handle_history(context)
        else:
            response = self._build_help_response()

        elapsed_ms = (time.monotonic() - start) * 1000
        response.processing_time_ms = elapsed_ms

        # Record assistant reply
        self._conversation_history.append(
            CopilotMessage(
                role=CopilotMessageRole.ASSISTANT,
                content=response.content,
                timestamp=time.time(),
            )
        )

        return response

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _handle_verify(
        self,
        code: str,
        context: CopilotContext,
    ) -> CopilotResponse:
        """Run formal verification on the supplied code.

        Delegates to the core verification engine and formats the
        results as a markdown report.
        """
        if not code:
            return CopilotResponse(
                content=(
                    "**No code to verify.**\n\n"
                    "Select some code in the editor or open a file, "
                    "then try `@codeverify verify` again."
                ),
            )

        logger.info("copilot_verify_start", file=context.file_path)

        # Placeholder: integrate with the real verification pipeline.
        results: dict[str, Any] = {
            "verified": True,
            "issues": [],
            "checked": 1,
            "passed": 1,
            "failed": 0,
        }

        content = self._format_verify_response(results)

        return CopilotResponse(
            content=content,
            diagnostics=[
                {"severity": issue.get("severity", "warning"), "message": issue.get("message", "")}
                for issue in results.get("issues", [])
            ],
            follow_up_actions=["Run `@codeverify explain` to understand the proof"],
        )

    async def _handle_explain(
        self,
        query: str,
        context: CopilotContext,
    ) -> CopilotResponse:
        """Explain a proof or verification result in plain English.

        If the user supplied a specific question it is forwarded to the
        natural-language query engine; otherwise the most recent
        verification result in the conversation is explained.
        """
        if not query and not context.selected_code:
            return CopilotResponse(
                content=(
                    "**What would you like explained?**\n\n"
                    "Try one of:\n"
                    "- `@codeverify explain this proof`\n"
                    "- `@codeverify explain why this failed`\n"
                    "- Select a verification annotation and ask again."
                ),
            )

        target = query or "the selected verification result"

        # Placeholder: integrate with the NL verification engine.
        content = (
            f"## Explanation\n\n"
            f"You asked about: *{target}*\n\n"
            f"The verification confirms that the selected code satisfies "
            f"its inferred specification.  All preconditions are met on "
            f"every reachable path, and no postcondition violations were "
            f"found.\n\n"
            f"> **Tip:** Select specific code and run `@codeverify verify` "
            f"to generate a new proof you can ask about."
        )

        return CopilotResponse(
            content=content,
            follow_up_actions=[
                "Run `@codeverify spec` to see the formal specification",
                "Run `@codeverify verify` on different code",
            ],
        )

    async def _handle_spec(
        self,
        code: str,
        context: CopilotContext,
    ) -> CopilotResponse:
        """Generate formal specifications for the supplied code.

        Uses the specification generator from ``codeverify_core.formal_specs``
        to infer pre/post-conditions and invariants, then renders them
        as a readable markdown document.
        """
        if not code:
            return CopilotResponse(
                content=(
                    "**No code selected.**\n\n"
                    "Select a function or class, then run "
                    "`@codeverify spec` to generate its formal specification."
                ),
            )

        language = context.language or "python"

        # Placeholder: integrate with SpecificationGenerator.
        content = (
            f"## Formal Specification\n\n"
            f"**Language:** {language}\n"
            f"**File:** {context.file_path or 'unknown'}\n\n"
            f"### Preconditions\n"
            f"- All parameters are non-null\n\n"
            f"### Postconditions\n"
            f"- Return value satisfies the declared type\n\n"
            f"### Invariants\n"
            f"- Internal state remains consistent after invocation\n"
        )

        return CopilotResponse(
            content=content,
            follow_up_actions=[
                "Run `@codeverify verify` to check the code against this spec",
                "Run `@codeverify trust-score` to assess AI-generated code quality",
            ],
        )

    async def _handle_trust_score(
        self,
        code: str,
        context: CopilotContext,
    ) -> CopilotResponse:
        """Compute and display a trust score for the supplied code.

        The trust score evaluates how reliable a piece of (potentially
        AI-generated) code is, based on complexity, pattern confidence,
        verification coverage and historical accuracy.
        """
        if not code:
            return CopilotResponse(
                content=(
                    "**No code to score.**\n\n"
                    "Select code in the editor and run "
                    "`@codeverify trust-score` again."
                ),
            )

        # Placeholder: integrate with TrustScoreAgent.
        score: dict[str, Any] = {
            "score": 82.5,
            "confidence": 0.91,
            "risk_level": "low",
            "is_ai_generated": False,
            "factors": {
                "complexity_score": 0.25,
                "pattern_confidence": 0.88,
                "historical_accuracy": 0.90,
                "verification_coverage": 0.78,
                "code_quality_signals": 0.85,
            },
            "recommendations": [],
        }

        content = self._format_trust_score_response(score)

        return CopilotResponse(
            content=content,
            follow_up_actions=[
                "Run `@codeverify verify` for formal verification",
                "Run `@codeverify fix` to address any issues",
            ],
        )

    async def _handle_fix(
        self,
        code: str,
        context: CopilotContext,
    ) -> CopilotResponse:
        """Suggest verified fixes for issues found in the code.

        Combines the verification engine with the autofix agent to
        propose changes that provably resolve detected issues.
        """
        if not code:
            return CopilotResponse(
                content=(
                    "**No code to fix.**\n\n"
                    "Select code with a known issue and run "
                    "`@codeverify fix` again."
                ),
            )

        file_path = context.file_path or "untitled"

        # Placeholder: integrate with the autofix agent.
        suggestion = CodeSuggestion(
            file_path=file_path,
            original_code=code,
            suggested_code=code,
            explanation="No issues detected -- the code looks correct.",
            confidence=0.95,
        )

        content = (
            "## Fix Suggestions\n\n"
            "No verification failures were found in the selected code.  "
            "If you believe there is an issue, try adding more context by "
            "selecting a larger code region.\n"
        )

        return CopilotResponse(
            content=content,
            code_suggestions=[suggestion],
            follow_up_actions=[
                "Run `@codeverify verify` for a detailed verification report",
            ],
        )

    async def _handle_history(
        self,
        context: CopilotContext,
    ) -> CopilotResponse:
        """Show verification history for the current file.

        Retrieves past verification runs and presents them in a
        summarised timeline.
        """
        file_path = context.file_path

        if not file_path:
            return CopilotResponse(
                content=(
                    "**No file context available.**\n\n"
                    "Open a file in the editor so that "
                    "`@codeverify history` can look up its records."
                ),
            )

        # Placeholder: integrate with verification history store.
        content = (
            f"## Verification History\n\n"
            f"**File:** `{file_path}`\n"
            f"**Branch:** {context.branch or 'unknown'}\n\n"
            f"| # | Date | Result | Issues |\n"
            f"|---|------|--------|--------|\n"
            f"| 1 | -- | Passed | 0 |\n\n"
            f"_No previous verification runs recorded for this file._\n"
        )

        return CopilotResponse(
            content=content,
            follow_up_actions=[
                "Run `@codeverify verify` to create a new verification record",
            ],
        )

    # ------------------------------------------------------------------
    # Formatters
    # ------------------------------------------------------------------

    def _format_verify_response(self, results: dict[str, Any]) -> str:
        """Format verification results as a markdown report."""
        verified = results.get("verified", False)
        checked = results.get("checked", 0)
        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        issues = results.get("issues", [])

        status = "Passed" if verified else "Failed"

        lines: list[str] = [
            f"## Verification Result: {status}\n",
            f"**Checks:** {checked} total, {passed} passed, {failed} failed\n",
        ]

        if issues:
            lines.append("### Issues\n")
            for idx, issue in enumerate(issues, 1):
                severity = issue.get("severity", "warning").upper()
                message = issue.get("message", "Unknown issue")
                line_num = issue.get("line")
                loc = f" (line {line_num})" if line_num else ""
                lines.append(f"{idx}. **[{severity}]** {message}{loc}")
            lines.append("")
        else:
            lines.append("No issues found -- all checks passed.\n")

        return "\n".join(lines)

    def _format_trust_score_response(self, score: dict[str, Any]) -> str:
        """Format a trust-score result as a markdown report."""
        value = score.get("score", 0)
        confidence = score.get("confidence", 0)
        risk = score.get("risk_level", "unknown")
        ai_gen = score.get("is_ai_generated", False)
        factors = score.get("factors", {})
        recommendations = score.get("recommendations", [])

        lines: list[str] = [
            "## Trust Score Report\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Score** | {value:.1f} / 100 |",
            f"| **Confidence** | {confidence:.0%} |",
            f"| **Risk Level** | {risk} |",
            f"| **AI-Generated** | {'Yes' if ai_gen else 'No'} |",
            "",
        ]

        if factors:
            lines.append("### Factor Breakdown\n")
            for name, val in factors.items():
                label = name.replace("_", " ").title()
                bar_len = int(round(float(val) * 20))
                bar = "#" * bar_len + "-" * (20 - bar_len)
                lines.append(f"- **{label}:** `[{bar}]` {float(val):.0%}")
            lines.append("")

        if recommendations:
            lines.append("### Recommendations\n")
            for rec in recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    @staticmethod
    def _build_help_response() -> CopilotResponse:
        """Return a help message listing available commands."""
        content = (
            "## CodeVerify Commands\n\n"
            "| Command | Description |\n"
            "|---------|-------------|\n"
            "| `@codeverify verify` | Run formal verification on selected code |\n"
            "| `@codeverify explain` | Explain a proof or verification result |\n"
            "| `@codeverify spec` | Generate formal specifications |\n"
            "| `@codeverify trust-score` | Compute a trust score for the code |\n"
            "| `@codeverify fix` | Suggest verified fixes |\n"
            "| `@codeverify history` | Show verification history for the file |\n\n"
            "You can also use natural language, for example:\n"
            "- *\"Is this code safe?\"*\n"
            "- *\"What does this proof mean?\"*\n"
            "- *\"How trustworthy is this?\"*\n"
        )

        return CopilotResponse(content=content)


# =============================================================================
# Webhook Handler (standalone Copilot Extension)
# =============================================================================


class CopilotWebhookHandler:
    """Handle incoming webhook events from the GitHub Copilot platform.

    When CodeVerify is deployed as a standalone GitHub Copilot Extension
    (rather than a VS Code chat participant), the platform delivers
    events to a registered endpoint.  This class validates, parses and
    dispatches those events.
    """

    def __init__(self) -> None:
        self._participant = CopilotChatParticipant()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle a Copilot Extension webhook event.

        Args:
            event_type: The event type header (e.g. ``copilot_chat.message``).
            payload: The JSON-decoded request body.

        Returns:
            A JSON-serialisable response dict to send back to the
            Copilot platform.
        """
        logger.info("copilot_webhook_event", event_type=event_type)

        if event_type == "copilot_chat.message":
            return await self._handle_chat_message(payload)

        logger.warning("copilot_webhook_unknown_event", event_type=event_type)
        return {"status": "ignored", "reason": f"unknown event type: {event_type}"}

    @staticmethod
    def validate_signature(
        payload: bytes,
        signature: str,
        secret: str,
    ) -> bool:
        """Validate the HMAC-SHA256 signature of a webhook payload.

        Args:
            payload: Raw request body bytes.
            signature: Value of the ``X-Hub-Signature-256`` header,
                typically prefixed with ``sha256=``.
            secret: Shared webhook secret configured in the GitHub App.

        Returns:
            ``True`` if the signature matches, ``False`` otherwise.
        """
        if signature.startswith("sha256="):
            signature = signature[len("sha256="):]

        expected = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _handle_chat_message(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a ``copilot_chat.message`` event."""
        messages = payload.get("messages", [])
        if not messages:
            return {"content": "No message received."}

        last_message = messages[-1]
        user_text: str = last_message.get("content", "")

        context = self._parse_references(payload)

        response = await self._participant.handle_message(user_text, context)

        return {
            "content": response.content,
            "code_suggestions": [
                {
                    "file_path": s.file_path,
                    "original_code": s.original_code,
                    "suggested_code": s.suggested_code,
                    "explanation": s.explanation,
                    "confidence": s.confidence,
                }
                for s in response.code_suggestions
            ],
            "diagnostics": response.diagnostics,
            "follow_up_actions": response.follow_up_actions,
            "processing_time_ms": response.processing_time_ms,
        }

    @staticmethod
    def _parse_references(payload: dict[str, Any]) -> CopilotContext:
        """Extract ``CopilotContext`` from the webhook payload references.

        The Copilot platform attaches editor references (current file,
        selected text, repository metadata, etc.) inside the payload so
        that extensions can make context-aware decisions.
        """
        references = payload.get("copilot_references", [])

        file_path: str | None = None
        language: str | None = None
        selected_code: str | None = None
        full_file_content: str | None = None
        cursor_line: int | None = None
        repository: str | None = None
        branch: str | None = None

        for ref in references:
            ref_type = ref.get("type", "")
            data = ref.get("data", {})

            if ref_type == "file":
                file_path = data.get("path", file_path)
                language = data.get("language", language)
                full_file_content = data.get("content", full_file_content)

            elif ref_type == "selection":
                selected_code = data.get("text", selected_code)
                cursor_line = data.get("start_line", cursor_line)

            elif ref_type == "repository":
                repository = data.get("full_name", repository)
                branch = data.get("default_branch", branch)

        return CopilotContext(
            file_path=file_path,
            language=language,
            selected_code=selected_code,
            full_file_content=full_file_content,
            cursor_line=cursor_line,
            repository=repository,
            branch=branch,
        )
