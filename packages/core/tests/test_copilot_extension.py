"""Tests for Copilot Chat Integration module."""

import pytest

from codeverify_core.copilot_extension import (
    CodeSuggestion,
    CommandParser,
    CopilotChatParticipant,
    CopilotCommand,
    CopilotContext,
    CopilotMessage,
    CopilotMessageRole,
    CopilotResponse,
    CopilotWebhookHandler,
)


class TestCopilotCommand:
    def test_commands_exist(self):
        assert CopilotCommand.VERIFY is not None
        assert CopilotCommand.TRUST_SCORE is not None
        assert CopilotCommand.EXPLAIN is not None


class TestCopilotMessageRole:
    def test_roles(self):
        assert CopilotMessageRole.USER is not None
        assert CopilotMessageRole.ASSISTANT is not None


class TestCopilotMessage:
    def test_creation(self):
        msg = CopilotMessage(
            role=CopilotMessageRole.USER,
            content="verify this function",
            timestamp=0.0,
        )
        assert msg.role == CopilotMessageRole.USER
        assert "verify" in msg.content


class TestCopilotContext:
    def test_creation(self):
        ctx = CopilotContext(
            file_path="src/main.py",
            language="python",
            selected_code="def foo(): pass",
            full_file_content="def foo(): pass",
        )
        assert ctx.language == "python"
        assert ctx.file_path == "src/main.py"


class TestCodeSuggestion:
    def test_creation(self):
        suggestion = CodeSuggestion(
            file_path="test.py",
            original_code="eval(x)",
            suggested_code="ast.literal_eval(x)",
            explanation="Replace eval with safe alternative",
            confidence=0.95,
        )
        assert suggestion.confidence == 0.95


class TestCopilotResponse:
    def test_creation(self):
        resp = CopilotResponse(
            content="No issues found",
            code_suggestions=[],
            diagnostics=[],
        )
        assert resp.content == "No issues found"


class TestCommandParser:
    def test_parse_verify_command(self):
        parser = CommandParser()
        result = parser.parse("verify this function")
        assert result is not None

    def test_parse_trust_score(self):
        parser = CommandParser()
        result = parser.parse("what's the trust score?")
        assert result is not None

    def test_parse_unknown(self):
        parser = CommandParser()
        result = parser.parse("hello world")
        assert result is not None


class TestCopilotChatParticipant:
    def test_creation(self):
        participant = CopilotChatParticipant()
        assert participant is not None

    @pytest.mark.asyncio
    async def test_handle_message(self):
        participant = CopilotChatParticipant()
        ctx = CopilotContext(
            file_path="main.py",
            language="python",
            selected_code="x = eval(input())",
            full_file_content="x = eval(input())",
        )
        response = await participant.handle_message("is this safe?", ctx)
        assert isinstance(response, CopilotResponse)


class TestCopilotWebhookHandler:
    def test_creation(self):
        handler = CopilotWebhookHandler()
        assert handler is not None

    def test_has_handle_event(self):
        handler = CopilotWebhookHandler()
        assert hasattr(handler, "handle_event")
        assert callable(handler.handle_event)
