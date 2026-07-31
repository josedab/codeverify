"""NL Conversation Sessions — Multi-turn natural language verification queries.

Provides conversational interface for verification queries with context
accumulation across turns, enabling developers to refine their questions
and explore code properties interactively.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class QueryIntent(str, Enum):
    """Classified intent of a user query."""

    NULL_CHECK = "null_check"
    BOUNDS_CHECK = "bounds_check"
    OVERFLOW_CHECK = "overflow_check"
    EXCEPTION_ANALYSIS = "exception_analysis"
    REACHABILITY = "reachability"
    INVARIANT = "invariant"
    COMPARISON = "comparison"
    GENERAL = "general"
    FOLLOWUP = "followup"
    CLARIFICATION = "clarification"


class AnswerConfidence(str, Enum):
    """Confidence level of a verification answer."""

    PROVEN = "proven"
    LIKELY = "likely"
    UNCERTAIN = "uncertain"
    DISPROVEN = "disproven"


@dataclass
class ConversationTurn:
    """A single turn in a conversation (query + response)."""

    turn_id: str
    query: str
    intent: QueryIntent
    code_context: str | None = None
    response: str = ""
    confidence: AnswerConfidence = AnswerConfidence.UNCERTAIN
    verification_result: dict[str, Any] | None = None
    suggestions: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "intent": self.intent.value,
            "response": self.response,
            "confidence": self.confidence.value,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConversationContext:
    """Accumulated context across conversation turns."""

    code_snippets: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)
    variables_mentioned: set[str] = field(default_factory=set)
    properties_checked: list[str] = field(default_factory=list)
    language: str = "python"

    def add_code(self, code: str, file_path: str | None = None) -> None:
        self.code_snippets.append(code)
        if file_path and file_path not in self.file_paths:
            self.file_paths.append(file_path)

    def get_summary(self) -> str:
        parts = []
        if self.file_paths:
            parts.append(f"Files: {', '.join(self.file_paths)}")
        if self.variables_mentioned:
            parts.append(f"Variables: {', '.join(self.variables_mentioned)}")
        if self.properties_checked:
            parts.append(f"Properties checked: {', '.join(self.properties_checked)}")
        return "; ".join(parts) if parts else "No context accumulated"


class IntentClassifier:
    """Classifies user query intent for routing to appropriate verification."""

    INTENT_PATTERNS: list[tuple[str, QueryIntent]] = [
        ("null", QueryIntent.NULL_CHECK),
        ("none", QueryIntent.NULL_CHECK),
        ("nil", QueryIntent.NULL_CHECK),
        ("undefined", QueryIntent.NULL_CHECK),
        ("bound", QueryIntent.BOUNDS_CHECK),
        ("index", QueryIntent.BOUNDS_CHECK),
        ("range", QueryIntent.BOUNDS_CHECK),
        ("out of", QueryIntent.BOUNDS_CHECK),
        ("overflow", QueryIntent.OVERFLOW_CHECK),
        ("underflow", QueryIntent.OVERFLOW_CHECK),
        ("wrap", QueryIntent.OVERFLOW_CHECK),
        ("exception", QueryIntent.EXCEPTION_ANALYSIS),
        ("error", QueryIntent.EXCEPTION_ANALYSIS),
        ("throw", QueryIntent.EXCEPTION_ANALYSIS),
        ("raise", QueryIntent.EXCEPTION_ANALYSIS),
        ("crash", QueryIntent.EXCEPTION_ANALYSIS),
        ("reach", QueryIntent.REACHABILITY),
        ("dead code", QueryIntent.REACHABILITY),
        ("unreachable", QueryIntent.REACHABILITY),
        ("always", QueryIntent.INVARIANT),
        ("never", QueryIntent.INVARIANT),
        ("invariant", QueryIntent.INVARIANT),
        ("guarantee", QueryIntent.INVARIANT),
        ("compare", QueryIntent.COMPARISON),
        ("same", QueryIntent.COMPARISON),
        ("equivalent", QueryIntent.COMPARISON),
        ("differ", QueryIntent.COMPARISON),
    ]

    FOLLOWUP_PATTERNS = [
        "what about",
        "also",
        "and what",
        "how about",
        "what if",
        "can it",
        "does it",
        "is it",
    ]

    CLARIFICATION_PATTERNS = [
        "what do you mean",
        "explain",
        "why",
        "show me",
        "give me an example",
        "can you elaborate",
    ]

    def classify(self, query: str, has_history: bool = False) -> QueryIntent:
        """Classify the intent of a user query."""
        lower = query.lower().strip()

        # Check for clarification
        for pattern in self.CLARIFICATION_PATTERNS:
            if lower.startswith(pattern):
                return QueryIntent.CLARIFICATION

        # Check for follow-up
        if has_history:
            for pattern in self.FOLLOWUP_PATTERNS:
                if lower.startswith(pattern):
                    return QueryIntent.FOLLOWUP

        # Pattern-based classification
        for keyword, intent in self.INTENT_PATTERNS:
            if keyword in lower:
                return intent

        return QueryIntent.GENERAL

    def extract_variables(self, query: str) -> list[str]:
        """Extract variable names mentioned in the query."""
        import re

        # Match backtick-quoted identifiers or common variable patterns
        backtick = re.findall(r"`(\w+)`", query)
        # Match "variable X" or "parameter X" patterns
        named = re.findall(
            r"(?:variable|parameter|argument|field|attr)\s+['\"]?(\w+)['\"]?",
            query,
            re.IGNORECASE,
        )
        return list(set(backtick + named))


class ResponseGenerator:
    """Generates human-readable responses for verification results."""

    def generate(
        self,
        intent: QueryIntent,
        query: str,
        code: str,
        context: ConversationContext,
    ) -> tuple[str, AnswerConfidence, list[str]]:
        """Generate a response for a verification query.

        Returns (response_text, confidence, follow_up_suggestions).
        """
        # Simplified local verification (in production this calls Z3/LLM)
        if intent == QueryIntent.NULL_CHECK:
            return self._check_null(query, code, context)
        elif intent == QueryIntent.BOUNDS_CHECK:
            return self._check_bounds(query, code, context)
        elif intent == QueryIntent.OVERFLOW_CHECK:
            return self._check_overflow(query, code, context)
        elif intent == QueryIntent.EXCEPTION_ANALYSIS:
            return self._check_exceptions(query, code, context)
        elif intent == QueryIntent.INVARIANT:
            return self._check_invariant(query, code, context)
        elif intent == QueryIntent.CLARIFICATION:
            return self._clarify(query, context)
        elif intent == QueryIntent.FOLLOWUP:
            return self._followup(query, context)
        else:
            return self._general_analysis(query, code, context)

    def _check_null(
        self, _query: str, code: str, ctx: ConversationContext
    ) -> tuple[str, AnswerConfidence, list[str]]:
        import re

        none_checks = re.findall(r"if\s+\w+\s+is\s+(?:not\s+)?None", code)
        optional_params = re.findall(r"(\w+)\s*:\s*.*\|\s*None", code)

        if none_checks:
            response = f"Found {len(none_checks)} null check(s) in the code. "
            if optional_params:
                response += f"Parameters {optional_params} accept None. "
            confidence = AnswerConfidence.LIKELY
            suggestions = [
                "Does this handle None in all branches?",
                "What about nested None values?",
            ]
        else:
            response = "No explicit null checks found. "
            if optional_params:
                response += f"⚠️ Parameters {optional_params} accept None but are not guarded."
                confidence = AnswerConfidence.DISPROVEN
            else:
                confidence = AnswerConfidence.UNCERTAIN
            suggestions = ["Show me which parameters could be None", "Can you add null safety?"]

        ctx.properties_checked.append("null_safety")
        return response, confidence, suggestions

    def _check_bounds(
        self, _query: str, code: str, ctx: ConversationContext
    ) -> tuple[str, AnswerConfidence, list[str]]:
        import re

        index_access = re.findall(r"(\w+)\[(\w+)\]", code)
        len_checks = re.findall(r"len\((\w+)\)", code)

        if index_access:
            response = f"Found {len(index_access)} index access(es): "
            response += ", ".join(f"{var}[{idx}]" for var, idx in index_access[:5])
            response += ". "
            if len_checks:
                response += f"Length checks exist for: {', '.join(len_checks)}."
                confidence = AnswerConfidence.LIKELY
            else:
                response += "⚠️ No length/bounds checks found before index access."
                confidence = AnswerConfidence.DISPROVEN
        else:
            response = "No direct index access found in this code."
            confidence = AnswerConfidence.PROVEN

        ctx.properties_checked.append("bounds_safety")
        return (
            response,
            confidence,
            ["Can any of these arrays be empty?", "What about negative indices?"],
        )

    def _check_overflow(
        self, _query: str, code: str, ctx: ConversationContext
    ) -> tuple[str, AnswerConfidence, list[str]]:
        import re

        arithmetic = re.findall(r"(\w+)\s*([+\-*/])\s*(\w+)", code)
        response = f"Found {len(arithmetic)} arithmetic operation(s). "
        if arithmetic:
            response += (
                "Python integers have arbitrary precision, so overflow is unlikely for int types. "
            )
            response += "However, if these interact with C extensions or fixed-width types, overflow is possible."
            confidence = AnswerConfidence.LIKELY
        else:
            response = "No arithmetic operations found in this code."
            confidence = AnswerConfidence.PROVEN

        ctx.properties_checked.append("overflow_safety")
        return (
            response,
            confidence,
            ["Are any of these values from external input?", "What about float precision?"],
        )

    def _check_exceptions(
        self, _query: str, code: str, ctx: ConversationContext
    ) -> tuple[str, AnswerConfidence, list[str]]:
        import re

        try_blocks = re.findall(r"\btry\b", code)
        raises = re.findall(r"\braise\s+(\w+)", code)
        bare_excepts = re.findall(r"except\s*:", code)

        response = f"Found {len(try_blocks)} try block(s), {len(raises)} raise statement(s). "
        if raises:
            response += f"Exceptions raised: {', '.join(set(raises))}. "
        if bare_excepts:
            response += (
                f"⚠️ {len(bare_excepts)} bare except clause(s) found — these catch everything."
            )

        confidence = AnswerConfidence.LIKELY if try_blocks else AnswerConfidence.UNCERTAIN
        ctx.properties_checked.append("exception_handling")
        return (
            response,
            confidence,
            ["What happens if an unexpected exception occurs?", "Are all exceptions logged?"],
        )

    def _check_invariant(
        self, _query: str, code: str, ctx: ConversationContext
    ) -> tuple[str, AnswerConfidence, list[str]]:
        import re

        asserts = re.findall(r"\bassert\s+(.+)", code)
        response = f"Found {len(asserts)} assertion(s) in the code. "
        if asserts:
            response += "Asserted conditions: " + "; ".join(a.strip()[:50] for a in asserts[:3])
            confidence = AnswerConfidence.LIKELY
        else:
            response = "No assertions or invariants found. Consider adding explicit checks."
            confidence = AnswerConfidence.UNCERTAIN

        ctx.properties_checked.append("invariants")
        return (
            response,
            confidence,
            ["What invariants should hold here?", "Can you verify this with Z3?"],
        )

    def _clarify(
        self, _query: str, ctx: ConversationContext
    ) -> tuple[str, AnswerConfidence, list[str]]:
        summary = ctx.get_summary()
        response = (
            f"Here's what we've covered so far: {summary}. What would you like me to clarify?"
        )
        return (
            response,
            AnswerConfidence.UNCERTAIN,
            ["Can you re-check null safety?", "Show me the code again"],
        )

    def _followup(
        self, _query: str, ctx: ConversationContext
    ) -> tuple[str, AnswerConfidence, list[str]]:
        unchecked = {
            "null_safety",
            "bounds_safety",
            "overflow_safety",
            "exception_handling",
            "invariants",
        } - set(ctx.properties_checked)
        if unchecked:
            response = f"We haven't checked: {', '.join(unchecked)}. Would you like me to check any of these?"
        else:
            response = "We've covered all major property checks. Would you like a summary?"
        return response, AnswerConfidence.UNCERTAIN, [f"Check {p}" for p in list(unchecked)[:3]]

    def _general_analysis(
        self, _query: str, code: str, _ctx: ConversationContext
    ) -> tuple[str, AnswerConfidence, list[str]]:
        import re

        lines = code.strip().split("\n")
        funcs = re.findall(r"def\s+(\w+)", code)
        classes = re.findall(r"class\s+(\w+)", code)

        response = f"Code has {len(lines)} lines"
        if funcs:
            response += f", {len(funcs)} function(s): {', '.join(funcs[:5])}"
        if classes:
            response += f", {len(classes)} class(es): {', '.join(classes[:5])}"
        response += ". What would you like to verify about this code?"

        return (
            response,
            AnswerConfidence.UNCERTAIN,
            [
                "Does this handle null values correctly?",
                "Can any array access go out of bounds?",
                "What exceptions can this code throw?",
            ],
        )


class ConversationSession:
    """Manages a multi-turn verification conversation.

    Example:
        >>> session = ConversationSession(code="def divide(a, b): return a / b")
        >>> turn = session.ask("Can this throw an exception?")
        >>> turn.response  # "Found 0 try blocks... raise statements..."
        >>> turn2 = session.ask("What about division by zero?")
    """

    def __init__(
        self,
        code: str = "",
        file_path: str | None = None,
        language: str = "python",
    ) -> None:
        self.session_id = str(uuid.uuid4())
        self.code = code
        self.context = ConversationContext(language=language)
        self._classifier = IntentClassifier()
        self._generator = ResponseGenerator()
        self._turns: list[ConversationTurn] = []

        if code:
            self.context.add_code(code, file_path)

    def ask(self, query: str, code: str | None = None) -> ConversationTurn:
        """Ask a verification question and get a response."""
        if code:
            self.code = code
            self.context.add_code(code)

        has_history = len(self._turns) > 0
        intent = self._classifier.classify(query, has_history)

        # Extract variables
        variables = self._classifier.extract_variables(query)
        self.context.variables_mentioned.update(variables)

        # Generate response
        response, confidence, suggestions = self._generator.generate(
            intent, query, self.code, self.context
        )

        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            query=query,
            intent=intent,
            code_context=self.code[:500] if self.code else None,
            response=response,
            confidence=confidence,
            suggestions=suggestions,
        )
        self._turns.append(turn)

        logger.info(
            "Conversation turn",
            session_id=self.session_id,
            turn=len(self._turns),
            intent=intent.value,
            confidence=confidence.value,
        )

        return turn

    def get_history(self) -> list[dict[str, Any]]:
        """Get conversation history."""
        return [t.to_dict() for t in self._turns]

    def get_summary(self) -> dict[str, Any]:
        """Get session summary."""
        return {
            "session_id": self.session_id,
            "turns": len(self._turns),
            "context": self.context.get_summary(),
            "properties_checked": self.context.properties_checked,
            "intents": [t.intent.value for t in self._turns],
        }

    def reset(self) -> None:
        """Reset conversation state but keep code context."""
        self._turns.clear()
        self.context.properties_checked.clear()
        self.context.variables_mentioned.clear()
