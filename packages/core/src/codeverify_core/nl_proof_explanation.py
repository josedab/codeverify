"""Natural Language Proof Explanation.

Generates human-readable explanations of Z3 proofs and counterexamples
using structured templates and LLM-compatible prompting.

Features:
- Counterexample-to-narrative translation
- Proof success summarization
- Context-aware explanation with source code references
- Multiple detail levels (brief, standard, detailed)
- PR comment formatting with Markdown
- Explanation caching for repeated patterns
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class DetailLevel(str, Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


class ExplanationType(str, Enum):
    COUNTEREXAMPLE = "counterexample"
    PROOF_SUCCESS = "proof_success"
    FINDING = "finding"
    FIX_SUGGESTION = "fix_suggestion"


@dataclass
class ProofExplanation:
    """A natural language explanation of a proof or counterexample."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    explanation_type: ExplanationType = ExplanationType.COUNTEREXAMPLE
    detail_level: DetailLevel = DetailLevel.STANDARD
    title: str = ""
    narrative: str = ""
    code_reference: str = ""
    line_number: int = 0
    fix_suggestion: str = ""
    markdown: str = ""
    content_hash: str = ""


@dataclass
class ExplanationContext:
    """Context for generating an explanation."""
    check_type: str = ""
    function_name: str = ""
    file_path: str = ""
    line: int = 0
    code_snippet: str = ""
    variable_assignments: dict[str, Any] = field(default_factory=dict)
    constraint_violated: str = ""
    z3_output: str = ""
    severity: str = "medium"


EXPLANATION_TEMPLATES: dict[str, dict[str, str]] = {
    "null_safety": {
        "brief": "**Null dereference**: `{var}` can be `None` at line {line}.",
        "standard": (
            "**Null Safety Violation** in `{func}` ({file}:{line})\n\n"
            "This function can fail when `{var}` is `None`. The Z3 solver found a concrete "
            "counterexample: `{var} = None`. When this value reaches line {line}, "
            "it will cause a `TypeError` or `AttributeError`.\n\n"
            "**Fix**: Add a null check before using `{var}`."
        ),
        "detailed": (
            "### Null Safety Violation\n\n"
            "**Location**: `{func}` in `{file}`, line {line}\n"
            "**Severity**: {severity}\n\n"
            "The Z3 SMT solver mathematically proved that this function can receive "
            "`{var} = None` as input. When this happens:\n\n"
            "1. The value `None` propagates to line {line}\n"
            "2. An operation on `{var}` (method call, attribute access, or arithmetic) "
            "will raise `TypeError` or `AttributeError`\n\n"
            "**Counterexample**: `{assignments}`\n\n"
            "**Recommended Fix**:\n```python\nif {var} is not None:\n    # ... safe to use {var}\n```"
        ),
    },
    "division_by_zero": {
        "brief": "**Division by zero**: `{var}` can be `0` at line {line}.",
        "standard": (
            "**Division by Zero** in `{func}` ({file}:{line})\n\n"
            "This function performs division where the divisor `{var}` can be zero. "
            "Z3 found: `{var} = 0`. This will raise `ZeroDivisionError` at runtime.\n\n"
            "**Fix**: Guard the division with `if {var} != 0`."
        ),
        "detailed": (
            "### Division by Zero\n\n"
            "**Location**: `{func}` in `{file}`, line {line}\n"
            "**Severity**: {severity}\n\n"
            "The Z3 solver proved that `{var}` can equal zero when used as a divisor. "
            "This is a mathematically proven bug, not a heuristic — there exists a "
            "concrete input where `{var} = 0`.\n\n"
            "**Counterexample**: `{assignments}`\n\n"
            "**Recommended Fix**:\n```python\nif {var} != 0:\n    result = numerator / {var}\nelse:\n    result = 0  # or raise ValueError\n```"
        ),
    },
    "array_bounds": {
        "brief": "**Out of bounds**: index `{var}` can exceed array length at line {line}.",
        "standard": (
            "**Array Bounds Violation** in `{func}` ({file}:{line})\n\n"
            "The index `{var}` can be out of bounds. Z3 found a case where "
            "`{var}` exceeds the array length, causing an `IndexError`.\n\n"
            "**Fix**: Add a bounds check: `if 0 <= {var} < len(array)`."
        ),
        "detailed": (
            "### Array Out of Bounds\n\n"
            "**Location**: `{func}` in `{file}`, line {line}\n\n"
            "**Counterexample**: `{assignments}`\n\n"
            "**Recommended Fix**:\n```python\nif 0 <= {var} < len(array):\n    value = array[{var}]\n```"
        ),
    },
    "proof_success": {
        "brief": "✅ `{func}` is verified: {check_type} guaranteed.",
        "standard": (
            "✅ **Verified**: `{func}` in `{file}`\n\n"
            "Z3 proved that {check_type} is satisfied for all possible inputs. "
            "This function is mathematically guaranteed to be safe."
        ),
        "detailed": (
            "### ✅ Verification Passed\n\n"
            "**Function**: `{func}` in `{file}`\n"
            "**Check**: {check_type}\n\n"
            "The Z3 SMT solver exhaustively proved that no input can violate the "
            "{check_type} property. This is a mathematical guarantee, not a test result."
        ),
    },
}


class ExplanationGenerator:
    """Generates natural language explanations."""

    def __init__(self) -> None:
        self._cache: dict[str, ProofExplanation] = {}

    def explain_counterexample(
        self, context: ExplanationContext, detail: DetailLevel = DetailLevel.STANDARD
    ) -> ProofExplanation:
        """Generate explanation for a counterexample."""
        cache_key = self._cache_key(context, detail)
        if cache_key in self._cache:
            return self._cache[cache_key]

        templates = EXPLANATION_TEMPLATES.get(context.check_type, {})
        template = templates.get(detail.value, "Finding in `{func}` at line {line}: {check_type} violation.")

        var = next(iter(context.variable_assignments.keys()), "unknown")
        assignments = ", ".join(f"{k}={v}" for k, v in context.variable_assignments.items())

        narrative = template.format(
            var=var, func=context.function_name, file=context.file_path,
            line=context.line, severity=context.severity,
            assignments=assignments, check_type=context.check_type,
        )

        explanation = ProofExplanation(
            explanation_type=ExplanationType.COUNTEREXAMPLE,
            detail_level=detail,
            title=f"{context.check_type} violation in {context.function_name}",
            narrative=narrative,
            code_reference=context.code_snippet,
            line_number=context.line,
            markdown=narrative,
            content_hash=cache_key,
        )

        self._cache[cache_key] = explanation
        return explanation

    def explain_proof_success(
        self, context: ExplanationContext, detail: DetailLevel = DetailLevel.STANDARD
    ) -> ProofExplanation:
        """Generate explanation for a successful proof."""
        templates = EXPLANATION_TEMPLATES.get("proof_success", {})
        template = templates.get(detail.value, "✅ `{func}` verified for {check_type}.")

        narrative = template.format(
            func=context.function_name, file=context.file_path,
            check_type=context.check_type,
        )

        return ProofExplanation(
            explanation_type=ExplanationType.PROOF_SUCCESS,
            detail_level=detail,
            title=f"{context.function_name} verified",
            narrative=narrative,
            markdown=narrative,
        )

    def generate_pr_comment(
        self, explanations: list[ProofExplanation]
    ) -> str:
        """Generate a formatted PR comment from multiple explanations."""
        if not explanations:
            return "✅ **CodeVerify**: No issues found."

        failures = [e for e in explanations if e.explanation_type == ExplanationType.COUNTEREXAMPLE]
        successes = [e for e in explanations if e.explanation_type == ExplanationType.PROOF_SUCCESS]

        parts: list[str] = ["## 🔍 CodeVerify Analysis\n"]

        if failures:
            parts.append(f"### ⚠️ {len(failures)} Issue{'s' if len(failures) > 1 else ''} Found\n")
            for e in failures:
                parts.append(e.markdown + "\n\n---\n")

        if successes:
            parts.append(f"\n### ✅ {len(successes)} Check{'s' if len(successes) > 1 else ''} Passed\n")
            for e in successes:
                parts.append(f"- {e.narrative}\n")

        return "\n".join(parts)

    def _cache_key(self, ctx: ExplanationContext, detail: DetailLevel) -> str:
        content = f"{ctx.check_type}:{ctx.function_name}:{ctx.line}:{detail.value}:{ctx.variable_assignments}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class NLProofExplanationService:
    """Main service for natural language proof explanations."""

    def __init__(self) -> None:
        self._generator = ExplanationGenerator()
        self._history: list[ProofExplanation] = []

    def explain(
        self, context: ExplanationContext, detail: DetailLevel = DetailLevel.STANDARD
    ) -> ProofExplanation:
        if context.variable_assignments:
            explanation = self._generator.explain_counterexample(context, detail)
        else:
            explanation = self._generator.explain_proof_success(context, detail)
        self._history.append(explanation)
        return explanation

    def generate_pr_comment(self, contexts: list[ExplanationContext]) -> str:
        explanations = [self.explain(ctx) for ctx in contexts]
        return self._generator.generate_pr_comment(explanations)

    def get_history(self) -> list[ProofExplanation]:
        return list(self._history)


# ─── Singleton Access ──────────────────────────────────────────────────

_nl_explain_instance: NLProofExplanationService | None = None

def get_nl_explanation_service() -> NLProofExplanationService:
    global _nl_explain_instance
    if _nl_explain_instance is None:
        _nl_explain_instance = NLProofExplanationService()
    return _nl_explain_instance

def reset_nl_explanation_service() -> None:
    global _nl_explain_instance
    _nl_explain_instance = None
