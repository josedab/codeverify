"""Verification-Aware Code Generation.

Provides context-aware prompting, inline assertion generation, and
verified suggestion ranking for AI code generation tools. Guides AI
to generate formally-correct code from the start.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AssertionType(str, Enum):
    """Type of inline assertion."""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    BOUNDS_CHECK = "bounds_check"
    NULL_CHECK = "null_check"
    TYPE_CHECK = "type_check"


class SuggestionRank(str, Enum):
    """Verification-based ranking for code suggestions."""
    VERIFIED = "verified"
    LIKELY_CORRECT = "likely_correct"
    UNVERIFIED = "unverified"
    RISKY = "risky"


@dataclass
class InlineAssertion:
    """An assertion generated for code."""
    assertion_type: AssertionType
    expression: str
    natural_language: str
    target_line: int = 0
    language: str = "python"

    def to_comment(self) -> str:
        comment_prefix = {"python": "#", "typescript": "//", "go": "//", "java": "//", "rust": "//"}
        prefix = comment_prefix.get(self.language, "#")
        return f"{prefix} {self.assertion_type.value}: {self.natural_language}"

    def to_code(self) -> str:
        if self.language == "python":
            return f"assert {self.expression}, '{self.natural_language}'"
        elif self.language == "rust":
            return f'debug_assert!({self.expression}, "{self.natural_language}");'
        elif self.language in ("go", "java", "typescript"):
            return f"// assert: {self.expression} — {self.natural_language}"
        return f"# assert: {self.expression}"


@dataclass
class VerifiedSuggestion:
    """A code suggestion with verification metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    code: str = ""
    rank: SuggestionRank = SuggestionRank.UNVERIFIED
    verification_score: float = 0.0
    assertions: list[InlineAssertion] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def is_verified(self) -> bool:
        return self.rank == SuggestionRank.VERIFIED


@dataclass
class GenerationContext:
    """Context extracted for verification-aware code generation."""
    function_name: str = ""
    parameters: list[dict[str, str]] = field(default_factory=list)
    return_type: str = ""
    docstring: str = ""
    existing_assertions: list[str] = field(default_factory=list)
    surrounding_code: str = ""
    language: str = "python"


class ContextExtractor:
    """Extracts verification context from code for prompt enrichment."""

    def extract(self, code: str, cursor_line: int, language: str = "python") -> GenerationContext:
        """Extract context from code around the cursor position."""
        lines = code.split("\n")
        ctx = GenerationContext(language=language)

        # Find the enclosing function
        for i in range(min(cursor_line, len(lines) - 1), -1, -1):
            line = lines[i]
            func_match = re.match(r"\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", line)
            if func_match:
                ctx.function_name = func_match.group(1)
                params_str = func_match.group(2)
                ctx.parameters = self._parse_params(params_str, language)
                # Check for return type
                ret_match = re.search(r"->\s*(.+?):", line)
                if ret_match:
                    ctx.return_type = ret_match.group(1).strip()
                # Check for docstring
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    doc_lines = []
                    for j in range(i + 1, min(i + 10, len(lines))):
                        doc_lines.append(lines[j])
                        if j > i + 1 and '"""' in lines[j]:
                            break
                    ctx.docstring = "\n".join(doc_lines)
                break

        # Extract existing assertions
        ctx.existing_assertions = [
            l.strip() for l in lines if "assert" in l.lower() and not l.strip().startswith("#")
        ]

        # Surrounding code context
        start = max(0, cursor_line - 10)
        end = min(len(lines), cursor_line + 10)
        ctx.surrounding_code = "\n".join(lines[start:end])

        return ctx

    def _parse_params(self, params_str: str, language: str) -> list[dict[str, str]]:
        params = []
        for p in params_str.split(","):
            p = p.strip()
            if not p or p == "self":
                continue
            if ":" in p:
                name, type_hint = p.split(":", 1)
                params.append({"name": name.strip(), "type": type_hint.strip()})
            else:
                params.append({"name": p, "type": "Any"})
        return params


class AssertionGenerator:
    """Generates inline assertions based on code context."""

    def generate(self, context: GenerationContext) -> list[InlineAssertion]:
        """Generate assertions for the given context."""
        assertions: list[InlineAssertion] = []

        for param in context.parameters:
            name = param["name"]
            ptype = param.get("type", "Any")

            # Null checks for optional types
            if "None" in ptype or "Optional" in ptype or ptype == "Any":
                assertions.append(InlineAssertion(
                    assertion_type=AssertionType.PRECONDITION,
                    expression=f"{name} is not None",
                    natural_language=f"{name} must not be None",
                    language=context.language,
                ))

            # Bounds checks for numeric types
            if ptype in ("int", "float", "number"):
                if "index" in name.lower() or "idx" in name.lower():
                    assertions.append(InlineAssertion(
                        assertion_type=AssertionType.BOUNDS_CHECK,
                        expression=f"{name} >= 0",
                        natural_language=f"{name} must be non-negative",
                        language=context.language,
                    ))

            # String non-empty checks
            if ptype == "str" and ("name" in name.lower() or "id" in name.lower()):
                assertions.append(InlineAssertion(
                    assertion_type=AssertionType.PRECONDITION,
                    expression=f"len({name}) > 0",
                    natural_language=f"{name} must not be empty",
                    language=context.language,
                ))

        # Return type assertions
        if context.return_type and context.return_type not in ("None", "void"):
            assertions.append(InlineAssertion(
                assertion_type=AssertionType.POSTCONDITION,
                expression="result is not None",
                natural_language=f"Function must return a valid {context.return_type}",
                language=context.language,
            ))

        return assertions


class SuggestionRanker:
    """Ranks code suggestions by verification score."""

    def rank_suggestions(
        self,
        suggestions: list[str],
        context: GenerationContext,
    ) -> list[VerifiedSuggestion]:
        """Rank multiple code suggestions by verification safety."""
        assertion_gen = AssertionGenerator()
        ranked: list[VerifiedSuggestion] = []

        for code in suggestions:
            score = self._compute_score(code, context)
            assertions = assertion_gen.generate(context)

            issues: list[str] = []
            if ".unwrap()" in code:
                issues.append("Uses .unwrap() which may panic")
                score -= 0.2
            if "# type: ignore" in code or "// @ts-ignore" in code:
                issues.append("Bypasses type checking")
                score -= 0.15
            if "eval(" in code or "exec(" in code:
                issues.append("Uses eval/exec which is a security risk")
                score -= 0.3

            score = max(0.0, min(1.0, score))

            if score >= 0.8:
                rank = SuggestionRank.VERIFIED
            elif score >= 0.5:
                rank = SuggestionRank.LIKELY_CORRECT
            elif score >= 0.3:
                rank = SuggestionRank.UNVERIFIED
            else:
                rank = SuggestionRank.RISKY

            ranked.append(VerifiedSuggestion(
                code=code,
                rank=rank,
                verification_score=score,
                assertions=assertions,
                issues=issues,
                explanation=f"Verification score: {score:.2f}",
            ))

        ranked.sort(key=lambda s: s.verification_score, reverse=True)
        return ranked

    def _compute_score(self, code: str, context: GenerationContext) -> float:
        """Compute a verification score for a code suggestion."""
        score = 0.5

        # Reward: has error handling
        if "try" in code or "except" in code or "catch" in code or "?" in code:
            score += 0.15
        # Reward: has type annotations
        if "->" in code or ":" in code:
            score += 0.1
        # Reward: has docstring
        if '"""' in code or "///" in code:
            score += 0.05
        # Reward: has assertions or guards
        if "assert" in code or "if " in code:
            score += 0.1
        # Penalty: very long (likely complex)
        if len(code.split("\n")) > 50:
            score -= 0.1

        return score


class VerificationAwareCodeGen:
    """Main entry point for verification-aware code generation."""

    def __init__(self) -> None:
        self._extractor = ContextExtractor()
        self._assertion_gen = AssertionGenerator()
        self._ranker = SuggestionRanker()

    def enrich_prompt(self, code: str, cursor_line: int, language: str = "python") -> str:
        """Generate a verification-enriched prompt for AI code generation."""
        context = self._extractor.extract(code, cursor_line, language)
        assertions = self._assertion_gen.generate(context)

        prompt_parts = [
            f"Generate code for function '{context.function_name}' that satisfies these constraints:",
        ]

        for a in assertions:
            prompt_parts.append(f"- {a.assertion_type.value}: {a.natural_language}")

        if context.docstring:
            prompt_parts.append(f"\nDocstring:\n{context.docstring}")
        if context.return_type:
            prompt_parts.append(f"\nReturn type: {context.return_type}")

        prompt_parts.append("\nGenerate code that is provably correct with respect to these constraints.")
        return "\n".join(prompt_parts)

    def rank_suggestions(self, suggestions: list[str], code: str, cursor_line: int, language: str = "python") -> list[VerifiedSuggestion]:
        context = self._extractor.extract(code, cursor_line, language)
        return self._ranker.rank_suggestions(suggestions, context)

    def generate_assertions(self, code: str, cursor_line: int, language: str = "python") -> list[InlineAssertion]:
        context = self._extractor.extract(code, cursor_line, language)
        return self._assertion_gen.generate(context)
