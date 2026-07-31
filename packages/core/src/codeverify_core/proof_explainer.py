"""LLM-Powered Proof Explainer.

Auto-generates plain-English explanations from Z3 counterexamples and proofs.
Converts cryptic SMT solver output into developer-friendly descriptions with
code examples, fix suggestions, and educational context.

Features:
- Z3 counterexample parsing and structuring
- Tiered explanation detail (summary → details → full proof)
- Fix suggestion generation from proof context
- Educational links for verification concepts
- Template-based explanation with LLM enrichment fallback
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ExplanationDetail(str, Enum):
    """Level of detail for proof explanations."""

    SUMMARY = "summary"
    DETAILED = "detailed"
    FULL_PROOF = "full_proof"


class ProofOutcome(str, Enum):
    """Outcome of a Z3 verification check."""

    PROVED_SAFE = "proved_safe"
    COUNTEREXAMPLE_FOUND = "counterexample_found"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class CheckCategory(str, Enum):
    """Category of the verification check."""

    NULL_SAFETY = "null_safety"
    ARRAY_BOUNDS = "array_bounds"
    INTEGER_OVERFLOW = "integer_overflow"
    DIVISION_BY_ZERO = "division_by_zero"
    TYPE_ERROR = "type_error"
    ASSERTION_VIOLATION = "assertion_violation"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    LOOP_INVARIANT = "loop_invariant"
    CUSTOM = "custom"


# Educational resources for each check category
_EDUCATIONAL_LINKS: dict[CheckCategory, dict[str, str]] = {
    CheckCategory.NULL_SAFETY: {
        "title": "Understanding Null Safety",
        "url": "https://docs.codeverify.dev/verification/null-safety",
        "description": "Null dereferences cause crashes when code accesses a variable that has no value.",
    },
    CheckCategory.ARRAY_BOUNDS: {
        "title": "Array Bounds Checking",
        "url": "https://docs.codeverify.dev/verification/array-bounds",
        "description": "Out-of-bounds access reads/writes memory beyond array limits, causing crashes or security vulnerabilities.",
    },
    CheckCategory.INTEGER_OVERFLOW: {
        "title": "Integer Overflow Prevention",
        "url": "https://docs.codeverify.dev/verification/integer-overflow",
        "description": "Arithmetic overflow wraps values unexpectedly, leading to incorrect calculations or security flaws.",
    },
    CheckCategory.DIVISION_BY_ZERO: {
        "title": "Division by Zero Prevention",
        "url": "https://docs.codeverify.dev/verification/division-by-zero",
        "description": "Division by zero causes runtime exceptions. Ensure divisors are validated before use.",
    },
    CheckCategory.TYPE_ERROR: {
        "title": "Type Safety Verification",
        "url": "https://docs.codeverify.dev/verification/type-safety",
        "description": "Type mismatches can cause unexpected behavior or crashes at runtime.",
    },
    CheckCategory.ASSERTION_VIOLATION: {
        "title": "Assertion Verification",
        "url": "https://docs.codeverify.dev/verification/assertions",
        "description": "Assertions express expected invariants. Violations indicate logic errors.",
    },
    CheckCategory.PRECONDITION: {
        "title": "Precondition Checking",
        "url": "https://docs.codeverify.dev/verification/preconditions",
        "description": "Preconditions define what must be true before a function runs.",
    },
    CheckCategory.POSTCONDITION: {
        "title": "Postcondition Checking",
        "url": "https://docs.codeverify.dev/verification/postconditions",
        "description": "Postconditions define what must be true after a function completes.",
    },
    CheckCategory.LOOP_INVARIANT: {
        "title": "Loop Invariant Verification",
        "url": "https://docs.codeverify.dev/verification/loop-invariants",
        "description": "Loop invariants are conditions that hold before and after each iteration.",
    },
    CheckCategory.CUSTOM: {
        "title": "Custom Verification Rules",
        "url": "https://docs.codeverify.dev/verification/custom-rules",
        "description": "Custom verification checks defined by your team's requirements.",
    },
}


@dataclass
class CounterexampleValue:
    """A single variable assignment in a counterexample."""

    variable: str
    value: Any
    type_name: str = "unknown"

    @property
    def display(self) -> str:
        if self.type_name == "string":
            return f'{self.variable} = "{self.value}"'
        return f"{self.variable} = {self.value}"


@dataclass
class ParsedCounterexample:
    """Structured representation of a Z3 counterexample."""

    values: list[CounterexampleValue] = field(default_factory=list)
    raw_output: str = ""

    def get_value(self, variable: str) -> CounterexampleValue | None:
        for v in self.values:
            if v.variable == variable:
                return v
        return None

    @property
    def variable_names(self) -> list[str]:
        return [v.variable for v in self.values]


@dataclass
class ProofExplanation:
    """A human-readable explanation of a Z3 verification result."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    check_category: CheckCategory = CheckCategory.CUSTOM
    outcome: ProofOutcome = ProofOutcome.UNKNOWN
    detail_level: ExplanationDetail = ExplanationDetail.SUMMARY
    summary: str = ""
    detailed_explanation: str = ""
    counterexample: ParsedCounterexample | None = None
    fix_suggestions: list[str] = field(default_factory=list)
    code_example: str = ""
    educational_link: dict[str, str] = field(default_factory=dict)
    function_name: str = ""
    file_path: str = ""
    line_number: int = 0
    raw_z3_output: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_safe(self) -> bool:
        return self.outcome == ProofOutcome.PROVED_SAFE

    def to_markdown(self) -> str:
        parts = []
        icon = (
            "✅"
            if self.is_safe
            else "❌"
            if self.outcome == ProofOutcome.COUNTEREXAMPLE_FOUND
            else "⚠️"
        )
        parts.append(f"{icon} **{self.check_category.value.replace('_', ' ').title()}**")
        parts.append(f"\n{self.summary}")

        if self.detail_level in (ExplanationDetail.DETAILED, ExplanationDetail.FULL_PROOF):
            if self.detailed_explanation:
                parts.append(f"\n**Details:** {self.detailed_explanation}")
            if self.counterexample and self.counterexample.values:
                parts.append("\n**Counterexample (values that trigger the bug):**")
                for v in self.counterexample.values:
                    parts.append(f"  - `{v.display}`")
            if self.fix_suggestions:
                parts.append("\n**Suggested Fixes:**")
                for i, fix in enumerate(self.fix_suggestions, 1):
                    parts.append(f"  {i}. {fix}")

        if self.detail_level == ExplanationDetail.FULL_PROOF:
            if self.code_example:
                parts.append(f"\n**Example Fix:**\n```python\n{self.code_example}\n```")
            if self.raw_z3_output:
                parts.append(
                    f"\n<details><summary>Raw Z3 Output</summary>\n\n```\n{self.raw_z3_output}\n```\n</details>"
                )

        if self.educational_link:
            parts.append(
                f"\n📚 [{self.educational_link.get('title', 'Learn more')}]({self.educational_link.get('url', '')})"
            )

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "check_category": self.check_category.value,
            "outcome": self.outcome.value,
            "detail_level": self.detail_level.value,
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "fix_suggestions": self.fix_suggestions,
            "code_example": self.code_example,
            "function_name": self.function_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "is_safe": self.is_safe,
        }


class CounterexampleParser:
    """Parses raw Z3 output into structured counterexamples."""

    _ASSIGNMENT_PATTERN = re.compile(r"(\w+)\s*(?:->|=|:=)\s*(.+?)(?:\s*\((\w+)\))?$", re.MULTILINE)
    _DEFINE_FUN_PATTERN = re.compile(r"\(define-fun\s+(\w+)\s*\(\)\s*(\w+)\s+(.+?)\)", re.DOTALL)

    def parse(self, raw_output: str) -> ParsedCounterexample:
        """Parse Z3 counterexample output into structured form."""
        if not raw_output or not raw_output.strip():
            return ParsedCounterexample(raw_output=raw_output)

        values: list[CounterexampleValue] = []

        # Try SMT-LIB define-fun format
        for match in self._DEFINE_FUN_PATTERN.finditer(raw_output):
            name, type_name, value = match.group(1), match.group(2), match.group(3).strip()
            value = self._clean_value(value)
            values.append(
                CounterexampleValue(
                    variable=name,
                    value=self._coerce_value(value, type_name),
                    type_name=type_name.lower(),
                )
            )

        # Try assignment format (name -> value)
        if not values:
            for match in self._ASSIGNMENT_PATTERN.finditer(raw_output):
                name = match.group(1)
                value = match.group(2).strip()
                type_name = match.group(3) or self._infer_type(value)
                values.append(
                    CounterexampleValue(
                        variable=name,
                        value=self._coerce_value(value, type_name),
                        type_name=type_name.lower(),
                    )
                )

        return ParsedCounterexample(values=values, raw_output=raw_output)

    def _clean_value(self, value: str) -> str:
        value = value.strip().strip("()")
        if value.startswith("-") and " " in value:
            parts = value.split()
            if len(parts) == 2 and parts[1].isdigit():
                return f"-{parts[1]}"
        return value

    def _coerce_value(self, value: str, type_name: str) -> Any:
        type_lower = type_name.lower()
        if type_lower in ("int", "integer", "bv32", "bv64"):
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        if type_lower in ("real", "float", "double"):
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if type_lower == "bool":
            return value.lower() in ("true", "1")
        return value

    def _infer_type(self, value: str) -> str:
        if value.lower() in ("true", "false"):
            return "bool"
        try:
            int(value)
            return "int"
        except ValueError:
            pass
        try:
            float(value)
            return "real"
        except ValueError:
            pass
        return "unknown"


# Template-based explanations for each category
_EXPLANATION_TEMPLATES: dict[CheckCategory, dict[str, str]] = {
    CheckCategory.NULL_SAFETY: {
        "safe": "The variable `{var}` is guaranteed to be non-null at this point. Z3 proved that no execution path reaches this dereference with a null value.",
        "unsafe": "The variable `{var}` can be null when accessed on line {line}. The verifier found an input where `{var}` is null, which would cause a NullReferenceError at runtime.",
        "fix": "Add a null check: `if {var} is not None:` before accessing `{var}`.",
    },
    CheckCategory.ARRAY_BOUNDS: {
        "safe": "Array access at index `{var}` is within bounds. Z3 proved the index is always between 0 and the array length.",
        "unsafe": "Array index `{var}` can be {value}, which is out of bounds for an array of this size. This would cause an IndexError at runtime.",
        "fix": "Add bounds checking: `if 0 <= {var} < len(array):` before the access.",
    },
    CheckCategory.INTEGER_OVERFLOW: {
        "safe": "The arithmetic operation is safe. Z3 proved the result stays within the representable range.",
        "unsafe": "The arithmetic operation involving `{var}` can overflow when the value is {value}. This wraps the result to an unexpected value.",
        "fix": "Use checked arithmetic or validate input ranges before the computation.",
    },
    CheckCategory.DIVISION_BY_ZERO: {
        "safe": "The divisor is guaranteed to be non-zero. Z3 proved no execution path reaches this division with a zero divisor.",
        "unsafe": "The divisor `{var}` can be zero (found input: {value}). This would cause a ZeroDivisionError at runtime.",
        "fix": "Add a zero check: `if {var} != 0:` before dividing, or provide a default value.",
    },
    CheckCategory.ASSERTION_VIOLATION: {
        "safe": "The assertion holds for all possible inputs. Z3 proved the condition is always true.",
        "unsafe": "The assertion can fail when {var} = {value}. This means the expected invariant does not hold for all inputs.",
        "fix": "Review the assertion condition and ensure the preceding logic guarantees it.",
    },
    CheckCategory.PRECONDITION: {
        "safe": "The precondition is satisfied by all callers. Z3 proved the required conditions hold at every call site.",
        "unsafe": "The precondition can be violated when {var} = {value}. Callers are not guaranteeing the expected input constraints.",
        "fix": "Validate inputs at the call site or add parameter validation at the function entry.",
    },
    CheckCategory.POSTCONDITION: {
        "safe": "The postcondition is guaranteed. Z3 proved the function always produces the expected output.",
        "unsafe": "The postcondition can be violated when {var} = {value}. The function does not always produce the expected result.",
        "fix": "Review the function logic to ensure the return value meets the postcondition for all inputs.",
    },
}


class ProofExplainerEngine:
    """Generates human-readable explanations from Z3 verification results."""

    def __init__(self) -> None:
        self._parser = CounterexampleParser()
        self._cache: dict[str, ProofExplanation] = {}

    def explain(
        self,
        check_category: CheckCategory,
        outcome: ProofOutcome,
        raw_z3_output: str = "",
        function_name: str = "",
        file_path: str = "",
        line_number: int = 0,
        variable_name: str = "",
        detail_level: ExplanationDetail = ExplanationDetail.DETAILED,
    ) -> ProofExplanation:
        """Generate a human-readable explanation for a verification result."""
        cache_key = hashlib.sha256(
            f"{check_category}:{outcome}:{raw_z3_output}:{detail_level}".encode()
        ).hexdigest()[:16]

        if cache_key in self._cache:
            return self._cache[cache_key]

        counterexample = None
        if outcome == ProofOutcome.COUNTEREXAMPLE_FOUND and raw_z3_output:
            counterexample = self._parser.parse(raw_z3_output)

        summary = self._generate_summary(
            check_category, outcome, counterexample, variable_name, line_number
        )
        detailed = self._generate_detailed(
            check_category, outcome, counterexample, variable_name, line_number
        )
        fixes = self._generate_fixes(check_category, outcome, counterexample, variable_name)
        code_example = self._generate_code_example(check_category, variable_name)
        edu_link = _EDUCATIONAL_LINKS.get(check_category, {})

        explanation = ProofExplanation(
            check_category=check_category,
            outcome=outcome,
            detail_level=detail_level,
            summary=summary,
            detailed_explanation=detailed,
            counterexample=counterexample,
            fix_suggestions=fixes,
            code_example=code_example,
            educational_link=edu_link,
            function_name=function_name,
            file_path=file_path,
            line_number=line_number,
            raw_z3_output=raw_z3_output,
        )

        self._cache[cache_key] = explanation
        logger.info("proof_explained", category=check_category.value, outcome=outcome.value)
        return explanation

    def explain_batch(
        self,
        results: list[dict[str, Any]],
        detail_level: ExplanationDetail = ExplanationDetail.DETAILED,
    ) -> list[ProofExplanation]:
        """Explain multiple verification results at once."""
        return [
            self.explain(
                check_category=CheckCategory(r.get("category", "custom")),
                outcome=ProofOutcome(r.get("outcome", "unknown")),
                raw_z3_output=r.get("raw_output", ""),
                function_name=r.get("function_name", ""),
                file_path=r.get("file_path", ""),
                line_number=r.get("line_number", 0),
                variable_name=r.get("variable_name", ""),
                detail_level=detail_level,
            )
            for r in results
        ]

    def clear_cache(self) -> None:
        self._cache.clear()

    def _generate_summary(
        self,
        category: CheckCategory,
        outcome: ProofOutcome,
        counterexample: ParsedCounterexample | None,
        variable: str,
        line: int,
    ) -> str:
        if outcome == ProofOutcome.PROVED_SAFE:
            return f"✅ Verified safe: {category.value.replace('_', ' ')} check passed. No violations possible."
        if outcome == ProofOutcome.TIMEOUT:
            return f"⚠️ Verification timed out for {category.value.replace('_', ' ')} check. Consider simplifying the code or increasing the timeout."
        if outcome == ProofOutcome.UNKNOWN:
            return f"⚠️ Verification result is inconclusive for {category.value.replace('_', ' ')} check."

        var_display = variable or "the expression"
        val_display = ""
        if counterexample and counterexample.values:
            first = counterexample.values[0]
            var_display = variable or first.variable
            val_display = str(first.value)

        templates = _EXPLANATION_TEMPLATES.get(category)
        if templates and "unsafe" in templates:
            return templates["unsafe"].format(var=var_display, value=val_display, line=line)

        return f"❌ Bug found: {category.value.replace('_', ' ')} violation detected for `{var_display}`."

    def _generate_detailed(
        self,
        category: CheckCategory,
        outcome: ProofOutcome,
        counterexample: ParsedCounterexample | None,
        variable: str,
        line: int,
    ) -> str:
        if outcome == ProofOutcome.PROVED_SAFE:
            templates = _EXPLANATION_TEMPLATES.get(category)
            if templates and "safe" in templates:
                return templates["safe"].format(var=variable or "the expression", line=line)
            return "The Z3 SMT solver exhaustively checked all possible inputs and proved this code is safe."

        if outcome != ProofOutcome.COUNTEREXAMPLE_FOUND:
            return ""

        parts = []
        parts.append(
            f"Z3 found a concrete input that triggers this {category.value.replace('_', ' ')} violation."
        )
        if counterexample and counterexample.values:
            parts.append("When the following values are used:")
            for v in counterexample.values:
                parts.append(f"  • {v.display}")
            parts.append("the code fails the safety check.")
        return " ".join(parts) if len(parts) <= 2 else "\n".join(parts)

    def _generate_fixes(
        self,
        category: CheckCategory,
        outcome: ProofOutcome,
        _counterexample: ParsedCounterexample | None,
        variable: str,
    ) -> list[str]:
        if outcome == ProofOutcome.PROVED_SAFE:
            return []

        fixes = []
        var = variable or "value"
        templates = _EXPLANATION_TEMPLATES.get(category)
        if templates and "fix" in templates:
            fixes.append(templates["fix"].format(var=var))

        if category == CheckCategory.NULL_SAFETY:
            fixes.append(f"Use a default value: `{var} = {var} or default_value`")
            fixes.append("Use Optional type annotation and handle None explicitly")
        elif category == CheckCategory.ARRAY_BOUNDS:
            fixes.append("Use `min(index, len(arr) - 1)` to clamp the index")
            fixes.append("Consider using `.get()` for dict-like access with defaults")
        elif category == CheckCategory.INTEGER_OVERFLOW:
            fixes.append("Use Python's arbitrary-precision integers where possible")
            fixes.append("Add range validation: `assert MIN <= value <= MAX`")
        elif category == CheckCategory.DIVISION_BY_ZERO:
            fixes.append(f"Use a safe division helper: `safe_div({var}, divisor, default=0)`")

        return fixes

    def _generate_code_example(self, category: CheckCategory, variable: str) -> str:
        var = variable or "value"
        examples: dict[CheckCategory, str] = {
            CheckCategory.NULL_SAFETY: f'# Before (unsafe)\nresult = {var}.strip()\n\n# After (safe)\nif {var} is not None:\n    result = {var}.strip()\nelse:\n    result = ""',
            CheckCategory.DIVISION_BY_ZERO: f"# Before (unsafe)\nresult = total / {var}\n\n# After (safe)\nif {var} != 0:\n    result = total / {var}\nelse:\n    result = 0  # or raise ValueError",
            CheckCategory.ARRAY_BOUNDS: f"# Before (unsafe)\nitem = arr[{var}]\n\n# After (safe)\nif 0 <= {var} < len(arr):\n    item = arr[{var}]\nelse:\n    item = None  # or handle error",
            CheckCategory.INTEGER_OVERFLOW: '# Before (unsafe)\nresult = a * b\n\n# After (safe)\nMAX_VAL = 2**31 - 1\nif a != 0 and abs(b) > MAX_VAL // abs(a):\n    raise OverflowError("multiplication overflow")\nresult = a * b',
        }
        return examples.get(category, "")


_engine: ProofExplainerEngine | None = None


def get_proof_explainer() -> ProofExplainerEngine:
    """Get the singleton ProofExplainerEngine instance."""
    global _engine
    if _engine is None:
        _engine = ProofExplainerEngine()
    return _engine


def reset_proof_explainer() -> None:
    """Reset the singleton (useful for testing)."""
    global _engine
    _engine = None
