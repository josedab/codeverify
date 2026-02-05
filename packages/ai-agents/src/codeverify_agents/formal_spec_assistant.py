"""Formal Specification Assistant - Natural language to Z3 constraint conversion.

This module provides an intuitive interface for generating formal specifications
from natural language descriptions, making Z3 verification accessible to developers
without formal methods expertise.

Features:
- Natural language to Z3 constraint conversion
- Interactive refinement with clarifying questions
- Reusable specification library and templates
- Domain-specific specification patterns
- Validation and counterexample generation
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from .base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


# =============================================================================
# Enums and Data Classes
# =============================================================================


class SpecDomain(str, Enum):
    """Domain categories for specifications."""

    GENERAL = "general"
    NUMERIC = "numeric"
    STRING = "string"
    COLLECTION = "collection"
    FINANCIAL = "financial"
    AUTHENTICATION = "authentication"
    DATA_VALIDATION = "data_validation"
    CONCURRENCY = "concurrency"
    MEMORY = "memory"


class SpecComplexity(str, Enum):
    """Complexity level of specifications."""

    SIMPLE = "simple"  # Single constraint
    MODERATE = "moderate"  # Multiple constraints, no quantifiers
    COMPLEX = "complex"  # Quantifiers, implications
    ADVANCED = "advanced"  # Nested quantifiers, complex logic


@dataclass
class SpecTemplate:
    """Reusable specification template."""

    id: str
    name: str
    domain: SpecDomain
    complexity: SpecComplexity

    # Natural language pattern (with placeholders)
    nl_pattern: str

    # Z3 expression template
    z3_template: str

    # SMT-LIB template
    smtlib_template: str

    # Python assertion template
    python_template: str

    # Variable placeholders
    variables: list[dict[str, str]]  # [{"name": "x", "type": "Int", "description": "..."}]

    # Example usages
    examples: list[dict[str, str]] = field(default_factory=list)

    # Usage count for popularity ranking
    usage_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain.value,
            "complexity": self.complexity.value,
            "nl_pattern": self.nl_pattern,
            "z3_template": self.z3_template,
            "smtlib_template": self.smtlib_template,
            "python_template": self.python_template,
            "variables": self.variables,
            "examples": self.examples,
            "usage_count": self.usage_count,
        }


@dataclass
class ParsedSpec:
    """Result of parsing natural language specification."""

    original_text: str
    normalized_text: str

    # Identified components
    subject: str | None = None  # What the spec is about
    predicate: str | None = None  # The constraint type
    objects: list[str] = field(default_factory=list)  # Values/variables involved

    # Matched template
    matched_template: SpecTemplate | None = None
    template_confidence: float = 0.0

    # Generated Z3
    z3_expr: str | None = None
    smtlib: str | None = None
    python_assert: str | None = None

    # Variables extracted
    variables: dict[str, str] = field(default_factory=dict)  # name -> type

    # Ambiguities detected
    ambiguities: list[str] = field(default_factory=list)

    # Suggested clarifications
    clarification_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "subject": self.subject,
            "predicate": self.predicate,
            "objects": self.objects,
            "matched_template": self.matched_template.to_dict() if self.matched_template else None,
            "template_confidence": self.template_confidence,
            "z3_expr": self.z3_expr,
            "smtlib": self.smtlib,
            "python_assert": self.python_assert,
            "variables": self.variables,
            "ambiguities": self.ambiguities,
            "clarification_questions": self.clarification_questions,
        }


@dataclass
class ConversionResult:
    """Result of natural language to Z3 conversion."""

    success: bool
    parsed_spec: ParsedSpec

    # Generated formal specs
    z3_expr: str | None = None
    smtlib: str | None = None
    python_assert: str | None = None

    # Explanation of the conversion
    explanation: str = ""

    # Confidence in the conversion
    confidence: float = 0.0

    # Validation result if validated
    validated: bool = False
    validation_result: str | None = None
    counterexample: dict[str, Any] | None = None

    # Processing time
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "parsed_spec": self.parsed_spec.to_dict(),
            "z3_expr": self.z3_expr,
            "smtlib": self.smtlib,
            "python_assert": self.python_assert,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "validated": self.validated,
            "validation_result": self.validation_result,
            "counterexample": self.counterexample,
            "processing_time_ms": self.processing_time_ms,
        }


# =============================================================================
# Specification Template Library
# =============================================================================


class SpecLibrary:
    """Library of reusable specification templates."""

    def __init__(self) -> None:
        """Initialize with built-in templates."""
        self.templates: dict[str, SpecTemplate] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """Load built-in specification templates."""
        builtin = [
            # Numeric constraints
            SpecTemplate(
                id="positive",
                name="Positive Number",
                domain=SpecDomain.NUMERIC,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="{var} must be positive",
                z3_template="{var} > 0",
                smtlib_template="(assert (> {var} 0))",
                python_template="assert {var} > 0",
                variables=[{"name": "var", "type": "Int", "description": "The variable to check"}],
                examples=[
                    {"nl": "x must be positive", "z3": "x > 0"},
                    {"nl": "count must be positive", "z3": "count > 0"},
                ],
            ),
            SpecTemplate(
                id="non_negative",
                name="Non-negative Number",
                domain=SpecDomain.NUMERIC,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="{var} must be non-negative",
                z3_template="{var} >= 0",
                smtlib_template="(assert (>= {var} 0))",
                python_template="assert {var} >= 0",
                variables=[{"name": "var", "type": "Int", "description": "The variable to check"}],
                examples=[
                    {"nl": "index must be non-negative", "z3": "index >= 0"},
                ],
            ),
            SpecTemplate(
                id="range",
                name="Value in Range",
                domain=SpecDomain.NUMERIC,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="{var} must be between {min} and {max}",
                z3_template="And({var} >= {min}, {var} <= {max})",
                smtlib_template="(assert (and (>= {var} {min}) (<= {var} {max})))",
                python_template="assert {min} <= {var} <= {max}",
                variables=[
                    {"name": "var", "type": "Int", "description": "The variable to check"},
                    {"name": "min", "type": "Int", "description": "Minimum value"},
                    {"name": "max", "type": "Int", "description": "Maximum value"},
                ],
                examples=[
                    {"nl": "age must be between 0 and 150", "z3": "And(age >= 0, age <= 150)"},
                ],
            ),
            SpecTemplate(
                id="less_than",
                name="Less Than",
                domain=SpecDomain.NUMERIC,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="{var1} must be less than {var2}",
                z3_template="{var1} < {var2}",
                smtlib_template="(assert (< {var1} {var2}))",
                python_template="assert {var1} < {var2}",
                variables=[
                    {"name": "var1", "type": "Int", "description": "First variable"},
                    {"name": "var2", "type": "Int", "description": "Second variable"},
                ],
                examples=[
                    {"nl": "start must be less than end", "z3": "start < end"},
                ],
            ),
            # Nullability
            SpecTemplate(
                id="not_null",
                name="Not Null",
                domain=SpecDomain.GENERAL,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="{var} must not be null",
                z3_template="{var} != None",
                smtlib_template="(assert (not (= {var} nil)))",
                python_template="assert {var} is not None",
                variables=[{"name": "var", "type": "Any", "description": "The variable to check"}],
                examples=[
                    {"nl": "user must not be null", "z3": "user != None"},
                ],
            ),
            SpecTemplate(
                id="not_empty",
                name="Not Empty",
                domain=SpecDomain.COLLECTION,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="{var} must not be empty",
                z3_template="Length({var}) > 0",
                smtlib_template="(assert (> (seq.len {var}) 0))",
                python_template="assert len({var}) > 0",
                variables=[{"name": "var", "type": "Seq", "description": "The collection to check"}],
                examples=[
                    {"nl": "items must not be empty", "z3": "Length(items) > 0"},
                ],
            ),
            # Array/Collection constraints
            SpecTemplate(
                id="valid_index",
                name="Valid Array Index",
                domain=SpecDomain.COLLECTION,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="{index} must be a valid index for {array}",
                z3_template="And({index} >= 0, {index} < Length({array}))",
                smtlib_template="(assert (and (>= {index} 0) (< {index} (seq.len {array}))))",
                python_template="assert 0 <= {index} < len({array})",
                variables=[
                    {"name": "index", "type": "Int", "description": "Array index"},
                    {"name": "array", "type": "Seq", "description": "Array/list"},
                ],
                examples=[
                    {"nl": "i must be a valid index for items", "z3": "And(i >= 0, i < Length(items))"},
                ],
            ),
            SpecTemplate(
                id="all_positive",
                name="All Elements Positive",
                domain=SpecDomain.COLLECTION,
                complexity=SpecComplexity.COMPLEX,
                nl_pattern="all elements of {array} must be positive",
                z3_template="ForAll([i], Implies(And(i >= 0, i < Length({array})), {array}[i] > 0))",
                smtlib_template="(assert (forall ((i Int)) (=> (and (>= i 0) (< i (seq.len {array}))) (> (seq.nth {array} i) 0))))",
                python_template="assert all(x > 0 for x in {array})",
                variables=[{"name": "array", "type": "Seq(Int)", "description": "Array of integers"}],
                examples=[
                    {"nl": "all elements of prices must be positive", "z3": "ForAll([i], Implies(And(i >= 0, i < Length(prices)), prices[i] > 0))"},
                ],
            ),
            # Implication patterns
            SpecTemplate(
                id="implies",
                name="If-Then Implication",
                domain=SpecDomain.GENERAL,
                complexity=SpecComplexity.MODERATE,
                nl_pattern="if {condition} then {consequence}",
                z3_template="Implies({condition}, {consequence})",
                smtlib_template="(assert (=> {condition} {consequence}))",
                python_template="assert not {condition} or {consequence}",
                variables=[
                    {"name": "condition", "type": "Bool", "description": "Condition"},
                    {"name": "consequence", "type": "Bool", "description": "Consequence"},
                ],
                examples=[
                    {"nl": "if is_admin then can_delete", "z3": "Implies(is_admin, can_delete)"},
                ],
            ),
            # Return value patterns
            SpecTemplate(
                id="returns_positive",
                name="Returns Positive",
                domain=SpecDomain.NUMERIC,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="the function returns a positive value",
                z3_template="result > 0",
                smtlib_template="(assert (> result 0))",
                python_template="assert result > 0",
                variables=[{"name": "result", "type": "Int", "description": "Return value"}],
                examples=[
                    {"nl": "the function returns a positive value", "z3": "result > 0"},
                ],
            ),
            SpecTemplate(
                id="preserves_sum",
                name="Preserves Sum",
                domain=SpecDomain.NUMERIC,
                complexity=SpecComplexity.MODERATE,
                nl_pattern="the sum of {var1} and {var2} must equal {total}",
                z3_template="{var1} + {var2} == {total}",
                smtlib_template="(assert (= (+ {var1} {var2}) {total}))",
                python_template="assert {var1} + {var2} == {total}",
                variables=[
                    {"name": "var1", "type": "Int", "description": "First value"},
                    {"name": "var2", "type": "Int", "description": "Second value"},
                    {"name": "total", "type": "Int", "description": "Expected total"},
                ],
                examples=[
                    {"nl": "the sum of credit and debit must equal balance", "z3": "credit + debit == balance"},
                ],
            ),
            # String patterns
            SpecTemplate(
                id="string_length_range",
                name="String Length in Range",
                domain=SpecDomain.STRING,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="{var} length must be between {min} and {max}",
                z3_template="And(Length({var}) >= {min}, Length({var}) <= {max})",
                smtlib_template="(assert (and (>= (str.len {var}) {min}) (<= (str.len {var}) {max})))",
                python_template="assert {min} <= len({var}) <= {max}",
                variables=[
                    {"name": "var", "type": "String", "description": "String variable"},
                    {"name": "min", "type": "Int", "description": "Minimum length"},
                    {"name": "max", "type": "Int", "description": "Maximum length"},
                ],
                examples=[
                    {"nl": "password length must be between 8 and 128", "z3": "And(Length(password) >= 8, Length(password) <= 128)"},
                ],
            ),
            # Financial patterns
            SpecTemplate(
                id="non_negative_balance",
                name="Non-negative Balance",
                domain=SpecDomain.FINANCIAL,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="{account} balance must never go negative",
                z3_template="{account}_balance >= 0",
                smtlib_template="(assert (>= {account}_balance 0))",
                python_template="assert {account}.balance >= 0",
                variables=[{"name": "account", "type": "Account", "description": "Account object"}],
                examples=[
                    {"nl": "checking balance must never go negative", "z3": "checking_balance >= 0"},
                ],
            ),
            # Authentication patterns
            SpecTemplate(
                id="authenticated",
                name="User Must Be Authenticated",
                domain=SpecDomain.AUTHENTICATION,
                complexity=SpecComplexity.SIMPLE,
                nl_pattern="{user} must be authenticated",
                z3_template="{user}_authenticated == True",
                smtlib_template="(assert {user}_authenticated)",
                python_template="assert {user}.is_authenticated",
                variables=[{"name": "user", "type": "User", "description": "User object"}],
                examples=[
                    {"nl": "current_user must be authenticated", "z3": "current_user_authenticated == True"},
                ],
            ),
            SpecTemplate(
                id="authorized",
                name="User Must Be Authorized",
                domain=SpecDomain.AUTHENTICATION,
                complexity=SpecComplexity.MODERATE,
                nl_pattern="{user} must have {permission} permission",
                z3_template="Contains({user}_permissions, {permission})",
                smtlib_template="(assert (str.contains {user}_permissions {permission}))",
                python_template="assert '{permission}' in {user}.permissions",
                variables=[
                    {"name": "user", "type": "User", "description": "User object"},
                    {"name": "permission", "type": "String", "description": "Required permission"},
                ],
                examples=[
                    {"nl": "admin must have delete permission", "z3": "Contains(admin_permissions, 'delete')"},
                ],
            ),
        ]

        for template in builtin:
            self.templates[template.id] = template

    def get_template(self, template_id: str) -> SpecTemplate | None:
        """Get template by ID."""
        return self.templates.get(template_id)

    def search_templates(
        self,
        query: str,
        domain: SpecDomain | None = None,
        max_results: int = 5,
    ) -> list[SpecTemplate]:
        """Search templates by natural language query."""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for template in self.templates.values():
            if domain and template.domain != domain:
                continue

            # Score based on pattern match
            pattern_lower = template.nl_pattern.lower()
            pattern_words = set(re.sub(r"\{[^}]+\}", "", pattern_lower).split())

            # Word overlap score
            overlap = len(query_words & pattern_words)
            name_match = 1.0 if query_lower in template.name.lower() else 0.0

            # Check examples
            example_match = 0.0
            for example in template.examples:
                if query_lower in example.get("nl", "").lower():
                    example_match = 0.5
                    break

            score = overlap * 0.5 + name_match + example_match + template.usage_count * 0.01

            if score > 0:
                results.append((score, template))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in results[:max_results]]

    def add_template(self, template: SpecTemplate) -> None:
        """Add a custom template."""
        self.templates[template.id] = template

    def increment_usage(self, template_id: str) -> None:
        """Increment usage count for a template."""
        if template_id in self.templates:
            self.templates[template_id].usage_count += 1

    def get_by_domain(self, domain: SpecDomain) -> list[SpecTemplate]:
        """Get all templates for a domain."""
        return [t for t in self.templates.values() if t.domain == domain]

    def export_library(self) -> list[dict[str, Any]]:
        """Export library as JSON-serializable list."""
        return [t.to_dict() for t in self.templates.values()]


# =============================================================================
# Natural Language Parser
# =============================================================================


class NLSpecParser:
    """Parser for natural language specifications."""

    # Common patterns for spec components
    PATTERNS = {
        "positive": r"(?:must be |is |should be )?positive",
        "negative": r"(?:must be |is |should be )?negative",
        "non_negative": r"(?:must be |is |should be )?(?:non-negative|>= ?0|not negative)",
        "not_null": r"(?:must not be |cannot be |is not |should not be )?(?:null|None|nil)",
        "not_empty": r"(?:must not be |cannot be |is not )?empty",
        "range": r"(?:must be |is |should be )?(?:between|in range) (\d+) (?:and|to) (\d+)",
        "less_than": r"(?:must be |is |should be )?less than (\w+|\d+)",
        "greater_than": r"(?:must be |is |should be )?greater than (\w+|\d+)",
        "equal": r"(?:must |should )?equal(?:s)? (\w+|\d+)",
        "implies": r"if (.+) then (.+)",
        "forall": r"(?:all|every|each) (\w+) (?:in |of )?(\w+) (?:must |should )?(.+)",
    }

    # Variable extraction patterns
    VAR_PATTERNS = [
        r"\b([a-z_][a-z0-9_]*)\b(?= must| should| is| cannot| has)",
        r"\bthe (\w+)\b",
        r"^(\w+)\b",
    ]

    def parse(self, text: str) -> ParsedSpec:
        """Parse natural language specification."""
        original = text
        normalized = self._normalize(text)

        spec = ParsedSpec(
            original_text=original,
            normalized_text=normalized,
        )

        # Extract subject (what is being constrained)
        spec.subject = self._extract_subject(normalized)

        # Extract predicate (type of constraint)
        spec.predicate, spec.objects = self._extract_predicate(normalized)

        # Extract variables
        spec.variables = self._extract_variables(normalized)

        # Detect ambiguities
        spec.ambiguities = self._detect_ambiguities(normalized, spec)

        # Generate clarification questions
        spec.clarification_questions = self._generate_clarifications(spec)

        return spec

    def _normalize(self, text: str) -> str:
        """Normalize input text."""
        # Lowercase
        text = text.lower().strip()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Standardize common phrases
        replacements = [
            (r"can't|cannot", "must not"),
            (r"shouldn't", "should not"),
            (r"isn't", "is not"),
            (r"won't", "will not"),
            (r"!=", "not equal to"),
            (r">=", "greater than or equal to"),
            (r"<=", "less than or equal to"),
            (r">", "greater than"),
            (r"<", "less than"),
        ]

        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

        return text

    def _extract_subject(self, text: str) -> str | None:
        """Extract the subject of the specification."""
        for pattern in self.VAR_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_predicate(self, text: str) -> tuple[str | None, list[str]]:
        """Extract predicate and objects."""
        for pred_name, pattern in self.PATTERNS.items():
            match = re.search(pattern, text)
            if match:
                objects = list(match.groups()) if match.groups() else []
                return pred_name, objects
        return None, []

    def _extract_variables(self, text: str) -> dict[str, str]:
        """Extract variables and infer types."""
        variables = {}

        # Find all potential variable names
        var_pattern = r"\b([a-z_][a-z0-9_]*)\b"
        for match in re.finditer(var_pattern, text):
            var_name = match.group(1)

            # Skip common words
            skip_words = {
                "must", "should", "be", "is", "not", "null", "none", "empty",
                "positive", "negative", "between", "and", "or", "if", "then",
                "all", "every", "each", "in", "of", "the", "a", "an", "to",
                "greater", "less", "than", "equal", "equals", "true", "false",
            }

            if var_name not in skip_words:
                # Infer type from context
                var_type = self._infer_type(var_name, text)
                variables[var_name] = var_type

        return variables

    def _infer_type(self, var_name: str, text: str) -> str:
        """Infer variable type from context."""
        # Check for type hints in text
        type_patterns = [
            (r"string|text|name|password|email", "String"),
            (r"array|list|items|elements|collection", "Seq"),
            (r"count|index|number|age|size|length|amount|balance|price", "Int"),
            (r"rate|percentage|ratio|factor", "Real"),
            (r"flag|is_|has_|can_|should_|enabled|active|valid", "Bool"),
        ]

        for pattern, typ in type_patterns:
            if re.search(pattern, var_name):
                return typ

        # Default to Int for numeric contexts, String for others
        if re.search(r"positive|negative|\d+|between|greater|less", text):
            return "Int"

        return "Any"

    def _detect_ambiguities(self, text: str, spec: ParsedSpec) -> list[str]:
        """Detect potential ambiguities."""
        ambiguities = []

        # No clear subject
        if not spec.subject:
            ambiguities.append("Could not identify the variable being constrained")

        # Unclear comparison
        if "than" in text and not spec.predicate:
            ambiguities.append("Comparison target is unclear")

        # Multiple constraints without clear combination
        constraint_words = ["and", "or", "but"]
        if sum(1 for w in constraint_words if w in text) > 1:
            ambiguities.append("Multiple constraints - unclear how they should be combined")

        # Range without both bounds
        if "between" in text or "range" in text:
            numbers = re.findall(r"\d+", text)
            if len(numbers) < 2:
                ambiguities.append("Range specified but missing upper or lower bound")

        return ambiguities

    def _generate_clarifications(self, spec: ParsedSpec) -> list[str]:
        """Generate clarification questions."""
        questions = []

        if not spec.subject:
            questions.append("Which variable should this constraint apply to?")

        if "positive" in (spec.predicate or "") and spec.subject:
            questions.append(f"Should {spec.subject} be strictly positive (> 0) or non-negative (>= 0)?")

        if spec.predicate == "range" and len(spec.objects) < 2:
            questions.append("What should the upper and lower bounds be?")

        for var, typ in spec.variables.items():
            if typ == "Any":
                questions.append(f"What is the type of '{var}'? (int, string, list, etc.)")

        return questions


# =============================================================================
# Formal Specification Assistant Agent
# =============================================================================


class FormalSpecAssistant(BaseAgent):
    """AI agent for converting natural language to formal Z3 specifications.

    This agent makes formal verification accessible by:
    1. Parsing natural language requirements
    2. Matching against specification templates
    3. Using LLM for complex conversions
    4. Validating generated specs with Z3
    5. Supporting interactive refinement

    Example usage:
        assistant = FormalSpecAssistant()
        result = await assistant.convert("x must be positive")
        print(result.z3_expr)  # "x > 0"
    """

    SYSTEM_PROMPT = """You are an expert in formal methods and program verification.
Your task is to convert natural language specifications into Z3 formal constraints.

RESPOND ONLY WITH VALID JSON in this format:
{
    "z3_expr": "Z3 Python expression using z3 library syntax",
    "smtlib": "SMT-LIB format assertion",
    "python_assert": "Python assert statement",
    "variables": {"var_name": "Z3_type"},
    "explanation": "Brief explanation of the conversion",
    "confidence": 0.9,
    "alternatives": ["alternative interpretation if ambiguous"]
}

Z3 SYNTAX GUIDE:
- Comparisons: x > 0, x >= y, x == y, x != y
- Logic: And(a, b), Or(a, b), Not(a), Implies(a, b)
- Quantifiers: ForAll([x], expr), Exists([x], expr)
- Arithmetic: x + y, x - y, x * y, x / y, x % y
- Sequences: Length(s), s[i], Concat(s1, s2)
- Strings: Length(s), Contains(s, sub), Prefix(s, pre)
- Conditional: If(cond, then_expr, else_expr)

Variable types:
- Int: integers
- Real: real numbers
- Bool: booleans
- String: strings (use StringSort())
- Array(K, V): arrays/maps
- Seq(T): sequences/lists

IMPORTANT:
- Use 'result' for return values
- Use descriptive variable names from the input
- Prefer simpler expressions when equivalent
- Include all necessary variable declarations
"""

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the formal spec assistant."""
        super().__init__(config)
        self.library = SpecLibrary()
        self.parser = NLSpecParser()
        self._conversion_cache: dict[str, ConversionResult] = {}

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """Analyze natural language spec in code context."""
        start_time = time.time()

        try:
            nl_spec = context.get("specification", code)
            result = await self.convert(nl_spec, context)

            return AgentResult(
                success=result.success,
                data=result.to_dict(),
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("Formal spec conversion failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def convert(
        self,
        natural_language: str,
        context: dict[str, Any] | None = None,
    ) -> ConversionResult:
        """Convert natural language specification to Z3.

        Args:
            natural_language: The specification in natural language
            context: Optional context (function signature, types, etc.)

        Returns:
            ConversionResult with Z3 expression and metadata
        """
        start_time = time.time()
        context = context or {}

        # Check cache
        cache_key = hashlib.md5(natural_language.encode()).hexdigest()
        if cache_key in self._conversion_cache:
            cached = self._conversion_cache[cache_key]
            cached.processing_time_ms = (time.time() - start_time) * 1000
            return cached

        # Parse the natural language
        parsed = self.parser.parse(natural_language)

        # Try template matching first (fast path)
        template_result = self._try_template_match(parsed, context)
        if template_result and template_result.confidence > 0.8:
            template_result.processing_time_ms = (time.time() - start_time) * 1000
            self._conversion_cache[cache_key] = template_result
            return template_result

        # Fall back to LLM conversion
        llm_result = await self._llm_convert(parsed, context)
        llm_result.processing_time_ms = (time.time() - start_time) * 1000

        # Cache result
        self._conversion_cache[cache_key] = llm_result

        logger.info(
            "Converted natural language to Z3",
            input=natural_language[:50],
            confidence=llm_result.confidence,
            processing_time_ms=llm_result.processing_time_ms,
        )

        return llm_result

    def _try_template_match(
        self,
        parsed: ParsedSpec,
        context: dict[str, Any],
    ) -> ConversionResult | None:
        """Try to match against template library."""
        # Search templates based on parsed spec
        search_query = f"{parsed.predicate or ''} {parsed.subject or ''}"
        templates = self.library.search_templates(search_query, max_results=3)

        if not templates:
            return None

        best_template = templates[0]

        # Extract variable values from parsed spec
        var_mapping = {}
        if parsed.subject:
            var_mapping["var"] = parsed.subject
            var_mapping["var1"] = parsed.subject

        for i, obj in enumerate(parsed.objects):
            var_mapping[f"var{i+2}"] = obj
            if i == 0:
                var_mapping["min"] = obj
            elif i == 1:
                var_mapping["max"] = obj

        # Fill in template
        try:
            z3_expr = best_template.z3_template
            smtlib = best_template.smtlib_template
            python_assert = best_template.python_template

            for key, value in var_mapping.items():
                z3_expr = z3_expr.replace(f"{{{key}}}", value)
                smtlib = smtlib.replace(f"{{{key}}}", value)
                python_assert = python_assert.replace(f"{{{key}}}", value)

            # Check if all placeholders filled
            if "{" in z3_expr:
                return None

            # Increment usage
            self.library.increment_usage(best_template.id)

            parsed.matched_template = best_template
            parsed.template_confidence = 0.9
            parsed.z3_expr = z3_expr
            parsed.smtlib = smtlib
            parsed.python_assert = python_assert

            return ConversionResult(
                success=True,
                parsed_spec=parsed,
                z3_expr=z3_expr,
                smtlib=smtlib,
                python_assert=python_assert,
                explanation=f"Matched template: {best_template.name}",
                confidence=0.9,
            )

        except Exception as e:
            logger.debug("Template matching failed", error=str(e))
            return None

    async def _llm_convert(
        self,
        parsed: ParsedSpec,
        context: dict[str, Any],
    ) -> ConversionResult:
        """Use LLM for conversion when templates don't match."""
        prompt = f"""Convert this natural language specification to Z3:

Input: "{parsed.original_text}"

Parsed components:
- Subject: {parsed.subject}
- Predicate: {parsed.predicate}
- Objects: {parsed.objects}
- Detected variables: {parsed.variables}

Additional context:
{json.dumps(context, indent=2) if context else "None"}

Convert to Z3 formal specification."""

        try:
            response = await self._call_llm(
                self.SYSTEM_PROMPT,
                prompt,
                json_mode=True,
            )

            data = json.loads(response["content"])

            parsed.z3_expr = data.get("z3_expr")
            parsed.smtlib = data.get("smtlib")
            parsed.python_assert = data.get("python_assert")

            # Update variables from LLM
            if data.get("variables"):
                parsed.variables.update(data["variables"])

            return ConversionResult(
                success=True,
                parsed_spec=parsed,
                z3_expr=data.get("z3_expr"),
                smtlib=data.get("smtlib"),
                python_assert=data.get("python_assert"),
                explanation=data.get("explanation", ""),
                confidence=data.get("confidence", 0.7),
            )

        except Exception as e:
            logger.error("LLM conversion failed", error=str(e))
            return ConversionResult(
                success=False,
                parsed_spec=parsed,
                explanation=f"Conversion failed: {str(e)}",
                confidence=0.0,
            )

    async def convert_batch(
        self,
        specifications: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[ConversionResult]:
        """Convert multiple specifications."""
        results = []
        for spec in specifications:
            result = await self.convert(spec, context)
            results.append(result)
        return results

    async def validate_spec(
        self,
        z3_expr: str,
        variables: dict[str, str],
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Validate a Z3 specification.

        Returns:
            Tuple of (is_satisfiable, message, model/counterexample)
        """
        try:
            from z3 import (
                And, Array, Bool, ForAll, If, Implies, Int, IntSort, Not, Or,
                Real, Solver, String, sat, unknown, unsat
            )

            solver = Solver()
            solver.set("timeout", 5000)

            # Create variables
            local_vars: dict[str, Any] = {
                "And": And, "Or": Or, "Not": Not, "Implies": Implies,
                "ForAll": ForAll, "If": If,
                "Int": Int, "Real": Real, "Bool": Bool, "String": String,
                "Array": Array, "IntSort": IntSort,
            }

            for var_name, var_type in variables.items():
                if var_type == "Int":
                    local_vars[var_name] = Int(var_name)
                elif var_type == "Real":
                    local_vars[var_name] = Real(var_name)
                elif var_type == "Bool":
                    local_vars[var_name] = Bool(var_name)
                else:
                    local_vars[var_name] = Int(var_name)  # Default

            # Evaluate and add constraint
            constraint = eval(z3_expr, {"__builtins__": {}}, local_vars)
            solver.add(constraint)

            result = solver.check()

            if result == sat:
                model = solver.model()
                model_dict = {str(d): str(model[d]) for d in model.decls()}
                return True, "Specification is satisfiable", model_dict
            elif result == unsat:
                return False, "Specification is unsatisfiable (contradictory)", None
            else:
                return False, "Verification timed out", None

        except ImportError:
            return False, "Z3 not available", None
        except Exception as e:
            return False, f"Validation error: {str(e)}", None

    async def refine_with_feedback(
        self,
        original_spec: str,
        feedback: str,
        current_result: ConversionResult,
    ) -> ConversionResult:
        """Refine a specification based on user feedback.

        Args:
            original_spec: Original natural language specification
            feedback: User's feedback or correction
            current_result: Current conversion result

        Returns:
            Refined ConversionResult
        """
        prompt = f"""Refine this specification based on user feedback:

Original specification: "{original_spec}"

Current Z3 expression: {current_result.z3_expr}

User feedback: "{feedback}"

Provide a corrected/refined specification that addresses the feedback."""

        try:
            response = await self._call_llm(
                self.SYSTEM_PROMPT,
                prompt,
                json_mode=True,
            )

            data = json.loads(response["content"])

            # Create new parsed spec
            parsed = current_result.parsed_spec
            parsed.z3_expr = data.get("z3_expr")
            parsed.smtlib = data.get("smtlib")
            parsed.python_assert = data.get("python_assert")

            return ConversionResult(
                success=True,
                parsed_spec=parsed,
                z3_expr=data.get("z3_expr"),
                smtlib=data.get("smtlib"),
                python_assert=data.get("python_assert"),
                explanation=f"Refined based on feedback: {data.get('explanation', '')}",
                confidence=data.get("confidence", 0.8),
            )

        except Exception as e:
            logger.error("Refinement failed", error=str(e))
            return current_result

    def suggest_specifications(
        self,
        function_signature: str,
        docstring: str | None = None,
    ) -> list[str]:
        """Suggest specifications based on function signature.

        Args:
            function_signature: Function signature (e.g., "def foo(x: int, y: int) -> int:")
            docstring: Optional docstring

        Returns:
            List of suggested natural language specifications
        """
        suggestions = []

        # Parse signature
        params = re.findall(r"(\w+)\s*:\s*(\w+)", function_signature)
        return_match = re.search(r"->\s*(\w+)", function_signature)

        for param_name, param_type in params:
            if param_type in ("int", "Int", "integer"):
                suggestions.append(f"{param_name} must be positive")
                suggestions.append(f"{param_name} must be non-negative")
            elif param_type in ("str", "String", "string"):
                suggestions.append(f"{param_name} must not be empty")
            elif "list" in param_type.lower() or "array" in param_type.lower():
                suggestions.append(f"{param_name} must not be empty")

        if return_match:
            return_type = return_match.group(1)
            if return_type in ("int", "Int"):
                suggestions.append("the function returns a positive value")
                suggestions.append("the function returns a non-negative value")

        # Add relationship suggestions if multiple params
        if len(params) >= 2:
            p1, p2 = params[0][0], params[1][0]
            suggestions.append(f"{p1} must be less than {p2}")

        return suggestions

    def get_template_library(self) -> list[dict[str, Any]]:
        """Get the specification template library."""
        return self.library.export_library()

    def get_templates_by_domain(self, domain: str) -> list[dict[str, Any]]:
        """Get templates for a specific domain."""
        try:
            domain_enum = SpecDomain(domain)
            templates = self.library.get_by_domain(domain_enum)
            return [t.to_dict() for t in templates]
        except ValueError:
            return []

    def get_statistics(self) -> dict[str, Any]:
        """Get assistant statistics."""
        return {
            "cached_conversions": len(self._conversion_cache),
            "template_count": len(self.library.templates),
            "popular_templates": [
                {"id": t.id, "name": t.name, "usage_count": t.usage_count}
                for t in sorted(
                    self.library.templates.values(),
                    key=lambda x: x.usage_count,
                    reverse=True,
                )[:5]
            ],
        }

    def clear_cache(self) -> None:
        """Clear conversion cache."""
        self._conversion_cache.clear()
