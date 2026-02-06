"""Verification Explainer Agent - Explains Z3 proofs in plain English.

This module provides an AI agent that explains Z3 proofs and counterexamples
in plain English, with interactive tutorials teaching developers formal methods.

Key features:
1. Proof Explanation: Convert Z3 output to natural language narratives
2. Counterexample Walkthrough: Step-by-step execution trace explanation
3. Interactive Tutorials: "Why formal verification matters" learning paths
4. IDE Integration: Hover tooltips and explanation panels
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class ExplanationLevel(str, Enum):
    """Level of explanation detail."""

    BEGINNER = "beginner"  # ELI5-style
    INTERMEDIATE = "intermediate"  # Some technical terms
    ADVANCED = "advanced"  # Full technical detail


class ExplanationType(str, Enum):
    """Type of explanation."""

    PROOF = "proof"
    COUNTEREXAMPLE = "counterexample"
    VERIFICATION_RESULT = "verification_result"
    CONSTRAINT = "constraint"
    CONCEPT = "concept"


@dataclass
class Z3ParsedProof:
    """Parsed Z3 proof for explanation."""

    status: str  # "sat", "unsat", "unknown"
    variables: list[dict[str, Any]]
    constraints: list[str]
    counterexample: dict[str, Any] | None = None
    proof_tree: list[dict[str, Any]] | None = None
    raw_output: str = ""


@dataclass
class ProofExplanation:
    """Human-readable explanation of a proof."""

    summary: str
    detailed_explanation: str
    why_it_matters: str
    analogy: str | None = None
    code_walkthrough: list[dict[str, str]] | None = None
    recommendations: list[str] = field(default_factory=list)
    level: ExplanationLevel = ExplanationLevel.INTERMEDIATE
    related_concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "why_it_matters": self.why_it_matters,
            "analogy": self.analogy,
            "code_walkthrough": self.code_walkthrough,
            "recommendations": self.recommendations,
            "level": self.level.value,
            "related_concepts": self.related_concepts,
        }


@dataclass
class CounterexampleTrace:
    """Step-by-step trace of a counterexample."""

    step_number: int
    description: str
    variable_states: dict[str, Any]
    code_line: str | None = None
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "description": self.description,
            "variable_states": self.variable_states,
            "code_line": self.code_line,
            "explanation": self.explanation,
        }


@dataclass
class TutorialStep:
    """A step in an interactive tutorial."""

    id: str
    title: str
    content: str
    code_example: str | None = None
    quiz_question: str | None = None
    quiz_options: list[str] | None = None
    quiz_answer: int | None = None
    next_step_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "code_example": self.code_example,
            "quiz_question": self.quiz_question,
            "quiz_options": self.quiz_options,
            "next_step_id": self.next_step_id,
        }


@dataclass
class Tutorial:
    """An interactive tutorial on formal verification."""

    id: str
    title: str
    description: str
    steps: list[TutorialStep]
    difficulty: str  # "beginner", "intermediate", "advanced"
    estimated_minutes: int
    prerequisites: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "difficulty": self.difficulty,
            "estimated_minutes": self.estimated_minutes,
            "prerequisites": self.prerequisites,
        }


# Pre-built explanations for common patterns
PATTERN_EXPLANATIONS = {
    "null_check": {
        ExplanationLevel.BEGINNER: (
            "Think of a variable like a box. Sometimes the box is empty (null). "
            "If you try to look inside an empty box, your program crashes. "
            "This check makes sure the box has something in it before you look."
        ),
        ExplanationLevel.INTERMEDIATE: (
            "A null check verifies that a reference points to valid memory before "
            "attempting to dereference it. Without this check, accessing a null "
            "reference would result in a NullPointerException or similar error."
        ),
        ExplanationLevel.ADVANCED: (
            "The Z3 solver models the nullability of references as Boolean variables. "
            "The verification proves that along all execution paths, any dereference "
            "of the variable is guarded by a prior null check, ensuring the safety "
            "property: ∀paths. deref(x) → ¬null(x)."
        ),
    },
    "bounds_check": {
        ExplanationLevel.BEGINNER: (
            "Imagine a bookshelf with 5 spots, numbered 0 to 4. If you try to "
            "grab a book from spot 10, you'll reach into thin air! This check "
            "makes sure we only look at spots that actually exist."
        ),
        ExplanationLevel.INTERMEDIATE: (
            "Array bounds checking ensures that indices are within the valid range "
            "[0, length-1]. Accessing memory outside this range leads to undefined "
            "behavior, security vulnerabilities, or crashes."
        ),
        ExplanationLevel.ADVANCED: (
            "The verification encodes array accesses as: access(arr, i) → 0 ≤ i < len(arr). "
            "Z3 proves this holds for all possible values of i by attempting to find "
            "a counterexample where the index violates the bounds."
        ),
    },
    "division_check": {
        ExplanationLevel.BEGINNER: (
            "What's 10 divided by 0? It's undefined! Like asking how many groups of "
            "zero fit into ten - it doesn't make sense. This check ensures we never "
            "divide by zero."
        ),
        ExplanationLevel.INTERMEDIATE: (
            "Division by zero is mathematically undefined and causes runtime errors "
            "in most languages. The verification ensures the divisor can never be "
            "zero when a division occurs."
        ),
        ExplanationLevel.ADVANCED: (
            "The constraint div(a, b) → b ≠ 0 is added for each division operation. "
            "Z3 attempts to find values where b = 0 at the point of division. If "
            "unsatisfiable, the code is safe; if sat, the counterexample shows how."
        ),
    },
}

# Pre-built tutorials
TUTORIALS = {
    "intro_formal_verification": Tutorial(
        id="intro_formal_verification",
        title="Introduction to Formal Verification",
        description="Learn the basics of formal verification and why it matters",
        difficulty="beginner",
        estimated_minutes=15,
        steps=[
            TutorialStep(
                id="step1",
                title="What is Formal Verification?",
                content="""
Formal verification is like having a super-powered code reviewer that can
mathematically prove your code is correct - not just test a few examples,
but prove it works for ALL possible inputs.

Think of it like this:
- **Testing**: "I tried 1000 inputs and they all worked"
- **Formal Verification**: "I proved this works for ALL possible inputs"

This is incredibly powerful for catching bugs that tests might miss!
                """,
                next_step_id="step2",
            ),
            TutorialStep(
                id="step2",
                title="A Simple Example",
                content="""
Let's look at a simple function that might have a bug:

```python
def get_first_element(items):
    return items[0]
```

What could go wrong? The list might be empty!

Formal verification can PROVE this is unsafe by showing there exists
a case where `items` is empty and we crash.
                """,
                code_example="def get_first_element(items):\n    return items[0]",
                quiz_question="What's the bug in this code?",
                quiz_options=[
                    "It returns the wrong element",
                    "It might crash on an empty list",
                    "It's too slow",
                    "There's no bug",
                ],
                quiz_answer=1,
                next_step_id="step3",
            ),
            TutorialStep(
                id="step3",
                title="How Z3 Helps",
                content="""
Z3 is an SMT (Satisfiability Modulo Theories) solver - it can answer
questions like "Is there ANY input that makes this crash?"

For our example:
- Z3 asks: "Can items have length 0 when we access items[0]?"
- Z3 answers: "YES! Here's an example: items = []"

This is called a **counterexample** - proof that the code can fail.

Once we know the bug, we can fix it:

```python
def get_first_element(items):
    if len(items) > 0:
        return items[0]
    return None
```

Now Z3 can prove this version is safe!
                """,
                next_step_id="step4",
            ),
            TutorialStep(
                id="step4",
                title="Why This Matters",
                content="""
Formal verification catches bugs that testing misses:

1. **Edge cases**: Empty lists, None values, integer overflow
2. **Race conditions**: Hard to reproduce timing issues
3. **Security vulnerabilities**: SQL injection, buffer overflows

Industries using formal verification:
- **Aerospace**: Boeing, Airbus verify flight control software
- **Finance**: Banks verify transaction processing
- **Healthcare**: Medical device manufacturers verify safety

You don't need a PhD to use it - tools like CodeVerify bring formal
verification to everyday coding!
                """,
            ),
        ],
    ),
}


class Z3OutputParser:
    """Parses Z3 output into structured format."""

    def parse(self, z3_output: str) -> Z3ParsedProof:
        """Parse Z3 output."""
        status = "unknown"
        variables = []
        constraints = []
        counterexample = None

        lines = z3_output.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Parse status
            if line == "sat":
                status = "sat"
            elif line == "unsat":
                status = "unsat"
            elif line == "unknown":
                status = "unknown"

            # Parse variable declarations
            var_match = re.match(r"\(declare-(?:const|fun)\s+(\w+)\s+(\w+)\)", line)
            if var_match:
                variables.append({
                    "name": var_match.group(1),
                    "type": var_match.group(2),
                })

            # Parse assertions
            assert_match = re.match(r"\(assert\s+(.+)\)", line)
            if assert_match:
                constraints.append(assert_match.group(1))

        # Parse counterexample (model) if sat
        if status == "sat" and "(model" in z3_output:
            counterexample = self._parse_model(z3_output)

        return Z3ParsedProof(
            status=status,
            variables=variables,
            constraints=constraints,
            counterexample=counterexample,
            raw_output=z3_output,
        )

    def _parse_model(self, output: str) -> dict[str, Any]:
        """Parse Z3 model (counterexample)."""
        model = {}

        # Find model section
        model_match = re.search(r"\(model(.*?)\)", output, re.DOTALL)
        if model_match:
            model_text = model_match.group(1)

            # Parse define-fun entries
            for match in re.finditer(r"\(define-fun\s+(\w+)\s+\(\)\s+\w+\s+(.+?)\)", model_text):
                name = match.group(1)
                value = match.group(2).strip()
                model[name] = value

        return model


class ProofExplainerEngine:
    """Engine for generating proof explanations."""

    def explain_proof(
        self,
        proof: Z3ParsedProof,
        context: dict[str, Any] | None = None,
        level: ExplanationLevel = ExplanationLevel.INTERMEDIATE,
    ) -> ProofExplanation:
        """Generate explanation for a proof."""
        # Determine verification type from context
        verification_type = context.get("type", "general") if context else "general"

        if proof.status == "unsat":
            return self._explain_verified(proof, verification_type, level)
        elif proof.status == "sat":
            return self._explain_counterexample(proof, verification_type, level, context)
        else:
            return self._explain_unknown(proof, level)

    def _explain_verified(
        self,
        proof: Z3ParsedProof,
        verification_type: str,
        level: ExplanationLevel,
    ) -> ProofExplanation:
        """Explain a verified (unsat) result."""
        # Get pre-built explanation if available
        if verification_type in PATTERN_EXPLANATIONS:
            pattern_exp = PATTERN_EXPLANATIONS[verification_type][level]
        else:
            pattern_exp = "The code has been mathematically proven correct."

        summary = f"✅ Verification passed: Your code is proven safe for {verification_type.replace('_', ' ')}."

        detailed = f"""
The Z3 solver attempted to find ANY input that could violate the safety property,
but it proved that no such input exists. This means your code is mathematically
guaranteed to be safe.

{pattern_exp}

**What was verified:**
- Constraints checked: {len(proof.constraints)}
- Variables analyzed: {len(proof.variables)}
- Result: No counterexample found (proof by refutation)
"""

        why = """
This proof gives you certainty that testing alone cannot provide. While tests
check specific inputs, formal verification proves correctness for ALL possible
inputs - including edge cases you might not have thought of.
"""

        return ProofExplanation(
            summary=summary,
            detailed_explanation=detailed.strip(),
            why_it_matters=why.strip(),
            analogy=self._get_analogy(verification_type, True, level),
            level=level,
            related_concepts=self._get_related_concepts(verification_type),
        )

    def _explain_counterexample(
        self,
        proof: Z3ParsedProof,
        verification_type: str,
        level: ExplanationLevel,
        context: dict[str, Any] | None,
    ) -> ProofExplanation:
        """Explain a counterexample (sat) result."""
        ce = proof.counterexample or {}

        summary = f"⚠️ Potential issue found: {verification_type.replace('_', ' ')} can fail."

        # Build counterexample explanation
        ce_lines = []
        for var, value in ce.items():
            ce_lines.append(f"  - {var} = {value}")

        ce_str = "\n".join(ce_lines) if ce_lines else "  (Details unavailable)"

        detailed = f"""
Z3 found a specific input that violates the safety property. This is called
a **counterexample** - concrete proof that the code CAN fail.

**Counterexample found:**
{ce_str}

**What this means:**
When the variables have these values, the code will exhibit the bug. This
isn't just a theoretical possibility - it's a concrete failing case.

**How to fix:**
Add appropriate checks before the problematic operation to handle this case.
"""

        why = """
This counterexample shows exactly how your code can fail. Unlike a test failure
that just says "something went wrong," this gives you the exact input that
triggers the bug, making it much easier to understand and fix.
"""

        # Generate walkthrough if we have code context
        walkthrough = None
        if context and "code" in context:
            walkthrough = self._generate_walkthrough(proof, context["code"])

        return ProofExplanation(
            summary=summary,
            detailed_explanation=detailed.strip(),
            why_it_matters=why.strip(),
            analogy=self._get_analogy(verification_type, False, level),
            code_walkthrough=walkthrough,
            recommendations=self._get_fix_recommendations(verification_type),
            level=level,
            related_concepts=self._get_related_concepts(verification_type),
        )

    def _explain_unknown(
        self,
        proof: Z3ParsedProof,
        level: ExplanationLevel,
    ) -> ProofExplanation:
        """Explain an unknown result."""
        return ProofExplanation(
            summary="⏳ Verification inconclusive",
            detailed_explanation="""
The Z3 solver could not determine whether the code is safe or unsafe within
the given time and resource limits. This can happen for several reasons:

1. **Complexity**: The code involves complex logic that's hard to analyze
2. **Loops**: Unbounded loops may require manual invariants
3. **External dependencies**: Calls to external code can't be verified

This doesn't mean there's a bug - it just means automated verification
couldn't prove safety. Consider:
- Adding loop invariants
- Simplifying complex conditions
- Using manual testing for now
""",
            why_it_matters="Incomplete verification still provides value by checking simpler properties.",
            level=level,
        )

    def _get_analogy(
        self,
        verification_type: str,
        is_safe: bool,
        level: ExplanationLevel,
    ) -> str:
        """Get an analogy for the verification result."""
        if level == ExplanationLevel.ADVANCED:
            return ""  # No analogies for advanced level

        analogies = {
            "null_check": {
                True: "Like checking if a door is locked before trying to open it.",
                False: "Like reaching for a door handle that doesn't exist.",
            },
            "bounds_check": {
                True: "Like a librarian who always checks if a book exists before fetching it.",
                False: "Like trying to grab the 10th item from a shelf with only 5 items.",
            },
            "division_check": {
                True: "Like a calculator that refuses to divide by zero.",
                False: "Like trying to split a pizza among zero people - impossible!",
            },
        }

        type_analogies = analogies.get(verification_type, {})
        return type_analogies.get(is_safe, "")

    def _get_related_concepts(self, verification_type: str) -> list[str]:
        """Get related concepts for further learning."""
        concepts = {
            "null_check": ["Optional types", "Null safety", "Defensive programming"],
            "bounds_check": ["Array safety", "Buffer overflow", "Input validation"],
            "division_check": ["Arithmetic safety", "Error handling", "Edge cases"],
        }
        return concepts.get(verification_type, ["Formal verification", "Static analysis"])

    def _get_fix_recommendations(self, verification_type: str) -> list[str]:
        """Get fix recommendations for a verification type."""
        recommendations = {
            "null_check": [
                "Add a null/None check before accessing the variable",
                "Use optional chaining (?.) in JavaScript/TypeScript",
                "Consider using Optional types or Maybe monads",
            ],
            "bounds_check": [
                "Check array length before accessing indices",
                "Use safe access methods like .get() that return None",
                "Validate input indices are within bounds",
            ],
            "division_check": [
                "Check if divisor is zero before dividing",
                "Handle the zero case explicitly",
                "Consider what the correct behavior should be for zero",
            ],
        }
        return recommendations.get(verification_type, ["Review the code for potential issues"])

    def _generate_walkthrough(
        self,
        proof: Z3ParsedProof,
        code: str,
    ) -> list[dict[str, str]]:
        """Generate step-by-step code walkthrough with counterexample."""
        walkthrough = []
        ce = proof.counterexample or {}

        # This would be more sophisticated in production
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if line.strip():
                step = {
                    "line_number": i + 1,
                    "code": line,
                    "explanation": f"Line {i+1}: {line.strip()[:50]}...",
                    "state": str(ce) if ce else "{}",
                }
                walkthrough.append(step)

        return walkthrough


class VerificationExplainerAgent(BaseAgent):
    """AI agent for explaining verification results.

    Usage:
        agent = VerificationExplainerAgent()

        # Explain a Z3 proof
        result = await agent.explain_z3_output(z3_output, context)

        # Get a tutorial
        tutorial = agent.get_tutorial("intro_formal_verification")

        # Explain a concept
        explanation = await agent.explain_concept("null safety")
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config)
        self._parser = Z3OutputParser()
        self._engine = ProofExplainerEngine()

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """Analyze and explain verification results."""
        import time
        start_time = time.time()

        try:
            z3_output = context.get("z3_output", "")
            level = ExplanationLevel(context.get("level", "intermediate"))

            explanation = await self.explain_z3_output(z3_output, context, level)

            return AgentResult(
                success=True,
                data=explanation.to_dict(),
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("Explanation failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def explain_z3_output(
        self,
        z3_output: str,
        context: dict[str, Any] | None = None,
        level: ExplanationLevel = ExplanationLevel.INTERMEDIATE,
    ) -> ProofExplanation:
        """Explain Z3 output in human-readable format."""
        # Parse Z3 output
        proof = self._parser.parse(z3_output)

        # Generate explanation
        return self._engine.explain_proof(proof, context, level)

    async def explain_counterexample(
        self,
        counterexample: dict[str, Any],
        code: str,
        language: str = "python",
        level: ExplanationLevel = ExplanationLevel.INTERMEDIATE,
    ) -> list[CounterexampleTrace]:
        """Generate step-by-step trace of a counterexample."""
        traces = []
        step = 0

        # Parse code to understand structure
        lines = code.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            step += 1
            trace = CounterexampleTrace(
                step_number=step,
                description=f"Executing line {i+1}",
                variable_states=counterexample,
                code_line=stripped,
                explanation=self._explain_line(stripped, counterexample, level),
            )
            traces.append(trace)

        return traces

    def _explain_line(
        self,
        line: str,
        counterexample: dict[str, Any],
        level: ExplanationLevel,
    ) -> str:
        """Explain what happens at a line given the counterexample."""
        # Check for common patterns
        if "if " in line:
            return "Evaluating condition..."
        elif "return " in line:
            return "Returning value..."
        elif "[" in line and "]" in line:
            return "Accessing array element..."
        elif "." in line:
            return "Accessing object property..."
        else:
            return "Executing statement..."

    async def explain_concept(
        self,
        concept: str,
        level: ExplanationLevel = ExplanationLevel.INTERMEDIATE,
    ) -> ProofExplanation:
        """Explain a verification concept."""
        # Map concept to verification type
        concept_lower = concept.lower()
        if "null" in concept_lower or "none" in concept_lower:
            verification_type = "null_check"
        elif "bound" in concept_lower or "index" in concept_lower or "array" in concept_lower:
            verification_type = "bounds_check"
        elif "divis" in concept_lower or "zero" in concept_lower:
            verification_type = "division_check"
        else:
            verification_type = "general"

        # Use LLM for custom explanations if available
        if self.config.openai_api_key or self.config.anthropic_api_key:
            return await self._llm_explain_concept(concept, level)

        # Fallback to pattern explanations
        if verification_type in PATTERN_EXPLANATIONS:
            exp = PATTERN_EXPLANATIONS[verification_type][level]
            return ProofExplanation(
                summary=f"Explanation of {concept}",
                detailed_explanation=exp,
                why_it_matters="Understanding this concept helps write safer code.",
                level=level,
                related_concepts=self._engine._get_related_concepts(verification_type),
            )

        return ProofExplanation(
            summary=f"Concept: {concept}",
            detailed_explanation="This concept is related to formal verification and code safety.",
            why_it_matters="Formal verification helps prove code correctness.",
            level=level,
        )

    async def _llm_explain_concept(
        self,
        concept: str,
        level: ExplanationLevel,
    ) -> ProofExplanation:
        """Use LLM to explain a concept."""
        level_desc = {
            ExplanationLevel.BEGINNER: "simple terms with analogies, suitable for beginners",
            ExplanationLevel.INTERMEDIATE: "moderate technical detail",
            ExplanationLevel.ADVANCED: "full technical depth with formal notation",
        }

        prompt = f"""Explain the concept "{concept}" in the context of formal verification and code safety.

Explanation level: {level_desc[level]}

Provide:
1. A one-sentence summary
2. A detailed explanation (2-3 paragraphs)
3. Why this matters for developers
4. A relatable analogy (if not advanced level)

Format as JSON with keys: summary, detailed_explanation, why_it_matters, analogy"""

        try:
            response = await self._call_llm(
                system_prompt="You are an expert at explaining formal verification concepts clearly.",
                user_prompt=prompt,
                json_mode=True,
            )

            parsed = self._parse_json_response(response)
            data = parsed.data

            return ProofExplanation(
                summary=data.get("summary", concept),
                detailed_explanation=data.get("detailed_explanation", ""),
                why_it_matters=data.get("why_it_matters", ""),
                analogy=data.get("analogy") if level != ExplanationLevel.ADVANCED else None,
                level=level,
            )

        except Exception as e:
            logger.error("LLM explanation failed", error=str(e))
            return ProofExplanation(
                summary=f"Concept: {concept}",
                detailed_explanation="Could not generate detailed explanation.",
                why_it_matters="",
                level=level,
            )

    def get_tutorial(self, tutorial_id: str) -> Tutorial | None:
        """Get a tutorial by ID."""
        return TUTORIALS.get(tutorial_id)

    def list_tutorials(self) -> list[dict[str, Any]]:
        """List all available tutorials."""
        return [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "difficulty": t.difficulty,
                "estimated_minutes": t.estimated_minutes,
            }
            for t in TUTORIALS.values()
        ]
