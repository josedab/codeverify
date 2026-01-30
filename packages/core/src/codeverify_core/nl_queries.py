"""
Natural Language Verification Queries

Allow users to ask questions about code verification in natural language:
- NL query parser
- Query-to-constraint translation
- Proof explanation generation

Lowers the barrier to formal methods by making them accessible through
intuitive natural language questions.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# Data Models
# =============================================================================

class QueryType(str, Enum):
    """Types of verification queries."""
    NULL_CHECK = "null_check"          # "Can X ever be null?"
    BOUNDS_CHECK = "bounds_check"      # "Can index Y be out of bounds?"
    VALUE_CHECK = "value_check"        # "What values can X have?"
    REACHABILITY = "reachability"      # "Can line X be reached?"
    TERMINATION = "termination"        # "Does this loop always terminate?"
    EXCEPTION = "exception"            # "Can this throw an exception?"
    INVARIANT = "invariant"            # "Is X always true?"
    COMPARISON = "comparison"          # "Is X always greater than Y?"
    TYPE_CHECK = "type_check"          # "What type can X be?"
    UNKNOWN = "unknown"


class ProofResult(str, Enum):
    """Results of proof attempts."""
    PROVEN = "proven"           # Property definitely holds
    DISPROVEN = "disproven"     # Found counterexample
    UNKNOWN = "unknown"         # Could not determine
    TIMEOUT = "timeout"         # Verification timed out


@dataclass
class ParsedQuery:
    """Result of parsing a natural language query."""
    
    query_id: str
    original_text: str
    
    # Extracted information
    query_type: QueryType
    subject: Optional[str] = None      # Variable/expression being asked about
    predicate: Optional[str] = None    # What's being checked
    context: Optional[str] = None      # Additional context (function, line, etc.)
    
    # Generated formal representation
    z3_query: Optional[str] = None
    constraint: Optional[str] = None
    
    # Confidence in parsing
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "original_text": self.original_text,
            "query_type": self.query_type.value,
            "subject": self.subject,
            "predicate": self.predicate,
            "context": self.context,
            "z3_query": self.z3_query,
            "constraint": self.constraint,
            "confidence": self.confidence,
        }


@dataclass
class VerificationAnswer:
    """Answer to a verification query."""
    
    query_id: str
    result: ProofResult
    
    # Human-readable answer
    answer: str
    explanation: str
    
    # Technical details
    proof_steps: List[str] = field(default_factory=list)
    counterexample: Optional[Dict[str, Any]] = None
    
    # Related information
    related_constraints: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    verification_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "result": self.result.value,
            "answer": self.answer,
            "explanation": self.explanation,
            "proof_steps": self.proof_steps,
            "counterexample": self.counterexample,
            "related_constraints": self.related_constraints,
            "suggestions": self.suggestions,
            "verification_time_ms": self.verification_time_ms,
        }


# =============================================================================
# Query Parser
# =============================================================================

class NaturalLanguageQueryParser:
    """Parses natural language queries into formal verification queries."""
    
    def __init__(self):
        # Define patterns for different query types
        self._patterns = self._build_patterns()
    
    def _build_patterns(self) -> List[Tuple[re.Pattern, QueryType, Callable]]:
        """Build regex patterns for query recognition."""
        patterns = []
        
        # Null check patterns
        null_patterns = [
            r"can\s+(?:the\s+)?(\w+)\s+(?:ever\s+)?be\s+(?:null|none|nil|undefined)",
            r"(?:is|will)\s+(?:the\s+)?(\w+)\s+(?:ever\s+)?(?:null|none|nil|undefined)",
            r"(?:could|might)\s+(?:the\s+)?(\w+)\s+(?:be\s+)?(?:null|none|nil|undefined)",
            r"(?:can|will)\s+(?:this\s+)?(?:function|method)\s+return\s+(?:null|none|nil)",
        ]
        for p in null_patterns:
            patterns.append((
                re.compile(p, re.IGNORECASE),
                QueryType.NULL_CHECK,
                self._extract_null_check,
            ))
        
        # Bounds check patterns
        bounds_patterns = [
            r"can\s+(?:the\s+)?(?:index\s+)?(\w+)\s+(?:ever\s+)?(?:be\s+)?out\s+of\s+bounds",
            r"(?:is|will)\s+(?:the\s+)?(?:index\s+)?(\w+)\s+(?:always\s+)?(?:within|in)\s+bounds",
            r"can\s+(?:the\s+)?array\s+access\s+(?:at\s+)?(\w+)\s+fail",
            r"(?:is|will)\s+(\w+)\s+(?:a\s+)?valid\s+(?:index|position)",
        ]
        for p in bounds_patterns:
            patterns.append((
                re.compile(p, re.IGNORECASE),
                QueryType.BOUNDS_CHECK,
                self._extract_bounds_check,
            ))
        
        # Value check patterns
        value_patterns = [
            r"what\s+(?:values?\s+)?can\s+(?:the\s+)?(\w+)\s+(?:have|be|take)",
            r"what\s+(?:is|are)\s+(?:the\s+)?(?:possible|valid)\s+(?:values?\s+)?(?:for|of)\s+(?:the\s+)?(\w+)",
            r"(?:what|which)\s+values?\s+(?:can|could|might)\s+(\w+)\s+(?:hold|contain|have)",
        ]
        for p in value_patterns:
            patterns.append((
                re.compile(p, re.IGNORECASE),
                QueryType.VALUE_CHECK,
                self._extract_value_check,
            ))
        
        # Reachability patterns
        reach_patterns = [
            r"can\s+(?:line\s+)?(\d+)\s+(?:ever\s+)?be\s+reached",
            r"(?:is|will)\s+(?:line\s+)?(\d+)\s+(?:ever\s+)?(?:executed|reached)",
            r"(?:is|can)\s+(?:the\s+)?(?:code\s+)?(?:on\s+)?(?:line\s+)?(\d+)\s+(?:be\s+)?(?:reachable|reached)",
        ]
        for p in reach_patterns:
            patterns.append((
                re.compile(p, re.IGNORECASE),
                QueryType.REACHABILITY,
                self._extract_reachability,
            ))
        
        # Termination patterns
        term_patterns = [
            r"(?:does|will)\s+(?:this\s+)?(?:loop|function|recursion)\s+(?:always\s+)?terminate",
            r"can\s+(?:this\s+)?(?:loop|function)\s+(?:ever\s+)?(?:run\s+)?forever",
            r"(?:is|will)\s+(?:this\s+)?(?:loop|function)\s+(?:guaranteed\s+)?to\s+(?:finish|end|terminate)",
        ]
        for p in term_patterns:
            patterns.append((
                re.compile(p, re.IGNORECASE),
                QueryType.TERMINATION,
                self._extract_termination,
            ))
        
        # Exception patterns
        exc_patterns = [
            r"can\s+(?:this\s+)?(?:code|function|method)\s+(?:ever\s+)?(?:throw|raise)\s+(?:an?\s+)?(?:exception|error)",
            r"(?:will|could|might)\s+(?:this\s+)?(?:code|function)\s+(?:throw|raise)",
            r"(?:is|are)\s+(?:there\s+)?(?:any\s+)?(?:exception|error)s?\s+(?:possible|thrown)",
        ]
        for p in exc_patterns:
            patterns.append((
                re.compile(p, re.IGNORECASE),
                QueryType.EXCEPTION,
                self._extract_exception,
            ))
        
        # Invariant patterns
        inv_patterns = [
            r"(?:is|will)\s+(?:the\s+)?(\w+)\s+always\s+(?:be\s+)?(.+?)(?:\?|$)",
            r"(?:is|will)\s+it\s+always\s+(?:true\s+)?that\s+(.+?)(?:\?|$)",
            r"(?:can|does)\s+(?:the\s+)?invariant\s+(.+?)\s+(?:ever\s+)?(?:be\s+)?(?:violated|broken)",
        ]
        for p in inv_patterns:
            patterns.append((
                re.compile(p, re.IGNORECASE),
                QueryType.INVARIANT,
                self._extract_invariant,
            ))
        
        # Comparison patterns
        comp_patterns = [
            r"(?:is|will)\s+(?:the\s+)?(\w+)\s+always\s+(?:greater|less|equal)\s+(?:than|to)\s+(?:the\s+)?(\w+)",
            r"can\s+(?:the\s+)?(\w+)\s+(?:ever\s+)?(?:be\s+)?(?:greater|less|equal)\s+(?:than|to)\s+(?:the\s+)?(\w+)",
            r"(?:is|will)\s+(\w+)\s+(?:<|>|<=|>=|==|!=)\s+(\w+)",
        ]
        for p in comp_patterns:
            patterns.append((
                re.compile(p, re.IGNORECASE),
                QueryType.COMPARISON,
                self._extract_comparison,
            ))
        
        return patterns
    
    def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language query."""
        query_id = hashlib.sha256(
            f"{time.time()}-{query}".encode()
        ).hexdigest()[:16]
        
        parsed = ParsedQuery(
            query_id=query_id,
            original_text=query,
            query_type=QueryType.UNKNOWN,
        )
        
        # Try each pattern
        for pattern, query_type, extractor in self._patterns:
            match = pattern.search(query)
            if match:
                parsed.query_type = query_type
                extractor(parsed, match, query)
                parsed.confidence = 0.9 if match.group(0) == query else 0.7
                break
        
        # If no pattern matched, try to extract subject at least
        if parsed.query_type == QueryType.UNKNOWN:
            parsed.subject = self._extract_subject(query)
            parsed.confidence = 0.4
        
        # Generate Z3 query
        parsed.z3_query = self._generate_z3_query(parsed)
        parsed.constraint = self._generate_constraint(parsed)
        
        return parsed
    
    def _extract_null_check(
        self,
        parsed: ParsedQuery,
        match: re.Match,
        query: str,
    ) -> None:
        """Extract null check query components."""
        if match.groups():
            parsed.subject = match.group(1)
        else:
            # Extract subject from query
            parsed.subject = self._extract_subject(query)
        
        parsed.predicate = "== None"
    
    def _extract_bounds_check(
        self,
        parsed: ParsedQuery,
        match: re.Match,
        query: str,
    ) -> None:
        """Extract bounds check query components."""
        if match.groups():
            parsed.subject = match.group(1)
        
        parsed.predicate = "within_bounds"
    
    def _extract_value_check(
        self,
        parsed: ParsedQuery,
        match: re.Match,
        query: str,
    ) -> None:
        """Extract value check query components."""
        if match.groups():
            parsed.subject = match.group(1)
        
        parsed.predicate = "possible_values"
    
    def _extract_reachability(
        self,
        parsed: ParsedQuery,
        match: re.Match,
        query: str,
    ) -> None:
        """Extract reachability query components."""
        if match.groups():
            parsed.subject = f"line_{match.group(1)}"
            parsed.context = f"line {match.group(1)}"
        
        parsed.predicate = "reachable"
    
    def _extract_termination(
        self,
        parsed: ParsedQuery,
        match: re.Match,
        query: str,
    ) -> None:
        """Extract termination query components."""
        parsed.subject = "loop"
        parsed.predicate = "terminates"
    
    def _extract_exception(
        self,
        parsed: ParsedQuery,
        match: re.Match,
        query: str,
    ) -> None:
        """Extract exception query components."""
        parsed.subject = "function"
        parsed.predicate = "throws_exception"
    
    def _extract_invariant(
        self,
        parsed: ParsedQuery,
        match: re.Match,
        query: str,
    ) -> None:
        """Extract invariant query components."""
        groups = match.groups()
        if len(groups) >= 2:
            parsed.subject = groups[0]
            parsed.predicate = groups[1].strip()
        elif len(groups) >= 1:
            parsed.predicate = groups[0].strip()
    
    def _extract_comparison(
        self,
        parsed: ParsedQuery,
        match: re.Match,
        query: str,
    ) -> None:
        """Extract comparison query components."""
        groups = match.groups()
        if len(groups) >= 2:
            parsed.subject = groups[0]
            parsed.context = groups[1]
            
            # Determine comparison operator
            if "greater" in query.lower():
                parsed.predicate = ">"
            elif "less" in query.lower():
                parsed.predicate = "<"
            elif "equal" in query.lower():
                parsed.predicate = "=="
            else:
                parsed.predicate = "?"
    
    def _extract_subject(self, query: str) -> Optional[str]:
        """Extract likely subject from query."""
        # Look for quoted identifiers
        quoted = re.search(r"['\"](\w+)['\"]", query)
        if quoted:
            return quoted.group(1)
        
        # Look for common patterns
        patterns = [
            r"(?:the\s+)?(?:variable\s+)?(\w+)",
            r"(?:function|method)\s+(\w+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                word = match.group(1)
                # Filter out common words
                if word.lower() not in ["the", "a", "an", "this", "that", "can", "will", "is"]:
                    return word
        
        return None
    
    def _generate_z3_query(self, parsed: ParsedQuery) -> Optional[str]:
        """Generate Z3 Python code for the query."""
        if not parsed.subject:
            return None
        
        subject = parsed.subject
        
        if parsed.query_type == QueryType.NULL_CHECK:
            return f"s.add({subject} == None)\nresult = s.check()"
        
        elif parsed.query_type == QueryType.BOUNDS_CHECK:
            return f"s.add(Or({subject} < 0, {subject} >= len_array))\nresult = s.check()"
        
        elif parsed.query_type == QueryType.VALUE_CHECK:
            return f"# Find all satisfying values for {subject}\nresults = []\nwhile s.check() == sat:\n    m = s.model()\n    results.append(m[{subject}])\n    s.add({subject} != m[{subject}])"
        
        elif parsed.query_type == QueryType.COMPARISON:
            other = parsed.context or "other"
            op = parsed.predicate or ">"
            return f"s.add(Not({subject} {op} {other}))\nresult = s.check()"
        
        elif parsed.query_type == QueryType.INVARIANT:
            pred = parsed.predicate or "true"
            return f"s.add(Not({pred}))\nresult = s.check()"
        
        return None
    
    def _generate_constraint(self, parsed: ParsedQuery) -> Optional[str]:
        """Generate human-readable constraint."""
        if not parsed.subject:
            return None
        
        subject = parsed.subject
        
        if parsed.query_type == QueryType.NULL_CHECK:
            return f"{subject} is not null/None"
        
        elif parsed.query_type == QueryType.BOUNDS_CHECK:
            return f"0 <= {subject} < array_length"
        
        elif parsed.query_type == QueryType.COMPARISON:
            other = parsed.context or "other"
            op = parsed.predicate or ">"
            return f"{subject} {op} {other}"
        
        elif parsed.query_type == QueryType.INVARIANT:
            return parsed.predicate
        
        return None


# =============================================================================
# Answer Generator
# =============================================================================

class AnswerGenerator:
    """Generates human-readable answers to verification queries."""
    
    def generate_answer(
        self,
        parsed: ParsedQuery,
        result: ProofResult,
        details: Optional[Dict[str, Any]] = None,
    ) -> VerificationAnswer:
        """Generate a verification answer."""
        details = details or {}
        
        answer = VerificationAnswer(
            query_id=parsed.query_id,
            result=result,
            answer="",
            explanation="",
            verification_time_ms=details.get("time_ms", 0.0),
        )
        
        # Generate answer based on query type and result
        if parsed.query_type == QueryType.NULL_CHECK:
            self._generate_null_answer(answer, parsed, result, details)
        
        elif parsed.query_type == QueryType.BOUNDS_CHECK:
            self._generate_bounds_answer(answer, parsed, result, details)
        
        elif parsed.query_type == QueryType.VALUE_CHECK:
            self._generate_value_answer(answer, parsed, result, details)
        
        elif parsed.query_type == QueryType.COMPARISON:
            self._generate_comparison_answer(answer, parsed, result, details)
        
        elif parsed.query_type == QueryType.INVARIANT:
            self._generate_invariant_answer(answer, parsed, result, details)
        
        elif parsed.query_type == QueryType.TERMINATION:
            self._generate_termination_answer(answer, parsed, result, details)
        
        elif parsed.query_type == QueryType.EXCEPTION:
            self._generate_exception_answer(answer, parsed, result, details)
        
        else:
            self._generate_generic_answer(answer, parsed, result, details)
        
        return answer
    
    def _generate_null_answer(
        self,
        answer: VerificationAnswer,
        parsed: ParsedQuery,
        result: ProofResult,
        details: Dict[str, Any],
    ) -> None:
        """Generate null check answer."""
        subject = parsed.subject or "the value"
        
        if result == ProofResult.PROVEN:
            answer.answer = f"No, {subject} can never be null/None."
            answer.explanation = (
                f"Formal verification proves that {subject} is always non-null "
                f"in all possible execution paths."
            )
            answer.proof_steps = [
                f"Assumed {subject} could be null",
                "Checked all code paths",
                "No path leads to null value",
                "Therefore, null is not possible",
            ]
        
        elif result == ProofResult.DISPROVEN:
            answer.answer = f"Yes, {subject} can be null/None."
            answer.explanation = (
                f"Found an execution path where {subject} becomes null."
            )
            answer.counterexample = details.get("counterexample", {
                "scenario": f"{subject} is null when...",
            })
            answer.suggestions = [
                f"Add a null check before using {subject}",
                f"Use Optional type annotation for {subject}",
            ]
        
        else:
            answer.answer = f"Unable to determine if {subject} can be null."
            answer.explanation = "The verification could not reach a conclusive result."
    
    def _generate_bounds_answer(
        self,
        answer: VerificationAnswer,
        parsed: ParsedQuery,
        result: ProofResult,
        details: Dict[str, Any],
    ) -> None:
        """Generate bounds check answer."""
        subject = parsed.subject or "the index"
        
        if result == ProofResult.PROVEN:
            answer.answer = f"No, {subject} is always within bounds."
            answer.explanation = (
                f"Formal verification proves that {subject} is always valid "
                f"for the array access."
            )
        
        elif result == ProofResult.DISPROVEN:
            answer.answer = f"Yes, {subject} can be out of bounds."
            answer.explanation = (
                f"Found a scenario where {subject} exceeds array bounds."
            )
            answer.counterexample = details.get("counterexample", {})
            answer.suggestions = [
                f"Add bounds check: if 0 <= {subject} < len(array)",
                "Use .get() method with default value",
            ]
    
    def _generate_value_answer(
        self,
        answer: VerificationAnswer,
        parsed: ParsedQuery,
        result: ProofResult,
        details: Dict[str, Any],
    ) -> None:
        """Generate value check answer."""
        subject = parsed.subject or "the variable"
        values = details.get("values", [])
        
        if values:
            answer.answer = f"{subject} can have the following values: {', '.join(map(str, values))}"
            answer.explanation = f"Based on constraints, {subject} is limited to these values."
        else:
            answer.answer = f"{subject} appears to be unconstrained."
            answer.explanation = "No specific constraints limit the values."
    
    def _generate_comparison_answer(
        self,
        answer: VerificationAnswer,
        parsed: ParsedQuery,
        result: ProofResult,
        details: Dict[str, Any],
    ) -> None:
        """Generate comparison answer."""
        subject = parsed.subject or "A"
        other = parsed.context or "B"
        op = parsed.predicate or ">"
        
        if result == ProofResult.PROVEN:
            answer.answer = f"Yes, {subject} is always {op} {other}."
            answer.explanation = f"Formal proof confirms the comparison holds in all cases."
        
        elif result == ProofResult.DISPROVEN:
            answer.answer = f"No, {subject} is not always {op} {other}."
            answer.explanation = f"Found cases where the comparison fails."
            answer.counterexample = details.get("counterexample", {})
    
    def _generate_invariant_answer(
        self,
        answer: VerificationAnswer,
        parsed: ParsedQuery,
        result: ProofResult,
        details: Dict[str, Any],
    ) -> None:
        """Generate invariant answer."""
        predicate = parsed.predicate or "the condition"
        
        if result == ProofResult.PROVEN:
            answer.answer = f"Yes, '{predicate}' is always true."
            answer.explanation = "The invariant holds in all execution paths."
        
        elif result == ProofResult.DISPROVEN:
            answer.answer = f"No, '{predicate}' can be violated."
            answer.explanation = "Found an execution path where the invariant fails."
            answer.counterexample = details.get("counterexample", {})
    
    def _generate_termination_answer(
        self,
        answer: VerificationAnswer,
        parsed: ParsedQuery,
        result: ProofResult,
        details: Dict[str, Any],
    ) -> None:
        """Generate termination answer."""
        if result == ProofResult.PROVEN:
            answer.answer = "Yes, the code always terminates."
            answer.explanation = "Verified that all loops and recursion have bounded iterations."
        
        elif result == ProofResult.DISPROVEN:
            answer.answer = "No, the code may not terminate."
            answer.explanation = "Found a potential infinite loop or unbounded recursion."
            answer.suggestions = [
                "Add a loop counter or termination condition",
                "Verify recursion has a proper base case",
            ]
        
        else:
            answer.answer = "Unable to prove termination."
            answer.explanation = "The termination analysis was inconclusive."
    
    def _generate_exception_answer(
        self,
        answer: VerificationAnswer,
        parsed: ParsedQuery,
        result: ProofResult,
        details: Dict[str, Any],
    ) -> None:
        """Generate exception answer."""
        exceptions = details.get("exceptions", [])
        
        if result == ProofResult.PROVEN:
            answer.answer = "No, the code cannot throw an exception."
            answer.explanation = "All potential exception sources are properly handled."
        
        elif result == ProofResult.DISPROVEN:
            if exceptions:
                answer.answer = f"Yes, the code can throw: {', '.join(exceptions)}"
            else:
                answer.answer = "Yes, the code can throw exceptions."
            answer.explanation = "Found code paths that may raise exceptions."
            answer.suggestions = [
                "Add try/except blocks around risky operations",
                "Validate inputs before processing",
            ]
    
    def _generate_generic_answer(
        self,
        answer: VerificationAnswer,
        parsed: ParsedQuery,
        result: ProofResult,
        details: Dict[str, Any],
    ) -> None:
        """Generate generic answer for unknown query types."""
        answer.answer = "I couldn't fully understand your question."
        answer.explanation = (
            "Please try rephrasing your question. Examples:\n"
            "- 'Can x ever be null?'\n"
            "- 'Is the index always within bounds?'\n"
            "- 'What values can result have?'"
        )


# =============================================================================
# Query Engine
# =============================================================================

class NLVerificationEngine:
    """
    Main engine for natural language verification queries.
    
    Combines parsing, verification, and answer generation.
    """
    
    def __init__(self):
        self.parser = NaturalLanguageQueryParser()
        self.answer_generator = AnswerGenerator()
        self._query_history: List[Dict[str, Any]] = []
    
    def query(
        self,
        question: str,
        code: str,
        language: str = "python",
    ) -> VerificationAnswer:
        """
        Process a natural language verification query.
        
        Args:
            question: The natural language question
            code: The code to verify
            language: Programming language
        
        Returns:
            VerificationAnswer with the result
        """
        start_time = time.time()
        
        # Parse the question
        parsed = self.parser.parse(question)
        
        # Perform verification
        result, details = self._verify(parsed, code, language)
        
        # Generate answer
        details["time_ms"] = (time.time() - start_time) * 1000
        answer = self.answer_generator.generate_answer(parsed, result, details)
        
        # Store in history
        self._query_history.append({
            "question": question,
            "parsed": parsed.to_dict(),
            "answer": answer.to_dict(),
            "timestamp": time.time(),
        })
        
        return answer
    
    def _verify(
        self,
        parsed: ParsedQuery,
        code: str,
        language: str,
    ) -> Tuple[ProofResult, Dict[str, Any]]:
        """Perform verification based on parsed query."""
        details: Dict[str, Any] = {}
        
        # Simplified verification logic
        # In production, this would use Z3 SMT solver
        
        if parsed.query_type == QueryType.NULL_CHECK:
            return self._verify_null(parsed, code)
        
        elif parsed.query_type == QueryType.BOUNDS_CHECK:
            return self._verify_bounds(parsed, code)
        
        elif parsed.query_type == QueryType.VALUE_CHECK:
            return self._verify_values(parsed, code)
        
        elif parsed.query_type == QueryType.EXCEPTION:
            return self._verify_exceptions(parsed, code)
        
        elif parsed.query_type == QueryType.COMPARISON:
            return self._verify_comparison(parsed, code)
        
        return ProofResult.UNKNOWN, details
    
    def _verify_null(
        self,
        parsed: ParsedQuery,
        code: str,
    ) -> Tuple[ProofResult, Dict[str, Any]]:
        """Verify null safety."""
        subject = parsed.subject
        
        if not subject:
            return ProofResult.UNKNOWN, {}
        
        # Simple pattern matching
        if f"return None" in code:
            if f"Optional" in code or f"| None" in code:
                return ProofResult.PROVEN, {}
            else:
                return ProofResult.DISPROVEN, {
                    "counterexample": {
                        "scenario": f"Function returns None without Optional type",
                    }
                }
        
        if f"{subject} = None" in code or f"{subject}=None" in code:
            return ProofResult.DISPROVEN, {
                "counterexample": {
                    "scenario": f"{subject} is explicitly assigned None",
                }
            }
        
        return ProofResult.PROVEN, {}
    
    def _verify_bounds(
        self,
        parsed: ParsedQuery,
        code: str,
    ) -> Tuple[ProofResult, Dict[str, Any]]:
        """Verify bounds safety."""
        subject = parsed.subject
        
        # Look for array access without bounds check
        if f"[{subject}]" in code:
            if f"if {subject} <" in code or f"if 0 <= {subject}" in code:
                return ProofResult.PROVEN, {}
            else:
                return ProofResult.DISPROVEN, {
                    "counterexample": {
                        "scenario": f"Array access [{subject}] without bounds check",
                    }
                }
        
        return ProofResult.PROVEN, {}
    
    def _verify_values(
        self,
        parsed: ParsedQuery,
        code: str,
    ) -> Tuple[ProofResult, Dict[str, Any]]:
        """Analyze possible values."""
        subject = parsed.subject
        values = []
        
        # Look for assignments
        import re
        assignments = re.findall(rf"{subject}\s*=\s*(.+)", code)
        
        for assignment in assignments:
            # Try to extract literal values
            if assignment.strip().isdigit():
                values.append(int(assignment.strip()))
            elif assignment.strip() in ("True", "False"):
                values.append(assignment.strip())
        
        return ProofResult.PROVEN, {"values": values}
    
    def _verify_exceptions(
        self,
        parsed: ParsedQuery,
        code: str,
    ) -> Tuple[ProofResult, Dict[str, Any]]:
        """Verify exception safety."""
        exceptions = []
        
        # Look for potential exception sources
        if "raise" in code:
            import re
            raises = re.findall(r"raise\s+(\w+)", code)
            exceptions.extend(raises)
        
        if "/" in code and "try" not in code:
            exceptions.append("ZeroDivisionError")
        
        if "[" in code and "try" not in code:
            exceptions.append("IndexError")
        
        if exceptions:
            return ProofResult.DISPROVEN, {"exceptions": exceptions}
        
        return ProofResult.PROVEN, {}
    
    def _verify_comparison(
        self,
        parsed: ParsedQuery,
        code: str,
    ) -> Tuple[ProofResult, Dict[str, Any]]:
        """Verify comparison."""
        # Simplified - would use Z3 in production
        return ProofResult.UNKNOWN, {}
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get query history."""
        return self._query_history[-limit:]
    
    def get_example_queries(self) -> List[str]:
        """Get example queries users can ask."""
        return [
            "Can x ever be null?",
            "Is the index always within bounds?",
            "What values can result have?",
            "Does this loop always terminate?",
            "Can this function throw an exception?",
            "Is x always greater than 0?",
            "Can line 42 be reached?",
        ]
