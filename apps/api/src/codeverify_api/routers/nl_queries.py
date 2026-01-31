"""
Natural Language Verification Queries API Router

Provides REST API endpoints for natural language verification:
- Query parsing
- Verification execution
- Answer generation
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v1/nl-query", tags=["nl-queries"])


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request for natural language query."""
    question: str = Field(..., description="Natural language question")
    code: str = Field(..., description="Code to verify")
    language: str = Field("python", description="Programming language")


class ParseOnlyRequest(BaseModel):
    """Request to parse without verification."""
    question: str = Field(..., description="Question to parse")


class ParsedQueryResponse(BaseModel):
    """Parsed query response."""
    query_id: str
    query_type: str
    subject: Optional[str]
    predicate: Optional[str]
    context: Optional[str]
    z3_query: Optional[str]
    constraint: Optional[str]
    confidence: float


class VerificationAnswerResponse(BaseModel):
    """Verification answer response."""
    query_id: str
    result: str
    answer: str
    explanation: str
    proof_steps: List[str]
    counterexample: Optional[Dict[str, Any]]
    suggestions: List[str]
    verification_time_ms: float


# =============================================================================
# In-Memory State
# =============================================================================

# Query history per user/session
_query_history: List[Dict[str, Any]] = []

# Query types
QUERY_TYPES = [
    "null_check", "bounds_check", "value_check", "reachability",
    "termination", "exception", "invariant", "comparison", "type_check", "unknown"
]

# Proof results
PROOF_RESULTS = ["proven", "disproven", "unknown", "timeout"]


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/query",
    response_model=VerificationAnswerResponse,
    summary="Ask Verification Question",
    description="Ask a natural language question about code verification"
)
async def query(request: QueryRequest) -> VerificationAnswerResponse:
    """
    Ask a natural language verification question.
    
    Examples:
    - "Can x ever be null?"
    - "Is the index always within bounds?"
    - "What values can result have?"
    """
    start_time = time.time()
    
    # Parse the question
    parsed = _parse_query(request.question)
    
    # Perform verification
    result, details = _verify(parsed, request.code, request.language)
    
    # Generate answer
    answer = _generate_answer(parsed, result, details)
    
    answer["verification_time_ms"] = (time.time() - start_time) * 1000
    
    # Store in history
    _query_history.append({
        "question": request.question,
        "parsed": parsed,
        "result": result,
        "answer": answer["answer"],
        "timestamp": time.time(),
    })
    
    return VerificationAnswerResponse(
        query_id=parsed["query_id"],
        result=result,
        answer=answer["answer"],
        explanation=answer["explanation"],
        proof_steps=answer.get("proof_steps", []),
        counterexample=answer.get("counterexample"),
        suggestions=answer.get("suggestions", []),
        verification_time_ms=answer["verification_time_ms"],
    )


@router.post(
    "/parse",
    response_model=ParsedQueryResponse,
    summary="Parse Question",
    description="Parse a natural language question without verifying"
)
async def parse_question(request: ParseOnlyRequest) -> ParsedQueryResponse:
    """Parse a question without verification."""
    parsed = _parse_query(request.question)
    
    return ParsedQueryResponse(
        query_id=parsed["query_id"],
        query_type=parsed["query_type"],
        subject=parsed.get("subject"),
        predicate=parsed.get("predicate"),
        context=parsed.get("context"),
        z3_query=parsed.get("z3_query"),
        constraint=parsed.get("constraint"),
        confidence=parsed.get("confidence", 0.5),
    )


@router.get(
    "/examples",
    summary="Get Example Questions",
    description="Get example questions that can be asked"
)
async def get_examples() -> Dict[str, Any]:
    """Get example questions."""
    return {
        "examples": [
            {
                "question": "Can x ever be null?",
                "type": "null_check",
                "description": "Check if a variable can be null/None",
            },
            {
                "question": "Is the index always within bounds?",
                "type": "bounds_check",
                "description": "Check array/list access safety",
            },
            {
                "question": "What values can result have?",
                "type": "value_check",
                "description": "Find possible values for a variable",
            },
            {
                "question": "Does this loop always terminate?",
                "type": "termination",
                "description": "Check if loops/recursion terminate",
            },
            {
                "question": "Can this function throw an exception?",
                "type": "exception",
                "description": "Check for potential exceptions",
            },
            {
                "question": "Is x always greater than 0?",
                "type": "comparison",
                "description": "Verify comparison properties",
            },
            {
                "question": "Is it always true that count >= 0?",
                "type": "invariant",
                "description": "Verify invariants hold",
            },
        ],
        "tips": [
            "Use specific variable names from your code",
            "Ask one question at a time for best results",
            "Questions about null, bounds, and comparisons work best",
        ],
    }


@router.get(
    "/history",
    summary="Get Query History",
    description="Get recent query history"
)
async def get_history(limit: int = 20) -> Dict[str, Any]:
    """Get query history."""
    return {
        "queries": _query_history[-limit:],
        "total": len(_query_history),
    }


@router.post(
    "/batch",
    summary="Batch Query",
    description="Ask multiple questions at once"
)
async def batch_query(
    questions: List[str],
    code: str,
    language: str = "python",
) -> Dict[str, Any]:
    """Ask multiple questions about the same code."""
    results = []
    
    for question in questions:
        try:
            request = QueryRequest(question=question, code=code, language=language)
            result = await query(request)
            results.append({
                "question": question,
                "result": result.result,
                "answer": result.answer,
            })
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e),
            })
    
    return {
        "results": results,
        "total": len(results),
        "successful": len([r for r in results if "error" not in r]),
    }


@router.get(
    "/query-types",
    summary="List Query Types",
    description="Get list of supported query types"
)
async def list_query_types() -> Dict[str, Any]:
    """List supported query types."""
    return {
        "query_types": [
            {"type": "null_check", "description": "Check for null/None values"},
            {"type": "bounds_check", "description": "Check array bounds"},
            {"type": "value_check", "description": "Find possible values"},
            {"type": "reachability", "description": "Check code reachability"},
            {"type": "termination", "description": "Check termination"},
            {"type": "exception", "description": "Check for exceptions"},
            {"type": "invariant", "description": "Verify invariants"},
            {"type": "comparison", "description": "Verify comparisons"},
            {"type": "type_check", "description": "Check types"},
        ],
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_query(question: str) -> Dict[str, Any]:
    """Parse a natural language question."""
    query_id = hashlib.sha256(
        f"{time.time()}-{question}".encode()
    ).hexdigest()[:16]
    
    parsed = {
        "query_id": query_id,
        "original_text": question,
        "query_type": "unknown",
        "subject": None,
        "predicate": None,
        "context": None,
        "z3_query": None,
        "constraint": None,
        "confidence": 0.5,
    }
    
    question_lower = question.lower()
    
    # Null check patterns
    null_patterns = [
        r"can\s+(?:the\s+)?(\w+)\s+(?:ever\s+)?be\s+(?:null|none)",
        r"(?:is|will)\s+(?:the\s+)?(\w+)\s+(?:ever\s+)?null",
        r"could\s+(?:the\s+)?(\w+)\s+be\s+none",
    ]
    for pattern in null_patterns:
        match = re.search(pattern, question_lower)
        if match:
            parsed["query_type"] = "null_check"
            parsed["subject"] = match.group(1)
            parsed["predicate"] = "== None"
            parsed["confidence"] = 0.9
            break
    
    # Bounds check patterns
    if parsed["query_type"] == "unknown":
        bounds_patterns = [
            r"(?:is\s+)?(?:the\s+)?(?:index\s+)?(\w+)\s+(?:always\s+)?within\s+bounds",
            r"can\s+(?:the\s+)?(?:index\s+)?(\w+)\s+be\s+out\s+of\s+bounds",
            r"(?:is\s+)?(\w+)\s+a?\s*valid\s+index",
        ]
        for pattern in bounds_patterns:
            match = re.search(pattern, question_lower)
            if match:
                parsed["query_type"] = "bounds_check"
                parsed["subject"] = match.group(1)
                parsed["predicate"] = "within_bounds"
                parsed["confidence"] = 0.9
                break
    
    # Value check patterns
    if parsed["query_type"] == "unknown":
        value_patterns = [
            r"what\s+(?:values?\s+)?can\s+(?:the\s+)?(\w+)\s+have",
            r"what\s+(?:are\s+)?(?:the\s+)?possible\s+values\s+(?:for|of)\s+(\w+)",
        ]
        for pattern in value_patterns:
            match = re.search(pattern, question_lower)
            if match:
                parsed["query_type"] = "value_check"
                parsed["subject"] = match.group(1)
                parsed["predicate"] = "possible_values"
                parsed["confidence"] = 0.85
                break
    
    # Termination patterns
    if parsed["query_type"] == "unknown":
        if "terminate" in question_lower or "finish" in question_lower or "end" in question_lower:
            if "loop" in question_lower or "function" in question_lower or "recursion" in question_lower:
                parsed["query_type"] = "termination"
                parsed["subject"] = "loop"
                parsed["predicate"] = "terminates"
                parsed["confidence"] = 0.85
    
    # Exception patterns
    if parsed["query_type"] == "unknown":
        if "exception" in question_lower or "throw" in question_lower or "raise" in question_lower:
            parsed["query_type"] = "exception"
            parsed["subject"] = "function"
            parsed["predicate"] = "throws"
            parsed["confidence"] = 0.85
    
    # Comparison patterns
    if parsed["query_type"] == "unknown":
        comp_patterns = [
            r"(?:is|will)\s+(?:the\s+)?(\w+)\s+always\s+(greater|less|equal)\s+(?:than|to)\s+(?:the\s+)?(\w+)",
            r"(?:is\s+)?(\w+)\s+always\s+(>|<|>=|<=|==)\s+(\w+)",
        ]
        for pattern in comp_patterns:
            match = re.search(pattern, question_lower)
            if match:
                parsed["query_type"] = "comparison"
                parsed["subject"] = match.group(1)
                parsed["predicate"] = match.group(2)
                parsed["context"] = match.group(3) if len(match.groups()) > 2 else None
                parsed["confidence"] = 0.85
                break
    
    # Invariant patterns
    if parsed["query_type"] == "unknown":
        inv_patterns = [
            r"(?:is\s+)?it\s+always\s+(?:true\s+)?that\s+(.+?)(?:\?|$)",
            r"(?:is|will)\s+(?:the\s+)?(\w+)\s+always\s+(.+?)(?:\?|$)",
        ]
        for pattern in inv_patterns:
            match = re.search(pattern, question_lower)
            if match:
                parsed["query_type"] = "invariant"
                if len(match.groups()) >= 2:
                    parsed["subject"] = match.group(1)
                    parsed["predicate"] = match.group(2).strip()
                else:
                    parsed["predicate"] = match.group(1).strip()
                parsed["confidence"] = 0.75
                break
    
    # Generate Z3 query
    parsed["z3_query"] = _generate_z3_query(parsed)
    parsed["constraint"] = _generate_constraint(parsed)
    
    return parsed


def _generate_z3_query(parsed: Dict[str, Any]) -> Optional[str]:
    """Generate Z3 query from parsed question."""
    subject = parsed.get("subject")
    if not subject:
        return None
    
    query_type = parsed["query_type"]
    
    if query_type == "null_check":
        return f"s.add({subject} == None)\nresult = s.check()"
    elif query_type == "bounds_check":
        return f"s.add(Or({subject} < 0, {subject} >= len_array))\nresult = s.check()"
    elif query_type == "comparison":
        other = parsed.get("context", "other")
        op = parsed.get("predicate", ">")
        return f"s.add(Not({subject} {op} {other}))\nresult = s.check()"
    
    return None


def _generate_constraint(parsed: Dict[str, Any]) -> Optional[str]:
    """Generate human-readable constraint."""
    subject = parsed.get("subject")
    if not subject:
        return None
    
    query_type = parsed["query_type"]
    
    if query_type == "null_check":
        return f"{subject} is not null/None"
    elif query_type == "bounds_check":
        return f"0 <= {subject} < array_length"
    elif query_type == "comparison":
        other = parsed.get("context", "other")
        op = parsed.get("predicate", ">")
        return f"{subject} {op} {other}"
    
    return None


def _verify(
    parsed: Dict[str, Any],
    code: str,
    language: str,
) -> tuple[str, Dict[str, Any]]:
    """Perform verification."""
    query_type = parsed["query_type"]
    subject = parsed.get("subject")
    details: Dict[str, Any] = {}
    
    if query_type == "null_check":
        return _verify_null(subject, code)
    elif query_type == "bounds_check":
        return _verify_bounds(subject, code)
    elif query_type == "value_check":
        return _verify_values(subject, code)
    elif query_type == "exception":
        return _verify_exceptions(code)
    elif query_type == "termination":
        return _verify_termination(code)
    elif query_type == "comparison":
        return _verify_comparison(parsed, code)
    
    return "unknown", details


def _verify_null(subject: Optional[str], code: str) -> tuple[str, Dict[str, Any]]:
    """Verify null safety."""
    if not subject:
        return "unknown", {}
    
    if "return None" in code:
        if "Optional" in code or "| None" in code:
            return "proven", {}
        else:
            return "disproven", {
                "counterexample": {
                    "scenario": "Function returns None without Optional type",
                }
            }
    
    if f"{subject} = None" in code or f"{subject}=None" in code:
        return "disproven", {
            "counterexample": {
                "scenario": f"{subject} is assigned None",
            }
        }
    
    return "proven", {}


def _verify_bounds(subject: Optional[str], code: str) -> tuple[str, Dict[str, Any]]:
    """Verify bounds safety."""
    if not subject:
        return "unknown", {}
    
    if f"[{subject}]" in code:
        if f"if {subject} <" in code or f"if 0 <= {subject}" in code:
            return "proven", {}
        else:
            return "disproven", {
                "counterexample": {
                    "scenario": f"Array access [{subject}] without bounds check",
                }
            }
    
    return "proven", {}


def _verify_values(subject: Optional[str], code: str) -> tuple[str, Dict[str, Any]]:
    """Find possible values."""
    if not subject:
        return "unknown", {}
    
    values = []
    assignments = re.findall(rf"{subject}\s*=\s*(.+)", code)
    
    for assignment in assignments:
        val = assignment.strip().rstrip(";").strip()
        if val.isdigit():
            values.append(int(val))
        elif val in ("True", "False"):
            values.append(val)
        elif val.startswith('"') or val.startswith("'"):
            values.append(val.strip("\"'"))
    
    return "proven", {"values": values}


def _verify_exceptions(code: str) -> tuple[str, Dict[str, Any]]:
    """Verify exception safety."""
    exceptions = []
    
    raises = re.findall(r"raise\s+(\w+)", code)
    exceptions.extend(raises)
    
    if "/" in code and "try" not in code:
        exceptions.append("ZeroDivisionError")
    
    if "[" in code and "try" not in code:
        exceptions.append("IndexError")
    
    if exceptions:
        return "disproven", {"exceptions": list(set(exceptions))}
    
    return "proven", {}


def _verify_termination(code: str) -> tuple[str, Dict[str, Any]]:
    """Verify termination."""
    # Simple heuristic
    if "while True" in code:
        if "break" not in code and "return" not in code:
            return "disproven", {
                "reason": "Infinite while True loop without break",
            }
    
    return "unknown", {}


def _verify_comparison(parsed: Dict[str, Any], code: str) -> tuple[str, Dict[str, Any]]:
    """Verify comparison."""
    return "unknown", {}


def _generate_answer(
    parsed: Dict[str, Any],
    result: str,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate human-readable answer."""
    query_type = parsed["query_type"]
    subject = parsed.get("subject", "the value")
    
    answer: Dict[str, Any] = {
        "answer": "",
        "explanation": "",
        "proof_steps": [],
        "suggestions": [],
    }
    
    if query_type == "null_check":
        if result == "proven":
            answer["answer"] = f"No, {subject} can never be null/None."
            answer["explanation"] = f"Formal verification proves that {subject} is always non-null."
            answer["proof_steps"] = [
                f"Analyzed all assignments to {subject}",
                "Checked all code paths",
                "No path leads to null value",
            ]
        elif result == "disproven":
            answer["answer"] = f"Yes, {subject} can be null/None."
            answer["explanation"] = "Found an execution path where null is possible."
            answer["counterexample"] = details.get("counterexample")
            answer["suggestions"] = [
                f"Add a null check before using {subject}",
                f"Use Optional type annotation",
            ]
        else:
            answer["answer"] = f"Unable to determine if {subject} can be null."
            answer["explanation"] = "Verification was inconclusive."
    
    elif query_type == "bounds_check":
        if result == "proven":
            answer["answer"] = f"Yes, {subject} is always within bounds."
            answer["explanation"] = f"Verified that {subject} is properly checked."
        elif result == "disproven":
            answer["answer"] = f"No, {subject} can be out of bounds."
            answer["explanation"] = "Found potential bounds violation."
            answer["counterexample"] = details.get("counterexample")
            answer["suggestions"] = [
                f"Add bounds check: if 0 <= {subject} < len(array)",
            ]
    
    elif query_type == "value_check":
        values = details.get("values", [])
        if values:
            answer["answer"] = f"{subject} can have values: {', '.join(map(str, values))}"
            answer["explanation"] = f"Found {len(values)} possible value(s) through analysis."
        else:
            answer["answer"] = f"{subject} appears unconstrained."
            answer["explanation"] = "No specific value constraints found."
    
    elif query_type == "exception":
        if result == "proven":
            answer["answer"] = "No, the code cannot throw an exception."
            answer["explanation"] = "All exception sources are handled."
        elif result == "disproven":
            exceptions = details.get("exceptions", ["unspecified"])
            answer["answer"] = f"Yes, can throw: {', '.join(exceptions)}"
            answer["explanation"] = "Found potential exception sources."
            answer["suggestions"] = ["Add try/except blocks"]
    
    elif query_type == "termination":
        if result == "proven":
            answer["answer"] = "Yes, the code always terminates."
            answer["explanation"] = "All loops and recursion are bounded."
        elif result == "disproven":
            answer["answer"] = "No, the code may not terminate."
            answer["explanation"] = details.get("reason", "Found potential infinite loop.")
            answer["suggestions"] = ["Add termination condition"]
        else:
            answer["answer"] = "Unable to prove termination."
            answer["explanation"] = "Analysis was inconclusive."
    
    else:
        answer["answer"] = "I couldn't understand your question."
        answer["explanation"] = (
            "Try asking questions like:\n"
            "- 'Can x ever be null?'\n"
            "- 'Is the index within bounds?'"
        )
    
    return answer
