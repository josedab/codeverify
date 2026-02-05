"""
Formal Specification API Router

Provides REST API endpoints for formal specification generation:
- Generate specs from code
- Export to SMT-LIB format
- Verify against specifications
- Natural language to Z3 conversion (Formal Spec Assistant)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import Formal Spec Assistant
try:
    from codeverify_agents.formal_spec_assistant import (
        FormalSpecAssistant,
        SpecDomain,
    )
    FORMAL_SPEC_ASSISTANT_AVAILABLE = True
except ImportError:
    FORMAL_SPEC_ASSISTANT_AVAILABLE = False
    FormalSpecAssistant = None
    SpecDomain = None


router = APIRouter(prefix="/api/v1/specs", tags=["formal-specs"])

# Singleton assistant instance
_formal_spec_assistant: Optional[FormalSpecAssistant] = None


def get_formal_spec_assistant() -> FormalSpecAssistant:
    """Get or create the formal spec assistant singleton."""
    global _formal_spec_assistant
    if _formal_spec_assistant is None and FORMAL_SPEC_ASSISTANT_AVAILABLE:
        _formal_spec_assistant = FormalSpecAssistant()
    return _formal_spec_assistant


# =============================================================================
# Request/Response Models
# =============================================================================

class GenerateSpecsRequest(BaseModel):
    """Request to generate specifications from code."""
    code: str = Field(..., description="Source code to analyze")
    language: str = Field("python", description="Programming language")
    include_inferred: bool = Field(
        True, description="Include inferred specifications"
    )
    include_type_specs: bool = Field(
        True, description="Include type-based specifications"
    )


class SpecificationResponse(BaseModel):
    """Response with generated specifications."""
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    module_invariants: List[Dict[str, Any]] = []
    generated_at: float
    total_specs: int


class ExportSMTLibRequest(BaseModel):
    """Request to export specifications as SMT-LIB."""
    specs: Dict[str, Any] = Field(..., description="Specifications to export")
    function_name: Optional[str] = Field(
        None, description="Export specific function only"
    )


class VerifyAgainstSpecsRequest(BaseModel):
    """Request to verify code against specifications."""
    code: str = Field(..., description="Code to verify")
    specs: Dict[str, Any] = Field(..., description="Specifications to verify against")
    language: str = Field("python", description="Programming language")


class VerificationResult(BaseModel):
    """Result of verification against specs."""
    verified: bool
    violations: List[Dict[str, Any]]
    checked: int
    passed: int
    failed: int


class ContractDocstringRequest(BaseModel):
    """Request to generate contract-style docstring."""
    function_name: str = Field(..., description="Function name")
    specs: Dict[str, Any] = Field(..., description="Specifications")


# =============================================================================
# NL-to-Z3 Conversion Models (Formal Spec Assistant)
# =============================================================================


class NLToZ3Request(BaseModel):
    """Request to convert natural language to Z3 specification."""
    specification: str = Field(..., description="Natural language specification", min_length=1)
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context (function signature, types, etc.)"
    )


class NLToZ3Response(BaseModel):
    """Response with Z3 conversion result."""
    success: bool
    z3_expr: Optional[str] = None
    smtlib: Optional[str] = None
    python_assert: Optional[str] = None
    explanation: str = ""
    confidence: float = 0.0
    variables: Dict[str, str] = {}
    ambiguities: List[str] = []
    clarification_questions: List[str] = []
    processing_time_ms: float = 0.0


class BatchNLToZ3Request(BaseModel):
    """Request to convert multiple specifications."""
    specifications: List[str] = Field(..., description="List of natural language specs", min_length=1)
    context: Optional[Dict[str, Any]] = Field(None, description="Shared context")


class ValidateSpecRequest(BaseModel):
    """Request to validate a Z3 specification."""
    z3_expr: str = Field(..., description="Z3 expression to validate")
    variables: Dict[str, str] = Field(..., description="Variable name to type mapping")


class ValidateSpecResponse(BaseModel):
    """Response with validation result."""
    is_satisfiable: bool
    message: Optional[str] = None
    model: Optional[Dict[str, Any]] = None


class RefineSpecRequest(BaseModel):
    """Request to refine a specification with feedback."""
    original_spec: str = Field(..., description="Original natural language spec")
    current_z3: str = Field(..., description="Current Z3 expression")
    feedback: str = Field(..., description="User feedback for refinement")


class SuggestSpecsRequest(BaseModel):
    """Request to suggest specifications from function signature."""
    function_signature: str = Field(..., description="Function signature")
    docstring: Optional[str] = Field(None, description="Function docstring")


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/generate",
    response_model=SpecificationResponse,
    summary="Generate Specifications",
    description="Generate formal specifications from source code"
)
async def generate_specifications(
    request: GenerateSpecsRequest,
) -> SpecificationResponse:
    """
    Generate formal specifications from code.
    
    Analyzes code to extract:
    - Preconditions
    - Postconditions
    - Invariants
    - Type constraints
    """
    result = _generate_specs(
        request.code,
        request.language,
        request.include_inferred,
        request.include_type_specs,
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Count total specs
    total = 0
    for func in result.get("functions", []):
        total += len(func.get("preconditions", []))
        total += len(func.get("postconditions", []))
        total += len(func.get("invariants", []))
    
    for cls in result.get("classes", []):
        total += len(cls.get("invariants", []))
    
    return SpecificationResponse(
        functions=result.get("functions", []),
        classes=result.get("classes", []),
        module_invariants=result.get("module_invariants", []),
        generated_at=result.get("generated_at", time.time()),
        total_specs=total,
    )


@router.post(
    "/export/smtlib",
    summary="Export to SMT-LIB",
    description="Export specifications to SMT-LIB format"
)
async def export_smtlib(request: ExportSMTLibRequest) -> Dict[str, Any]:
    """Export specifications to SMT-LIB format for Z3 solver."""
    functions = request.specs.get("functions", [])
    
    if request.function_name:
        functions = [
            f for f in functions
            if f.get("function_name") == request.function_name
        ]
        if not functions:
            raise HTTPException(
                status_code=404,
                detail=f"Function not found: {request.function_name}"
            )
    
    exports = {}
    for func in functions:
        smt_lib = _generate_smtlib(func)
        exports[func["function_name"]] = smt_lib
    
    return {
        "exports": exports,
        "function_count": len(exports),
    }


@router.post(
    "/verify",
    response_model=VerificationResult,
    summary="Verify Against Specs",
    description="Verify code against formal specifications"
)
async def verify_against_specs(
    request: VerifyAgainstSpecsRequest,
) -> VerificationResult:
    """
    Verify code against specifications.
    
    Checks if code satisfies all preconditions, postconditions, and invariants.
    """
    result = _verify_code(
        request.code,
        request.specs,
        request.language,
    )
    
    return VerificationResult(
        verified=result["verified"],
        violations=result["violations"],
        checked=result["checked"],
        passed=result["passed"],
        failed=result["failed"],
    )


@router.post(
    "/docstring",
    summary="Generate Contract Docstring",
    description="Generate contract-style docstring from specifications"
)
async def generate_docstring(
    request: ContractDocstringRequest,
) -> Dict[str, Any]:
    """Generate a contract-style docstring from specifications."""
    functions = request.specs.get("functions", [])
    func = next(
        (f for f in functions if f.get("function_name") == request.function_name),
        None,
    )
    
    if not func:
        raise HTTPException(
            status_code=404,
            detail=f"Function not found: {request.function_name}"
        )
    
    docstring = _generate_docstring(func)
    
    return {
        "function_name": request.function_name,
        "docstring": docstring,
    }


@router.post(
    "/z3-assertions",
    summary="Generate Z3 Assertions",
    description="Generate Z3 Python code for contract verification"
)
async def generate_z3_assertions(
    request: ContractDocstringRequest,
) -> Dict[str, Any]:
    """Generate Z3 Python assertions for contract verification."""
    functions = request.specs.get("functions", [])
    func = next(
        (f for f in functions if f.get("function_name") == request.function_name),
        None,
    )
    
    if not func:
        raise HTTPException(
            status_code=404,
            detail=f"Function not found: {request.function_name}"
        )
    
    z3_code = _generate_z3_code(func)
    
    return {
        "function_name": request.function_name,
        "z3_code": z3_code,
    }


@router.get(
    "/templates",
    summary="Get Spec Templates",
    description="Get common specification templates"
)
async def get_templates() -> Dict[str, Any]:
    """Get common specification templates."""
    return {
        "templates": [
            {
                "name": "null_check",
                "description": "Parameter must not be null/None",
                "precondition": "{param} is not None",
                "z3": "{param} != None",
                "python": "assert {param} is not None",
            },
            {
                "name": "positive",
                "description": "Value must be positive",
                "precondition": "{param} > 0",
                "z3": "{param} > 0",
                "python": "assert {param} > 0",
            },
            {
                "name": "non_negative",
                "description": "Value must be non-negative",
                "precondition": "{param} >= 0",
                "z3": "{param} >= 0",
                "python": "assert {param} >= 0",
            },
            {
                "name": "non_empty",
                "description": "Collection must be non-empty",
                "precondition": "len({param}) > 0",
                "z3": "Length({param}) > 0",
                "python": "assert len({param}) > 0",
            },
            {
                "name": "in_range",
                "description": "Value must be in range",
                "precondition": "{min} <= {param} <= {max}",
                "z3": "And({param} >= {min}, {param} <= {max})",
                "python": "assert {min} <= {param} <= {max}",
            },
            {
                "name": "valid_index",
                "description": "Index must be valid for collection",
                "precondition": "0 <= {index} < len({collection})",
                "z3": "And({index} >= 0, {index} < Length({collection}))",
                "python": "assert 0 <= {index} < len({collection})",
            },
        ],
    }


# =============================================================================
# NL-to-Z3 Conversion Endpoints (Formal Spec Assistant)
# =============================================================================


@router.post(
    "/nl-to-z3",
    response_model=NLToZ3Response,
    summary="Convert Natural Language to Z3",
    description="Convert a natural language specification to Z3 formal specification"
)
async def convert_nl_to_z3(request: NLToZ3Request) -> NLToZ3Response:
    """
    Convert natural language specification to Z3.

    Example inputs:
    - "x must be positive"
    - "the sum of a and b must equal total"
    - "if user is admin then can_delete must be true"
    """
    if not FORMAL_SPEC_ASSISTANT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Formal Spec Assistant is not available"
        )

    assistant = get_formal_spec_assistant()
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Formal Spec Assistant"
        )

    result = await assistant.convert(
        request.specification,
        request.context or {},
    )

    return NLToZ3Response(
        success=result.success,
        z3_expr=result.z3_expr,
        smtlib=result.smtlib,
        python_assert=result.python_assert,
        explanation=result.explanation,
        confidence=result.confidence,
        variables=result.parsed_spec.variables,
        ambiguities=result.parsed_spec.ambiguities,
        clarification_questions=result.parsed_spec.clarification_questions,
        processing_time_ms=result.processing_time_ms,
    )


@router.post(
    "/nl-to-z3/batch",
    summary="Batch Convert NL to Z3",
    description="Convert multiple natural language specifications to Z3"
)
async def convert_nl_to_z3_batch(request: BatchNLToZ3Request) -> Dict[str, Any]:
    """Convert multiple natural language specifications to Z3."""
    if not FORMAL_SPEC_ASSISTANT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Formal Spec Assistant is not available"
        )

    assistant = get_formal_spec_assistant()
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Formal Spec Assistant"
        )

    results = await assistant.convert_batch(
        request.specifications,
        request.context,
    )

    return {
        "results": [
            {
                "success": r.success,
                "z3_expr": r.z3_expr,
                "smtlib": r.smtlib,
                "python_assert": r.python_assert,
                "confidence": r.confidence,
                "explanation": r.explanation,
            }
            for r in results
        ],
        "total": len(results),
        "successful": sum(1 for r in results if r.success),
    }


@router.post(
    "/nl-to-z3/validate",
    response_model=ValidateSpecResponse,
    summary="Validate Z3 Specification",
    description="Validate a Z3 specification using the Z3 solver"
)
async def validate_z3_spec(request: ValidateSpecRequest) -> ValidateSpecResponse:
    """
    Validate a Z3 specification.

    Checks if the specification is satisfiable and returns a model if so.
    """
    if not FORMAL_SPEC_ASSISTANT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Formal Spec Assistant is not available"
        )

    assistant = get_formal_spec_assistant()
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Formal Spec Assistant"
        )

    is_sat, message, model = await assistant.validate_spec(
        request.z3_expr,
        request.variables,
    )

    return ValidateSpecResponse(
        is_satisfiable=is_sat,
        message=message,
        model=model,
    )


@router.post(
    "/nl-to-z3/refine",
    response_model=NLToZ3Response,
    summary="Refine Specification",
    description="Refine a specification based on user feedback"
)
async def refine_spec(request: RefineSpecRequest) -> NLToZ3Response:
    """
    Refine a specification based on user feedback.

    Useful for interactive refinement when the initial conversion
    doesn't match user intent.
    """
    if not FORMAL_SPEC_ASSISTANT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Formal Spec Assistant is not available"
        )

    assistant = get_formal_spec_assistant()
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Formal Spec Assistant"
        )

    # First convert to get current result
    current_result = await assistant.convert(request.original_spec, {})
    current_result.z3_expr = request.current_z3

    # Refine with feedback
    result = await assistant.refine_with_feedback(
        request.original_spec,
        request.feedback,
        current_result,
    )

    return NLToZ3Response(
        success=result.success,
        z3_expr=result.z3_expr,
        smtlib=result.smtlib,
        python_assert=result.python_assert,
        explanation=result.explanation,
        confidence=result.confidence,
        variables=result.parsed_spec.variables,
        ambiguities=result.parsed_spec.ambiguities,
        clarification_questions=result.parsed_spec.clarification_questions,
        processing_time_ms=result.processing_time_ms,
    )


@router.post(
    "/nl-to-z3/suggest",
    summary="Suggest Specifications",
    description="Suggest specifications based on function signature"
)
async def suggest_specs(request: SuggestSpecsRequest) -> Dict[str, Any]:
    """
    Suggest specifications based on function signature.

    Analyzes the function signature and docstring to suggest
    relevant preconditions and postconditions.
    """
    if not FORMAL_SPEC_ASSISTANT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Formal Spec Assistant is not available"
        )

    assistant = get_formal_spec_assistant()
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Formal Spec Assistant"
        )

    suggestions = assistant.suggest_specifications(
        request.function_signature,
        request.docstring,
    )

    return {
        "suggestions": suggestions,
        "count": len(suggestions),
    }


@router.get(
    "/nl-to-z3/templates",
    summary="Get NL Template Library",
    description="Get the full NL-to-Z3 template library"
)
async def get_nl_template_library() -> Dict[str, Any]:
    """Get the full template library for NL-to-Z3 conversion."""
    if not FORMAL_SPEC_ASSISTANT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Formal Spec Assistant is not available"
        )

    assistant = get_formal_spec_assistant()
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Formal Spec Assistant"
        )

    templates = assistant.get_template_library()

    return {
        "templates": templates,
        "count": len(templates),
    }


@router.get(
    "/nl-to-z3/templates/{domain}",
    summary="Get Templates by Domain",
    description="Get NL-to-Z3 templates for a specific domain"
)
async def get_templates_by_domain(domain: str) -> Dict[str, Any]:
    """Get templates for a specific domain (numeric, string, collection, etc.)."""
    if not FORMAL_SPEC_ASSISTANT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Formal Spec Assistant is not available"
        )

    assistant = get_formal_spec_assistant()
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Formal Spec Assistant"
        )

    templates = assistant.get_templates_by_domain(domain)

    if not templates:
        valid_domains = ["general", "numeric", "string", "collection", "financial",
                        "authentication", "data_validation", "concurrency", "memory"]
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain}' not found. Valid domains: {valid_domains}"
        )

    return {
        "domain": domain,
        "templates": templates,
        "count": len(templates),
    }


@router.get(
    "/nl-to-z3/stats",
    summary="Get Assistant Statistics",
    description="Get statistics about the Formal Spec Assistant"
)
async def get_assistant_stats() -> Dict[str, Any]:
    """Get statistics about the Formal Spec Assistant."""
    if not FORMAL_SPEC_ASSISTANT_AVAILABLE:
        return {
            "available": False,
            "message": "Formal Spec Assistant is not available",
        }

    assistant = get_formal_spec_assistant()
    if not assistant:
        return {
            "available": False,
            "message": "Failed to initialize assistant",
        }

    stats = assistant.get_statistics()
    stats["available"] = True

    return stats


# =============================================================================
# Helper Functions
# =============================================================================

def _generate_specs(
    code: str,
    language: str,
    include_inferred: bool,
    include_type_specs: bool,
) -> Dict[str, Any]:
    """Generate specifications from code."""
    import ast
    import re
    
    result = {
        "functions": [],
        "classes": [],
        "module_invariants": [],
        "generated_at": time.time(),
    }
    
    if language == "python":
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
        
        spec_counter = [0]
        
        def gen_id():
            spec_counter[0] += 1
            return f"spec_{spec_counter[0]:04d}"
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_spec = _analyze_python_function(
                    node,
                    gen_id,
                    include_inferred,
                    include_type_specs,
                )
                result["functions"].append(func_spec)
            
            elif isinstance(node, ast.ClassDef):
                class_spec = _analyze_python_class(node, gen_id)
                result["classes"].append(class_spec)
    
    elif language in ("typescript", "javascript"):
        result = _analyze_typescript(
            code, include_inferred, include_type_specs
        )
    
    return result


def _analyze_python_function(
    node: ast.FunctionDef,
    gen_id,
    include_inferred: bool,
    include_type_specs: bool,
) -> Dict[str, Any]:
    """Analyze a Python function for specifications."""
    func_spec = {
        "function_name": node.name,
        "parameters": [],
        "return_type": None,
        "preconditions": [],
        "postconditions": [],
        "invariants": [],
        "modifies": [],
        "raises": [],
    }
    
    # Extract parameters
    for arg in node.args.args:
        param = {"name": arg.arg, "type": None}
        if arg.annotation:
            param["type"] = ast.unparse(arg.annotation)
        func_spec["parameters"].append(param)
    
    # Extract return type
    if node.returns:
        func_spec["return_type"] = ast.unparse(node.returns)
    
    # Parse docstring for specs
    docstring = ast.get_docstring(node)
    if docstring:
        _extract_docstring_specs(docstring, func_spec, gen_id)
    
    if include_inferred:
        # Infer from validation code
        _infer_preconditions(node, func_spec, gen_id)
        
        # Infer from return statements
        _infer_postconditions(node, func_spec, gen_id)
    
    if include_type_specs:
        # Add type-based specs
        _add_type_specs(node, func_spec, gen_id)
    
    # Find raised exceptions
    _find_raises(node, func_spec)
    
    # Find modifications
    _find_modifications(node, func_spec)
    
    return func_spec


def _extract_docstring_specs(
    docstring: str,
    func_spec: Dict[str, Any],
    gen_id,
) -> None:
    """Extract specs from docstring."""
    import re
    
    # Look for "Requires:" or "Precondition:"
    requires_match = re.search(
        r'(?:Requires|Precondition|Pre):\s*(.*?)(?:\n\n|\n(?=[A-Z])|$)',
        docstring,
        re.DOTALL | re.IGNORECASE,
    )
    if requires_match:
        for line in requires_match.group(1).split('\n'):
            line = line.strip(' -')
            if line:
                func_spec["preconditions"].append({
                    "spec_id": gen_id(),
                    "spec_type": "precondition",
                    "description": line,
                    "source": "documented",
                    "confidence": 0.95,
                })
    
    # Look for "Ensures:" or "Postcondition:"
    ensures_match = re.search(
        r'(?:Ensures|Postcondition|Post|Returns):\s*(.*?)(?:\n\n|\n(?=[A-Z])|$)',
        docstring,
        re.DOTALL | re.IGNORECASE,
    )
    if ensures_match:
        for line in ensures_match.group(1).split('\n'):
            line = line.strip(' -')
            if line:
                func_spec["postconditions"].append({
                    "spec_id": gen_id(),
                    "spec_type": "postcondition",
                    "description": line,
                    "source": "documented",
                    "confidence": 0.95,
                })


def _infer_preconditions(
    node: ast.FunctionDef,
    func_spec: Dict[str, Any],
    gen_id,
) -> None:
    """Infer preconditions from code."""
    import ast
    
    for stmt in node.body[:5]:  # Check first few statements
        # Look for: if x < 0: raise ValueError
        if isinstance(stmt, ast.If):
            raises = any(isinstance(s, ast.Raise) for s in stmt.body)
            if raises and isinstance(stmt.test, ast.Compare):
                try:
                    condition = ast.unparse(stmt.test)
                    negated = _negate_condition(condition)
                    func_spec["preconditions"].append({
                        "spec_id": gen_id(),
                        "spec_type": "precondition",
                        "description": f"Requires: {negated}",
                        "python_assertion": f"assert {negated}",
                        "z3_formula": negated,
                        "source": "inferred",
                        "confidence": 0.9,
                    })
                except Exception:
                    pass
        
        # Direct assertions
        elif isinstance(stmt, ast.Assert):
            try:
                condition = ast.unparse(stmt.test)
                func_spec["preconditions"].append({
                    "spec_id": gen_id(),
                    "spec_type": "precondition",
                    "description": f"Asserts: {condition}",
                    "python_assertion": f"assert {condition}",
                    "z3_formula": condition,
                    "source": "explicit",
                    "confidence": 1.0,
                })
            except Exception:
                pass


def _negate_condition(condition: str) -> str:
    """Negate a Python condition."""
    negations = {
        " < ": " >= ",
        " > ": " <= ",
        " <= ": " > ",
        " >= ": " < ",
        " == ": " != ",
        " != ": " == ",
        " is None": " is not None",
        " is not None": " is None",
    }
    
    for old, new in negations.items():
        if old in condition:
            return condition.replace(old, new)
    
    return f"not ({condition})"


def _infer_postconditions(
    node: ast.FunctionDef,
    func_spec: Dict[str, Any],
    gen_id,
) -> None:
    """Infer postconditions from return type and statements."""
    import ast
    
    if node.returns:
        return_type = ast.unparse(node.returns)
        
        if "Optional" in return_type or "None" in return_type:
            func_spec["postconditions"].append({
                "spec_id": gen_id(),
                "spec_type": "postcondition",
                "description": "May return None",
                "source": "type_annotation",
                "confidence": 1.0,
            })
        else:
            func_spec["postconditions"].append({
                "spec_id": gen_id(),
                "spec_type": "postcondition",
                "description": f"Returns value of type {return_type}",
                "source": "type_annotation",
                "confidence": 1.0,
            })


def _add_type_specs(
    node: ast.FunctionDef,
    func_spec: Dict[str, Any],
    gen_id,
) -> None:
    """Add type-based specifications."""
    import ast
    
    for arg in node.args.args:
        if arg.annotation:
            type_str = ast.unparse(arg.annotation)
            func_spec["preconditions"].append({
                "spec_id": gen_id(),
                "spec_type": "type_constraint",
                "description": f"{arg.arg} is of type {type_str}",
                "variables": [arg.arg],
                "source": "type_annotation",
                "confidence": 1.0,
            })


def _find_raises(node: ast.FunctionDef, func_spec: Dict[str, Any]) -> None:
    """Find exceptions raised by function."""
    import ast
    
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Raise):
            if stmt.exc:
                if isinstance(stmt.exc, ast.Call):
                    if isinstance(stmt.exc.func, ast.Name):
                        exc_name = stmt.exc.func.id
                        if exc_name not in func_spec["raises"]:
                            func_spec["raises"].append(exc_name)
                elif isinstance(stmt.exc, ast.Name):
                    exc_name = stmt.exc.id
                    if exc_name not in func_spec["raises"]:
                        func_spec["raises"].append(exc_name)


def _find_modifications(node: ast.FunctionDef, func_spec: Dict[str, Any]) -> None:
    """Find what the function modifies."""
    import ast
    
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Attribute):
                    if isinstance(target.value, ast.Name):
                        mod = f"{target.value.id}.{target.attr}"
                        if mod not in func_spec["modifies"]:
                            func_spec["modifies"].append(mod)


def _analyze_python_class(node: ast.ClassDef, gen_id) -> Dict[str, Any]:
    """Analyze a Python class for invariants."""
    return {
        "class_name": node.name,
        "invariants": [],
    }


def _analyze_typescript(
    code: str,
    include_inferred: bool,
    include_type_specs: bool,
) -> Dict[str, Any]:
    """Analyze TypeScript code for specifications."""
    import re
    
    result = {
        "functions": [],
        "classes": [],
        "interfaces": [],
        "generated_at": time.time(),
    }
    
    spec_counter = [0]
    
    def gen_id():
        spec_counter[0] += 1
        return f"spec_{spec_counter[0]:04d}"
    
    # Find functions
    func_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^\{]+))?\s*\{'
    for match in re.finditer(func_pattern, code):
        func_name = match.group(1)
        params_str = match.group(2)
        return_type = match.group(3).strip() if match.group(3) else None
        
        func_spec = {
            "function_name": func_name,
            "parameters": _parse_ts_params(params_str),
            "return_type": return_type,
            "preconditions": [],
            "postconditions": [],
            "invariants": [],
        }
        
        if return_type and include_type_specs:
            if "null" in return_type or "undefined" in return_type:
                func_spec["postconditions"].append({
                    "spec_id": gen_id(),
                    "description": "May return null/undefined",
                    "source": "type_annotation",
                    "confidence": 1.0,
                })
        
        result["functions"].append(func_spec)
    
    return result


def _parse_ts_params(params_str: str) -> List[Dict[str, Any]]:
    """Parse TypeScript parameters."""
    params = []
    if not params_str.strip():
        return params
    
    for param in params_str.split(','):
        param = param.strip()
        if ':' in param:
            parts = param.split(':')
            name = parts[0].strip().replace('?', '')
            type_str = parts[1].strip()
            params.append({"name": name, "type": type_str})
        elif param:
            params.append({"name": param, "type": None})
    
    return params


def _generate_smtlib(func: Dict[str, Any]) -> str:
    """Generate SMT-LIB for a function specification."""
    lines = [
        "; SMT-LIB Specification",
        f"; Function: {func['function_name']}",
        "; Generated by CodeVerify",
        "",
    ]
    
    # Declare variables from parameters
    for param in func.get("parameters", []):
        param_type = param.get("type", "Int")
        smt_type = "Int"
        if param_type:
            if "str" in param_type.lower():
                smt_type = "String"
            elif "bool" in param_type.lower():
                smt_type = "Bool"
            elif "float" in param_type.lower():
                smt_type = "Real"
        lines.append(f"(declare-const {param['name']} {smt_type})")
    
    lines.append("")
    
    # Add preconditions
    if func.get("preconditions"):
        lines.append("; Preconditions")
        for pre in func["preconditions"]:
            z3 = pre.get("z3_formula", "")
            desc = pre.get("description", "")
            if z3:
                smt = _z3_to_smtlib(z3)
                lines.append(f"(assert {smt}) ; {desc}")
    
    lines.extend([
        "",
        "(check-sat)",
        "(get-model)",
    ])
    
    return "\n".join(lines)


def _z3_to_smtlib(z3_formula: str) -> str:
    """Convert Z3-style formula to SMT-LIB."""
    import re
    
    smt = z3_formula
    
    # Handle comparisons
    comparisons = [
        (r"(\w+)\s*>=\s*(\w+)", r"(>= \1 \2)"),
        (r"(\w+)\s*<=\s*(\w+)", r"(<= \1 \2)"),
        (r"(\w+)\s*>\s*(\w+)", r"(> \1 \2)"),
        (r"(\w+)\s*<\s*(\w+)", r"(< \1 \2)"),
        (r"(\w+)\s*==\s*(\w+)", r"(= \1 \2)"),
        (r"(\w+)\s*!=\s*(\w+)", r"(distinct \1 \2)"),
    ]
    
    for pattern, replacement in comparisons:
        smt = re.sub(pattern, replacement, smt)
    
    return smt


def _generate_docstring(func: Dict[str, Any]) -> str:
    """Generate contract-style docstring."""
    lines = ['"""']
    
    if func.get("preconditions"):
        lines.append("")
        lines.append("Requires:")
        for pre in func["preconditions"]:
            lines.append(f"    - {pre.get('description', '')}")
    
    if func.get("postconditions"):
        lines.append("")
        lines.append("Ensures:")
        for post in func["postconditions"]:
            lines.append(f"    - {post.get('description', '')}")
    
    if func.get("modifies"):
        lines.append("")
        lines.append(f"Modifies: {', '.join(func['modifies'])}")
    
    if func.get("raises"):
        lines.append("")
        lines.append("Raises:")
        for exc in func["raises"]:
            lines.append(f"    - {exc}")
    
    lines.append('"""')
    return "\n".join(lines)


def _generate_z3_code(func: Dict[str, Any]) -> str:
    """Generate Z3 Python code for verification."""
    lines = [
        "from z3 import *",
        "",
        f"# Verification for: {func['function_name']}",
        "",
    ]
    
    # Declare variables
    for param in func.get("parameters", []):
        param_type = param.get("type", "")
        z3_type = "Int"
        if "str" in param_type.lower():
            z3_type = "String"
        elif "bool" in param_type.lower():
            z3_type = "Bool"
        elif "float" in param_type.lower():
            z3_type = "Real"
        lines.append(f"{param['name']} = {z3_type}('{param['name']}')")
    
    lines.extend(["", "s = Solver()", ""])
    
    if func.get("preconditions"):
        lines.append("# Preconditions")
        for pre in func["preconditions"]:
            z3 = pre.get("z3_formula", "")
            desc = pre.get("description", "")
            if z3:
                lines.append(f"s.add({z3})  # {desc}")
    
    lines.extend([
        "",
        "# Check satisfiability",
        "result = s.check()",
        "print(f'Result: {result}')",
        "if result == sat:",
        "    print(f'Model: {s.model()}')",
    ])
    
    return "\n".join(lines)


def _verify_code(
    code: str,
    specs: Dict[str, Any],
    language: str,
) -> Dict[str, Any]:
    """Verify code against specifications."""
    result = {
        "verified": True,
        "violations": [],
        "checked": 0,
        "passed": 0,
        "failed": 0,
    }
    
    for func_spec in specs.get("functions", []):
        func_name = func_spec.get("function_name", "")
        
        for pre in func_spec.get("preconditions", []):
            result["checked"] += 1
            
            # Check if precondition is enforced
            python_assert = pre.get("python_assertion", "")
            desc = pre.get("description", "")
            
            enforced = False
            if python_assert and python_assert in code:
                enforced = True
            elif desc:
                # Look for related validation
                for var in pre.get("variables", []):
                    if f"if {var}" in code or f"if not {var}" in code:
                        enforced = True
                        break
            
            if enforced:
                result["passed"] += 1
            else:
                result["failed"] += 1
                result["violations"].append({
                    "function": func_name,
                    "type": "missing_precondition",
                    "spec": pre,
                    "message": f"Precondition not enforced: {desc}",
                })
    
    result["verified"] = result["failed"] == 0
    return result
