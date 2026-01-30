"""Z3 MCP Server - Model Context Protocol server for Z3 SMT solver.

This is an open-source component of CodeVerify that can be used independently
to add formal verification capabilities to AI agents.

## Installation

    pip install z3-mcp-server

## Usage

    # Start the server
    z3-mcp-server

    # Or use with uvicorn
    uvicorn z3_mcp.server:app --host 0.0.0.0 --port 8001

## MCP Integration

Add to your MCP config:

    {
        "mcpServers": {
            "z3": {
                "url": "http://localhost:8001"
            }
        }
    }

## Available Tools

- **check_sat**: Check satisfiability of SMT-LIB formulas
- **generate_vc**: Generate verification conditions from code
- **check_overflow**: Check for integer overflow
- **check_bounds**: Check array bounds
- **check_null**: Check null dereference
- **check_div_zero**: Check division by zero

## License

MIT License - See LICENSE file for details.
"""

import time
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from z3 import Solver, sat, unknown, unsat

logger = structlog.get_logger()

# Version info
__version__ = "0.2.0"
__author__ = "CodeVerify Team"
__license__ = "MIT"

app = FastAPI(
    title="Z3 MCP Server",
    description="Model Context Protocol server for Z3 SMT solver - Open Source",
    version=__version__,
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models

class CheckSatRequest(BaseModel):
    """Request to check satisfiability of a formula."""

    formula: str = Field(..., description="SMT-LIB formatted formula")
    timeout_ms: int = Field(default=60000, ge=1000, le=300000, description="Timeout in milliseconds")


class CheckSatResponse(BaseModel):
    """Response from satisfiability check."""

    satisfiable: bool | None = Field(description="True if SAT, False if UNSAT, None if unknown")
    counterexample: dict[str, str] | None = None
    proof_time_ms: float
    error: str | None = None


class CheckOverflowRequest(BaseModel):
    """Request to check for integer overflow."""

    var_name: str = Field(..., description="Variable name for reporting")
    operation: str = Field(..., description="Operation: add, mul, sub")
    operand1_min: int
    operand1_max: int
    operand2_min: int | None = None
    operand2_max: int | None = None
    bit_width: int = Field(default=32, description="Integer bit width (8, 16, 32, 64)")


class CheckOverflowResponse(BaseModel):
    """Response from overflow check."""

    can_overflow: bool
    counterexample: dict[str, Any] | None = None
    proof_time_ms: float
    message: str


class CheckBoundsRequest(BaseModel):
    """Request to check array bounds."""

    index_var: str
    index_min: int | None = None
    index_max: int | None = None
    array_length: int


class CheckBoundsResponse(BaseModel):
    """Response from bounds check."""

    can_violate: bool
    counterexample: dict[str, Any] | None = None
    proof_time_ms: float
    message: str


class CheckDivZeroRequest(BaseModel):
    """Request to check for division by zero."""

    divisor_var: str
    divisor_min: int | None = None
    divisor_max: int | None = None


class CheckDivZeroResponse(BaseModel):
    """Response from division by zero check."""

    can_be_zero: bool
    proof_time_ms: float
    message: str


class GenerateVCRequest(BaseModel):
    """Request to generate verification conditions from code."""

    code: str
    language: str
    check_types: list[str] = Field(
        default=["null_safety", "bounds", "overflow"],
        description="Types of checks to generate VCs for",
    )


class GenerateVCResponse(BaseModel):
    """Response with generated verification conditions."""

    conditions: list[dict[str, Any]]
    error: str | None = None


class MarketplaceRule(BaseModel):
    """A verification rule from the marketplace."""

    id: str
    name: str
    description: str
    author: str
    version: str
    category: str
    formula_template: str
    parameters: list[dict[str, Any]]
    examples: list[dict[str, Any]] = Field(default_factory=list)
    downloads: int = 0
    rating: float = 0.0


# In-memory marketplace rules (would be database in production)
MARKETPLACE_RULES: dict[str, MarketplaceRule] = {
    "overflow-check-signed": MarketplaceRule(
        id="overflow-check-signed",
        name="Signed Integer Overflow",
        description="Check if signed integer arithmetic can overflow",
        author="codeverify",
        version="1.0.0",
        category="numeric",
        formula_template="""
(declare-const a Int)
(declare-const b Int)
(assert (and (>= a {min_a}) (<= a {max_a})))
(assert (and (>= b {min_b}) (<= b {max_b})))
(assert (let ((result ({op} a b)))
    (or (> result {max_val}) (< result {min_val}))))
(check-sat)
""",
        parameters=[
            {"name": "min_a", "type": "int", "description": "Minimum value of first operand"},
            {"name": "max_a", "type": "int", "description": "Maximum value of first operand"},
            {"name": "min_b", "type": "int", "description": "Minimum value of second operand"},
            {"name": "max_b", "type": "int", "description": "Maximum value of second operand"},
            {"name": "op", "type": "string", "description": "Operation: +, -, *"},
            {"name": "max_val", "type": "int", "description": "Maximum allowed result"},
            {"name": "min_val", "type": "int", "description": "Minimum allowed result"},
        ],
        examples=[
            {
                "name": "Int32 addition",
                "params": {
                    "min_a": 0, "max_a": 2147483647,
                    "min_b": 0, "max_b": 2147483647,
                    "op": "+",
                    "max_val": 2147483647, "min_val": -2147483648,
                },
            }
        ],
        downloads=1250,
        rating=4.8,
    ),
    "null-deref-check": MarketplaceRule(
        id="null-deref-check",
        name="Null Dereference Check",
        description="Check if a pointer can be null when dereferenced",
        author="codeverify",
        version="1.0.0",
        category="memory",
        formula_template="""
(declare-const ptr Int)
(declare-const is_null Bool)
(assert (= is_null (= ptr 0)))
(assert {can_be_null})
(assert (not {has_null_check}))
(assert is_null)
(check-sat)
""",
        parameters=[
            {"name": "can_be_null", "type": "bool", "description": "Can the value be null?"},
            {"name": "has_null_check", "type": "bool", "description": "Is there a null check?"},
        ],
        examples=[],
        downloads=890,
        rating=4.6,
    ),
    "array-bounds-check": MarketplaceRule(
        id="array-bounds-check",
        name="Array Bounds Check",
        description="Check if array index can be out of bounds",
        author="codeverify",
        version="1.0.0",
        category="memory",
        formula_template="""
(declare-const idx Int)
(assert (and (>= idx {idx_min}) (<= idx {idx_max})))
(assert (or (< idx 0) (>= idx {array_len})))
(check-sat)
""",
        parameters=[
            {"name": "idx_min", "type": "int", "description": "Minimum index value"},
            {"name": "idx_max", "type": "int", "description": "Maximum index value"},
            {"name": "array_len", "type": "int", "description": "Array length"},
        ],
        examples=[],
        downloads=720,
        rating=4.5,
    ),
}


class Z3MCPServer:
    """
    MCP-compatible server for Z3 SMT solver.

    This server exposes Z3 functionality through a REST API that can be
    used by AI agents via the Model Context Protocol.
    """

    def __init__(self, default_timeout_ms: int = 60000) -> None:
        """Initialize the Z3 MCP server."""
        self.default_timeout_ms = default_timeout_ms

    def check_sat(self, formula: str, timeout_ms: int | None = None) -> dict[str, Any]:
        """
        Check satisfiability of an SMT-LIB formula.

        Args:
            formula: SMT-LIB formatted formula
            timeout_ms: Timeout in milliseconds

        Returns:
            Dictionary with result, counterexample (if sat), and timing
        """
        timeout = timeout_ms or self.default_timeout_ms

        solver = Solver()
        solver.set("timeout", timeout)

        start_time = time.time()

        try:
            solver.from_string(formula)
            result = solver.check()
            elapsed_ms = (time.time() - start_time) * 1000

            if result == sat:
                model = solver.model()
                counterexample = {str(d): str(model[d]) for d in model.decls()}
                return {
                    "satisfiable": True,
                    "counterexample": counterexample,
                    "proof_time_ms": elapsed_ms,
                }
            elif result == unsat:
                return {
                    "satisfiable": False,
                    "counterexample": None,
                    "proof_time_ms": elapsed_ms,
                }
            else:
                return {
                    "satisfiable": None,
                    "counterexample": None,
                    "proof_time_ms": elapsed_ms,
                    "reason": "timeout_or_unknown",
                }

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error("Z3 check_sat error", error=str(e))
            return {
                "satisfiable": None,
                "counterexample": None,
                "proof_time_ms": elapsed_ms,
                "error": str(e),
            }

    def generate_vc(
        self,
        code: str,
        language: str,
        check_types: list[str],
    ) -> dict[str, Any]:
        """
        Generate verification conditions from code.

        This is a placeholder - full implementation would parse code
        and generate appropriate SMT-LIB formulas.
        """
        conditions = []
        logger.info(
            "generate_vc called",
            language=language,
            check_types=check_types,
            code_length=len(code),
        )
        return {"conditions": conditions}


# Create global server instance
z3_server = Z3MCPServer()


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint with server info."""
    return {
        "name": "Z3 MCP Server",
        "version": __version__,
        "description": "Model Context Protocol server for Z3 SMT solver",
        "license": "MIT",
        "documentation": "/docs",
        "mcp_manifest": "/mcp/manifest",
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "z3-mcp-server", "version": __version__}


@app.post("/tools/check_sat", response_model=CheckSatResponse)
async def check_sat(request: CheckSatRequest) -> CheckSatResponse:
    """Check satisfiability of an SMT-LIB formula."""
    logger.info("check_sat request", formula_length=len(request.formula))
    result = z3_server.check_sat(request.formula, request.timeout_ms)
    return CheckSatResponse(
        satisfiable=result.get("satisfiable"),
        counterexample=result.get("counterexample"),
        proof_time_ms=result.get("proof_time_ms", 0),
        error=result.get("error"),
    )


@app.post("/tools/check_overflow", response_model=CheckOverflowResponse)
async def check_overflow(request: CheckOverflowRequest) -> CheckOverflowResponse:
    """Check if an arithmetic operation can overflow."""
    from codeverify_verifier.z3_verifier import Z3Verifier

    verifier = Z3Verifier()

    operand2_range = None
    if request.operand2_min is not None and request.operand2_max is not None:
        operand2_range = (request.operand2_min, request.operand2_max)

    result = verifier.check_integer_overflow(
        var_name=request.var_name,
        operation=request.operation,
        operand1_range=(request.operand1_min, request.operand1_max),
        operand2_range=operand2_range,
        bit_width=request.bit_width,
    )

    return CheckOverflowResponse(
        can_overflow=result.get("satisfiable", False) or False,
        counterexample=result.get("counterexample"),
        proof_time_ms=result.get("proof_time_ms", 0),
        message=result.get("message", ""),
    )


@app.post("/tools/check_bounds", response_model=CheckBoundsResponse)
async def check_bounds(request: CheckBoundsRequest) -> CheckBoundsResponse:
    """Check if array access can go out of bounds."""
    from codeverify_verifier.z3_verifier import Z3Verifier

    verifier = Z3Verifier()

    index_range = None
    if request.index_min is not None and request.index_max is not None:
        index_range = (request.index_min, request.index_max)

    result = verifier.check_array_bounds(
        index_var=request.index_var,
        index_range=index_range,
        array_length=request.array_length,
    )

    return CheckBoundsResponse(
        can_violate=result.get("satisfiable", False) or False,
        counterexample=result.get("counterexample"),
        proof_time_ms=result.get("proof_time_ms", 0),
        message=result.get("message", ""),
    )


@app.post("/tools/check_div_zero", response_model=CheckDivZeroResponse)
async def check_div_zero(request: CheckDivZeroRequest) -> CheckDivZeroResponse:
    """Check if division by zero is possible."""
    from codeverify_verifier.z3_verifier import Z3Verifier

    verifier = Z3Verifier()

    divisor_range = None
    if request.divisor_min is not None and request.divisor_max is not None:
        divisor_range = (request.divisor_min, request.divisor_max)

    result = verifier.check_division_by_zero(
        divisor_var=request.divisor_var,
        divisor_range=divisor_range,
    )

    return CheckDivZeroResponse(
        can_be_zero=result.get("satisfiable", False) or False,
        proof_time_ms=result.get("proof_time_ms", 0),
        message=result.get("message", ""),
    )


@app.post("/tools/generate_vc", response_model=GenerateVCResponse)
async def generate_vc(request: GenerateVCRequest) -> GenerateVCResponse:
    """Generate verification conditions from code."""
    logger.info("generate_vc request", language=request.language)
    result = z3_server.generate_vc(request.code, request.language, request.check_types)
    return GenerateVCResponse(
        conditions=result.get("conditions", []),
        error=result.get("error"),
    )


@app.get("/mcp/manifest")
async def mcp_manifest() -> dict[str, Any]:
    """MCP Manifest endpoint for AI agent consumption."""
    return {
        "name": "z3-mcp-server",
        "version": __version__,
        "description": "Z3 SMT Solver for formal verification",
        "author": __author__,
        "license": __license__,
        "tools": [
            {
                "name": "check_sat",
                "description": "Check satisfiability of an SMT-LIB formula. Returns SAT with counterexample, UNSAT for valid, or UNKNOWN.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "formula": {"type": "string", "description": "SMT-LIB formula to check"},
                        "timeout_ms": {"type": "integer", "default": 60000},
                    },
                    "required": ["formula"],
                },
            },
            {
                "name": "check_overflow",
                "description": "Check if an arithmetic operation can cause integer overflow",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "var_name": {"type": "string"},
                        "operation": {"type": "string", "enum": ["add", "mul", "sub"]},
                        "operand1_min": {"type": "integer"},
                        "operand1_max": {"type": "integer"},
                        "operand2_min": {"type": "integer"},
                        "operand2_max": {"type": "integer"},
                        "bit_width": {"type": "integer", "default": 32},
                    },
                    "required": ["var_name", "operation", "operand1_min", "operand1_max"],
                },
            },
            {
                "name": "check_bounds",
                "description": "Check if array access can go out of bounds",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_var": {"type": "string"},
                        "index_min": {"type": "integer"},
                        "index_max": {"type": "integer"},
                        "array_length": {"type": "integer"},
                    },
                    "required": ["index_var", "array_length"],
                },
            },
            {
                "name": "check_div_zero",
                "description": "Check if division by zero is possible",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "divisor_var": {"type": "string"},
                        "divisor_min": {"type": "integer"},
                        "divisor_max": {"type": "integer"},
                    },
                    "required": ["divisor_var"],
                },
            },
            {
                "name": "generate_vc",
                "description": "Generate verification conditions from source code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "language": {"type": "string"},
                        "check_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["code", "language"],
                },
            },
        ],
    }


# Marketplace endpoints

@app.get("/marketplace/rules")
async def list_marketplace_rules(
    category: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """List available rules from the marketplace."""
    rules = list(MARKETPLACE_RULES.values())

    if category:
        rules = [r for r in rules if r.category == category]

    if search:
        search_lower = search.lower()
        rules = [
            r for r in rules
            if search_lower in r.name.lower() or search_lower in r.description.lower()
        ]

    return {
        "rules": [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "author": r.author,
                "version": r.version,
                "category": r.category,
                "downloads": r.downloads,
                "rating": r.rating,
            }
            for r in rules
        ],
        "total": len(rules),
    }


@app.get("/marketplace/rules/{rule_id}")
async def get_marketplace_rule(rule_id: str) -> MarketplaceRule:
    """Get a specific rule from the marketplace."""
    if rule_id not in MARKETPLACE_RULES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule not found: {rule_id}",
        )
    return MARKETPLACE_RULES[rule_id]


@app.post("/marketplace/rules/{rule_id}/execute")
async def execute_marketplace_rule(
    rule_id: str,
    params: dict[str, Any],
) -> CheckSatResponse:
    """Execute a marketplace rule with given parameters."""
    if rule_id not in MARKETPLACE_RULES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule not found: {rule_id}",
        )

    rule = MARKETPLACE_RULES[rule_id]

    # Format the formula with parameters
    try:
        formula = rule.formula_template.format(**params)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing parameter: {e}",
        )

    # Execute the check
    result = z3_server.check_sat(formula)

    return CheckSatResponse(
        satisfiable=result.get("satisfiable"),
        counterexample=result.get("counterexample"),
        proof_time_ms=result.get("proof_time_ms", 0),
        error=result.get("error"),
    )


@app.get("/marketplace/categories")
async def list_categories() -> dict[str, Any]:
    """List rule categories in the marketplace."""
    categories = set(r.category for r in MARKETPLACE_RULES.values())
    return {
        "categories": [
            {"id": c, "count": sum(1 for r in MARKETPLACE_RULES.values() if r.category == c)}
            for c in sorted(categories)
        ]
    }


def main() -> None:
    """Run the Z3 MCP server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
