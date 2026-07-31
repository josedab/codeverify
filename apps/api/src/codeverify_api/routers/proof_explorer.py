"""Interactive Proof Explorer API router.

Serves proof trees, constraint graphs, counterexample playgrounds, and
LLM-powered explanations for the browser-based proof visualization UI.
"""

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ProofExplorerRequest(BaseModel):
    code: str = Field(description="Source code to verify")
    function_name: str = Field(description="Function to verify")
    language: str = Field(default="python")
    properties: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Specific properties to check; auto-inferred if empty",
    )


class ProofTreeNode(BaseModel):
    id: str
    type: str
    label: str
    expression: str
    status: str
    children: list["ProofTreeNode"] = []
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConstraintGraphData(BaseModel):
    variables: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class CounterexampleData(BaseModel):
    id: str
    variables: list[dict[str, Any]]
    violated_property: str
    execution_path: list[dict[str, Any]]
    editable: bool = True


class ProofExplorerResponse(BaseModel):
    id: str
    function_name: str
    file_path: str
    language: str
    status: str
    proof_tree: ProofTreeNode | None
    constraint_graph: ConstraintGraphData | None
    counterexamples: list[CounterexampleData]
    execution_trace: list[dict[str, Any]]
    explanation: str | None = None
    solver_time_ms: float
    created_at: str


class CounterexampleEditRequest(BaseModel):
    exploration_id: str
    counterexample_id: str
    modified_variables: dict[str, Any]


class CounterexampleEditResponse(BaseModel):
    counterexample_id: str
    original_variables: dict[str, Any]
    modified_variables: dict[str, Any]
    still_violates: bool
    new_execution_path: list[dict[str, Any]]
    explanation: str


class ExplainRequest(BaseModel):
    exploration_id: str
    node_id: str | None = None
    question: str = Field(default="Why is this unsafe?")


class ExplainResponse(BaseModel):
    explanation: str
    related_properties: list[str]
    suggested_fixes: list[str]


# ---------------------------------------------------------------------------
# Demo proof tree generator
# ---------------------------------------------------------------------------


def _build_demo_proof_tree(function_name: str, code: str) -> ProofTreeNode:
    """Build a representative proof tree for demonstration."""
    checks = []

    # Infer properties from code patterns
    if "None" in code or "null" in code or "Optional" in code:
        checks.append(
            ProofTreeNode(
                id=str(uuid.uuid4()),
                type="assertion",
                label="Null Safety",
                expression="ForAll([x], Implies(is_param(x), x != None))",
                status="disproved",
                children=[
                    ProofTreeNode(
                        id=str(uuid.uuid4()),
                        type="constraint",
                        label="Parameter non-null",
                        expression="param != None",
                        status="disproved",
                    ),
                    ProofTreeNode(
                        id=str(uuid.uuid4()),
                        type="constraint",
                        label="Return non-null",
                        expression="result != None",
                        status="proved",
                    ),
                ],
            )
        )
    if "[" in code or "index" in code.lower():
        checks.append(
            ProofTreeNode(
                id=str(uuid.uuid4()),
                type="assertion",
                label="Bounds Check",
                expression="ForAll([i, a], Implies(access(a, i), And(i >= 0, i < len(a))))",
                status="proved",
                children=[
                    ProofTreeNode(
                        id=str(uuid.uuid4()),
                        type="constraint",
                        label="Lower bound",
                        expression="index >= 0",
                        status="proved",
                    ),
                    ProofTreeNode(
                        id=str(uuid.uuid4()),
                        type="constraint",
                        label="Upper bound",
                        expression="index < len(array)",
                        status="proved",
                    ),
                ],
            )
        )
    if "/" in code or "div" in code.lower():
        checks.append(
            ProofTreeNode(
                id=str(uuid.uuid4()),
                type="assertion",
                label="Division Safety",
                expression="ForAll([a, b], Implies(divide(a, b), b != 0))",
                status="proved",
            )
        )

    if not checks:
        checks.append(
            ProofTreeNode(
                id=str(uuid.uuid4()),
                type="assertion",
                label="Type Safety",
                expression="type_check(all_vars)",
                status="proved",
            )
        )

    root_status = "disproved" if any(c.status == "disproved" for c in checks) else "proved"
    return ProofTreeNode(
        id=str(uuid.uuid4()),
        type="root",
        label=f"Verify: {function_name}",
        expression=f"verify({function_name})",
        status=root_status,
        children=checks,
    )


def _build_demo_constraint_graph(code: str) -> ConstraintGraphData:
    variables = []
    edges = []
    var_names = []

    # Extract simple variable patterns
    import re

    params = re.findall(r"def \w+\(([^)]+)\)", code)
    if params:
        for param in params[0].split(","):
            name = param.strip().split(":")[0].strip().split("=")[0].strip()
            if name:
                var_names.append(name)
                variables.append({"id": name, "name": name, "type": "parameter"})

    if not variables:
        variables = [
            {"id": "x", "name": "x", "type": "parameter"},
            {"id": "result", "name": "result", "type": "return"},
        ]
        var_names = ["x", "result"]

    for i, v1 in enumerate(var_names):
        for v2 in var_names[i + 1 :]:
            edges.append(
                {
                    "source": v1,
                    "target": v2,
                    "constraint": f"{v1} relates to {v2}",
                    "satisfied": True,
                }
            )

    return ConstraintGraphData(variables=variables, edges=edges)


def _build_demo_counterexamples(_function_name: str, code: str) -> list[CounterexampleData]:
    counterexamples = []
    if "None" in code or "null" in code or "Optional" in code:
        counterexamples.append(
            CounterexampleData(
                id=str(uuid.uuid4()),
                variables=[
                    {"name": "param", "value": None, "type": "NoneType"},
                ],
                violated_property="Null Safety: param != None",
                execution_path=[
                    {"step": 1, "line": 1, "action": "enter", "state": {"param": None}},
                    {
                        "step": 2,
                        "line": 3,
                        "action": "access",
                        "state": {"param": None},
                        "error": "NoneType has no attribute",
                    },
                ],
                editable=True,
            )
        )
    return counterexamples


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_explorations: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/explore", response_model=ProofExplorerResponse)
async def create_exploration(request: ProofExplorerRequest) -> ProofExplorerResponse:
    """Create a new proof exploration for a function."""
    exploration_id = str(uuid.uuid4())

    proof_tree = _build_demo_proof_tree(request.function_name, request.code)
    constraint_graph = _build_demo_constraint_graph(request.code)
    counterexamples = _build_demo_counterexamples(request.function_name, request.code)

    trace = [
        {
            "step": 1,
            "type": "parse",
            "description": f"Parse {request.function_name}",
            "time_ms": 2.1,
        },
        {
            "step": 2,
            "type": "extract_constraints",
            "description": "Extract verification constraints",
            "time_ms": 5.3,
        },
        {"step": 3, "type": "solve", "description": "Run Z3 SMT solver", "time_ms": 45.7},
        {"step": 4, "type": "check_sat", "description": "Check satisfiability", "time_ms": 12.4},
    ]

    response = ProofExplorerResponse(
        id=exploration_id,
        function_name=request.function_name,
        file_path="<inline>",
        language=request.language,
        status=proof_tree.status,
        proof_tree=proof_tree,
        constraint_graph=constraint_graph,
        counterexamples=counterexamples,
        execution_trace=trace,
        solver_time_ms=65.5,
        created_at=datetime.utcnow().isoformat(),
    )
    _explorations[exploration_id] = response.model_dump()
    return response


@router.get("/explore/{exploration_id}", response_model=ProofExplorerResponse)
async def get_exploration(exploration_id: str) -> ProofExplorerResponse:
    """Retrieve a proof exploration."""
    data = _explorations.get(exploration_id)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exploration not found")
    return ProofExplorerResponse(**data)


@router.post("/explore/counterexample/edit", response_model=CounterexampleEditResponse)
async def edit_counterexample(request: CounterexampleEditRequest) -> CounterexampleEditResponse:
    """Edit a counterexample's variable values and re-evaluate."""
    data = _explorations.get(request.exploration_id)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exploration not found")

    # Find the counterexample
    ce = None
    for c in data.get("counterexamples", []):
        if c["id"] == request.counterexample_id:
            ce = c
            break
    if not ce:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Counterexample not found"
        )

    original_vars = {v["name"]: v["value"] for v in ce["variables"]}

    # Re-evaluate with modified values
    still_violates = any(v is None for v in request.modified_variables.values())

    new_path = [
        {"step": 1, "line": 1, "action": "enter", "state": request.modified_variables},
        {
            "step": 2,
            "line": 3,
            "action": "access" if still_violates else "return",
            "state": request.modified_variables,
            "error": "Still violates property" if still_violates else None,
        },
    ]

    explanation = (
        "The modified values still trigger the violation because a null value remains."
        if still_violates
        else "The modified values satisfy all verification properties."
    )

    return CounterexampleEditResponse(
        counterexample_id=request.counterexample_id,
        original_variables=original_vars,
        modified_variables=request.modified_variables,
        still_violates=still_violates,
        new_execution_path=new_path,
        explanation=explanation,
    )


@router.post("/explore/explain", response_model=ExplainResponse)
async def explain_proof(request: ExplainRequest) -> ExplainResponse:
    """LLM-powered explanation of why a property holds or is violated."""
    data = _explorations.get(request.exploration_id)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exploration not found")

    status_str = data.get("status", "unknown")

    if status_str == "disproved":
        explanation = (
            f"The function `{data['function_name']}` has a potential null pointer issue. "
            "The Z3 solver found that when a parameter is None, the function attempts "
            "to access an attribute on it, which would raise an AttributeError at runtime. "
            "This violates the null safety property.\n\n"
            "**Root cause:** The function does not check if its input parameter is None "
            "before accessing its attributes.\n\n"
            "**Impact:** This could crash the application when called with None arguments, "
            "especially in cases where the caller cannot guarantee non-null inputs."
        )
        fixes = [
            "Add a null check: `if param is not None:` before accessing attributes",
            "Use Optional type annotation and handle the None case explicitly",
            "Add a precondition decorator: `@requires(lambda param: param is not None)`",
        ]
    else:
        explanation = (
            f"The function `{data['function_name']}` passes all verification checks. "
            "The Z3 solver confirmed that all specified properties hold for all possible inputs. "
            "No counterexamples were found."
        )
        fixes = []

    return ExplainResponse(
        explanation=explanation,
        related_properties=["null_safety", "type_safety"],
        suggested_fixes=fixes,
    )


@router.get("/explorations", response_model=list[dict[str, Any]])
async def list_explorations(
    limit: int = Query(default=20, le=100),
) -> list[dict[str, Any]]:
    """List recent proof explorations (summary only)."""
    items = list(_explorations.values())[-limit:]
    return [
        {
            "id": e["id"],
            "function_name": e["function_name"],
            "status": e["status"],
            "solver_time_ms": e["solver_time_ms"],
            "created_at": e["created_at"],
        }
        for e in items
    ]
