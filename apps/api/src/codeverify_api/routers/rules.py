"""Custom Rules API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


class RuleConditionRequest(BaseModel):
    """Request model for a rule condition."""

    field: str = Field(..., description="Field to check: code, line, imports, etc.")
    operator: str = Field(..., description="Operator: contains, matches, equals, etc.")
    value: str | int | float | bool | list[str] = Field(..., description="Value to compare")
    case_sensitive: bool = Field(default=True)
    description: str = Field(default="")


class RuleActionRequest(BaseModel):
    """Request model for a rule action."""

    action_type: str = Field(default="report", description="Action type: report, suggest_fix, block")
    message: str = Field(..., description="Message to display when rule matches")
    fix_template: str | None = Field(default=None, description="Template for suggested fix")


class CreateRuleRequest(BaseModel):
    """Request to create a custom rule."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="")
    rule_type: str = Field(default="pattern", description="Rule type: pattern, ast, semantic, composite")
    severity: str = Field(default="medium", description="Severity: critical, high, medium, low, info")
    scope: str = Field(default="line", description="Scope: file, function, class, block, line")
    conditions: list[RuleConditionRequest] = Field(default_factory=list)
    actions: list[RuleActionRequest] = Field(default_factory=list)
    condition_logic: str = Field(default="AND", description="Logic for combining conditions: AND or OR")
    enabled: bool = Field(default=True)
    languages: list[str] = Field(default_factory=list, description="Languages this rule applies to")
    file_patterns: list[str] = Field(default_factory=list, description="Glob patterns for files")
    exclude_patterns: list[str] = Field(default_factory=list, description="Patterns to exclude")
    tags: list[str] = Field(default_factory=list)


class RuleResponse(BaseModel):
    """Response model for a custom rule."""

    id: str
    name: str
    description: str
    rule_type: str
    severity: str
    scope: str
    conditions: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    condition_logic: str
    enabled: bool
    languages: list[str]
    file_patterns: list[str]
    exclude_patterns: list[str]
    tags: list[str]
    created_at: str
    updated_at: str
    version: int


class TestRuleRequest(BaseModel):
    """Request to test a rule against code."""

    rule: CreateRuleRequest
    code: str = Field(..., description="Code to test the rule against")
    file_path: str = Field(default="test.py", description="Simulated file path")
    language: str = Field(default="python", description="Programming language")


class TestRuleResponse(BaseModel):
    """Response from testing a rule."""

    matches: bool
    violations: list[dict[str, Any]]
    execution_time_ms: float


class EvaluateRulesRequest(BaseModel):
    """Request to evaluate multiple rules against code."""

    rule_ids: list[str] = Field(default_factory=list, description="IDs of rules to evaluate (empty = all)")
    code: str = Field(..., description="Code to evaluate")
    file_path: str = Field(..., description="File path")
    language: str = Field(default="python")


# In-memory rule storage (should use database in production)
_rules: dict[str, dict[str, Any]] = {}


@router.post("", response_model=RuleResponse, status_code=status.HTTP_201_CREATED)
async def create_rule(request: CreateRuleRequest) -> RuleResponse:
    """Create a new custom rule."""
    from uuid import uuid4
    from datetime import datetime

    from codeverify_core.rules import (
        CustomRule,
        RuleType,
        RuleSeverity,
        RuleScope,
        RuleCondition,
        RuleAction,
        ConditionOperator,
    )

    # Convert request to CustomRule
    conditions = [
        RuleCondition(
            id=str(uuid4()),
            field=c.field,
            operator=ConditionOperator(c.operator),
            value=c.value,
            case_sensitive=c.case_sensitive,
            description=c.description,
        )
        for c in request.conditions
    ]

    actions = [
        RuleAction(
            action_type=a.action_type,
            message=a.message,
            fix_template=a.fix_template,
        )
        for a in request.actions
    ]

    rule = CustomRule(
        id=uuid4(),
        name=request.name,
        description=request.description,
        rule_type=RuleType(request.rule_type),
        severity=RuleSeverity(request.severity),
        scope=RuleScope(request.scope),
        conditions=conditions,
        actions=actions,
        condition_logic=request.condition_logic,
        enabled=request.enabled,
        languages=request.languages,
        file_patterns=request.file_patterns,
        exclude_patterns=request.exclude_patterns,
        tags=request.tags,
    )

    # Store rule
    rule_dict = rule.to_dict()
    _rules[str(rule.id)] = rule_dict

    return RuleResponse(**rule_dict)


@router.get("", response_model=list[RuleResponse])
async def list_rules(
    enabled_only: bool = Query(default=False, description="Only return enabled rules"),
    tag: str | None = Query(default=None, description="Filter by tag"),
    language: str | None = Query(default=None, description="Filter by language"),
    severity: str | None = Query(default=None, description="Filter by severity"),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[RuleResponse]:
    """List custom rules with optional filtering."""
    rules = list(_rules.values())

    # Apply filters
    if enabled_only:
        rules = [r for r in rules if r.get("enabled", True)]

    if tag:
        rules = [r for r in rules if tag in r.get("tags", [])]

    if language:
        rules = [
            r for r in rules
            if not r.get("languages") or language in r.get("languages", [])
        ]

    if severity:
        rules = [r for r in rules if r.get("severity") == severity]

    # Pagination
    total = len(rules)
    rules = rules[offset:offset + limit]

    return [RuleResponse(**r) for r in rules]


@router.get("/templates")
async def get_rule_templates() -> dict[str, Any]:
    """Get pre-built rule templates."""
    from codeverify_core.rules import RULE_TEMPLATES

    templates = {}
    for name, rule in RULE_TEMPLATES.items():
        templates[name] = {
            "name": rule.name,
            "description": rule.description,
            "severity": rule.severity.value,
            "tags": rule.tags,
            "rule": rule.to_dict(),
        }

    return {"templates": templates}


@router.get("/{rule_id}", response_model=RuleResponse)
async def get_rule(rule_id: str) -> RuleResponse:
    """Get a specific rule by ID."""
    if rule_id not in _rules:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule not found: {rule_id}",
        )

    return RuleResponse(**_rules[rule_id])


@router.put("/{rule_id}", response_model=RuleResponse)
async def update_rule(rule_id: str, request: CreateRuleRequest) -> RuleResponse:
    """Update an existing rule."""
    from datetime import datetime

    if rule_id not in _rules:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule not found: {rule_id}",
        )

    existing = _rules[rule_id]

    # Update fields
    updated = {
        **existing,
        "name": request.name,
        "description": request.description,
        "rule_type": request.rule_type,
        "severity": request.severity,
        "scope": request.scope,
        "conditions": [c.dict() for c in request.conditions],
        "actions": [a.dict() for a in request.actions],
        "condition_logic": request.condition_logic,
        "enabled": request.enabled,
        "languages": request.languages,
        "file_patterns": request.file_patterns,
        "exclude_patterns": request.exclude_patterns,
        "tags": request.tags,
        "updated_at": datetime.utcnow().isoformat(),
        "version": existing.get("version", 1) + 1,
    }

    _rules[rule_id] = updated
    return RuleResponse(**updated)


@router.delete("/{rule_id}")
async def delete_rule(rule_id: str) -> dict[str, Any]:
    """Delete a rule."""
    if rule_id not in _rules:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule not found: {rule_id}",
        )

    del _rules[rule_id]
    return {"deleted": True, "rule_id": rule_id}


@router.post("/{rule_id}/toggle")
async def toggle_rule(rule_id: str, enabled: bool) -> dict[str, Any]:
    """Enable or disable a rule."""
    if rule_id not in _rules:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule not found: {rule_id}",
        )

    _rules[rule_id]["enabled"] = enabled
    return {"rule_id": rule_id, "enabled": enabled}


@router.post("/test", response_model=TestRuleResponse)
async def test_rule(request: TestRuleRequest) -> TestRuleResponse:
    """Test a rule against sample code without saving it."""
    import time
    from uuid import uuid4

    from codeverify_core.rules import (
        CustomRule,
        RuleType,
        RuleSeverity,
        RuleScope,
        RuleCondition,
        RuleAction,
        ConditionOperator,
        RuleEvaluator,
    )

    start_time = time.time()

    # Convert request to CustomRule
    conditions = [
        RuleCondition(
            id=str(uuid4()),
            field=c.field,
            operator=ConditionOperator(c.operator),
            value=c.value,
            case_sensitive=c.case_sensitive,
            description=c.description,
        )
        for c in request.rule.conditions
    ]

    actions = [
        RuleAction(
            action_type=a.action_type,
            message=a.message,
            fix_template=a.fix_template,
        )
        for a in request.rule.actions
    ]

    rule = CustomRule(
        id=uuid4(),
        name=request.rule.name,
        description=request.rule.description,
        rule_type=RuleType(request.rule.rule_type),
        severity=RuleSeverity(request.rule.severity),
        scope=RuleScope(request.rule.scope),
        conditions=conditions,
        actions=actions,
        condition_logic=request.rule.condition_logic,
        enabled=True,
        languages=request.rule.languages,
        file_patterns=request.rule.file_patterns,
        exclude_patterns=request.rule.exclude_patterns,
        tags=request.rule.tags,
    )

    # Evaluate
    evaluator = RuleEvaluator([rule])
    violations = evaluator.evaluate(
        code=request.code,
        file_path=request.file_path,
        language=request.language,
    )

    execution_time = (time.time() - start_time) * 1000

    return TestRuleResponse(
        matches=len(violations) > 0,
        violations=violations,
        execution_time_ms=execution_time,
    )


@router.post("/evaluate", response_model=list[dict[str, Any]])
async def evaluate_rules(request: EvaluateRulesRequest) -> list[dict[str, Any]]:
    """Evaluate rules against code."""
    from codeverify_core.rules import CustomRule, RuleEvaluator

    # Get rules to evaluate
    if request.rule_ids:
        rules_to_eval = [
            _rules[rid] for rid in request.rule_ids
            if rid in _rules
        ]
    else:
        rules_to_eval = list(_rules.values())

    # Convert to CustomRule objects
    custom_rules = [CustomRule.from_dict(r) for r in rules_to_eval]

    # Evaluate
    evaluator = RuleEvaluator(custom_rules)
    violations = evaluator.evaluate(
        code=request.code,
        file_path=request.file_path,
        language=request.language,
    )

    return violations


@router.post("/import")
async def import_rules(rules: list[CreateRuleRequest]) -> dict[str, Any]:
    """Import multiple rules at once."""
    imported = []
    errors = []

    for rule_request in rules:
        try:
            response = await create_rule(rule_request)
            imported.append(response.id)
        except Exception as e:
            errors.append({"name": rule_request.name, "error": str(e)})

    return {
        "imported": len(imported),
        "rule_ids": imported,
        "errors": errors,
    }


@router.get("/export")
async def export_rules(
    format: str = Query(default="json", description="Export format: json or yaml"),
) -> dict[str, Any]:
    """Export all rules."""
    from codeverify_core.rules import CustomRule

    if format == "yaml":
        import yaml
        rules_yaml = []
        for rule_dict in _rules.values():
            rule = CustomRule.from_dict(rule_dict)
            rules_yaml.append(rule.to_yaml())
        return {"format": "yaml", "rules": "\n---\n".join(rules_yaml)}
    else:
        return {"format": "json", "rules": list(_rules.values())}
