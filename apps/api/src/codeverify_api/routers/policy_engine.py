"""Verification policy-as-code API router."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class PolicyConditionModel(BaseModel):
    field: str = Field(description="Context field to evaluate (e.g., file_path, severity, language)")
    operator: str = Field(description="Comparison operator: equals, not_equals, contains, matches, greater_than, less_than, in")
    value: Any = Field(description="Value to compare against")


class PolicyRuleModel(BaseModel):
    id: str
    name: str
    description: str
    conditions: list[PolicyConditionModel]
    action: str = Field(description="Action: allow, deny, warn")
    scope: str = Field(default="file", description="Scope: file, function, module, repository")
    verification_depth: str | None = Field(default=None, description="Depth: pattern, static, ai, formal, full")
    priority: int = Field(default=0)
    enabled: bool = True


class PolicySetModel(BaseModel):
    name: str
    version: str = "1"
    description: str = ""
    rules: list[PolicyRuleModel]
    default_action: str = "warn"


class EvaluateRequest(BaseModel):
    policy: PolicySetModel
    context: dict[str, Any] = Field(description="Evaluation context: file_path, severity, language, findings, etc.")


class EvaluationResult(BaseModel):
    rule_id: str
    rule_name: str
    action: str
    matched: bool
    reason: str
    matched_conditions: list[str]


class EvaluateResponse(BaseModel):
    results: list[EvaluationResult]
    overall_action: str
    verification_depth: str | None
    rules_evaluated: int
    rules_matched: int


class VerificationDepthRequest(BaseModel):
    policy: PolicySetModel
    file_path: str
    language: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


# Built-in policy templates
BUILT_IN_POLICIES: list[dict[str, Any]] = [
    {
        "id": "auth-files-require-formal",
        "name": "Auth files require formal verification",
        "description": "Authentication and authorization files must pass formal verification",
        "conditions": [{"field": "file_path", "operator": "matches", "value": ".*(auth|login|session|permission|oauth|jwt).*"}],
        "action": "deny",
        "verification_depth": "formal",
        "priority": 100,
    },
    {
        "id": "test-files-warn-only",
        "name": "Test files only warn",
        "description": "Test files should only produce warnings, never block",
        "conditions": [{"field": "file_path", "operator": "matches", "value": ".*(test_|_test|spec\\.|__tests__).*"}],
        "action": "warn",
        "verification_depth": "static",
        "priority": 90,
    },
    {
        "id": "critical-findings-block",
        "name": "Block on critical findings",
        "description": "Any critical severity finding blocks the PR",
        "conditions": [{"field": "severity", "operator": "equals", "value": "critical"}],
        "action": "deny",
        "priority": 100,
    },
    {
        "id": "generated-code-strict",
        "name": "AI-generated code strict verification",
        "description": "Code flagged as AI-generated gets full verification",
        "conditions": [{"field": "is_ai_generated", "operator": "equals", "value": True}],
        "action": "deny",
        "verification_depth": "full",
        "priority": 80,
    },
    {
        "id": "config-files-skip",
        "name": "Skip config file verification",
        "description": "Configuration files don't need verification",
        "conditions": [{"field": "file_path", "operator": "matches", "value": ".*(config|settings|\\.env|\\.yml|\\.yaml|\\.json|\\.toml).*"}],
        "action": "allow",
        "verification_depth": "pattern",
        "priority": 70,
    },
    {
        "id": "api-endpoints-security",
        "name": "API endpoints require security scan",
        "description": "Files with API route definitions require security analysis",
        "conditions": [{"field": "file_path", "operator": "matches", "value": ".*(router|route|endpoint|controller|handler|view).*"}],
        "action": "warn",
        "verification_depth": "ai",
        "priority": 75,
    },
    {
        "id": "migration-files-skip",
        "name": "Skip migration files",
        "description": "Database migration files are auto-generated",
        "conditions": [{"field": "file_path", "operator": "contains", "value": "migration"}],
        "action": "allow",
        "verification_depth": "pattern",
        "priority": 60,
    },
    {
        "id": "high-complexity-formal",
        "name": "High complexity requires formal verification",
        "description": "Files with high cyclomatic complexity need formal proofs",
        "conditions": [{"field": "complexity_score", "operator": "greater_than", "value": 15}],
        "action": "warn",
        "verification_depth": "formal",
        "priority": 65,
    },
    {
        "id": "dependency-changes-strict",
        "name": "Dependency changes need scrutiny",
        "description": "Changes to lock files and dependency manifests require review",
        "conditions": [{"field": "file_path", "operator": "matches", "value": ".*(package-lock|yarn\\.lock|Pipfile\\.lock|poetry\\.lock|go\\.sum|pom\\.xml).*"}],
        "action": "warn",
        "verification_depth": "ai",
        "priority": 85,
    },
    {
        "id": "documentation-skip",
        "name": "Skip documentation files",
        "description": "Documentation doesn't need code verification",
        "conditions": [{"field": "file_path", "operator": "matches", "value": ".*\\.(md|rst|txt|adoc)$"}],
        "action": "allow",
        "verification_depth": "pattern",
        "priority": 50,
    },
]

import re


def _match_condition(condition: PolicyConditionModel, context: dict[str, Any]) -> bool:
    """Evaluate a single policy condition against context."""
    value = context.get(condition.field)
    if value is None:
        return False

    if condition.operator == "equals":
        return value == condition.value
    elif condition.operator == "not_equals":
        return value != condition.value
    elif condition.operator == "contains":
        return isinstance(value, str) and condition.value in value
    elif condition.operator == "matches":
        return isinstance(value, str) and bool(re.search(condition.value, value))
    elif condition.operator == "greater_than":
        return isinstance(value, (int, float)) and value > condition.value
    elif condition.operator == "less_than":
        return isinstance(value, (int, float)) and value < condition.value
    elif condition.operator == "in":
        return value in condition.value if isinstance(condition.value, list) else False
    return False


def _evaluate_policy(policy: PolicySetModel, context: dict[str, Any]) -> list[EvaluationResult]:
    """Evaluate all rules in a policy set against context."""
    results = []
    sorted_rules = sorted(policy.rules, key=lambda r: r.priority, reverse=True)

    for rule in sorted_rules:
        if not rule.enabled:
            continue

        matched_conditions = []
        all_match = True

        for cond in rule.conditions:
            if _match_condition(cond, context):
                matched_conditions.append(f"{cond.field} {cond.operator} {cond.value}")
            else:
                all_match = False

        reason = f"All conditions matched: {', '.join(matched_conditions)}" if all_match else "Not all conditions matched"
        results.append(EvaluationResult(
            rule_id=rule.id,
            rule_name=rule.name,
            action=rule.action if all_match else policy.default_action,
            matched=all_match,
            reason=reason,
            matched_conditions=matched_conditions,
        ))

    return results


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_policy(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate a policy set against a given context."""
    results = _evaluate_policy(request.policy, request.context)
    matched = [r for r in results if r.matched]

    # Determine overall action (highest priority matched rule wins)
    overall_action = request.policy.default_action
    verification_depth = None
    if matched:
        overall_action = matched[0].action
        # Find verification depth from first matched rule
        for rule in sorted(request.policy.rules, key=lambda r: r.priority, reverse=True):
            if rule.id == matched[0].rule_id and rule.verification_depth:
                verification_depth = rule.verification_depth
                break

    return EvaluateResponse(
        results=results,
        overall_action=overall_action,
        verification_depth=verification_depth,
        rules_evaluated=len(results),
        rules_matched=len(matched),
    )


@router.get("/built-in", response_model=list[PolicyRuleModel])
async def list_built_in_policies() -> list[PolicyRuleModel]:
    """List all built-in policy templates."""
    return [
        PolicyRuleModel(
            id=p["id"],
            name=p["name"],
            description=p["description"],
            conditions=[PolicyConditionModel(**c) for c in p["conditions"]],
            action=p["action"],
            verification_depth=p.get("verification_depth"),
            priority=p.get("priority", 0),
        )
        for p in BUILT_IN_POLICIES
    ]


@router.post("/verification-depth")
async def get_verification_depth(request: VerificationDepthRequest) -> dict[str, Any]:
    """Determine the appropriate verification depth for a file based on policy."""
    context = {**request.context, "file_path": request.file_path}
    if request.language:
        context["language"] = request.language

    results = _evaluate_policy(request.policy, context)
    matched = [r for r in results if r.matched]

    depth = "static"  # default
    for rule in sorted(request.policy.rules, key=lambda r: r.priority, reverse=True):
        if any(r.rule_id == rule.id and r.matched for r in results) and rule.verification_depth:
            depth = rule.verification_depth
            break

    return {
        "file_path": request.file_path,
        "verification_depth": depth,
        "matched_rules": [r.rule_id for r in matched],
        "reason": matched[0].reason if matched else "No rules matched, using default depth",
    }
