"""Multi-Repository Governance API router.

Centralized policy management across GitHub orgs: policy sets, repo groups,
inheritance, merge-blocking enforcement, exception workflows, and audit log.
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

class PolicyRule(BaseModel):
    """A single verification policy rule."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    check_type: str = Field(description="null_safety, bounds_check, injection, etc.")
    severity_threshold: str = Field(default="medium", description="Minimum severity to enforce")
    action: str = Field(default="block", description="block, warn, observe")
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class PolicySet(BaseModel):
    """A named collection of policy rules."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    org_id: str
    name: str
    description: str = ""
    version: int = 1
    rules: list[PolicyRule] = Field(default_factory=list)
    is_default: bool = False
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RepoGroup(BaseModel):
    """A group of repositories that share a policy set."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    org_id: str
    name: str
    description: str = ""
    policy_set_id: str
    repo_patterns: list[str] = Field(default_factory=list, description="Glob patterns or repo names")
    repos: list[str] = Field(default_factory=list, description="Explicit repo full_names")
    inherit_from: str | None = Field(default=None, description="Parent group ID for inheritance")
    mode: str = Field(default="enforce", description="enforce or observe")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PolicyViolation(BaseModel):
    """A detected policy violation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    org_id: str
    repo_full_name: str
    pr_number: int
    policy_set_id: str
    rule_id: str
    rule_name: str
    severity: str
    action_taken: str = Field(description="blocked, warned, observed")
    finding_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PolicyException(BaseModel):
    """An exception allowing a PR to bypass a policy rule."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    org_id: str
    violation_id: str
    repo_full_name: str
    pr_number: int
    rule_id: str
    reason: str
    requested_by: str
    approved_by: str | None = None
    status: str = Field(default="pending", description="pending, approved, rejected, expired")
    expires_at: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class MergeCheckRequest(BaseModel):
    """Request to check if a PR can be merged under current policies."""

    org_id: str
    repo_full_name: str
    pr_number: int
    findings: list[dict[str, Any]] = Field(default_factory=list)


class MergeCheckResponse(BaseModel):
    """Result of a merge policy check."""

    can_merge: bool
    violations: list[PolicyViolation]
    exceptions_applied: list[str]
    mode: str
    policy_set_name: str


class GovernanceAuditEntry(BaseModel):
    id: str
    org_id: str
    action: str
    actor: str
    resource_type: str
    resource_id: str
    details: dict[str, Any]
    timestamp: str


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

_policy_sets: dict[str, PolicySet] = {}
_repo_groups: dict[str, RepoGroup] = {}
_violations: list[PolicyViolation] = []
_exceptions: dict[str, PolicyException] = {}
_governance_audit: list[GovernanceAuditEntry] = []


def _audit(org_id: str, action: str, actor: str, resource_type: str, resource_id: str, details: dict[str, Any]) -> None:
    _governance_audit.append(GovernanceAuditEntry(
        id=str(uuid.uuid4()),
        org_id=org_id,
        action=action,
        actor=actor,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        timestamp=datetime.utcnow().isoformat(),
    ))


def _find_policy_for_repo(org_id: str, repo_full_name: str) -> tuple[PolicySet | None, RepoGroup | None]:
    """Find the policy set that applies to a repo, considering group inheritance."""
    for group in _repo_groups.values():
        if group.org_id != org_id:
            continue
        if repo_full_name in group.repos:
            ps = _policy_sets.get(group.policy_set_id)
            return ps, group
        for pattern in group.repo_patterns:
            if _glob_match(repo_full_name, pattern):
                ps = _policy_sets.get(group.policy_set_id)
                return ps, group

    # Fallback to org default policy
    for ps in _policy_sets.values():
        if ps.org_id == org_id and ps.is_default:
            return ps, None
    return None, None


def _glob_match(name: str, pattern: str) -> bool:
    """Simple glob matching (production would use fnmatch)."""
    if pattern == "*":
        return True
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    if pattern.startswith("*"):
        return name.endswith(pattern[1:])
    return name == pattern


# ---------------------------------------------------------------------------
# Policy Set CRUD
# ---------------------------------------------------------------------------

@router.post("/policies", response_model=PolicySet, status_code=201)
async def create_policy_set(policy_set: PolicySet) -> PolicySet:
    """Create a new policy set."""
    _policy_sets[policy_set.id] = policy_set
    _audit(policy_set.org_id, "policy.created", "system", "policy_set", policy_set.id,
           {"name": policy_set.name, "rules_count": len(policy_set.rules)})
    return policy_set


@router.get("/policies", response_model=list[PolicySet])
async def list_policy_sets(
    org_id: str = Query(description="Organization ID"),
) -> list[PolicySet]:
    """List all policy sets for an organization."""
    return [ps for ps in _policy_sets.values() if ps.org_id == org_id]


@router.get("/policies/{policy_id}", response_model=PolicySet)
async def get_policy_set(policy_id: str) -> PolicySet:
    """Get a policy set by ID."""
    ps = _policy_sets.get(policy_id)
    if not ps:
        raise HTTPException(status_code=404, detail="Policy set not found")
    return ps


@router.put("/policies/{policy_id}", response_model=PolicySet)
async def update_policy_set(policy_id: str, update: PolicySet) -> PolicySet:
    """Update a policy set (creates a new version)."""
    existing = _policy_sets.get(policy_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Policy set not found")

    update.id = policy_id
    update.version = existing.version + 1
    update.updated_at = datetime.utcnow().isoformat()
    _policy_sets[policy_id] = update

    _audit(update.org_id, "policy.updated", "system", "policy_set", policy_id,
           {"version": update.version, "rules_count": len(update.rules)})
    return update


@router.delete("/policies/{policy_id}", status_code=204)
async def delete_policy_set(policy_id: str) -> None:
    """Delete a policy set."""
    ps = _policy_sets.pop(policy_id, None)
    if not ps:
        raise HTTPException(status_code=404, detail="Policy set not found")
    _audit(ps.org_id, "policy.deleted", "system", "policy_set", policy_id, {"name": ps.name})


# ---------------------------------------------------------------------------
# Repo Group CRUD
# ---------------------------------------------------------------------------

@router.post("/groups", response_model=RepoGroup, status_code=201)
async def create_repo_group(group: RepoGroup) -> RepoGroup:
    """Create a repo group with a policy assignment."""
    if group.policy_set_id not in _policy_sets:
        raise HTTPException(status_code=400, detail="Policy set not found")
    _repo_groups[group.id] = group
    _audit(group.org_id, "group.created", "system", "repo_group", group.id,
           {"name": group.name, "repos": len(group.repos), "mode": group.mode})
    return group


@router.get("/groups", response_model=list[RepoGroup])
async def list_repo_groups(org_id: str = Query(description="Organization ID")) -> list[RepoGroup]:
    """List all repo groups for an organization."""
    return [g for g in _repo_groups.values() if g.org_id == org_id]


@router.put("/groups/{group_id}", response_model=RepoGroup)
async def update_repo_group(group_id: str, update: RepoGroup) -> RepoGroup:
    """Update a repo group."""
    if group_id not in _repo_groups:
        raise HTTPException(status_code=404, detail="Repo group not found")
    update.id = group_id
    _repo_groups[group_id] = update
    _audit(update.org_id, "group.updated", "system", "repo_group", group_id, {"name": update.name})
    return update


@router.delete("/groups/{group_id}", status_code=204)
async def delete_repo_group(group_id: str) -> None:
    """Delete a repo group."""
    group = _repo_groups.pop(group_id, None)
    if not group:
        raise HTTPException(status_code=404, detail="Repo group not found")


# ---------------------------------------------------------------------------
# Merge-blocking enforcement
# ---------------------------------------------------------------------------

@router.post("/check-merge", response_model=MergeCheckResponse)
async def check_merge_eligibility(request: MergeCheckRequest) -> MergeCheckResponse:
    """Check if a PR can be merged under the org's governance policies."""
    policy_set, group = _find_policy_for_repo(request.org_id, request.repo_full_name)

    if not policy_set:
        return MergeCheckResponse(
            can_merge=True,
            violations=[],
            exceptions_applied=[],
            mode="none",
            policy_set_name="(no policy)",
        )

    mode = group.mode if group else "enforce"
    violations: list[PolicyViolation] = []
    exceptions_applied: list[str] = []

    for rule in policy_set.rules:
        if not rule.enabled:
            continue

        # Check findings against the rule
        for finding in request.findings:
            finding_type = finding.get("type", "")
            finding_severity = finding.get("severity", "low")

            if finding_type != rule.check_type:
                continue

            severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            if severity_order.get(finding_severity, 0) < severity_order.get(rule.severity_threshold, 0):
                continue

            action = rule.action if mode == "enforce" else "observed"

            violation = PolicyViolation(
                org_id=request.org_id,
                repo_full_name=request.repo_full_name,
                pr_number=request.pr_number,
                policy_set_id=policy_set.id,
                rule_id=rule.id,
                rule_name=rule.name,
                severity=finding_severity,
                action_taken=action,
                finding_id=finding.get("id"),
                details=finding,
            )
            violations.append(violation)
            _violations.append(violation)

    # Check for approved exceptions
    blocking_violations = [v for v in violations if v.action_taken == "block"]
    for v in blocking_violations:
        for exc in _exceptions.values():
            if (exc.status == "approved" and exc.rule_id == v.rule_id
                    and exc.repo_full_name == v.repo_full_name
                    and exc.pr_number == v.pr_number):
                exceptions_applied.append(exc.id)
                v.action_taken = "exception_applied"
                break

    can_merge = all(v.action_taken != "block" for v in violations)

    _audit(request.org_id, "merge.checked", "system", "pull_request",
           f"{request.repo_full_name}#{request.pr_number}",
           {"can_merge": can_merge, "violations": len(violations), "mode": mode})

    return MergeCheckResponse(
        can_merge=can_merge,
        violations=violations,
        exceptions_applied=exceptions_applied,
        mode=mode,
        policy_set_name=policy_set.name,
    )


# ---------------------------------------------------------------------------
# Exception workflows
# ---------------------------------------------------------------------------

@router.post("/exceptions", response_model=PolicyException, status_code=201)
async def request_exception(exception: PolicyException) -> PolicyException:
    """Request an exception to bypass a policy rule."""
    _exceptions[exception.id] = exception
    _audit(exception.org_id, "exception.requested", exception.requested_by,
           "policy_exception", exception.id,
           {"rule_id": exception.rule_id, "reason": exception.reason})
    return exception


@router.put("/exceptions/{exception_id}/approve")
async def approve_exception(
    exception_id: str,
    approver: str = Query(description="Username of the approver"),
) -> PolicyException:
    """Approve a policy exception."""
    exc = _exceptions.get(exception_id)
    if not exc:
        raise HTTPException(status_code=404, detail="Exception not found")
    if exc.status != "pending":
        raise HTTPException(status_code=400, detail=f"Exception is already {exc.status}")

    exc.status = "approved"
    exc.approved_by = approver
    _audit(exc.org_id, "exception.approved", approver, "policy_exception", exception_id,
           {"rule_id": exc.rule_id})
    return exc


@router.put("/exceptions/{exception_id}/reject")
async def reject_exception(
    exception_id: str,
    rejector: str = Query(description="Username of the rejector"),
) -> PolicyException:
    """Reject a policy exception."""
    exc = _exceptions.get(exception_id)
    if not exc:
        raise HTTPException(status_code=404, detail="Exception not found")
    exc.status = "rejected"
    _audit(exc.org_id, "exception.rejected", rejector, "policy_exception", exception_id,
           {"rule_id": exc.rule_id})
    return exc


@router.get("/exceptions", response_model=list[PolicyException])
async def list_exceptions(
    org_id: str = Query(description="Organization ID"),
    status_filter: str | None = Query(default=None, alias="status"),
) -> list[PolicyException]:
    """List policy exceptions."""
    results = [e for e in _exceptions.values() if e.org_id == org_id]
    if status_filter:
        results = [e for e in results if e.status == status_filter]
    return results


# ---------------------------------------------------------------------------
# Violations & Audit
# ---------------------------------------------------------------------------

@router.get("/violations", response_model=list[PolicyViolation])
async def list_violations(
    org_id: str = Query(description="Organization ID"),
    repo: str | None = Query(default=None),
    limit: int = Query(default=50, le=500),
) -> list[PolicyViolation]:
    """List policy violations."""
    results = [v for v in _violations if v.org_id == org_id]
    if repo:
        results = [v for v in results if v.repo_full_name == repo]
    return results[-limit:]


@router.get("/audit", response_model=list[GovernanceAuditEntry])
async def get_governance_audit(
    org_id: str = Query(description="Organization ID"),
    limit: int = Query(default=50, le=500),
    action: str | None = Query(default=None),
) -> list[GovernanceAuditEntry]:
    """Get governance audit log."""
    entries = [e for e in _governance_audit if e.org_id == org_id]
    if action:
        entries = [e for e in entries if action in e.action]
    return entries[-limit:]


@router.get("/overview/{org_id}")
async def get_governance_overview(org_id: str) -> dict[str, Any]:
    """Get a high-level overview of governance status for an org."""
    policies = [ps for ps in _policy_sets.values() if ps.org_id == org_id]
    groups = [g for g in _repo_groups.values() if g.org_id == org_id]
    org_violations = [v for v in _violations if v.org_id == org_id]
    org_exceptions = [e for e in _exceptions.values() if e.org_id == org_id]

    repos_covered = set()
    for g in groups:
        repos_covered.update(g.repos)

    return {
        "org_id": org_id,
        "policy_sets": len(policies),
        "repo_groups": len(groups),
        "repos_governed": len(repos_covered),
        "total_violations": len(org_violations),
        "blocking_violations": sum(1 for v in org_violations if v.action_taken == "block"),
        "pending_exceptions": sum(1 for e in org_exceptions if e.status == "pending"),
        "observe_mode_groups": sum(1 for g in groups if g.mode == "observe"),
        "enforce_mode_groups": sum(1 for g in groups if g.mode == "enforce"),
    }
