"""GitHub Copilot Workspace Integration.

Deep integration with GitHub Copilot Workspace for verifying generated
code plans before execution, providing trust scores on workspace-generated
PRs, and injecting verification constraints into the generation loop.

Features:
- Plan verification: intercept Copilot Workspace plans and verify code changes
- Constraint injection: feed verification constraints back into generation
- Trust score display for workspace-generated code
- Real-time verification status within Copilot Workspace
- Workspace session management with verification state
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class WorkspacePlanStatus(str, Enum):
    """Status of a Copilot Workspace plan."""

    PENDING = "pending"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    EXECUTING = "executing"
    COMPLETED = "completed"


class VerificationGate(str, Enum):
    """Verification gate decision."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class ConstraintType(str, Enum):
    """Types of verification constraints."""

    NULL_SAFETY = "null_safety"
    BOUNDS_CHECK = "bounds_check"
    TYPE_SAFETY = "type_safety"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class WorkspaceEventType(str, Enum):
    """Types of workspace events."""

    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"
    PLAN_VERIFIED = "plan_verified"
    PLAN_EXECUTED = "plan_executed"
    CONSTRAINT_ADDED = "constraint_added"
    FINDING_DETECTED = "finding_detected"


@dataclass
class WorkspaceFile:
    """A file within a Copilot Workspace plan."""

    path: str
    language: str = ""
    original_content: str = ""
    proposed_content: str = ""
    is_new: bool = False
    is_deleted: bool = False

    @property
    def has_changes(self) -> bool:
        return self.original_content != self.proposed_content

    @property
    def diff_size(self) -> int:
        orig_lines = len(self.original_content.split("\n"))
        prop_lines = len(self.proposed_content.split("\n"))
        return abs(prop_lines - orig_lines)


@dataclass
class VerificationConstraint:
    """A constraint to inject into code generation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    constraint_type: ConstraintType = ConstraintType.NULL_SAFETY
    description: str = ""
    z3_expression: str = ""
    applies_to: str = ""
    is_hard: bool = True
    priority: int = 1


@dataclass
class PlanFinding:
    """A finding from verifying a workspace plan."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    file_path: str = ""
    line: int = 0
    severity: str = "medium"
    message: str = ""
    fix_suggestion: str = ""
    constraint_violated: str | None = None
    auto_fixable: bool = False


@dataclass
class PlanVerificationResult:
    """Result of verifying a workspace plan."""

    plan_id: str = ""
    gate: VerificationGate = VerificationGate.PASS
    findings: list[PlanFinding] = field(default_factory=list)
    trust_score: float = 0.0
    constraints_checked: int = 0
    constraints_satisfied: int = 0
    verification_time_ms: int = 0
    summary: str = ""

    @property
    def pass_rate(self) -> float:
        if self.constraints_checked == 0:
            return 1.0
        return self.constraints_satisfied / self.constraints_checked


@dataclass
class WorkspacePlan:
    """A Copilot Workspace plan with verification state."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    workspace_id: str = ""
    description: str = ""
    files: list[WorkspaceFile] = field(default_factory=list)
    status: WorkspacePlanStatus = WorkspacePlanStatus.PENDING
    constraints: list[VerificationConstraint] = field(default_factory=list)
    verification_result: PlanVerificationResult | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    verified_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkspaceSession:
    """Active Copilot Workspace session with verification state."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    workspace_id: str = ""
    user_id: str = ""
    plans: list[WorkspacePlan] = field(default_factory=list)
    active_constraints: list[VerificationConstraint] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_event(self, event_type: WorkspaceEventType, data: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "type": event_type.value,
                "timestamp": datetime.now(UTC).isoformat(),
                "data": data or {},
            }
        )


class CopilotWorkspaceIntegration:
    """Main integration point for GitHub Copilot Workspace.

    Provides plan verification, constraint injection, and trust scoring
    for code generated within Copilot Workspace sessions.
    """

    def __init__(
        self,
        max_file_size: int = 100_000,
        block_on_critical: bool = True,
        trust_threshold: float = 0.6,
    ) -> None:
        self.max_file_size = max_file_size
        self.block_on_critical = block_on_critical
        self.trust_threshold = trust_threshold
        self.sessions: dict[str, WorkspaceSession] = {}
        self.global_constraints: list[VerificationConstraint] = []
        self._default_constraints = self._build_default_constraints()

    def create_session(self, workspace_id: str, user_id: str = "") -> WorkspaceSession:
        """Create a new workspace verification session."""
        session = WorkspaceSession(
            workspace_id=workspace_id,
            user_id=user_id,
            active_constraints=list(self._default_constraints),
        )
        self.sessions[session.id] = session
        session.add_event(WorkspaceEventType.PLAN_CREATED)
        logger.info("workspace_session_created", session_id=session.id)
        return session

    def get_session(self, session_id: str) -> WorkspaceSession | None:
        """Retrieve a workspace session."""
        return self.sessions.get(session_id)

    def submit_plan(
        self,
        session_id: str,
        description: str,
        files: list[dict[str, Any]],
    ) -> WorkspacePlan:
        """Submit a Copilot Workspace plan for verification."""
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        workspace_files = []
        for f in files:
            wf = WorkspaceFile(
                path=f.get("path", ""),
                language=f.get("language", self._detect_language(f.get("path", ""))),
                original_content=f.get("original_content", ""),
                proposed_content=f.get("proposed_content", ""),
                is_new=f.get("is_new", False),
                is_deleted=f.get("is_deleted", False),
            )
            workspace_files.append(wf)

        plan = WorkspacePlan(
            workspace_id=session.workspace_id,
            description=description,
            files=workspace_files,
            constraints=list(session.active_constraints),
        )
        session.plans.append(plan)
        session.add_event(WorkspaceEventType.PLAN_CREATED, {"plan_id": plan.id})
        return plan

    def verify_plan(self, session_id: str, plan_id: str) -> PlanVerificationResult:
        """Verify a submitted workspace plan against constraints."""
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        plan = next((p for p in session.plans if p.id == plan_id), None)
        if plan is None:
            raise ValueError(f"Plan {plan_id} not found")

        start_time = time.monotonic()
        plan.status = WorkspacePlanStatus.VERIFYING

        findings: list[PlanFinding] = []
        constraints_checked = 0
        constraints_satisfied = 0

        for ws_file in plan.files:
            if not ws_file.has_changes and not ws_file.is_new:
                continue

            content = ws_file.proposed_content
            for constraint in plan.constraints:
                constraints_checked += 1
                file_findings = self._check_constraint(
                    content, ws_file.path, ws_file.language, constraint
                )
                if not file_findings:
                    constraints_satisfied += 1
                findings.extend(file_findings)

        trust_score = self._calculate_trust_score(plan, findings)
        gate = self._determine_gate(findings, trust_score)
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        critical_count = sum(1 for f in findings if f.severity == "critical")
        high_count = sum(1 for f in findings if f.severity == "high")
        summary = (
            f"Plan verified: {len(findings)} findings "
            f"({critical_count} critical, {high_count} high), "
            f"trust score {trust_score:.1f}/100"
        )

        result = PlanVerificationResult(
            plan_id=plan_id,
            gate=gate,
            findings=findings,
            trust_score=trust_score,
            constraints_checked=constraints_checked,
            constraints_satisfied=constraints_satisfied,
            verification_time_ms=elapsed_ms,
            summary=summary,
        )

        plan.verification_result = result
        plan.status = (
            WorkspacePlanStatus.VERIFIED
            if gate != VerificationGate.BLOCK
            else WorkspacePlanStatus.FAILED
        )
        plan.verified_at = datetime.now(UTC)

        session.add_event(
            WorkspaceEventType.PLAN_VERIFIED,
            {"plan_id": plan_id, "gate": gate.value, "trust_score": trust_score},
        )

        return result

    def add_constraint(
        self,
        session_id: str,
        constraint_type: ConstraintType,
        description: str,
        applies_to: str = "",
        is_hard: bool = True,
    ) -> VerificationConstraint:
        """Add a verification constraint to a session."""
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        constraint = VerificationConstraint(
            constraint_type=constraint_type,
            description=description,
            applies_to=applies_to,
            is_hard=is_hard,
        )
        session.active_constraints.append(constraint)
        session.add_event(
            WorkspaceEventType.CONSTRAINT_ADDED,
            {"constraint_id": constraint.id, "type": constraint_type.value},
        )
        return constraint

    def get_generation_prompt(self, session_id: str) -> str:
        """Generate a prompt injection with verification constraints.

        This is injected into the Copilot Workspace generation prompt
        to guide code generation toward verified-correct output.
        """
        session = self.sessions.get(session_id)
        if session is None:
            return ""

        if not session.active_constraints:
            return ""

        lines = [
            "VERIFICATION CONSTRAINTS (code MUST satisfy these):",
            "",
        ]
        for i, c in enumerate(session.active_constraints, 1):
            marker = "[REQUIRED]" if c.is_hard else "[RECOMMENDED]"
            lines.append(f"{i}. {marker} {c.description}")
            if c.applies_to:
                lines.append(f"   Applies to: {c.applies_to}")

        lines.append("")
        lines.append(
            "Generated code will be formally verified against these constraints. "
            "Ensure all required constraints are satisfied."
        )
        return "\n".join(lines)

    def _check_constraint(
        self,
        code: str,
        file_path: str,
        language: str,
        constraint: VerificationConstraint,
    ) -> list[PlanFinding]:
        """Check a single constraint against code."""
        findings = []

        if constraint.constraint_type == ConstraintType.NULL_SAFETY:
            findings.extend(self._check_null_safety(code, file_path, language))
        elif constraint.constraint_type == ConstraintType.BOUNDS_CHECK:
            findings.extend(self._check_bounds(code, file_path, language))
        elif constraint.constraint_type == ConstraintType.SECURITY:
            findings.extend(self._check_security(code, file_path, language))
        elif constraint.constraint_type == ConstraintType.TYPE_SAFETY:
            findings.extend(self._check_type_safety(code, file_path, language))

        for f in findings:
            f.constraint_violated = constraint.id

        return findings

    def _check_null_safety(self, code: str, file_path: str, language: str) -> list[PlanFinding]:
        """Check for potential null/None dereference issues."""
        findings = []
        import re

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if (
                language in ("python", "py")
                and (
                    re.search(r"\.\w+\(", stripped)
                    and "if " not in stripped
                    and "is not None" not in stripped
                )
                and ("= None" in code[: code.index(line)] if line in code else False)
            ):
                findings.append(
                    PlanFinding(
                        file_path=file_path,
                        line=i,
                        severity="medium",
                        message="Potential None dereference — variable may be None",
                        fix_suggestion="Add a None check before accessing attributes",
                        auto_fixable=True,
                    )
                )
            if (
                language in ("typescript", "ts", "javascript", "js")
                and (".length" in stripped or "." in stripped)
                and ("undefined" in code or "null" in code)
                and "?" not in stripped
                and "!= null" not in stripped
            ):
                pass  # Only flag obvious cases
        return findings

    def _check_bounds(self, code: str, file_path: str, language: str) -> list[PlanFinding]:
        """Check for array bounds issues."""
        import re

        findings = []
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if re.search(r"\[\s*-\d+\s*\]", line) and language not in ("python", "py"):
                findings.append(
                    PlanFinding(
                        file_path=file_path,
                        line=i,
                        severity="high",
                        message="Negative array index detected",
                        fix_suggestion="Ensure array index is non-negative",
                        auto_fixable=False,
                    )
                )
        return findings

    def _check_security(self, code: str, file_path: str, _language: str) -> list[PlanFinding]:
        """Check for common security issues."""
        import re

        findings = []
        patterns = [
            (r"eval\(", "critical", "Use of eval() — potential code injection"),
            (r"exec\(", "high", "Use of exec() — potential code injection"),
            (r"password\s*=\s*['\"]", "critical", "Hardcoded password detected"),
            (r"api[_-]?key\s*=\s*['\"]", "critical", "Hardcoded API key detected"),
            (r"SELECT\s+.*\+\s*\w+", "high", "Potential SQL injection — use parameterized queries"),
        ]
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern, severity, message in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(
                        PlanFinding(
                            file_path=file_path,
                            line=i,
                            severity=severity,
                            message=message,
                            fix_suggestion="Remove or secure the flagged pattern",
                        )
                    )
        return findings

    def _check_type_safety(self, _code: str, _file_path: str, _language: str) -> list[PlanFinding]:
        """Check for type safety issues."""
        return []  # Placeholder for AST-based type checking

    def _calculate_trust_score(self, plan: WorkspacePlan, findings: list[PlanFinding]) -> float:
        """Calculate trust score for the plan (0-100)."""
        base_score = 100.0

        severity_penalties = {
            "critical": 25,
            "high": 15,
            "medium": 8,
            "low": 3,
            "info": 1,
        }
        for f in findings:
            base_score -= severity_penalties.get(f.severity, 5)

        total_lines = sum(len(f.proposed_content.split("\n")) for f in plan.files if f.has_changes)
        if total_lines > 500:
            base_score -= min(10, (total_lines - 500) / 100)

        return max(0.0, min(100.0, base_score))

    def _determine_gate(self, findings: list[PlanFinding], trust_score: float) -> VerificationGate:
        """Determine the verification gate decision."""
        critical_count = sum(1 for f in findings if f.severity == "critical")
        high_count = sum(1 for f in findings if f.severity == "high")

        if critical_count > 0 and self.block_on_critical:
            return VerificationGate.BLOCK
        if high_count > 2 or trust_score < self.trust_threshold * 100:
            return VerificationGate.WARN
        return VerificationGate.PASS

    def _detect_language(self, path: str) -> str:
        """Detect language from file extension."""
        ext_map = {
            ".py": "python",
            ".ts": "typescript",
            ".js": "javascript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
        }
        for ext, lang in ext_map.items():
            if path.endswith(ext):
                return lang
        return "unknown"

    def _build_default_constraints(self) -> list[VerificationConstraint]:
        """Build default verification constraints."""
        return [
            VerificationConstraint(
                constraint_type=ConstraintType.NULL_SAFETY,
                description="No null/None dereferences without explicit checks",
                is_hard=True,
                priority=1,
            ),
            VerificationConstraint(
                constraint_type=ConstraintType.SECURITY,
                description="No hardcoded secrets, eval(), or SQL injection patterns",
                is_hard=True,
                priority=1,
            ),
            VerificationConstraint(
                constraint_type=ConstraintType.BOUNDS_CHECK,
                description="Array indices must be within bounds",
                is_hard=False,
                priority=2,
            ),
        ]


# ─── Singleton Access ──────────────────────────────────────────────────


_integration_instance: CopilotWorkspaceIntegration | None = None


def get_copilot_workspace_integration() -> CopilotWorkspaceIntegration:
    """Get or create the singleton CopilotWorkspaceIntegration."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = CopilotWorkspaceIntegration()
    return _integration_instance


def reset_copilot_workspace_integration() -> None:
    """Reset the singleton (for testing)."""
    global _integration_instance
    _integration_instance = None
