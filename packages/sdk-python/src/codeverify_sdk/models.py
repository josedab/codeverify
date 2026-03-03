"""Pydantic models for CodeVerify SDK responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A single verification finding."""

    id: str
    type: str
    severity: str
    title: str
    description: str
    file_path: str = "<inline>"
    line_start: int | None = None
    line_end: int | None = None
    fix_suggestion: str | None = None
    confidence: float = 1.0


class VerificationResult(BaseModel):
    """Result of a full verification run."""

    passed: bool
    findings: list[Finding] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)
    language: str = "python"
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity in ("critical", "high"))

    @property
    def has_issues(self) -> bool:
        return len(self.findings) > 0


class SafetyResult(BaseModel):
    """Result of a safety check."""

    safe: bool
    risk_level: str = Field(description="low, medium, high, critical")
    issues: list[Finding] = Field(default_factory=list)
    categories_checked: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


class ProofResult(BaseModel):
    """Result of formal proof generation."""

    status: str = Field(description="proved, disproved, unknown, timeout")
    property_checked: str = ""
    counterexamples: list[dict] = Field(default_factory=list)
    proof_tree: dict | None = None
    solver_time_ms: float = 0.0
    explanation: str | None = None


class CheckResult(BaseModel):
    """Result of a single property check (used by pytest plugin)."""

    property_name: str
    holds: bool
    counterexample: dict | None = None
    message: str = ""
