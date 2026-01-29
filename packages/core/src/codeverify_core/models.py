"""Core data models for CodeVerify."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Timestamp mixins for consistent datetime handling
# ============================================================================


class TimestampMixin(BaseModel):
    """Mixin providing created_at and updated_at timestamps for Pydantic models."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        """Update the updated_at timestamp to now."""
        self.updated_at = datetime.utcnow()


@dataclass
class DataclassTimestampMixin:
    """Mixin providing timestamps for dataclasses. Use as first base class."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        """Update the updated_at timestamp to now."""
        self.updated_at = datetime.utcnow()


def parse_iso_datetime(value: str | None, default: datetime | None = None) -> datetime | None:
    """Parse ISO datetime string, handling Z suffix.
    
    Args:
        value: ISO datetime string (may end in Z)
        default: Default value if parsing fails or value is None
        
    Returns:
        Parsed datetime or default
    """
    if not value:
        return default
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return default


# ============================================================================
# Enums
# ============================================================================


class AnalysisStatus(str, Enum):
    """Status of an analysis."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Import FindingSeverity from severity module for backwards compatibility
from codeverify_core.severity import (
    FindingSeverity,
    SEVERITY_ORDER,
    SEVERITY_EMOJI,
    SEVERITY_LABELS,
    parse_severity,
    compare_severity,
    is_blocking_severity,
    is_above_threshold,
    get_severity_emoji,
    get_severity_label,
    sort_by_severity,
)


class FindingCategory(str, Enum):
    """Category of a finding."""

    SECURITY = "security"
    LOGIC_ERROR = "logic_error"
    NULL_SAFETY = "null_safety"
    TYPE_ERROR = "type_error"
    OVERFLOW = "overflow"
    BOUNDS = "bounds"
    RESOURCE_LEAK = "resource_leak"
    CONCURRENCY = "concurrency"
    PERFORMANCE = "performance"
    STYLE = "style"
    DOCUMENTATION = "documentation"


class VerificationType(str, Enum):
    """Type of verification that produced the finding."""

    FORMAL = "formal"  # SMT solver proof
    AI = "ai"  # LLM-based analysis
    PATTERN = "pattern"  # Pattern matching
    HYBRID = "hybrid"  # Combination of methods


class RiskLevel(str, Enum):
    """Risk level for trust scoring."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VCSProvider(str, Enum):
    """Version control system provider."""

    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"


class CodeLocation(BaseModel):
    """Location of code in a file."""

    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    column_start: int | None = None
    column_end: int | None = None


class TrustScoreFactors(BaseModel):
    """Factors contributing to a trust score."""

    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    pattern_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    historical_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    code_quality_signals: float = Field(default=0.0, ge=0.0, le=1.0)
    ai_detection_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TrustScore(BaseModel):
    """Trust score for a code block or file."""

    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    factors: TrustScoreFactors
    recommendations: list[str] = Field(default_factory=list)
    is_ai_generated: bool = False
    code_hash: str | None = None
    calculated_at: datetime | None = None


class Finding(BaseModel):
    """A finding from code analysis."""

    id: UUID | None = None
    category: FindingCategory
    severity: FindingSeverity
    title: str
    description: str
    location: CodeLocation
    code_snippet: str | None = None
    fix_suggestion: str | None = None
    fix_diff: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    verification_type: VerificationType
    verification_proof: str | None = None
    trust_score: TrustScore | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    dismissed: bool = False
    dismissed_reason: str | None = None


class AnalysisStage(BaseModel):
    """A stage in the analysis pipeline."""

    name: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class Analysis(BaseModel):
    """A complete code analysis result."""

    id: UUID | None = None
    repo_full_name: str
    pr_number: int
    pr_title: str | None = None
    head_sha: str
    base_sha: str | None = None
    status: AnalysisStatus = AnalysisStatus.PENDING
    findings: list[Finding] = Field(default_factory=list)
    stages: list[AnalysisStage] = Field(default_factory=list)
    trust_scores: dict[str, TrustScore] = Field(default_factory=dict)  # file_path -> score
    overall_trust_score: TrustScore | None = None
    vcs_provider: VCSProvider = VCSProvider.GITHUB
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Calculate analysis duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def critical_count(self) -> int:
        """Count critical severity findings."""
        return sum(1 for f in self.findings if f.severity == FindingSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count high severity findings."""
        return sum(1 for f in self.findings if f.severity == FindingSeverity.HIGH)

    def passed(self, max_critical: int = 0, max_high: int = 0) -> bool:
        """Check if analysis passed based on thresholds."""
        return self.critical_count <= max_critical and self.high_count <= max_high


class VerificationCondition(BaseModel):
    """A verification condition to be checked by SMT solver."""

    id: str
    description: str
    formula: str  # SMT-LIB format
    source_location: CodeLocation
    check_type: str  # null_check, bounds_check, overflow_check, etc.


class VerificationResult(BaseModel):
    """Result of verifying a condition."""

    condition_id: str
    satisfied: bool | None  # None means timeout/unknown
    counterexample: dict[str, Any] | None = None
    proof_time_ms: float
    error: str | None = None


class DiffSummary(BaseModel):
    """AI-generated summary of a code diff."""

    summary: str
    behavioral_changes: list[str] = Field(default_factory=list)
    risk_assessment: str | None = None
    suggested_pr_description: str | None = None
    changelog_entry: str | None = None
    affected_components: list[str] = Field(default_factory=list)
    breaking_changes: list[str] = Field(default_factory=list)


class WebhookEvent(BaseModel):
    """Generic webhook event for multiple VCS providers."""

    provider: VCSProvider
    event_type: str
    repo_full_name: str
    repo_id: int | str
    sender: str
    payload: dict[str, Any]
    signature: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Result type pattern for consistent error handling
# ============================================================================

from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass
class Result(Generic[T, E]):
    """A Result type for explicit success/failure handling.
    
    Inspired by Rust's Result type. Use this instead of exceptions
    for expected error conditions that callers should handle.
    
    Usage:
        def parse_config(path: str) -> Result[Config, str]:
            try:
                config = load(path)
                return Result.ok(config)
            except FileNotFoundError:
                return Result.err("Config file not found")
                
        result = parse_config("config.yaml")
        if result.is_ok:
            config = result.unwrap()
        else:
            print(f"Error: {result.error}")
    """
    
    _value: T | None = None
    _error: E | None = None
    _is_ok: bool = True
    
    @classmethod
    def ok(cls, value: T) -> "Result[T, E]":
        """Create a successful result."""
        return cls(_value=value, _error=None, _is_ok=True)
    
    @classmethod
    def err(cls, error: E) -> "Result[T, E]":
        """Create a failed result."""
        return cls(_value=None, _error=error, _is_ok=False)
    
    @property
    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self._is_ok
    
    @property
    def is_err(self) -> bool:
        """Check if result is an error."""
        return not self._is_ok
    
    @property
    def value(self) -> T | None:
        """Get the value if successful, None otherwise."""
        return self._value
    
    @property
    def error(self) -> E | None:
        """Get the error if failed, None otherwise."""
        return self._error
    
    def unwrap(self) -> T:
        """Get the value, raising ValueError if result is an error."""
        if not self._is_ok:
            raise ValueError(f"Called unwrap on error result: {self._error}")
        return self._value  # type: ignore
    
    def unwrap_or(self, default: T) -> T:
        """Get the value or a default if result is an error."""
        return self._value if self._is_ok else default  # type: ignore
    
    def unwrap_err(self) -> E:
        """Get the error, raising ValueError if result is successful."""
        if self._is_ok:
            raise ValueError("Called unwrap_err on successful result")
        return self._error  # type: ignore
    
    def map(self, func: "callable[[T], T]") -> "Result[T, E]":
        """Apply a function to the value if successful."""
        if self._is_ok:
            return Result.ok(func(self._value))  # type: ignore
        return self  # type: ignore
    
    def map_err(self, func: "callable[[E], E]") -> "Result[T, E]":
        """Apply a function to the error if failed."""
        if not self._is_ok:
            return Result.err(func(self._error))  # type: ignore
        return self  # type: ignore


# Common Result type aliases
StrResult = Result[str, str]
BoolResult = Result[bool, str]


class OperationResult(BaseModel):
    """Pydantic model for serializable operation results.
    
    Use this for API responses where you need JSON serialization.
    For internal code, prefer the Result dataclass.
    """
    
    success: bool
    data: Any | None = None
    error: str | None = None
    error_code: str | None = None
    
    @classmethod
    def ok(cls, data: Any = None) -> "OperationResult":
        """Create a successful result."""
        return cls(success=True, data=data)
    
    @classmethod
    def err(cls, error: str, error_code: str | None = None) -> "OperationResult":
        """Create a failed result."""
        return cls(success=False, error=error, error_code=error_code)
