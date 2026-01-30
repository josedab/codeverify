"""Gradual Verification Ramp - Warnings-only onboarding mode.

Provides a gentle introduction to CodeVerify for new repositories/teams:
- Warnings-only mode (no blocking) for configurable period
- Progressive enforcement based on severity
- Customizable ramp schedule
- Baseline establishment and metric tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from codeverify_core.models import Analysis, Finding, FindingSeverity

logger = structlog.get_logger()


class EnforcementLevel(str, Enum):
    """Level of enforcement during ramp."""
    SHADOW = "shadow"  # Silent observation only
    WARN = "warn"  # Warnings, never block
    SOFT_BLOCK = "soft_block"  # Block critical only
    MEDIUM_BLOCK = "medium_block"  # Block critical and error
    FULL = "full"  # Block all configured severities


class RampPhase(str, Enum):
    """Phase of the verification ramp."""
    BASELINE = "baseline"  # Establishing baseline
    OBSERVATION = "observation"  # Observing with warnings
    TRANSITION = "transition"  # Gradually increasing enforcement
    ENFORCING = "enforcing"  # Full enforcement


@dataclass
class RampSchedule:
    """Schedule for verification ramp-up."""
    baseline_days: int = 7  # Days to establish baseline
    observation_days: int = 14  # Days of warning-only mode
    transition_days: int = 14  # Days of gradual enforcement
    
    # Severity enforcement schedule (day when enforcement starts)
    critical_enforcement_day: int = 21
    error_enforcement_day: int = 28
    warning_enforcement_day: int = 35
    info_enforcement_day: int | None = None  # Never block on info by default


@dataclass
class BaselineMetrics:
    """Baseline metrics for a repository."""
    repository: str
    start_date: datetime
    end_date: datetime | None
    total_prs: int = 0
    total_findings: int = 0
    findings_by_severity: dict[str, int] = field(default_factory=dict)
    findings_by_category: dict[str, int] = field(default_factory=dict)
    avg_findings_per_pr: float = 0.0
    most_common_categories: list[str] = field(default_factory=list)


@dataclass
class RampState:
    """Current state of the verification ramp."""
    repository: str
    enabled: bool
    start_date: datetime
    current_phase: RampPhase
    enforcement_level: EnforcementLevel
    days_elapsed: int
    schedule: RampSchedule
    baseline: BaselineMetrics | None = None
    
    # Progress tracking
    findings_while_ramping: int = 0
    warnings_issued: int = 0
    would_have_blocked: int = 0
    
    # Customization
    paused: bool = False
    skip_teams: list[str] = field(default_factory=list)
    early_enforcement_repos: list[str] = field(default_factory=list)


@dataclass
class EnforcementDecision:
    """Decision about whether to block a PR."""
    should_block: bool
    enforcement_level: EnforcementLevel
    phase: RampPhase
    reason: str
    blocking_findings: list[Finding] = field(default_factory=list)
    warning_findings: list[Finding] = field(default_factory=list)
    days_until_enforcement: int | None = None


@dataclass
class RampProgress:
    """Progress report for the ramp."""
    repository: str
    current_phase: RampPhase
    enforcement_level: EnforcementLevel
    days_elapsed: int
    days_remaining: int
    percent_complete: float
    metrics: dict[str, Any]
    next_milestone: str
    recommendations: list[str]


class BaselineCollector:
    """Collects baseline metrics for a repository."""

    def __init__(self) -> None:
        """Initialize collector."""
        self._findings: list[tuple[datetime, Finding]] = []
        self._pr_count = 0

    def record_pr(self, findings: list[Finding]) -> None:
        """Record a PR with its findings."""
        self._pr_count += 1
        now = datetime.utcnow()
        for finding in findings:
            self._findings.append((now, finding))

    def compute_baseline(self, repository: str, start_date: datetime) -> BaselineMetrics:
        """Compute baseline metrics."""
        findings_by_severity: dict[str, int] = {}
        findings_by_category: dict[str, int] = {}
        
        for _, finding in self._findings:
            sev = finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity)
            findings_by_severity[sev] = findings_by_severity.get(sev, 0) + 1
            
            cat = finding.category.value if hasattr(finding.category, 'value') else str(finding.category)
            findings_by_category[cat] = findings_by_category.get(cat, 0) + 1
        
        total = len(self._findings)
        avg = total / self._pr_count if self._pr_count > 0 else 0
        
        # Top 3 categories
        sorted_cats = sorted(
            findings_by_category.items(),
            key=lambda x: x[1],
            reverse=True
        )
        most_common = [cat for cat, _ in sorted_cats[:3]]
        
        return BaselineMetrics(
            repository=repository,
            start_date=start_date,
            end_date=datetime.utcnow(),
            total_prs=self._pr_count,
            total_findings=total,
            findings_by_severity=findings_by_severity,
            findings_by_category=findings_by_category,
            avg_findings_per_pr=avg,
            most_common_categories=most_common,
        )


class GradualVerificationRamp:
    """
    Manages gradual verification ramp-up for repositories.
    
    Provides warnings-only mode during onboarding, then progressively
    increases enforcement over time.
    """

    def __init__(self, default_schedule: RampSchedule | None = None) -> None:
        """Initialize the ramp manager."""
        self.default_schedule = default_schedule or RampSchedule()
        self._ramp_states: dict[str, RampState] = {}
        self._baseline_collectors: dict[str, BaselineCollector] = {}

    def start_ramp(
        self,
        repository: str,
        schedule: RampSchedule | None = None,
        start_date: datetime | None = None,
    ) -> RampState:
        """Start verification ramp for a repository."""
        schedule = schedule or self.default_schedule
        start = start_date or datetime.utcnow()
        
        state = RampState(
            repository=repository,
            enabled=True,
            start_date=start,
            current_phase=RampPhase.BASELINE,
            enforcement_level=EnforcementLevel.SHADOW,
            days_elapsed=0,
            schedule=schedule,
        )
        
        self._ramp_states[repository] = state
        self._baseline_collectors[repository] = BaselineCollector()
        
        logger.info(
            "Verification ramp started",
            repository=repository,
            schedule_baseline=schedule.baseline_days,
            schedule_observation=schedule.observation_days,
            schedule_transition=schedule.transition_days,
        )
        
        return state

    def get_state(self, repository: str) -> RampState | None:
        """Get current ramp state for a repository."""
        state = self._ramp_states.get(repository)
        if state:
            self._update_state(state)
        return state

    def evaluate_enforcement(
        self,
        repository: str,
        findings: list[Finding],
    ) -> EnforcementDecision:
        """
        Evaluate whether to block a PR based on ramp status.
        
        Args:
            repository: Repository identifier
            findings: Findings from verification
            
        Returns:
            EnforcementDecision with block decision and details
        """
        state = self.get_state(repository)
        
        # No ramp configured - full enforcement
        if not state or not state.enabled:
            return self._full_enforcement(findings)
        
        # Paused - warning only
        if state.paused:
            return EnforcementDecision(
                should_block=False,
                enforcement_level=EnforcementLevel.WARN,
                phase=state.current_phase,
                reason="Ramp is paused",
                warning_findings=findings,
            )
        
        # Update baseline if still collecting
        if state.current_phase == RampPhase.BASELINE:
            collector = self._baseline_collectors.get(repository)
            if collector:
                collector.record_pr(findings)
        
        # Determine blocking based on phase and severity
        return self._evaluate_by_phase(state, findings)

    def _update_state(self, state: RampState) -> None:
        """Update state based on elapsed time."""
        now = datetime.utcnow()
        days_elapsed = (now - state.start_date).days
        state.days_elapsed = days_elapsed
        
        schedule = state.schedule
        
        # Determine phase
        if days_elapsed < schedule.baseline_days:
            state.current_phase = RampPhase.BASELINE
            state.enforcement_level = EnforcementLevel.SHADOW
            
        elif days_elapsed < schedule.baseline_days + schedule.observation_days:
            state.current_phase = RampPhase.OBSERVATION
            state.enforcement_level = EnforcementLevel.WARN
            
            # Compute baseline if transitioning from baseline
            if state.baseline is None:
                collector = self._baseline_collectors.get(state.repository)
                if collector:
                    state.baseline = collector.compute_baseline(
                        state.repository,
                        state.start_date,
                    )
                    
        elif days_elapsed < (
            schedule.baseline_days +
            schedule.observation_days +
            schedule.transition_days
        ):
            state.current_phase = RampPhase.TRANSITION
            
            # Progressive enforcement
            if days_elapsed >= schedule.critical_enforcement_day:
                if days_elapsed >= schedule.error_enforcement_day:
                    if days_elapsed >= schedule.warning_enforcement_day:
                        state.enforcement_level = EnforcementLevel.FULL
                    else:
                        state.enforcement_level = EnforcementLevel.MEDIUM_BLOCK
                else:
                    state.enforcement_level = EnforcementLevel.SOFT_BLOCK
            else:
                state.enforcement_level = EnforcementLevel.WARN
        else:
            state.current_phase = RampPhase.ENFORCING
            state.enforcement_level = EnforcementLevel.FULL

    def _evaluate_by_phase(
        self,
        state: RampState,
        findings: list[Finding],
    ) -> EnforcementDecision:
        """Evaluate enforcement based on current phase."""
        blocking_findings = []
        warning_findings = []
        
        for finding in findings:
            should_block = self._should_block_finding(
                finding, state.enforcement_level, state.schedule
            )
            
            if should_block:
                blocking_findings.append(finding)
            else:
                warning_findings.append(finding)
        
        # Track statistics
        state.findings_while_ramping += len(findings)
        state.warnings_issued += len(warning_findings)
        if blocking_findings:
            state.would_have_blocked += 1
        
        should_block = len(blocking_findings) > 0
        
        # Calculate days until next enforcement milestone
        days_until = self._days_until_enforcement(state, findings)
        
        reason = self._generate_reason(state, blocking_findings, warning_findings)
        
        return EnforcementDecision(
            should_block=should_block,
            enforcement_level=state.enforcement_level,
            phase=state.current_phase,
            reason=reason,
            blocking_findings=blocking_findings,
            warning_findings=warning_findings,
            days_until_enforcement=days_until,
        )

    def _should_block_finding(
        self,
        finding: Finding,
        level: EnforcementLevel,
        schedule: RampSchedule,
    ) -> bool:
        """Determine if a finding should block based on enforcement level."""
        if level == EnforcementLevel.SHADOW:
            return False
        
        if level == EnforcementLevel.WARN:
            return False
        
        # Get severity value
        sev = finding.severity
        if hasattr(sev, 'value'):
            sev_value = sev.value.lower()
        else:
            sev_value = str(sev).lower()
        
        if level == EnforcementLevel.SOFT_BLOCK:
            return sev_value == "critical"
        
        if level == EnforcementLevel.MEDIUM_BLOCK:
            return sev_value in ("critical", "error")
        
        if level == EnforcementLevel.FULL:
            # Block all except info (unless configured)
            if sev_value == "info" and schedule.info_enforcement_day is None:
                return False
            return sev_value in ("critical", "error", "warning")
        
        return False

    def _days_until_enforcement(
        self,
        state: RampState,
        findings: list[Finding],
    ) -> int | None:
        """Calculate days until findings would block."""
        if state.enforcement_level == EnforcementLevel.FULL:
            return None  # Already enforcing
        
        # Find the next milestone that would affect current findings
        schedule = state.schedule
        days = state.days_elapsed
        
        severities = set()
        for f in findings:
            sev = f.severity
            if hasattr(sev, 'value'):
                severities.add(sev.value.lower())
            else:
                severities.add(str(sev).lower())
        
        milestones = []
        if "critical" in severities and days < schedule.critical_enforcement_day:
            milestones.append(schedule.critical_enforcement_day - days)
        if "error" in severities and days < schedule.error_enforcement_day:
            milestones.append(schedule.error_enforcement_day - days)
        if "warning" in severities and days < schedule.warning_enforcement_day:
            milestones.append(schedule.warning_enforcement_day - days)
        
        return min(milestones) if milestones else None

    def _generate_reason(
        self,
        state: RampState,
        blocking: list[Finding],
        warnings: list[Finding],
    ) -> str:
        """Generate explanation for the decision."""
        if state.current_phase == RampPhase.BASELINE:
            return f"Baseline collection (day {state.days_elapsed}/{state.schedule.baseline_days})"
        
        if state.current_phase == RampPhase.OBSERVATION:
            return f"Observation period - {len(warnings)} warnings issued"
        
        if state.current_phase == RampPhase.TRANSITION:
            if blocking:
                return f"Transition phase - blocking {len(blocking)} critical/error findings"
            return f"Transition phase - {len(warnings)} warnings (not blocking yet)"
        
        if blocking:
            return f"Full enforcement - blocking {len(blocking)} findings"
        return "Full enforcement - no blocking findings"

    def _full_enforcement(self, findings: list[Finding]) -> EnforcementDecision:
        """Apply full enforcement (no ramp)."""
        blocking = []
        warnings = []
        
        for finding in findings:
            sev = finding.severity
            if hasattr(sev, 'value'):
                sev_value = sev.value.lower()
            else:
                sev_value = str(sev).lower()
            
            if sev_value in ("critical", "error", "warning"):
                blocking.append(finding)
            else:
                warnings.append(finding)
        
        return EnforcementDecision(
            should_block=len(blocking) > 0,
            enforcement_level=EnforcementLevel.FULL,
            phase=RampPhase.ENFORCING,
            reason="Full enforcement active (no ramp configured)",
            blocking_findings=blocking,
            warning_findings=warnings,
        )

    def get_progress_report(self, repository: str) -> RampProgress | None:
        """Get progress report for a repository's ramp."""
        state = self.get_state(repository)
        if not state:
            return None
        
        schedule = state.schedule
        total_days = (
            schedule.baseline_days +
            schedule.observation_days +
            schedule.transition_days
        )
        
        days_remaining = max(0, total_days - state.days_elapsed)
        percent = min(100, (state.days_elapsed / total_days) * 100) if total_days > 0 else 100
        
        # Determine next milestone
        if state.current_phase == RampPhase.BASELINE:
            next_milestone = f"Observation starts in {schedule.baseline_days - state.days_elapsed} days"
        elif state.current_phase == RampPhase.OBSERVATION:
            days_to_transition = (
                schedule.baseline_days + schedule.observation_days - state.days_elapsed
            )
            next_milestone = f"Transition starts in {days_to_transition} days"
        elif state.current_phase == RampPhase.TRANSITION:
            next_milestone = f"Full enforcement in {days_remaining} days"
        else:
            next_milestone = "Full enforcement active"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(state)
        
        metrics = {
            "findings_during_ramp": state.findings_while_ramping,
            "warnings_issued": state.warnings_issued,
            "would_have_blocked": state.would_have_blocked,
        }
        
        if state.baseline:
            metrics["baseline_avg_per_pr"] = state.baseline.avg_findings_per_pr
            metrics["baseline_total"] = state.baseline.total_findings
            metrics["common_categories"] = state.baseline.most_common_categories
        
        return RampProgress(
            repository=repository,
            current_phase=state.current_phase,
            enforcement_level=state.enforcement_level,
            days_elapsed=state.days_elapsed,
            days_remaining=days_remaining,
            percent_complete=percent,
            metrics=metrics,
            next_milestone=next_milestone,
            recommendations=recommendations,
        )

    def _generate_recommendations(self, state: RampState) -> list[str]:
        """Generate recommendations based on ramp progress."""
        recommendations = []
        
        if state.baseline:
            avg = state.baseline.avg_findings_per_pr
            if avg > 5:
                recommendations.append(
                    "High finding rate detected. Consider team training before full enforcement."
                )
            
            if state.baseline.most_common_categories:
                top_cat = state.baseline.most_common_categories[0]
                recommendations.append(
                    f"Focus on {top_cat} issues - most common category in baseline."
                )
        
        if state.would_have_blocked > 0:
            block_rate = state.would_have_blocked / max(state.findings_while_ramping, 1) * 100
            if block_rate > 20:
                recommendations.append(
                    "Many PRs would have been blocked. Review findings before enforcement."
                )
        
        if state.current_phase == RampPhase.TRANSITION:
            recommendations.append(
                "Transition phase - monitor team feedback and adjust if needed."
            )
        
        return recommendations

    def pause_ramp(self, repository: str) -> bool:
        """Pause ramp for a repository."""
        state = self._ramp_states.get(repository)
        if state:
            state.paused = True
            logger.info("Ramp paused", repository=repository)
            return True
        return False

    def resume_ramp(self, repository: str) -> bool:
        """Resume paused ramp."""
        state = self._ramp_states.get(repository)
        if state:
            state.paused = False
            logger.info("Ramp resumed", repository=repository)
            return True
        return False

    def extend_ramp(self, repository: str, extra_days: int) -> bool:
        """Extend ramp by additional days."""
        state = self._ramp_states.get(repository)
        if state:
            # Add to transition days
            state.schedule.transition_days += extra_days
            state.schedule.warning_enforcement_day += extra_days
            
            # Recalculate state
            self._update_state(state)
            
            logger.info(
                "Ramp extended",
                repository=repository,
                extra_days=extra_days,
            )
            return True
        return False

    def end_ramp(self, repository: str) -> bool:
        """End ramp and move to full enforcement."""
        state = self._ramp_states.get(repository)
        if state:
            state.current_phase = RampPhase.ENFORCING
            state.enforcement_level = EnforcementLevel.FULL
            
            logger.info("Ramp ended - full enforcement", repository=repository)
            return True
        return False

    def get_all_ramps(self) -> list[RampState]:
        """Get all active ramps."""
        result = []
        for repo, state in self._ramp_states.items():
            self._update_state(state)
            result.append(state)
        return result

    def format_pr_comment(self, decision: EnforcementDecision) -> str:
        """Format a PR comment explaining the ramp status."""
        lines = [
            "## CodeVerify Onboarding Status",
            "",
        ]
        
        # Phase indicator
        phase_emoji = {
            RampPhase.BASELINE: "📊",
            RampPhase.OBSERVATION: "👀",
            RampPhase.TRANSITION: "🔄",
            RampPhase.ENFORCING: "✅",
        }
        
        emoji = phase_emoji.get(decision.phase, "ℹ️")
        lines.append(f"{emoji} **Phase:** {decision.phase.value.title()}")
        lines.append(f"**Enforcement Level:** {decision.enforcement_level.value.replace('_', ' ').title()}")
        lines.append("")
        
        # Warning findings
        if decision.warning_findings:
            lines.append(f"### ⚠️ Warnings ({len(decision.warning_findings)})")
            lines.append("*These findings will become blocking in future phases.*")
            lines.append("")
            for f in decision.warning_findings[:5]:  # Show first 5
                sev = f.severity.value if hasattr(f.severity, 'value') else str(f.severity)
                lines.append(f"- **{sev.upper()}**: {f.message}")
            
            if len(decision.warning_findings) > 5:
                lines.append(f"- ... and {len(decision.warning_findings) - 5} more")
            lines.append("")
        
        # Blocking findings
        if decision.blocking_findings:
            lines.append(f"### 🚫 Blocking ({len(decision.blocking_findings)})")
            for f in decision.blocking_findings:
                sev = f.severity.value if hasattr(f.severity, 'value') else str(f.severity)
                lines.append(f"- **{sev.upper()}**: {f.message}")
            lines.append("")
        
        # Days until enforcement
        if decision.days_until_enforcement:
            lines.append(f"📅 *Some warnings will become blocking in {decision.days_until_enforcement} days.*")
            lines.append("")
        
        lines.append("---")
        lines.append("*CodeVerify is gradually ramping up verification for this repository.*")
        
        return "\n".join(lines)
