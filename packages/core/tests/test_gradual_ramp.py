"""Tests for Gradual Verification Ramp module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from codeverify_core.gradual_ramp import (
    BaselineCollector,
    BaselineMetrics,
    EnforcementDecision,
    EnforcementLevel,
    GradualVerificationRamp,
    RampPhase,
    RampProgress,
    RampSchedule,
    RampState,
)
from codeverify_core.models import Finding, FindingCategory, FindingSeverity


def create_finding(
    severity: str = "warning",
    category: str = "correctness",
    message: str = "Test finding",
) -> Finding:
    """Create a test finding."""
    return Finding(
        id=f"finding-{id(message)}",
        message=message,
        category=FindingCategory(category),
        severity=FindingSeverity(severity),
        file_path="src/test.py",
        line_number=10,
    )


class TestEnforcementLevel:
    """Tests for EnforcementLevel enum."""

    def test_all_levels_exist(self):
        """All expected levels exist."""
        assert EnforcementLevel.SHADOW.value == "shadow"
        assert EnforcementLevel.WARN.value == "warn"
        assert EnforcementLevel.SOFT_BLOCK.value == "soft_block"
        assert EnforcementLevel.MEDIUM_BLOCK.value == "medium_block"
        assert EnforcementLevel.FULL.value == "full"


class TestRampPhase:
    """Tests for RampPhase enum."""

    def test_all_phases_exist(self):
        """All expected phases exist."""
        assert RampPhase.BASELINE.value == "baseline"
        assert RampPhase.OBSERVATION.value == "observation"
        assert RampPhase.TRANSITION.value == "transition"
        assert RampPhase.ENFORCING.value == "enforcing"


class TestRampSchedule:
    """Tests for RampSchedule dataclass."""

    def test_default_schedule(self):
        """Default schedule values."""
        schedule = RampSchedule()
        assert schedule.baseline_days == 7
        assert schedule.observation_days == 14
        assert schedule.transition_days == 14

    def test_custom_schedule(self):
        """Custom schedule values."""
        schedule = RampSchedule(
            baseline_days=3,
            observation_days=7,
            transition_days=7,
            critical_enforcement_day=10,
        )
        assert schedule.baseline_days == 3
        assert schedule.critical_enforcement_day == 10


class TestBaselineCollector:
    """Tests for BaselineCollector."""

    def test_empty_baseline(self):
        """Empty baseline has zero findings."""
        collector = BaselineCollector()
        baseline = collector.compute_baseline("repo", datetime.utcnow())
        
        assert baseline.total_findings == 0
        assert baseline.total_prs == 0

    def test_record_pr(self):
        """Records PR findings."""
        collector = BaselineCollector()
        
        findings = [
            create_finding(severity="error"),
            create_finding(severity="warning"),
        ]
        collector.record_pr(findings)
        
        baseline = collector.compute_baseline("repo", datetime.utcnow())
        assert baseline.total_findings == 2
        assert baseline.total_prs == 1

    def test_multiple_prs(self):
        """Records multiple PRs."""
        collector = BaselineCollector()
        
        for _ in range(5):
            collector.record_pr([create_finding()])
        
        baseline = collector.compute_baseline("repo", datetime.utcnow())
        assert baseline.total_prs == 5
        assert baseline.total_findings == 5

    def test_findings_by_severity(self):
        """Counts findings by severity."""
        collector = BaselineCollector()
        
        collector.record_pr([
            create_finding(severity="critical"),
            create_finding(severity="error"),
            create_finding(severity="error"),
            create_finding(severity="warning"),
        ])
        
        baseline = collector.compute_baseline("repo", datetime.utcnow())
        
        assert baseline.findings_by_severity.get("critical", 0) == 1
        assert baseline.findings_by_severity.get("error", 0) == 2
        assert baseline.findings_by_severity.get("warning", 0) == 1

    def test_avg_findings_per_pr(self):
        """Calculates average findings per PR."""
        collector = BaselineCollector()
        
        collector.record_pr([create_finding()] * 4)
        collector.record_pr([create_finding()] * 6)
        
        baseline = collector.compute_baseline("repo", datetime.utcnow())
        
        assert baseline.avg_findings_per_pr == 5.0


class TestGradualVerificationRamp:
    """Tests for GradualVerificationRamp."""

    def test_start_ramp(self):
        """Starts a new ramp."""
        ramp = GradualVerificationRamp()
        state = ramp.start_ramp("my-repo")
        
        assert state.repository == "my-repo"
        assert state.enabled is True
        assert state.current_phase == RampPhase.BASELINE

    def test_custom_schedule(self):
        """Uses custom schedule."""
        schedule = RampSchedule(baseline_days=3)
        ramp = GradualVerificationRamp(default_schedule=schedule)
        
        state = ramp.start_ramp("repo")
        assert state.schedule.baseline_days == 3

    def test_get_state_not_found(self):
        """Returns None for unknown repo."""
        ramp = GradualVerificationRamp()
        state = ramp.get_state("unknown-repo")
        assert state is None

    def test_phase_progression_baseline(self):
        """Baseline phase at start."""
        ramp = GradualVerificationRamp()
        state = ramp.start_ramp("repo")
        
        assert state.current_phase == RampPhase.BASELINE
        assert state.enforcement_level == EnforcementLevel.SHADOW

    def test_phase_progression_observation(self):
        """Progresses to observation phase."""
        ramp = GradualVerificationRamp()
        state = ramp.start_ramp(
            "repo",
            start_date=datetime.utcnow() - timedelta(days=8),
        )
        
        state = ramp.get_state("repo")
        assert state.current_phase == RampPhase.OBSERVATION
        assert state.enforcement_level == EnforcementLevel.WARN

    def test_phase_progression_transition(self):
        """Progresses to transition phase."""
        schedule = RampSchedule(
            baseline_days=7,
            observation_days=14,
            critical_enforcement_day=21,
        )
        ramp = GradualVerificationRamp(default_schedule=schedule)
        
        state = ramp.start_ramp(
            "repo",
            start_date=datetime.utcnow() - timedelta(days=22),
        )
        
        state = ramp.get_state("repo")
        assert state.current_phase == RampPhase.TRANSITION

    def test_phase_progression_enforcing(self):
        """Progresses to full enforcement."""
        schedule = RampSchedule(
            baseline_days=1,
            observation_days=1,
            transition_days=1,
        )
        ramp = GradualVerificationRamp(default_schedule=schedule)
        
        state = ramp.start_ramp(
            "repo",
            start_date=datetime.utcnow() - timedelta(days=10),
        )
        
        state = ramp.get_state("repo")
        assert state.current_phase == RampPhase.ENFORCING
        assert state.enforcement_level == EnforcementLevel.FULL


class TestEnforcementDecisions:
    """Tests for enforcement decisions."""

    def test_baseline_no_block(self):
        """Baseline phase never blocks."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo")
        
        findings = [
            create_finding(severity="critical"),
            create_finding(severity="error"),
        ]
        
        decision = ramp.evaluate_enforcement("repo", findings)
        
        assert decision.should_block is False
        assert decision.phase == RampPhase.BASELINE

    def test_observation_no_block(self):
        """Observation phase never blocks."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp(
            "repo",
            start_date=datetime.utcnow() - timedelta(days=10),
        )
        
        findings = [create_finding(severity="critical")]
        decision = ramp.evaluate_enforcement("repo", findings)
        
        assert decision.should_block is False
        assert len(decision.warning_findings) == 1

    def test_soft_block_critical_only(self):
        """Soft block only blocks critical."""
        schedule = RampSchedule(
            baseline_days=1,
            observation_days=1,
            transition_days=30,
            critical_enforcement_day=2,
            error_enforcement_day=10,
        )
        ramp = GradualVerificationRamp(default_schedule=schedule)
        
        ramp.start_ramp(
            "repo",
            start_date=datetime.utcnow() - timedelta(days=3),
        )
        
        # Critical should block
        critical_decision = ramp.evaluate_enforcement(
            "repo",
            [create_finding(severity="critical")],
        )
        assert critical_decision.should_block is True
        
        # Error should not block yet
        error_decision = ramp.evaluate_enforcement(
            "repo",
            [create_finding(severity="error")],
        )
        assert error_decision.should_block is False

    def test_full_enforcement_blocks(self):
        """Full enforcement blocks appropriate severities."""
        schedule = RampSchedule(
            baseline_days=1,
            observation_days=1,
            transition_days=1,
        )
        ramp = GradualVerificationRamp(default_schedule=schedule)
        
        ramp.start_ramp(
            "repo",
            start_date=datetime.utcnow() - timedelta(days=100),
        )
        
        findings = [
            create_finding(severity="error"),
            create_finding(severity="warning"),
        ]
        
        decision = ramp.evaluate_enforcement("repo", findings)
        
        assert decision.should_block is True
        assert decision.enforcement_level == EnforcementLevel.FULL

    def test_info_never_blocks(self):
        """Info severity never blocks by default."""
        schedule = RampSchedule(
            baseline_days=1,
            observation_days=1,
            transition_days=1,
            info_enforcement_day=None,  # Never enforce info
        )
        ramp = GradualVerificationRamp(default_schedule=schedule)
        
        ramp.start_ramp(
            "repo",
            start_date=datetime.utcnow() - timedelta(days=100),
        )
        
        decision = ramp.evaluate_enforcement(
            "repo",
            [create_finding(severity="info")],
        )
        
        assert decision.should_block is False

    def test_no_ramp_full_enforcement(self):
        """No ramp configured means full enforcement."""
        ramp = GradualVerificationRamp()
        
        # No start_ramp called
        decision = ramp.evaluate_enforcement(
            "unknown-repo",
            [create_finding(severity="error")],
        )
        
        assert decision.should_block is True
        assert decision.phase == RampPhase.ENFORCING


class TestRampManagement:
    """Tests for ramp management operations."""

    def test_pause_ramp(self):
        """Can pause a ramp."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo")
        
        assert ramp.pause_ramp("repo") is True
        
        state = ramp.get_state("repo")
        assert state.paused is True

    def test_paused_ramp_no_block(self):
        """Paused ramp never blocks."""
        schedule = RampSchedule(
            baseline_days=1,
            observation_days=1,
            transition_days=1,
        )
        ramp = GradualVerificationRamp(default_schedule=schedule)
        
        ramp.start_ramp(
            "repo",
            start_date=datetime.utcnow() - timedelta(days=100),
        )
        ramp.pause_ramp("repo")
        
        decision = ramp.evaluate_enforcement(
            "repo",
            [create_finding(severity="critical")],
        )
        
        assert decision.should_block is False
        assert "paused" in decision.reason.lower()

    def test_resume_ramp(self):
        """Can resume a paused ramp."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo")
        ramp.pause_ramp("repo")
        ramp.resume_ramp("repo")
        
        state = ramp.get_state("repo")
        assert state.paused is False

    def test_extend_ramp(self):
        """Can extend ramp duration."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo")
        
        original_transition = ramp.get_state("repo").schedule.transition_days
        
        ramp.extend_ramp("repo", 7)
        
        new_transition = ramp.get_state("repo").schedule.transition_days
        assert new_transition == original_transition + 7

    def test_end_ramp(self):
        """Can end ramp early."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo")
        
        ramp.end_ramp("repo")
        
        state = ramp.get_state("repo")
        assert state.current_phase == RampPhase.ENFORCING
        assert state.enforcement_level == EnforcementLevel.FULL

    def test_get_all_ramps(self):
        """Gets all active ramps."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo1")
        ramp.start_ramp("repo2")
        ramp.start_ramp("repo3")
        
        all_ramps = ramp.get_all_ramps()
        
        assert len(all_ramps) == 3
        repos = [r.repository for r in all_ramps]
        assert "repo1" in repos
        assert "repo2" in repos


class TestProgressReport:
    """Tests for progress reporting."""

    def test_progress_report(self):
        """Generates progress report."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp(
            "repo",
            start_date=datetime.utcnow() - timedelta(days=10),
        )
        
        progress = ramp.get_progress_report("repo")
        
        assert progress is not None
        assert progress.repository == "repo"
        assert progress.days_elapsed == 10
        assert 0 <= progress.percent_complete <= 100

    def test_progress_report_not_found(self):
        """Returns None for unknown repo."""
        ramp = GradualVerificationRamp()
        progress = ramp.get_progress_report("unknown")
        assert progress is None

    def test_progress_metrics(self):
        """Progress includes metrics."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo")
        
        # Record some findings
        ramp.evaluate_enforcement("repo", [create_finding()])
        ramp.evaluate_enforcement("repo", [create_finding()])
        
        progress = ramp.get_progress_report("repo")
        
        assert "findings_during_ramp" in progress.metrics
        assert progress.metrics["findings_during_ramp"] == 2

    def test_progress_recommendations(self):
        """Progress includes recommendations."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo")
        
        progress = ramp.get_progress_report("repo")
        
        assert isinstance(progress.recommendations, list)


class TestPRComment:
    """Tests for PR comment formatting."""

    def test_format_pr_comment(self):
        """Formats PR comment."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo")
        
        decision = ramp.evaluate_enforcement(
            "repo",
            [create_finding(severity="error")],
        )
        
        comment = ramp.format_pr_comment(decision)
        
        assert "CodeVerify" in comment
        assert "Phase" in comment

    def test_comment_shows_warnings(self):
        """Comment shows warning findings."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo")
        
        decision = ramp.evaluate_enforcement(
            "repo",
            [create_finding(message="Test warning message")],
        )
        
        comment = ramp.format_pr_comment(decision)
        
        assert "Warning" in comment

    def test_comment_shows_days_until_enforcement(self):
        """Comment shows days until enforcement."""
        schedule = RampSchedule(
            baseline_days=7,
            observation_days=14,
            critical_enforcement_day=21,
        )
        ramp = GradualVerificationRamp(default_schedule=schedule)
        
        ramp.start_ramp(
            "repo",
            start_date=datetime.utcnow() - timedelta(days=10),
        )
        
        decision = ramp.evaluate_enforcement(
            "repo",
            [create_finding(severity="critical")],
        )
        
        if decision.days_until_enforcement:
            comment = ramp.format_pr_comment(decision)
            assert "days" in comment.lower()


class TestGradualRampEdgeCases:
    """Edge case tests for gradual ramp."""

    def test_no_findings(self):
        """Handles no findings."""
        ramp = GradualVerificationRamp()
        ramp.start_ramp("repo")
        
        decision = ramp.evaluate_enforcement("repo", [])
        
        assert decision.should_block is False
        assert len(decision.blocking_findings) == 0
        assert len(decision.warning_findings) == 0

    def test_pause_unknown_repo(self):
        """Pause returns False for unknown repo."""
        ramp = GradualVerificationRamp()
        assert ramp.pause_ramp("unknown") is False

    def test_extend_unknown_repo(self):
        """Extend returns False for unknown repo."""
        ramp = GradualVerificationRamp()
        assert ramp.extend_ramp("unknown", 7) is False

    def test_concurrent_repos(self):
        """Handles multiple repos independently."""
        ramp = GradualVerificationRamp()
        
        # Start ramps at different times
        ramp.start_ramp(
            "repo1",
            start_date=datetime.utcnow() - timedelta(days=5),
        )
        ramp.start_ramp(
            "repo2",
            start_date=datetime.utcnow() - timedelta(days=30),
        )
        
        state1 = ramp.get_state("repo1")
        state2 = ramp.get_state("repo2")
        
        # Should be in different phases
        assert state1.days_elapsed != state2.days_elapsed
