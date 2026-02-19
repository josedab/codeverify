"""Verification Replay & Regression Testing.

Records verification sessions as replayable artifacts, replays proofs
against new code versions, and detects proof regressions.

Features:
- Session recording with content-addressed storage
- Replay engine for re-running proofs against new code
- Proof regression detection (previously-proven properties failing)
- Regression dashboard data generation
- CI/CD integration for proof-based quality gates
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ReplayResult(str, Enum):
    STILL_VALID = "still_valid"
    REGRESSION = "regression"
    IMPROVED = "improved"
    CHANGED = "changed"
    ERROR = "error"


class SessionStatus(str, Enum):
    RECORDED = "recorded"
    REPLAYING = "replaying"
    REPLAYED = "replayed"
    FAILED = "failed"


@dataclass
class VerificationSnapshot:
    """Snapshot of a single verification check."""
    check_type: str = ""
    function_name: str = ""
    file_path: str = ""
    result: str = ""  # "pass" or "fail"
    counterexample: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    content_hash: str = ""


@dataclass
class RecordedSession:
    """A recorded verification session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    repo: str = ""
    commit_sha: str = ""
    snapshots: list[VerificationSnapshot] = field(default_factory=list)
    status: SessionStatus = SessionStatus.RECORDED
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    code_hashes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        content = json.dumps([s.content_hash for s in self.snapshots], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ReplayComparison:
    """Comparison between original and replayed verification."""
    snapshot_index: int = 0
    check_type: str = ""
    function_name: str = ""
    original_result: str = ""
    replay_result: str = ""
    comparison: ReplayResult = ReplayResult.STILL_VALID
    details: str = ""


@dataclass
class ReplayReport:
    """Report from replaying a session against new code."""
    session_id: str = ""
    original_commit: str = ""
    replay_commit: str = ""
    comparisons: list[ReplayComparison] = field(default_factory=list)
    regressions: int = 0
    improvements: int = 0
    unchanged: int = 0
    total_checks: int = 0
    replayed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def regression_rate(self) -> float:
        return round(self.regressions / self.total_checks, 3) if self.total_checks > 0 else 0.0


@dataclass
class RegressionTrend:
    """Trend data for proof regressions over time."""
    period: str = ""
    sessions_replayed: int = 0
    total_regressions: int = 0
    total_improvements: int = 0
    regression_rate: float = 0.0


class SessionRecorder:
    """Records verification sessions."""

    def record(
        self,
        repo: str,
        commit_sha: str,
        checks: list[dict[str, Any]],
        code_files: dict[str, str] | None = None,
    ) -> RecordedSession:
        """Record a verification session."""
        snapshots: list[VerificationSnapshot] = []
        for check in checks:
            code_content = check.get("code", "")
            snapshots.append(VerificationSnapshot(
                check_type=check.get("check_type", ""),
                function_name=check.get("function_name", ""),
                file_path=check.get("file_path", ""),
                result=check.get("result", "pass"),
                counterexample=check.get("counterexample", {}),
                constraints=check.get("constraints", []),
                content_hash=hashlib.sha256(code_content.encode()).hexdigest()[:12],
            ))

        code_hashes = {}
        if code_files:
            for path, content in code_files.items():
                code_hashes[path] = hashlib.sha256(content.encode()).hexdigest()[:12]

        return RecordedSession(
            repo=repo, commit_sha=commit_sha,
            snapshots=snapshots, code_hashes=code_hashes,
        )


class ReplayEngine:
    """Replays recorded sessions against new code."""

    def replay(
        self,
        session: RecordedSession,
        new_code_files: dict[str, str],
        new_commit_sha: str = "",
    ) -> ReplayReport:
        """Replay a session against new code."""
        comparisons: list[ReplayComparison] = []
        regressions = 0
        improvements = 0
        unchanged = 0

        new_hashes: dict[str, str] = {}
        for path, content in new_code_files.items():
            new_hashes[path] = hashlib.sha256(content.encode()).hexdigest()[:12]

        for i, snapshot in enumerate(session.snapshots):
            old_hash = session.code_hashes.get(snapshot.file_path, "")
            new_hash = new_hashes.get(snapshot.file_path, "")
            new_content = new_code_files.get(snapshot.file_path, "")

            if old_hash == new_hash:
                comparisons.append(ReplayComparison(
                    snapshot_index=i, check_type=snapshot.check_type,
                    function_name=snapshot.function_name,
                    original_result=snapshot.result, replay_result=snapshot.result,
                    comparison=ReplayResult.STILL_VALID,
                    details="Code unchanged — proof still valid",
                ))
                unchanged += 1
                continue

            # Code changed — re-verify
            replay_result = self._simulate_verification(
                snapshot, new_content
            )

            if snapshot.result == "pass" and replay_result == "fail":
                comparison = ReplayResult.REGRESSION
                regressions += 1
                details = f"Previously passing check now fails — {snapshot.check_type} regression"
            elif snapshot.result == "fail" and replay_result == "pass":
                comparison = ReplayResult.IMPROVED
                improvements += 1
                details = f"Previously failing check now passes — bug fixed"
            elif snapshot.result == replay_result:
                comparison = ReplayResult.STILL_VALID
                unchanged += 1
                details = "Result unchanged despite code changes"
            else:
                comparison = ReplayResult.CHANGED
                unchanged += 1
                details = "Result changed"

            comparisons.append(ReplayComparison(
                snapshot_index=i, check_type=snapshot.check_type,
                function_name=snapshot.function_name,
                original_result=snapshot.result, replay_result=replay_result,
                comparison=comparison, details=details,
            ))

        return ReplayReport(
            session_id=session.id,
            original_commit=session.commit_sha,
            replay_commit=new_commit_sha,
            comparisons=comparisons,
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
            total_checks=len(session.snapshots),
        )

    def _simulate_verification(
        self, snapshot: VerificationSnapshot, new_code: str
    ) -> str:
        """Simulate re-verification of a check against new code."""
        if not new_code:
            return "fail"
        if snapshot.function_name and snapshot.function_name not in new_code:
            return "fail"
        if snapshot.check_type == "null_safety":
            if "is not None" in new_code or ".get(" in new_code:
                return "pass"
            if "None" in str(snapshot.counterexample):
                return "fail"
        if snapshot.check_type == "division_by_zero":
            if "!= 0" in new_code or "if " in new_code:
                return "pass"
        return snapshot.result


class VerificationReplayService:
    """Main service for verification replay and regression testing."""

    def __init__(self) -> None:
        self._recorder = SessionRecorder()
        self._engine = ReplayEngine()
        self._sessions: dict[str, RecordedSession] = {}
        self._reports: list[ReplayReport] = []

    def record_session(
        self, repo: str, commit_sha: str,
        checks: list[dict[str, Any]],
        code_files: dict[str, str] | None = None,
    ) -> RecordedSession:
        session = self._recorder.record(repo, commit_sha, checks, code_files)
        self._sessions[session.id] = session
        return session

    def replay_session(
        self, session_id: str,
        new_code_files: dict[str, str],
        new_commit_sha: str = "",
    ) -> ReplayReport | None:
        session = self._sessions.get(session_id)
        if not session:
            return None
        report = self._engine.replay(session, new_code_files, new_commit_sha)
        self._reports.append(report)
        return report

    def get_regressions(self) -> list[ReplayComparison]:
        regressions: list[ReplayComparison] = []
        for report in self._reports:
            regressions.extend(c for c in report.comparisons if c.comparison == ReplayResult.REGRESSION)
        return regressions

    def get_trend(self) -> list[RegressionTrend]:
        if not self._reports:
            return []
        total_reg = sum(r.regressions for r in self._reports)
        total_imp = sum(r.improvements for r in self._reports)
        total_checks = sum(r.total_checks for r in self._reports)
        return [RegressionTrend(
            period="current",
            sessions_replayed=len(self._reports),
            total_regressions=total_reg,
            total_improvements=total_imp,
            regression_rate=round(total_reg / total_checks, 3) if total_checks > 0 else 0.0,
        )]

    def get_session(self, session_id: str) -> RecordedSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, repo: str = "") -> list[RecordedSession]:
        sessions = list(self._sessions.values())
        if repo:
            sessions = [s for s in sessions if s.repo == repo]
        return sessions


# ─── Singleton Access ──────────────────────────────────────────────────

_replay_instance: VerificationReplayService | None = None

def get_verification_replay_service() -> VerificationReplayService:
    global _replay_instance
    if _replay_instance is None:
        _replay_instance = VerificationReplayService()
    return _replay_instance

def reset_verification_replay_service() -> None:
    global _replay_instance
    _replay_instance = None
