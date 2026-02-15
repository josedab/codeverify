"""IDE Copilot Undo with Proof Preservation.

Save point system for Copilot suggestions with one-click rollback
and preserved verification state. Time-travel debugging for AI
suggestions with proof audit trail.

Features:
- Save point creation on Copilot suggestion acceptance
- Lightweight snapshots with diff + proof metadata
- One-click rollback to previous verified state
- Trust score trend tracking over save points
- Branching save points for comparing suggestions
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class SavePointType(str, Enum):
    """Type of save point."""

    COPILOT_ACCEPT = "copilot_accept"
    MANUAL = "manual"
    AUTO = "auto"
    BRANCH = "branch"


class SavePointStatus(str, Enum):
    """Status of a save point."""

    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    SUPERSEDED = "superseded"
    PRUNED = "pruned"


class VerificationState(str, Enum):
    """Verification state at save point time."""

    VERIFIED = "verified"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"
    SKIPPED = "skipped"


@dataclass
class CodeDiff:
    """A code diff between save points."""

    file_path: str
    before: str = ""
    after: str = ""
    line_start: int = 0
    line_end: int = 0
    diff_hash: str = ""

    def __post_init__(self) -> None:
        if not self.diff_hash:
            content = f"{self.file_path}:{self.before}:{self.after}"
            self.diff_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def lines_changed(self) -> int:
        before_lines = len(self.before.strip().split("\n")) if self.before.strip() else 0
        after_lines = len(self.after.strip().split("\n")) if self.after.strip() else 0
        return abs(after_lines - before_lines) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file_path,
            "lines_changed": self.lines_changed,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "diff_hash": self.diff_hash,
        }


@dataclass
class ProofSnapshot:
    """Snapshot of verification state at a point in time."""

    trust_score: float = 0.0
    verification_state: VerificationState = VerificationState.PENDING
    findings_count: int = 0
    critical_findings: int = 0
    proof_hashes: list[str] = field(default_factory=list)
    constraints_verified: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trust_score": round(self.trust_score, 2),
            "state": self.verification_state.value,
            "findings": self.findings_count,
            "critical": self.critical_findings,
            "constraints_verified": self.constraints_verified,
        }


@dataclass
class SavePoint:
    """A save point capturing code state and verification proof."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    save_type: SavePointType = SavePointType.COPILOT_ACCEPT
    status: SavePointStatus = SavePointStatus.ACTIVE
    diff: CodeDiff | None = None
    proof_snapshot: ProofSnapshot = field(default_factory=ProofSnapshot)
    parent_id: str | None = None
    branch_label: str | None = None
    copilot_suggestion_id: str | None = None
    description: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.save_type.value,
            "status": self.status.value,
            "trust_score": self.proof_snapshot.trust_score,
            "verification_state": self.proof_snapshot.verification_state.value,
            "parent_id": self.parent_id,
            "branch_label": self.branch_label,
            "description": self.description,
        }


@dataclass
class TrustScoreTrend:
    """Trust score trend over save points."""

    scores: list[float] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)

    @property
    def direction(self) -> str:
        if len(self.scores) < 2:
            return "stable"
        recent = self.scores[-3:] if len(self.scores) >= 3 else self.scores
        if recent[-1] > recent[0] + 5:
            return "improving"
        elif recent[-1] < recent[0] - 5:
            return "declining"
        return "stable"

    @property
    def current(self) -> float:
        return self.scores[-1] if self.scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": round(self.current, 2),
            "direction": self.direction,
            "data_points": len(self.scores),
        }


class CopilotUndoManager:
    """Manages save points for Copilot suggestion undo with proof preservation."""

    MAX_SAVE_POINTS = 50

    def __init__(self) -> None:
        self._save_points: dict[str, SavePoint] = {}
        self._chronological: list[str] = []
        self._trust_trend = TrustScoreTrend()
        self._rollback_count: int = 0

    def create_save_point(
        self,
        diff: CodeDiff,
        proof_snapshot: ProofSnapshot | None = None,
        save_type: SavePointType = SavePointType.COPILOT_ACCEPT,
        copilot_suggestion_id: str | None = None,
        description: str = "",
    ) -> SavePoint:
        """Create a new save point when a Copilot suggestion is accepted."""
        parent_id = self._chronological[-1] if self._chronological else None

        sp = SavePoint(
            save_type=save_type,
            diff=diff,
            proof_snapshot=proof_snapshot or ProofSnapshot(),
            parent_id=parent_id,
            copilot_suggestion_id=copilot_suggestion_id,
            description=description or f"Save point for {diff.file_path}",
        )
        self._save_points[sp.id] = sp
        self._chronological.append(sp.id)

        self._trust_trend.scores.append(sp.proof_snapshot.trust_score)
        self._trust_trend.timestamps.append(sp.created_at.isoformat())

        self._prune_if_needed()

        logger.info(
            "save_point_created",
            save_point_id=sp.id,
            type=save_type.value,
            trust_score=sp.proof_snapshot.trust_score,
        )
        return sp

    def rollback(self, save_point_id: str) -> SavePoint | None:
        """Rollback to a specific save point."""
        target = self._save_points.get(save_point_id)
        if not target or target.status != SavePointStatus.ACTIVE:
            return None

        idx = self._chronological.index(save_point_id) if save_point_id in self._chronological else -1
        if idx < 0:
            return None

        # Mark all subsequent save points as rolled back
        for sp_id in self._chronological[idx + 1:]:
            sp = self._save_points.get(sp_id)
            if sp:
                sp.status = SavePointStatus.ROLLED_BACK

        self._chronological = self._chronological[:idx + 1]
        self._rollback_count += 1

        logger.info(
            "rollback_performed",
            target_id=save_point_id,
            total_rollbacks=self._rollback_count,
        )
        return target

    def create_branch(
        self,
        parent_id: str,
        diff: CodeDiff,
        branch_label: str,
        proof_snapshot: ProofSnapshot | None = None,
    ) -> SavePoint | None:
        """Create a branching save point for comparing suggestions."""
        parent = self._save_points.get(parent_id)
        if not parent:
            return None

        sp = SavePoint(
            save_type=SavePointType.BRANCH,
            diff=diff,
            proof_snapshot=proof_snapshot or ProofSnapshot(),
            parent_id=parent_id,
            branch_label=branch_label,
            description=f"Branch: {branch_label}",
        )
        self._save_points[sp.id] = sp
        return sp

    def compare_branches(
        self, branch_a_id: str, branch_b_id: str,
    ) -> dict[str, Any]:
        """Compare two branching save points."""
        a = self._save_points.get(branch_a_id)
        b = self._save_points.get(branch_b_id)
        if not a or not b:
            return {"error": "Branch not found"}

        return {
            "branch_a": {
                "label": a.branch_label,
                "trust_score": a.proof_snapshot.trust_score,
                "state": a.proof_snapshot.verification_state.value,
                "findings": a.proof_snapshot.findings_count,
            },
            "branch_b": {
                "label": b.branch_label,
                "trust_score": b.proof_snapshot.trust_score,
                "state": b.proof_snapshot.verification_state.value,
                "findings": b.proof_snapshot.findings_count,
            },
            "recommendation": (
                a.branch_label if a.proof_snapshot.trust_score >= b.proof_snapshot.trust_score
                else b.branch_label
            ),
        }

    def get_history(self, limit: int = 20) -> list[SavePoint]:
        """Get save point history (most recent first)."""
        ids = self._chronological[-limit:]
        return [self._save_points[sid] for sid in reversed(ids) if sid in self._save_points]

    @property
    def trust_trend(self) -> TrustScoreTrend:
        return self._trust_trend

    @property
    def save_point_count(self) -> int:
        return len(self._save_points)

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    def _prune_if_needed(self) -> None:
        """Prune old save points if over limit."""
        while len(self._chronological) > self.MAX_SAVE_POINTS:
            old_id = self._chronological.pop(0)
            sp = self._save_points.get(old_id)
            if sp:
                sp.status = SavePointStatus.PRUNED

    def get_summary(self) -> dict[str, Any]:
        """Get undo manager summary."""
        active = [sp for sp in self._save_points.values() if sp.status == SavePointStatus.ACTIVE]
        branches = [sp for sp in self._save_points.values() if sp.save_type == SavePointType.BRANCH]
        return {
            "total_save_points": len(self._save_points),
            "active_save_points": len(active),
            "branches": len(branches),
            "rollbacks_performed": self._rollback_count,
            "trust_trend": self._trust_trend.to_dict(),
        }


_default_manager: CopilotUndoManager | None = None


def get_copilot_undo_manager() -> CopilotUndoManager:
    """Get the singleton Copilot undo manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = CopilotUndoManager()
    return _default_manager


def reset_copilot_undo_manager() -> None:
    """Reset the singleton (for testing)."""
    global _default_manager
    _default_manager = None
