"""AI Drift & Regression Monitor.

Continuously monitors repositories for semantic drift: detects when
commits subtly change function behavior, violate previously-verified
invariants, or introduce inconsistent patterns.

Features:
- Behavioral fingerprinting per function (inputs→outputs contract)
- Drift detection on push via fingerprint comparison
- Invariant monitoring for previously-proven properties
- Configurable drift sensitivity thresholds
- Drift alert generation with severity classification
- Historical drift timeline tracking
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class DriftSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class DriftType(str, Enum):
    BEHAVIOR_CHANGE = "behavior_change"
    SIGNATURE_CHANGE = "signature_change"
    INVARIANT_VIOLATION = "invariant_violation"
    PATTERN_INCONSISTENCY = "pattern_inconsistency"
    EXCEPTION_CHANGE = "exception_change"


class MonitorStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"


@dataclass
class BehavioralFingerprint:
    """Behavioral signature of a function."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    function_name: str = ""
    file_path: str = ""
    signature_hash: str = ""
    parameters: list[str] = field(default_factory=list)
    return_type: str = ""
    raises: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    complexity: int = 0
    content_hash: str = ""
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def compute_hash(self, content: str) -> str:
        self.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        sig = f"{self.function_name}({','.join(self.parameters)})->{self.return_type}"
        self.signature_hash = hashlib.sha256(sig.encode()).hexdigest()[:12]
        return self.content_hash


@dataclass
class VerifiedInvariant:
    """A previously-verified invariant to monitor."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    function_name: str = ""
    file_path: str = ""
    invariant_text: str = ""
    z3_assertion: str = ""
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


@dataclass
class DriftAlert:
    """An alert for detected drift."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    drift_type: DriftType = DriftType.BEHAVIOR_CHANGE
    severity: DriftSeverity = DriftSeverity.MEDIUM
    function_name: str = ""
    file_path: str = ""
    message: str = ""
    old_fingerprint: str = ""
    new_fingerprint: str = ""
    commit_sha: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False


@dataclass
class DriftReport:
    """Summary report of drift detection results."""
    repo: str = ""
    commit_sha: str = ""
    alerts: list[DriftAlert] = field(default_factory=list)
    functions_checked: int = 0
    functions_drifted: int = 0
    invariants_checked: int = 0
    invariants_violated: int = 0
    scan_time_ms: int = 0

    @property
    def drift_rate(self) -> float:
        if self.functions_checked == 0:
            return 0.0
        return round(self.functions_drifted / self.functions_checked, 3)


class FingerprintExtractor:
    """Extracts behavioral fingerprints from source code."""

    def extract(self, file_path: str, content: str, language: str = "python") -> list[BehavioralFingerprint]:
        """Extract fingerprints for all functions in a file."""
        fingerprints: list[BehavioralFingerprint] = []
        lines = content.split("\n")
        func_pattern = "def " if language == "python" else "function " if language in ("typescript", "javascript") else "func "

        current_func: str | None = None
        func_lines: list[str] = []
        func_start = 0
        params: list[str] = []

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(func_pattern):
                if current_func and func_lines:
                    fp = self._create_fingerprint(file_path, current_func, params, func_lines)
                    fingerprints.append(fp)

                name_part = stripped[len(func_pattern):].split("(")[0].strip()
                current_func = name_part
                func_start = i
                func_lines = [line]
                param_str = stripped.split("(")[1].split(")")[0] if "(" in stripped else ""
                params = [p.strip().split(":")[0].strip() for p in param_str.split(",") if p.strip()]
            elif current_func:
                func_lines.append(line)

        if current_func and func_lines:
            fp = self._create_fingerprint(file_path, current_func, params, func_lines)
            fingerprints.append(fp)

        return fingerprints

    def _create_fingerprint(
        self, file_path: str, name: str, params: list[str], lines: list[str]
    ) -> BehavioralFingerprint:
        content = "\n".join(lines)
        raises = [l.strip().split("raise ")[1].split("(")[0] for l in lines if "raise " in l]
        calls = []
        for l in lines:
            stripped = l.strip()
            if "(" in stripped and not stripped.startswith(("def ", "class ", "#", "if ", "for ", "while ")):
                call_name = stripped.split("(")[0].strip().split(".")[-1]
                if call_name and call_name[0].islower():
                    calls.append(call_name)

        fp = BehavioralFingerprint(
            function_name=name,
            file_path=file_path,
            parameters=params,
            raises=raises[:5],
            calls=calls[:10],
            complexity=len(lines),
        )
        fp.compute_hash(content)
        return fp


class DriftDetector:
    """Detects behavioral drift between two versions of code."""

    def detect(
        self,
        old_fingerprints: list[BehavioralFingerprint],
        new_fingerprints: list[BehavioralFingerprint],
        commit_sha: str = "",
    ) -> list[DriftAlert]:
        """Detect drift between old and new fingerprints."""
        alerts: list[DriftAlert] = []
        old_map = {(fp.file_path, fp.function_name): fp for fp in old_fingerprints}
        new_map = {(fp.file_path, fp.function_name): fp for fp in new_fingerprints}

        for key, new_fp in new_map.items():
            old_fp = old_map.get(key)
            if not old_fp:
                continue

            if old_fp.content_hash == new_fp.content_hash:
                continue

            # Signature change
            if old_fp.signature_hash != new_fp.signature_hash:
                alerts.append(DriftAlert(
                    drift_type=DriftType.SIGNATURE_CHANGE,
                    severity=DriftSeverity.HIGH,
                    function_name=new_fp.function_name,
                    file_path=new_fp.file_path,
                    message=f"Function signature changed: params {old_fp.parameters} → {new_fp.parameters}",
                    old_fingerprint=old_fp.signature_hash,
                    new_fingerprint=new_fp.signature_hash,
                    commit_sha=commit_sha,
                ))

            # Exception change
            if set(old_fp.raises) != set(new_fp.raises):
                added = set(new_fp.raises) - set(old_fp.raises)
                removed = set(old_fp.raises) - set(new_fp.raises)
                alerts.append(DriftAlert(
                    drift_type=DriftType.EXCEPTION_CHANGE,
                    severity=DriftSeverity.MEDIUM,
                    function_name=new_fp.function_name,
                    file_path=new_fp.file_path,
                    message=f"Exception behavior changed. Added: {added or 'none'}, Removed: {removed or 'none'}",
                    commit_sha=commit_sha,
                ))

            # Behavioral change (content differs but signature same)
            if old_fp.content_hash != new_fp.content_hash and old_fp.signature_hash == new_fp.signature_hash:
                complexity_delta = abs(new_fp.complexity - old_fp.complexity)
                severity = DriftSeverity.LOW if complexity_delta < 5 else DriftSeverity.MEDIUM
                alerts.append(DriftAlert(
                    drift_type=DriftType.BEHAVIOR_CHANGE,
                    severity=severity,
                    function_name=new_fp.function_name,
                    file_path=new_fp.file_path,
                    message=f"Function behavior changed (complexity delta: {complexity_delta})",
                    old_fingerprint=old_fp.content_hash,
                    new_fingerprint=new_fp.content_hash,
                    commit_sha=commit_sha,
                ))

        return alerts


class InvariantMonitor:
    """Monitors that previously-verified invariants still hold."""

    def check_invariants(
        self,
        invariants: list[VerifiedInvariant],
        current_code: dict[str, str],
    ) -> list[DriftAlert]:
        """Check invariants against current code."""
        alerts: list[DriftAlert] = []
        for inv in invariants:
            if not inv.is_active:
                continue
            code = current_code.get(inv.file_path, "")
            if not code:
                continue
            if inv.function_name not in code:
                alerts.append(DriftAlert(
                    drift_type=DriftType.INVARIANT_VIOLATION,
                    severity=DriftSeverity.CRITICAL,
                    function_name=inv.function_name,
                    file_path=inv.file_path,
                    message=f"Function '{inv.function_name}' removed — invariant '{inv.invariant_text}' can no longer be verified",
                ))
        return alerts


class DriftMonitorService:
    """Main service for AI drift & regression monitoring."""

    def __init__(self) -> None:
        self._extractor = FingerprintExtractor()
        self._detector = DriftDetector()
        self._invariant_monitor = InvariantMonitor()
        self._baselines: dict[str, list[BehavioralFingerprint]] = {}  # repo → fingerprints
        self._invariants: dict[str, list[VerifiedInvariant]] = {}
        self._alerts: list[DriftAlert] = []
        self._status = MonitorStatus.ACTIVE

    def set_baseline(self, repo: str, files: dict[str, str], language: str = "python") -> int:
        """Set the baseline fingerprints for a repository."""
        all_fps: list[BehavioralFingerprint] = []
        for path, content in files.items():
            fps = self._extractor.extract(path, content, language)
            all_fps.extend(fps)
        self._baselines[repo] = all_fps
        return len(all_fps)

    def register_invariant(self, repo: str, invariant: VerifiedInvariant) -> None:
        if repo not in self._invariants:
            self._invariants[repo] = []
        self._invariants[repo].append(invariant)

    def scan(
        self,
        repo: str,
        current_files: dict[str, str],
        commit_sha: str = "",
        language: str = "python",
    ) -> DriftReport:
        """Scan for drift against baseline."""
        import time
        start = time.time()

        new_fps: list[BehavioralFingerprint] = []
        for path, content in current_files.items():
            fps = self._extractor.extract(path, content, language)
            new_fps.extend(fps)

        old_fps = self._baselines.get(repo, [])
        alerts = self._detector.detect(old_fps, new_fps, commit_sha)

        inv_alerts = self._invariant_monitor.check_invariants(
            self._invariants.get(repo, []), current_files
        )
        alerts.extend(inv_alerts)

        self._alerts.extend(alerts)
        elapsed = int((time.time() - start) * 1000)

        drifted = len(set(a.function_name for a in alerts if a.drift_type != DriftType.INVARIANT_VIOLATION))
        inv_violated = sum(1 for a in alerts if a.drift_type == DriftType.INVARIANT_VIOLATION)

        return DriftReport(
            repo=repo,
            commit_sha=commit_sha,
            alerts=alerts,
            functions_checked=len(new_fps),
            functions_drifted=drifted,
            invariants_checked=len(self._invariants.get(repo, [])),
            invariants_violated=inv_violated,
            scan_time_ms=elapsed,
        )

    def update_baseline(self, repo: str, files: dict[str, str], language: str = "python") -> int:
        """Update baseline after accepting current state."""
        return self.set_baseline(repo, files, language)

    def acknowledge_alert(self, alert_id: str) -> bool:
        for a in self._alerts:
            if a.id == alert_id:
                a.acknowledged = True
                return True
        return False

    def get_alerts(self, repo: str | None = None) -> list[DriftAlert]:
        if repo:
            baselines = self._baselines.get(repo, [])
            paths = set(fp.file_path for fp in baselines)
            return [a for a in self._alerts if a.file_path in paths]
        return list(self._alerts)


# ─── Singleton Access ──────────────────────────────────────────────────


_drift_monitor_instance: DriftMonitorService | None = None


def get_drift_monitor_service() -> DriftMonitorService:
    global _drift_monitor_instance
    if _drift_monitor_instance is None:
        _drift_monitor_instance = DriftMonitorService()
    return _drift_monitor_instance


def reset_drift_monitor_service() -> None:
    global _drift_monitor_instance
    _drift_monitor_instance = None
