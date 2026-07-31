"""CI/CD Verification Agent — Pipeline connectors, incremental verification, and change impact.

Provides CI pipeline integration for automated code verification with support
for GitHub Actions, GitLab CI, Jenkins, CircleCI, Azure DevOps, and Buildkite.
Includes change detection, proof caching, quality gate evaluation, and
configuration generation for popular CI providers.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CIProvider(str, Enum):
    """Supported CI/CD providers."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    CIRCLE_CI = "circle_ci"
    AZURE_DEVOPS = "azure_devops"
    BUILDKITE = "buildkite"


class PipelineStage(str, Enum):
    """Stages within a verification pipeline."""

    CHECKOUT = "checkout"
    ANALYZE = "analyze"
    VERIFY = "verify"
    REPORT = "report"
    GATE = "gate"


class GateDecision(str, Enum):
    """Quality-gate outcome."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    MANUAL_REVIEW = "manual_review"


class VerificationScope(str, Enum):
    """Scope of verification to perform."""

    FULL = "full"
    INCREMENTAL = "incremental"
    AFFECTED_ONLY = "affected_only"
    CRITICAL_PATHS = "critical_paths"


class ChangeCategory(str, Enum):
    """Category of a file change."""

    NEW_CODE = "new_code"
    MODIFIED = "modified"
    REFACTORED = "refactored"
    DELETED = "deleted"
    DEPENDENCY_UPDATE = "dependency_update"
    CONFIG_CHANGE = "config_change"


@dataclass
class CIConfig:
    """Configuration for a CI verification run."""

    provider: CIProvider
    repo_url: str
    branch: str = "main"
    verification_scope: VerificationScope = VerificationScope.INCREMENTAL
    fail_on_critical: bool = True
    fail_on_high: bool = False
    timeout_seconds: int = 300
    parallel_jobs: int = 4
    cache_proofs: bool = True
    notify_on_failure: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in self.__dict__.items()}


@dataclass
class FileChange:
    """A single file change extracted from a diff."""

    path: str
    category: ChangeCategory
    additions: int = 0
    deletions: int = 0
    risk_score: float = 0.0
    requires_verification: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "category": self.category.value,
            "additions": self.additions,
            "deletions": self.deletions,
            "risk_score": round(self.risk_score, 2),
            "requires_verification": self.requires_verification,
        }


@dataclass
class ChangeImpact:
    """Aggregated impact analysis for a set of file changes."""

    changed_files: list[FileChange]
    affected_modules: list[str]
    affected_tests: list[str]
    risk_level: str = "low"
    recommended_scope: VerificationScope = VerificationScope.INCREMENTAL
    estimated_time_seconds: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_files": [f.to_dict() for f in self.changed_files],
            "affected_modules": self.affected_modules,
            "affected_tests": self.affected_tests,
            "risk_level": self.risk_level,
            "recommended_scope": self.recommended_scope.value,
            "estimated_time_seconds": self.estimated_time_seconds,
        }


@dataclass
class PipelineRun:
    """State for a single pipeline execution."""

    id: str
    config: CIConfig
    commit_sha: str
    branch: str
    trigger: str = "push"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    gate_decision: GateDecision = GateDecision.PASS
    findings: list[dict[str, Any]] = field(default_factory=list)
    cached_proofs_used: int = 0
    verification_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "trigger": self.trigger,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "gate_decision": self.gate_decision.value,
            "findings_count": len(self.findings),
            "cached_proofs_used": self.cached_proofs_used,
            "verification_time_ms": round(self.verification_time_ms, 1),
            "stages": self.stages,
        }


@dataclass
class GatePolicy:
    """Quality-gate thresholds."""

    max_critical: int = 0
    max_high: int = 0
    max_medium: int = -1  # -1 = unlimited
    min_verification_coverage: float = 80.0
    require_formal_proof: bool = False
    allow_override: bool = True
    reviewers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_critical": self.max_critical,
            "max_high": self.max_high,
            "max_medium": self.max_medium,
            "min_verification_coverage": self.min_verification_coverage,
            "require_formal_proof": self.require_formal_proof,
            "allow_override": self.allow_override,
            "reviewers": self.reviewers,
        }


@dataclass
class PipelineReport:
    """Summary report produced at the end of a pipeline run."""

    run_id: str
    commit_sha: str
    gate_decision: GateDecision
    total_findings: int
    critical_count: int
    high_count: int
    files_verified: int
    files_skipped: int
    coverage_percent: float
    cached_proofs: int
    total_time_seconds: float
    pr_comment_markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "commit_sha": self.commit_sha,
            "gate_decision": self.gate_decision.value,
            "total_findings": self.total_findings,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "files_verified": self.files_verified,
            "files_skipped": self.files_skipped,
            "coverage_percent": round(self.coverage_percent, 1),
            "cached_proofs": self.cached_proofs,
            "total_time_seconds": round(self.total_time_seconds, 2),
        }


@dataclass
class ProofCache:
    """A cached verification proof for a single file."""

    file_path: str
    content_hash: str
    proof_result: dict[str, Any]
    cached_at: datetime
    valid_until: datetime | None = None

    @property
    def is_valid(self) -> bool:
        return self.valid_until is None or datetime.now(UTC) < self.valid_until

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "proof_result": self.proof_result,
            "cached_at": self.cached_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "is_valid": self.is_valid,
        }


# Patterns for recognising config / dependency / test files
_CONFIG_PATTERNS = re.compile(
    r"(\.ya?ml|\.toml|\.ini|\.cfg|\.json|Makefile|Dockerfile|\.env)$", re.IGNORECASE
)
_DEPENDENCY_PATTERNS = re.compile(
    r"(requirements.*\.txt|Pipfile|poetry\.lock|package\.json|go\.sum|Cargo\.lock)$",
    re.IGNORECASE,
)
_TEST_PATTERNS = re.compile(r"(test_|_test\.|\.test\.|spec\.|\.spec\.)", re.IGNORECASE)


class ChangeDetector:
    """Detect and classify code changes for incremental verification."""

    def __init__(self) -> None:
        self._hunk_header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        self._diff_file = re.compile(r"^diff --git a/(.*) b/(.*)")

    def analyze_diff(self, diff_text: str) -> list[FileChange]:
        """Parse a unified diff and return a list of *FileChange* objects."""
        changes: list[FileChange] = []
        current_path: str | None = None
        additions = deletions = 0

        for line in diff_text.splitlines():
            file_match = self._diff_file.match(line)
            if file_match:
                if current_path is not None:
                    changes.append(self._build_change(current_path, additions, deletions))
                current_path = file_match.group(2)
                additions = deletions = 0
                continue
            if current_path is None:
                continue
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1

        if current_path is not None:
            changes.append(self._build_change(current_path, additions, deletions))
        for change in changes:
            change.risk_score = self._calculate_file_risk(change)
        logger.info(
            "diff_analyzed",
            files_changed=len(changes),
            total_additions=sum(c.additions for c in changes),
            total_deletions=sum(c.deletions for c in changes),
        )
        return changes

    def calculate_impact(self, changes: list[FileChange]) -> ChangeImpact:
        """Calculate the aggregate impact of a set of changes."""
        affected_modules = self._find_affected_modules(changes)
        affected_tests = self._find_affected_tests(changes)
        max_risk = max((c.risk_score for c in changes), default=0.0)

        if max_risk >= 0.8:
            risk_level = "critical"
        elif max_risk >= 0.6:
            risk_level = "high"
        elif max_risk >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        if risk_level in ("critical", "high"):
            scope = VerificationScope.FULL
        elif len(changes) <= 3:
            scope = VerificationScope.AFFECTED_ONLY
        else:
            scope = VerificationScope.INCREMENTAL

        impact = ChangeImpact(
            changed_files=changes,
            affected_modules=affected_modules,
            affected_tests=affected_tests,
            risk_level=risk_level,
            recommended_scope=scope,
            estimated_time_seconds=max(30, len(changes) * 15),
        )
        logger.info(
            "impact_calculated",
            risk_level=risk_level,
            modules=len(affected_modules),
            tests=len(affected_tests),
        )
        return impact

    def _build_change(self, path: str, additions: int, deletions: int) -> FileChange:
        cat = self._classify_change(path, additions, deletions)
        return FileChange(
            path=path,
            category=cat,
            additions=additions,
            deletions=deletions,
            requires_verification=cat not in (ChangeCategory.DELETED, ChangeCategory.CONFIG_CHANGE),
        )

    def _classify_change(self, path: str, additions: int, deletions: int) -> ChangeCategory:
        if _DEPENDENCY_PATTERNS.search(path):
            return ChangeCategory.DEPENDENCY_UPDATE
        if _CONFIG_PATTERNS.search(path):
            return ChangeCategory.CONFIG_CHANGE
        if deletions > 0 and additions == 0:
            return ChangeCategory.DELETED
        if additions > 0 and deletions == 0:
            return ChangeCategory.NEW_CODE
        if (
            additions > 0
            and deletions > 0
            and min(additions, deletions) / max(additions, deletions) > 0.7
        ):
            return ChangeCategory.REFACTORED
        return ChangeCategory.MODIFIED

    def _calculate_file_risk(self, change: FileChange) -> float:
        """Return a 0.0-1.0 risk score for a file change."""
        score = min(0.3, (change.additions + change.deletions) / 500)
        weights = {
            ChangeCategory.NEW_CODE: 0.3,
            ChangeCategory.MODIFIED: 0.25,
            ChangeCategory.REFACTORED: 0.2,
            ChangeCategory.DELETED: 0.1,
            ChangeCategory.DEPENDENCY_UPDATE: 0.35,
            ChangeCategory.CONFIG_CHANGE: 0.15,
        }
        score += weights.get(change.category, 0.2)
        high_risk = ("auth", "security", "crypto", "payment", "billing", "admin")
        if any(tok in change.path.lower() for tok in high_risk):
            score += 0.25
        return min(1.0, score)

    def _find_affected_modules(self, changes: list[FileChange]) -> list[str]:
        modules: set[str] = set()
        for c in changes:
            parts = c.path.replace("\\", "/").split("/")
            if len(parts) >= 2:
                modules.add(parts[0] if parts[0] != "." else parts[1])
        return sorted(modules)

    def _find_affected_tests(self, changes: list[FileChange]) -> list[str]:
        tests: list[str] = []
        for c in changes:
            if _TEST_PATTERNS.search(os.path.basename(c.path)):
                tests.append(c.path)
            else:
                # Infer a companion test file
                base = os.path.splitext(os.path.basename(c.path))[0]
                tests.append(f"test_{base}.py")
        return tests


class ProofCacheManager:
    """Cache and reuse verification proofs for unchanged code."""

    def __init__(self) -> None:
        self._cache: dict[str, ProofCache] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get_cached_proof(self, file_path: str, content_hash: str) -> ProofCache | None:
        """Return a valid cached proof or ``None``."""
        entry = self._cache.get(f"{file_path}:{content_hash}")
        if entry is not None and entry.is_valid:
            self._hits += 1
            return entry
        self._misses += 1
        return None

    def store_proof(
        self, file_path: str, content_hash: str, proof_result: dict[str, Any]
    ) -> ProofCache:
        """Store a verification proof in the cache."""
        entry = ProofCache(
            file_path=file_path,
            content_hash=content_hash,
            proof_result=proof_result,
            cached_at=datetime.now(UTC),
        )
        self._cache[f"{file_path}:{content_hash}"] = entry
        return entry

    def invalidate(self, file_path: str) -> bool:
        """Invalidate all cached proofs for *file_path*."""
        keys = [k for k in self._cache if k.startswith(f"{file_path}:")]
        for k in keys:
            del self._cache[k]
        return len(keys) > 0

    def invalidate_all(self) -> int:
        """Invalidate the entire cache."""
        count = len(self._cache)
        self._cache.clear()
        self._hits = self._misses = 0
        return count

    def get_cache_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }

    @staticmethod
    def _compute_hash(content: str) -> str:
        """SHA-256 based content hash (first 32 hex chars)."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


class PipelineOrchestrator:
    """Orchestrate CI/CD verification pipeline."""

    def __init__(self, config: CIConfig | None = None) -> None:
        self.config = config or CIConfig(
            provider=CIProvider.GITHUB_ACTIONS,
            repo_url="https://github.com/example/repo",
        )
        self._detector = ChangeDetector()
        self._cache = ProofCacheManager()
        self._runs: list[PipelineRun] = []

    def run_pipeline(
        self,
        commit_sha: str,
        diff_text: str,
        files: dict[str, str],
    ) -> PipelineRun:
        """Execute the full verification pipeline and return a *PipelineRun*."""
        run = PipelineRun(
            id=str(uuid.uuid4())[:12],
            config=self.config,
            commit_sha=commit_sha,
            branch=self.config.branch,
        )
        t_start = time.monotonic()
        logger.info("pipeline_started", run_id=run.id, commit=commit_sha[:8])

        run.stages.append(self._stage_checkout(run, files))
        changes = self._detector.analyze_diff(diff_text)
        run.stages.append(self._stage_analyze(run, changes))
        run.stages.append(self._stage_verify(run, files, changes))
        run.stages.append(self._stage_report(run))
        run.stages.append(self._stage_gate(run, GatePolicy()))

        run.completed_at = datetime.now(UTC)
        run.verification_time_ms = (time.monotonic() - t_start) * 1000
        self._runs.append(run)
        logger.info(
            "pipeline_completed",
            run_id=run.id,
            gate=run.gate_decision.value,
            findings=len(run.findings),
            time_ms=round(run.verification_time_ms, 1),
        )
        return run

    def _stage_checkout(self, _run: PipelineRun, files: dict[str, str]) -> dict[str, Any]:
        return {
            "stage": PipelineStage.CHECKOUT.value,
            "status": "success",
            "files_received": len(files),
        }

    def _stage_analyze(self, _run: PipelineRun, changes: list[FileChange]) -> dict[str, Any]:
        impact = self._detector.calculate_impact(changes)
        return {
            "stage": PipelineStage.ANALYZE.value,
            "status": "success",
            "files_changed": len(changes),
            "risk_level": impact.risk_level,
            "recommended_scope": impact.recommended_scope.value,
        }

    def _stage_verify(
        self,
        run: PipelineRun,
        files: dict[str, str],
        changes: list[FileChange],
    ) -> dict[str, Any]:
        verified, skipped, cached = 0, 0, 0
        paths_to_verify = {c.path for c in changes if c.requires_verification}

        for path, content in files.items():
            if path not in paths_to_verify:
                skipped += 1
                continue
            content_hash = ProofCacheManager._compute_hash(content)
            if (
                self.config.cache_proofs
                and self._cache.get_cached_proof(path, content_hash) is not None
            ):
                cached += 1
                verified += 1
                continue
            file_findings = self._verify_file(path, content)
            run.findings.extend(file_findings)
            if self.config.cache_proofs:
                self._cache.store_proof(
                    path,
                    content_hash,
                    {"status": "verified", "findings": len(file_findings)},
                )
            verified += 1

        run.cached_proofs_used = cached
        return {
            "stage": PipelineStage.VERIFY.value,
            "status": "success",
            "files_verified": verified,
            "files_skipped": skipped,
            "cached_proofs_used": cached,
        }

    def _stage_report(self, run: PipelineRun) -> dict[str, Any]:
        return {
            "stage": PipelineStage.REPORT.value,
            "status": "success",
            "total_findings": len(run.findings),
        }

    def _stage_gate(self, run: PipelineRun, policy: GatePolicy) -> dict[str, Any]:
        evaluator = GateEvaluator(policy)
        vs = next((s for s in run.stages if s.get("stage") == PipelineStage.VERIFY.value), {})
        verified, skipped = vs.get("files_verified", 0), vs.get("files_skipped", 0)
        total = verified + skipped
        coverage = (verified / total * 100) if total > 0 else 100.0
        decision = evaluator.evaluate(run.findings, coverage)
        run.gate_decision = decision
        return {"stage": PipelineStage.GATE.value, "status": "success", "decision": decision.value}

    def _verify_file(self, path: str, content: str) -> list[dict[str, Any]]:
        """Run lightweight heuristic checks on a single file."""
        findings: list[dict[str, Any]] = []
        for idx, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if re.search(r"\b(TODO|FIXME|HACK|XXX)\b", stripped):
                findings.append(
                    {
                        "file": path,
                        "line": idx,
                        "severity": "low",
                        "message": "Unresolved marker comment",
                        "rule": "marker-comment",
                    }
                )
            if re.search(
                r"(password|secret|api_key|token)\s*=\s*['\"].{4,}['\"]", stripped, re.IGNORECASE
            ):
                findings.append(
                    {
                        "file": path,
                        "line": idx,
                        "severity": "critical",
                        "message": "Potential hardcoded secret",
                        "rule": "hardcoded-secret",
                    }
                )
            if re.search(r"\b(eval|exec)\s*\(", stripped):
                findings.append(
                    {
                        "file": path,
                        "line": idx,
                        "severity": "high",
                        "message": "Use of eval/exec is a security risk",
                        "rule": "dangerous-call",
                    }
                )
        return findings


class GateEvaluator:
    """Evaluate quality gate decisions."""

    def __init__(self, policy: GatePolicy | None = None) -> None:
        self.policy = policy or GatePolicy()

    def evaluate(self, findings: list[dict[str, Any]], coverage: float) -> GateDecision:
        """Return a *GateDecision* based on findings and coverage."""
        counts = self._count_by_severity(findings)

        if counts.get("critical", 0) > self.policy.max_critical:
            return GateDecision.FAIL
        if counts.get("high", 0) > self.policy.max_high:
            return GateDecision.FAIL
        if self.policy.max_medium >= 0 and counts.get("medium", 0) > self.policy.max_medium:
            return GateDecision.FAIL
        if coverage < self.policy.min_verification_coverage:
            if self.policy.allow_override:
                return GateDecision.MANUAL_REVIEW
            return GateDecision.FAIL

        # Warn when findings exist but thresholds are not breached
        if counts.get("high", 0) > 0 or counts.get("medium", 0) > 0:
            return GateDecision.WARN

        return GateDecision.PASS

    def generate_pr_comment(self, run: PipelineRun) -> str:
        """Generate a GitHub/GitLab PR comment in Markdown."""
        counts = self._count_by_severity(run.findings)
        badge = {
            GateDecision.PASS: "✅ **PASSED**",
            GateDecision.WARN: "⚠️ **WARNING**",
            GateDecision.FAIL: "❌ **FAILED**",
            GateDecision.MANUAL_REVIEW: "👀 **MANUAL REVIEW REQUIRED**",
        }.get(run.gate_decision, "❓ Unknown")

        vs = next((s for s in run.stages if s.get("stage") == PipelineStage.VERIFY.value), {})
        verified, skipped = vs.get("files_verified", 0), vs.get("files_skipped", 0)
        total = verified + skipped
        coverage = (verified / total * 100) if total > 0 else 100.0

        lines: list[str] = [
            f"## CodeVerify — {badge}",
            "",
            f"**Commit:** `{run.commit_sha[:8]}`  ",
            f"**Branch:** `{run.branch}`  ",
            f"**Time:** {run.verification_time_ms / 1000:.1f}s",
            "",
            "### Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total findings | {len(run.findings)} |",
            f"| Critical | {counts.get('critical', 0)} |",
            f"| High | {counts.get('high', 0)} |",
            f"| Medium | {counts.get('medium', 0)} |",
            f"| Low | {counts.get('low', 0)} |",
            f"| Files verified | {verified} |",
            f"| Files skipped | {skipped} |",
            f"| Coverage | {coverage:.1f}% |",
            f"| Cached proofs | {run.cached_proofs_used} |",
        ]
        if run.findings:
            lines.extend(
                [
                    "",
                    "### Findings",
                    "",
                    "| File | Line | Severity | Message |",
                    "|------|------|----------|---------|",
                ]
            )
            for f in run.findings[:20]:
                lines.append(
                    f"| `{f.get('file', '')}` | {f.get('line', '-')} "
                    f"| {f.get('severity', 'unknown')} | {f.get('message', '')} |"
                )
            if len(run.findings) > 20:
                lines.append(f"| … | … | … | *{len(run.findings) - 20} more findings* |")
        lines.extend(["", "---", f"*Generated by CodeVerify CI Agent • Run `{run.id}`*"])
        return "\n".join(lines)

    def _count_by_severity(self, findings: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in findings:
            sev = f.get("severity", "unknown")
            counts[sev] = counts.get(sev, 0) + 1
        return counts


class CIConfigGenerator:
    """Generate CI configuration files for different providers."""

    def __init__(self) -> None:
        self._generators: dict[CIProvider, Callable[[CIConfig], str]] = {
            CIProvider.GITHUB_ACTIONS: self._github_actions,
            CIProvider.GITLAB_CI: self._gitlab_ci,
        }

    def generate(self, config: CIConfig) -> str:
        """Return a YAML configuration string for the requested provider."""
        generator = self._generators.get(config.provider)
        if generator is None:
            logger.warning("unsupported_provider", provider=config.provider.value)
            return f"# Configuration generation not yet supported for {config.provider.value}\n"
        return generator(config)

    @staticmethod
    def _github_actions(config: CIConfig) -> str:
        timeout_min = config.timeout_seconds // 60 or 5
        fail_flags = ""
        if config.fail_on_critical:
            fail_flags += " --fail-on-critical"
        if config.fail_on_high:
            fail_flags += " --fail-on-high"
        return (
            "name: CodeVerify\n"
            "\n"
            "on:\n"
            f"  push:\n    branches: [{config.branch}]\n"
            f"  pull_request:\n    branches: [{config.branch}]\n"
            "\n"
            "permissions:\n  contents: read\n  pull-requests: write\n"
            "\n"
            "jobs:\n"
            "  codeverify:\n"
            f"    runs-on: ubuntu-latest\n"
            f"    timeout-minutes: {timeout_min}\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "        with:\n          fetch-depth: 0\n"
            "\n"
            "      - name: Set up Python\n"
            "        uses: actions/setup-python@v5\n"
            "        with:\n          python-version: '3.12'\n"
            "\n"
            "      - name: Install CodeVerify\n"
            "        run: pip install codeverify\n"
            "\n"
            "      - name: Run verification\n"
            "        run: |\n"
            f"          codeverify verify \\\n"
            f"            --scope {config.verification_scope.value} \\\n"
            f"            --parallel {config.parallel_jobs}{fail_flags}\n"
            "\n"
            "      - name: Upload results\n"
            "        if: always()\n"
            "        uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: codeverify-report\n"
            "          path: codeverify-report.json\n"
        )

    @staticmethod
    def _gitlab_ci(config: CIConfig) -> str:
        timeout_min = config.timeout_seconds // 60 or 5
        return (
            "stages:\n  - verify\n"
            "\n"
            "codeverify:\n"
            "  stage: verify\n"
            "  image: python:3.12-slim\n"
            f"  timeout: {timeout_min}m\n"
            "  before_script:\n    - pip install codeverify\n"
            "  script:\n    - |\n"
            f"      codeverify verify \\\n"
            f"        --scope {config.verification_scope.value} \\\n"
            f"        --parallel {config.parallel_jobs}\n"
            "  artifacts:\n    when: always\n"
            "    paths:\n      - codeverify-report.json\n"
            "    reports:\n      codequality: codeverify-report.json\n"
            "  rules:\n"
            f"    - if: '$CI_COMMIT_BRANCH == \"{config.branch}\"'\n"
            "    - if: '$CI_PIPELINE_SOURCE == \"merge_request_event\"'\n"
        )


class CIVerificationAgent:
    """Main orchestrator for CI/CD verification.

    Ties together change detection, proof caching, pipeline orchestration,
    gate evaluation, and CI configuration generation.
    """

    def __init__(
        self,
        config: CIConfig | None = None,
        policy: GatePolicy | None = None,
    ) -> None:
        self.config = config or CIConfig(
            provider=CIProvider.GITHUB_ACTIONS,
            repo_url="https://github.com/example/repo",
        )
        self.policy = policy or GatePolicy()
        self._orchestrator = PipelineOrchestrator(self.config)
        self._evaluator = GateEvaluator(self.policy)
        self._config_gen = CIConfigGenerator()

        logger.info(
            "ci_agent_initialized",
            provider=self.config.provider.value,
            scope=self.config.verification_scope.value,
        )

    def verify_commit(
        self,
        commit_sha: str,
        diff_text: str,
        files: dict[str, str],
    ) -> PipelineReport:
        """Run a full verification pipeline and return a *PipelineReport*."""
        run = self._orchestrator.run_pipeline(commit_sha, diff_text, files)
        counts = self._evaluator._count_by_severity(run.findings)
        vs = next((s for s in run.stages if s.get("stage") == PipelineStage.VERIFY.value), {})
        verified, skipped = vs.get("files_verified", 0), vs.get("files_skipped", 0)
        total = verified + skipped
        coverage = (verified / total * 100) if total > 0 else 100.0
        pr_comment = self._evaluator.generate_pr_comment(run)

        report = PipelineReport(
            run_id=run.id,
            commit_sha=commit_sha,
            gate_decision=run.gate_decision,
            total_findings=len(run.findings),
            critical_count=counts.get("critical", 0),
            high_count=counts.get("high", 0),
            files_verified=verified,
            files_skipped=skipped,
            coverage_percent=coverage,
            cached_proofs=run.cached_proofs_used,
            total_time_seconds=run.verification_time_ms / 1000,
            pr_comment_markdown=pr_comment,
        )
        logger.info(
            "commit_verified",
            commit=commit_sha[:8],
            decision=report.gate_decision.value,
            findings=report.total_findings,
        )
        return report

    def generate_config(self, provider: CIProvider) -> str:
        """Generate a CI configuration file for the given provider."""
        cfg = CIConfig(
            provider=provider,
            repo_url=self.config.repo_url,
            branch=self.config.branch,
            verification_scope=self.config.verification_scope,
            fail_on_critical=self.config.fail_on_critical,
            fail_on_high=self.config.fail_on_high,
            timeout_seconds=self.config.timeout_seconds,
            parallel_jobs=self.config.parallel_jobs,
            cache_proofs=self.config.cache_proofs,
            notify_on_failure=self.config.notify_on_failure,
        )
        return self._config_gen.generate(cfg)

    def get_cache_stats(self) -> dict[str, Any]:
        """Return proof-cache statistics."""
        return self._orchestrator._cache.get_cache_stats()

    def get_run_history(self) -> list[PipelineRun]:
        """Return all completed pipeline runs."""
        return list(self._orchestrator._runs)
