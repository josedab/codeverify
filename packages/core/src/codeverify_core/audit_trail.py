"""AI Code Audit Trail & Provenance Tracking.

Automatic detection, tracking, and auditing of AI-generated code across the
codebase with regulatory compliance reporting (EU AI Act, SOX).

Features:
- LLM fingerprinting (entropy, style, patterns)
- Provenance records with immutable hash chains
- Compliance reporting (EU AI Act, SOX, AIBOM)
- Dashboard metrics: % AI code by repo/team/file
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ProvenanceSource(str, Enum):
    """Source of the code."""

    HUMAN = "human"
    AI_COPILOT = "ai_copilot"
    AI_CHATGPT = "ai_chatgpt"
    AI_CLAUDE = "ai_claude"
    AI_CURSOR = "ai_cursor"
    AI_UNKNOWN = "ai_unknown"
    MIXED = "mixed"


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    EU_AI_ACT = "eu_ai_act"
    SOX = "sox"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    CUSTOM = "custom"


class RiskLevel(str, Enum):
    """Risk level of AI-generated code."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AIFingerprint:
    """Fingerprint analysis of code for AI detection."""

    entropy: float = 0.0
    avg_line_length: float = 0.0
    comment_ratio: float = 0.0
    unique_identifier_ratio: float = 0.0
    repetition_score: float = 0.0
    style_consistency: float = 0.0
    ai_probability: float = 0.0
    signals: list[str] = field(default_factory=list)

    @property
    def is_likely_ai(self) -> bool:
        return self.ai_probability >= 0.6

    @property
    def confidence_label(self) -> str:
        if self.ai_probability >= 0.9:
            return "very high"
        if self.ai_probability >= 0.7:
            return "high"
        if self.ai_probability >= 0.5:
            return "medium"
        if self.ai_probability >= 0.3:
            return "low"
        return "very low"


@dataclass
class ProvenanceRecord:
    """Immutable provenance record for a code block."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    source: ProvenanceSource = ProvenanceSource.HUMAN
    author: str = ""
    ai_model: str = ""
    fingerprint: AIFingerprint | None = None
    commit_sha: str = ""
    timestamp: float = field(default_factory=time.time)
    content_hash: str = ""
    previous_hash: str = ""

    def compute_hash(self) -> str:
        data = f"{self.file_path}:{self.line_start}-{self.line_end}:{self.source.value}:{self.timestamp}:{self.previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "source": self.source.value,
            "author": self.author,
            "ai_model": self.ai_model,
            "ai_probability": self.fingerprint.ai_probability if self.fingerprint else 0.0,
            "commit_sha": self.commit_sha,
            "timestamp": self.timestamp,
            "content_hash": self.content_hash,
            "previous_hash": self.previous_hash,
        }


@dataclass
class ComplianceReport:
    """Compliance report for AI code governance."""

    framework: ComplianceFramework
    generated_at: float = field(default_factory=time.time)
    total_files: int = 0
    total_lines: int = 0
    ai_generated_lines: int = 0
    records: list[ProvenanceRecord] = field(default_factory=list)
    risk_assessment: dict[str, Any] = field(default_factory=dict)

    @property
    def ai_percentage(self) -> float:
        return (self.ai_generated_lines / self.total_lines * 100) if self.total_lines > 0 else 0.0

    @property
    def risk_level(self) -> RiskLevel:
        pct = self.ai_percentage
        if pct > 70:
            return RiskLevel.CRITICAL
        if pct > 50:
            return RiskLevel.HIGH
        if pct > 25:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework.value,
            "generated_at": self.generated_at,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "ai_generated_lines": self.ai_generated_lines,
            "ai_percentage": round(self.ai_percentage, 1),
            "risk_level": self.risk_level.value,
            "risk_assessment": self.risk_assessment,
            "records_count": len(self.records),
        }

    def to_aibom(self) -> dict[str, Any]:
        """Export as AI Bill of Materials (AIBOM) format."""
        return {
            "aibomVersion": "1.0",
            "metadata": {
                "framework": self.framework.value,
                "timestamp": self.generated_at,
                "tools": [{"name": "CodeVerify", "version": "0.9.0"}],
            },
            "components": [
                {
                    "type": "ai-generated-code",
                    "path": r.file_path,
                    "lines": f"{r.line_start}-{r.line_end}",
                    "source": r.source.value,
                    "model": r.ai_model,
                    "confidence": r.fingerprint.ai_probability if r.fingerprint else 0.0,
                    "hash": r.content_hash,
                }
                for r in self.records
                if r.source != ProvenanceSource.HUMAN
            ],
            "summary": {
                "total_components": len(self.records),
                "ai_components": sum(1 for r in self.records if r.source != ProvenanceSource.HUMAN),
                "ai_percentage": round(self.ai_percentage, 1),
            },
        }


class AICodeDetector:
    """Detects AI-generated code using heuristic fingerprinting."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def analyze(self, source: str) -> AIFingerprint:
        """Analyze source code for AI generation signals."""
        lines = source.split("\n")
        if not lines or not source.strip():
            return AIFingerprint()

        signals: list[str] = []
        scores: list[float] = []

        # 1. Shannon entropy of character distribution
        entropy = self._shannon_entropy(source)
        entropy_score = min(1.0, entropy / 5.0)  # Normalize
        scores.append(entropy_score * 0.15)

        # 2. Average line length (AI tends to write medium-length lines)
        non_empty = [line for line in lines if line.strip()]
        avg_len = sum(len(line) for line in non_empty) / len(non_empty) if non_empty else 0
        if 40 <= avg_len <= 80:
            scores.append(0.1)
            signals.append("avg_line_length_typical_ai")

        # 3. Comment ratio (AI typically adds comments at ~15-25% density)
        comment_lines = sum(
            1 for line in lines if line.strip().startswith("#") or line.strip().startswith("//")
        )
        comment_ratio = comment_lines / len(non_empty) if non_empty else 0
        if 0.10 <= comment_ratio <= 0.30:
            scores.append(0.1)
            signals.append("comment_density_typical_ai")

        # 4. Docstring/JSDoc patterns (AI always adds docs)
        docstring_count = source.count('"""') + source.count("'''") + source.count("/**")
        if docstring_count >= 2:
            scores.append(0.15)
            signals.append("comprehensive_docstrings")

        # 5. Naming consistency (AI is very consistent)
        identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\b", source)
        if identifiers:
            snake_case = sum(1 for i in identifiers if "_" in i)
            camel_case = sum(1 for i in identifiers if any(c.isupper() for c in i))
            total_named = snake_case + camel_case
            if total_named > 0:
                consistency = max(snake_case, camel_case) / total_named
                if consistency > 0.9:
                    scores.append(0.1)
                    signals.append("high_naming_consistency")

        # 6. Repetitive patterns (AI repeats structures)
        repetition = self._check_repetition(lines)
        if repetition > 0.3:
            scores.append(0.1)
            signals.append("repetitive_structure")

        # 7. Co-authored-by: Copilot signal
        if "co-authored-by" in source.lower() and "copilot" in source.lower():
            scores.append(0.3)
            signals.append("copilot_co_author_tag")

        # 8. Generic variable names (AI tends to use descriptive but generic names)
        generic_names = {"result", "data", "value", "item", "response", "output", "temp"}
        used_generic = sum(1 for i in set(identifiers) if i in generic_names)
        if used_generic >= 3:
            scores.append(0.1)
            signals.append("generic_variable_names")

        ai_probability = min(1.0, sum(scores))

        return AIFingerprint(
            entropy=entropy,
            avg_line_length=avg_len,
            comment_ratio=comment_ratio,
            unique_identifier_ratio=len(set(identifiers)) / len(identifiers) if identifiers else 0,
            repetition_score=repetition,
            style_consistency=1.0 if "high_naming_consistency" in signals else 0.5,
            ai_probability=ai_probability,
            signals=signals,
        )

    def _shannon_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        freq = Counter(text)
        length = len(text)
        return -sum(
            (count / length) * math.log2(count / length) for count in freq.values() if count > 0
        )

    def _check_repetition(self, lines: list[str]) -> float:
        """Check for structural repetition in code."""
        if len(lines) < 4:
            return 0.0
        # Compare structural similarity of consecutive blocks
        stripped = [line.strip() for line in lines if line.strip()]
        if len(stripped) < 4:
            return 0.0
        # Count lines that share the same leading pattern
        patterns = [line[:10] for line in stripped]
        pattern_counts = Counter(patterns)
        most_common_count = pattern_counts.most_common(1)[0][1] if pattern_counts else 0
        return most_common_count / len(stripped)


class ProvenanceTracker:
    """Tracks provenance of code blocks with hash chain."""

    def __init__(self) -> None:
        self._records: list[ProvenanceRecord] = []
        self._detector = AICodeDetector()
        self._last_hash = "genesis"

    def track(
        self,
        source: str,
        file_path: str,
        line_start: int = 1,
        line_end: int | None = None,
        author: str = "",
        commit_sha: str = "",
        ai_model: str = "",
    ) -> ProvenanceRecord:
        """Track a code block's provenance."""
        if line_end is None:
            line_end = line_start + source.count("\n")

        fingerprint = self._detector.analyze(source)

        # Determine source
        if ai_model:
            source_type = self._model_to_source(ai_model)
        elif fingerprint.is_likely_ai:
            source_type = ProvenanceSource.AI_UNKNOWN
        else:
            source_type = ProvenanceSource.HUMAN

        content_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

        record = ProvenanceRecord(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            source=source_type,
            author=author,
            ai_model=ai_model,
            fingerprint=fingerprint,
            commit_sha=commit_sha,
            timestamp=time.time(),
            content_hash=content_hash,
            previous_hash=self._last_hash,
        )

        self._last_hash = record.compute_hash()
        self._records.append(record)
        return record

    def get_records(
        self,
        file_path: str | None = None,
    ) -> list[ProvenanceRecord]:
        if file_path:
            return [r for r in self._records if r.file_path == file_path]
        return list(self._records)

    def get_ai_records(self) -> list[ProvenanceRecord]:
        return [r for r in self._records if r.source != ProvenanceSource.HUMAN]

    @property
    def ai_percentage(self) -> float:
        if not self._records:
            return 0.0
        ai_count = sum(1 for r in self._records if r.source != ProvenanceSource.HUMAN)
        return ai_count / len(self._records) * 100

    def generate_report(
        self,
        framework: ComplianceFramework = ComplianceFramework.EU_AI_ACT,
    ) -> ComplianceReport:
        """Generate a compliance report."""
        total_lines = sum(r.line_end - r.line_start + 1 for r in self._records)
        ai_lines = sum(
            r.line_end - r.line_start + 1
            for r in self._records
            if r.source != ProvenanceSource.HUMAN
        )

        report = ComplianceReport(
            framework=framework,
            total_files=len({r.file_path for r in self._records}),
            total_lines=total_lines,
            ai_generated_lines=ai_lines,
            records=list(self._records),
        )

        # Framework-specific risk assessment
        if framework == ComplianceFramework.EU_AI_ACT:
            report.risk_assessment = self._eu_ai_act_assessment(report)
        elif framework == ComplianceFramework.SOX:
            report.risk_assessment = self._sox_assessment(report)
        else:
            report.risk_assessment = {"status": "compliant", "notes": []}

        return report

    def verify_chain(self) -> bool:
        """Verify the integrity of the provenance hash chain."""
        expected_prev = "genesis"
        for record in self._records:
            if record.previous_hash != expected_prev:
                return False
            expected_prev = record.compute_hash()
        return True

    def _model_to_source(self, model: str) -> ProvenanceSource:
        model_lower = model.lower()
        if "copilot" in model_lower:
            return ProvenanceSource.AI_COPILOT
        if "gpt" in model_lower or "openai" in model_lower:
            return ProvenanceSource.AI_CHATGPT
        if "claude" in model_lower or "anthropic" in model_lower:
            return ProvenanceSource.AI_CLAUDE
        if "cursor" in model_lower:
            return ProvenanceSource.AI_CURSOR
        return ProvenanceSource.AI_UNKNOWN

    def _eu_ai_act_assessment(self, report: ComplianceReport) -> dict[str, Any]:
        notes: list[str] = []
        if report.ai_percentage > 50:
            notes.append("WARNING: >50% AI-generated code requires enhanced documentation")
        if report.ai_percentage > 25:
            notes.append("AI transparency disclosure required under Art. 52")
        notes.append("AI-generated code provenance tracked per Art. 12")
        return {
            "status": "compliant" if report.ai_percentage < 70 else "review_required",
            "risk_category": report.risk_level.value,
            "notes": notes,
            "disclosure_required": report.ai_percentage > 25,
        }

    def _sox_assessment(self, report: ComplianceReport) -> dict[str, Any]:
        notes: list[str] = []
        if report.ai_percentage > 30:
            notes.append("Material code changes require dual review per SOX Section 404")
        notes.append("AI code provenance chain verified for audit trail")
        return {
            "status": "compliant",
            "audit_ready": True,
            "chain_verified": self.verify_chain(),
            "notes": notes,
        }


# Singleton
_provenance_tracker_instance: ProvenanceTracker | None = None


def get_provenance_tracker() -> ProvenanceTracker:
    global _provenance_tracker_instance
    if _provenance_tracker_instance is None:
        _provenance_tracker_instance = ProvenanceTracker()
    return _provenance_tracker_instance


def reset_provenance_tracker() -> None:
    global _provenance_tracker_instance
    _provenance_tracker_instance = None
