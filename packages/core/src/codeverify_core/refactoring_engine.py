"""AI-Powered Refactoring Suggestions — Detect code smells and generate plans.

Provides automated code smell detection using regex-based heuristics,
complexity analysis, and structured refactoring plan generation.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()

# =============================================================================
# Enums
# =============================================================================


class SmellType(str, Enum):
    """Category of detected code smell."""

    GOD_CLASS = "god_class"
    LONG_METHOD = "long_method"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMP = "data_clump"
    SHOTGUN_SURGERY = "shotgun_surgery"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    DEAD_CODE = "dead_code"
    DUPLICATED_CODE = "duplicated_code"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    MIDDLE_MAN = "middle_man"


class RefactoringType(str, Enum):
    """Type of refactoring action."""

    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    INLINE_METHOD = "inline_method"
    MOVE_METHOD = "move_method"
    RENAME = "rename"
    INTRODUCE_PARAMETER_OBJECT = "introduce_parameter_object"
    REPLACE_CONDITIONAL = "replace_conditional"
    DECOMPOSE_CONDITIONAL = "decompose_conditional"
    PULL_UP_METHOD = "pull_up_method"
    ENCAPSULATE_FIELD = "encapsulate_field"


class RefactoringRisk(str, Enum):
    """Risk level associated with a refactoring plan."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RefactoringStatus(str, Enum):
    """Lifecycle status of a refactoring plan."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


# =============================================================================
# Data models
# =============================================================================


@dataclass
class CodeSmell:
    """A detected code smell in the analysed source."""

    id: str
    smell_type: SmellType
    file_path: str
    line_start: int
    line_end: int
    description: str
    severity: float
    confidence: float
    affected_symbols: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "smell_type": self.smell_type.value,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "description": self.description,
            "severity": round(self.severity, 2),
            "confidence": round(self.confidence, 2),
            "affected_symbols": self.affected_symbols,
            "metrics": {k: round(v, 2) for k, v in self.metrics.items()},
        }


@dataclass
class ComplexityMetrics:
    """Aggregated complexity measurements for a code unit."""

    cyclomatic_complexity: int
    cognitive_complexity: int
    nesting_depth: int
    lines_of_code: int
    parameter_count: int
    dependency_count: int
    coupling_score: float
    cohesion_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "nesting_depth": self.nesting_depth,
            "lines_of_code": self.lines_of_code,
            "parameter_count": self.parameter_count,
            "dependency_count": self.dependency_count,
            "coupling_score": round(self.coupling_score, 2),
            "cohesion_score": round(self.cohesion_score, 2),
        }


@dataclass
class RefactoringStep:
    """A single atomic step inside a refactoring plan."""

    order: int
    description: str
    file_path: str
    original_code: str
    refactored_code: str
    refactoring_type: RefactoringType

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "description": self.description,
            "file_path": self.file_path,
            "original_code": self.original_code,
            "refactored_code": self.refactored_code,
            "refactoring_type": self.refactoring_type.value,
        }


@dataclass
class RefactoringPlan:
    """A complete refactoring plan targeting one or more code smells."""

    id: str
    name: str
    description: str
    target_smells: list[str]
    steps: list[RefactoringStep]
    risk: RefactoringRisk
    status: RefactoringStatus = RefactoringStatus.PROPOSED
    estimated_improvement: dict[str, float] = field(default_factory=dict)
    affected_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "target_smells": self.target_smells,
            "steps": [s.to_dict() for s in self.steps],
            "risk": self.risk.value,
            "status": self.status.value,
            "estimated_improvement": {
                k: round(v, 2) for k, v in self.estimated_improvement.items()
            },
            "affected_files": self.affected_files,
            "dependencies": self.dependencies,
        }


@dataclass
class RefactoringReport:
    """Top-level report produced by the refactoring engine."""

    project_path: str
    smells_detected: list[CodeSmell]
    plans_generated: list[RefactoringPlan]
    overall_metrics: ComplexityMetrics
    technical_debt_score: float
    improvement_potential: float
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_path": self.project_path,
            "smells_detected": [s.to_dict() for s in self.smells_detected],
            "plans_generated": [p.to_dict() for p in self.plans_generated],
            "overall_metrics": self.overall_metrics.to_dict(),
            "technical_debt_score": round(self.technical_debt_score, 2),
            "improvement_potential": round(self.improvement_potential, 2),
            "recommendations": self.recommendations,
        }


# =============================================================================
# Thresholds
# =============================================================================

_LONG_METHOD_LINES = 30
_GOD_CLASS_METHODS = 10
_GOD_CLASS_LINES = 300
_HIGH_CYCLOMATIC = 10
_DUPLICATE_MIN_LINES = 4
_MAX_PARAMS = 5


# =============================================================================
# Code smell detection
# =============================================================================


class CodeSmellDetector:
    """Regex-based heuristic detector for common code smells."""

    def __init__(self) -> None:
        self._detectors = [
            self._detect_long_method,
            self._detect_god_class,
            self._detect_duplicated_code,
            self._detect_dead_code,
            self._detect_feature_envy,
        ]

    def detect_smells(self, code: str, file_path: str, language: str = "python") -> list[CodeSmell]:
        """Run all detectors against *code* and return discovered smells."""
        smells: list[CodeSmell] = []
        for detector in self._detectors:
            try:
                smells.extend(detector(code, file_path))
            except Exception:
                logger.warning(
                    "smell_detector_failed", detector=detector.__name__, file_path=file_path
                )
        logger.info("smells_detected", file_path=file_path, count=len(smells), language=language)
        return smells

    def _detect_long_method(self, code: str, file_path: str) -> list[CodeSmell]:
        """Detect functions/methods exceeding the line threshold."""
        smells: list[CodeSmell] = []
        pattern = re.compile(r"^( *)(?:def|async\s+def)\s+(\w+)\s*\(", re.MULTILINE)
        lines = code.splitlines()
        for match in pattern.finditer(code):
            indent = len(match.group(1))
            func_name = match.group(2)
            start_line = code[: match.start()].count("\n") + 1
            end_line = start_line
            for i in range(start_line, len(lines)):
                stripped = lines[i].rstrip()
                if stripped == "":
                    continue
                cur_indent = len(lines[i]) - len(lines[i].lstrip())
                if i > start_line and cur_indent <= indent and stripped:
                    break
                end_line = i + 1
            length = end_line - start_line + 1
            if length > _LONG_METHOD_LINES:
                smells.append(
                    CodeSmell(
                        id=str(uuid.uuid4()),
                        smell_type=SmellType.LONG_METHOD,
                        file_path=file_path,
                        line_start=start_line,
                        line_end=end_line,
                        description=f"Method '{func_name}' is {length} lines (threshold: {_LONG_METHOD_LINES})",
                        severity=min(1.0, length / (_LONG_METHOD_LINES * 3)),
                        confidence=0.9,
                        affected_symbols=[func_name],
                        metrics={"lines": float(length)},
                    )
                )
        return smells

    def _detect_god_class(self, code: str, file_path: str) -> list[CodeSmell]:
        """Detect classes with too many methods or lines."""
        smells: list[CodeSmell] = []
        class_pat = re.compile(r"^class\s+(\w+)\s*[\(:]", re.MULTILINE)
        method_pat = re.compile(r"^\s+(?:def|async\s+def)\s+(\w+)\s*\(", re.MULTILINE)
        lines = code.splitlines()
        class_matches = list(class_pat.finditer(code))
        for idx, match in enumerate(class_matches):
            class_name = match.group(1)
            start_line = code[: match.start()].count("\n") + 1
            if idx + 1 < len(class_matches):
                class_body = code[match.start() : class_matches[idx + 1].start()]
                end_line = start_line + class_body.count("\n")
            else:
                class_body = code[match.start() :]
                end_line = len(lines)
            class_lines = end_line - start_line + 1
            methods = method_pat.findall(class_body)
            method_count = len(methods)
            if method_count > _GOD_CLASS_METHODS or class_lines > _GOD_CLASS_LINES:
                severity = min(
                    1.0,
                    max(
                        method_count / (_GOD_CLASS_METHODS * 2),
                        class_lines / (_GOD_CLASS_LINES * 2),
                    ),
                )
                smells.append(
                    CodeSmell(
                        id=str(uuid.uuid4()),
                        smell_type=SmellType.GOD_CLASS,
                        file_path=file_path,
                        line_start=start_line,
                        line_end=end_line,
                        description=f"Class '{class_name}' has {method_count} methods and {class_lines} lines",
                        severity=severity,
                        confidence=0.85,
                        affected_symbols=[class_name] + methods,
                        metrics={
                            "method_count": float(method_count),
                            "class_lines": float(class_lines),
                        },
                    )
                )
        return smells

    def _detect_duplicated_code(self, code: str, file_path: str) -> list[CodeSmell]:
        """Detect duplicated blocks using line-hash fingerprinting."""
        smells: list[CodeSmell] = []
        lines = code.splitlines()
        normalised = [re.sub(r"\s+", " ", ln.strip()) for ln in lines]
        fingerprints: dict[str, list[int]] = {}
        for i in range(len(normalised) - _DUPLICATE_MIN_LINES + 1):
            window = "\n".join(normalised[i : i + _DUPLICATE_MIN_LINES])
            if len(window.strip()) < 20:
                continue
            digest = hashlib.md5(window.encode()).hexdigest()  # noqa: S324
            fingerprints.setdefault(digest, []).append(i + 1)
        reported: set[str] = set()
        for _digest, positions in fingerprints.items():
            if len(positions) < 2:
                continue
            key_str = str(tuple(sorted(positions)))
            if key_str in reported:
                continue
            reported.add(key_str)
            smells.append(
                CodeSmell(
                    id=str(uuid.uuid4()),
                    smell_type=SmellType.DUPLICATED_CODE,
                    file_path=file_path,
                    line_start=positions[0],
                    line_end=positions[0] + _DUPLICATE_MIN_LINES - 1,
                    description=f"Duplicated block ({_DUPLICATE_MIN_LINES}+ lines) at lines {positions}",
                    severity=min(1.0, len(positions) * 0.3),
                    confidence=0.8,
                    metrics={
                        "duplicate_count": float(len(positions)),
                        "block_size": float(_DUPLICATE_MIN_LINES),
                    },
                )
            )
        return smells

    def _detect_dead_code(self, code: str, file_path: str) -> list[CodeSmell]:
        """Detect potentially unused private functions/methods."""
        smells: list[CodeSmell] = []
        def_pat = re.compile(r"^( *)(?:def|async\s+def)\s+(_[a-zA-Z]\w*)\s*\(", re.MULTILINE)
        for match in def_pat.finditer(code):
            name = match.group(2)
            if name.startswith("__") and name.endswith("__"):
                continue
            references = len(re.findall(r"(?<!\bdef\s)" + re.escape(name), code)) - 1
            if references <= 0:
                start_line = code[: match.start()].count("\n") + 1
                smells.append(
                    CodeSmell(
                        id=str(uuid.uuid4()),
                        smell_type=SmellType.DEAD_CODE,
                        file_path=file_path,
                        line_start=start_line,
                        line_end=start_line,
                        description=f"Private function '{name}' appears unused",
                        severity=0.4,
                        confidence=0.6,
                        affected_symbols=[name],
                        metrics={"references": 0.0},
                    )
                )
        return smells

    def _detect_feature_envy(self, code: str, file_path: str) -> list[CodeSmell]:
        """Detect methods that reference external objects more than self."""
        smells: list[CodeSmell] = []
        func_pat = re.compile(r"^( *)(?:def|async\s+def)\s+(\w+)\s*\(self[^)]*\)", re.MULTILINE)
        lines = code.splitlines()
        for match in func_pat.finditer(code):
            indent = len(match.group(1))
            func_name = match.group(2)
            start_line = code[: match.start()].count("\n") + 1
            body_lines: list[str] = []
            for i in range(start_line, len(lines)):
                stripped = lines[i].rstrip()
                if stripped == "":
                    body_lines.append(stripped)
                    continue
                cur_indent = len(lines[i]) - len(lines[i].lstrip())
                if i > start_line and cur_indent <= indent and stripped:
                    break
                body_lines.append(stripped)
            body = "\n".join(body_lines)
            self_refs = len(re.findall(r"\bself\.\w+", body))
            other_refs = len(re.findall(r"\b(?!self\b)[a-z_]\w*\.\w+", body))
            if other_refs > self_refs * 2 and other_refs > 4:
                smells.append(
                    CodeSmell(
                        id=str(uuid.uuid4()),
                        smell_type=SmellType.FEATURE_ENVY,
                        file_path=file_path,
                        line_start=start_line,
                        line_end=start_line + len(body_lines) - 1,
                        description=f"Method '{func_name}' references external objects ({other_refs}) more than self ({self_refs})",
                        severity=min(1.0, other_refs / 15),
                        confidence=0.7,
                        affected_symbols=[func_name],
                        metrics={
                            "self_references": float(self_refs),
                            "external_references": float(other_refs),
                        },
                    )
                )
        return smells


# =============================================================================
# Complexity analysis
# =============================================================================


class ComplexityAnalyzer:
    """Regex-based complexity metrics calculator."""

    _BRANCH_KW = re.compile(r"\b(if|elif|else|for|while|except|with|case|and|or)\b")
    _FLOW_KW = re.compile(r"^\s*(if|elif|else|for|while|try|except|with|match)\b")

    def __init__(self) -> None:
        self._import_pat = re.compile(r"^\s*(?:import|from)\s+(\S+)", re.MULTILINE)
        self._param_pat = re.compile(r"(?:def|async\s+def)\s+\w+\s*\(([^)]*)\)", re.MULTILINE)
        self._attr_pat = re.compile(r"\bself\.(\w+)\b")
        self._method_pat = re.compile(r"^\s+(?:def|async\s+def)\s+(\w+)", re.MULTILINE)

    def analyze(self, code: str, _language: str = "python") -> ComplexityMetrics:
        """Return aggregated complexity metrics for *code*."""
        loc = sum(1 for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#"))
        max_params = 0
        for m in self._param_pat.finditer(code):
            params = [
                p.strip()
                for p in m.group(1).split(",")
                if p.strip() and p.strip() not in ("self", "cls")
            ]
            max_params = max(max_params, len(params))
        return ComplexityMetrics(
            cyclomatic_complexity=self._calculate_cyclomatic(code),
            cognitive_complexity=self._calculate_cognitive(code),
            nesting_depth=self._calculate_nesting_depth(code),
            lines_of_code=loc,
            parameter_count=max_params,
            dependency_count=len(set(self._import_pat.findall(code))),
            coupling_score=self._calculate_coupling(code),
            cohesion_score=self._calculate_cohesion(code),
        )

    def _calculate_cyclomatic(self, code: str) -> int:
        """CC = 1 + count of branching keywords (if, elif, for, while, …)."""
        count = 1
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped.startswith("#"):
                count += len(self._BRANCH_KW.findall(stripped))
        return count

    def _calculate_cognitive(self, code: str) -> int:
        """Approximate cognitive complexity with nesting-level penalties."""
        complexity = 0
        nesting = 0
        prev_indent = 0
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            cur_indent = len(line) - len(line.lstrip())
            if cur_indent > prev_indent:
                nesting += 1
            elif cur_indent < prev_indent:
                nesting = max(0, nesting - (prev_indent - cur_indent) // 4)
            if self._FLOW_KW.match(line):
                complexity += 1 + nesting
            prev_indent = cur_indent
        return complexity

    def _calculate_nesting_depth(self, code: str) -> int:
        """Return the maximum indentation-based nesting depth."""
        max_depth = 0
        base: int | None = None
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if base is None:
                base = indent
            max_depth = max(max_depth, (indent - (base or 0)) // 4)
        return max_depth

    def _calculate_coupling(self, code: str) -> float:
        """Estimate coupling as a 0-1 score based on import and call diversity."""
        imports = set(self._import_pat.findall(code))
        external = set(re.findall(r"\b([A-Z]\w+)\.\w+", code))
        return min(1.0, (len(imports) + len(external)) / 30)

    def _calculate_cohesion(self, code: str) -> float:
        """LCOM-like cohesion score (1 = highly cohesive)."""
        methods = list(self._method_pat.finditer(code))
        all_attrs = set(self._attr_pat.findall(code))
        if not methods or not all_attrs:
            return 1.0
        lines = code.splitlines()
        ratios: list[float] = []
        for i, m in enumerate(methods):
            start = code[: m.start()].count("\n")
            end = code[: methods[i + 1].start()].count("\n") if i + 1 < len(methods) else len(lines)
            body = "\n".join(lines[start:end])
            used = set(self._attr_pat.findall(body))
            ratios.append(len(used & all_attrs) / len(all_attrs))
        return sum(ratios) / len(ratios) if ratios else 1.0


# =============================================================================
# Refactoring planning
# =============================================================================


class RefactoringPlanner:
    """Generate refactoring plans from detected code smells."""

    _SMELL_TO_REFACTORING: dict[SmellType, RefactoringType] = {
        SmellType.LONG_METHOD: RefactoringType.EXTRACT_METHOD,
        SmellType.GOD_CLASS: RefactoringType.EXTRACT_CLASS,
        SmellType.DUPLICATED_CODE: RefactoringType.EXTRACT_METHOD,
        SmellType.FEATURE_ENVY: RefactoringType.MOVE_METHOD,
        SmellType.DEAD_CODE: RefactoringType.INLINE_METHOD,
        SmellType.DATA_CLUMP: RefactoringType.INTRODUCE_PARAMETER_OBJECT,
        SmellType.PRIMITIVE_OBSESSION: RefactoringType.INTRODUCE_PARAMETER_OBJECT,
        SmellType.MIDDLE_MAN: RefactoringType.INLINE_METHOD,
    }

    def __init__(self) -> None:
        self._planners = {
            SmellType.LONG_METHOD: self._plan_extract_method,
            SmellType.GOD_CLASS: self._plan_extract_class,
            SmellType.DUPLICATED_CODE: self._plan_extract_method,
        }

    def create_plan(
        self, smells: list[CodeSmell], code: str, language: str = "python"
    ) -> list[RefactoringPlan]:
        """Create refactoring plans for the given smells."""
        plans: list[RefactoringPlan] = []
        for smell in smells:
            planner = self._planners.get(smell.smell_type)
            try:
                if planner is not None:
                    plans.append(planner(smell, code))
                else:
                    plans.append(self._plan_decompose_conditional(smell, code))
            except Exception:
                logger.warning("plan_creation_failed", smell_type=smell.smell_type.value)
        plans = self._prioritize_plans(plans)
        logger.info("plans_created", count=len(plans), language=language)
        return plans

    def _plan_extract_method(self, smell: CodeSmell, code: str) -> RefactoringPlan:
        """Generate plan to extract a long method into smaller pieces."""
        lines = code.splitlines()
        start = max(0, smell.line_start - 1)
        end = min(len(lines), smell.line_end)
        mid = (start + end) // 2
        original = "\n".join(lines[start:end])
        first_half = "\n".join(lines[start:mid])
        second_half = "\n".join(lines[mid:end])
        func_name = smell.affected_symbols[0] if smell.affected_symbols else "target"
        extracted = f"_extracted_from_{func_name}"
        steps = [
            RefactoringStep(
                order=1,
                description=f"Extract lines {smell.line_start}-{mid + 1} into '{extracted}'",
                file_path=smell.file_path,
                original_code=first_half,
                refactored_code=f"def {extracted}(self):\n    {first_half}",
                refactoring_type=RefactoringType.EXTRACT_METHOD,
            ),
            RefactoringStep(
                order=2,
                description=f"Replace extracted lines with call to '{extracted}'",
                file_path=smell.file_path,
                original_code=original,
                refactored_code=f"self.{extracted}()\n{second_half}",
                refactoring_type=RefactoringType.EXTRACT_METHOD,
            ),
        ]
        risk = RefactoringRisk.MEDIUM if smell.severity > 0.5 else RefactoringRisk.LOW
        return RefactoringPlan(
            id=str(uuid.uuid4()),
            name=f"Extract method from '{func_name}'",
            description=f"Break '{func_name}' into smaller methods to reduce complexity",
            target_smells=[smell.id],
            steps=steps,
            risk=risk,
            affected_files=[smell.file_path],
        )

    def _plan_extract_class(self, smell: CodeSmell, _code: str) -> RefactoringPlan:
        """Generate plan to split a god class into cohesive units."""
        class_name = smell.affected_symbols[0] if smell.affected_symbols else "Target"
        methods = smell.affected_symbols[1:] if len(smell.affected_symbols) > 1 else []
        split = len(methods) // 2 if methods else 0
        keep, move = methods[:split], methods[split:]
        new_class = f"{class_name}Helper"
        steps = [
            RefactoringStep(
                order=1,
                description=f"Create new class '{new_class}'",
                file_path=smell.file_path,
                original_code="",
                refactored_code=f"class {new_class}:\n    pass",
                refactoring_type=RefactoringType.EXTRACT_CLASS,
            )
        ]
        for i, method in enumerate(move):
            steps.append(
                RefactoringStep(
                    order=i + 2,
                    description=f"Move method '{method}' to '{new_class}'",
                    file_path=smell.file_path,
                    original_code=f"def {method}(self",
                    refactored_code=f"# Moved to {new_class}\n# def {method}(self",
                    refactoring_type=RefactoringType.MOVE_METHOD,
                )
            )
        risk = RefactoringRisk.HIGH if smell.severity > 0.7 else RefactoringRisk.MEDIUM
        return RefactoringPlan(
            id=str(uuid.uuid4()),
            name=f"Extract class from '{class_name}'",
            description=f"Split '{class_name}' into '{class_name}' (keeps {len(keep)} methods) and '{new_class}' (receives {len(move)} methods)",
            target_smells=[smell.id],
            steps=steps,
            risk=risk,
            affected_files=[smell.file_path],
        )

    def _plan_decompose_conditional(self, smell: CodeSmell, _code: str) -> RefactoringPlan:
        """Fallback plan that proposes decomposing complex logic."""
        ref_type = self._SMELL_TO_REFACTORING.get(
            smell.smell_type, RefactoringType.DECOMPOSE_CONDITIONAL
        )
        func_name = smell.affected_symbols[0] if smell.affected_symbols else "target"
        step = RefactoringStep(
            order=1,
            description=f"Refactor '{func_name}' to address {smell.smell_type.value}",
            file_path=smell.file_path,
            original_code="",
            refactored_code="",
            refactoring_type=ref_type,
        )
        risk = (
            RefactoringRisk.HIGH
            if smell.severity > 0.7
            else (RefactoringRisk.MEDIUM if smell.severity > 0.4 else RefactoringRisk.LOW)
        )
        return RefactoringPlan(
            id=str(uuid.uuid4()),
            name=f"Refactor '{func_name}' ({smell.smell_type.value})",
            description=smell.description,
            target_smells=[smell.id],
            steps=[step],
            risk=risk,
            affected_files=[smell.file_path],
        )

    def _prioritize_plans(self, plans: list[RefactoringPlan]) -> list[RefactoringPlan]:
        """Sort plans by risk (low first) then by number of target smells."""
        risk_order = {
            RefactoringRisk.LOW: 0,
            RefactoringRisk.MEDIUM: 1,
            RefactoringRisk.HIGH: 2,
            RefactoringRisk.CRITICAL: 3,
        }
        return sorted(plans, key=lambda p: (risk_order.get(p.risk, 99), -len(p.target_smells)))


# =============================================================================
# Refactoring engine (orchestrator)
# =============================================================================


class RefactoringEngine:
    """Top-level orchestrator that ties detection, analysis, and planning."""

    def __init__(self) -> None:
        self.detector = CodeSmellDetector()
        self.analyzer = ComplexityAnalyzer()
        self.planner = RefactoringPlanner()

    def analyze_project(self, files: dict[str, str], language: str = "python") -> RefactoringReport:
        """Analyse all *files* and produce a consolidated refactoring report."""
        all_smells: list[CodeSmell] = []
        all_plans: list[RefactoringPlan] = []
        aggregated_code = ""
        for path, code in files.items():
            smells = self.detector.detect_smells(code, path, language)
            all_smells.extend(smells)
            if smells:
                plans = self.planner.create_plan(smells, code, language)
                all_plans.extend(plans)
            aggregated_code += code + "\n"
        overall = self.analyzer.analyze(aggregated_code, language)
        debt = self.calculate_technical_debt(overall)
        improvements: list[dict[str, float]] = []
        for plan in all_plans:
            est = self.estimate_improvement(plan, overall)
            plan.estimated_improvement = est
            improvements.append(est)
        avg_improvement = 0.0
        if improvements:
            totals: defaultdict[str, float] = defaultdict(float)
            for imp in improvements:
                for k, v in imp.items():
                    totals[k] += v
            avg_improvement = sum(totals.values()) / (len(totals) or 1)
        recommendations = self._generate_recommendations(all_smells, overall, debt)
        logger.info(
            "project_analysis_complete",
            files=len(files),
            smells=len(all_smells),
            plans=len(all_plans),
            debt=round(debt, 2),
        )
        return RefactoringReport(
            project_path=next(iter(files), ""),
            smells_detected=all_smells,
            plans_generated=all_plans,
            overall_metrics=overall,
            technical_debt_score=debt,
            improvement_potential=min(1.0, avg_improvement),
            recommendations=recommendations,
        )

    def apply_plan(self, plan: RefactoringPlan, code: str) -> str:
        """Apply the steps of *plan* to *code* and return the result."""
        result = code
        for step in sorted(plan.steps, key=lambda s: s.order):
            if step.original_code and step.original_code in result:
                result = result.replace(step.original_code, step.refactored_code, 1)
                logger.debug("step_applied", order=step.order, type=step.refactoring_type.value)
        plan.status = RefactoringStatus.COMPLETED
        return result

    def estimate_improvement(
        self, plan: RefactoringPlan, _original_metrics: ComplexityMetrics
    ) -> dict[str, float]:
        """Estimate the percentage improvement a plan would bring."""
        factor = {
            RefactoringRisk.LOW: 0.15,
            RefactoringRisk.MEDIUM: 0.10,
            RefactoringRisk.HIGH: 0.08,
            RefactoringRisk.CRITICAL: 0.05,
        }
        base = factor.get(plan.risk, 0.10) * len(plan.steps)
        improvements: dict[str, float] = {
            "cyclomatic_reduction": min(0.5, base * 1.2),
            "cognitive_reduction": min(0.5, base * 1.0),
            "coupling_reduction": min(0.3, base * 0.5),
        }
        if len(plan.target_smells) > 1:
            improvements = {k: min(1.0, v * 1.3) for k, v in improvements.items()}
        return improvements

    def calculate_technical_debt(self, metrics: ComplexityMetrics) -> float:
        """Return a 0-1 technical debt score derived from complexity metrics."""
        debt = (
            min(1.0, metrics.cyclomatic_complexity / 50) * 0.25
            + min(1.0, metrics.cognitive_complexity / 80) * 0.25
            + min(1.0, metrics.nesting_depth / 8) * 0.15
            + min(1.0, metrics.lines_of_code / 1000) * 0.10
            + metrics.coupling_score * 0.15
            + (1.0 - metrics.cohesion_score) * 0.10
        )
        return min(1.0, max(0.0, debt))

    def _generate_recommendations(
        self, smells: list[CodeSmell], metrics: ComplexityMetrics, debt: float
    ) -> list[str]:
        """Produce human-readable recommendations based on analysis."""
        recs: list[str] = []
        counts: Counter[SmellType] = Counter(s.smell_type for s in smells)
        _SMELL_RECS = {
            SmellType.LONG_METHOD: "Consider extracting helper methods to improve readability.",
            SmellType.GOD_CLASS: "Split responsibilities into focused classes.",
            SmellType.DUPLICATED_CODE: "Extract shared logic into reusable functions.",
            SmellType.DEAD_CODE: "Remove unused code to reduce maintenance burden.",
            SmellType.FEATURE_ENVY: "Consider moving these methods to the class they reference most.",
        }
        for smell_type, advice in _SMELL_RECS.items():
            n = counts.get(smell_type, 0)
            if n > 0:
                recs.append(f"Found {n} {smell_type.value} smell(s). {advice}")
        if metrics.cyclomatic_complexity > _HIGH_CYCLOMATIC:
            recs.append(
                f"Cyclomatic complexity is {metrics.cyclomatic_complexity} (threshold: {_HIGH_CYCLOMATIC}). Simplify control flow."
            )
        if metrics.coupling_score > 0.6:
            recs.append(
                "High coupling detected. Introduce abstractions to reduce direct dependencies."
            )
        if metrics.cohesion_score < 0.4:
            recs.append(
                "Low cohesion detected. Ensure classes have a single, well-defined responsibility."
            )
        if debt > 0.7:
            recs.append(
                f"Technical debt score is {debt:.0%}. Prioritise refactoring to prevent increasing costs."
            )
        if not recs:
            recs.append("Code quality looks good — no major issues detected.")
        return recs
