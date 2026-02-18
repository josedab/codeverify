"""Runtime Verification Bridge.

Translates Z3 constraints into lightweight runtime assertions,
captures violations in production, and feeds runtime data back
to improve static analysis accuracy.

Features:
- Z3 constraint to runtime assertion translation (Python decorators, TS checks)
- Configurable assertion density and sampling rate
- Runtime violation capture with execution context serialization
- Feedback loop: confirmed violations boost confidence, non-violations reduce FPs
- Performance-aware instrumentation (<1ms overhead target)
- Multi-language assertion generation (Python, TypeScript, Go)
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AssertionLanguage(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"


class ViolationSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FeedbackEffect(str, Enum):
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    INCONCLUSIVE = "inconclusive"


@dataclass
class RuntimeAssertion:
    """A runtime assertion derived from a Z3 constraint."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    constraint_id: str = ""
    check_type: str = ""
    function_name: str = ""
    file_path: str = ""
    language: AssertionLanguage = AssertionLanguage.PYTHON
    assertion_code: str = ""
    original_z3: str = ""
    variables: list[str] = field(default_factory=list)
    is_active: bool = True
    sample_rate: float = 1.0


@dataclass
class RuntimeViolation:
    """A captured runtime assertion violation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    assertion_id: str = ""
    function_name: str = ""
    file_path: str = ""
    variable_values: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    stack_trace: str = ""
    severity: ViolationSeverity = ViolationSeverity.MEDIUM
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    environment: str = "production"


@dataclass
class FeedbackRecord:
    """Feedback from runtime data improving static analysis."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    constraint_id: str = ""
    effect: FeedbackEffect = FeedbackEffect.INCONCLUSIVE
    static_prediction: str = ""
    runtime_outcome: str = ""
    confidence_delta: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class InstrumentationConfig:
    """Configuration for runtime instrumentation."""

    enabled: bool = True
    sample_rate: float = 1.0
    max_assertions_per_function: int = 5
    overhead_budget_ms: float = 1.0
    report_endpoint: str = ""
    batch_size: int = 100
    environments: list[str] = field(default_factory=lambda: ["staging", "production"])


@dataclass
class BridgeStats:
    """Statistics for the runtime verification bridge."""

    assertions_generated: int = 0
    violations_captured: int = 0
    confirmed_bugs: int = 0
    false_positives_detected: int = 0
    avg_overhead_ms: float = 0.0
    feedback_records: int = 0


class AssertionTranslator:
    """Translates Z3 constraints to runtime assertions."""

    PYTHON_TEMPLATES: dict[str, str] = {
        "null_safety": "assert {var} is not None, 'CodeVerify: {var} must not be None at {func}'",
        "division_by_zero": "assert {var} != 0, 'CodeVerify: {var} must not be zero at {func}'",
        "array_bounds": "assert 0 <= {var} < len({array}), 'CodeVerify: {var} out of bounds at {func}'",
        "integer_overflow": "assert abs({var}) <= 2**31 - 1, 'CodeVerify: {var} may overflow at {func}'",
        "positive": "assert {var} > 0, 'CodeVerify: {var} must be positive at {func}'",
        "non_negative": "assert {var} >= 0, 'CodeVerify: {var} must be non-negative at {func}'",
    }

    TS_TEMPLATES: dict[str, str] = {
        "null_safety": "if ({var} === null || {var} === undefined) throw new Error('CodeVerify: {var} must not be null at {func}');",
        "division_by_zero": "if ({var} === 0) throw new Error('CodeVerify: {var} must not be zero at {func}');",
        "array_bounds": "if ({var} < 0 || {var} >= {array}.length) throw new RangeError('CodeVerify: {var} out of bounds at {func}');",
    }

    GO_TEMPLATES: dict[str, str] = {
        "null_safety": 'if {var} == nil {{ panic("CodeVerify: {var} must not be nil at {func}") }}',
        "division_by_zero": 'if {var} == 0 {{ panic("CodeVerify: {var} must not be zero at {func}") }}',
    }

    def translate(
        self,
        check_type: str,
        function_name: str,
        file_path: str,
        variables: list[str],
        language: AssertionLanguage = AssertionLanguage.PYTHON,
        z3_assertion: str = "",
    ) -> list[RuntimeAssertion]:
        """Translate a Z3 constraint into runtime assertions."""
        templates = {
            AssertionLanguage.PYTHON: self.PYTHON_TEMPLATES,
            AssertionLanguage.TYPESCRIPT: self.TS_TEMPLATES,
            AssertionLanguage.GO: self.GO_TEMPLATES,
        }.get(language, self.PYTHON_TEMPLATES)

        template = templates.get(check_type)
        if not template:
            return []

        assertions: list[RuntimeAssertion] = []
        for var in variables:
            code = template.format(var=var, func=function_name, array="data")
            assertions.append(RuntimeAssertion(
                constraint_id=hashlib.sha256(f"{check_type}:{var}:{function_name}".encode()).hexdigest()[:8],
                check_type=check_type,
                function_name=function_name,
                file_path=file_path,
                language=language,
                assertion_code=code,
                original_z3=z3_assertion,
                variables=[var],
            ))
        return assertions

    def generate_decorator(self, assertions: list[RuntimeAssertion]) -> str:
        """Generate a Python decorator that runs all assertions."""
        if not assertions:
            return ""
        checks = "\n        ".join(a.assertion_code for a in assertions)
        return (
            "def codeverify_check(func):\n"
            "    def wrapper(*args, **kwargs):\n"
            f"        {checks}\n"
            "        return func(*args, **kwargs)\n"
            "    return wrapper"
        )


class ViolationCollector:
    """Captures and batches runtime violations."""

    def __init__(self, config: InstrumentationConfig | None = None) -> None:
        self._config = config or InstrumentationConfig()
        self._violations: list[RuntimeViolation] = []
        self._batch: list[RuntimeViolation] = []

    def capture(
        self,
        assertion: RuntimeAssertion,
        variable_values: dict[str, Any],
        error_message: str = "",
        stack_trace: str = "",
    ) -> RuntimeViolation:
        """Capture a runtime violation."""
        violation = RuntimeViolation(
            assertion_id=assertion.id,
            function_name=assertion.function_name,
            file_path=assertion.file_path,
            variable_values=variable_values,
            error_message=error_message or f"Assertion failed: {assertion.check_type}",
            stack_trace=stack_trace,
            severity=self._classify_severity(assertion.check_type),
        )
        self._violations.append(violation)
        self._batch.append(violation)

        if len(self._batch) >= self._config.batch_size:
            self._flush_batch()

        return violation

    def get_violations(self, assertion_id: str | None = None) -> list[RuntimeViolation]:
        if assertion_id:
            return [v for v in self._violations if v.assertion_id == assertion_id]
        return list(self._violations)

    def _classify_severity(self, check_type: str) -> ViolationSeverity:
        severity_map = {
            "null_safety": ViolationSeverity.HIGH,
            "division_by_zero": ViolationSeverity.CRITICAL,
            "array_bounds": ViolationSeverity.HIGH,
            "integer_overflow": ViolationSeverity.MEDIUM,
        }
        return severity_map.get(check_type, ViolationSeverity.MEDIUM)

    def _flush_batch(self) -> None:
        logger.info("violations_batch_flushed", count=len(self._batch))
        self._batch = []


class FeedbackEngine:
    """Feeds runtime results back to improve static analysis."""

    def __init__(self) -> None:
        self._records: list[FeedbackRecord] = []
        self._confidence_adjustments: dict[str, float] = {}

    def process_violation(
        self, constraint_id: str, static_severity: str, runtime_confirmed: bool
    ) -> FeedbackRecord:
        """Process a runtime result and generate feedback."""
        if runtime_confirmed:
            effect = FeedbackEffect.CONFIRMED
            delta = 0.15
        else:
            effect = FeedbackEffect.FALSE_POSITIVE
            delta = -0.2

        record = FeedbackRecord(
            constraint_id=constraint_id,
            effect=effect,
            static_prediction=static_severity,
            runtime_outcome="violation_confirmed" if runtime_confirmed else "no_violation",
            confidence_delta=delta,
        )
        self._records.append(record)

        current = self._confidence_adjustments.get(constraint_id, 0.0)
        self._confidence_adjustments[constraint_id] = round(current + delta, 3)

        return record

    def get_confidence_adjustment(self, constraint_id: str) -> float:
        return self._confidence_adjustments.get(constraint_id, 0.0)

    def get_records(self) -> list[FeedbackRecord]:
        return list(self._records)

    def get_false_positive_rate(self) -> float:
        if not self._records:
            return 0.0
        fp = sum(1 for r in self._records if r.effect == FeedbackEffect.FALSE_POSITIVE)
        return round(fp / len(self._records), 3)


class RuntimeVerificationBridgeService:
    """Main service for the runtime verification bridge."""

    def __init__(self, config: InstrumentationConfig | None = None) -> None:
        self._config = config or InstrumentationConfig()
        self._translator = AssertionTranslator()
        self._collector = ViolationCollector(self._config)
        self._feedback = FeedbackEngine()
        self._assertions: dict[str, RuntimeAssertion] = {}

    @property
    def feedback(self) -> FeedbackEngine:
        return self._feedback

    def instrument_function(
        self,
        check_type: str,
        function_name: str,
        file_path: str,
        variables: list[str],
        language: AssertionLanguage = AssertionLanguage.PYTHON,
        z3_assertion: str = "",
    ) -> list[RuntimeAssertion]:
        """Generate runtime assertions for a function."""
        assertions = self._translator.translate(
            check_type, function_name, file_path, variables, language, z3_assertion
        )
        for a in assertions[:self._config.max_assertions_per_function]:
            a.sample_rate = self._config.sample_rate
            self._assertions[a.id] = a
        return assertions

    def report_violation(
        self,
        assertion_id: str,
        variable_values: dict[str, Any],
        error_message: str = "",
    ) -> RuntimeViolation | None:
        """Report a runtime assertion violation."""
        assertion = self._assertions.get(assertion_id)
        if not assertion:
            return None
        violation = self._collector.capture(assertion, variable_values, error_message)
        self._feedback.process_violation(assertion.constraint_id, "medium", runtime_confirmed=True)
        return violation

    def report_pass(self, assertion_id: str) -> FeedbackRecord | None:
        """Report that a runtime assertion passed (expected violation didn't occur)."""
        assertion = self._assertions.get(assertion_id)
        if not assertion:
            return None
        return self._feedback.process_violation(assertion.constraint_id, "medium", runtime_confirmed=False)

    def generate_instrumented_code(
        self, function_name: str, file_path: str
    ) -> str:
        """Generate instrumented code with all assertions for a function."""
        func_assertions = [
            a for a in self._assertions.values()
            if a.function_name == function_name and a.file_path == file_path
        ]
        return self._translator.generate_decorator(func_assertions)

    def get_stats(self) -> BridgeStats:
        violations = self._collector.get_violations()
        records = self._feedback.get_records()
        confirmed = sum(1 for r in records if r.effect == FeedbackEffect.CONFIRMED)
        fps = sum(1 for r in records if r.effect == FeedbackEffect.FALSE_POSITIVE)
        return BridgeStats(
            assertions_generated=len(self._assertions),
            violations_captured=len(violations),
            confirmed_bugs=confirmed,
            false_positives_detected=fps,
            feedback_records=len(records),
        )

    def get_assertion(self, assertion_id: str) -> RuntimeAssertion | None:
        return self._assertions.get(assertion_id)


# ─── Singleton Access ──────────────────────────────────────────────────

_runtime_bridge_instance: RuntimeVerificationBridgeService | None = None

def get_runtime_bridge_service() -> RuntimeVerificationBridgeService:
    global _runtime_bridge_instance
    if _runtime_bridge_instance is None:
        _runtime_bridge_instance = RuntimeVerificationBridgeService()
    return _runtime_bridge_instance

def reset_runtime_bridge_service() -> None:
    global _runtime_bridge_instance
    _runtime_bridge_instance = None
