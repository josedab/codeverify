"""Runtime Verification Probes - Inject Z3 proofs as runtime assertions.

This module bridges the gap between compile-time formal verification and runtime
reality by converting Z3 proofs into lightweight runtime assertions that can be
injected into production code.

Key features:
1. Proof-to-Assert Compiler: Converts Z3 constraints to language-native assertions
2. Instrumentation Engine: Injects probes at strategic code points
3. Runtime Collector: Aggregates violation data with minimal performance impact
4. Feedback Integration: Routes runtime failures back to verification dashboard
"""

import ast
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class ProbeType(str, Enum):
    """Type of runtime probe."""

    NULL_CHECK = "null_check"
    BOUNDS_CHECK = "bounds_check"
    OVERFLOW_CHECK = "overflow_check"
    DIVISION_CHECK = "division_check"
    TYPE_CHECK = "type_check"
    INVARIANT = "invariant"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"


class ProbeMode(str, Enum):
    """Execution mode for probes."""

    ASSERT = "assert"  # Raises AssertionError
    LOG = "log"  # Logs violation but continues
    SAMPLE = "sample"  # Only checks on sample of executions
    DISABLED = "disabled"  # No-op


class ProbeSeverity(str, Enum):
    """Severity of a probe violation."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Z3Constraint:
    """A Z3 constraint to be converted to runtime probe."""

    id: str
    expression: str  # Z3 expression string
    variables: list[str]
    constraint_type: ProbeType
    location: tuple[int, int]  # (line_start, line_end)
    description: str = ""
    severity: ProbeSeverity = ProbeSeverity.MEDIUM

    @classmethod
    def from_z3_proof(cls, proof: dict[str, Any]) -> "Z3Constraint":
        """Create constraint from Z3 verification proof."""
        return cls(
            id=proof.get("id", str(uuid.uuid4())),
            expression=proof.get("expression", ""),
            variables=proof.get("variables", []),
            constraint_type=ProbeType(proof.get("type", "invariant")),
            location=tuple(proof.get("location", [0, 0])),
            description=proof.get("description", ""),
            severity=ProbeSeverity(proof.get("severity", "medium")),
        )


@dataclass
class RuntimeProbe:
    """A runtime probe ready for injection."""

    id: str
    constraint_id: str
    probe_type: ProbeType
    code: str  # The assertion/check code
    location: tuple[int, int]
    mode: ProbeMode = ProbeMode.ASSERT
    sample_rate: float = 1.0  # For SAMPLE mode
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "constraint_id": self.constraint_id,
            "probe_type": self.probe_type.value,
            "code": self.code,
            "location": self.location,
            "mode": self.mode.value,
            "sample_rate": self.sample_rate,
        }


@dataclass
class ProbeViolation:
    """Record of a runtime probe violation."""

    probe_id: str
    constraint_id: str
    timestamp: datetime
    file_path: str
    line_number: int
    variables: dict[str, Any]
    message: str
    stack_trace: str | None = None
    severity: ProbeSeverity = ProbeSeverity.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "constraint_id": self.constraint_id,
            "timestamp": self.timestamp.isoformat(),
            "file_path": self.file_path,
            "line_number": self.line_number,
            "variables": self.variables,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "severity": self.severity.value,
        }


@dataclass
class InstrumentationResult:
    """Result of code instrumentation."""

    success: bool
    instrumented_code: str
    probes_inserted: list[RuntimeProbe]
    error: str | None = None
    original_lines: int = 0
    instrumented_lines: int = 0


class ProofToAssertCompiler:
    """Compiles Z3 constraints to language-native assertions."""

    # Z3 operators to Python equivalents
    Z3_TO_PYTHON = {
        "And": "and",
        "Or": "or",
        "Not": "not",
        "Implies": "implies",
        "==": "==",
        "!=": "!=",
        "<": "<",
        "<=": "<=",
        ">": ">",
        ">=": ">=",
        "+": "+",
        "-": "-",
        "*": "*",
        "/": "/",
        "%": "%",
    }

    def __init__(self, language: str = "python") -> None:
        self.language = language

    def compile(self, constraint: Z3Constraint) -> RuntimeProbe:
        """Compile a Z3 constraint to a runtime probe."""
        if self.language == "python":
            return self._compile_python(constraint)
        elif self.language in ("typescript", "javascript"):
            return self._compile_typescript(constraint)
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def _compile_python(self, constraint: Z3Constraint) -> RuntimeProbe:
        """Compile constraint to Python assertion."""
        # Convert Z3 expression to Python
        python_expr = self._z3_to_python(constraint.expression)

        # Generate probe code based on type
        if constraint.constraint_type == ProbeType.NULL_CHECK:
            code = self._generate_null_check_python(constraint, python_expr)
        elif constraint.constraint_type == ProbeType.BOUNDS_CHECK:
            code = self._generate_bounds_check_python(constraint, python_expr)
        elif constraint.constraint_type == ProbeType.OVERFLOW_CHECK:
            code = self._generate_overflow_check_python(constraint, python_expr)
        elif constraint.constraint_type == ProbeType.DIVISION_CHECK:
            code = self._generate_division_check_python(constraint, python_expr)
        else:
            code = self._generate_generic_check_python(constraint, python_expr)

        return RuntimeProbe(
            id=str(uuid.uuid4()),
            constraint_id=constraint.id,
            probe_type=constraint.constraint_type,
            code=code,
            location=constraint.location,
            metadata={
                "original_expression": constraint.expression,
                "description": constraint.description,
            },
        )

    def _compile_typescript(self, constraint: Z3Constraint) -> RuntimeProbe:
        """Compile constraint to TypeScript assertion."""
        ts_expr = self._z3_to_typescript(constraint.expression)

        if constraint.constraint_type == ProbeType.NULL_CHECK:
            code = self._generate_null_check_ts(constraint, ts_expr)
        elif constraint.constraint_type == ProbeType.BOUNDS_CHECK:
            code = self._generate_bounds_check_ts(constraint, ts_expr)
        else:
            code = self._generate_generic_check_ts(constraint, ts_expr)

        return RuntimeProbe(
            id=str(uuid.uuid4()),
            constraint_id=constraint.id,
            probe_type=constraint.constraint_type,
            code=code,
            location=constraint.location,
            metadata={
                "original_expression": constraint.expression,
                "description": constraint.description,
            },
        )

    def _z3_to_python(self, expr: str) -> str:
        """Convert Z3 expression to Python."""
        result = expr

        # Replace Z3 functions
        result = re.sub(r'\bAnd\s*\(([^)]+)\)', r'(\1)', result)
        result = result.replace(' And ', ' and ')

        result = re.sub(r'\bOr\s*\(([^)]+)\)', r'(\1)', result)
        result = result.replace(' Or ', ' or ')

        result = re.sub(r'\bNot\s*\(([^)]+)\)', r'not (\1)', result)

        result = re.sub(r'\bImplies\s*\(([^,]+),\s*([^)]+)\)',
                        r'(not (\1) or (\2))', result)

        # Handle integer types
        result = re.sub(r'\bInt\s*\(([^)]+)\)', r'\1', result)

        return result

    def _z3_to_typescript(self, expr: str) -> str:
        """Convert Z3 expression to TypeScript."""
        result = self._z3_to_python(expr)

        # Python to TS adjustments
        result = result.replace(' and ', ' && ')
        result = result.replace(' or ', ' || ')
        result = result.replace('not ', '!')
        result = result.replace('None', 'null')
        result = result.replace('True', 'true')
        result = result.replace('False', 'false')

        return result

    def _generate_null_check_python(
        self,
        constraint: Z3Constraint,
        expr: str,
    ) -> str:
        """Generate Python null check probe."""
        var = constraint.variables[0] if constraint.variables else "value"
        return f'''# CodeVerify Runtime Probe: {constraint.id}
if {var} is None:
    _codeverify_probe_violation(
        probe_id="{constraint.id}",
        message="Null safety violation: {var} is None",
        variables={{"var": "{var}", "value": None}},
        severity="{constraint.severity.value}"
    )
'''

    def _generate_bounds_check_python(
        self,
        constraint: Z3Constraint,
        expr: str,
    ) -> str:
        """Generate Python bounds check probe."""
        # Extract array and index from expression
        arr = constraint.variables[0] if len(constraint.variables) > 0 else "arr"
        idx = constraint.variables[1] if len(constraint.variables) > 1 else "idx"

        return f'''# CodeVerify Runtime Probe: {constraint.id}
if not (0 <= {idx} < len({arr})):
    _codeverify_probe_violation(
        probe_id="{constraint.id}",
        message=f"Bounds check violation: index {{{idx}}} out of range for array of length {{len({arr})}}",
        variables={{"array": "{arr}", "index": {idx}, "length": len({arr})}},
        severity="{constraint.severity.value}"
    )
'''

    def _generate_overflow_check_python(
        self,
        constraint: Z3Constraint,
        expr: str,
    ) -> str:
        """Generate Python overflow check probe."""
        var = constraint.variables[0] if constraint.variables else "value"
        max_val = 2**31 - 1  # int32 max

        return f'''# CodeVerify Runtime Probe: {constraint.id}
if abs({var}) > {max_val}:
    _codeverify_probe_violation(
        probe_id="{constraint.id}",
        message=f"Integer overflow: {{{var}}} exceeds 32-bit range",
        variables={{"var": "{var}", "value": {var}}},
        severity="{constraint.severity.value}"
    )
'''

    def _generate_division_check_python(
        self,
        constraint: Z3Constraint,
        expr: str,
    ) -> str:
        """Generate Python division by zero check probe."""
        divisor = constraint.variables[0] if constraint.variables else "divisor"

        return f'''# CodeVerify Runtime Probe: {constraint.id}
if {divisor} == 0:
    _codeverify_probe_violation(
        probe_id="{constraint.id}",
        message="Division by zero violation",
        variables={{"divisor": "{divisor}", "value": {divisor}}},
        severity="{constraint.severity.value}"
    )
'''

    def _generate_generic_check_python(
        self,
        constraint: Z3Constraint,
        expr: str,
    ) -> str:
        """Generate generic Python check probe."""
        return f'''# CodeVerify Runtime Probe: {constraint.id}
if not ({expr}):
    _codeverify_probe_violation(
        probe_id="{constraint.id}",
        message="Invariant violation: {constraint.description}",
        variables={{{", ".join(f'"{v}": {v}' for v in constraint.variables)}}},
        severity="{constraint.severity.value}"
    )
'''

    def _generate_null_check_ts(
        self,
        constraint: Z3Constraint,
        expr: str,
    ) -> str:
        """Generate TypeScript null check probe."""
        var = constraint.variables[0] if constraint.variables else "value"
        return f'''// CodeVerify Runtime Probe: {constraint.id}
if ({var} === null || {var} === undefined) {{
    _codeverifyProbeViolation({{
        probeId: "{constraint.id}",
        message: `Null safety violation: {var} is null/undefined`,
        variables: {{ var: "{var}", value: {var} }},
        severity: "{constraint.severity.value}"
    }});
}}
'''

    def _generate_bounds_check_ts(
        self,
        constraint: Z3Constraint,
        expr: str,
    ) -> str:
        """Generate TypeScript bounds check probe."""
        arr = constraint.variables[0] if len(constraint.variables) > 0 else "arr"
        idx = constraint.variables[1] if len(constraint.variables) > 1 else "idx"

        return f'''// CodeVerify Runtime Probe: {constraint.id}
if ({idx} < 0 || {idx} >= {arr}.length) {{
    _codeverifyProbeViolation({{
        probeId: "{constraint.id}",
        message: `Bounds check violation: index ${{{idx}}} out of range for array of length ${{{arr}.length}}`,
        variables: {{ array: "{arr}", index: {idx}, length: {arr}.length }},
        severity: "{constraint.severity.value}"
    }});
}}
'''

    def _generate_generic_check_ts(
        self,
        constraint: Z3Constraint,
        expr: str,
    ) -> str:
        """Generate generic TypeScript check probe."""
        vars_obj = ", ".join(f'{v}: {v}' for v in constraint.variables)
        return f'''// CodeVerify Runtime Probe: {constraint.id}
if (!({expr})) {{
    _codeverifyProbeViolation({{
        probeId: "{constraint.id}",
        message: "{constraint.description}",
        variables: {{ {vars_obj} }},
        severity: "{constraint.severity.value}"
    }});
}}
'''


class InstrumentationEngine:
    """Injects runtime probes into source code."""

    def __init__(
        self,
        language: str = "python",
        mode: ProbeMode = ProbeMode.LOG,
        sample_rate: float = 1.0,
    ) -> None:
        self.language = language
        self.mode = mode
        self.sample_rate = sample_rate
        self._compiler = ProofToAssertCompiler(language)

    def instrument(
        self,
        code: str,
        constraints: list[Z3Constraint],
        include_runtime: bool = True,
    ) -> InstrumentationResult:
        """Instrument code with runtime probes."""
        if self.language == "python":
            return self._instrument_python(code, constraints, include_runtime)
        elif self.language in ("typescript", "javascript"):
            return self._instrument_typescript(code, constraints, include_runtime)
        else:
            return InstrumentationResult(
                success=False,
                instrumented_code=code,
                probes_inserted=[],
                error=f"Unsupported language: {self.language}",
            )

    def _instrument_python(
        self,
        code: str,
        constraints: list[Z3Constraint],
        include_runtime: bool,
    ) -> InstrumentationResult:
        """Instrument Python code."""
        try:
            lines = code.split("\n")
            probes: list[RuntimeProbe] = []

            # Compile all constraints to probes
            for constraint in constraints:
                probe = self._compiler.compile(constraint)
                probe.mode = self.mode
                probe.sample_rate = self.sample_rate
                probes.append(probe)

            # Sort probes by line number (descending) to insert from bottom up
            probes.sort(key=lambda p: p.location[0], reverse=True)

            # Insert probes
            for probe in probes:
                line_num = probe.location[0]
                if 0 <= line_num < len(lines):
                    # Get indentation of target line
                    target_line = lines[line_num]
                    indent = len(target_line) - len(target_line.lstrip())
                    indent_str = " " * indent

                    # Indent probe code
                    probe_lines = probe.code.split("\n")
                    indented_probe = "\n".join(
                        indent_str + line if line.strip() else line
                        for line in probe_lines
                    )

                    # Insert before target line
                    lines.insert(line_num, indented_probe)

            # Add runtime library if requested
            if include_runtime:
                runtime_code = self._get_python_runtime()
                lines.insert(0, runtime_code)

            instrumented = "\n".join(lines)

            return InstrumentationResult(
                success=True,
                instrumented_code=instrumented,
                probes_inserted=probes,
                original_lines=len(code.split("\n")),
                instrumented_lines=len(instrumented.split("\n")),
            )

        except Exception as e:
            logger.error("Python instrumentation failed", error=str(e))
            return InstrumentationResult(
                success=False,
                instrumented_code=code,
                probes_inserted=[],
                error=str(e),
            )

    def _instrument_typescript(
        self,
        code: str,
        constraints: list[Z3Constraint],
        include_runtime: bool,
    ) -> InstrumentationResult:
        """Instrument TypeScript code."""
        try:
            lines = code.split("\n")
            probes: list[RuntimeProbe] = []

            for constraint in constraints:
                probe = self._compiler.compile(constraint)
                probe.mode = self.mode
                probe.sample_rate = self.sample_rate
                probes.append(probe)

            probes.sort(key=lambda p: p.location[0], reverse=True)

            for probe in probes:
                line_num = probe.location[0]
                if 0 <= line_num < len(lines):
                    target_line = lines[line_num]
                    indent = len(target_line) - len(target_line.lstrip())
                    indent_str = " " * indent

                    probe_lines = probe.code.split("\n")
                    indented_probe = "\n".join(
                        indent_str + line if line.strip() else line
                        for line in probe_lines
                    )

                    lines.insert(line_num, indented_probe)

            if include_runtime:
                runtime_code = self._get_typescript_runtime()
                lines.insert(0, runtime_code)

            instrumented = "\n".join(lines)

            return InstrumentationResult(
                success=True,
                instrumented_code=instrumented,
                probes_inserted=probes,
                original_lines=len(code.split("\n")),
                instrumented_lines=len(instrumented.split("\n")),
            )

        except Exception as e:
            logger.error("TypeScript instrumentation failed", error=str(e))
            return InstrumentationResult(
                success=False,
                instrumented_code=code,
                probes_inserted=[],
                error=str(e),
            )

    def _get_python_runtime(self) -> str:
        """Get Python runtime probe support code."""
        if self.mode == ProbeMode.DISABLED:
            return "def _codeverify_probe_violation(**kwargs): pass\n"

        if self.mode == ProbeMode.ASSERT:
            return '''
import traceback
import random
from datetime import datetime

_CODEVERIFY_PROBE_SAMPLE_RATE = {sample_rate}
_CODEVERIFY_VIOLATIONS = []

def _codeverify_probe_violation(probe_id: str, message: str, variables: dict, severity: str):
    """Record a probe violation."""
    if _CODEVERIFY_PROBE_SAMPLE_RATE < 1.0 and random.random() > _CODEVERIFY_PROBE_SAMPLE_RATE:
        return

    violation = {{
        "probe_id": probe_id,
        "message": message,
        "variables": variables,
        "severity": severity,
        "timestamp": datetime.utcnow().isoformat(),
        "stack_trace": traceback.format_stack(),
    }}
    _CODEVERIFY_VIOLATIONS.append(violation)

    # In ASSERT mode, raise
    raise AssertionError(f"CodeVerify probe violation: {{message}}")

'''.format(sample_rate=self.sample_rate)

        # LOG mode
        return '''
import traceback
import random
import logging
from datetime import datetime

_CODEVERIFY_PROBE_SAMPLE_RATE = {sample_rate}
_CODEVERIFY_VIOLATIONS = []
_codeverify_logger = logging.getLogger("codeverify.probes")

def _codeverify_probe_violation(probe_id: str, message: str, variables: dict, severity: str):
    """Record a probe violation (logging mode)."""
    if _CODEVERIFY_PROBE_SAMPLE_RATE < 1.0 and random.random() > _CODEVERIFY_PROBE_SAMPLE_RATE:
        return

    violation = {{
        "probe_id": probe_id,
        "message": message,
        "variables": variables,
        "severity": severity,
        "timestamp": datetime.utcnow().isoformat(),
        "stack_trace": traceback.format_stack(),
    }}
    _CODEVERIFY_VIOLATIONS.append(violation)

    # Log violation
    _codeverify_logger.warning(f"Probe violation [{{severity}}]: {{message}}", extra=violation)

'''.format(sample_rate=self.sample_rate)

    def _get_typescript_runtime(self) -> str:
        """Get TypeScript runtime probe support code."""
        if self.mode == ProbeMode.DISABLED:
            return "function _codeverifyProbeViolation(opts: any) {}\n"

        if self.mode == ProbeMode.ASSERT:
            return f'''
const _CODEVERIFY_PROBE_SAMPLE_RATE = {self.sample_rate};
const _CODEVERIFY_VIOLATIONS: any[] = [];

interface ProbeViolationOpts {{
    probeId: string;
    message: string;
    variables: Record<string, any>;
    severity: string;
}}

function _codeverifyProbeViolation(opts: ProbeViolationOpts): void {{
    if (_CODEVERIFY_PROBE_SAMPLE_RATE < 1.0 && Math.random() > _CODEVERIFY_PROBE_SAMPLE_RATE) {{
        return;
    }}

    const violation = {{
        ...opts,
        timestamp: new Date().toISOString(),
        stackTrace: new Error().stack,
    }};
    _CODEVERIFY_VIOLATIONS.push(violation);

    throw new Error(`CodeVerify probe violation: ${{opts.message}}`);
}}

'''

        # LOG mode
        return f'''
const _CODEVERIFY_PROBE_SAMPLE_RATE = {self.sample_rate};
const _CODEVERIFY_VIOLATIONS: any[] = [];

interface ProbeViolationOpts {{
    probeId: string;
    message: string;
    variables: Record<string, any>;
    severity: string;
}}

function _codeverifyProbeViolation(opts: ProbeViolationOpts): void {{
    if (_CODEVERIFY_PROBE_SAMPLE_RATE < 1.0 && Math.random() > _CODEVERIFY_PROBE_SAMPLE_RATE) {{
        return;
    }}

    const violation = {{
        ...opts,
        timestamp: new Date().toISOString(),
        stackTrace: new Error().stack,
    }};
    _CODEVERIFY_VIOLATIONS.push(violation);

    console.warn(`[CodeVerify] Probe violation [${{opts.severity}}]: ${{opts.message}}`, violation);
}}

'''


class RuntimeCollector:
    """Collects and aggregates runtime probe violations."""

    def __init__(self, max_violations: int = 10000) -> None:
        self._violations: list[ProbeViolation] = []
        self._max_violations = max_violations
        self._violation_counts: dict[str, int] = {}

    def record(self, violation: ProbeViolation) -> None:
        """Record a violation."""
        if len(self._violations) >= self._max_violations:
            # Evict oldest
            self._violations.pop(0)

        self._violations.append(violation)

        # Track counts by probe
        self._violation_counts[violation.probe_id] = (
            self._violation_counts.get(violation.probe_id, 0) + 1
        )

    def get_violations(
        self,
        probe_id: str | None = None,
        severity: ProbeSeverity | None = None,
        since: datetime | None = None,
    ) -> list[ProbeViolation]:
        """Get violations with optional filters."""
        result = self._violations

        if probe_id:
            result = [v for v in result if v.probe_id == probe_id]

        if severity:
            result = [v for v in result if v.severity == severity]

        if since:
            result = [v for v in result if v.timestamp >= since]

        return result

    def get_statistics(self) -> dict[str, Any]:
        """Get violation statistics."""
        by_severity = {}
        by_type = {}

        for v in self._violations:
            by_severity[v.severity.value] = by_severity.get(v.severity.value, 0) + 1

        return {
            "total_violations": len(self._violations),
            "by_severity": by_severity,
            "by_probe": self._violation_counts,
            "unique_probes": len(self._violation_counts),
        }

    def clear(self) -> None:
        """Clear all violations."""
        self._violations.clear()
        self._violation_counts.clear()


class RuntimeVerificationProbes:
    """Main interface for runtime verification probes.

    Usage:
        probes = RuntimeVerificationProbes()

        # From Z3 verification results
        constraints = probes.extract_constraints(z3_results)

        # Instrument code
        result = probes.instrument(code, constraints)

        # Deploy instrumented code...

        # Collect violations at runtime
        violations = probes.get_violations()
    """

    def __init__(
        self,
        language: str = "python",
        mode: ProbeMode = ProbeMode.LOG,
        sample_rate: float = 1.0,
    ) -> None:
        self._engine = InstrumentationEngine(language, mode, sample_rate)
        self._compiler = ProofToAssertCompiler(language)
        self._collector = RuntimeCollector()
        self.language = language

    def extract_constraints(
        self,
        verification_results: list[dict[str, Any]],
    ) -> list[Z3Constraint]:
        """Extract Z3 constraints from verification results."""
        constraints = []

        for result in verification_results:
            # Handle different verification result formats
            if "constraints" in result:
                for c in result["constraints"]:
                    constraints.append(Z3Constraint.from_z3_proof(c))

            elif "proofs" in result:
                for proof in result["proofs"]:
                    if proof.get("status") == "verified":
                        constraints.append(Z3Constraint(
                            id=proof.get("id", str(uuid.uuid4())),
                            expression=proof.get("assertion", ""),
                            variables=proof.get("variables", []),
                            constraint_type=self._infer_constraint_type(proof),
                            location=tuple(proof.get("location", [0, 0])),
                            description=proof.get("description", ""),
                        ))

            elif "findings" in result:
                for finding in result["findings"]:
                    constraint_type = self._finding_to_constraint_type(finding)
                    if constraint_type:
                        constraints.append(Z3Constraint(
                            id=finding.get("id", str(uuid.uuid4())),
                            expression=finding.get("constraint", "True"),
                            variables=finding.get("variables", []),
                            constraint_type=constraint_type,
                            location=(
                                finding.get("line_start", 0),
                                finding.get("line_end", 0),
                            ),
                            description=finding.get("message", ""),
                            severity=ProbeSeverity(
                                finding.get("severity", "medium").lower()
                            ),
                        ))

        return constraints

    def _infer_constraint_type(self, proof: dict[str, Any]) -> ProbeType:
        """Infer constraint type from proof."""
        check = proof.get("check", "").lower()
        assertion = proof.get("assertion", "").lower()

        if "null" in check or "none" in assertion:
            return ProbeType.NULL_CHECK
        elif "bounds" in check or "index" in assertion:
            return ProbeType.BOUNDS_CHECK
        elif "overflow" in check:
            return ProbeType.OVERFLOW_CHECK
        elif "div" in check or "zero" in assertion:
            return ProbeType.DIVISION_CHECK
        else:
            return ProbeType.INVARIANT

    def _finding_to_constraint_type(
        self,
        finding: dict[str, Any],
    ) -> ProbeType | None:
        """Convert finding category to constraint type."""
        category = finding.get("category", "").lower()

        mapping = {
            "null_safety": ProbeType.NULL_CHECK,
            "array_bounds": ProbeType.BOUNDS_CHECK,
            "integer_overflow": ProbeType.OVERFLOW_CHECK,
            "division_by_zero": ProbeType.DIVISION_CHECK,
        }

        return mapping.get(category)

    def instrument(
        self,
        code: str,
        constraints: list[Z3Constraint],
        include_runtime: bool = True,
    ) -> InstrumentationResult:
        """Instrument code with runtime probes."""
        return self._engine.instrument(code, constraints, include_runtime)

    def compile_single(self, constraint: Z3Constraint) -> RuntimeProbe:
        """Compile a single constraint to a probe."""
        return self._compiler.compile(constraint)

    def record_violation(self, violation: ProbeViolation) -> None:
        """Record a runtime violation."""
        self._collector.record(violation)

    def get_violations(
        self,
        probe_id: str | None = None,
        severity: ProbeSeverity | None = None,
        since: datetime | None = None,
    ) -> list[ProbeViolation]:
        """Get recorded violations."""
        return self._collector.get_violations(probe_id, severity, since)

    def get_statistics(self) -> dict[str, Any]:
        """Get violation statistics."""
        return self._collector.get_statistics()

    def export_violations(self, format: str = "json") -> str:
        """Export violations in specified format."""
        import json

        violations = self._collector.get_violations()

        if format == "json":
            return json.dumps([v.to_dict() for v in violations], indent=2)
        elif format == "sarif":
            return self._export_sarif(violations)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_sarif(self, violations: list[ProbeViolation]) -> str:
        """Export violations in SARIF format."""
        import json

        sarif = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "CodeVerify Runtime Probes",
                        "version": "1.0.0",
                    }
                },
                "results": [
                    {
                        "ruleId": v.probe_id,
                        "level": "warning" if v.severity in (ProbeSeverity.LOW, ProbeSeverity.MEDIUM) else "error",
                        "message": {"text": v.message},
                        "locations": [{
                            "physicalLocation": {
                                "artifactLocation": {"uri": v.file_path},
                                "region": {"startLine": v.line_number},
                            }
                        }],
                    }
                    for v in violations
                ],
            }],
        }

        return json.dumps(sarif, indent=2)
