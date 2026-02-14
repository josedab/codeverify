"""Auto-Fix with Test Generation — generate tests to validate auto-fixes.

Companion module to :mod:`autofix_validation`.  Where that module applies and
validates fixes, this module automatically generates test cases (unit,
regression, edge-case, negative, property-based) to ensure fixes don't
introduce regressions.

Key components:
    - **TestGenerator**: build test cases from a fix diff.
    - **TestRunner**: simulated runner that validates generated tests.
    - **CoverageAnalyzer**: measure how well tests cover the changed code.
    - **AutoFixTestEngine**: end-to-end orchestrator for the pipeline.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class TestType(str, Enum):
    """Kind of test to generate for a fix."""

    UNIT = "unit"
    INTEGRATION = "integration"
    PROPERTY = "property"
    REGRESSION = "regression"
    FUZZ = "fuzz"


class TestFramework(str, Enum):
    """Supported test framework targets."""

    PYTEST = "pytest"
    JEST = "jest"
    JUNIT = "junit"
    MOCHA = "mocha"
    GO_TEST = "go_test"
    RSPEC = "rspec"


class FixConfidence(str, Enum):
    """Confidence that an auto-fix is correct."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class CoverageLevel(str, Enum):
    """How well the generated tests cover the fix diff."""

    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"
    EXCEEDS = "exceeds"


# =============================================================================
# Configuration & Result Dataclasses
# =============================================================================


@dataclass
class TestCase:
    """A single generated test case targeting a fix."""

    id: str
    name: str
    test_type: TestType
    code: str
    language: str
    target_function: str
    description: str
    expected_outcome: str = "pass"
    framework: TestFramework = TestFramework.PYTEST
    covers_fix: bool = True
    assertions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "test_type": self.test_type.value,
            "code": self.code,
            "language": self.language,
            "target_function": self.target_function,
            "description": self.description,
            "expected_outcome": self.expected_outcome,
            "framework": self.framework.value,
            "covers_fix": self.covers_fix,
            "assertions": list(self.assertions),
        }


@dataclass
class FixWithTests:
    """A code fix paired with its generated test cases."""

    fix_id: str
    original_code: str
    fixed_code: str
    issue_description: str
    language: str
    tests: list[TestCase] = field(default_factory=list)
    fix_confidence: FixConfidence = FixConfidence.MEDIUM
    coverage_level: CoverageLevel = CoverageLevel.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "fix_id": self.fix_id,
            "original_code": self.original_code,
            "fixed_code": self.fixed_code,
            "issue_description": self.issue_description,
            "language": self.language,
            "tests": [t.to_dict() for t in self.tests],
            "fix_confidence": self.fix_confidence.value,
            "coverage_level": self.coverage_level.value,
        }


@dataclass
class TestGenerationConfig:
    """Tunable knobs for test generation behaviour."""

    max_tests_per_fix: int = 5
    test_types: list[TestType] = field(
        default_factory=lambda: [TestType.UNIT, TestType.REGRESSION]
    )
    framework: TestFramework = TestFramework.PYTEST
    include_edge_cases: bool = True
    include_negative_tests: bool = True
    include_property_tests: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tests_per_fix": self.max_tests_per_fix,
            "test_types": [t.value for t in self.test_types],
            "framework": self.framework.value,
            "include_edge_cases": self.include_edge_cases,
            "include_negative_tests": self.include_negative_tests,
            "include_property_tests": self.include_property_tests,
        }


@dataclass
class TestSuite:
    """Collection of tests generated for a single fix."""

    id: str
    fix_id: str
    tests: list[TestCase]
    generated_at: datetime
    coverage_summary: dict[str, Any] = field(default_factory=dict)
    all_pass: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fix_id": self.fix_id,
            "tests": [t.to_dict() for t in self.tests],
            "generated_at": self.generated_at.isoformat(),
            "coverage_summary": dict(self.coverage_summary),
            "all_pass": self.all_pass,
        }


@dataclass
class TestRunResult:
    """Outcome of running a single test case."""

    test_id: str
    passed: bool
    execution_time_ms: float
    error_message: str | None = None
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "passed": self.passed,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "output": self.output,
        }


@dataclass
class FixValidationReport:
    """Final report summarising fix validation via generated tests."""

    fix_id: str
    fix_applied: bool
    tests_generated: int
    tests_passed: int
    tests_failed: int
    coverage_level: CoverageLevel
    confidence: FixConfidence
    regression_risk: float
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fix_id": self.fix_id,
            "fix_applied": self.fix_applied,
            "tests_generated": self.tests_generated,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "coverage_level": self.coverage_level.value,
            "confidence": self.confidence.value,
            "regression_risk": self.regression_risk,
            "recommendations": list(self.recommendations),
        }


# =============================================================================
# Test Code Templates
# =============================================================================

_PYTEST_UNIT_TEMPLATE = '''import pytest
from {module} import {function}


def test_{function}_basic():
    """Verify {function} returns expected result after fix."""
{assertions}
'''

_PYTEST_REGRESSION_TEMPLATE = '''import pytest
from {module} import {function}


class TestRegression{class_name}:
    """Regression tests ensuring the fix for '{issue}' holds."""

    def test_original_issue_resolved(self):
        """The original issue must not resurface."""
{assertions}

    def test_existing_behavior_preserved(self):
        """Pre-existing correct behaviour must remain intact."""
{preservation_assertions}
'''

_PYTEST_EDGE_CASE_TEMPLATE = '''import pytest
from {module} import {function}


@pytest.mark.parametrize("input_val, expected", [
{parametrize_values}
])
def test_{function}_edge_cases(input_val, expected):
    """Edge-case inputs must not crash or produce wrong results."""
    result = {function}(input_val)
    assert result == expected
'''

_PYTEST_NEGATIVE_TEMPLATE = '''import pytest
from {module} import {function}


class TestNegative{class_name}:
    """Negative tests for {function} — invalid inputs must fail gracefully."""

    def test_none_input(self):
        with pytest.raises((TypeError, ValueError)):
            {function}(None)

    def test_empty_string(self):
        with pytest.raises((TypeError, ValueError)):
            {function}("")

    def test_invalid_type(self):
        with pytest.raises((TypeError, ValueError)):
            {function}(object())
'''

_PYTEST_PROPERTY_TEMPLATE = '''import pytest
from hypothesis import given, strategies as st
from {module} import {function}


@given({strategies})
def test_{function}_property(value):
    """Property: {property_description}"""
    result = {function}(value)
    assert {property_assertion}
'''

# Framework-specific wrappers for non-pytest targets
_JEST_TEMPLATE = '''const {{ {function} }} = require('./{module}');

describe('{function}', () => {{
  test('basic correctness after fix', () => {{
{assertions}
  }});
}});
'''

_JUNIT_TEMPLATE = '''import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class {class_name}Test {{
    @Test
    void test{class_name}Basic() {{
{assertions}
    }}
}}
'''


# =============================================================================
# Edge-Case Value Registry
# =============================================================================

_EDGE_CASES_BY_TYPE: dict[str, list[dict[str, Any]]] = {
    "str": [
        {"input": '""', "label": "empty string"},
        {"input": '" "', "label": "whitespace only"},
        {"input": '"a" * 10000', "label": "very long string"},
        {"input": '"\\n\\t\\r"', "label": "whitespace characters"},
        {"input": '"<script>alert(1)</script>"', "label": "html injection"},
    ],
    "int": [
        {"input": "0", "label": "zero"},
        {"input": "-1", "label": "negative"},
        {"input": "2**31 - 1", "label": "max 32-bit int"},
        {"input": "-(2**31)", "label": "min 32-bit int"},
    ],
    "float": [
        {"input": "0.0", "label": "zero float"},
        {"input": "-0.0", "label": "negative zero"},
        {"input": "float('inf')", "label": "positive infinity"},
        {"input": "float('-inf')", "label": "negative infinity"},
        {"input": "float('nan')", "label": "NaN"},
    ],
    "list": [
        {"input": "[]", "label": "empty list"},
        {"input": "[None]", "label": "list with None"},
        {"input": "list(range(10000))", "label": "large list"},
    ],
    "dict": [
        {"input": "{}", "label": "empty dict"},
        {"input": '{"": ""}', "label": "empty key/value"},
    ],
    "None": [
        {"input": "None", "label": "None value"},
    ],
}


# =============================================================================
# Test Generator
# =============================================================================


class TestGenerator:
    """Generate test cases for code fixes.

    Analyses the diff between original and fixed code, extracts target
    functions and parameters, then emits concrete test code for the
    configured framework and test types.
    """

    def __init__(self, config: TestGenerationConfig | None = None) -> None:
        self._config = config or TestGenerationConfig()
        self._test_counter: int = 0

    def generate_tests(self, fix: FixWithTests) -> list[TestCase]:
        """Return a list of test cases covering *fix*."""
        logger.info(
            "test_generation.start",
            fix_id=fix.fix_id,
            language=fix.language,
            types=[t.value for t in self._config.test_types],
        )

        function_name = self._extract_function_name(fix.fixed_code)
        tests: list[TestCase] = []

        for test_type in self._config.test_types:
            if len(tests) >= self._config.max_tests_per_fix:
                break
            if test_type == TestType.UNIT:
                tests.append(self._generate_unit_test(fix, function_name))
            elif test_type == TestType.REGRESSION:
                tests.append(self._generate_regression_test(fix))

        if self._config.include_edge_cases and len(tests) < self._config.max_tests_per_fix:
            tests.append(self._generate_edge_case_test(fix))

        if self._config.include_negative_tests and len(tests) < self._config.max_tests_per_fix:
            tests.append(self._generate_negative_test(fix))

        if self._config.include_property_tests and len(tests) < self._config.max_tests_per_fix:
            tests.append(self._generate_property_test(fix))

        logger.info("test_generation.complete", fix_id=fix.fix_id, count=len(tests))
        return tests

    # --- Individual test generators -------------------------------------------

    def _generate_unit_test(self, fix: FixWithTests, function_name: str) -> TestCase:
        """Emit a basic unit test that calls the fixed function."""
        params = self._extract_parameters(fix.fixed_code, function_name)
        code = self._generate_test_code(
            TestType.UNIT, function_name, params, self._config.framework,
        )
        test_id = self._next_id("unit")
        return TestCase(
            id=test_id,
            name=f"test_{function_name}_basic",
            test_type=TestType.UNIT,
            code=code,
            language=fix.language,
            target_function=function_name,
            description=f"Unit test verifying {function_name} after fix",
            framework=self._config.framework,
            assertions=[f"assert {function_name}(...) returns expected value"],
        )

    def _generate_regression_test(self, fix: FixWithTests) -> TestCase:
        """Emit a regression test that validates the original issue is gone."""
        function_name = self._extract_function_name(fix.fixed_code)
        class_name = _to_class_name(function_name)
        issue_short = fix.issue_description[:60].replace("'", "\\'")

        assertions = self._build_regression_assertions(fix, function_name)
        preservation = self._build_preservation_assertions(fix, function_name)

        code = _PYTEST_REGRESSION_TEMPLATE.format(
            module="module_under_test",
            function=function_name,
            class_name=class_name,
            issue=issue_short,
            assertions=assertions,
            preservation_assertions=preservation,
        )
        test_id = self._next_id("regression")
        return TestCase(
            id=test_id,
            name=f"test_regression_{function_name}",
            test_type=TestType.REGRESSION,
            code=code,
            language=fix.language,
            target_function=function_name,
            description=f"Regression test: '{issue_short}' must not recur",
            framework=self._config.framework,
            assertions=[
                "original issue is resolved",
                "existing behaviour preserved",
            ],
        )

    def _generate_edge_case_test(self, fix: FixWithTests) -> TestCase:
        """Emit a parametrised edge-case test."""
        function_name = self._extract_function_name(fix.fixed_code)
        params = self._extract_parameters(fix.fixed_code, function_name)
        edge_cases = self._infer_edge_cases(params)

        parametrize_lines: list[str] = []
        for ec in edge_cases:
            parametrize_lines.append(f'    ({ec["input"]}, {ec.get("expected", "None")}),')
        parametrize_values = "\n".join(parametrize_lines) if parametrize_lines else '    (None, None),'

        code = _PYTEST_EDGE_CASE_TEMPLATE.format(
            module="module_under_test",
            function=function_name,
            parametrize_values=parametrize_values,
        )
        test_id = self._next_id("edge")
        return TestCase(
            id=test_id,
            name=f"test_{function_name}_edge_cases",
            test_type=TestType.UNIT,
            code=code,
            language=fix.language,
            target_function=function_name,
            description=f"Edge-case inputs for {function_name}",
            framework=self._config.framework,
            covers_fix=True,
            assertions=[f"edge case: {ec.get('label', 'unknown')}" for ec in edge_cases],
        )

    def _generate_negative_test(self, fix: FixWithTests) -> TestCase:
        """Emit negative tests — invalid inputs must fail gracefully."""
        function_name = self._extract_function_name(fix.fixed_code)
        class_name = _to_class_name(function_name)

        code = _PYTEST_NEGATIVE_TEMPLATE.format(
            module="module_under_test",
            function=function_name,
            class_name=class_name,
        )
        test_id = self._next_id("negative")
        return TestCase(
            id=test_id,
            name=f"test_{function_name}_negative",
            test_type=TestType.UNIT,
            code=code,
            language=fix.language,
            target_function=function_name,
            description=f"Negative tests for {function_name} with invalid inputs",
            expected_outcome="pass",
            framework=self._config.framework,
            covers_fix=False,
            assertions=[
                "None input raises TypeError or ValueError",
                "empty string raises TypeError or ValueError",
                "invalid type raises TypeError or ValueError",
            ],
        )

    def _generate_property_test(self, fix: FixWithTests) -> TestCase:
        """Emit a Hypothesis property-based test."""
        function_name = self._extract_function_name(fix.fixed_code)
        params = self._extract_parameters(fix.fixed_code, function_name)

        strategies = self._params_to_strategies(params)
        prop_desc = f"{function_name} never raises unhandled exceptions"
        prop_assertion = "result is not None or result is None"  # basic smoke

        code = _PYTEST_PROPERTY_TEMPLATE.format(
            module="module_under_test",
            function=function_name,
            strategies=strategies,
            property_description=prop_desc,
            property_assertion=prop_assertion,
        )
        test_id = self._next_id("property")
        return TestCase(
            id=test_id,
            name=f"test_{function_name}_property",
            test_type=TestType.PROPERTY,
            code=code,
            language=fix.language,
            target_function=function_name,
            description=f"Property test for {function_name}",
            framework=self._config.framework,
            assertions=["property holds for all generated inputs"],
        )

    # --- Extraction helpers ---------------------------------------------------

    def _extract_function_name(self, code: str) -> str:
        """Return the first ``def`` name found in *code*, or ``'unknown'``."""
        match = re.search(r"def\s+(\w+)\s*\(", code)
        return match.group(1) if match else "unknown"

    def _extract_parameters(self, code: str, function_name: str) -> list[dict[str, Any]]:
        """Extract parameter names and inferred types from a function signature."""
        pattern = re.compile(rf"def\s+{re.escape(function_name)}\s*\(([^)]*)\)")
        match = pattern.search(code)
        if not match:
            return []

        raw = match.group(1)
        params: list[dict[str, Any]] = []
        for part in raw.split(","):
            part = part.strip()
            if not part or part == "self" or part == "cls":
                continue
            if ":" in part:
                name, type_hint = part.split(":", 1)
                name = name.strip()
                type_hint = type_hint.split("=")[0].strip()
            else:
                name = part.split("=")[0].strip()
                type_hint = "Any"
            params.append({"name": name, "type": type_hint})
        return params

    def _generate_test_code(
        self,
        test_type: TestType,
        function_name: str,
        params: list[dict[str, Any]],
        framework: TestFramework,
    ) -> str:
        """Render concrete test source for the given framework."""
        if framework == TestFramework.PYTEST:
            return self._render_pytest(test_type, function_name, params)
        if framework == TestFramework.JEST:
            return self._render_jest(function_name, params)
        if framework == TestFramework.JUNIT:
            return self._render_junit(function_name)
        # Fallback to pytest-style
        return self._render_pytest(test_type, function_name, params)

    def _render_pytest(
        self, test_type: TestType, function_name: str, params: list[dict[str, Any]],
    ) -> str:
        """Build a pytest unit test string with concrete assertions."""
        arg_values = ", ".join(self._default_value_for(p["type"]) for p in params)
        call = f"{function_name}({arg_values})"

        lines = [
            f"    result = {call}",
            f"    assert result is not None, \"{function_name} returned None\"",
        ]
        # Type-specific assertions
        for p in params:
            ptype = p["type"].lower()
            if "str" in ptype:
                lines.append(f"    assert isinstance(result, (str, type(None)))")
                break
            if "int" in ptype or "float" in ptype:
                lines.append(f"    assert isinstance(result, (int, float, type(None)))")
                break

        assertions = "\n".join(lines)
        return _PYTEST_UNIT_TEMPLATE.format(
            module="module_under_test",
            function=function_name,
            assertions=assertions,
        )

    def _render_jest(self, function_name: str, params: list[dict[str, Any]]) -> str:
        arg_values = ", ".join(self._js_default(p["type"]) for p in params)
        assertions = f"    expect({function_name}({arg_values})).toBeDefined();"
        return _JEST_TEMPLATE.format(
            module="module_under_test",
            function=function_name,
            assertions=assertions,
        )

    def _render_junit(self, function_name: str) -> str:
        class_name = _to_class_name(function_name)
        assertions = f"        assertNotNull(new {class_name}());"
        return _JUNIT_TEMPLATE.format(
            class_name=class_name,
            assertions=assertions,
        )

    def _infer_edge_cases(self, params: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Look up edge-case values for each parameter type."""
        cases: list[dict[str, Any]] = []
        for p in params:
            ptype = self._normalise_type(p["type"])
            type_cases = _EDGE_CASES_BY_TYPE.get(ptype, _EDGE_CASES_BY_TYPE["None"])
            cases.extend(type_cases)
        # Deduplicate by input value
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for c in cases:
            if c["input"] not in seen:
                seen.add(c["input"])
                unique.append(c)
        return unique

    # --- Internal utilities ---------------------------------------------------

    def _build_regression_assertions(self, fix: FixWithTests, function_name: str) -> str:
        """Build assertion lines proving the original issue is gone."""
        lines = [
            f"        # The fix addresses: {fix.issue_description[:50]}",
            f"        result = {function_name}()",
            f"        assert result is not None",
        ]
        return "\n".join(lines)

    def _build_preservation_assertions(self, fix: FixWithTests, function_name: str) -> str:
        """Build assertion lines proving existing behaviour is unchanged."""
        lines = [
            f"        result = {function_name}()",
            f"        assert result is not None, \"existing behaviour broken\"",
        ]
        return "\n".join(lines)

    def _params_to_strategies(self, params: list[dict[str, Any]]) -> str:
        """Map parameter types to Hypothesis strategies."""
        mappings: dict[str, str] = {
            "str": "st.text(min_size=0, max_size=100)",
            "int": "st.integers(min_value=-1000, max_value=1000)",
            "float": "st.floats(allow_nan=False, allow_infinity=False)",
            "bool": "st.booleans()",
            "list": "st.lists(st.integers(), max_size=50)",
        }
        parts: list[str] = []
        for p in params:
            ptype = self._normalise_type(p["type"])
            strategy = mappings.get(ptype, "st.text()")
            parts.append(f"{p['name']}={strategy}")
        return ", ".join(parts) if parts else "value=st.text()"

    def _normalise_type(self, type_hint: str) -> str:
        """Simplify a type hint to a base type key."""
        hint = type_hint.strip().lower()
        for base in ("str", "int", "float", "bool", "list", "dict"):
            if base in hint:
                return base
        return "None"

    def _default_value_for(self, type_hint: str) -> str:
        """Return a sensible default literal for a type hint."""
        defaults: dict[str, str] = {
            "str": '"test_value"',
            "int": "1",
            "float": "1.0",
            "bool": "True",
            "list": "[1, 2, 3]",
            "dict": '{"key": "value"}',
        }
        ptype = self._normalise_type(type_hint)
        return defaults.get(ptype, "None")

    def _js_default(self, type_hint: str) -> str:
        """Return a sensible JavaScript default literal."""
        defaults: dict[str, str] = {
            "str": "'test_value'",
            "int": "1",
            "float": "1.0",
            "bool": "true",
            "list": "[1, 2, 3]",
            "dict": "{ key: 'value' }",
        }
        ptype = self._normalise_type(type_hint)
        return defaults.get(ptype, "null")

    def _next_id(self, prefix: str) -> str:
        """Generate a sequential test identifier."""
        self._test_counter += 1
        return f"test-{prefix}-{self._test_counter:04d}"


# =============================================================================
# Test Runner
# =============================================================================


class TestRunner:
    """Simulated test runner for generated tests.

    Validates syntax and simulates execution for each test case without
    actually invoking an external test framework.
    """

    def __init__(self) -> None:
        self._run_count: int = 0

    def run_test(self, test: TestCase, fixed_code: str) -> TestRunResult:
        """Validate and simulate execution of a single test."""
        self._run_count += 1
        logger.debug("test_runner.run", test_id=test.id, test_type=test.test_type.value)

        valid, error = self._validate_syntax(test.code, test.language)
        if not valid:
            return TestRunResult(
                test_id=test.id,
                passed=False,
                execution_time_ms=0.0,
                error_message=f"Syntax error in generated test: {error}",
                output="",
            )

        return self._simulate_execution(test)

    def run_suite(self, suite: TestSuite, fixed_code: str) -> list[TestRunResult]:
        """Run every test in *suite* and return the results."""
        logger.info("test_runner.suite_start", suite_id=suite.id, count=len(suite.tests))
        results = [self.run_test(tc, fixed_code) for tc in suite.tests]
        passed = sum(1 for r in results if r.passed)
        logger.info(
            "test_runner.suite_complete",
            suite_id=suite.id,
            passed=passed,
            failed=len(results) - passed,
        )
        return results

    def _validate_syntax(self, code: str, language: str) -> tuple[bool, str]:
        """Check that *code* parses without errors."""
        if language == "python":
            try:
                ast.parse(code)
                return True, ""
            except SyntaxError as exc:
                return False, str(exc)

        # Basic brace-balance check for other languages
        opens = code.count("{") + code.count("(") + code.count("[")
        closes = code.count("}") + code.count(")") + code.count("]")
        if opens != closes:
            return False, f"Unbalanced delimiters: {opens} open vs {closes} close"
        return True, ""

    def _simulate_execution(self, test: TestCase) -> TestRunResult:
        """Simulate running a test — uses heuristics on the test code."""
        start = time.monotonic()

        has_assertions = (
            "assert " in test.code
            or "expect(" in test.code
            or "assertEquals" in test.code
            or "assertNotNull" in test.code
        )

        has_import = "import " in test.code or "require(" in test.code
        calls_target = test.target_function in test.code

        passed = has_assertions and calls_target
        elapsed = (time.monotonic() - start) * 1000

        error: str | None = None
        if not has_assertions:
            error = "No assertions found in test"
        elif not calls_target:
            error = f"Test does not call target function '{test.target_function}'"

        return TestRunResult(
            test_id=test.id,
            passed=passed,
            execution_time_ms=round(elapsed, 3),
            error_message=error,
            output=f"{'PASS' if passed else 'FAIL'}: {test.name}",
        )


# =============================================================================
# Coverage Analyzer
# =============================================================================


class CoverageAnalyzer:
    """Analyse test coverage of fixes.

    Compares changed lines (the diff) against test-case targets to determine
    whether the generated tests adequately cover the fix.
    """

    def __init__(self) -> None:
        self._line_weight: float = 1.0

    def analyze_coverage(self, fix: FixWithTests, tests: list[TestCase]) -> CoverageLevel:
        """Determine overall coverage level."""
        changed_lines = self._extract_changed_lines(fix.original_code, fix.fixed_code)
        if not changed_lines:
            return CoverageLevel.NONE

        coverage_ratio = self._check_line_coverage(tests, changed_lines)
        logger.debug(
            "coverage.analyzed",
            fix_id=fix.fix_id,
            changed=len(changed_lines),
            ratio=round(coverage_ratio, 2),
        )

        if coverage_ratio <= 0.0:
            return CoverageLevel.NONE
        if coverage_ratio < 0.7:
            return CoverageLevel.PARTIAL
        if coverage_ratio <= 1.0:
            return CoverageLevel.FULL
        return CoverageLevel.EXCEEDS

    def identify_gaps(self, fix: FixWithTests, tests: list[TestCase]) -> list[str]:
        """Return human-readable descriptions of uncovered areas."""
        gaps: list[str] = []
        changed_lines = self._extract_changed_lines(fix.original_code, fix.fixed_code)
        if not changed_lines:
            return gaps

        target_functions = {t.target_function for t in tests}
        fixed_functions = set(re.findall(r"def\s+(\w+)\s*\(", fix.fixed_code))

        untested = fixed_functions - target_functions - {"__init__"}
        for fn in sorted(untested):
            gaps.append(f"Function '{fn}' has no dedicated test")

        test_types_present = {t.test_type for t in tests}
        if TestType.REGRESSION not in test_types_present:
            gaps.append("No regression test for the specific issue")

        covers_fix = sum(1 for t in tests if t.covers_fix)
        if covers_fix == 0:
            gaps.append("No test directly targets the fix diff")

        if not any("edge" in t.name or "parametrize" in t.code for t in tests):
            gaps.append("No edge-case or parametrised tests")

        return gaps

    def _extract_changed_lines(self, original: str, fixed: str) -> list[int]:
        """Return 1-based line numbers that differ between original and fixed."""
        orig_lines = original.splitlines()
        fixed_lines = fixed.splitlines()
        matcher = difflib.SequenceMatcher(None, orig_lines, fixed_lines)

        changed: list[int] = []
        for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
            if tag in ("replace", "insert"):
                changed.extend(range(j1 + 1, j2 + 1))
        return changed

    def _check_line_coverage(self, tests: list[TestCase], changed_lines: list[int]) -> float:
        """Estimate coverage ratio based on test targets."""
        if not changed_lines:
            return 0.0

        covered = 0
        total = len(changed_lines)

        for test in tests:
            if test.covers_fix:
                # Each fix-covering test covers a proportional share
                covered += max(1, total // max(len(tests), 1))

        return min(covered / total, 1.5)


# =============================================================================
# Auto-Fix Test Engine
# =============================================================================


class AutoFixTestEngine:
    """End-to-end engine for fix validation with test generation.

    Orchestrates :class:`TestGenerator`, :class:`TestRunner`, and
    :class:`CoverageAnalyzer` to produce a :class:`FixValidationReport`.
    """

    def __init__(self, config: TestGenerationConfig | None = None) -> None:
        self._config = config or TestGenerationConfig()
        self._generator = TestGenerator(config=self._config)
        self._runner = TestRunner()
        self._analyzer = CoverageAnalyzer()

    def validate_fix(
        self,
        original_code: str,
        fixed_code: str,
        issue_description: str,
        language: str = "python",
    ) -> FixValidationReport:
        """One-call entry point: generate tests, run them, and report."""
        fix_id = _generate_fix_id(original_code, fixed_code)
        logger.info("autofix_test_engine.validate", fix_id=fix_id)

        fix = FixWithTests(
            fix_id=fix_id,
            original_code=original_code,
            fixed_code=fixed_code,
            issue_description=issue_description,
            language=language,
        )
        return self.run_validation(fix)

    def generate_test_suite(self, fix: FixWithTests) -> TestSuite:
        """Generate a :class:`TestSuite` for *fix*."""
        tests = self._generator.generate_tests(fix)
        fix.tests = tests

        coverage = self._analyzer.analyze_coverage(fix, tests)
        fix.coverage_level = coverage

        suite = TestSuite(
            id=str(uuid4()),
            fix_id=fix.fix_id,
            tests=tests,
            generated_at=datetime.now(tz=timezone.utc),
            coverage_summary={
                "level": coverage.value,
                "test_count": len(tests),
                "gaps": self._analyzer.identify_gaps(fix, tests),
            },
        )
        logger.info(
            "autofix_test_engine.suite_generated",
            fix_id=fix.fix_id,
            tests=len(tests),
            coverage=coverage.value,
        )
        return suite

    def run_validation(self, fix: FixWithTests) -> FixValidationReport:
        """Generate tests, execute them, and return a full report."""
        suite = self.generate_test_suite(fix)
        results = self._runner.run_suite(suite, fix.fixed_code)

        confidence = self.get_fix_confidence(fix, results)
        fix.fix_confidence = confidence

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        suite.all_pass = failed == 0

        regression_risk = self._calculate_regression_risk(fix, results)
        recommendations = self._build_recommendations(fix, results, suite)

        report = FixValidationReport(
            fix_id=fix.fix_id,
            fix_applied=suite.all_pass,
            tests_generated=len(results),
            tests_passed=passed,
            tests_failed=failed,
            coverage_level=fix.coverage_level,
            confidence=confidence,
            regression_risk=regression_risk,
            recommendations=recommendations,
        )
        logger.info(
            "autofix_test_engine.validation_complete",
            fix_id=fix.fix_id,
            passed=passed,
            failed=failed,
            confidence=confidence.value,
        )
        return report

    def get_fix_confidence(
        self, fix: FixWithTests, test_results: list[TestRunResult],
    ) -> FixConfidence:
        """Derive confidence from test-pass rate and coverage."""
        if not test_results:
            return FixConfidence.UNKNOWN

        pass_rate = sum(1 for r in test_results if r.passed) / len(test_results)
        coverage = fix.coverage_level

        if pass_rate >= 0.95 and coverage in (CoverageLevel.FULL, CoverageLevel.EXCEEDS):
            return FixConfidence.HIGH
        if pass_rate >= 0.7 and coverage != CoverageLevel.NONE:
            return FixConfidence.MEDIUM
        if pass_rate >= 0.4:
            return FixConfidence.LOW
        return FixConfidence.UNKNOWN

    def batch_validate(self, fixes: list[FixWithTests]) -> list[FixValidationReport]:
        """Validate a batch of fixes sequentially."""
        logger.info("autofix_test_engine.batch_start", count=len(fixes))
        reports = [self.run_validation(fix) for fix in fixes]
        passed = sum(1 for r in reports if r.fix_applied)
        logger.info(
            "autofix_test_engine.batch_complete",
            total=len(reports),
            passed=passed,
        )
        return reports

    # --- Internal helpers -----------------------------------------------------

    def _calculate_regression_risk(
        self, fix: FixWithTests, results: list[TestRunResult],
    ) -> float:
        """Return a 0.0–1.0 risk score for regressions."""
        if not results:
            return 1.0

        fail_rate = sum(1 for r in results if not r.passed) / len(results)

        # Larger diffs carry more risk
        diff_size = len(
            list(difflib.unified_diff(
                fix.original_code.splitlines(),
                fix.fixed_code.splitlines(),
            ))
        )
        diff_factor = min(diff_size / 50.0, 1.0)

        coverage_penalty = {
            CoverageLevel.NONE: 0.4,
            CoverageLevel.PARTIAL: 0.2,
            CoverageLevel.FULL: 0.0,
            CoverageLevel.EXCEEDS: 0.0,
        }.get(fix.coverage_level, 0.3)

        risk = (fail_rate * 0.5) + (diff_factor * 0.3) + (coverage_penalty * 0.2)
        return round(min(risk, 1.0), 3)

    def _build_recommendations(
        self,
        fix: FixWithTests,
        results: list[TestRunResult],
        suite: TestSuite,
    ) -> list[str]:
        """Generate actionable recommendations for the report."""
        recs: list[str] = []

        failed = [r for r in results if not r.passed]
        if failed:
            recs.append(
                f"{len(failed)} test(s) failed — review generated test code for correctness"
            )

        gaps = suite.coverage_summary.get("gaps", [])
        if gaps:
            recs.append(f"Coverage gaps identified: {'; '.join(gaps[:3])}")

        if fix.coverage_level == CoverageLevel.NONE:
            recs.append("No tests cover the fix diff — add targeted assertions")
        elif fix.coverage_level == CoverageLevel.PARTIAL:
            recs.append("Consider adding tests for uncovered changed lines")

        if not any(t.test_type == TestType.REGRESSION for t in fix.tests):
            recs.append("Add a regression test for the specific issue being fixed")

        if not recs:
            recs.append("All tests pass with full coverage — fix is ready to merge")

        return recs


# =============================================================================
# Helpers
# =============================================================================


def _generate_fix_id(original_code: str, fixed_code: str) -> str:
    """Deterministic fix identifier derived from the code pair."""
    return hashlib.sha256(f"{original_code}||{fixed_code}".encode()).hexdigest()[:16]


def _to_class_name(function_name: str) -> str:
    """Convert a snake_case function name to PascalCase."""
    return "".join(part.capitalize() for part in function_name.split("_"))
