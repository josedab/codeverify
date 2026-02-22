"""Integration tests: Verification-Guided Fuzzing with real execution.

Tests that the guided fuzzing service can generate test inputs from
counterexamples and execute them against actual Python functions to
confirm bugs vs false positives.

Unlike the core unit tests which simulate execution, these tests
call real functions with generated inputs.
"""

from __future__ import annotations

import pytest


# ─── Target functions with known bugs ──────────────────────────────────


def divide(a: int, b: int) -> float:
    """Intentionally buggy: no zero-division guard."""
    return a / b


def get_item(items: list, index: int):
    """Intentionally buggy: no bounds check."""
    return items[index]


def safe_divide(a: int, b: int) -> float:
    """Fixed version: has zero-division guard."""
    if b == 0:
        return 0.0
    return a / b


def process_user(user: dict | None) -> str:
    """Intentionally buggy: no null check."""
    return user["name"]


def safe_process_user(user: dict | None) -> str:
    """Fixed version."""
    if user is None:
        return "anonymous"
    return user.get("name", "unknown")


# ─── Real Execution Fuzzer ─────────────────────────────────────────────


class RealFuzzExecutor:
    """Executes fuzz inputs against actual Python functions."""

    def execute_with_inputs(
        self, func, inputs: dict, expected_exception: type | None = None
    ) -> tuple[bool, str]:
        """Execute a function with inputs. Returns (raised_expected, detail)."""
        try:
            result = func(**inputs)
            if expected_exception:
                return False, f"No exception raised, got result: {result}"
            return True, f"Executed successfully: {result}"
        except Exception as exc:
            if expected_exception and isinstance(exc, expected_exception):
                return True, f"Expected exception raised: {exc}"
            return False, f"Unexpected exception: {type(exc).__name__}: {exc}"


# ─── Tests ─────────────────────────────────────────────────────────────


class TestRealExecutionFuzzing:
    """Test guided fuzzing with actual function execution."""

    def test_division_by_zero_confirmed(self):
        """Z3 counterexample {b: 0} should trigger ZeroDivisionError."""
        from codeverify_core.guided_fuzzing import InputGenerator

        gen = InputGenerator()
        inputs = gen.from_counterexample({"a": 10, "b": 0})
        input_dict = {inp.variable_name: inp.value for inp in inputs}

        executor = RealFuzzExecutor()
        raised, detail = executor.execute_with_inputs(
            divide, input_dict, expected_exception=ZeroDivisionError
        )
        assert raised is True
        assert "division by zero" in detail.lower()

    def test_division_safe_version_passes(self):
        """Same inputs on fixed function should not raise."""
        executor = RealFuzzExecutor()
        raised, detail = executor.execute_with_inputs(
            safe_divide, {"a": 10, "b": 0}
        )
        assert raised is True
        assert "0.0" in detail

    def test_null_dereference_confirmed(self):
        """Z3 counterexample {user: None} should trigger TypeError."""
        executor = RealFuzzExecutor()
        raised, detail = executor.execute_with_inputs(
            process_user, {"user": None}, expected_exception=TypeError
        )
        assert raised is True

    def test_null_safe_version_passes(self):
        executor = RealFuzzExecutor()
        raised, detail = executor.execute_with_inputs(
            safe_process_user, {"user": None}
        )
        assert raised is True
        assert "anonymous" in detail

    def test_index_out_of_bounds_confirmed(self):
        """Z3 counterexample {index: -1} triggers IndexError on real list."""
        from codeverify_core.guided_fuzzing import InputGenerator

        gen = InputGenerator()
        inputs_raw = gen.from_counterexample({"index": 10})
        input_dict = {inp.variable_name: inp.value for inp in inputs_raw}
        input_dict["items"] = [1, 2, 3]

        executor = RealFuzzExecutor()
        raised, detail = executor.execute_with_inputs(
            get_item, input_dict, expected_exception=IndexError
        )
        assert raised is True

    def test_mutation_exploration(self):
        """Mutated inputs from counterexample explore nearby values."""
        from codeverify_core.guided_fuzzing import InputGenerator

        gen = InputGenerator()
        base = gen.from_counterexample({"b": 0})
        mutations = gen.generate_mutations(base, rounds=5)

        executor = RealFuzzExecutor()
        confirmed = 0
        for inputs in mutations:
            input_dict = {"a": 10, **{i.variable_name: i.value for i in inputs}}
            raised, _ = executor.execute_with_inputs(
                divide, input_dict, expected_exception=ZeroDivisionError
            )
            if raised:
                confirmed += 1

        # At least the original (b=0) should confirm
        assert confirmed >= 1

    def test_end_to_end_campaign_with_real_execution(self):
        """Full pipeline: counterexample → inputs → real execution → classification."""
        from codeverify_core.guided_fuzzing import FuzzResult, VerificationGuidedFuzzingService

        svc = VerificationGuidedFuzzingService()
        campaign = svc.fuzz_counterexample(
            "divide", "math.py", "division_by_zero",
            {"b": 0}, counterexample_id="ce-001",
        )

        # The simulated executor confirms it; verify with real execution too
        executor = RealFuzzExecutor()
        real_confirmed = 0
        for tc in campaign.test_cases:
            input_dict = {"a": 10, **{i.variable_name: i.value for i in tc.inputs}}
            raised, _ = executor.execute_with_inputs(
                divide, input_dict, expected_exception=ZeroDivisionError
            )
            if raised:
                real_confirmed += 1

        assert real_confirmed >= 1
        assert svc.classify_finding(campaign) == FuzzResult.CONFIRMED_BUG

    def test_false_positive_detection_via_real_execution(self):
        """Counterexample that doesn't trigger on fixed code = false positive."""
        executor = RealFuzzExecutor()

        # Z3 says b=0 is dangerous, but safe_divide handles it
        raised, detail = executor.execute_with_inputs(
            safe_divide, {"a": 10, "b": 0}
        )
        assert raised is True  # No exception
        assert "0.0" in detail  # Returns safe default

        # This means the Z3 finding is addressed by the fix
        # The original finding was valid for `divide`, but `safe_divide` is safe

    def test_generated_test_code_is_valid_python(self):
        """Verify generated pytest code is syntactically valid."""
        from codeverify_core.guided_fuzzing import FuzzInput, InputType, TestCodeGenerator

        gen = TestCodeGenerator()
        inputs = [
            FuzzInput(variable_name="a", value=10, input_type=InputType.INTEGER),
            FuzzInput(variable_name="b", value=0, input_type=InputType.INTEGER),
        ]
        code = gen.generate_pytest("divide", inputs, "division_by_zero")

        # Verify it's valid Python
        compile(code, "<test>", "exec")
