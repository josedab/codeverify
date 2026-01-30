"""Tests for the Z3 Verifier."""

import pytest

from codeverify_verifier import Z3Verifier


class TestZ3Verifier:
    """Test suite for Z3Verifier class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.verifier = Z3Verifier(timeout_ms=5000)

    def test_simple_sat_formula(self) -> None:
        """Test satisfiable formula detection."""
        formula = """
        (declare-const x Int)
        (assert (> x 10))
        (assert (< x 20))
        (check-sat)
        """
        result = self.verifier.verify_condition(formula, "x between 10 and 20")

        assert result["satisfiable"] is True
        assert result["counterexample"] is not None
        assert "x" in result["counterexample"]

    def test_simple_unsat_formula(self) -> None:
        """Test unsatisfiable formula detection."""
        formula = """
        (declare-const x Int)
        (assert (> x 10))
        (assert (< x 5))
        (check-sat)
        """
        result = self.verifier.verify_condition(formula, "x > 10 and x < 5")

        assert result["satisfiable"] is False
        assert result["counterexample"] is None

    def test_integer_overflow_detection(self) -> None:
        """Test integer overflow detection."""
        result = self.verifier.check_integer_overflow(
            var_name="total",
            operation="mul",
            operand1_range=(0, 100000),
            operand2_range=(0, 100000),
            bit_width=32,
        )

        # Multiplication of large values should be able to overflow
        assert result["satisfiable"] is True
        assert result["counterexample"] is not None

    def test_no_overflow_small_values(self) -> None:
        """Test that small values don't overflow."""
        result = self.verifier.check_integer_overflow(
            var_name="total",
            operation="add",
            operand1_range=(0, 100),
            operand2_range=(0, 100),
            bit_width=32,
        )

        # Adding small values shouldn't overflow
        assert result["satisfiable"] is False

    def test_array_bounds_violation(self) -> None:
        """Test array bounds checking."""
        result = self.verifier.check_array_bounds(
            index_var="i",
            index_range=(0, 20),
            array_length=10,
        )

        # Index can go up to 20 but array is only 10 elements
        assert result["satisfiable"] is True
        assert result["counterexample"]["index"] >= 10

    def test_array_bounds_safe(self) -> None:
        """Test safe array access."""
        result = self.verifier.check_array_bounds(
            index_var="i",
            index_range=(0, 9),
            array_length=10,
        )

        # Index is always in bounds
        assert result["satisfiable"] is False

    def test_division_by_zero_possible(self) -> None:
        """Test division by zero detection."""
        result = self.verifier.check_division_by_zero(
            divisor_var="divisor",
            divisor_range=(-10, 10),
        )

        # Divisor can be 0
        assert result["satisfiable"] is True

    def test_division_by_zero_safe(self) -> None:
        """Test safe division."""
        result = self.verifier.check_division_by_zero(
            divisor_var="divisor",
            divisor_range=(1, 100),
        )

        # Divisor cannot be 0
        assert result["satisfiable"] is False

    def test_null_dereference_possible(self) -> None:
        """Test null dereference detection."""
        result = self.verifier.check_null_dereference(
            var_name="user",
            can_be_null=True,
            null_check_exists=False,
        )

        assert result["satisfiable"] is True
        assert result["counterexample"]["user"] == "null"

    def test_null_dereference_safe_with_check(self) -> None:
        """Test null check protects against dereference."""
        result = self.verifier.check_null_dereference(
            var_name="user",
            can_be_null=True,
            null_check_exists=True,
        )

        assert result["satisfiable"] is False

    def test_null_dereference_safe_not_nullable(self) -> None:
        """Test non-nullable type is safe."""
        result = self.verifier.check_null_dereference(
            var_name="user",
            can_be_null=False,
            null_check_exists=False,
        )

        assert result["satisfiable"] is False
