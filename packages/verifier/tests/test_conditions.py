"""Tests for verification condition generators."""

from codeverify_verifier.conditions import (
    generate_bounds_check,
    generate_null_check,
    generate_overflow_check,
    generate_division_check,
)


class TestConditionGenerators:
    """Test suite for verification condition generators."""

    def test_generate_null_check(self) -> None:
        """Test null check generation."""
        formula = generate_null_check("user", "User")

        assert "user_is_null" in formula
        assert "Bool" in formula
        assert "check-sat" in formula

    def test_generate_bounds_check_with_int_length(self) -> None:
        """Test bounds check with integer array length."""
        formula = generate_bounds_check("i", "arr", 10)

        assert "index" in formula
        assert "length" in formula
        assert "10" in formula
        assert "check-sat" in formula

    def test_generate_overflow_check_add(self) -> None:
        """Test overflow check for addition."""
        formula = generate_overflow_check("+", "a", "b", "int32")

        assert "Int" in formula
        assert "-2147483648" in formula or "2147483647" in formula
        assert "check-sat" in formula

    def test_generate_division_check(self) -> None:
        """Test division by zero check generation."""
        formula = generate_division_check("numerator", "denominator")

        assert "divisor" in formula
        assert "= divisor 0" in formula
        assert "check-sat" in formula
