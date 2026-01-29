"""Tests for new code quality abstractions."""

import pytest
from datetime import datetime

from codeverify_core.models import (
    FindingSeverity,
    Result,
    OperationResult,
    TimestampMixin,
    DataclassTimestampMixin,
    parse_severity,
    compare_severity,
    is_blocking_severity,
    parse_iso_datetime,
)
from codeverify_core.repositories import (
    InMemoryRepository,
    InMemoryScanResultRepository,
    InMemoryNotificationConfigRepository,
    get_scan_result_repository,
    set_scan_result_repository,
)
from codeverify_core.rules import (
    RuleEvaluator,
    RuleBuilder,
    RuleType,
    RuleSeverity,
    PatternRuleStrategy,
    CompositeRuleStrategy,
)


# ============================================================================
# Result Type Tests
# ============================================================================


class TestResult:
    """Tests for the Result type pattern."""

    def test_ok_result(self):
        """Test creating and using successful results."""
        result = Result.ok(42)
        assert result.is_ok
        assert not result.is_err
        assert result.value == 42
        assert result.error is None
        assert result.unwrap() == 42

    def test_err_result(self):
        """Test creating and using error results."""
        result = Result.err("Something went wrong")
        assert not result.is_ok
        assert result.is_err
        assert result.value is None
        assert result.error == "Something went wrong"
        assert result.unwrap_err() == "Something went wrong"

    def test_unwrap_on_error_raises(self):
        """Test that unwrap on error raises ValueError."""
        result = Result.err("error")
        with pytest.raises(ValueError, match="error result"):
            result.unwrap()

    def test_unwrap_err_on_ok_raises(self):
        """Test that unwrap_err on ok raises ValueError."""
        result = Result.ok("value")
        with pytest.raises(ValueError, match="successful result"):
            result.unwrap_err()

    def test_unwrap_or(self):
        """Test unwrap_or with default values."""
        ok_result = Result.ok(42)
        err_result = Result.err("error")
        
        assert ok_result.unwrap_or(0) == 42
        assert err_result.unwrap_or(0) == 0

    def test_map_on_ok(self):
        """Test mapping over successful results."""
        result = Result.ok(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.unwrap() == 10

    def test_map_on_err(self):
        """Test mapping over error results (should not apply)."""
        result = Result.err("error")
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_err
        assert mapped.error == "error"


class TestOperationResult:
    """Tests for the OperationResult Pydantic model."""

    def test_ok_operation(self):
        """Test creating successful operation results."""
        result = OperationResult.ok({"key": "value"})
        assert result.success
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_err_operation(self):
        """Test creating error operation results."""
        result = OperationResult.err("File not found", "NOT_FOUND")
        assert not result.success
        assert result.data is None
        assert result.error == "File not found"
        assert result.error_code == "NOT_FOUND"

    def test_json_serialization(self):
        """Test that OperationResult can be serialized to JSON."""
        result = OperationResult.ok({"test": True})
        json_data = result.model_dump_json()
        assert '"success": true' in json_data.lower() or '"success":true' in json_data.lower()


# ============================================================================
# Severity Utility Tests
# ============================================================================


class TestSeverityUtilities:
    """Tests for severity handling utilities."""

    def test_parse_severity_from_enum(self):
        """Test parsing when already an enum."""
        result = parse_severity(FindingSeverity.CRITICAL)
        assert result == FindingSeverity.CRITICAL

    def test_parse_severity_from_string(self):
        """Test parsing from valid string."""
        assert parse_severity("high") == FindingSeverity.HIGH
        assert parse_severity("CRITICAL") == FindingSeverity.CRITICAL

    def test_parse_severity_with_alias(self):
        """Test parsing severity aliases."""
        assert parse_severity("warning") == FindingSeverity.MEDIUM
        assert parse_severity("error") == FindingSeverity.HIGH

    def test_parse_severity_invalid_uses_default(self):
        """Test that invalid values use default."""
        result = parse_severity("invalid", FindingSeverity.LOW)
        assert result == FindingSeverity.LOW

    def test_compare_severity(self):
        """Test severity comparison."""
        assert compare_severity("low", "high") == -1
        assert compare_severity("high", "low") == 1
        assert compare_severity("medium", "medium") == 0

    def test_is_blocking_severity(self):
        """Test blocking severity check."""
        assert is_blocking_severity("critical")
        assert is_blocking_severity("high")
        assert is_blocking_severity("error")  # alias
        assert not is_blocking_severity("medium")
        assert not is_blocking_severity("low")


# ============================================================================
# Timestamp Utility Tests
# ============================================================================


class TestTimestampUtilities:
    """Tests for timestamp handling utilities."""

    def test_parse_iso_datetime_standard(self):
        """Test parsing standard ISO datetime."""
        result = parse_iso_datetime("2024-01-15T10:30:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_iso_datetime_with_z(self):
        """Test parsing ISO datetime with Z suffix."""
        result = parse_iso_datetime("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.year == 2024

    def test_parse_iso_datetime_none_value(self):
        """Test parsing None returns default."""
        result = parse_iso_datetime(None, datetime(2000, 1, 1))
        assert result == datetime(2000, 1, 1)

    def test_parse_iso_datetime_invalid(self):
        """Test parsing invalid string returns default."""
        result = parse_iso_datetime("not a date", datetime(2000, 1, 1))
        assert result == datetime(2000, 1, 1)


# ============================================================================
# Repository Pattern Tests
# ============================================================================


class TestRepositoryPattern:
    """Tests for the repository pattern abstractions."""

    @pytest.mark.asyncio
    async def test_notification_config_repository(self):
        """Test InMemoryNotificationConfigRepository."""
        repo = InMemoryNotificationConfigRepository()
        
        # Add configs for a repo
        config1 = {"channel": "slack", "enabled": True}
        await repo.add_for_repo("owner/repo", config1)
        
        config2 = {"channel": "teams", "enabled": False}
        await repo.add_for_repo("owner/repo", config2)
        
        # Get configs
        configs = await repo.get_by_repo("owner/repo")
        assert len(configs) == 2
        
        # Different repo should be empty
        other_configs = await repo.get_by_repo("other/repo")
        assert len(other_configs) == 0


# ============================================================================
# Rule Evaluation Strategy Tests
# ============================================================================


class TestRuleEvaluationStrategies:
    """Tests for rule evaluation strategy pattern."""

    def test_pattern_strategy_can_evaluate(self):
        """Test PatternRuleStrategy.can_evaluate."""
        strategy = PatternRuleStrategy()
        assert strategy.can_evaluate(RuleType.PATTERN)
        assert not strategy.can_evaluate(RuleType.COMPOSITE)

    def test_composite_strategy_can_evaluate(self):
        """Test CompositeRuleStrategy.can_evaluate."""
        strategy = CompositeRuleStrategy()
        assert strategy.can_evaluate(RuleType.COMPOSITE)
        assert not strategy.can_evaluate(RuleType.PATTERN)

    def test_evaluator_with_custom_strategies(self):
        """Test that RuleEvaluator accepts custom strategies."""
        custom_strategies = [PatternRuleStrategy()]
        rule = (
            RuleBuilder()
            .name("test-rule")
            .pattern(r"TODO")
            .build()
        )
        
        evaluator = RuleEvaluator([rule], strategies=custom_strategies)
        assert len(evaluator._strategies) == 1

    def test_pattern_rule_evaluation(self):
        """Test pattern-based rule evaluation."""
        rule = (
            RuleBuilder()
            .name("no-print")
            .description("Avoid print statements")
            .pattern(r"print\s*\(")
            .severity(RuleSeverity.LOW)
            .build()
        )
        
        evaluator = RuleEvaluator([rule])
        code = """
def greet():
    print("Hello")
"""
        violations = evaluator.evaluate(code, "test.py", "python")
        assert len(violations) == 1
        assert "no-print" in violations[0]["rule_name"]


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for refactored components."""

    def test_severity_with_result(self):
        """Test using severity utilities with Result type."""
        def validate_severity(sev: str) -> Result[FindingSeverity, str]:
            try:
                parsed = parse_severity(sev)
                if parsed == FindingSeverity.INFO:
                    # Treat INFO as invalid for this context
                    return Result.err("INFO severity not allowed")
                return Result.ok(parsed)
            except Exception as e:
                return Result.err(str(e))
        
        result = validate_severity("high")
        assert result.is_ok
        assert result.unwrap() == FindingSeverity.HIGH
        
        result = validate_severity("info")
        assert result.is_err
