"""Tests for config_validator module."""

from __future__ import annotations

import pytest

from codeverify_core.config_validator import (
    DEFAULT_CONFIG,
    ConfigValidator,
)


@pytest.fixture
def validator():
    return ConfigValidator()


class TestConfigValidator:
    def test_valid_default_config(self, validator):
        result = validator.validate(DEFAULT_CONFIG)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_required_version(self, validator):
        result = validator.validate({"ai": {"enabled": True}})
        assert result.valid is False
        assert any("version" in e.message for e in result.errors)

    def test_invalid_version_value(self, validator):
        result = validator.validate({"version": "99"})
        assert result.valid is False
        assert any("not in allowed" in e.message for e in result.errors)

    def test_wrong_type_timeout(self, validator):
        config = {"version": "2", "verification": {"timeout_seconds": "abc"}}
        result = validator.validate(config)
        assert result.valid is False
        assert any("integer" in e.message.lower() for e in result.errors)

    def test_out_of_range_timeout(self, validator):
        config = {"version": "2", "verification": {"timeout_seconds": 9999}}
        result = validator.validate(config)
        assert result.valid is False
        assert any("maximum" in e.message for e in result.errors)

    def test_invalid_check_name(self, validator):
        config = {"version": "2", "verification": {"checks": ["nonexistent_check"]}}
        result = validator.validate(config)
        assert result.valid is False
        assert any("not in allowed" in e.message for e in result.errors)

    def test_unknown_key_warning(self, validator):
        config = {"version": "2", "mystery_key": True}
        result = validator.validate(config)
        assert result.valid is True  # Warnings don't block
        assert len(result.warnings) > 0

    def test_invalid_language(self, validator):
        config = {"version": "2", "languages": ["python", "brainfuck"]}
        result = validator.validate(config)
        assert result.valid is False

    def test_valid_custom_rules(self, validator):
        config = {
            "version": "2",
            "custom_rules": [
                {"id": "no-eval", "pattern": "eval\\(", "severity": "critical"},
            ],
        }
        result = validator.validate(config)
        assert result.valid is True

    def test_custom_rule_missing_required(self, validator):
        config = {
            "version": "2",
            "custom_rules": [{"severity": "high"}],
        }
        result = validator.validate(config)
        assert result.valid is False
        assert any("id" in e.message for e in result.errors)

    def test_invalid_temperature(self, validator):
        config = {"version": "2", "ai": {"temperature": 5.0}}
        result = validator.validate(config)
        assert result.valid is False
        assert any("maximum" in e.message for e in result.errors)

    def test_boolean_type_check(self, validator):
        config = {"version": "2", "verification": {"enabled": "yes"}}
        result = validator.validate(config)
        assert result.valid is False

    def test_nested_dict_validation(self, validator):
        config = {"version": "2", "severity_thresholds": {"critical": -1}}
        result = validator.validate(config)
        assert result.valid is False
        assert any("minimum" in e.message for e in result.errors)


class TestConfigGeneration:
    def test_generate_default(self, validator):
        default = validator.generate_default()
        assert default["version"] == "2"
        assert default["verification"]["enabled"] is True
        result = validator.validate(default)
        assert result.valid is True

    def test_generate_default_is_deep_copy(self, validator):
        d1 = validator.generate_default()
        d2 = validator.generate_default()
        d1["version"] = "1"
        assert d2["version"] == "2"


class TestMigration:
    def test_v1_to_v2_timeout(self, validator):
        v1 = {"version": "1", "timeout": 60}
        v2 = validator.migrate_v1_to_v2(v1)
        assert v2["version"] == "2"
        assert v2["verification"]["timeout_seconds"] == 60
        assert "timeout" not in v2

    def test_v1_to_v2_flat_checks(self, validator):
        v1 = {"version": "1", "checks": ["null_safety"]}
        v2 = validator.migrate_v1_to_v2(v1)
        assert v2["verification"]["checks"] == ["null_safety"]
        assert "checks" not in v2

    def test_migration_preserves_other_keys(self, validator):
        v1 = {"version": "1", "languages": ["python"]}
        v2 = validator.migrate_v1_to_v2(v1)
        assert v2["languages"] == ["python"]
