"""Unit tests for cli/config.py — YAML config loading & validation."""

from __future__ import annotations

import pytest
from codeverify_cli.config import CLIConfig, load_config, merge_configs, validate_config


class TestCLIConfigDefaults:
    """Test CLIConfig default values."""

    def test_default_version(self):
        c = CLIConfig()
        assert c.version == "1"

    def test_default_languages(self):
        c = CLIConfig()
        assert "python" in c.languages
        assert "typescript" in c.languages

    def test_default_thresholds(self):
        c = CLIConfig()
        assert c.thresholds["critical"] == 0
        assert c.thresholds["high"] == 0
        assert c.thresholds["medium"] == 5
        assert c.thresholds["low"] == 10

    def test_default_verification(self):
        c = CLIConfig()
        assert c.verification["enabled"] is True
        assert c.verification["timeout_seconds"] == 30

    def test_default_ai_disabled(self):
        c = CLIConfig()
        assert c.ai["enabled"] is False


class TestLoadConfig:
    """Test load_config from YAML files."""

    def test_returns_defaults_for_nonexistent_file(self, tmp_path):
        cfg = load_config(tmp_path / "missing.yml")
        assert cfg.version == "1"
        assert cfg.languages == ["python", "typescript"]

    def test_loads_valid_yaml(self, tmp_path):
        config_file = tmp_path / ".codeverify.yml"
        config_file.write_text(
            'version: "1"\nlanguages:\n  - go\n  - rust\nexclude:\n  - "vendor/**"\n'
        )
        cfg = load_config(config_file)
        assert cfg.languages == ["go", "rust"]
        assert "vendor/**" in cfg.exclude

    def test_loads_partial_config(self, tmp_path):
        config_file = tmp_path / ".codeverify.yml"
        config_file.write_text('version: "1"\n')
        cfg = load_config(config_file)
        assert cfg.version == "1"
        assert cfg.languages == ["python", "typescript"]

    def test_handles_empty_yaml(self, tmp_path):
        config_file = tmp_path / ".codeverify.yml"
        config_file.write_text("")
        cfg = load_config(config_file)
        assert cfg.version == "1"

    def test_raises_for_malformed_yaml(self, tmp_path):
        config_file = tmp_path / ".codeverify.yml"
        config_file.write_text("{{invalid: yaml: [[[")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(config_file)

    def test_loads_custom_rules(self, tmp_path):
        config_file = tmp_path / ".codeverify.yml"
        config_file.write_text(
            "custom_rules:\n  - id: no-print\n    name: No Print\n    pattern: 'print\\\\('\n"
        )
        cfg = load_config(config_file)
        assert len(cfg.custom_rules) == 1
        assert cfg.custom_rules[0]["id"] == "no-print"

    def test_loads_thresholds(self, tmp_path):
        config_file = tmp_path / ".codeverify.yml"
        config_file.write_text("thresholds:\n  critical: 1\n  high: 2\n")
        cfg = load_config(config_file)
        assert cfg.thresholds == {"critical": 1, "high": 2}


class TestValidateConfig:
    """Test validate_config error and warning detection."""

    def test_missing_file_returns_error(self, tmp_path):
        errors, warnings = validate_config(tmp_path / "missing.yml")
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_invalid_yaml_returns_error(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("{{bad yaml")
        errors, warnings = validate_config(f)
        assert any("Invalid YAML" in e for e in errors)

    def test_non_dict_returns_error(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("- item1\n- item2\n")
        errors, warnings = validate_config(f)
        assert any("must be a YAML object" in e for e in errors)

    def test_unknown_version_warns(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("version: 99\n")
        errors, warnings = validate_config(f)
        assert any("Unknown config version" in w for w in warnings)

    def test_known_version_no_warning(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text('version: "1"\n')
        errors, warnings = validate_config(f)
        assert not errors
        assert not warnings

    def test_unknown_language_warns(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("languages:\n  - brainfuck\n")
        errors, warnings = validate_config(f)
        assert any("Unknown language" in w for w in warnings)

    def test_non_integer_threshold_errors(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("thresholds:\n  critical: 'many'\n")
        errors, warnings = validate_config(f)
        assert any("must be an integer" in e for e in errors)

    def test_unknown_threshold_key_warns(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("thresholds:\n  ultralow: 5\n")
        errors, warnings = validate_config(f)
        assert any("Unknown threshold key" in w for w in warnings)

    def test_high_timeout_warns(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("verification:\n  timeout_seconds: 999\n")
        errors, warnings = validate_config(f)
        assert any("very high" in w for w in warnings)

    def test_negative_timeout_errors(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("verification:\n  timeout_seconds: -1\n")
        errors, warnings = validate_config(f)
        assert any("positive integer" in e for e in errors)

    def test_unknown_check_warns(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("verification:\n  checks:\n    - telekinesis\n")
        errors, warnings = validate_config(f)
        assert any("Unknown verification check" in w for w in warnings)

    def test_custom_rule_missing_id_errors(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("custom_rules:\n  - name: Test\n")
        errors, warnings = validate_config(f)
        assert any("missing required 'id'" in e for e in errors)

    def test_custom_rule_missing_name_errors(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("custom_rules:\n  - id: test\n")
        errors, warnings = validate_config(f)
        assert any("missing required 'name'" in e for e in errors)

    def test_custom_rule_no_pattern_warns(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("custom_rules:\n  - id: test\n    name: Test\n")
        errors, warnings = validate_config(f)
        assert any("no 'pattern' or 'prompt'" in w for w in warnings)

    def test_custom_rule_invalid_severity_warns(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("custom_rules:\n  - id: test\n    name: Test\n    severity: extreme\n")
        errors, warnings = validate_config(f)
        assert any("invalid severity" in w for w in warnings)

    def test_ignore_missing_pattern_errors(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text("ignore:\n  - reason: testing\n")
        errors, warnings = validate_config(f)
        assert any("missing required 'pattern'" in e for e in errors)

    def test_valid_config_no_errors(self, tmp_path):
        f = tmp_path / ".codeverify.yml"
        f.write_text(
            'version: "1"\n'
            "languages:\n  - python\n"
            "thresholds:\n  critical: 0\n"
            "verification:\n  timeout_seconds: 30\n  checks:\n    - null_safety\n"
        )
        errors, warnings = validate_config(f)
        assert not errors
        assert not warnings


class TestMergeConfigs:
    """Test merge_configs."""

    def test_override_takes_precedence(self):
        base = CLIConfig(languages=["python"])
        override = CLIConfig(languages=["go"])
        merged = merge_configs(base, override)
        assert merged.languages == ["go"]

    def test_excludes_are_combined(self):
        base = CLIConfig(exclude=["a"])
        override = CLIConfig(exclude=["b"])
        merged = merge_configs(base, override)
        assert set(merged.exclude) == {"a", "b"}

    def test_thresholds_are_merged(self):
        base = CLIConfig(thresholds={"critical": 0, "high": 0})
        override = CLIConfig(thresholds={"high": 5, "medium": 10})
        merged = merge_configs(base, override)
        assert merged.thresholds == {"critical": 0, "high": 5, "medium": 10}

    def test_custom_rules_are_concatenated(self):
        base = CLIConfig(custom_rules=[{"id": "a"}])
        override = CLIConfig(custom_rules=[{"id": "b"}])
        merged = merge_configs(base, override)
        assert len(merged.custom_rules) == 2
