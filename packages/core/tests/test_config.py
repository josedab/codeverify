"""Tests for configuration module."""
import pytest
from codeverify_core.config import (
    CodeVerifyConfig,
    parse_config,
    should_analyze_file,
    should_ignore_finding,
    passes_thresholds,
)


class TestParseConfig:
    """Tests for parse_config function."""
    
    def test_parse_empty_config(self):
        """Empty config returns defaults."""
        config = parse_config("")
        assert config.version == "1"
        assert "python" in config.languages
        assert config.verification.enabled is True
        assert config.ai.enabled is True
    
    def test_parse_minimal_config(self):
        """Minimal YAML config parses correctly."""
        yaml_content = """
version: "2"
languages:
  - python
"""
        config = parse_config(yaml_content)
        assert config.version == "2"
        assert config.languages == ["python"]
    
    def test_parse_full_config(self):
        """Full config with all options."""
        yaml_content = """
version: "1"
languages:
  - python
  - typescript
include:
  - "src/**/*.py"
exclude:
  - "venv/**"
verification:
  enabled: true
  timeout: 60
  checks:
    - null_safety
ai:
  enabled: false
  model: claude-3
thresholds:
  critical: 0
  high: 2
  medium: 10
  low: 50
ignore:
  - pattern: "migrations/**"
    reason: "Generated"
rules:
  - id: test-rule
    name: Test Rule
    description: A test rule
    severity: high
    pattern: "test_pattern"
auto_approve: true
comment_on_pass: false
"""
        config = parse_config(yaml_content)
        
        assert config.languages == ["python", "typescript"]
        assert config.include_patterns == ["src/**/*.py"]
        assert config.exclude_patterns == ["venv/**"]
        assert config.verification.timeout_seconds == 60
        assert config.verification.checks == ["null_safety"]
        assert config.ai.enabled is False
        assert config.ai.model == "claude-3"
        assert config.thresholds.high == 2
        assert len(config.ignore_rules) == 1
        assert config.ignore_rules[0].pattern == "migrations/**"
        assert len(config.custom_rules) == 1
        assert config.custom_rules[0].id == "test-rule"
        assert config.auto_approve is True
        assert config.comment_on_pass is False


class TestShouldAnalyzeFile:
    """Tests for should_analyze_file function."""
    
    def test_include_python_file(self):
        """Python file in default config is included."""
        config = CodeVerifyConfig()
        assert should_analyze_file(config, "src/main.py") is True
    
    def test_exclude_node_modules(self):
        """Node modules are excluded."""
        config = CodeVerifyConfig()
        assert should_analyze_file(config, "node_modules/lodash/index.js") is False
    
    def test_exclude_venv(self):
        """Virtual environment is excluded."""
        config = CodeVerifyConfig()
        assert should_analyze_file(config, "venv/lib/site-packages/foo.py") is False
    
    def test_exclude_takes_precedence(self):
        """Exclude patterns override include patterns."""
        config = CodeVerifyConfig(
            include_patterns=["**/*.py"],
            exclude_patterns=["tests/**"],
        )
        assert should_analyze_file(config, "tests/test_main.py") is False


class TestShouldIgnoreFinding:
    """Tests for should_ignore_finding function."""
    
    def test_no_ignore_rules(self):
        """No ignore rules means nothing is ignored."""
        config = CodeVerifyConfig()
        ignored, reason = should_ignore_finding(config, "src/main.py", "security")
        assert ignored is False
    
    def test_ignore_by_pattern(self):
        """Ignore rule by file pattern."""
        from codeverify_core.config import IgnoreRule
        config = CodeVerifyConfig(ignore_rules=[
            IgnoreRule(pattern="migrations/**", reason="Auto-generated")
        ])
        
        ignored, reason = should_ignore_finding(
            config, "migrations/001_initial.py", "security"
        )
        assert ignored is True
        assert reason == "Auto-generated"
    
    def test_ignore_by_category(self):
        """Ignore rule by category."""
        from codeverify_core.config import IgnoreRule
        config = CodeVerifyConfig(ignore_rules=[
            IgnoreRule(
                pattern="tests/**",
                categories=["security"],
                reason="Test code"
            )
        ])
        
        # Security in tests is ignored
        ignored, reason = should_ignore_finding(
            config, "tests/test_auth.py", "security"
        )
        assert ignored is True
        
        # Logic errors in tests are NOT ignored
        ignored, reason = should_ignore_finding(
            config, "tests/test_auth.py", "logic_error"
        )
        assert ignored is False


class TestPassesThresholds:
    """Tests for passes_thresholds function."""
    
    def test_passes_with_no_findings(self):
        """No findings passes."""
        config = CodeVerifyConfig()
        passed, msg = passes_thresholds(config, {})
        assert passed is True
    
    def test_passes_within_thresholds(self):
        """Findings within thresholds pass."""
        config = CodeVerifyConfig()
        config.thresholds.medium = 5
        config.thresholds.low = 10
        
        passed, msg = passes_thresholds(config, {
            "critical": 0,
            "high": 0,
            "medium": 3,
            "low": 5,
        })
        assert passed is True
    
    def test_fails_critical(self):
        """Any critical finding fails by default."""
        config = CodeVerifyConfig()
        passed, msg = passes_thresholds(config, {"critical": 1})
        assert passed is False
        assert "Critical" in msg
    
    def test_fails_high(self):
        """High findings over threshold fails."""
        config = CodeVerifyConfig()
        passed, msg = passes_thresholds(config, {"high": 1})
        assert passed is False
        assert "High" in msg
    
    def test_custom_thresholds(self):
        """Custom thresholds are respected."""
        from codeverify_core.config import SeverityThresholds
        config = CodeVerifyConfig(thresholds=SeverityThresholds(
            critical=1,  # Allow 1 critical
            high=5,
            medium=20,
            low=100,
        ))
        
        passed, msg = passes_thresholds(config, {
            "critical": 1,
            "high": 3,
            "medium": 15,
            "low": 50,
        })
        assert passed is True
