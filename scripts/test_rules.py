#!/usr/bin/env python3
"""
Rule Testing Tool for CodeVerify

Validates custom rule definitions and tests them against sample code.

Usage:
    python scripts/test_rules.py validate rules/my-rule.yml
    python scripts/test_rules.py test rules/my-rule.yml sample.py
    python scripts/test_rules.py generate-template > new-rule.yml
    python scripts/test_rules.py lint .codeverify.yml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


class RuleValidator:
    """Validator for CodeVerify custom rules."""
    
    VALID_SEVERITIES = {"critical", "high", "medium", "low", "info"}
    VALID_CHECKS = {"null_safety", "array_bounds", "integer_overflow", "division_by_zero", "loop_termination"}
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: list[str] = []
        self.warnings: list[str] = []
    
    def validate_rule(self, rule_data: dict[str, Any]) -> bool:
        """Validate a single rule definition."""
        self.errors = []
        self.warnings = []
        
        # Required fields
        if "id" not in rule_data:
            self.errors.append("Missing required field: 'id'")
        else:
            rule_id = rule_data["id"]
            if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", rule_id):
                self.errors.append(f"Invalid rule ID: '{rule_id}' (use lowercase, numbers, hyphens, no leading/trailing hyphens)")
        
        if "name" not in rule_data:
            self.errors.append("Missing required field: 'name'")
        
        # Detection method
        has_pattern = "pattern" in rule_data
        has_prompt = "prompt" in rule_data
        
        if not has_pattern and not has_prompt:
            self.errors.append("Rule must define either 'pattern' (regex) or 'prompt' (AI)")
        
        if has_pattern and has_prompt:
            self.warnings.append("Rule has both 'pattern' and 'prompt'; pattern takes precedence")
        
        # Validate pattern
        if has_pattern:
            pattern = rule_data["pattern"]
            try:
                re.compile(pattern)
            except re.error as e:
                self.errors.append(f"Invalid regex pattern: {e}")
        
        # Validate severity
        severity = rule_data.get("severity", "medium")
        if severity not in self.VALID_SEVERITIES:
            self.errors.append(f"Invalid severity: '{severity}' (must be one of: {', '.join(self.VALID_SEVERITIES)})")
        
        # Validate enabled flag
        enabled = rule_data.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            self.errors.append(f"'enabled' must be a boolean, got: {type(enabled).__name__}")
        
        # Validate apply_to patterns
        apply_to = rule_data.get("apply_to", [])
        if apply_to and not isinstance(apply_to, list):
            self.errors.append("'apply_to' must be a list of file patterns")
        
        # Validate exclude patterns
        exclude = rule_data.get("exclude", [])
        if exclude and not isinstance(exclude, list):
            self.errors.append("'exclude' must be a list of file patterns")
        
        # Validate tags
        tags = rule_data.get("tags", [])
        if tags and not isinstance(tags, list):
            self.errors.append("'tags' must be a list of strings")
        
        return len(self.errors) == 0
    
    def validate_config(self, config_data: dict[str, Any]) -> bool:
        """Validate a full .codeverify.yml configuration."""
        self.errors = []
        self.warnings = []
        
        # Check version
        version = config_data.get("version")
        if version is None:
            self.warnings.append("Missing 'version' field (defaulting to '1')")
        elif str(version) not in ("1", "1.0"):
            self.warnings.append(f"Unknown config version: {version}")
        
        # Check languages
        languages = config_data.get("languages", [])
        valid_languages = {"python", "typescript", "javascript", "go", "java", "rust", "cpp", "c", "csharp"}
        for lang in languages:
            if lang not in valid_languages:
                self.warnings.append(f"Unknown language: {lang}")
        
        # Check thresholds
        thresholds = config_data.get("thresholds", {})
        valid_threshold_keys = {"critical", "high", "medium", "low"}
        for key, value in thresholds.items():
            if key not in valid_threshold_keys:
                self.warnings.append(f"Unknown threshold: {key}")
            elif not isinstance(value, int) or value < 0:
                self.errors.append(f"Threshold '{key}' must be a non-negative integer")
        
        # Check verification settings
        verification = config_data.get("verification", {})
        if verification:
            timeout = verification.get("timeout_seconds")
            if timeout is not None:
                if not isinstance(timeout, int) or timeout < 1:
                    self.errors.append("verification.timeout_seconds must be a positive integer")
                elif timeout > 300:
                    self.warnings.append("verification.timeout_seconds is very high (>300s)")
            
            checks = verification.get("checks", [])
            for check in checks:
                if check not in self.VALID_CHECKS:
                    self.warnings.append(f"Unknown verification check: {check}")
        
        # Validate custom rules
        custom_rules = config_data.get("custom_rules", [])
        for i, rule in enumerate(custom_rules):
            if not isinstance(rule, dict):
                self.errors.append(f"custom_rules[{i}] must be an object")
                continue
            
            rule_valid = self.validate_rule(rule)
            if not rule_valid:
                for error in self.errors:
                    if not error.startswith("custom_rules"):
                        self.errors.append(f"custom_rules[{i}]: {error}")
        
        # Validate ignore rules
        ignore = config_data.get("ignore", [])
        for i, rule in enumerate(ignore):
            if not isinstance(rule, dict):
                self.errors.append(f"ignore[{i}] must be an object")
                continue
            if "pattern" not in rule:
                self.errors.append(f"ignore[{i}]: missing required 'pattern' field")
        
        return len(self.errors) == 0


class RuleTester:
    """Test rules against sample code."""
    
    def __init__(self, rule_data: dict[str, Any]):
        self.rule = rule_data
        self.pattern = None
        if "pattern" in rule_data:
            self.pattern = re.compile(rule_data["pattern"])
    
    def test_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Test rule against a file."""
        if not self.pattern:
            return []
        
        matches: list[dict[str, Any]] = []
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
            return []
        
        lines = content.split("\n")
        
        for i, line in enumerate(lines, 1):
            match = self.pattern.search(line)
            if match:
                matches.append({
                    "line": i,
                    "column": match.start() + 1,
                    "match": match.group(),
                    "snippet": line.strip()[:100],
                })
        
        return matches
    
    def test_string(self, code: str) -> list[dict[str, Any]]:
        """Test rule against a string of code."""
        if not self.pattern:
            return []
        
        matches: list[dict[str, Any]] = []
        lines = code.split("\n")
        
        for i, line in enumerate(lines, 1):
            match = self.pattern.search(line)
            if match:
                matches.append({
                    "line": i,
                    "column": match.start() + 1,
                    "match": match.group(),
                    "snippet": line.strip()[:100],
                })
        
        return matches


def generate_template() -> str:
    """Generate a template rule file."""
    return """# CodeVerify Custom Rule Definition
# Documentation: https://codeverify.dev/docs/custom-rules

# Unique identifier for this rule (lowercase, numbers, hyphens)
id: my-custom-rule

# Human-readable name
name: My Custom Rule

# Detailed description
description: |
  This rule checks for...
  
  Why this matters:
  - Reason 1
  - Reason 2

# Severity level: critical, high, medium, low, info
severity: medium

# Detection: use either 'pattern' (regex) or 'prompt' (AI-based)
# Pattern-based detection (regex)
pattern: "dangerous_function\\s*\\("

# AI-based detection (requires API connection)
# prompt: |
#   Check if this code contains calls to dangerous functions.
#   Return findings with specific line numbers.

# Enable/disable the rule
enabled: true

# Optional: file patterns to apply this rule to
apply_to:
  - "*.py"
  - "*.js"
  - "*.ts"

# Optional: file patterns to exclude
exclude:
  - "**/tests/**"
  - "**/test_*.py"
  - "**/*.test.ts"

# Optional: categorization tags
tags:
  - security
  - best-practices

# Optional: suggested fix template
fix_template: |
  Replace dangerous_function() with safe_alternative()

# Optional: links to documentation
references:
  - https://example.com/security-best-practices
  - https://owasp.org/relevant-topic
"""


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a rule file."""
    rule_path = Path(args.rule_file)
    
    if not rule_path.exists():
        print(f"Error: File not found: {rule_path}", file=sys.stderr)
        return 1
    
    try:
        content = rule_path.read_text()
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML: {e}", file=sys.stderr)
        return 1
    
    validator = RuleValidator(verbose=args.verbose)
    is_valid = validator.validate_rule(data)
    
    if validator.errors:
        print("Errors:")
        for error in validator.errors:
            print(f"  ✗ {error}")
    
    if validator.warnings:
        print("Warnings:")
        for warning in validator.warnings:
            print(f"  ⚠ {warning}")
    
    if is_valid:
        print(f"✓ Rule '{data.get('id', 'unknown')}' is valid")
        return 0
    else:
        print(f"✗ Rule validation failed")
        return 1


def cmd_test(args: argparse.Namespace) -> int:
    """Test a rule against sample code."""
    rule_path = Path(args.rule_file)
    test_path = Path(args.test_file)
    
    if not rule_path.exists():
        print(f"Error: Rule file not found: {rule_path}", file=sys.stderr)
        return 1
    
    if not test_path.exists():
        print(f"Error: Test file not found: {test_path}", file=sys.stderr)
        return 1
    
    try:
        rule_data = yaml.safe_load(rule_path.read_text())
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML: {e}", file=sys.stderr)
        return 1
    
    # Validate first
    validator = RuleValidator()
    if not validator.validate_rule(rule_data):
        print("Error: Invalid rule definition")
        for error in validator.errors:
            print(f"  ✗ {error}")
        return 1
    
    # Test
    tester = RuleTester(rule_data)
    matches = tester.test_file(test_path)
    
    print(f"Testing rule '{rule_data.get('id')}' against {test_path}")
    print(f"Pattern: {rule_data.get('pattern', 'N/A')}")
    print()
    
    if matches:
        print(f"Found {len(matches)} match(es):")
        for match in matches:
            print(f"  Line {match['line']}: {match['snippet']}")
    else:
        print("No matches found")
    
    if args.output_json:
        print("\nJSON output:")
        print(json.dumps({
            "rule_id": rule_data.get("id"),
            "test_file": str(test_path),
            "matches": matches,
        }, indent=2))
    
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    """Lint a .codeverify.yml configuration."""
    config_path = Path(args.config_file)
    
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        return 1
    
    try:
        content = config_path.read_text()
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML: {e}", file=sys.stderr)
        return 1
    
    validator = RuleValidator(verbose=args.verbose)
    is_valid = validator.validate_config(data)
    
    if validator.errors:
        print("Errors:")
        for error in validator.errors:
            print(f"  ✗ {error}")
    
    if validator.warnings:
        print("Warnings:")
        for warning in validator.warnings:
            print(f"  ⚠ {warning}")
    
    if is_valid and not validator.warnings:
        print(f"✓ Configuration is valid")
        return 0
    elif is_valid:
        print(f"✓ Configuration is valid (with warnings)")
        return 0
    else:
        print(f"✗ Configuration validation failed")
        return 1


def cmd_generate_template(args: argparse.Namespace) -> int:
    """Generate a rule template."""
    print(generate_template())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CodeVerify Rule Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate rules/no-eval.yml
  %(prog)s test rules/no-eval.yml src/app.py
  %(prog)s lint .codeverify.yml
  %(prog)s generate-template > new-rule.yml
        """
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a rule file")
    validate_parser.add_argument("rule_file", help="Path to rule YAML file")
    
    # test command
    test_parser = subparsers.add_parser("test", help="Test rule against sample code")
    test_parser.add_argument("rule_file", help="Path to rule YAML file")
    test_parser.add_argument("test_file", help="Path to test file")
    test_parser.add_argument("--json", dest="output_json", action="store_true", help="Output JSON")
    
    # lint command
    lint_parser = subparsers.add_parser("lint", help="Lint .codeverify.yml")
    lint_parser.add_argument("config_file", nargs="?", default=".codeverify.yml", help="Config file path")
    
    # generate-template command
    template_parser = subparsers.add_parser("generate-template", help="Generate rule template")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "validate":
        return cmd_validate(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "lint":
        return cmd_lint(args)
    elif args.command == "generate-template":
        return cmd_generate_template(args)
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
