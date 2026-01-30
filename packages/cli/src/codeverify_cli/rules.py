"""Custom rule handling for CLI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def test_custom_rule(rule_path: Path, test_file: Path | None = None) -> dict[str, Any]:
    """Test a custom rule definition.
    
    Args:
        rule_path: Path to YAML file containing rule definition
        test_file: Optional file to test the rule against
    
    Returns:
        Dict with validation results
    """
    results: dict[str, Any] = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "matches": [],
    }
    
    # Load rule file
    try:
        content = rule_path.read_text()
        rule_data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        results["errors"].append(f"Invalid YAML: {e}")
        return results
    except Exception as e:
        results["errors"].append(f"Failed to read file: {e}")
        return results
    
    if not isinstance(rule_data, dict):
        results["errors"].append("Rule must be a YAML object")
        return results
    
    # Validate required fields
    required_fields = ["id", "name"]
    for field in required_fields:
        if field not in rule_data:
            results["errors"].append(f"Missing required field: {field}")
    
    if results["errors"]:
        return results
    
    # Validate rule content
    rule_id = rule_data.get("id", "")
    if not re.match(r"^[a-z0-9-]+$", rule_id):
        results["errors"].append(f"Invalid rule ID format: {rule_id} (use lowercase letters, numbers, hyphens)")
    
    # Check for pattern or prompt
    pattern = rule_data.get("pattern")
    prompt = rule_data.get("prompt")
    
    if not pattern and not prompt:
        results["errors"].append("Rule must have either 'pattern' or 'prompt'")
        return results
    
    # Validate pattern if provided
    if pattern:
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            results["errors"].append(f"Invalid regex pattern: {e}")
            return results
    
    # Validate severity
    severity = rule_data.get("severity", "medium")
    valid_severities = {"critical", "high", "medium", "low", "info"}
    if severity not in valid_severities:
        results["warnings"].append(f"Unknown severity: {severity}")
    
    # Mark as valid if no errors
    if not results["errors"]:
        results["valid"] = True
    
    # Test against file if provided
    if test_file and pattern and results["valid"]:
        try:
            test_content = test_file.read_text()
            compiled = re.compile(pattern)
            
            for i, line in enumerate(test_content.split("\n"), 1):
                if compiled.search(line):
                    results["matches"].append({
                        "line": i,
                        "snippet": line.strip(),
                    })
        except Exception as e:
            results["warnings"].append(f"Test failed: {e}")
    
    return results


def load_custom_rules(config_path: Path) -> list[dict[str, Any]]:
    """Load custom rules from configuration."""
    if not config_path.exists():
        return []
    
    try:
        content = config_path.read_text()
        data = yaml.safe_load(content) or {}
        return data.get("custom_rules", [])
    except Exception:
        return []


def apply_custom_rules(content: str, file_path: Path, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply custom rules to file content.
    
    Returns list of findings.
    """
    findings: list[dict[str, Any]] = []
    lines = content.split("\n")
    
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        
        pattern = rule.get("pattern")
        if not pattern:
            continue
        
        try:
            compiled = re.compile(pattern)
        except re.error:
            continue
        
        for i, line in enumerate(lines, 1):
            if compiled.search(line):
                findings.append({
                    "category": "custom",
                    "severity": rule.get("severity", "medium"),
                    "title": rule.get("name", rule.get("id", "Custom Rule")),
                    "description": rule.get("description", ""),
                    "file_path": str(file_path),
                    "line_start": i,
                    "line_end": i,
                    "confidence": 1.0,
                    "verification_type": "custom_rule",
                    "rule_id": rule.get("id"),
                })
    
    return findings


def create_rule_template() -> str:
    """Generate a template for custom rule definition."""
    return """# Custom Rule Definition
# Save this as a .yml file and reference it in your .codeverify.yml

id: my-custom-rule
name: My Custom Rule
description: |
  Detailed description of what this rule checks for
  and why it's important.

# Severity: critical, high, medium, low, info
severity: medium

# Pattern-based detection (regex)
pattern: "dangerous_function\\s*\\("

# OR AI-based detection (requires API connection)
# prompt: |
#   Check if the code contains calls to dangerous functions
#   that should be avoided in production code.

# Enable/disable the rule
enabled: true

# Optional: specific file patterns to apply this rule
# apply_to:
#   - "*.py"
#   - "src/**/*.ts"

# Optional: file patterns to exclude
# exclude:
#   - "**/tests/**"
#   - "**/*.test.*"

# Optional: tags for organization
tags:
  - security
  - best-practices
"""
