"""Configuration handling for CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CLIConfig:
    """CLI configuration loaded from .codeverify.yml."""
    
    version: str = "1"
    languages: list[str] = field(default_factory=lambda: ["python", "typescript"])
    include: list[str] = field(default_factory=lambda: ["**/*"])
    exclude: list[str] = field(default_factory=list)
    thresholds: dict[str, int] = field(default_factory=lambda: {
        "critical": 0,
        "high": 0,
        "medium": 5,
        "low": 10,
    })
    verification: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "timeout_seconds": 30,
        "checks": ["null_safety", "array_bounds", "integer_overflow", "division_by_zero"],
    })
    ai: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,  # Disabled by default for local CLI
        "semantic_analysis": False,
        "security_analysis": False,
    })
    custom_rules: list[dict[str, Any]] = field(default_factory=list)
    ignore: list[dict[str, Any]] = field(default_factory=list)


def load_config(path: Path) -> CLIConfig:
    """Load configuration from YAML file."""
    if not path.exists():
        return CLIConfig()
    
    try:
        content = path.read_text()
        data = yaml.safe_load(content) or {}
        
        config = CLIConfig(
            version=data.get("version", "1"),
            languages=data.get("languages", ["python", "typescript"]),
            include=data.get("include", ["**/*"]),
            exclude=data.get("exclude", []),
            thresholds=data.get("thresholds", {}),
            verification=data.get("verification", {}),
            ai=data.get("ai", {}),
            custom_rules=data.get("custom_rules", []),
            ignore=data.get("ignore", []),
        )
        
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")


def validate_config(path: Path) -> tuple[list[str], list[str]]:
    """Validate configuration file.
    
    Returns (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []
    
    if not path.exists():
        errors.append(f"Configuration file not found: {path}")
        return errors, warnings
    
    try:
        content = path.read_text()
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML syntax: {e}")
        return errors, warnings
    
    if not isinstance(data, dict):
        errors.append("Configuration must be a YAML object")
        return errors, warnings
    
    # Check version
    version = data.get("version")
    if version and str(version) not in ("1", "1.0"):
        warnings.append(f"Unknown config version: {version}")
    
    # Check languages
    languages = data.get("languages", [])
    valid_languages = {"python", "typescript", "javascript", "go", "java", "rust", "cpp", "c"}
    if languages:
        for lang in languages:
            if lang not in valid_languages:
                warnings.append(f"Unknown language: {lang}")
    
    # Check thresholds
    thresholds = data.get("thresholds", {})
    if thresholds:
        for key in thresholds:
            if key not in ("critical", "high", "medium", "low"):
                warnings.append(f"Unknown threshold key: {key}")
            elif not isinstance(thresholds[key], int):
                errors.append(f"Threshold '{key}' must be an integer")
    
    # Check verification config
    verification = data.get("verification", {})
    if verification:
        if "timeout_seconds" in verification:
            timeout = verification["timeout_seconds"]
            if not isinstance(timeout, int) or timeout < 1:
                errors.append("verification.timeout_seconds must be a positive integer")
            elif timeout > 300:
                warnings.append("verification.timeout_seconds is very high (>300s)")
        
        checks = verification.get("checks", [])
        valid_checks = {"null_safety", "array_bounds", "integer_overflow", "division_by_zero", "loop_termination"}
        for check in checks:
            if check not in valid_checks:
                warnings.append(f"Unknown verification check: {check}")
    
    # Check custom rules
    custom_rules = data.get("custom_rules", [])
    for i, rule in enumerate(custom_rules):
        if not isinstance(rule, dict):
            errors.append(f"custom_rules[{i}] must be an object")
            continue
        
        if "id" not in rule:
            errors.append(f"custom_rules[{i}] missing required 'id' field")
        if "name" not in rule:
            errors.append(f"custom_rules[{i}] missing required 'name' field")
        
        if "pattern" not in rule and "prompt" not in rule:
            warnings.append(f"custom_rules[{i}] has no 'pattern' or 'prompt'")
        
        severity = rule.get("severity", "medium")
        if severity not in ("critical", "high", "medium", "low", "info"):
            warnings.append(f"custom_rules[{i}] has invalid severity: {severity}")
    
    # Check ignore rules
    ignore = data.get("ignore", [])
    for i, rule in enumerate(ignore):
        if not isinstance(rule, dict):
            errors.append(f"ignore[{i}] must be an object")
            continue
        
        if "pattern" not in rule:
            errors.append(f"ignore[{i}] missing required 'pattern' field")
    
    return errors, warnings


def merge_configs(base: CLIConfig, override: CLIConfig) -> CLIConfig:
    """Merge two configurations, with override taking precedence."""
    return CLIConfig(
        version=override.version or base.version,
        languages=override.languages or base.languages,
        include=override.include or base.include,
        exclude=list(set(base.exclude + override.exclude)),
        thresholds={**base.thresholds, **override.thresholds},
        verification={**base.verification, **override.verification},
        ai={**base.ai, **override.ai},
        custom_rules=base.custom_rules + override.custom_rules,
        ignore=base.ignore + override.ignore,
    )
