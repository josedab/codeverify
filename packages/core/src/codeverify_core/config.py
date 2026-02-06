"""Configuration file parser for .codeverify.yml files."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SeverityThresholds:
    """Thresholds for pass/fail determination."""
    
    critical: int = 0  # Max allowed critical findings
    high: int = 0      # Max allowed high findings
    medium: int = 5    # Max allowed medium findings
    low: int = 10      # Max allowed low findings


@dataclass
class VerificationConfig:
    """Configuration for formal verification."""
    
    enabled: bool = True
    timeout_seconds: int = 30
    checks: list[str] = field(default_factory=lambda: [
        "null_safety",
        "array_bounds",
        "integer_overflow",
        "division_by_zero",
    ])


@dataclass
class AIConfig:
    """Configuration for AI analysis."""
    
    enabled: bool = True
    semantic_analysis: bool = True
    security_analysis: bool = True
    model: str = "gpt-4"
    custom_prompts: dict[str, str] = field(default_factory=dict)


@dataclass
class IgnoreRule:
    """A rule for ignoring findings."""
    
    pattern: str          # Glob pattern for file paths
    categories: list[str] | None = None  # Categories to ignore
    reason: str = ""      # Documentation


@dataclass
class CustomRule:
    """A custom analysis rule."""
    
    id: str
    name: str
    description: str
    severity: str = "medium"
    pattern: str | None = None  # Regex pattern
    prompt: str | None = None   # AI prompt for detection
    enabled: bool = True


@dataclass
class CacheConfig:
    """Configuration for incremental verification cache."""

    enabled: bool = True
    backend: str = "memory"  # "memory" or "redis"
    redis_url: str = "redis://localhost:6379/1"
    ttl_seconds: int = 86400
    max_entries: int = 10000


@dataclass
class PolicyConfig:
    """Configuration for verification policy-as-code."""

    enabled: bool = False
    policy_file: str | None = None
    use_built_in: list[str] = field(default_factory=list)
    default_action: str = "warn"


@dataclass
class StreamingConfig:
    """Configuration for real-time streaming verification."""

    enabled: bool = True
    debounce_ms: int = 500
    stages: list[str] = field(default_factory=lambda: ["pattern", "ai", "formal"])


@dataclass
class AutoFixConfig:
    """Configuration for agentic auto-fix pipeline."""

    enabled: bool = False
    max_attempts: int = 3
    require_verification: bool = True
    require_test: bool = True
    auto_create_pr: bool = False
    min_confidence: str = "medium"


@dataclass
class HallucinationConfig:
    """Configuration for AI hallucination detection."""

    enabled: bool = True
    check_registry: bool = False
    custom_allowlist: list[str] = field(default_factory=list)


@dataclass
class TelemetryConfig:
    """Configuration for verification telemetry and ROI."""

    enabled: bool = True
    track_lifecycle: bool = True
    cost_multipliers: dict[str, float] = field(default_factory=lambda: {
        "critical": 15000.0,
        "high": 5000.0,
        "medium": 1500.0,
        "low": 500.0,
    })


@dataclass
class ImpactConfig:
    """Configuration for cross-repository impact analysis."""

    enabled: bool = False
    org_repos: list[str] = field(default_factory=list)
    notify_downstream: bool = True


@dataclass
class CopilotConfig:
    """Configuration for GitHub Copilot Extension integration."""

    enabled: bool = False
    webhook_secret: str = ""
    commands: list[str] = field(default_factory=lambda: [
        "verify", "explain", "spec", "trust_score", "fix", "history",
    ])


@dataclass
class CodeVerifyConfig:
    """Complete CodeVerify configuration."""

    version: str = "1"

    # Languages to analyze
    languages: list[str] = field(default_factory=lambda: ["python", "typescript", "go", "java"])
    
    # File patterns
    include_patterns: list[str] = field(default_factory=lambda: ["**/*.py", "**/*.ts", "**/*.tsx"])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "node_modules/**",
        "venv/**",
        "__pycache__/**",
        "dist/**",
        "build/**",
        "*.min.js",
        "*.generated.*",
    ])
    
    # Analysis settings
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    
    # Thresholds
    thresholds: SeverityThresholds = field(default_factory=SeverityThresholds)
    
    # Ignore rules
    ignore_rules: list[IgnoreRule] = field(default_factory=list)
    
    # Custom rules
    custom_rules: list[CustomRule] = field(default_factory=list)
    
    # PR settings
    auto_approve: bool = False
    comment_on_pass: bool = True
    collapse_findings: bool = True
    max_inline_comments: int = 10

    # Next-gen feature configs
    cache: CacheConfig = field(default_factory=CacheConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    autofix: AutoFixConfig = field(default_factory=AutoFixConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
    telemetry_config: TelemetryConfig = field(default_factory=TelemetryConfig)
    impact: ImpactConfig = field(default_factory=ImpactConfig)
    copilot: CopilotConfig = field(default_factory=CopilotConfig)


def parse_config(content: str | dict[str, Any]) -> CodeVerifyConfig:
    """Parse configuration from YAML string or dict."""
    if isinstance(content, str):
        data = yaml.safe_load(content) or {}
    else:
        data = content
    
    config = CodeVerifyConfig()
    
    # Version
    config.version = str(data.get("version", "1"))
    
    # Languages
    if "languages" in data:
        config.languages = list(data["languages"])
    
    # File patterns
    if "include" in data:
        config.include_patterns = list(data["include"])
    if "exclude" in data:
        config.exclude_patterns = list(data["exclude"])
    
    # Verification config
    if "verification" in data:
        v = data["verification"]
        config.verification = VerificationConfig(
            enabled=v.get("enabled", True),
            timeout_seconds=v.get("timeout", 30),
            checks=v.get("checks", config.verification.checks),
        )
    
    # AI config
    if "ai" in data:
        a = data["ai"]
        config.ai = AIConfig(
            enabled=a.get("enabled", True),
            semantic_analysis=a.get("semantic", True),
            security_analysis=a.get("security", True),
            model=a.get("model", "gpt-4"),
            custom_prompts=a.get("prompts", {}),
        )
    
    # Thresholds
    if "thresholds" in data:
        t = data["thresholds"]
        config.thresholds = SeverityThresholds(
            critical=t.get("critical", 0),
            high=t.get("high", 0),
            medium=t.get("medium", 5),
            low=t.get("low", 10),
        )
    
    # Ignore rules
    if "ignore" in data:
        for rule in data["ignore"]:
            config.ignore_rules.append(IgnoreRule(
                pattern=rule.get("pattern", "**/*"),
                categories=rule.get("categories"),
                reason=rule.get("reason", ""),
            ))
    
    # Custom rules
    if "rules" in data:
        for rule in data["rules"]:
            config.custom_rules.append(CustomRule(
                id=rule["id"],
                name=rule.get("name", rule["id"]),
                description=rule.get("description", ""),
                severity=rule.get("severity", "medium"),
                pattern=rule.get("pattern"),
                prompt=rule.get("prompt"),
                enabled=rule.get("enabled", True),
            ))
    
    # PR settings
    config.auto_approve = data.get("auto_approve", False)
    config.comment_on_pass = data.get("comment_on_pass", True)
    config.collapse_findings = data.get("collapse_findings", True)
    config.max_inline_comments = data.get("max_inline_comments", 10)

    # Cache config
    if "cache" in data:
        c = data["cache"]
        config.cache = CacheConfig(
            enabled=c.get("enabled", True),
            backend=c.get("backend", "memory"),
            redis_url=c.get("redis_url", "redis://localhost:6379/1"),
            ttl_seconds=c.get("ttl_seconds", 86400),
            max_entries=c.get("max_entries", 10000),
        )

    # Policy config
    if "policy" in data:
        p = data["policy"]
        config.policy = PolicyConfig(
            enabled=p.get("enabled", False),
            policy_file=p.get("policy_file"),
            use_built_in=p.get("use_built_in", []),
            default_action=p.get("default_action", "warn"),
        )

    # Streaming config
    if "streaming" in data:
        s = data["streaming"]
        config.streaming = StreamingConfig(
            enabled=s.get("enabled", True),
            debounce_ms=s.get("debounce_ms", 500),
            stages=s.get("stages", ["pattern", "ai", "formal"]),
        )

    # Auto-fix config
    if "autofix" in data:
        af = data["autofix"]
        config.autofix = AutoFixConfig(
            enabled=af.get("enabled", False),
            max_attempts=af.get("max_attempts", 3),
            require_verification=af.get("require_verification", True),
            auto_create_pr=af.get("auto_create_pr", False),
            min_confidence=af.get("min_confidence", "medium"),
        )

    # Hallucination config
    if "hallucination" in data:
        h = data["hallucination"]
        config.hallucination = HallucinationConfig(
            enabled=h.get("enabled", True),
            check_registry=h.get("check_registry", False),
            custom_allowlist=h.get("allowlist", []),
        )

    # Telemetry config
    if "telemetry" in data:
        t = data["telemetry"]
        config.telemetry_config = TelemetryConfig(
            enabled=t.get("enabled", True),
            track_lifecycle=t.get("track_lifecycle", True),
        )

    # Impact analysis config
    if "impact" in data:
        i = data["impact"]
        config.impact = ImpactConfig(
            enabled=i.get("enabled", False),
            org_repos=i.get("org_repos", []),
            notify_downstream=i.get("notify_downstream", True),
        )

    # Copilot config
    if "copilot" in data:
        cp = data["copilot"]
        config.copilot = CopilotConfig(
            enabled=cp.get("enabled", False),
            webhook_secret=cp.get("webhook_secret", ""),
            commands=cp.get("commands", ["verify", "explain", "spec", "trust_score", "fix", "history"]),
        )

    return config


def load_config(repo_path: Path | str) -> CodeVerifyConfig:
    """Load configuration from repository."""
    repo_path = Path(repo_path)
    
    # Try different config file names
    config_names = [".codeverify.yml", ".codeverify.yaml", "codeverify.yml"]
    
    for name in config_names:
        config_file = repo_path / name
        if config_file.exists():
            logger.info(f"Loading config from {config_file}")
            return parse_config(config_file.read_text())
    
    # Return defaults if no config file
    logger.info("No config file found, using defaults")
    return CodeVerifyConfig()


def should_analyze_file(config: CodeVerifyConfig, file_path: str) -> bool:
    """Check if a file should be analyzed based on config patterns."""
    from fnmatch import fnmatch
    
    # Check exclusions first
    for pattern in config.exclude_patterns:
        if fnmatch(file_path, pattern):
            return False
    
    # Check inclusions
    for pattern in config.include_patterns:
        if fnmatch(file_path, pattern):
            return True
    
    return False


def should_ignore_finding(
    config: CodeVerifyConfig,
    file_path: str,
    category: str
) -> tuple[bool, str]:
    """Check if a finding should be ignored based on rules."""
    from fnmatch import fnmatch
    
    for rule in config.ignore_rules:
        if fnmatch(file_path, rule.pattern):
            # If no categories specified, ignore all for this pattern
            if rule.categories is None:
                return True, rule.reason
            # Check if category matches
            if category in rule.categories:
                return True, rule.reason
    
    return False, ""


def passes_thresholds(
    config: CodeVerifyConfig,
    findings: dict[str, int]
) -> tuple[bool, str]:
    """Check if findings pass the configured thresholds."""
    critical = findings.get("critical", 0)
    high = findings.get("high", 0)
    medium = findings.get("medium", 0)
    low = findings.get("low", 0)
    
    if critical > config.thresholds.critical:
        return False, f"Critical findings ({critical}) exceed threshold ({config.thresholds.critical})"
    
    if high > config.thresholds.high:
        return False, f"High findings ({high}) exceed threshold ({config.thresholds.high})"
    
    if medium > config.thresholds.medium:
        return False, f"Medium findings ({medium}) exceed threshold ({config.thresholds.medium})"
    
    if low > config.thresholds.low:
        return False, f"Low findings ({low}) exceed threshold ({config.thresholds.low})"
    
    return True, "All findings within thresholds"


# Example configuration for documentation
EXAMPLE_CONFIG = """
# .codeverify.yml - CodeVerify Configuration
version: "1"

# Languages to analyze
languages:
  - python
  - typescript

# File patterns
include:
  - "**/*.py"
  - "**/*.ts"
  - "**/*.tsx"

exclude:
  - "node_modules/**"
  - "venv/**"
  - "tests/**"
  - "*.min.js"

# Formal verification settings
verification:
  enabled: true
  timeout: 30  # seconds per check
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero

# AI analysis settings
ai:
  enabled: true
  semantic: true
  security: true
  model: gpt-4
  prompts:
    security: "Focus on authentication and authorization issues"

# Severity thresholds (max allowed to pass)
thresholds:
  critical: 0
  high: 0
  medium: 5
  low: 10

# Ignore rules
ignore:
  - pattern: "migrations/**"
    reason: "Auto-generated migrations"
  - pattern: "tests/**"
    categories:
      - security
    reason: "Test code can have different standards"

# Custom rules
rules:
  - id: no-print
    name: "No print statements"
    description: "Use logging instead of print"
    severity: low
    pattern: "\\bprint\\s*\\("
    enabled: true
  
  - id: auth-required
    name: "Endpoints require authentication"
    description: "All API endpoints should have authentication"
    severity: high
    prompt: "Check if this API endpoint has proper authentication"

# PR behavior
auto_approve: false
comment_on_pass: true
collapse_findings: true
max_inline_comments: 10
"""
