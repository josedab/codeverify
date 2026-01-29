"""Severity utilities for consistent handling across the codebase.

This module provides centralized severity constants, parsing, and comparison
functions used by findings, rules, and analysis reports.
"""

from __future__ import annotations

from enum import Enum


class FindingSeverity(str, Enum):
    """Severity level of a finding."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# Severity ordering for comparison (higher number = more severe)
SEVERITY_ORDER: dict[str, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
    # Also handle alternative severity names
    "warning": 2,  # maps to medium
    "error": 3,    # maps to high
}

# Emoji mapping for display purposes
SEVERITY_EMOJI: dict[str, str] = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
    "info": "⚪",
}

# Labels for UI display
SEVERITY_LABELS: dict[str, str] = {
    "critical": "Critical",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "info": "Info",
}


def parse_severity(
    value: str | FindingSeverity,
    default: FindingSeverity = FindingSeverity.MEDIUM,
) -> FindingSeverity:
    """Parse a severity value to FindingSeverity enum.
    
    Handles string values with fallback to default for invalid values.
    
    Args:
        value: String or FindingSeverity value
        default: Default severity if parsing fails
        
    Returns:
        FindingSeverity enum value
        
    Examples:
        >>> parse_severity("high")
        FindingSeverity.HIGH
        >>> parse_severity("error")  # alias
        FindingSeverity.HIGH
        >>> parse_severity("invalid")
        FindingSeverity.MEDIUM
    """
    if isinstance(value, FindingSeverity):
        return value
    
    try:
        return FindingSeverity(value.lower())
    except (ValueError, AttributeError):
        # Map common aliases
        aliases = {
            "warning": FindingSeverity.MEDIUM,
            "error": FindingSeverity.HIGH,
            "severe": FindingSeverity.CRITICAL,
        }
        return aliases.get(value.lower(), default) if isinstance(value, str) else default


def compare_severity(sev1: str | FindingSeverity, sev2: str | FindingSeverity) -> int:
    """Compare two severity values.
    
    Args:
        sev1: First severity
        sev2: Second severity
        
    Returns:
        -1 if sev1 < sev2, 0 if equal, 1 if sev1 > sev2
        
    Examples:
        >>> compare_severity("high", "low")
        1
        >>> compare_severity("low", "critical")
        -1
        >>> compare_severity("medium", "warning")  # warning = medium
        0
    """
    val1 = SEVERITY_ORDER.get(str(sev1).lower(), 0)
    val2 = SEVERITY_ORDER.get(str(sev2).lower(), 0)
    
    if val1 < val2:
        return -1
    elif val1 > val2:
        return 1
    return 0


def is_blocking_severity(severity: str | FindingSeverity) -> bool:
    """Check if a severity level should block merging.
    
    Args:
        severity: Severity to check
        
    Returns:
        True if severity is high or critical
        
    Examples:
        >>> is_blocking_severity("critical")
        True
        >>> is_blocking_severity("medium")
        False
    """
    sev_str = str(severity).lower()
    return sev_str in ("critical", "high", "error", "severe")


def is_above_threshold(
    severity: str | FindingSeverity,
    threshold: str | FindingSeverity,
) -> bool:
    """Check if severity is at or above a threshold.
    
    Args:
        severity: Severity to check
        threshold: Minimum severity threshold
        
    Returns:
        True if severity >= threshold
        
    Examples:
        >>> is_above_threshold("high", "medium")
        True
        >>> is_above_threshold("low", "medium")
        False
    """
    return compare_severity(severity, threshold) >= 0


def get_severity_emoji(severity: str | FindingSeverity) -> str:
    """Get emoji for severity level.
    
    Args:
        severity: Severity to get emoji for
        
    Returns:
        Emoji string
    """
    return SEVERITY_EMOJI.get(str(severity).lower(), "⚪")


def get_severity_label(severity: str | FindingSeverity) -> str:
    """Get display label for severity level.
    
    Args:
        severity: Severity to get label for
        
    Returns:
        Human-readable label
    """
    return SEVERITY_LABELS.get(str(severity).lower(), "Unknown")


def sort_by_severity(
    items: list,
    key: str = "severity",
    descending: bool = True,
) -> list:
    """Sort items by severity.
    
    Args:
        items: List of dicts or objects with severity attribute/key
        key: Key or attribute name containing severity
        descending: If True, most severe first
        
    Returns:
        Sorted list
    """
    def get_order(item):
        if isinstance(item, dict):
            sev = item.get(key, "info")
        else:
            sev = getattr(item, key, "info")
        return SEVERITY_ORDER.get(str(sev).lower(), 0)
    
    return sorted(items, key=get_order, reverse=descending)
