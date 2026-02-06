#!/usr/bin/env python3
"""
CodeVerify Tier Configuration

Defines verification capabilities and limits for each tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VerificationTier(str, Enum):
    """Verification tier levels."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Limits and quotas for a verification tier."""

    max_files_per_run: int
    max_file_size_kb: int
    max_lines_per_file: int
    monthly_runs: int | None  # None = unlimited
    concurrent_runs: int
    retention_days: int


@dataclass
class TierCapabilities:
    """Capabilities enabled for a verification tier."""

    # Analysis capabilities
    pattern_analysis: bool = True
    ai_analysis: bool = False
    z3_verification: bool = False
    supply_chain_audit: bool = False
    runtime_probes: bool = False

    # Output capabilities
    sarif_output: bool = True
    pr_comments: bool = True
    detailed_reports: bool = False
    proof_certificates: bool = False

    # Integration capabilities
    github_security_tab: bool = True
    slack_notifications: bool = False
    custom_webhooks: bool = False
    api_access: bool = False

    # Advanced features
    auto_fix_suggestions: bool = False
    formal_proofs: bool = False
    custom_rules: bool = False
    priority_support: bool = False


@dataclass
class TierPricing:
    """Pricing information for a tier."""

    monthly_price_usd: float
    annual_price_usd: float
    per_seat_pricing: bool = False
    enterprise_contact: bool = False


@dataclass
class TierConfiguration:
    """Complete configuration for a verification tier."""

    tier: VerificationTier
    display_name: str
    description: str
    limits: TierLimits
    capabilities: TierCapabilities
    pricing: TierPricing
    badge_color: str = "blue"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tier": self.tier.value,
            "display_name": self.display_name,
            "description": self.description,
            "limits": {
                "max_files_per_run": self.limits.max_files_per_run,
                "max_file_size_kb": self.limits.max_file_size_kb,
                "max_lines_per_file": self.limits.max_lines_per_file,
                "monthly_runs": self.limits.monthly_runs,
                "concurrent_runs": self.limits.concurrent_runs,
                "retention_days": self.limits.retention_days
            },
            "capabilities": {
                "pattern_analysis": self.capabilities.pattern_analysis,
                "ai_analysis": self.capabilities.ai_analysis,
                "z3_verification": self.capabilities.z3_verification,
                "supply_chain_audit": self.capabilities.supply_chain_audit,
                "runtime_probes": self.capabilities.runtime_probes,
                "sarif_output": self.capabilities.sarif_output,
                "pr_comments": self.capabilities.pr_comments,
                "detailed_reports": self.capabilities.detailed_reports,
                "proof_certificates": self.capabilities.proof_certificates,
                "github_security_tab": self.capabilities.github_security_tab,
                "slack_notifications": self.capabilities.slack_notifications,
                "custom_webhooks": self.capabilities.custom_webhooks,
                "api_access": self.capabilities.api_access,
                "auto_fix_suggestions": self.capabilities.auto_fix_suggestions,
                "formal_proofs": self.capabilities.formal_proofs,
                "custom_rules": self.capabilities.custom_rules,
                "priority_support": self.capabilities.priority_support
            },
            "pricing": {
                "monthly_price_usd": self.pricing.monthly_price_usd,
                "annual_price_usd": self.pricing.annual_price_usd,
                "per_seat_pricing": self.pricing.per_seat_pricing,
                "enterprise_contact": self.pricing.enterprise_contact
            }
        }


# Tier configurations
FREE_TIER = TierConfiguration(
    tier=VerificationTier.FREE,
    display_name="Free",
    description="Pattern-based static analysis for open source projects",
    limits=TierLimits(
        max_files_per_run=100,
        max_file_size_kb=500,
        max_lines_per_file=5000,
        monthly_runs=100,
        concurrent_runs=1,
        retention_days=7
    ),
    capabilities=TierCapabilities(
        pattern_analysis=True,
        ai_analysis=False,
        z3_verification=False,
        supply_chain_audit=False,
        sarif_output=True,
        pr_comments=True,
        detailed_reports=False,
        github_security_tab=True
    ),
    pricing=TierPricing(
        monthly_price_usd=0,
        annual_price_usd=0
    ),
    badge_color="green"
)

PRO_TIER = TierConfiguration(
    tier=VerificationTier.PRO,
    display_name="Pro",
    description="AI-powered semantic analysis for teams",
    limits=TierLimits(
        max_files_per_run=1000,
        max_file_size_kb=2000,
        max_lines_per_file=20000,
        monthly_runs=1000,
        concurrent_runs=5,
        retention_days=30
    ),
    capabilities=TierCapabilities(
        pattern_analysis=True,
        ai_analysis=True,
        z3_verification=False,
        supply_chain_audit=True,
        sarif_output=True,
        pr_comments=True,
        detailed_reports=True,
        github_security_tab=True,
        slack_notifications=True,
        auto_fix_suggestions=True
    ),
    pricing=TierPricing(
        monthly_price_usd=29,
        annual_price_usd=290,
        per_seat_pricing=False
    ),
    badge_color="blue"
)

ENTERPRISE_TIER = TierConfiguration(
    tier=VerificationTier.ENTERPRISE,
    display_name="Enterprise",
    description="Full Z3 formal verification with proofs for security-critical applications",
    limits=TierLimits(
        max_files_per_run=10000,
        max_file_size_kb=10000,
        max_lines_per_file=100000,
        monthly_runs=None,  # Unlimited
        concurrent_runs=20,
        retention_days=365
    ),
    capabilities=TierCapabilities(
        pattern_analysis=True,
        ai_analysis=True,
        z3_verification=True,
        supply_chain_audit=True,
        runtime_probes=True,
        sarif_output=True,
        pr_comments=True,
        detailed_reports=True,
        proof_certificates=True,
        github_security_tab=True,
        slack_notifications=True,
        custom_webhooks=True,
        api_access=True,
        auto_fix_suggestions=True,
        formal_proofs=True,
        custom_rules=True,
        priority_support=True
    ),
    pricing=TierPricing(
        monthly_price_usd=199,
        annual_price_usd=1990,
        per_seat_pricing=True,
        enterprise_contact=True
    ),
    badge_color="purple"
)

# Tier registry
TIER_CONFIGS: dict[VerificationTier, TierConfiguration] = {
    VerificationTier.FREE: FREE_TIER,
    VerificationTier.PRO: PRO_TIER,
    VerificationTier.ENTERPRISE: ENTERPRISE_TIER
}


def get_tier_config(tier: str | VerificationTier) -> TierConfiguration:
    """Get configuration for a tier."""
    if isinstance(tier, str):
        tier = VerificationTier(tier)
    return TIER_CONFIGS[tier]


def check_tier_capability(tier: str | VerificationTier, capability: str) -> bool:
    """Check if a capability is enabled for a tier."""
    config = get_tier_config(tier)
    return getattr(config.capabilities, capability, False)


def check_tier_limit(tier: str | VerificationTier, limit: str, value: int) -> bool:
    """Check if a value is within tier limits."""
    config = get_tier_config(tier)
    limit_value = getattr(config.limits, limit, None)

    if limit_value is None:  # Unlimited
        return True

    return value <= limit_value


def get_tier_comparison() -> list[dict[str, Any]]:
    """Get comparison data for all tiers."""
    return [config.to_dict() for config in TIER_CONFIGS.values()]


def validate_api_key(api_key: str, tier: VerificationTier) -> bool:
    """
    Validate API key for a tier.

    In production, this would verify against the CodeVerify API.
    """
    if tier == VerificationTier.FREE:
        return True  # No API key required for free tier

    if not api_key:
        return False

    # Placeholder validation - in production, verify with API
    return api_key.startswith("cv_") and len(api_key) >= 32
