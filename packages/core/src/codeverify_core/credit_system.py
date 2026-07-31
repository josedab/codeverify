"""Verification Credit System.

Org-level gamification: earn credits for verification coverage,
fix rate, contributions. Credits unlock features and reduce pricing.

Features:
- Credit earning rules (coverage, fix rate, contributions)
- Credit redemption (feature unlocks, pricing discounts)
- Org leaderboard
- Anti-gaming rules
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

import structlog

logger = structlog.get_logger()


class CreditSource(str, Enum):
    VERIFICATION_COVERAGE = "verification_coverage"
    FIX_RATE = "fix_rate"
    FEDERATED_CONTRIBUTION = "federated_contribution"
    ZERO_CRITICAL_STREAK = "zero_critical_streak"
    TRAINING_COMPLETION = "training_completion"


class RedemptionType(str, Enum):
    FEATURE_UNLOCK = "feature_unlock"
    PRICING_DISCOUNT = "pricing_discount"
    BADGE = "badge"
    PRIORITY_SUPPORT = "priority_support"


@dataclass
class CreditTransaction:
    """A credit earning or spending transaction."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    org_id: str = ""
    amount: int = 0
    source: CreditSource | None = None
    redemption: RedemptionType | None = None
    description: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class OrgCredits:
    """Credit balance for an organization."""

    org_id: str = ""
    org_name: str = ""
    balance: int = 0
    total_earned: int = 0
    total_spent: int = 0
    transactions: list[CreditTransaction] = field(default_factory=list)
    badges: list[str] = field(default_factory=list)


@dataclass
class CreditRule:
    """Rule for earning credits."""

    source: CreditSource = CreditSource.VERIFICATION_COVERAGE
    threshold: float = 0.8
    credits_awarded: int = 100
    description: str = ""
    max_per_period: int = 1


@dataclass
class LeaderboardEntry:
    """Leaderboard entry for an org."""

    org_id: str = ""
    org_name: str = ""
    balance: int = 0
    total_earned: int = 0
    rank: int = 0
    badges: int = 0


CREDIT_RULES = [
    CreditRule(CreditSource.VERIFICATION_COVERAGE, 0.8, 100, "80%+ verification coverage", 1),
    CreditRule(CreditSource.VERIFICATION_COVERAGE, 0.95, 200, "95%+ verification coverage", 1),
    CreditRule(CreditSource.FIX_RATE, 0.9, 150, "90%+ finding fix rate", 1),
    CreditRule(CreditSource.FEDERATED_CONTRIBUTION, 0, 50, "Contributed to federated learning", 5),
    CreditRule(
        CreditSource.ZERO_CRITICAL_STREAK, 30, 300, "30-day zero critical findings streak", 1
    ),
    CreditRule(CreditSource.TRAINING_COMPLETION, 0, 75, "Completed security training module", 10),
]

REDEMPTION_COSTS = {
    RedemptionType.FEATURE_UNLOCK: 500,
    RedemptionType.PRICING_DISCOUNT: 1000,
    RedemptionType.BADGE: 100,
    RedemptionType.PRIORITY_SUPPORT: 750,
}


class VerificationCreditService:
    """Main service for the verification credit system."""

    def __init__(self) -> None:
        self._orgs: dict[str, OrgCredits] = {}

    def register_org(self, org_id: str, org_name: str = "") -> OrgCredits:
        if org_id not in self._orgs:
            self._orgs[org_id] = OrgCredits(org_id=org_id, org_name=org_name or org_id)
        return self._orgs[org_id]

    def award_credits(
        self, org_id: str, source: CreditSource, amount: int, description: str = ""
    ) -> CreditTransaction:
        org = self._orgs.get(org_id)
        if not org:
            org = self.register_org(org_id)
        tx = CreditTransaction(org_id=org_id, amount=amount, source=source, description=description)
        org.balance += amount
        org.total_earned += amount
        org.transactions.append(tx)
        return tx

    def evaluate_rules(self, org_id: str, metrics: dict[str, float]) -> list[CreditTransaction]:
        awarded: list[CreditTransaction] = []
        for rule in CREDIT_RULES:
            metric_val = metrics.get(rule.source.value, 0.0)
            if metric_val >= rule.threshold:
                tx = self.award_credits(org_id, rule.source, rule.credits_awarded, rule.description)
                awarded.append(tx)
        return awarded

    def redeem(self, org_id: str, redemption: RedemptionType) -> CreditTransaction | None:
        org = self._orgs.get(org_id)
        if not org:
            return None
        cost = REDEMPTION_COSTS.get(redemption, 0)
        if org.balance < cost:
            return None
        tx = CreditTransaction(
            org_id=org_id,
            amount=-cost,
            redemption=redemption,
            description=f"Redeemed: {redemption.value}",
        )
        org.balance -= cost
        org.total_spent += cost
        org.transactions.append(tx)
        if redemption == RedemptionType.BADGE and "verified_org" not in org.badges:
            org.badges.append("verified_org")
        return tx

    def get_balance(self, org_id: str) -> int:
        org = self._orgs.get(org_id)
        return org.balance if org else 0

    def get_leaderboard(self) -> list[LeaderboardEntry]:
        entries = sorted(self._orgs.values(), key=lambda o: o.total_earned, reverse=True)
        return [
            LeaderboardEntry(
                org_id=o.org_id,
                org_name=o.org_name,
                balance=o.balance,
                total_earned=o.total_earned,
                rank=i + 1,
                badges=len(o.badges),
            )
            for i, o in enumerate(entries)
        ]

    def get_org(self, org_id: str) -> OrgCredits | None:
        return self._orgs.get(org_id)


_credit_instance: VerificationCreditService | None = None


def get_credit_service() -> VerificationCreditService:
    global _credit_instance
    if _credit_instance is None:
        _credit_instance = VerificationCreditService()
    return _credit_instance


def reset_credit_service() -> None:
    global _credit_instance
    _credit_instance = None
