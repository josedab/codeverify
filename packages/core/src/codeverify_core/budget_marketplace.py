"""Verification Budget Marketplace.

Secondary market for verification credits with dynamic pricing. Teams buy/sell
verification capacity, optimizing global utilization and cost.

Features:
- Credit system with complexity-normalized units
- Marketplace with listings and bidding
- Wallet management with balance tracking
- Dynamic pricing based on supply/demand
- Capacity leasing for idle verification workers

.. deprecated::
    This module is superseded by ``codeverify_core.proof_artifact_marketplace``.
    It remains importable for backward compatibility but will be
    removed in a future release.
"""

from __future__ import annotations

import warnings as _warnings
_warnings.warn(
    "codeverify_core.budget_marketplace is deprecated. Use codeverify_core.proof_artifact_marketplace instead.",
    DeprecationWarning,
    stacklevel=2,
)


import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class OrderSide(str, Enum):
    """Side of a marketplace order."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Status of a marketplace order."""

    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TransactionType(str, Enum):
    """Type of wallet transaction."""

    PURCHASE = "purchase"
    SALE = "sale"
    USAGE = "usage"
    REFUND = "refund"
    GRANT = "grant"
    LEASE_INCOME = "lease_income"
    LEASE_EXPENSE = "lease_expense"


@dataclass
class VerificationCredit:
    """A normalized verification credit unit.

    1 credit = 1 verification of average complexity (10 constraints, 50 LOC).
    """

    amount: float = 0.0
    complexity_factor: float = 1.0

    @property
    def effective_credits(self) -> float:
        return self.amount * self.complexity_factor

    @staticmethod
    def from_verification(
        constraints: int, loc: int, base_rate: float = 1.0,
    ) -> VerificationCredit:
        """Calculate credits needed for a verification task."""
        complexity = (constraints / 10.0) * (loc / 50.0)
        return VerificationCredit(
            amount=base_rate,
            complexity_factor=max(0.1, complexity),
        )


@dataclass
class WalletTransaction:
    """A transaction in a team's credit wallet."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    transaction_type: TransactionType = TransactionType.PURCHASE
    amount: float = 0.0
    price_per_credit: float = 0.0
    description: str = ""
    counterparty: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def total_cost(self) -> float:
        return self.amount * self.price_per_credit

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.transaction_type.value,
            "amount": self.amount,
            "price_per_credit": self.price_per_credit,
            "total_cost": self.total_cost,
            "description": self.description,
            "counterparty": self.counterparty,
            "timestamp": self.timestamp,
        }


@dataclass
class Wallet:
    """A team's verification credit wallet."""

    team_id: str = ""
    balance: float = 0.0
    reserved: float = 0.0
    transactions: list[WalletTransaction] = field(default_factory=list)

    @property
    def available(self) -> float:
        return self.balance - self.reserved

    def deposit(self, amount: float, tx_type: TransactionType, **kwargs: Any) -> WalletTransaction:
        tx = WalletTransaction(
            transaction_type=tx_type,
            amount=amount,
            description=kwargs.get("description", ""),
            counterparty=kwargs.get("counterparty", ""),
            price_per_credit=kwargs.get("price_per_credit", 0.0),
        )
        self.balance += amount
        self.transactions.append(tx)
        return tx

    def withdraw(self, amount: float, tx_type: TransactionType, **kwargs: Any) -> WalletTransaction | None:
        if amount > self.available:
            return None
        tx = WalletTransaction(
            transaction_type=tx_type,
            amount=-amount,
            description=kwargs.get("description", ""),
            counterparty=kwargs.get("counterparty", ""),
            price_per_credit=kwargs.get("price_per_credit", 0.0),
        )
        self.balance -= amount
        self.transactions.append(tx)
        return tx

    def reserve(self, amount: float) -> bool:
        if amount > self.available:
            return False
        self.reserved += amount
        return True

    def release_reserve(self, amount: float) -> None:
        self.reserved = max(0, self.reserved - amount)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "balance": self.balance,
            "reserved": self.reserved,
            "available": self.available,
            "transaction_count": len(self.transactions),
        }


@dataclass
class MarketOrder:
    """A buy or sell order on the marketplace."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    team_id: str = ""
    side: OrderSide = OrderSide.BUY
    credits: float = 0.0
    price_per_credit: float = 0.0
    filled: float = 0.0
    status: OrderStatus = OrderStatus.OPEN
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    @property
    def remaining(self) -> float:
        return self.credits - self.filled

    @property
    def is_expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at

    @property
    def total_value(self) -> float:
        return self.credits * self.price_per_credit

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "team_id": self.team_id,
            "side": self.side.value,
            "credits": self.credits,
            "price_per_credit": self.price_per_credit,
            "filled": self.filled,
            "remaining": self.remaining,
            "status": self.status.value,
            "total_value": self.total_value,
        }


@dataclass
class Trade:
    """A completed trade between buyer and seller."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    buyer_id: str = ""
    seller_id: str = ""
    credits: float = 0.0
    price_per_credit: float = 0.0
    buy_order_id: str = ""
    sell_order_id: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def total_value(self) -> float:
        return self.credits * self.price_per_credit

    @property
    def platform_fee(self) -> float:
        return self.total_value * 0.05  # 5% platform fee

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "credits": self.credits,
            "price_per_credit": self.price_per_credit,
            "total_value": self.total_value,
            "platform_fee": self.platform_fee,
            "timestamp": self.timestamp,
        }


@dataclass
class MarketStats:
    """Current marketplace statistics."""

    total_volume: float = 0.0
    trades_count: int = 0
    avg_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    spread: float = 0.0
    open_buy_orders: int = 0
    open_sell_orders: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_volume": round(self.total_volume, 2),
            "trades_count": self.trades_count,
            "avg_price": round(self.avg_price, 4),
            "bid_price": round(self.bid_price, 4),
            "ask_price": round(self.ask_price, 4),
            "spread": round(self.spread, 4),
            "open_buy_orders": self.open_buy_orders,
            "open_sell_orders": self.open_sell_orders,
        }


@dataclass
class CapacityLease:
    """A lease of idle verification capacity."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    provider_id: str = ""
    consumer_id: str = ""
    credits_per_hour: float = 0.0
    price_per_credit: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    active: bool = True

    @property
    def duration_hours(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return (end - self.start_time) / 3600

    @property
    def total_credits(self) -> float:
        return self.credits_per_hour * self.duration_hours

    @property
    def total_cost(self) -> float:
        return self.total_credits * self.price_per_credit

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "consumer_id": self.consumer_id,
            "credits_per_hour": self.credits_per_hour,
            "price_per_credit": self.price_per_credit,
            "active": self.active,
            "duration_hours": round(self.duration_hours, 2),
            "total_cost": round(self.total_cost, 2),
        }


class VerificationMarketplace:
    """Credit marketplace with order matching and wallet management."""

    def __init__(self, platform_fee_rate: float = 0.05) -> None:
        self._wallets: dict[str, Wallet] = {}
        self._orders: list[MarketOrder] = []
        self._trades: list[Trade] = []
        self._leases: list[CapacityLease] = []
        self._platform_fee_rate = platform_fee_rate

    def get_or_create_wallet(self, team_id: str) -> Wallet:
        if team_id not in self._wallets:
            self._wallets[team_id] = Wallet(team_id=team_id)
        return self._wallets[team_id]

    def grant_credits(self, team_id: str, amount: float) -> WalletTransaction:
        """Grant free credits to a team (e.g., welcome bonus)."""
        wallet = self.get_or_create_wallet(team_id)
        return wallet.deposit(amount, TransactionType.GRANT, description="Credit grant")

    def use_credits(self, team_id: str, credits: VerificationCredit) -> bool:
        """Consume credits for a verification task."""
        wallet = self.get_or_create_wallet(team_id)
        amount = credits.effective_credits
        tx = wallet.withdraw(amount, TransactionType.USAGE, description="Verification usage")
        return tx is not None

    def place_order(
        self, team_id: str, side: OrderSide, credits: float,
        price_per_credit: float, ttl_seconds: float = 3600,
    ) -> MarketOrder:
        """Place a buy or sell order."""
        order = MarketOrder(
            team_id=team_id,
            side=side,
            credits=credits,
            price_per_credit=price_per_credit,
            expires_at=time.time() + ttl_seconds if ttl_seconds > 0 else 0,
        )

        # Reserve credits for sell orders
        if side == OrderSide.SELL:
            wallet = self.get_or_create_wallet(team_id)
            if not wallet.reserve(credits):
                order.status = OrderStatus.CANCELLED
                return order

        self._orders.append(order)

        # Try to match
        self._match_orders()

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        for order in self._orders:
            if order.id == order_id and order.status == OrderStatus.OPEN:
                order.status = OrderStatus.CANCELLED
                if order.side == OrderSide.SELL:
                    wallet = self.get_or_create_wallet(order.team_id)
                    wallet.release_reserve(order.remaining)
                return True
        return False

    def _match_orders(self) -> None:
        """Match buy and sell orders (price-time priority)."""
        buy_orders = sorted(
            [o for o in self._orders if o.side == OrderSide.BUY and o.status == OrderStatus.OPEN],
            key=lambda o: (-o.price_per_credit, o.created_at),
        )
        sell_orders = sorted(
            [o for o in self._orders if o.side == OrderSide.SELL and o.status == OrderStatus.OPEN],
            key=lambda o: (o.price_per_credit, o.created_at),
        )

        for buy in buy_orders:
            for sell in sell_orders:
                if buy.team_id == sell.team_id:
                    continue
                if buy.price_per_credit < sell.price_per_credit:
                    continue
                if buy.remaining <= 0 or sell.remaining <= 0:
                    continue

                # Execute trade
                trade_credits = min(buy.remaining, sell.remaining)
                trade_price = sell.price_per_credit  # Sell price wins

                trade = Trade(
                    buyer_id=buy.team_id,
                    seller_id=sell.team_id,
                    credits=trade_credits,
                    price_per_credit=trade_price,
                    buy_order_id=buy.id,
                    sell_order_id=sell.id,
                )
                self._trades.append(trade)

                # Update orders
                buy.filled += trade_credits
                sell.filled += trade_credits

                if buy.remaining <= 0:
                    buy.status = OrderStatus.FILLED
                else:
                    buy.status = OrderStatus.PARTIALLY_FILLED

                if sell.remaining <= 0:
                    sell.status = OrderStatus.FILLED
                else:
                    sell.status = OrderStatus.PARTIALLY_FILLED

                # Transfer credits
                seller_wallet = self.get_or_create_wallet(sell.team_id)
                buyer_wallet = self.get_or_create_wallet(buy.team_id)

                seller_wallet.release_reserve(trade_credits)
                seller_wallet.withdraw(
                    trade_credits, TransactionType.SALE,
                    counterparty=buy.team_id,
                    price_per_credit=trade_price,
                    description=f"Sold {trade_credits} credits",
                )
                buyer_wallet.deposit(
                    trade_credits, TransactionType.PURCHASE,
                    counterparty=sell.team_id,
                    price_per_credit=trade_price,
                    description=f"Bought {trade_credits} credits",
                )

    def create_lease(
        self, provider_id: str, consumer_id: str,
        credits_per_hour: float, price_per_credit: float,
    ) -> CapacityLease:
        """Create a capacity lease."""
        lease = CapacityLease(
            provider_id=provider_id,
            consumer_id=consumer_id,
            credits_per_hour=credits_per_hour,
            price_per_credit=price_per_credit,
        )
        self._leases.append(lease)
        return lease

    def end_lease(self, lease_id: str) -> CapacityLease | None:
        """End an active lease and settle credits."""
        for lease in self._leases:
            if lease.id == lease_id and lease.active:
                lease.active = False
                lease.end_time = time.time()

                # Settle
                provider_wallet = self.get_or_create_wallet(lease.provider_id)
                consumer_wallet = self.get_or_create_wallet(lease.consumer_id)

                provider_wallet.deposit(
                    lease.total_credits, TransactionType.LEASE_INCOME,
                    counterparty=lease.consumer_id,
                    description=f"Lease income for {lease.duration_hours:.1f}h",
                )
                consumer_wallet.deposit(
                    lease.total_credits, TransactionType.LEASE_EXPENSE,
                    counterparty=lease.provider_id,
                    description=f"Lease capacity for {lease.duration_hours:.1f}h",
                )

                return lease
        return None

    def get_market_stats(self) -> MarketStats:
        """Get current marketplace statistics."""
        open_buys = [
            o for o in self._orders
            if o.side == OrderSide.BUY and o.status == OrderStatus.OPEN
        ]
        open_sells = [
            o for o in self._orders
            if o.side == OrderSide.SELL and o.status == OrderStatus.OPEN
        ]

        bid_price = max((o.price_per_credit for o in open_buys), default=0.0)
        ask_price = min((o.price_per_credit for o in open_sells), default=0.0)

        total_volume = sum(t.total_value for t in self._trades)
        avg_price = (
            sum(t.price_per_credit for t in self._trades) / len(self._trades)
            if self._trades else 0.0
        )

        return MarketStats(
            total_volume=total_volume,
            trades_count=len(self._trades),
            avg_price=avg_price,
            bid_price=bid_price,
            ask_price=ask_price,
            spread=ask_price - bid_price if ask_price > 0 and bid_price > 0 else 0.0,
            open_buy_orders=len(open_buys),
            open_sell_orders=len(open_sells),
        )

    def get_trades(self, team_id: str | None = None) -> list[Trade]:
        if team_id:
            return [
                t for t in self._trades
                if t.buyer_id == team_id or t.seller_id == team_id
            ]
        return list(self._trades)

    def expire_orders(self) -> int:
        """Expire old orders and release reserved credits."""
        expired = 0
        for order in self._orders:
            if order.status == OrderStatus.OPEN and order.is_expired:
                order.status = OrderStatus.EXPIRED
                if order.side == OrderSide.SELL:
                    wallet = self.get_or_create_wallet(order.team_id)
                    wallet.release_reserve(order.remaining)
                expired += 1
        return expired


# Singleton
_marketplace_instance: VerificationMarketplace | None = None


def get_marketplace() -> VerificationMarketplace:
    global _marketplace_instance
    if _marketplace_instance is None:
        _marketplace_instance = VerificationMarketplace()
    return _marketplace_instance


def reset_marketplace() -> None:
    global _marketplace_instance
    _marketplace_instance = None
