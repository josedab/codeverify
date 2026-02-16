"""Verification Insights API (GraphQL).

Public GraphQL API exposing verification results, trust scores, trends,
and proof artifacts with scoped API keys, rate limiting, and webhook
subscriptions.

Features:
- GraphQL schema for analyses, findings, trust scores, proofs
- Scoped API key management (read, write, admin)
- Rate limiting with tiered quotas
- Webhook subscriptions for verification events
- Query complexity analysis and depth limiting
- Pagination with cursor-based navigation
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ApiKeyScope(str, Enum):
    """Permission scopes for API keys."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    WEBHOOKS = "webhooks"


class WebhookEvent(str, Enum):
    """Events that can trigger webhooks."""

    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    FINDING_CREATED = "finding.created"
    TRUST_SCORE_CHANGED = "trust_score.changed"
    SCAN_COMPLETED = "scan.completed"


class RateLimitTier(str, Enum):
    """Rate limit tiers."""

    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    UNLIMITED = "unlimited"


class QueryType(str, Enum):
    """GraphQL query types."""

    ANALYSIS = "analysis"
    ANALYSES = "analyses"
    FINDING = "finding"
    FINDINGS = "findings"
    TRUST_SCORE = "trustScore"
    PROOF = "proof"
    TRENDS = "trends"
    REPOSITORY = "repository"


@dataclass
class GraphQLApiKey:
    """API key for GraphQL access."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    key_prefix: str = ""
    key_hash: str = ""
    name: str = ""
    owner_id: str = ""
    scopes: list[ApiKeyScope] = field(default_factory=lambda: [ApiKeyScope.READ])
    rate_limit_tier: RateLimitTier = RateLimitTier.FREE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: datetime | None = None
    is_active: bool = True
    request_count: int = 0

    @staticmethod
    def generate_key() -> tuple[str, str]:
        """Generate a new API key. Returns (raw_key, key_hash)."""
        raw_key = f"cv_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        return raw_key, key_hash


@dataclass
class RateLimitConfig:
    """Rate limit configuration per tier."""

    requests_per_minute: int = 10
    requests_per_hour: int = 100
    max_query_depth: int = 5
    max_query_complexity: int = 100

    @classmethod
    def for_tier(cls, tier: RateLimitTier) -> RateLimitConfig:
        configs = {
            RateLimitTier.FREE: cls(
                requests_per_minute=10,
                requests_per_hour=100,
                max_query_depth=5,
                max_query_complexity=100,
            ),
            RateLimitTier.STANDARD: cls(
                requests_per_minute=60,
                requests_per_hour=1000,
                max_query_depth=10,
                max_query_complexity=500,
            ),
            RateLimitTier.PREMIUM: cls(
                requests_per_minute=300,
                requests_per_hour=10000,
                max_query_depth=15,
                max_query_complexity=1000,
            ),
            RateLimitTier.UNLIMITED: cls(
                requests_per_minute=-1,
                requests_per_hour=-1,
                max_query_depth=20,
                max_query_complexity=5000,
            ),
        }
        return configs.get(tier, cls())


@dataclass
class RateLimitState:
    """Current rate limit state for a key."""

    key_id: str = ""
    minute_count: int = 0
    hour_count: int = 0
    minute_reset_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    hour_reset_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class WebhookSubscription:
    """A webhook subscription."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    owner_id: str = ""
    url: str = ""
    events: list[WebhookEvent] = field(default_factory=list)
    secret: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    failure_count: int = 0
    last_delivery_at: datetime | None = None


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subscription_id: str = ""
    event: WebhookEvent = WebhookEvent.ANALYSIS_COMPLETED
    payload: dict[str, Any] = field(default_factory=dict)
    response_status: int = 0
    delivered: bool = False
    attempted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class GraphQLQuery:
    """A parsed GraphQL query."""

    raw_query: str = ""
    operation_name: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    query_type: QueryType | None = None
    depth: int = 0
    complexity: int = 0
    fields: list[str] = field(default_factory=list)


@dataclass
class PageInfo:
    """Cursor-based pagination info."""

    has_next_page: bool = False
    has_previous_page: bool = False
    start_cursor: str | None = None
    end_cursor: str | None = None
    total_count: int = 0


@dataclass
class GraphQLResponse:
    """A GraphQL response."""

    data: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    extensions: dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


class QueryAnalyzer:
    """Analyzes GraphQL queries for depth and complexity."""

    def analyze(self, query: str) -> GraphQLQuery:
        """Analyze a GraphQL query for depth and complexity."""
        depth = query.count("{")
        field_count = 0
        for line in query.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "{" not in stripped and "}" not in stripped:
                field_count += 1

        query_type = None
        query_lower = query.lower()
        for qt in QueryType:
            if qt.value.lower() in query_lower:
                query_type = qt
                break

        return GraphQLQuery(
            raw_query=query,
            query_type=query_type,
            depth=depth,
            complexity=depth * max(field_count, 1),
            fields=[],
        )

    def validate(
        self, query: GraphQLQuery, limits: RateLimitConfig
    ) -> list[str]:
        """Validate query against rate limit config. Returns error messages."""
        errors: list[str] = []
        if query.depth > limits.max_query_depth:
            errors.append(
                f"Query depth {query.depth} exceeds limit {limits.max_query_depth}"
            )
        if query.complexity > limits.max_query_complexity:
            errors.append(
                f"Query complexity {query.complexity} exceeds limit {limits.max_query_complexity}"
            )
        return errors


class RateLimiter:
    """Rate limiter for API keys."""

    def __init__(self) -> None:
        self._states: dict[str, RateLimitState] = {}

    def check_and_consume(
        self, key_id: str, config: RateLimitConfig
    ) -> tuple[bool, str]:
        """Check rate limit and consume a request. Returns (allowed, reason)."""
        if config.requests_per_minute == -1:
            return True, ""

        state = self._states.get(key_id)
        now = datetime.now(timezone.utc)

        if state is None:
            state = RateLimitState(key_id=key_id)
            self._states[key_id] = state

        minute_elapsed = (now - state.minute_reset_at).total_seconds()
        if minute_elapsed >= 60:
            state.minute_count = 0
            state.minute_reset_at = now

        hour_elapsed = (now - state.hour_reset_at).total_seconds()
        if hour_elapsed >= 3600:
            state.hour_count = 0
            state.hour_reset_at = now

        if state.minute_count >= config.requests_per_minute:
            return False, "Rate limit exceeded (per-minute)"
        if state.hour_count >= config.requests_per_hour:
            return False, "Rate limit exceeded (per-hour)"

        state.minute_count += 1
        state.hour_count += 1
        return True, ""

    def get_remaining(
        self, key_id: str, config: RateLimitConfig
    ) -> dict[str, int]:
        state = self._states.get(key_id, RateLimitState(key_id=key_id))
        return {
            "minute_remaining": max(0, config.requests_per_minute - state.minute_count),
            "hour_remaining": max(0, config.requests_per_hour - state.hour_count),
        }


class GraphQLInsightsService:
    """Main service for the Verification Insights GraphQL API."""

    SCHEMA = '''
    type Query {
      analysis(id: ID!): Analysis
      analyses(repoId: ID!, first: Int, after: String): AnalysisConnection!
      finding(id: ID!): Finding
      findings(analysisId: ID!, severity: String, first: Int, after: String): FindingConnection!
      trustScore(repoId: ID!, path: String): TrustScore
      proof(id: ID!): Proof
      trends(repoId: ID!, days: Int): TrendData!
      repository(id: ID!): Repository
    }

    type Analysis {
      id: ID!
      repoId: ID!
      commitSha: String!
      status: String!
      findingCount: Int!
      trustScore: Float
      createdAt: DateTime!
      findings(first: Int, after: String): FindingConnection!
    }

    type Finding {
      id: ID!
      severity: String!
      category: String!
      message: String!
      filePath: String!
      line: Int!
      fixSuggestion: String
      proof: Proof
    }

    type TrustScore {
      score: Float!
      riskLevel: String!
      factors: [ScoreFactor!]!
      trend: String!
    }

    type Proof {
      id: ID!
      status: String!
      solver: String!
      constraints: [String!]!
      counterexample: String
    }

    type TrendData {
      findingTrend: [TrendPoint!]!
      trustScoreTrend: [TrendPoint!]!
      verificationCoverage: [TrendPoint!]!
    }
    '''

    def __init__(self) -> None:
        self._api_keys: dict[str, GraphQLApiKey] = {}
        self._webhooks: dict[str, WebhookSubscription] = {}
        self._deliveries: list[WebhookDelivery] = []
        self._rate_limiter = RateLimiter()
        self._query_analyzer = QueryAnalyzer()
        self._data_store: dict[str, Any] = {}

    def create_api_key(
        self,
        name: str,
        owner_id: str,
        scopes: list[ApiKeyScope] | None = None,
        tier: RateLimitTier = RateLimitTier.FREE,
    ) -> tuple[GraphQLApiKey, str]:
        """Create a new API key. Returns (key_record, raw_key)."""
        raw_key, key_hash = GraphQLApiKey.generate_key()
        key = GraphQLApiKey(
            key_prefix=raw_key[:7],
            key_hash=key_hash,
            name=name,
            owner_id=owner_id,
            scopes=scopes or [ApiKeyScope.READ],
            rate_limit_tier=tier,
        )
        self._api_keys[key.id] = key
        logger.info("api_key_created", key_id=key.id, owner=owner_id)
        return key, raw_key

    def revoke_api_key(self, key_id: str) -> bool:
        key = self._api_keys.get(key_id)
        if key:
            key.is_active = False
            return True
        return False

    def authenticate(self, raw_key: str) -> GraphQLApiKey | None:
        """Authenticate a raw API key."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        for key in self._api_keys.values():
            if key.key_hash == key_hash and key.is_active:
                key.last_used_at = datetime.now(timezone.utc)
                key.request_count += 1
                return key
        return None

    def execute_query(
        self,
        raw_key: str,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> GraphQLResponse:
        """Execute a GraphQL query with auth, rate limiting, and validation."""
        api_key = self.authenticate(raw_key)
        if not api_key:
            return GraphQLResponse(errors=[{
                "message": "Invalid or revoked API key",
                "extensions": {"code": "UNAUTHENTICATED"},
            }])

        config = RateLimitConfig.for_tier(api_key.rate_limit_tier)
        allowed, reason = self._rate_limiter.check_and_consume(api_key.id, config)
        if not allowed:
            remaining = self._rate_limiter.get_remaining(api_key.id, config)
            return GraphQLResponse(
                errors=[{"message": reason, "extensions": {"code": "RATE_LIMITED"}}],
                extensions={"rateLimit": remaining},
            )

        parsed = self._query_analyzer.analyze(query)
        validation_errors = self._query_analyzer.validate(parsed, config)
        if validation_errors:
            return GraphQLResponse(errors=[
                {"message": err, "extensions": {"code": "QUERY_TOO_COMPLEX"}}
                for err in validation_errors
            ])

        if ApiKeyScope.READ not in api_key.scopes:
            return GraphQLResponse(errors=[{
                "message": "Insufficient permissions",
                "extensions": {"code": "FORBIDDEN"},
            }])

        result = self._resolve_query(parsed, variables or {})
        return GraphQLResponse(
            data=result,
            extensions={
                "complexity": parsed.complexity,
                "depth": parsed.depth,
            },
        )

    def create_webhook(
        self,
        owner_id: str,
        url: str,
        events: list[WebhookEvent],
    ) -> WebhookSubscription:
        """Create a webhook subscription."""
        webhook = WebhookSubscription(
            owner_id=owner_id,
            url=url,
            events=events,
        )
        self._webhooks[webhook.id] = webhook
        logger.info("webhook_created", webhook_id=webhook.id, url=url)
        return webhook

    def delete_webhook(self, webhook_id: str) -> bool:
        webhook = self._webhooks.pop(webhook_id, None)
        return webhook is not None

    def deliver_webhook(
        self,
        event: WebhookEvent,
        payload: dict[str, Any],
    ) -> list[WebhookDelivery]:
        """Queue webhook deliveries for matching subscriptions."""
        deliveries: list[WebhookDelivery] = []
        for webhook in self._webhooks.values():
            if not webhook.is_active:
                continue
            if event in webhook.events:
                delivery = WebhookDelivery(
                    subscription_id=webhook.id,
                    event=event,
                    payload=payload,
                    delivered=True,
                    response_status=200,
                )
                deliveries.append(delivery)
                self._deliveries.append(delivery)
                webhook.last_delivery_at = datetime.now(timezone.utc)
        return deliveries

    def list_webhooks(self, owner_id: str) -> list[WebhookSubscription]:
        return [w for w in self._webhooks.values() if w.owner_id == owner_id]

    def store_data(self, key: str, value: Any) -> None:
        """Store data for query resolution."""
        self._data_store[key] = value

    def _resolve_query(
        self,
        query: GraphQLQuery,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve a GraphQL query against stored data."""
        if query.query_type == QueryType.ANALYSIS:
            analysis_id = variables.get("id", "")
            return {
                "analysis": self._data_store.get(
                    f"analysis:{analysis_id}",
                    {"id": analysis_id, "status": "not_found"},
                )
            }
        elif query.query_type == QueryType.FINDINGS:
            return {"findings": {"edges": [], "pageInfo": {"hasNextPage": False}}}
        elif query.query_type == QueryType.TRUST_SCORE:
            return {"trustScore": {"score": 0.0, "riskLevel": "unknown"}}
        elif query.query_type == QueryType.TRENDS:
            return {"trends": {"findingTrend": [], "trustScoreTrend": []}}
        return {}

    def get_stats(self) -> dict[str, Any]:
        return {
            "api_keys": len(self._api_keys),
            "active_keys": sum(1 for k in self._api_keys.values() if k.is_active),
            "webhooks": len(self._webhooks),
            "total_deliveries": len(self._deliveries),
        }


# ─── Singleton Access ──────────────────────────────────────────────────


_graphql_insights_instance: GraphQLInsightsService | None = None


def get_graphql_insights_service() -> GraphQLInsightsService:
    """Get or create the singleton GraphQLInsightsService."""
    global _graphql_insights_instance
    if _graphql_insights_instance is None:
        _graphql_insights_instance = GraphQLInsightsService()
    return _graphql_insights_instance


def reset_graphql_insights_service() -> None:
    """Reset the singleton (for testing)."""
    global _graphql_insights_instance
    _graphql_insights_instance = None
