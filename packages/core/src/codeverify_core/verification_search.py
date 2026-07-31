"""Verification-Aware Code Search.

Semantic code search by verification status, trust score, proof
coverage, and finding history.

Features:
- Search by verification status (verified, unverified, failing)
- Search by trust score range
- Search by finding category/severity
- Natural language queries
- Structured queries with filters
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class SearchField(str, Enum):
    FILE_PATH = "file_path"
    FUNCTION_NAME = "function_name"
    VERIFICATION_STATUS = "verification_status"
    TRUST_SCORE = "trust_score"
    FINDING_CATEGORY = "finding_category"
    SEVERITY = "severity"
    LAST_VERIFIED = "last_verified"
    PROOF_COVERAGE = "proof_coverage"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILING = "failing"
    PARTIAL = "partial"


@dataclass
class CodeEntity:
    """An indexed code entity (file or function)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    file_path: str = ""
    function_name: str = ""
    language: str = "python"
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    trust_score: float = 0.0
    proof_coverage: float = 0.0
    finding_count: int = 0
    critical_findings: int = 0
    categories: list[str] = field(default_factory=list)
    last_verified: datetime | None = None
    last_modified: datetime | None = None
    lines_of_code: int = 0


@dataclass
class SearchQuery:
    """A structured search query."""

    raw_query: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    sort_by: str = "relevance"
    limit: int = 50
    offset: int = 0


@dataclass
class SearchHit:
    """A search result."""

    entity: CodeEntity | None = None
    relevance: float = 0.0
    matched_fields: list[str] = field(default_factory=list)
    snippet: str = ""


@dataclass
class SearchResults:
    """Complete search results."""

    query: str = ""
    hits: list[SearchHit] = field(default_factory=list)
    total_count: int = 0
    elapsed_ms: int = 0


class QueryParser:
    """Parses natural language and structured queries."""

    NL_PATTERNS: list[tuple[str, dict[str, str | int | float]]] = [
        (
            r"unverified (?:functions?|code) in (.+)",
            {"verification_status": "unverified", "file_path_contains": 1},
        ),
        (r"functions? with (?:low|declining) trust", {"trust_score_max": 0.5}),
        (r"critical findings? in (.+)", {"severity": "critical", "file_path_contains": 1}),
        (
            r"verified functions? in (.+)",
            {"verification_status": "verified", "file_path_contains": 1},
        ),
        (
            r"(?:no|without) (?:null safety|null) proofs?",
            {"finding_category": "null_safety", "verification_status": "unverified"},
        ),
        (r"high coverage", {"proof_coverage_min": 0.8}),
        (r"(?:recently|last) modified", {"sort_by": "last_modified"}),
    ]

    def parse(self, query: str) -> SearchQuery:
        filters: dict[str, Any] = {}
        sort_by = "relevance"
        q_lower = query.lower().strip()

        for pattern, template in self.NL_PATTERNS:
            match = re.search(pattern, q_lower)
            if match:
                for key, value in template.items():
                    if key == "sort_by" and isinstance(value, str):
                        sort_by = value
                    elif isinstance(value, int):
                        filters[key] = match.group(value)
                    else:
                        filters[key] = value
                break

        if not filters:
            filters["text_search"] = q_lower

        return SearchQuery(raw_query=query, filters=filters, sort_by=sort_by)


class SearchIndex:
    """In-memory search index for code entities."""

    def __init__(self) -> None:
        self._entities: dict[str, CodeEntity] = {}

    def index(self, entity: CodeEntity) -> None:
        key = (
            f"{entity.file_path}:{entity.function_name}"
            if entity.function_name
            else entity.file_path
        )
        self._entities[key] = entity

    def index_batch(self, entities: list[CodeEntity]) -> int:
        for e in entities:
            self.index(e)
        return len(entities)

    def search(self, query: SearchQuery) -> SearchResults:
        import time

        start = time.time()
        candidates = list(self._entities.values())
        matched: list[SearchHit] = []

        for entity in candidates:
            relevance = 0.0
            fields: list[str] = []

            f = query.filters
            if "verification_status" in f:
                if entity.verification_status.value == f["verification_status"]:
                    relevance += 0.5
                    fields.append("verification_status")
                else:
                    continue

            if "file_path_contains" in f:
                if f["file_path_contains"] in entity.file_path.lower():
                    relevance += 0.3
                    fields.append("file_path")
                else:
                    continue

            if "severity" in f and f["severity"] == "critical":
                if entity.critical_findings > 0:
                    relevance += 0.4
                    fields.append("severity")
                else:
                    continue

            if "finding_category" in f and f["finding_category"] in entity.categories:
                relevance += 0.3
                fields.append("finding_category")

            if "trust_score_max" in f:
                if entity.trust_score <= f["trust_score_max"]:
                    relevance += 0.3
                    fields.append("trust_score")
                else:
                    continue

            if "proof_coverage_min" in f:
                if entity.proof_coverage >= f["proof_coverage_min"]:
                    relevance += 0.3
                    fields.append("proof_coverage")
                else:
                    continue

            if "text_search" in f:
                text = f["text_search"]
                searchable = f"{entity.file_path} {entity.function_name} {' '.join(entity.categories)}".lower()
                if text in searchable:
                    relevance += 0.4
                    fields.append("text")

            if relevance > 0 or not query.filters:
                if not fields:
                    relevance = 0.1
                    fields = ["default"]
                matched.append(
                    SearchHit(entity=entity, relevance=round(relevance, 3), matched_fields=fields)
                )

        if query.sort_by == "trust_score":
            matched.sort(key=lambda h: h.entity.trust_score if h.entity else 0, reverse=True)
        elif query.sort_by == "last_modified":
            matched.sort(
                key=lambda h: (
                    h.entity.last_modified or datetime.min.replace(tzinfo=UTC)
                    if h.entity
                    else datetime.min.replace(tzinfo=UTC)
                ),
                reverse=True,
            )
        else:
            matched.sort(key=lambda h: h.relevance, reverse=True)

        offset = query.offset
        limit = query.limit
        page = matched[offset : offset + limit]

        elapsed = int((time.time() - start) * 1000)
        return SearchResults(
            query=query.raw_query, hits=page, total_count=len(matched), elapsed_ms=elapsed
        )

    def count(self) -> int:
        return len(self._entities)


class VerificationSearchService:
    """Main service for verification-aware code search."""

    def __init__(self) -> None:
        self._parser = QueryParser()
        self._index = SearchIndex()

    def index_entity(self, entity: CodeEntity) -> None:
        self._index.index(entity)

    def index_batch(self, entities: list[CodeEntity]) -> int:
        return self._index.index_batch(entities)

    def search(self, query: str, limit: int = 50) -> SearchResults:
        parsed = self._parser.parse(query)
        parsed.limit = limit
        return self._index.search(parsed)

    def search_structured(self, filters: dict[str, Any], limit: int = 50) -> SearchResults:
        q = SearchQuery(filters=filters, limit=limit)
        return self._index.search(q)

    def get_index_size(self) -> int:
        return self._index.count()


_search_instance: VerificationSearchService | None = None


def get_verification_search_service() -> VerificationSearchService:
    global _search_instance
    if _search_instance is None:
        _search_instance = VerificationSearchService()
    return _search_instance


def reset_verification_search_service() -> None:
    global _search_instance
    _search_instance = None
