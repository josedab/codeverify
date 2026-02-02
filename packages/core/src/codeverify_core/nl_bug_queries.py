"""Natural Language Bug Queries - Semantic search across verification results.

This module provides:
1. Semantic search across historical findings
2. Natural language query to Z3 constraint translation
3. Similar bug pattern discovery
4. Query autocomplete and suggestions
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import hashlib
import json
import re
import structlog

logger = structlog.get_logger()


class QueryIntent(str, Enum):
    """Types of query intents."""
    FIND_BUGS = "find_bugs"  # "Show me all null pointer bugs"
    CHECK_PROPERTY = "check_property"  # "Can x be null?"
    COMPARE = "compare"  # "How does this compare to the other PR?"
    TREND = "trend"  # "Are null bugs increasing?"
    EXPLAIN = "explain"  # "Why is this a bug?"
    FIX = "fix"  # "How do I fix this?"


class BugCategory(str, Enum):
    """Bug categories for search."""
    NULL_SAFETY = "null_safety"
    BOUNDS = "bounds"
    OVERFLOW = "overflow"
    DIVISION = "division"
    SECURITY = "security"
    CONCURRENCY = "concurrency"
    MEMORY = "memory"
    TYPE = "type"
    LOGIC = "logic"
    ALL = "all"


@dataclass
class SemanticQuery:
    """A parsed semantic query."""
    query_id: str = ""
    original_text: str = ""
    intent: QueryIntent = QueryIntent.FIND_BUGS
    
    # Extracted entities
    bug_category: BugCategory | None = None
    variable_name: str | None = None
    file_pattern: str | None = None
    severity: str | None = None
    time_range: tuple[datetime, datetime] | None = None
    
    # Search parameters
    keywords: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    
    # Confidence
    confidence: float = 0.5


@dataclass
class SearchResult:
    """A search result."""
    finding_id: str = ""
    score: float = 0.0
    category: str = ""
    severity: str = ""
    title: str = ""
    description: str = ""
    file_path: str = ""
    line_number: int = 0
    code_snippet: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    repository: str = ""
    
    # Why this result matched
    match_reason: str = ""


@dataclass
class QueryResponse:
    """Response to a semantic query."""
    query_id: str = ""
    intent: str = ""
    answer: str = ""
    explanation: str = ""
    results: list[SearchResult] = field(default_factory=list)
    total_results: int = 0
    suggestions: list[str] = field(default_factory=list)
    related_queries: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


# Query patterns for intent detection
INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
    QueryIntent.FIND_BUGS: [
        r"show\s+(?:me\s+)?(?:all\s+)?(.+?)\s+bugs?",
        r"find\s+(?:all\s+)?(.+?)\s+(?:bugs?|issues?|problems?)",
        r"list\s+(?:all\s+)?(.+?)\s+(?:bugs?|findings?)",
        r"what\s+(.+?)\s+bugs?\s+(?:are\s+there|exist)",
        r"search\s+for\s+(.+)",
    ],
    QueryIntent.CHECK_PROPERTY: [
        r"can\s+(\w+)\s+(?:ever\s+)?be\s+(.+)",
        r"is\s+(\w+)\s+always\s+(.+)",
        r"will\s+(\w+)\s+(?:ever\s+)?(.+)",
        r"does\s+(\w+)\s+(?:ever\s+)?(.+)",
    ],
    QueryIntent.COMPARE: [
        r"compare\s+(.+?)\s+(?:to|with)\s+(.+)",
        r"how\s+does\s+(.+?)\s+compare",
        r"difference\s+between\s+(.+?)\s+and\s+(.+)",
    ],
    QueryIntent.TREND: [
        r"are\s+(.+?)\s+(?:bugs?|issues?)\s+(?:increasing|decreasing)",
        r"trend\s+(?:for|of|in)\s+(.+)",
        r"how\s+(?:have|has)\s+(.+?)\s+changed",
    ],
    QueryIntent.EXPLAIN: [
        r"why\s+is\s+(.+?)\s+a\s+(?:bug|issue|problem)",
        r"explain\s+(.+)",
        r"what\s+does\s+(.+?)\s+mean",
        r"how\s+does\s+(.+?)\s+work",
    ],
    QueryIntent.FIX: [
        r"how\s+(?:do\s+I|to)\s+fix\s+(.+)",
        r"fix\s+(?:for|suggestion\s+for)\s+(.+)",
        r"solution\s+(?:for|to)\s+(.+)",
    ],
}

# Category patterns
CATEGORY_PATTERNS: dict[BugCategory, list[str]] = {
    BugCategory.NULL_SAFETY: ["null", "none", "undefined", "nil", "npe", "null pointer"],
    BugCategory.BOUNDS: ["bounds", "index", "array", "out of bounds", "oob", "buffer"],
    BugCategory.OVERFLOW: ["overflow", "underflow", "integer overflow"],
    BugCategory.DIVISION: ["division", "divide", "zero", "div by zero"],
    BugCategory.SECURITY: ["security", "sql injection", "xss", "csrf", "vulnerability"],
    BugCategory.CONCURRENCY: ["race", "deadlock", "concurrent", "thread", "lock"],
    BugCategory.MEMORY: ["memory", "leak", "allocation", "free"],
    BugCategory.TYPE: ["type", "cast", "conversion"],
    BugCategory.LOGIC: ["logic", "incorrect", "wrong"],
}


class QueryParser:
    """Parses natural language queries into structured search parameters."""

    def parse(self, query_text: str) -> SemanticQuery:
        """Parse a natural language query."""
        query = SemanticQuery(
            query_id=hashlib.sha256(f"{datetime.utcnow()}-{query_text}".encode()).hexdigest()[:16],
            original_text=query_text,
        )
        
        query_lower = query_text.lower().strip()
        
        # Detect intent
        query.intent, query.confidence = self._detect_intent(query_lower)
        
        # Extract category
        query.bug_category = self._extract_category(query_lower)
        
        # Extract variable names
        query.variable_name = self._extract_variable(query_lower)
        
        # Extract file patterns
        query.file_pattern = self._extract_file_pattern(query_lower)
        
        # Extract severity
        query.severity = self._extract_severity(query_lower)
        
        # Extract keywords
        query.keywords = self._extract_keywords(query_lower)
        
        # Build filters
        query.filters = self._build_filters(query)
        
        return query

    def _detect_intent(self, query: str) -> tuple[QueryIntent, float]:
        """Detect the query intent."""
        best_intent = QueryIntent.FIND_BUGS
        best_confidence = 0.3
        
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent, 0.9
        
        return best_intent, best_confidence

    def _extract_category(self, query: str) -> BugCategory | None:
        """Extract bug category from query."""
        for category, keywords in CATEGORY_PATTERNS.items():
            for keyword in keywords:
                if keyword in query:
                    return category
        return None

    def _extract_variable(self, query: str) -> str | None:
        """Extract variable name from query."""
        # Pattern: "can X be null", "is X always"
        match = re.search(r"(?:can|is|will|does)\s+(\w+)\s+", query)
        if match:
            var = match.group(1)
            if var.lower() not in ("the", "this", "that", "it", "a", "an"):
                return var
        return None

    def _extract_file_pattern(self, query: str) -> str | None:
        """Extract file pattern from query."""
        # Pattern: "in *.py files", "in src/", "in file.ts"
        match = re.search(r"in\s+(?:files?\s+)?([^\s]+\.(?:py|ts|js|java|go|rs))", query)
        if match:
            return match.group(1)
        
        match = re.search(r"in\s+(\S+/)", query)
        if match:
            return match.group(1) + "*"
        
        return None

    def _extract_severity(self, query: str) -> str | None:
        """Extract severity filter from query."""
        severities = ["critical", "high", "medium", "low"]
        for sev in severities:
            if sev in query:
                return sev
        return None

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract search keywords."""
        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "show", "me", "all", "find", "list", "search", "for", "in",
            "to", "of", "and", "or", "with", "that", "this", "it",
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return list(set(keywords))

    def _build_filters(self, query: SemanticQuery) -> dict[str, Any]:
        """Build search filters from parsed query."""
        filters = {}
        
        if query.bug_category and query.bug_category != BugCategory.ALL:
            filters["category"] = query.bug_category.value
        
        if query.severity:
            filters["severity"] = query.severity
        
        if query.file_pattern:
            filters["file_pattern"] = query.file_pattern
        
        if query.variable_name:
            filters["variable"] = query.variable_name
        
        return filters


class FindingsIndex:
    """In-memory index of findings for semantic search."""

    def __init__(self) -> None:
        self._findings: dict[str, dict[str, Any]] = {}
        self._category_index: dict[str, set[str]] = {}
        self._severity_index: dict[str, set[str]] = {}
        self._keyword_index: dict[str, set[str]] = {}

    def add_finding(self, finding: dict[str, Any]) -> None:
        """Add a finding to the index."""
        finding_id = finding.get("id", str(hash(json.dumps(finding, default=str))))
        self._findings[finding_id] = finding
        
        # Index by category
        category = finding.get("category", "unknown")
        if category not in self._category_index:
            self._category_index[category] = set()
        self._category_index[category].add(finding_id)
        
        # Index by severity
        severity = finding.get("severity", "unknown")
        if severity not in self._severity_index:
            self._severity_index[severity] = set()
        self._severity_index[severity].add(finding_id)
        
        # Index by keywords
        text = f"{finding.get('title', '')} {finding.get('description', '')}".lower()
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            if len(word) > 3:
                if word not in self._keyword_index:
                    self._keyword_index[word] = set()
                self._keyword_index[word].add(finding_id)

    def search(
        self,
        keywords: list[str] | None = None,
        category: str | None = None,
        severity: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search findings."""
        candidate_ids: set[str] | None = None
        
        # Filter by category
        if category and category in self._category_index:
            category_ids = self._category_index[category]
            candidate_ids = category_ids if candidate_ids is None else candidate_ids & category_ids
        
        # Filter by severity
        if severity and severity in self._severity_index:
            severity_ids = self._severity_index[severity]
            candidate_ids = severity_ids if candidate_ids is None else candidate_ids & severity_ids
        
        # If no filters, start with all findings
        if candidate_ids is None:
            candidate_ids = set(self._findings.keys())
        
        # Score by keyword matches
        scores: dict[str, float] = {}
        for finding_id in candidate_ids:
            scores[finding_id] = 0.0
            
            if keywords:
                for keyword in keywords:
                    if keyword in self._keyword_index:
                        if finding_id in self._keyword_index[keyword]:
                            scores[finding_id] += 1.0
        
        # Sort by score
        sorted_ids = sorted(candidate_ids, key=lambda x: scores[x], reverse=True)
        
        return [self._findings[fid] for fid in sorted_ids[:limit]]

    def get_finding(self, finding_id: str) -> dict[str, Any] | None:
        """Get a finding by ID."""
        return self._findings.get(finding_id)

    def get_category_counts(self) -> dict[str, int]:
        """Get counts by category."""
        return {cat: len(ids) for cat, ids in self._category_index.items()}

    def get_severity_counts(self) -> dict[str, int]:
        """Get counts by severity."""
        return {sev: len(ids) for sev, ids in self._severity_index.items()}


class NLQueryEngine:
    """Natural language query engine for bug search."""

    def __init__(self) -> None:
        self.parser = QueryParser()
        self.index = FindingsIndex()
        self._query_history: list[SemanticQuery] = []

    def index_finding(self, finding: dict[str, Any]) -> None:
        """Add a finding to the searchable index."""
        self.index.add_finding(finding)

    def index_findings(self, findings: list[dict[str, Any]]) -> None:
        """Add multiple findings to the index."""
        for finding in findings:
            self.index.add_finding(finding)

    async def query(self, query_text: str) -> QueryResponse:
        """Execute a natural language query."""
        import time
        start = time.time()
        
        # Parse query
        parsed = self.parser.parse(query_text)
        self._query_history.append(parsed)
        
        # Route based on intent
        if parsed.intent == QueryIntent.FIND_BUGS:
            response = self._handle_find_bugs(parsed)
        elif parsed.intent == QueryIntent.CHECK_PROPERTY:
            response = self._handle_check_property(parsed)
        elif parsed.intent == QueryIntent.EXPLAIN:
            response = self._handle_explain(parsed)
        elif parsed.intent == QueryIntent.FIX:
            response = self._handle_fix(parsed)
        elif parsed.intent == QueryIntent.TREND:
            response = self._handle_trend(parsed)
        else:
            response = self._handle_find_bugs(parsed)  # Default to search
        
        response.query_id = parsed.query_id
        response.intent = parsed.intent.value
        response.processing_time_ms = (time.time() - start) * 1000
        
        # Add suggestions
        response.related_queries = self._generate_related_queries(parsed)
        
        return response

    def _handle_find_bugs(self, query: SemanticQuery) -> QueryResponse:
        """Handle bug search queries."""
        results = self.index.search(
            keywords=query.keywords,
            category=query.filters.get("category"),
            severity=query.filters.get("severity"),
            limit=20,
        )
        
        search_results = [
            SearchResult(
                finding_id=r.get("id", ""),
                score=1.0,
                category=r.get("category", ""),
                severity=r.get("severity", ""),
                title=r.get("title", ""),
                description=r.get("description", ""),
                file_path=r.get("file_path", ""),
                line_number=r.get("line_start", 0),
                code_snippet=r.get("code_snippet", ""),
                match_reason=f"Matched keywords: {', '.join(query.keywords[:3])}",
            )
            for r in results
        ]
        
        # Generate answer
        if search_results:
            answer = f"Found {len(search_results)} matching bugs"
            if query.bug_category:
                answer += f" in category '{query.bug_category.value}'"
            if query.severity:
                answer += f" with severity '{query.severity}'"
        else:
            answer = "No matching bugs found"
        
        return QueryResponse(
            answer=answer,
            explanation=f"Searched for: {', '.join(query.keywords)}" if query.keywords else "No specific keywords",
            results=search_results,
            total_results=len(search_results),
            suggestions=self._generate_suggestions(query),
        )

    def _handle_check_property(self, query: SemanticQuery) -> QueryResponse:
        """Handle property check queries."""
        var = query.variable_name or "variable"
        
        # Search for related findings
        if query.bug_category:
            keywords = [var] if var != "variable" else []
            results = self.index.search(
                keywords=keywords,
                category=query.bug_category.value,
                limit=5,
            )
            
            if results:
                answer = f"Yes, {var} could potentially have issues. Found {len(results)} related findings."
            else:
                answer = f"No known issues found for {var}."
        else:
            answer = f"Unable to determine property for {var}. Try being more specific."
        
        return QueryResponse(
            answer=answer,
            explanation="Checked historical findings for similar patterns.",
            results=[],
            suggestions=[
                f"Try: 'Show me all bugs involving {var}'",
                f"Try: 'Can {var} be null?'",
            ],
        )

    def _handle_explain(self, query: SemanticQuery) -> QueryResponse:
        """Handle explanation queries."""
        category = query.bug_category
        
        explanations = {
            BugCategory.NULL_SAFETY: (
                "Null safety bugs occur when code assumes a value exists but it might be null/None. "
                "This can cause crashes or unexpected behavior. Always check for null before accessing properties."
            ),
            BugCategory.BOUNDS: (
                "Bounds checking bugs happen when accessing arrays/lists with an index that could be "
                "outside valid range. This causes IndexError in Python or crashes in other languages."
            ),
            BugCategory.DIVISION: (
                "Division by zero occurs when the divisor could be zero. This causes ZeroDivisionError "
                "and should be guarded with a check before dividing."
            ),
            BugCategory.OVERFLOW: (
                "Integer overflow happens when arithmetic results exceed the maximum value for the type. "
                "In some languages this wraps around, in others it may cause errors."
            ),
            BugCategory.SECURITY: (
                "Security vulnerabilities include SQL injection, XSS, CSRF, and other attacks. "
                "Always sanitize user input and use parameterized queries."
            ),
        }
        
        if category:
            answer = explanations.get(category, f"No detailed explanation available for {category.value}.")
        else:
            answer = "Please specify what type of bug you want explained."
        
        return QueryResponse(
            answer=answer,
            explanation="",
            suggestions=[
                "Try: 'Explain null pointer bugs'",
                "Try: 'Why is division by zero dangerous?'",
            ],
        )

    def _handle_fix(self, query: SemanticQuery) -> QueryResponse:
        """Handle fix suggestion queries."""
        fixes = {
            BugCategory.NULL_SAFETY: [
                "Add null check: `if value is not None:`",
                "Use Optional type hints: `def func(x: Optional[T])`",
                "Use default values: `value = x or default`",
            ],
            BugCategory.BOUNDS: [
                "Check bounds: `if 0 <= index < len(array):`",
                "Use get() for dicts: `d.get(key, default)`",
                "Use try/except IndexError",
            ],
            BugCategory.DIVISION: [
                "Guard against zero: `if divisor != 0:`",
                "Use default: `result = x / divisor if divisor else 0`",
            ],
        }
        
        category = query.bug_category
        if category and category in fixes:
            suggestions = fixes[category]
            answer = f"To fix {category.value} bugs:\n" + "\n".join(f"• {s}" for s in suggestions)
        else:
            answer = "Specify the type of bug you want to fix."
        
        return QueryResponse(
            answer=answer,
            explanation="Common fix patterns for this bug category.",
            suggestions=["Try: 'How to fix null pointer bugs?'"],
        )

    def _handle_trend(self, query: SemanticQuery) -> QueryResponse:
        """Handle trend queries."""
        counts = self.index.get_category_counts()
        
        if query.bug_category:
            count = counts.get(query.bug_category.value, 0)
            answer = f"Found {count} {query.bug_category.value} bugs in the index."
        else:
            total = sum(counts.values())
            answer = f"Total bugs indexed: {total}\n"
            answer += "\n".join(f"• {cat}: {cnt}" for cat, cnt in sorted(counts.items(), key=lambda x: -x[1])[:5])
        
        return QueryResponse(
            answer=answer,
            explanation="Bug counts from indexed findings.",
        )

    def _generate_suggestions(self, query: SemanticQuery) -> list[str]:
        """Generate search suggestions."""
        suggestions = []
        
        if not query.bug_category:
            suggestions.append("Filter by category: 'null safety bugs', 'bounds check issues'")
        
        if not query.severity:
            suggestions.append("Filter by severity: 'high severity bugs'")
        
        return suggestions

    def _generate_related_queries(self, query: SemanticQuery) -> list[str]:
        """Generate related queries."""
        related = []
        
        if query.bug_category:
            related.append(f"How to fix {query.bug_category.value} bugs?")
            related.append(f"Explain {query.bug_category.value} issues")
        
        if query.variable_name:
            related.append(f"All bugs involving {query.variable_name}")
        
        return related[:3]

    def get_autocomplete(self, prefix: str, limit: int = 5) -> list[str]:
        """Get autocomplete suggestions."""
        suggestions = [
            "Show me all null safety bugs",
            "Find high severity issues",
            "Can variable be null?",
            "How to fix bounds check errors",
            "Explain division by zero",
            "Search for security vulnerabilities",
            "List all critical bugs",
            "Compare this PR to previous",
        ]
        
        prefix_lower = prefix.lower()
        matches = [s for s in suggestions if s.lower().startswith(prefix_lower)]
        
        return matches[:limit]


# Global engine instance
_nl_query_engine: NLQueryEngine | None = None


def get_nl_query_engine() -> NLQueryEngine:
    """Get or create the global NL query engine."""
    global _nl_query_engine
    if _nl_query_engine is None:
        _nl_query_engine = NLQueryEngine()
    return _nl_query_engine


def reset_nl_query_engine() -> None:
    """Reset the global NL query engine (for testing)."""
    global _nl_query_engine
    _nl_query_engine = None
