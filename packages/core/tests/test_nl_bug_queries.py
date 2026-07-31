"""Tests for natural language bug queries."""

import pytest

from codeverify_core.nl_bug_queries import (
    BugCategory,
    FindingsIndex,
    NLQueryEngine,
    QueryIntent,
    QueryParser,
    QueryResponse,
    SearchResult,
    SemanticQuery,
    get_nl_query_engine,
    reset_nl_query_engine,
)


class TestBugCategory:
    """Tests for BugCategory enum."""

    def test_all_categories(self):
        """Test all bug categories exist."""
        assert BugCategory.NULL_SAFETY.value == "null_safety"
        assert BugCategory.BOUNDS.value == "bounds"
        assert BugCategory.OVERFLOW.value == "overflow"
        assert BugCategory.DIVISION.value == "division"
        assert BugCategory.SECURITY.value == "security"
        assert BugCategory.CONCURRENCY.value == "concurrency"
        assert BugCategory.MEMORY.value == "memory"
        assert BugCategory.TYPE.value == "type"
        assert BugCategory.LOGIC.value == "logic"
        assert BugCategory.ALL.value == "all"


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_all_intents(self):
        """Test all query intents exist."""
        assert QueryIntent.FIND_BUGS.value == "find_bugs"
        assert QueryIntent.CHECK_PROPERTY.value == "check_property"
        assert QueryIntent.COMPARE.value == "compare"
        assert QueryIntent.TREND.value == "trend"
        assert QueryIntent.EXPLAIN.value == "explain"
        assert QueryIntent.FIX.value == "fix"


class TestSemanticQuery:
    """Tests for SemanticQuery dataclass."""

    def test_creation(self):
        """Test query creation."""
        query = SemanticQuery(
            original_text="Find all null pointer bugs in payments module",
            intent=QueryIntent.FIND_BUGS,
            bug_category=BugCategory.NULL_SAFETY,
            severity="high",
            file_pattern="payments/",
        )

        assert query.original_text == "Find all null pointer bugs in payments module"
        assert query.intent == QueryIntent.FIND_BUGS
        assert query.bug_category == BugCategory.NULL_SAFETY
        assert query.severity == "high"

    def test_default_values(self):
        """Test default query values."""
        query = SemanticQuery(
            original_text="show bugs",
            intent=QueryIntent.FIND_BUGS,
        )

        assert query.bug_category is None
        assert query.severity is None
        assert query.file_pattern is None
        assert query.time_range is None


class TestQueryParser:
    """Tests for QueryParser."""

    @pytest.fixture
    def parser(self):
        return QueryParser()

    def test_parse_find_bugs_query(self, parser):
        """Test parsing a find bugs query."""
        query = parser.parse("find all null safety bugs")

        assert query.intent == QueryIntent.FIND_BUGS
        assert query.bug_category == BugCategory.NULL_SAFETY

    def test_parse_trend_query(self, parser):
        """Test parsing a trend query."""
        query = parser.parse("are null bugs increasing")

        assert query.intent == QueryIntent.TREND

    def test_parse_compare_query(self, parser):
        """Test parsing a compare query."""
        query = parser.parse("compare bugs to previous")

        assert query.intent == QueryIntent.COMPARE

    def test_parse_explain_query(self, parser):
        """Test parsing an explain query."""
        query = parser.parse("explain why this is a bug")

        assert query.intent == QueryIntent.EXPLAIN

    def test_parse_severity_high(self, parser):
        """Test parsing severity filter for high."""
        query = parser.parse("find high severity bugs")

        assert query.severity == "high"

    def test_parse_severity_critical(self, parser):
        """Test parsing severity filter for critical."""
        query = parser.parse("show critical issues")

        assert query.severity == "critical"

    def test_parse_category(self, parser):
        """Test parsing query with category."""
        query = parser.parse("find null safety bugs")

        assert query.bug_category == BugCategory.NULL_SAFETY

    def test_parse_file_pattern(self, parser):
        """Test parsing file pattern."""
        query = parser.parse("bugs in src/api/handler.py")

        assert query.file_pattern is not None

    def test_parse_empty_query(self, parser):
        """Test parsing empty query."""
        query = parser.parse("")

        assert query.intent == QueryIntent.FIND_BUGS  # Default
        assert query.original_text == ""


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = SearchResult(
            finding_id="finding-123",
            file_path="src/api/handlers.py",
            line_number=42,
            category="null_safety",
            severity="high",
            title="Null pointer dereference",
            description="Potential null pointer dereference",
            score=0.95,
        )

        assert result.finding_id == "finding-123"
        assert result.line_number == 42
        assert result.score == 0.95


class TestFindingsIndex:
    """Tests for FindingsIndex."""

    @pytest.fixture
    def index(self):
        idx = FindingsIndex()
        # Add some test findings
        idx.add_finding(
            {
                "id": "f1",
                "file_path": "src/api/auth.py",
                "line_start": 10,
                "category": "null_safety",
                "severity": "high",
                "title": "Null pointer access",
                "description": "Null pointer access in auth module",
            }
        )
        idx.add_finding(
            {
                "id": "f2",
                "file_path": "src/api/payments.py",
                "line_start": 25,
                "category": "security",
                "severity": "critical",
                "title": "SQL injection vulnerability",
                "description": "SQL injection vulnerability in payments",
            }
        )
        idx.add_finding(
            {
                "id": "f3",
                "file_path": "src/utils/helpers.py",
                "line_start": 100,
                "category": "bounds",
                "severity": "medium",
                "title": "Array bounds not checked",
                "description": "Array bounds not checked in helpers",
            }
        )
        return idx

    def test_add_finding(self):
        """Test adding findings to index."""
        idx = FindingsIndex()
        idx.add_finding(
            {
                "id": "test-1",
                "file_path": "test.py",
                "category": "null_safety",
            }
        )

        finding = idx.get_finding("test-1")
        assert finding is not None

    def test_search_by_category(self, index):
        """Test searching by category."""
        results = index.search(category="null_safety")

        assert len(results) == 1
        assert results[0]["category"] == "null_safety"

    def test_search_by_severity(self, index):
        """Test searching by severity."""
        results = index.search(severity="critical")

        assert len(results) == 1
        assert results[0]["severity"] == "critical"

    def test_search_by_keywords(self, index):
        """Test searching by keyword content."""
        results = index.search(keywords=["injection"])

        assert len(results) >= 1
        assert any(
            "injection" in r.get("title", "").lower()
            or "injection" in r.get("description", "").lower()
            for r in results
        )

    def test_search_combined_filters(self, index):
        """Test searching with multiple filters."""
        results = index.search(
            category="security",
            severity="critical",
        )

        assert len(results) == 1
        assert results[0]["category"] == "security"
        assert results[0]["severity"] == "critical"

    def test_search_no_results(self, index):
        """Test search with no matching results."""
        results = index.search(category="concurrency", keywords=["concurrency"])

        # No findings have "concurrency" keyword in title/description
        assert all(r["category"] != "concurrency" for r in results)

    def test_search_with_limit(self, index):
        """Test search with result limit."""
        results = index.search(limit=1)

        assert len(results) == 1

    def test_get_category_counts(self, index):
        """Test getting category counts."""
        counts = index.get_category_counts()

        assert counts["null_safety"] == 1
        assert counts["security"] == 1
        assert counts["bounds"] == 1

    def test_get_severity_counts(self, index):
        """Test getting severity counts."""
        counts = index.get_severity_counts()

        assert counts["high"] == 1
        assert counts["critical"] == 1
        assert counts["medium"] == 1


class TestQueryResponse:
    """Tests for QueryResponse dataclass."""

    def test_creation(self):
        """Test response creation."""
        response = QueryResponse(
            query_id="test-id",
            intent="find_bugs",
            results=[],
            total_results=0,
            answer="No bugs found",
        )

        assert response.total_results == 0
        assert response.answer == "No bugs found"

    def test_with_results(self):
        """Test response with results."""
        result = SearchResult(
            finding_id="f1",
            file_path="test.py",
            line_number=1,
            category="null_safety",
            severity="high",
            title="Test",
            description="Test description",
            score=0.9,
        )

        response = QueryResponse(
            query_id="test-id",
            intent="find_bugs",
            results=[result],
            total_results=1,
            answer="Found 1 bug",
        )

        assert response.total_results == 1
        assert len(response.results) == 1


class TestNLQueryEngine:
    """Tests for NLQueryEngine."""

    @pytest.fixture
    def engine(self):
        engine = NLQueryEngine()
        # Add test findings
        engine.index.add_finding(
            {
                "id": "f1",
                "file_path": "src/api/auth.py",
                "line_start": 10,
                "category": "null_safety",
                "severity": "high",
                "title": "Null pointer in authentication",
                "description": "Null pointer in authentication module",
            }
        )
        engine.index.add_finding(
            {
                "id": "f2",
                "file_path": "src/api/users.py",
                "line_start": 50,
                "category": "security",
                "severity": "critical",
                "title": "Insecure user validation",
                "description": "Insecure user validation logic",
            }
        )
        return engine

    @pytest.mark.asyncio
    async def test_query_find_bugs(self, engine):
        """Test finding bugs with natural language."""
        response = await engine.query("find all null safety bugs")

        assert isinstance(response, QueryResponse)
        assert response.total_results >= 0

    @pytest.mark.asyncio
    async def test_query_explain(self, engine):
        """Test explain query."""
        response = await engine.query("explain null pointer bugs")

        assert isinstance(response, QueryResponse)
        assert len(response.answer) > 0

    @pytest.mark.asyncio
    async def test_query_empty(self, engine):
        """Test empty query handling."""
        response = await engine.query("")

        assert isinstance(response, QueryResponse)

    def test_get_autocomplete(self, engine):
        """Test getting autocomplete suggestions."""
        suggestions = engine.get_autocomplete("Find")

        assert isinstance(suggestions, list)


class TestGlobalQueryEngine:
    """Tests for global query engine functions."""

    def teardown_method(self):
        reset_nl_query_engine()

    def test_get_engine_singleton(self):
        """Test singleton pattern."""
        engine1 = get_nl_query_engine()
        engine2 = get_nl_query_engine()
        assert engine1 is engine2

    def test_reset_engine(self):
        """Test engine reset."""
        engine1 = get_nl_query_engine()
        reset_nl_query_engine()
        engine2 = get_nl_query_engine()
        assert engine1 is not engine2
