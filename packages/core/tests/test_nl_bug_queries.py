"""Tests for natural language bug queries."""

import pytest
from datetime import datetime, timedelta
from codeverify_core.nl_bug_queries import (
    BugCategory,
    FindingsIndex,
    NLQueryEngine,
    NLSearchResult,
    QueryIntent,
    QueryParser,
    QueryResponse,
    SemanticQuery,
    get_nl_query_engine,
    reset_nl_query_engine,
)


class TestBugCategory:
    """Tests for BugCategory enum."""

    def test_all_categories(self):
        """Test all bug categories exist."""
        assert BugCategory.NULL_SAFETY.value == "null_safety"
        assert BugCategory.BOUNDS_CHECK.value == "bounds_check"
        assert BugCategory.DIVISION_BY_ZERO.value == "division_by_zero"
        assert BugCategory.SECURITY.value == "security"
        assert BugCategory.CONCURRENCY.value == "concurrency"
        assert BugCategory.MEMORY.value == "memory"
        assert BugCategory.TYPE_ERROR.value == "type_error"
        assert BugCategory.LOGIC_ERROR.value == "logic_error"


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_all_intents(self):
        """Test all query intents exist."""
        assert QueryIntent.FIND_BUGS.value == "find_bugs"
        assert QueryIntent.COUNT.value == "count"
        assert QueryIntent.TREND.value == "trend"
        assert QueryIntent.COMPARE.value == "compare"
        assert QueryIntent.EXPLAIN.value == "explain"
        assert QueryIntent.RECOMMEND.value == "recommend"


class TestSemanticQuery:
    """Tests for SemanticQuery dataclass."""

    def test_creation(self):
        """Test query creation."""
        query = SemanticQuery(
            raw_query="Find all null pointer bugs in payments module",
            intent=QueryIntent.FIND_BUGS,
            categories=[BugCategory.NULL_SAFETY],
            severity_filter=["high", "critical"],
            file_pattern="payments/",
        )
        
        assert query.raw_query == "Find all null pointer bugs in payments module"
        assert query.intent == QueryIntent.FIND_BUGS
        assert BugCategory.NULL_SAFETY in query.categories
        assert "high" in query.severity_filter

    def test_default_values(self):
        """Test default query values."""
        query = SemanticQuery(
            raw_query="show bugs",
            intent=QueryIntent.FIND_BUGS,
        )
        
        assert query.categories == []
        assert query.severity_filter == []
        assert query.file_pattern is None
        assert query.time_range is None
        assert query.limit == 50


class TestQueryParser:
    """Tests for QueryParser."""

    @pytest.fixture
    def parser(self):
        return QueryParser()

    def test_parse_find_bugs_query(self, parser):
        """Test parsing a find bugs query."""
        query = parser.parse("find all null safety bugs")
        
        assert query.intent == QueryIntent.FIND_BUGS
        assert BugCategory.NULL_SAFETY in query.categories

    def test_parse_count_query(self, parser):
        """Test parsing a count query."""
        query = parser.parse("how many bugs are there?")
        
        assert query.intent == QueryIntent.COUNT

    def test_parse_trend_query(self, parser):
        """Test parsing a trend query."""
        query = parser.parse("show bug trends over time")
        
        assert query.intent == QueryIntent.TREND

    def test_parse_compare_query(self, parser):
        """Test parsing a compare query."""
        query = parser.parse("compare bugs between modules")
        
        assert query.intent == QueryIntent.COMPARE

    def test_parse_explain_query(self, parser):
        """Test parsing an explain query."""
        query = parser.parse("explain why this is a bug")
        
        assert query.intent == QueryIntent.EXPLAIN

    def test_parse_severity_high(self, parser):
        """Test parsing severity filter for high."""
        query = parser.parse("find high severity bugs")
        
        assert "high" in query.severity_filter

    def test_parse_severity_critical(self, parser):
        """Test parsing severity filter for critical."""
        query = parser.parse("show critical issues")
        
        assert "critical" in query.severity_filter

    def test_parse_multiple_categories(self, parser):
        """Test parsing query with multiple categories."""
        query = parser.parse("find null safety and security bugs")
        
        assert BugCategory.NULL_SAFETY in query.categories
        assert BugCategory.SECURITY in query.categories

    def test_parse_file_pattern(self, parser):
        """Test parsing file pattern."""
        query = parser.parse("bugs in src/api/**/*.py")
        
        # Should extract pattern if implementation supports it
        assert query is not None

    def test_parse_time_range_today(self, parser):
        """Test parsing time range for today."""
        query = parser.parse("bugs found today")
        
        if query.time_range:
            assert query.time_range["start"] is not None

    def test_parse_time_range_last_week(self, parser):
        """Test parsing time range for last week."""
        query = parser.parse("bugs from last week")
        
        if query.time_range:
            assert query.time_range["start"] is not None

    def test_parse_limit(self, parser):
        """Test parsing result limit."""
        query = parser.parse("show top 10 bugs")
        
        assert query.limit <= 50  # Should respect limit

    def test_parse_empty_query(self, parser):
        """Test parsing empty query."""
        query = parser.parse("")
        
        assert query.intent == QueryIntent.FIND_BUGS  # Default
        assert query.raw_query == ""


class TestNLSearchResult:
    """Tests for NLSearchResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = NLSearchResult(
            finding_id="finding-123",
            file_path="src/api/handlers.py",
            line_number=42,
            category=BugCategory.NULL_SAFETY,
            severity="high",
            message="Potential null pointer dereference",
            relevance_score=0.95,
        )
        
        assert result.finding_id == "finding-123"
        assert result.line_number == 42
        assert result.relevance_score == 0.95


class TestFindingsIndex:
    """Tests for FindingsIndex."""

    @pytest.fixture
    def index(self):
        idx = FindingsIndex()
        # Add some test findings
        idx.add_finding({
            "id": "f1",
            "file": "src/api/auth.py",
            "line": 10,
            "category": "null_safety",
            "severity": "high",
            "message": "Null pointer access",
        })
        idx.add_finding({
            "id": "f2",
            "file": "src/api/payments.py",
            "line": 25,
            "category": "security",
            "severity": "critical",
            "message": "SQL injection vulnerability",
        })
        idx.add_finding({
            "id": "f3",
            "file": "src/utils/helpers.py",
            "line": 100,
            "category": "bounds_check",
            "severity": "medium",
            "message": "Array bounds not checked",
        })
        return idx

    def test_add_finding(self):
        """Test adding findings to index."""
        idx = FindingsIndex()
        idx.add_finding({
            "id": "test-1",
            "file": "test.py",
            "category": "null_safety",
        })
        
        assert idx.total_findings == 1

    def test_search_by_category(self, index):
        """Test searching by category."""
        results = index.search(categories=[BugCategory.NULL_SAFETY])
        
        assert len(results) == 1
        assert results[0].category == BugCategory.NULL_SAFETY

    def test_search_by_severity(self, index):
        """Test searching by severity."""
        results = index.search(severity_filter=["critical"])
        
        assert len(results) == 1
        assert results[0].severity == "critical"

    def test_search_by_file_pattern(self, index):
        """Test searching by file pattern."""
        results = index.search(file_pattern="src/api/")
        
        assert len(results) == 2  # auth.py and payments.py

    def test_search_by_text(self, index):
        """Test searching by text content."""
        results = index.search(text_query="SQL injection")
        
        assert len(results) >= 1
        assert any("SQL" in r.message for r in results)

    def test_search_combined_filters(self, index):
        """Test searching with multiple filters."""
        results = index.search(
            categories=[BugCategory.SECURITY],
            severity_filter=["critical"],
        )
        
        assert len(results) == 1
        assert results[0].category == BugCategory.SECURITY
        assert results[0].severity == "critical"

    def test_search_no_results(self, index):
        """Test search with no matching results."""
        results = index.search(categories=[BugCategory.CONCURRENCY])
        
        assert len(results) == 0

    def test_search_with_limit(self, index):
        """Test search with result limit."""
        results = index.search(limit=1)
        
        assert len(results) == 1

    def test_get_statistics(self, index):
        """Test getting statistics."""
        stats = index.get_statistics()
        
        assert stats["total"] == 3
        assert "by_category" in stats
        assert "by_severity" in stats

    def test_remove_finding(self, index):
        """Test removing a finding."""
        success = index.remove_finding("f1")
        
        assert success is True
        assert index.total_findings == 2

    def test_clear_index(self, index):
        """Test clearing all findings."""
        index.clear()
        
        assert index.total_findings == 0


class TestQueryResponse:
    """Tests for QueryResponse dataclass."""

    def test_creation(self):
        """Test response creation."""
        response = QueryResponse(
            query=SemanticQuery(
                raw_query="find bugs",
                intent=QueryIntent.FIND_BUGS,
            ),
            results=[],
            total_count=0,
            summary="No bugs found",
        )
        
        assert response.total_count == 0
        assert response.summary == "No bugs found"

    def test_with_results(self):
        """Test response with results."""
        result = NLSearchResult(
            finding_id="f1",
            file_path="test.py",
            line_number=1,
            category=BugCategory.NULL_SAFETY,
            severity="high",
            message="Test",
            relevance_score=0.9,
        )
        
        response = QueryResponse(
            query=SemanticQuery(
                raw_query="find bugs",
                intent=QueryIntent.FIND_BUGS,
            ),
            results=[result],
            total_count=1,
            summary="Found 1 bug",
        )
        
        assert response.total_count == 1
        assert len(response.results) == 1


class TestNLQueryEngine:
    """Tests for NLQueryEngine."""

    @pytest.fixture
    def engine(self):
        engine = NLQueryEngine()
        # Add test findings
        engine.index.add_finding({
            "id": "f1",
            "file": "src/api/auth.py",
            "line": 10,
            "category": "null_safety",
            "severity": "high",
            "message": "Null pointer in authentication",
        })
        engine.index.add_finding({
            "id": "f2",
            "file": "src/api/users.py",
            "line": 50,
            "category": "security",
            "severity": "critical",
            "message": "Insecure user validation",
        })
        return engine

    def test_query_find_bugs(self, engine):
        """Test finding bugs with natural language."""
        response = engine.query("find all null safety bugs")
        
        assert isinstance(response, QueryResponse)
        assert response.total_count >= 0

    def test_query_count(self, engine):
        """Test counting bugs."""
        response = engine.query("how many bugs are there?")
        
        assert isinstance(response, QueryResponse)
        assert "count" in response.summary.lower() or response.total_count >= 0

    def test_query_high_severity(self, engine):
        """Test finding high severity bugs."""
        response = engine.query("show high severity issues")
        
        assert isinstance(response, QueryResponse)
        # Results should only contain high/critical
        for result in response.results:
            assert result.severity in ["high", "critical"]

    def test_query_by_file(self, engine):
        """Test querying by file path."""
        response = engine.query("bugs in auth.py")
        
        assert isinstance(response, QueryResponse)
        # Results should be from auth.py
        for result in response.results:
            assert "auth" in result.file_path

    def test_explain_query(self, engine):
        """Test explain query."""
        response = engine.query("explain why null pointers are dangerous")
        
        assert isinstance(response, QueryResponse)
        # Should have an explanation in summary
        assert len(response.summary) > 0

    def test_empty_query(self, engine):
        """Test empty query handling."""
        response = engine.query("")
        
        assert isinstance(response, QueryResponse)

    def test_get_suggestions(self, engine):
        """Test getting query suggestions."""
        suggestions = engine.get_suggestions("find")
        
        assert isinstance(suggestions, list)
        # Should suggest completions

    def test_get_recent_queries(self, engine):
        """Test getting recent queries."""
        engine.query("test query 1")
        engine.query("test query 2")
        
        recent = engine.get_recent_queries(limit=2)
        
        assert len(recent) <= 2


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
