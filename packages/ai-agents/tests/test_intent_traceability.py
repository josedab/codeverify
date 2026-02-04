"""Tests for Intent-to-Code Traceability."""

import pytest
from datetime import datetime

from codeverify_agents.intent_traceability import (
    AlignmentChecker,
    ChangeScope,
    CodeChangeAnalyzer,
    CodeChangeSummary,
    ExtractedIntent,
    IntentExtractor,
    IntentTraceabilityEngine,
    IssueDetails,
    IssueProvider,
    TraceabilityFinding,
    TraceabilityResult,
    TraceabilityStatus,
    create_traceability_engine,
)


class TestIssueDetails:
    """Tests for IssueDetails."""
    
    def test_to_dict(self):
        issue = IssueDetails(
            id="123",
            provider=IssueProvider.JIRA,
            key="PROJ-123",
            title="Add user authentication",
            description="Implement OAuth2 login",
            issue_type="feature",
            status="In Progress",
            labels=["security", "auth"],
            url="https://jira.example.com/PROJ-123",
        )
        
        d = issue.to_dict()
        
        assert d["key"] == "PROJ-123"
        assert d["provider"] == "jira"
        assert "auth" in d["labels"]


class TestIntentExtractor:
    """Tests for IntentExtractor."""
    
    @pytest.fixture
    def extractor(self):
        return IntentExtractor()
    
    def test_extract_feature_intent(self, extractor):
        issue = IssueDetails(
            id="1",
            provider=IssueProvider.JIRA,
            key="PROJ-1",
            title="Add rate limiting to API endpoints",
            description="""
            Implement rate limiting for all public API endpoints.
            
            - Add RateLimiter class in utils/
            - Apply to /api/v1/* routes
            - Configure limits via environment variables
            
            Should not modify authentication logic.
            """,
            issue_type="feature",
            status="Open",
            acceptance_criteria=["Rate limit configurable", "Returns 429 when exceeded"],
        )
        
        intent = extractor.extract_intent(issue)
        
        assert intent.change_scope == ChangeScope.FEATURE
        assert "rate" in intent.keywords or "limit" in intent.keywords
        assert len(intent.expected_changes) > 0
        assert len(intent.constraints) > 0
    
    def test_extract_bug_fix_intent(self, extractor):
        issue = IssueDetails(
            id="2",
            provider=IssueProvider.GITHUB,
            key="#42",
            title="Fix null pointer in user service",
            description="Users are getting crashes when profile is not set.",
            issue_type="bug",
            status="Open",
        )
        
        intent = extractor.extract_intent(issue)
        
        assert intent.change_scope == ChangeScope.BUG_FIX
    
    def test_extract_security_intent(self, extractor):
        issue = IssueDetails(
            id="3",
            provider=IssueProvider.LINEAR,
            key="SEC-123",
            title="Patch SQL injection vulnerability",
            description="Fix CVE-2024-1234 in user input handling",
            issue_type="security",
            status="Urgent",
        )
        
        intent = extractor.extract_intent(issue)
        
        assert intent.change_scope == ChangeScope.SECURITY
    
    def test_extract_affected_areas(self, extractor):
        issue = IssueDetails(
            id="4",
            provider=IssueProvider.JIRA,
            key="PROJ-4",
            title="Update logging",
            description="""
            Update logging in these files:
            - src/services/auth.py
            - src/api/users.ts
            
            Component: AuthService
            Module: authentication
            """,
            issue_type="task",
            status="Open",
        )
        
        intent = extractor.extract_intent(issue)
        
        assert len(intent.affected_areas) > 0
        # Should find file paths and component names
    
    def test_extract_expected_changes(self, extractor):
        issue = IssueDetails(
            id="5",
            provider=IssueProvider.JIRA,
            key="PROJ-5",
            title="Refactor database layer",
            description="""
            Changes needed:
            - Migrate from raw SQL to ORM
            - Add connection pooling
            - Update all repository classes
            
            1. Create base repository class
            2. Implement caching layer
            3. Add metrics
            """,
            issue_type="task",
            status="Open",
        )
        
        intent = extractor.extract_intent(issue)
        
        assert len(intent.expected_changes) >= 3


class TestCodeChangeAnalyzer:
    """Tests for CodeChangeAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return CodeChangeAnalyzer()
    
    def test_analyze_diff(self, analyzer):
        diff = """
diff --git a/src/auth.py b/src/auth.py
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,12 @@
+def validate_token(token):
+    if not token:
+        return False
+    return True
"""
        files_changed = [
            {"filename": "src/auth.py", "status": "modified", "additions": 6, "deletions": 0},
            {"filename": "tests/test_auth.py", "status": "added", "additions": 20, "deletions": 0},
        ]
        
        summary = analyzer.analyze_diff(diff, files_changed)
        
        assert "src/auth.py" in summary.files_modified
        assert "tests/test_auth.py" in summary.files_added
        assert summary.lines_added == 26
        assert "validate_token" in summary.functions_modified
    
    def test_detect_scope_from_security_diff(self, analyzer):
        diff = "+password = encrypt(user_input)"
        files = ["src/auth.py"]
        
        scope = analyzer._detect_scope_from_diff(diff, files)
        
        assert scope == ChangeScope.SECURITY
    
    def test_detect_scope_from_test_files(self, analyzer):
        diff = "+def test_feature():\n+    assert True"
        files = ["tests/test_feature.py", "spec/feature_spec.py"]
        
        scope = analyzer._detect_scope_from_diff(diff, files)
        
        assert scope == ChangeScope.TESTING
    
    def test_detect_scope_from_config_files(self, analyzer):
        diff = "+DEBUG=true"
        files = [".env", "config.yaml"]
        
        scope = analyzer._detect_scope_from_diff(diff, files)
        
        assert scope == ChangeScope.CONFIGURATION
    
    def test_extract_functions_from_python(self, analyzer):
        diff = """
+def new_function():
+    pass
+
+def another_function(x):
+    return x
"""
        functions = analyzer._extract_modified_functions(diff)
        
        assert "new_function" in functions
        assert "another_function" in functions
    
    def test_extract_classes(self, analyzer):
        diff = """
+class NewClass:
+    pass
+
+class AnotherClass(BaseClass):
+    pass
"""
        classes = analyzer._extract_modified_classes(diff)
        
        assert "NewClass" in classes
        assert "AnotherClass" in classes


class TestAlignmentChecker:
    """Tests for AlignmentChecker."""
    
    @pytest.fixture
    def checker(self):
        return AlignmentChecker()
    
    @pytest.fixture
    def feature_intent(self):
        return ExtractedIntent(
            primary_intent="Add rate limiting",
            expected_changes=["Add RateLimiter class", "Apply to API routes"],
            affected_areas=["src/utils/", "src/api/"],
            change_scope=ChangeScope.FEATURE,
            keywords=["rate", "limit", "api", "throttle"],
            acceptance_criteria=["Configurable limits", "Returns 429"],
            constraints=["Do not modify authentication"],
            confidence=0.8,
        )
    
    def test_aligned_changes(self, checker, feature_intent):
        changes = CodeChangeSummary(
            files_modified=["src/utils/rate_limiter.py", "src/api/routes.py"],
            files_added=["src/utils/throttle.py"],
            files_deleted=[],
            functions_modified=["rate_limit", "apply_limits"],
            classes_modified=["RateLimiter"],
            detected_scope=ChangeScope.FEATURE,
            change_keywords=["rate", "limit", "throttle", "api"],
            lines_added=100,
            lines_deleted=10,
        )
        
        score, findings = checker.check_alignment(feature_intent, changes)
        
        assert score > 0.7  # Should be well-aligned
        assert not any(f.type == "scope_mismatch" for f in findings)
    
    def test_scope_mismatch(self, checker, feature_intent):
        changes = CodeChangeSummary(
            files_modified=["src/auth.py"],
            files_added=[],
            files_deleted=[],
            functions_modified=["fix_bug"],
            classes_modified=[],
            detected_scope=ChangeScope.BUG_FIX,  # Mismatch!
            change_keywords=["fix", "bug", "error"],
            lines_added=5,
            lines_deleted=2,
        )
        
        score, findings = checker.check_alignment(feature_intent, changes)
        
        assert any(f.type == "scope_mismatch" for f in findings)
    
    def test_unexpected_files(self, checker, feature_intent):
        changes = CodeChangeSummary(
            files_modified=[
                "src/utils/rate.py",  # Expected
                "src/database/models.py",  # Unexpected!
                "src/billing/payments.py",  # Unexpected!
            ],
            files_added=[],
            files_deleted=[],
            functions_modified=[],
            classes_modified=[],
            detected_scope=ChangeScope.FEATURE,
            change_keywords=["rate", "database", "payment"],
            lines_added=50,
            lines_deleted=10,
        )
        
        score, findings = checker.check_alignment(feature_intent, changes)
        
        assert any(f.type == "unexpected_files" for f in findings)
    
    def test_constraint_violation(self, checker, feature_intent):
        changes = CodeChangeSummary(
            files_modified=[
                "src/utils/rate.py",
                "src/auth/authentication.py",  # Violates constraint!
            ],
            files_added=[],
            files_deleted=[],
            functions_modified=["authenticate"],  # Violates constraint!
            classes_modified=[],
            detected_scope=ChangeScope.FEATURE,
            change_keywords=["rate", "auth"],
            lines_added=50,
            lines_deleted=10,
        )
        
        score, findings = checker.check_alignment(feature_intent, changes)
        
        assert any(f.type == "constraint_violation" for f in findings)
    
    def test_keyword_mismatch(self, checker, feature_intent):
        changes = CodeChangeSummary(
            files_modified=["src/utils/something.py"],
            files_added=[],
            files_deleted=[],
            functions_modified=[],
            classes_modified=[],
            detected_scope=ChangeScope.FEATURE,
            change_keywords=["completely", "different", "keywords"],
            lines_added=10,
            lines_deleted=5,
        )
        
        score, findings = checker.check_alignment(feature_intent, changes)
        
        assert any(f.type == "keyword_mismatch" for f in findings)


class TestIntentTraceabilityEngine:
    """Tests for IntentTraceabilityEngine."""
    
    @pytest.fixture
    def engine(self):
        return IntentTraceabilityEngine(provider_configs={
            "github": {},
        })
    
    @pytest.mark.asyncio
    async def test_check_traceability_no_ticket(self, engine):
        pr_data = {
            "title": "Some changes",
            "body": "Made some updates",
        }
        
        result = await engine.check_traceability(
            pr_data=pr_data,
            diff="+ some code",
            files_changed=[{"filename": "test.py", "status": "modified"}],
        )
        
        assert result.status == TraceabilityStatus.NO_TICKET
        assert any("Link a ticket" in r for r in result.recommendations)
    
    @pytest.mark.asyncio
    async def test_check_traceability_with_ticket(self, engine):
        pr_data = {
            "title": "PROJ-123: Add feature",
            "body": "Implements the feature from PROJ-123",
            "head": {"ref": "feature/PROJ-123-new-feature"},
        }
        
        result = await engine.check_traceability(
            pr_data=pr_data,
            diff="""
+def new_feature():
+    pass
""",
            files_changed=[
                {"filename": "src/feature.py", "status": "added", "additions": 10},
            ],
        )
        
        # Should at least process and return a result
        assert result.status in [
            TraceabilityStatus.ALIGNED,
            TraceabilityStatus.PARTIAL,
            TraceabilityStatus.MISALIGNED,
            TraceabilityStatus.ERROR,
        ]


class TestTraceabilityResult:
    """Tests for TraceabilityResult."""
    
    def test_to_dict(self):
        result = TraceabilityResult(
            status=TraceabilityStatus.ALIGNED,
            issue=IssueDetails(
                id="1",
                provider=IssueProvider.JIRA,
                key="PROJ-1",
                title="Test",
                description="Test desc",
                issue_type="feature",
                status="Done",
            ),
            extracted_intent=ExtractedIntent(
                primary_intent="Test intent",
                expected_changes=["Change 1"],
                affected_areas=["src/"],
                change_scope=ChangeScope.FEATURE,
                keywords=["test"],
                acceptance_criteria=["Criterion 1"],
                constraints=[],
                confidence=0.9,
            ),
            code_summary=CodeChangeSummary(
                files_modified=["test.py"],
                files_added=[],
                files_deleted=[],
                functions_modified=["test"],
                classes_modified=[],
                detected_scope=ChangeScope.FEATURE,
                change_keywords=["test"],
                lines_added=10,
                lines_deleted=0,
            ),
            alignment_score=0.85,
            findings=[],
            recommendations=["Good job!"],
        )
        
        d = result.to_dict()
        
        assert d["status"] == "aligned"
        assert d["alignment_score"] == 0.85
        assert d["issue"]["key"] == "PROJ-1"


class TestTraceabilityFinding:
    """Tests for TraceabilityFinding."""
    
    def test_to_dict(self):
        finding = TraceabilityFinding(
            type="scope_mismatch",
            severity="warning",
            description="Scope doesn't match",
            expected="feature",
            actual="bug_fix",
        )
        
        d = finding.to_dict()
        
        assert d["type"] == "scope_mismatch"
        assert d["severity"] == "warning"


class TestExtractedIntent:
    """Tests for ExtractedIntent."""
    
    def test_to_dict(self):
        intent = ExtractedIntent(
            primary_intent="Add feature",
            expected_changes=["Change 1", "Change 2"],
            affected_areas=["src/", "tests/"],
            change_scope=ChangeScope.FEATURE,
            keywords=["add", "feature"],
            acceptance_criteria=["Works correctly"],
            constraints=["Don't break existing"],
            confidence=0.75,
        )
        
        d = intent.to_dict()
        
        assert d["primary_intent"] == "Add feature"
        assert d["change_scope"] == "feature"
        assert d["confidence"] == 0.75


class TestCreateTraceabilityEngine:
    """Tests for create_traceability_engine function."""
    
    def test_create_default(self):
        engine = create_traceability_engine()
        
        assert engine is not None
        assert engine.intent_extractor is not None
    
    def test_create_with_configs(self):
        engine = create_traceability_engine(provider_configs={
            "jira": {"url": "https://jira.example.com", "token": "xxx"},
            "github": {"token": "xxx"},
        })
        
        assert len(engine._providers) >= 1
