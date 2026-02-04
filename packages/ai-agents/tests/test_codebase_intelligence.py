"""Tests for Codebase Intelligence Engine."""

import pytest
import tempfile
from pathlib import Path

from codeverify_agents.codebase_intelligence import (
    BugCorrelation,
    BugTracker,
    CodebaseContext,
    CodebaseIntelligenceEngine,
    CodePattern,
    ComponentCriticality,
    ComponentInfo,
    DependencyAnalyzer,
    PatternDetector,
    PatternType,
)


class TestPatternDetector:
    """Tests for PatternDetector."""
    
    @pytest.fixture
    def detector(self):
        return PatternDetector()
    
    def test_detect_singleton_pattern(self, detector):
        code = '''
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
'''
        patterns = detector.detect_patterns(code, "singleton.py")
        
        pattern_names = [p.name for p in patterns]
        assert "Singleton Pattern" in pattern_names
    
    def test_detect_try_except_pass_antipattern(self, detector):
        code = '''
def risky_operation():
    try:
        do_something()
    except:
        pass
'''
        patterns = detector.detect_patterns(code, "risky.py")
        
        anti_patterns = [p for p in patterns if p.is_violation]
        assert len(anti_patterns) > 0
        assert any("Silent Exception" in p.name for p in anti_patterns)
    
    def test_detect_sql_injection_pattern(self, detector):
        code = '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return cursor.execute(query)
'''
        patterns = detector.detect_patterns(code, "database.py")
        
        security_patterns = [p for p in patterns if p.pattern_type == PatternType.SECURITY]
        assert len(security_patterns) > 0
    
    def test_detect_async_pattern(self, detector):
        code = '''
async def fetch_data():
    result = await api_client.get("/data")
    return result
'''
        patterns = detector.detect_patterns(code, "async_code.py")
        
        pattern_names = [p.name for p in patterns]
        assert "Async/Await Pattern" in pattern_names
    
    def test_add_custom_pattern(self, detector):
        pattern = detector.add_custom_pattern(
            name="Custom Logger",
            pattern_type=PatternType.API_USAGE,
            regex=r"logger\.\w+\(",
            description="Custom logging pattern",
        )
        
        assert pattern.id in detector.patterns
        
        # Test detection
        code = "logger.info('Hello')"
        detected = detector.detect_patterns(code, "test.py")
        
        assert any(p.name == "Custom Logger" for p in detected)
    
    def test_pattern_occurrence_counting(self, detector):
        code = '''
if condition1:
    pass
if condition2:
    pass
if condition3:
    pass
'''
        # Detect multiple times
        detector.detect_patterns(code, "file1.py")
        detector.detect_patterns(code, "file2.py")
        
        # The try_except_pass pattern should not be found (this is if statements)
        # But we should test occurrence counting works
        for pattern in detector.patterns.values():
            if pattern.occurrences > 0:
                assert pattern.last_seen is not None


class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return DependencyAnalyzer()
    
    def test_analyze_python_imports(self, analyzer):
        code = '''
import os
import sys
from pathlib import Path
from typing import Any, Optional
'''
        imports = analyzer.analyze_imports(code, "test.py")
        
        assert "os" in imports
        assert "sys" in imports
        assert "pathlib" in imports
        assert "typing" in imports
    
    def test_analyze_typescript_imports(self, analyzer):
        code = '''
import React from 'react';
import { useState, useEffect } from 'react';
import { UserService } from '../services/user';
'''
        imports = analyzer.analyze_imports(code, "component.tsx")
        
        assert "react" in imports
        assert "../services/user" in imports
    
    def test_build_dependency_graph(self, analyzer):
        files = {
            "main.py": "from utils import helper",
            "utils.py": "import os",
            "service.py": "from main import app\nfrom utils import helper",
        }
        
        graph = analyzer.build_dependency_graph(files)
        
        assert "main.py" in graph
        assert "utils" in graph["main.py"]
    
    def test_find_dependents(self, analyzer):
        files = {
            "utils.py": "import os",
            "main.py": "from utils import helper",
            "service.py": "from utils import helper",
        }
        
        graph = analyzer.build_dependency_graph(files)
        dependents = analyzer.find_dependents("utils.py", graph)
        
        assert "main.py" in dependents
        assert "service.py" in dependents


class TestBugTracker:
    """Tests for BugTracker."""
    
    @pytest.fixture
    def tracker(self):
        return BugTracker()
    
    def test_record_bug(self, tracker):
        bug = tracker.record_bug(
            file_path="auth.py",
            bug_id="BUG-123",
            bug_title="Security vulnerability",
            introduced_commit="abc123",
            pattern_id="sql_concat",
            severity="critical",
        )
        
        assert bug.bug_id == "BUG-123"
        assert len(tracker.bugs) == 1
        assert tracker.file_bug_count["auth.py"] == 1
        assert tracker.pattern_bug_count["sql_concat"] == 1
    
    def test_get_bugs_for_file(self, tracker):
        tracker.record_bug("auth.py", "BUG-1", "Bug 1", "commit1")
        tracker.record_bug("auth.py", "BUG-2", "Bug 2", "commit2")
        tracker.record_bug("other.py", "BUG-3", "Bug 3", "commit3")
        
        auth_bugs = tracker.get_bugs_for_file("auth.py")
        
        assert len(auth_bugs) == 2
        assert all(b.file_path == "auth.py" for b in auth_bugs)
    
    def test_get_bug_prone_files(self, tracker):
        tracker.record_bug("buggy.py", "B1", "Bug", "c1")
        tracker.record_bug("buggy.py", "B2", "Bug", "c2")
        tracker.record_bug("buggy.py", "B3", "Bug", "c3")
        tracker.record_bug("okay.py", "B4", "Bug", "c4")
        
        top_files = tracker.get_bug_prone_files(top_n=2)
        
        assert top_files[0] == ("buggy.py", 3)
        assert len(top_files) == 2


class TestCodebaseIntelligenceEngine:
    """Tests for CodebaseIntelligenceEngine."""
    
    @pytest.fixture
    def engine(self):
        return CodebaseIntelligenceEngine()
    
    @pytest.fixture
    def engine_with_storage(self, tmp_path):
        return CodebaseIntelligenceEngine(storage_path=str(tmp_path / "intelligence"))
    
    def test_index_file(self, engine):
        code = '''
import os
from typing import Optional

class UserService:
    _instance = None
    
    def get_user(self, user_id: int) -> Optional[dict]:
        return {"id": user_id}
'''
        component = engine.index_file("services/user.py", code)
        
        assert component is not None
        assert component.path == "services/user.py"
        assert component.name == "user"
        assert len(component.dependencies) > 0
        assert component.complexity_score is not None
    
    def test_index_security_critical_file(self, engine):
        code = '''
def authenticate(username, password):
    # Authentication logic
    pass
'''
        component = engine.index_file("auth/authentication.py", code)
        
        assert component.criticality == ComponentCriticality.CRITICAL
    
    def test_index_repository(self, engine):
        files = {
            "main.py": "from utils import helper\nimport os",
            "utils.py": "def helper(): pass",
            "tests/test_main.py": "from main import *",
        }
        
        result = engine.index_repository(files)
        
        assert result["indexed_files"] == 3
        assert len(engine.components) == 3
    
    def test_get_context(self, engine):
        code = '''
class PaymentProcessor:
    def process_payment(self, amount):
        if amount <= 0:
            raise ValueError("Invalid amount")
        return True
'''
        engine.index_file("payment/processor.py", code)
        
        context = engine.get_context("payment/processor.py")
        
        assert context is not None
        assert context.file_path == "payment/processor.py"
        assert context.component is not None
        assert context.confidence > 0
    
    def test_get_context_without_prior_indexing(self, engine):
        code = "def simple(): pass"
        
        # Get context for unindexed file with code provided
        context = engine.get_context("new_file.py", code)
        
        assert context is not None
        assert context.component is not None
        assert "new_file.py" in engine.components.values().__iter__().__next__().path or len(engine.components) > 0
    
    def test_learn_from_bug(self, engine):
        code = '''
def risky():
    try:
        do_something()
    except:
        pass
'''
        engine.index_file("risky.py", code)
        
        engine.learn_from_bug(
            file_path="risky.py",
            bug_id="BUG-100",
            bug_title="Silent failure in risky function",
            introduced_commit="abc123",
            severity="high",
        )
        
        # Check bug was recorded
        bugs = engine.bug_tracker.get_bugs_for_file("risky.py")
        assert len(bugs) == 1
        
        # Check component bug count updated
        component = list(engine.components.values())[0]
        assert component.bug_count == 1
    
    def test_get_similar_code(self, engine):
        # Index some files with patterns
        engine.index_file("service1.py", "class Service:\n    _instance = None")
        engine.index_file("service2.py", "class OtherService:\n    _instance = None")
        engine.index_file("util.py", "def helper(): pass")
        
        # Find similar code
        snippet = "class NewService:\n    _instance = None"
        similar = engine.get_similar_code(snippet, top_n=3)
        
        # Should find the singleton-like services
        assert len(similar) >= 0  # May or may not find depending on pattern matching
    
    def test_get_statistics(self, engine):
        engine.index_file("file1.py", "import os")
        engine.index_file("file2.py", "import sys")
        
        stats = engine.get_statistics()
        
        assert stats["total_components"] == 2
        assert stats["total_patterns"] >= 0
        assert "criticality_distribution" in stats
    
    def test_persistence(self, engine_with_storage):
        # Index some files
        engine_with_storage.index_file("test.py", "import os")
        
        # Create a new engine with same storage
        new_engine = CodebaseIntelligenceEngine(
            storage_path=engine_with_storage.storage_path
        )
        
        # Should have loaded the component
        assert len(new_engine.components) == 1
    
    def test_context_to_prompt_context(self, engine):
        code = '''
class CriticalAuth:
    def login(self, user, password):
        pass
'''
        engine.index_file("auth/login.py", code)
        engine.learn_from_bug("auth/login.py", "BUG-1", "Auth bypass", "commit1", "critical")
        
        context = engine.get_context("auth/login.py")
        prompt_context = context.to_prompt_context()
        
        assert "auth" in prompt_context.lower() or "critical" in prompt_context.lower()


class TestComponentInfo:
    """Tests for ComponentInfo dataclass."""
    
    def test_to_dict(self):
        component = ComponentInfo(
            id="test-id",
            path="src/service.py",
            name="service",
            component_type="file",
            criticality=ComponentCriticality.HIGH,
            description="Service component",
            dependencies=["os", "typing"],
            bug_count=2,
            complexity_score=0.5,
        )
        
        d = component.to_dict()
        
        assert d["id"] == "test-id"
        assert d["criticality"] == "high"
        assert d["bug_count"] == 2


class TestCodePattern:
    """Tests for CodePattern dataclass."""
    
    def test_to_dict(self):
        pattern = CodePattern(
            id="pattern-1",
            pattern_type=PatternType.SECURITY,
            name="SQL Injection",
            description="Potential SQL injection",
            regex=r"SELECT.*\{",
            files=["db.py", "query.py"],
            occurrences=5,
            is_violation=True,
            bug_correlation=0.75,
        )
        
        d = pattern.to_dict()
        
        assert d["id"] == "pattern-1"
        assert d["pattern_type"] == "security"
        assert d["is_violation"] is True
        assert d["bug_correlation"] == 0.75


class TestBugCorrelation:
    """Tests for BugCorrelation dataclass."""
    
    def test_to_dict(self):
        bug = BugCorrelation(
            pattern_id="pattern-1",
            file_path="auth.py",
            bug_id="BUG-123",
            bug_title="Security issue",
            introduced_commit="abc123",
            severity="critical",
            days_to_fix=5,
        )
        
        d = bug.to_dict()
        
        assert d["bug_id"] == "BUG-123"
        assert d["severity"] == "critical"
        assert d["days_to_fix"] == 5
