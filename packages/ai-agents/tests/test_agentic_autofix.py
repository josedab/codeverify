"""Tests for Agentic Auto-Fix PRs."""

import pytest

from codeverify_agents.agentic_autofix import (
    AgenticAutoFix,
    Finding,
    FixCategory,
    FixGenerator,
    FixStatus,
    FixTemplates,
    FixVerifier,
    TestGenerator,
)


class TestFixTemplates:
    """Tests for fix templates."""
    
    def test_get_null_safety_template(self):
        template = FixTemplates.get_template(
            FixCategory.NULL_SAFETY, "python", "guard_check"
        )
        assert template is not None
        assert "{var}" in template
    
    def test_get_bounds_check_template(self):
        template = FixTemplates.get_template(
            FixCategory.BOUNDS_CHECK, "python", "bounds_guard"
        )
        assert template is not None
        assert "len(" in template
    
    def test_nonexistent_template(self):
        template = FixTemplates.get_template(
            FixCategory.STYLE, "python", "nonexistent"
        )
        assert template is None


class TestFixGenerator:
    """Tests for fix generation."""
    
    @pytest.fixture
    def generator(self):
        return FixGenerator()
    
    @pytest.fixture
    def null_safety_finding(self):
        return Finding(
            id="test-finding-1",
            title="Potential null dereference",
            description="Variable 'data' may be None",
            category=FixCategory.NULL_SAFETY,
            severity="high",
            file_path="test.py",
            line_start=5,
            line_end=5,
            code_snippet="result = data.process()",
        )
    
    @pytest.fixture
    def bounds_finding(self):
        return Finding(
            id="test-finding-2",
            title="Potential index out of bounds",
            description="Array access may exceed bounds",
            category=FixCategory.BOUNDS_CHECK,
            severity="high",
            file_path="test.py",
            line_start=3,
            line_end=3,
            code_snippet="value = items[i]",
        )
    
    @pytest.mark.asyncio
    async def test_generate_null_safety_fix(self, generator, null_safety_finding):
        code = """def process_data(data):
    # Process the data
    result = data.process()
    return result
"""
        # Adjust line numbers for our test code
        null_safety_finding.line_start = 3
        
        fixes = await generator.generate_fix(null_safety_finding, code, "python")
        
        assert len(fixes) > 0
        best_fix = fixes[0]
        assert best_fix.status == FixStatus.PENDING
        assert best_fix.original_code == code
        assert best_fix.fixed_code != code
        assert best_fix.diff != ""
    
    @pytest.mark.asyncio
    async def test_generate_bounds_fix(self, generator, bounds_finding):
        code = """def get_item(items, i):
    value = items[i]
    return value
"""
        # Adjust line number
        bounds_finding.line_start = 2
        
        fixes = await generator.generate_fix(bounds_finding, code, "python")
        
        assert len(fixes) > 0
        best_fix = fixes[0]
        assert best_fix.fixed_code != code
    
    def test_generate_diff(self, generator):
        original = "x = 1\ny = 2"
        fixed = "x = 1\ny = 3"
        
        diff = generator._generate_diff(original, fixed)
        
        assert "-y = 2" in diff or "- y = 2" in diff
        assert "+y = 3" in diff or "+ y = 3" in diff
    
    @pytest.mark.asyncio
    async def test_rank_and_deduplicate(self, generator, null_safety_finding):
        from codeverify_agents.agentic_autofix import GeneratedFix
        
        fix1 = GeneratedFix(
            id="1",
            finding_id="test",
            status=FixStatus.PENDING,
            original_code="x",
            fixed_code="y",
            diff="diff1",
            explanation="fix1",
            confidence=0.9,
        )
        fix2 = GeneratedFix(
            id="2",
            finding_id="test",
            status=FixStatus.PENDING,
            original_code="x",
            fixed_code="y",
            diff="diff1",  # Same diff = duplicate
            explanation="fix2",
            confidence=0.7,
        )
        fix3 = GeneratedFix(
            id="3",
            finding_id="test",
            status=FixStatus.PENDING,
            original_code="x",
            fixed_code="z",
            diff="diff2",
            explanation="fix3",
            confidence=0.8,
        )
        
        result = generator._rank_and_deduplicate([fix1, fix2, fix3])
        
        # Should deduplicate and sort by confidence
        assert len(result) == 2  # fix1 and fix3 (fix2 is duplicate)
        assert result[0].confidence == 0.9  # Highest first


class TestFixVerifier:
    """Tests for fix verification."""
    
    @pytest.fixture
    def verifier(self):
        return FixVerifier(timeout_ms=5000)
    
    @pytest.fixture
    def null_fix(self):
        from codeverify_agents.agentic_autofix import GeneratedFix
        return GeneratedFix(
            id="test-fix-1",
            finding_id="test-finding-1",
            status=FixStatus.PENDING,
            original_code="result = data.process()",
            fixed_code="if data is not None:\n    result = data.process()",
            diff="",
            explanation="Added null check",
            confidence=0.7,
        )
    
    @pytest.fixture
    def null_finding(self):
        return Finding(
            id="test-finding-1",
            title="Null dereference",
            description="",
            category=FixCategory.NULL_SAFETY,
            severity="high",
            file_path="test.py",
            line_start=1,
            line_end=1,
            code_snippet="",
        )
    
    def test_verify_null_safety_fix(self, verifier, null_fix, null_finding):
        result = verifier.verify(null_fix, null_finding)
        
        assert result.status == FixStatus.VERIFIED
        assert result.verification_result is not None
        assert result.verification_result["verified"] is True
    
    def test_verify_invalid_fix(self, verifier, null_finding):
        from codeverify_agents.agentic_autofix import GeneratedFix
        
        bad_fix = GeneratedFix(
            id="bad-fix",
            finding_id="test",
            status=FixStatus.PENDING,
            original_code="result = data.process()",
            fixed_code="result = data.process()",  # No change = no fix
            diff="",
            explanation="No fix applied",
            confidence=0.5,
        )
        
        result = verifier.verify(bad_fix, null_finding)
        
        assert result.status == FixStatus.VERIFICATION_FAILED
    
    def test_verify_syntax(self, verifier):
        result = verifier._verify_syntax_fix_code("x = 1")
        
        # Should handle code verification
        assert isinstance(result, dict)
    
    def _verify_syntax_fix_code(self, code):
        """Helper to test syntax verification."""
        from codeverify_agents.agentic_autofix import GeneratedFix
        
        fix = GeneratedFix(
            id="test",
            finding_id="test",
            status=FixStatus.PENDING,
            original_code="",
            fixed_code=code,
            diff="",
            explanation="",
            confidence=0.5,
        )
        return self.verifier._verify_syntax(fix)


class TestTestGenerator:
    """Tests for test generation."""
    
    @pytest.fixture
    def generator(self):
        return TestGenerator()
    
    @pytest.fixture
    def sample_fix(self):
        from codeverify_agents.agentic_autofix import GeneratedFix
        return GeneratedFix(
            id="test-fix",
            finding_id="test-finding",
            status=FixStatus.VERIFIED,
            original_code="def foo(x): return x.bar()",
            fixed_code="def foo(x):\n    if x is None:\n        return None\n    return x.bar()",
            diff="",
            explanation="Added null check",
            confidence=0.9,
        )
    
    @pytest.fixture
    def null_finding(self):
        return Finding(
            id="test-finding",
            title="Null dereference in foo",
            description="x may be None",
            category=FixCategory.NULL_SAFETY,
            severity="high",
            file_path="test.py",
            line_start=1,
            line_end=1,
            code_snippet="",
        )
    
    @pytest.mark.asyncio
    async def test_generate_null_safety_tests(self, generator, sample_fix, null_finding):
        tests = await generator.generate_tests(sample_fix, null_finding)
        
        assert len(tests) >= 1
        assert any("None" in t for t in tests)
        assert any("def test_" in t for t in tests)
    
    @pytest.mark.asyncio
    async def test_generate_bounds_tests(self, generator, sample_fix):
        bounds_finding = Finding(
            id="bounds-finding",
            title="Index out of bounds",
            description="",
            category=FixCategory.BOUNDS_CHECK,
            severity="high",
            file_path="test.py",
            line_start=1,
            line_end=1,
            code_snippet="",
        )
        
        tests = await generator.generate_tests(sample_fix, bounds_finding)
        
        assert len(tests) >= 1
        assert any("negative" in t.lower() or "overflow" in t.lower() for t in tests)


class TestAgenticAutoFix:
    """Tests for the main agent."""
    
    @pytest.fixture
    def agent(self):
        return AgenticAutoFix()
    
    @pytest.mark.asyncio
    async def test_analyze(self, agent):
        code = """def process(data):
    result = data.process()
    return result
"""
        context = {
            "findings": [
                {
                    "id": "finding-1",
                    "title": "Null dereference",
                    "description": "data may be None",
                    "category": "null_safety",
                    "severity": "high",
                    "file_path": "test.py",
                    "line_start": 2,
                    "line_end": 2,
                    "code_snippet": "result = data.process()",
                }
            ],
            "language": "python",
        }
        
        result = await agent.analyze(code, context)
        
        assert result.success
        assert "fixes" in result.data
    
    @pytest.mark.asyncio
    async def test_auto_fix_with_dict_findings(self, agent):
        code = """def get_value(arr, idx):
    return arr[idx]
"""
        findings = [
            {
                "id": "f1",
                "title": "Out of bounds",
                "description": "idx may exceed array length",
                "category": "bounds_check",
                "severity": "high",
                "file_path": "test.py",
                "line_start": 2,
                "line_end": 2,
                "code_snippet": "return arr[idx]",
            }
        ]
        
        result = await agent.auto_fix(code, findings, "python")
        
        assert result.total_findings == 1
    
    @pytest.mark.asyncio
    async def test_auto_fix_with_finding_objects(self, agent):
        code = """def fetch(data):
    return data.get_value()
"""
        findings = [
            Finding(
                id="f1",
                title="Null dereference",
                description="data may be None",
                category=FixCategory.NULL_SAFETY,
                severity="high",
                file_path="test.py",
                line_start=2,
                line_end=2,
                code_snippet="return data.get_value()",
            )
        ]
        
        result = await agent.auto_fix(code, findings, "python")
        
        assert result.total_findings == 1
    
    @pytest.mark.asyncio
    async def test_fix_single_finding(self, agent):
        code = """def process(x):
    return x.value
"""
        finding = Finding(
            id="single-finding",
            title="Null dereference",
            description="x may be None",
            category=FixCategory.NULL_SAFETY,
            severity="high",
            file_path="test.py",
            line_start=2,
            line_end=2,
            code_snippet="return x.value",
        )
        
        fix = await agent.fix_single_finding(code, finding, "python")
        
        # May or may not generate a fix depending on pattern matching
        # The important thing is it doesn't raise an exception


class TestGeneratedFix:
    """Tests for GeneratedFix dataclass."""
    
    def test_to_dict(self):
        from codeverify_agents.agentic_autofix import GeneratedFix
        
        fix = GeneratedFix(
            id="test-id",
            finding_id="finding-id",
            status=FixStatus.VERIFIED,
            original_code="x = 1",
            fixed_code="x = 2",
            diff="-x = 1\n+x = 2",
            explanation="Changed value",
            confidence=0.95,
        )
        
        d = fix.to_dict()
        
        assert d["id"] == "test-id"
        assert d["status"] == "verified"
        assert d["confidence"] == 0.95
        assert "created_at" in d


class TestAutoFixResult:
    """Tests for AutoFixResult dataclass."""
    
    def test_to_dict(self):
        from codeverify_agents.agentic_autofix import AutoFixResult, GeneratedFix
        
        fix = GeneratedFix(
            id="fix-1",
            finding_id="finding-1",
            status=FixStatus.READY_FOR_PR,
            original_code="",
            fixed_code="",
            diff="",
            explanation="",
            confidence=0.9,
        )
        
        result = AutoFixResult(
            success=True,
            fixes=[fix],
            total_findings=1,
            fixes_verified=1,
            fixes_ready=1,
        )
        
        d = result.to_dict()
        
        assert d["success"] is True
        assert len(d["fixes"]) == 1
        assert d["fixes_ready"] == 1
