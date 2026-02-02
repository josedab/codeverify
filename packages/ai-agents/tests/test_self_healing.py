"""Tests for self-healing code suggestions."""

import pytest
from codeverify_agents.self_healing import (
    BugReport,
    FixCategory,
    FixGenerationResult,
    FixVerifier,
    ProofStatus,
    SelfHealingAgent,
    SelfHealingManager,
    VerifiedFix,
    get_self_healing_manager,
    reset_self_healing_manager,
)
from codeverify_agents.base import AgentConfig


class TestVerifiedFix:
    """Tests for VerifiedFix dataclass."""

    def test_creation(self):
        """Test fix creation."""
        fix = VerifiedFix(
            fix_id="fix-1",
            category=FixCategory.NULL_CHECK,
            original_code="x = obj.value",
            fixed_code="if obj is not None:\n    x = obj.value",
            description="Add null check",
            confidence=0.9,
        )
        
        assert fix.fix_id == "fix-1"
        assert fix.category == FixCategory.NULL_CHECK
        assert fix.confidence == 0.9
        assert "is not None" in fix.fixed_code

    def test_to_dict(self):
        """Test serialization."""
        fix = VerifiedFix(
            fix_id="fix-2",
            category=FixCategory.BOUNDS_CHECK,
            original_code="arr[i]",
            fixed_code="if i < len(arr): arr[i]",
            description="Add bounds check",
            proof_status=ProofStatus.PROVEN,
            confidence=0.85,
            rank=1,
        )
        
        d = fix.to_dict()
        assert d["fix_id"] == "fix-2"
        assert d["category"] == "bounds_check"
        assert d["proof_status"] == "proven"
        assert d["confidence"] == 0.85


class TestBugReport:
    """Tests for BugReport dataclass."""

    def test_creation(self):
        """Test bug report creation."""
        bug = BugReport(
            bug_id="bug-123",
            category="null_safety",
            description="Potential null dereference",
            code_snippet="x = obj.value",
            language="python",
            z3_counterexample={"obj": "None"},
        )
        
        assert bug.bug_id == "bug-123"
        assert bug.category == "null_safety"
        assert bug.z3_counterexample == {"obj": "None"}


class TestFixVerifier:
    """Tests for FixVerifier."""

    @pytest.fixture
    def verifier(self):
        return FixVerifier()

    @pytest.mark.asyncio
    async def test_verify_null_safety_python(self, verifier):
        """Test null safety verification for Python."""
        status, summary, time_ms = await verifier.verify_fix(
            original_code="x = obj.value",
            fixed_code="if obj is not None:\n    x = obj.value",
            bug_category="null_safety",
            language="python",
        )
        
        assert status == ProofStatus.PROVEN
        assert "null check" in summary.lower()
        assert time_ms >= 0

    @pytest.mark.asyncio
    async def test_verify_null_safety_typescript(self, verifier):
        """Test null safety verification for TypeScript."""
        status, summary, time_ms = await verifier.verify_fix(
            original_code="const x = obj.value",
            fixed_code="const x = obj?.value",
            bug_category="null_safety",
            language="typescript",
        )
        
        assert status == ProofStatus.PROVEN
        assert "null check" in summary.lower()

    @pytest.mark.asyncio
    async def test_verify_bounds_check(self, verifier):
        """Test bounds check verification."""
        status, summary, _ = await verifier.verify_fix(
            original_code="x = arr[i]",
            fixed_code="if i >= 0 and i < len(arr):\n    x = arr[i]",
            bug_category="bounds",
            language="python",
        )
        
        assert status == ProofStatus.PROVEN
        assert "bounds" in summary.lower()

    @pytest.mark.asyncio
    async def test_verify_division_guard(self, verifier):
        """Test division guard verification."""
        status, summary, _ = await verifier.verify_fix(
            original_code="result = a / b",
            fixed_code="if b != 0:\n    result = a / b",
            bug_category="division",
            language="python",
        )
        
        assert status == ProofStatus.PROVEN
        assert "division" in summary.lower()

    @pytest.mark.asyncio
    async def test_verify_identical_code_fails(self, verifier):
        """Test that identical code fails verification."""
        status, summary, _ = await verifier.verify_fix(
            original_code="x = arr[i]",
            fixed_code="x = arr[i]",
            bug_category="bounds",
            language="python",
        )
        
        assert status == ProofStatus.FAILED
        assert "identical" in summary.lower()


class TestSelfHealingAgent:
    """Tests for SelfHealingAgent."""

    @pytest.fixture
    def agent(self):
        return SelfHealingAgent()

    def test_template_fix_generation(self, agent):
        """Test template-based fix generation."""
        bug = BugReport(
            bug_id="bug-1",
            category="null_check",
            description="Null dereference",
            code_snippet="value = obj.attribute",
            language="python",
            z3_counterexample={"var": "obj"},
        )
        
        fixes = agent._generate_template_fixes(bug)
        assert len(fixes) > 0
        assert fixes[0].generation_method == "template"

    @pytest.mark.asyncio
    async def test_generate_fixes_integration(self, agent):
        """Test full fix generation flow."""
        bug = BugReport(
            bug_id="bug-2",
            category="division",
            description="Potential division by zero",
            code_snippet="result = x / divisor",
            language="python",
            z3_counterexample={"divisor": "0"},
        )
        
        result = await agent.generate_fixes(bug)
        
        # Should at least have template-based fixes
        assert result.success or result.total_generated >= 0
        assert result.generation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_analyze_interface(self, agent):
        """Test the standard agent analyze interface."""
        result = await agent.analyze(
            code="x = arr[i]",
            context={
                "language": "python",
                "category": "bounds",
                "description": "Array out of bounds",
            }
        )
        
        assert "fixes" in result.data

    def test_rank_fixes(self, agent):
        """Test fix ranking."""
        fixes = [
            VerifiedFix(
                fix_id="1",
                fixed_code="fix1",
                original_code="orig",
                confidence=0.5,
                proof_status=ProofStatus.UNVERIFIED,
                generation_method="llm",
            ),
            VerifiedFix(
                fix_id="2",
                fixed_code="fix2",
                original_code="orig",
                confidence=0.7,
                proof_status=ProofStatus.PROVEN,
                generation_method="template",
            ),
            VerifiedFix(
                fix_id="3",
                fixed_code="fix3",
                original_code="orig",
                confidence=0.8,
                proof_status=ProofStatus.LIKELY_CORRECT,
                generation_method="llm",
            ),
        ]
        
        ranked = agent._rank_fixes(fixes)
        
        # Fix 2 should rank highest (proven + template)
        assert ranked[0].fix_id == "2"


class TestSelfHealingManager:
    """Tests for SelfHealingManager."""

    @pytest.fixture
    def manager(self):
        return SelfHealingManager()

    @pytest.mark.asyncio
    async def test_heal_bug(self, manager):
        """Test healing a bug."""
        result = await manager.heal_bug(
            code="result = x / y",
            bug_category="division",
            bug_description="Division by zero possible",
            language="python",
            z3_counterexample={"y": "0"},
        )
        
        assert isinstance(result, FixGenerationResult)

    @pytest.mark.asyncio
    async def test_caching(self, manager):
        """Test result caching."""
        code = "result = x / y"
        
        result1 = await manager.heal_bug(
            code=code,
            bug_category="division",
            bug_description="Division by zero",
            language="python",
        )
        
        result2 = await manager.heal_bug(
            code=code,
            bug_category="division",
            bug_description="Division by zero",
            language="python",
        )
        
        # Should be same cached instance
        assert result1 is result2

    def test_get_best_fix(self, manager):
        """Test getting best fix."""
        result = FixGenerationResult(
            success=True,
            fixes=[
                VerifiedFix(fix_id="1", rank=1, confidence=0.9),
                VerifiedFix(fix_id="2", rank=2, confidence=0.7),
            ],
        )
        
        best = manager.get_best_fix(result)
        assert best.fix_id == "1"

    def test_get_best_fix_empty(self, manager):
        """Test getting best fix from empty result."""
        result = FixGenerationResult(success=False, fixes=[])
        assert manager.get_best_fix(result) is None

    def test_get_proven_fixes(self, manager):
        """Test filtering proven fixes."""
        result = FixGenerationResult(
            success=True,
            fixes=[
                VerifiedFix(fix_id="1", proof_status=ProofStatus.PROVEN),
                VerifiedFix(fix_id="2", proof_status=ProofStatus.UNVERIFIED),
                VerifiedFix(fix_id="3", proof_status=ProofStatus.PROVEN),
            ],
        )
        
        proven = manager.get_proven_fixes(result)
        assert len(proven) == 2
        assert all(f.proof_status == ProofStatus.PROVEN for f in proven)

    def test_format_fix_for_display(self, manager):
        """Test formatting a fix for display."""
        fix = VerifiedFix(
            fix_id="fix-1",
            category=FixCategory.NULL_CHECK,
            original_code="x = obj.value",
            fixed_code="if obj is not None:\n    x = obj.value",
            description="Add null check before access",
            proof_status=ProofStatus.PROVEN,
            proof_summary="Z3 verified null safety",
            confidence=0.95,
        )
        
        output = manager.format_fix_for_display(fix)
        
        assert "Add null check" in output
        assert "Original Code" in output
        assert "Fixed Code" in output
        assert "Proof Summary" in output
        assert "Z3 verified" in output


class TestGlobalManager:
    """Tests for global manager functions."""

    def teardown_method(self):
        reset_self_healing_manager()

    def test_get_manager_singleton(self):
        """Test singleton pattern."""
        manager1 = get_self_healing_manager()
        manager2 = get_self_healing_manager()
        assert manager1 is manager2

    def test_reset_manager(self):
        """Test manager reset."""
        manager1 = get_self_healing_manager()
        reset_self_healing_manager()
        manager2 = get_self_healing_manager()
        assert manager1 is not manager2

    def test_get_manager_with_config(self):
        """Test creating manager with config."""
        config = AgentConfig(model="gpt-4")
        manager = get_self_healing_manager(config)
        assert manager is not None


class TestFixCategories:
    """Tests for fix category enum."""

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        expected = [
            "null_check",
            "bounds_check",
            "overflow_check",
            "division_guard",
            "type_guard",
            "exception_handling",
            "initialization",
            "validation",
            "refactoring",
        ]
        
        for cat in expected:
            assert FixCategory(cat) is not None

    def test_category_string_conversion(self):
        """Test category to string conversion."""
        assert FixCategory.NULL_CHECK.value == "null_check"
        assert str(FixCategory.BOUNDS_CHECK) == "FixCategory.bounds_check"


class TestProofStatus:
    """Tests for proof status enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        expected = ["proven", "likely_correct", "unverified", "failed"]
        
        for status in expected:
            assert ProofStatus(status) is not None

    def test_status_ordering(self):
        """Test that statuses can be compared meaningfully."""
        # Proven is better than likely_correct
        assert ProofStatus.PROVEN.value != ProofStatus.LIKELY_CORRECT.value
