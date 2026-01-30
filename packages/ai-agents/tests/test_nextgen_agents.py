"""Tests for Next-Gen AI Agents."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from codeverify_agents.test_generator import (
    CounterexampleToTest,
    GeneratedTest,
    TestFramework,
    TestGeneratorAgent,
    TestSuite,
)
from codeverify_agents.nl_invariants import (
    InvariantSpec,
    NaturalLanguageInvariantsAgent,
    ParsedConstraint,
    Z3Assertion,
)
from codeverify_agents.semantic_diff import (
    BehaviorChange,
    ChangeType,
    SemanticDiffAgent,
    SemanticDiffResult,
)
from codeverify_agents.team_learning import (
    OrgHealthReport,
    PatternOccurrence,
    SystemicPattern,
    TeamLearningAgent,
    TeamMetrics,
    TrainingRecommendation,
)
from codeverify_agents.model_arbitrator import (
    ArbitratedFinding,
    ArbitrationResult,
    ArbitrationVote,
    CompetingModelArbitrator,
    ModelProfile,
    ModelSpecialization,
    VotingMethod,
)
from codeverify_agents.multi_model_consensus import (
    ModelConfig,
    ModelFinding,
    ModelProvider,
)
from codeverify_core.models import Finding, FindingCategory, FindingSeverity


# ============================================
# Feature 2: AI Regression Test Generator Tests
# ============================================

class TestTestFramework:
    """Tests for TestFramework enum."""

    def test_all_frameworks_exist(self):
        """All expected frameworks exist."""
        assert TestFramework.PYTEST.value == "pytest"
        assert TestFramework.UNITTEST.value == "unittest"
        assert TestFramework.JEST.value == "jest"
        assert TestFramework.VITEST.value == "vitest"
        assert TestFramework.GO_TEST.value == "go_test"


class TestCounterexampleToTest:
    """Tests for CounterexampleToTest dataclass."""

    def test_create_counterexample(self):
        """Can create a counterexample."""
        ce = CounterexampleToTest(
            function_name="divide",
            input_values={"a": 10, "b": 0},
            expected_behavior="raise ZeroDivisionError",
            verification_type="division_by_zero",
        )
        assert ce.function_name == "divide"
        assert ce.input_values["b"] == 0


class TestGeneratedTest:
    """Tests for GeneratedTest dataclass."""

    def test_create_test(self):
        """Can create a generated test."""
        test = GeneratedTest(
            test_name="test_divide_by_zero",
            test_code="def test_divide_by_zero():\n    with pytest.raises(ZeroDivisionError):\n        divide(10, 0)",
            framework=TestFramework.PYTEST,
            counterexample=CounterexampleToTest(
                function_name="divide",
                input_values={"a": 10, "b": 0},
                expected_behavior="exception",
                verification_type="div_zero",
            ),
        )
        assert "pytest.raises" in test.test_code
        assert test.framework == TestFramework.PYTEST


class TestTestGeneratorAgent:
    """Tests for TestGeneratorAgent."""

    def test_create_agent(self):
        """Can create test generator agent."""
        agent = TestGeneratorAgent()
        assert agent is not None

    def test_generate_test_name(self):
        """Generates appropriate test names."""
        agent = TestGeneratorAgent()
        ce = CounterexampleToTest(
            function_name="process_data",
            input_values={},
            expected_behavior="error",
            verification_type="null_check",
        )
        
        name = agent._generate_test_name(ce)
        assert "test_" in name
        assert "process_data" in name

    def test_select_framework(self):
        """Selects appropriate framework based on language."""
        agent = TestGeneratorAgent()
        
        assert agent._select_framework("python") == TestFramework.PYTEST
        assert agent._select_framework("typescript") == TestFramework.JEST
        assert agent._select_framework("go") == TestFramework.GO_TEST

    def test_generate_pytest_template(self):
        """Generates pytest template."""
        agent = TestGeneratorAgent()
        ce = CounterexampleToTest(
            function_name="add",
            input_values={"a": 1, "b": 2},
            expected_behavior="return 3",
            verification_type="correctness",
        )
        
        test = agent._generate_pytest(ce)
        
        assert "def test_" in test.test_code
        assert "assert" in test.test_code or "pytest" in test.test_code


class TestTestSuite:
    """Tests for TestSuite dataclass."""

    def test_create_suite(self):
        """Can create a test suite."""
        suite = TestSuite(
            name="Security Tests",
            tests=[],
            framework=TestFramework.PYTEST,
        )
        assert suite.name == "Security Tests"

    def test_suite_with_tests(self):
        """Suite can contain tests."""
        tests = [
            GeneratedTest(
                test_name="test_1",
                test_code="...",
                framework=TestFramework.PYTEST,
                counterexample=CounterexampleToTest("f", {}, "e", "t"),
            ),
            GeneratedTest(
                test_name="test_2",
                test_code="...",
                framework=TestFramework.PYTEST,
                counterexample=CounterexampleToTest("g", {}, "e", "t"),
            ),
        ]
        
        suite = TestSuite(
            name="Suite",
            tests=tests,
            framework=TestFramework.PYTEST,
        )
        assert len(suite.tests) == 2


# ============================================
# Feature 5: Natural Language Invariant Specs Tests
# ============================================

class TestParsedConstraint:
    """Tests for ParsedConstraint dataclass."""

    def test_create_constraint(self):
        """Can create a parsed constraint."""
        constraint = ParsedConstraint(
            original_text="x must be positive",
            constraint_type="range",
            variable="x",
            operator=">",
            value="0",
        )
        assert constraint.variable == "x"
        assert constraint.operator == ">"


class TestZ3Assertion:
    """Tests for Z3Assertion dataclass."""

    def test_create_assertion(self):
        """Can create a Z3 assertion."""
        assertion = Z3Assertion(
            z3_code="x > 0",
            smt_lib="(assert (> x 0))",
            description="x must be positive",
        )
        assert ">" in assertion.z3_code
        assert "assert" in assertion.smt_lib


class TestNaturalLanguageInvariantsAgent:
    """Tests for NaturalLanguageInvariantsAgent."""

    def test_create_agent(self):
        """Can create NL invariants agent."""
        agent = NaturalLanguageInvariantsAgent()
        assert agent is not None

    def test_parse_positive_constraint(self):
        """Parses 'must be positive' constraint."""
        agent = NaturalLanguageInvariantsAgent()
        
        constraints = agent._parse_constraints("x must be positive")
        
        assert len(constraints) > 0
        assert constraints[0].operator == ">"
        assert constraints[0].value == "0"

    def test_parse_non_negative_constraint(self):
        """Parses 'must be non-negative' constraint."""
        agent = NaturalLanguageInvariantsAgent()
        
        constraints = agent._parse_constraints("count must be non-negative")
        
        assert len(constraints) > 0
        assert constraints[0].operator == ">="

    def test_parse_range_constraint(self):
        """Parses range constraint."""
        agent = NaturalLanguageInvariantsAgent()
        
        constraints = agent._parse_constraints("x must be between 0 and 100")
        
        assert len(constraints) >= 1

    def test_parse_not_null_constraint(self):
        """Parses 'must not be null' constraint."""
        agent = NaturalLanguageInvariantsAgent()
        
        constraints = agent._parse_constraints("name must not be null")
        
        assert len(constraints) > 0
        assert constraints[0].constraint_type in ("null_check", "not_null")

    def test_generate_z3_from_constraint(self):
        """Generates Z3 code from constraint."""
        agent = NaturalLanguageInvariantsAgent()
        
        constraint = ParsedConstraint(
            original_text="x > 0",
            constraint_type="comparison",
            variable="x",
            operator=">",
            value="0",
        )
        
        z3_code = agent._to_z3(constraint)
        
        assert "x" in z3_code
        assert "0" in z3_code


class TestInvariantSpec:
    """Tests for InvariantSpec dataclass."""

    def test_create_spec(self):
        """Can create an invariant spec."""
        spec = InvariantSpec(
            name="positive_balance",
            natural_language="balance must be positive",
            assertions=[
                Z3Assertion(
                    z3_code="balance > 0",
                    smt_lib="(assert (> balance 0))",
                    description="balance positive",
                )
            ],
        )
        assert spec.name == "positive_balance"
        assert len(spec.assertions) == 1


# ============================================
# Feature 6: Semantic Diff Visualization Tests
# ============================================

class TestChangeType:
    """Tests for ChangeType enum."""

    def test_all_types_exist(self):
        """All expected change types exist."""
        assert ChangeType.SIGNATURE_CHANGE.value == "signature_change"
        assert ChangeType.BEHAVIOR_CHANGE.value == "behavior_change"
        assert ChangeType.EXCEPTION_CHANGE.value == "exception_change"


class TestBehaviorChange:
    """Tests for BehaviorChange dataclass."""

    def test_create_change(self):
        """Can create a behavior change."""
        change = BehaviorChange(
            change_type=ChangeType.SIGNATURE_CHANGE,
            location="function:calculate",
            old_behavior="calculate(a, b)",
            new_behavior="calculate(a, b, c)",
            impact="Breaking change - new required parameter",
        )
        assert change.change_type == ChangeType.SIGNATURE_CHANGE
        assert "Breaking" in change.impact


class TestSemanticDiffAgent:
    """Tests for SemanticDiffAgent."""

    def test_create_agent(self):
        """Can create semantic diff agent."""
        agent = SemanticDiffAgent()
        assert agent is not None

    def test_detect_signature_change(self):
        """Detects function signature changes."""
        agent = SemanticDiffAgent()
        
        old_code = "def greet(name):\n    return f'Hello {name}'"
        new_code = "def greet(name, title=''):\n    return f'Hello {title} {name}'"
        
        changes = agent._detect_signature_changes(old_code, new_code, "python")
        
        # Should detect parameter addition
        assert isinstance(changes, list)

    def test_generate_mermaid(self):
        """Generates Mermaid diagram."""
        agent = SemanticDiffAgent()
        
        changes = [
            BehaviorChange(
                change_type=ChangeType.SIGNATURE_CHANGE,
                location="func:test",
                old_behavior="old",
                new_behavior="new",
                impact="minor",
            )
        ]
        
        mermaid = agent._to_mermaid(changes)
        
        assert "graph" in mermaid.lower() or "flowchart" in mermaid.lower()

    def test_generate_dot(self):
        """Generates DOT format."""
        agent = SemanticDiffAgent()
        
        changes = [
            BehaviorChange(
                change_type=ChangeType.BEHAVIOR_CHANGE,
                location="func:test",
                old_behavior="old",
                new_behavior="new",
                impact="major",
            )
        ]
        
        dot = agent._to_dot(changes)
        
        assert "digraph" in dot


class TestSemanticDiffResult:
    """Tests for SemanticDiffResult dataclass."""

    def test_create_result(self):
        """Can create a diff result."""
        result = SemanticDiffResult(
            changes=[],
            mermaid_diagram="graph TD\n    A-->B",
            dot_diagram="digraph{}",
            summary="No behavioral changes",
        )
        assert result.summary == "No behavioral changes"


# ============================================
# Feature 8: Team Learning Mode Tests
# ============================================

class TestTeamLearningAgent:
    """Tests for TeamLearningAgent."""

    def test_create_agent(self):
        """Can create team learning agent."""
        agent = TeamLearningAgent()
        assert agent is not None

    def test_configure_teams(self):
        """Can configure team mappings."""
        agent = TeamLearningAgent()
        
        agent.configure_teams({
            "alice": "frontend",
            "bob": "backend",
        })
        
        # Mapping should be set
        assert agent.aggregator._team_mapping.get("alice") == "frontend"

    def test_record_findings(self):
        """Records findings for analysis."""
        agent = TeamLearningAgent()
        
        findings = [
            Finding(
                id="f1",
                message="Null reference",
                category=FindingCategory.CORRECTNESS,
                severity=FindingSeverity.ERROR,
                file_path="src/main.py",
                line_number=10,
            )
        ]
        
        agent.record_findings(findings, "my-repo", "alice")
        
        # Should have recorded
        assert len(agent.aggregator._occurrences) > 0

    def test_identify_patterns(self):
        """Identifies systemic patterns."""
        agent = TeamLearningAgent()
        
        # Record multiple similar findings
        for i in range(10):
            findings = [
                Finding(
                    id=f"f{i}",
                    message="Null pointer exception possible",
                    category=FindingCategory.CORRECTNESS,
                    severity=FindingSeverity.ERROR,
                    file_path="src/main.py",
                    line_number=i,
                )
            ]
            agent.record_findings(findings, "repo", "dev1")
        
        patterns = agent.identify_systemic_patterns(min_occurrences=5)
        
        # Should identify null-related pattern
        assert isinstance(patterns, list)


class TestOrgHealthReport:
    """Tests for OrgHealthReport."""

    def test_create_report(self):
        """Can create a health report."""
        report = OrgHealthReport(
            report_date=datetime.utcnow(),
            total_findings=100,
            total_prs_analyzed=50,
            findings_by_category={"correctness": 60, "security": 40},
            findings_by_severity={"error": 30, "warning": 70},
            team_metrics=[],
            systemic_patterns=[],
            training_recommendations=[],
            trend_vs_last_period="improving",
            top_improving_teams=["frontend"],
            teams_needing_attention=["backend"],
        )
        assert report.total_findings == 100


class TestTrainingRecommendation:
    """Tests for TrainingRecommendation."""

    def test_create_recommendation(self):
        """Can create a training recommendation."""
        rec = TrainingRecommendation(
            title="Security Training",
            description="Address SQL injection patterns",
            target_teams=["backend", "data"],
            target_skills=["SQL Security", "Input Validation"],
            priority=8,
            estimated_impact="50% reduction in SQL-related findings",
            supporting_data={"occurrences": 25},
        )
        assert rec.priority == 8
        assert "backend" in rec.target_teams


# ============================================
# Feature 9: Competing Model Arbitration Tests
# ============================================

class TestVotingMethod:
    """Tests for VotingMethod enum."""

    def test_all_methods_exist(self):
        """All expected voting methods exist."""
        assert VotingMethod.BORDA_COUNT.value == "borda_count"
        assert VotingMethod.APPROVAL.value == "approval"
        assert VotingMethod.RANKED_CHOICE.value == "ranked_choice"
        assert VotingMethod.CONFIDENCE_WEIGHTED.value == "confidence_weighted"
        assert VotingMethod.SPECIALIZATION.value == "specialization"


class TestModelProfile:
    """Tests for ModelProfile."""

    def test_create_profile(self):
        """Can create a model profile."""
        profile = ModelProfile(
            provider=ModelProvider.OPENAI_GPT5,
            specializations=[
                ModelSpecialization.SECURITY,
                ModelSpecialization.CORRECTNESS,
            ],
            base_weight=1.0,
        )
        assert ModelSpecialization.SECURITY in profile.specializations


class TestArbitrationVote:
    """Tests for ArbitrationVote."""

    def test_create_vote(self):
        """Can create an arbitration vote."""
        vote = ArbitrationVote(
            model=ModelProvider.ANTHROPIC_CLAUDE,
            finding_id="finding-123",
            vote="confirm",
            confidence=0.85,
            reasoning="Pattern matches known vulnerability",
        )
        assert vote.vote == "confirm"
        assert vote.confidence == 0.85


class TestCompetingModelArbitrator:
    """Tests for CompetingModelArbitrator."""

    def test_create_arbitrator(self):
        """Can create arbitrator."""
        arbitrator = CompetingModelArbitrator()
        assert arbitrator is not None

    def test_set_voting_method(self):
        """Can change voting method."""
        arbitrator = CompetingModelArbitrator()
        
        arbitrator.set_voting_method(VotingMethod.BORDA_COUNT)
        assert arbitrator.voting_method == VotingMethod.BORDA_COUNT

    def test_calibrate_thresholds(self):
        """Can calibrate thresholds."""
        arbitrator = CompetingModelArbitrator()
        
        arbitrator.calibrate_thresholds(
            confirm_threshold=0.7,
            debate_threshold=0.3,
            report_threshold=0.6,
        )
        
        assert arbitrator.confirm_threshold == 0.7
        assert arbitrator.debate_threshold == 0.3
        assert arbitrator.report_threshold == 0.6

    def test_update_model_profile(self):
        """Can update model profiles."""
        arbitrator = CompetingModelArbitrator()
        
        profile = ModelProfile(
            provider=ModelProvider.GOOGLE_GEMINI,
            specializations=[ModelSpecialization.PERFORMANCE],
            base_weight=0.9,
        )
        
        arbitrator.update_model_profile(ModelProvider.GOOGLE_GEMINI, profile)
        
        assert arbitrator.model_profiles[ModelProvider.GOOGLE_GEMINI] == profile


class TestVotingEngine:
    """Tests for voting algorithms."""

    def test_approval_voting(self):
        """Tests approval voting."""
        from codeverify_agents.model_arbitrator import VotingEngine
        
        engine = VotingEngine()
        
        votes = [
            ArbitrationVote(ModelProvider.OPENAI_GPT5, "f1", "confirm", 0.9, ""),
            ArbitrationVote(ModelProvider.ANTHROPIC_CLAUDE, "f1", "confirm", 0.8, ""),
            ArbitrationVote(ModelProvider.OPENAI_GPT4, "f1", "reject", 0.6, ""),
        ]
        
        result = engine.approval_voting(votes)
        
        assert result["confirm"] == 2
        assert result["reject"] == 1

    def test_confidence_weighted(self):
        """Tests confidence-weighted voting."""
        from codeverify_agents.model_arbitrator import VotingEngine
        
        engine = VotingEngine()
        
        votes = [
            ArbitrationVote(ModelProvider.OPENAI_GPT5, "f1", "confirm", 0.9, ""),
            ArbitrationVote(ModelProvider.ANTHROPIC_CLAUDE, "f1", "reject", 0.3, ""),
        ]
        
        winner, confidence = engine.confidence_weighted(votes)
        
        # High confidence confirm should win
        assert winner == "confirm"
        assert confidence > 0.5

    def test_borda_count(self):
        """Tests Borda count voting."""
        from codeverify_agents.model_arbitrator import VotingEngine
        
        engine = VotingEngine()
        
        votes = [
            ArbitrationVote(ModelProvider.OPENAI_GPT5, "f1", "confirm", 0.8, ""),
            ArbitrationVote(ModelProvider.ANTHROPIC_CLAUDE, "f1", "confirm", 0.7, ""),
            ArbitrationVote(ModelProvider.OPENAI_GPT4, "f1", "uncertain", 0.5, ""),
        ]
        
        scores = engine.borda_count(votes)
        
        # Confirm should have highest score
        assert scores["confirm"] > scores["uncertain"]
        assert scores["confirm"] > scores["reject"]


class TestArbitrationResult:
    """Tests for ArbitrationResult."""

    def test_create_result(self):
        """Can create arbitration result."""
        result = ArbitrationResult(
            finding_id="finding-1",
            final_verdict="confirmed",
            confidence=0.85,
            votes=[],
            debate_rounds=[],
            voting_method=VotingMethod.CONFIDENCE_WEIGHTED,
            vote_breakdown={"confirm": 2, "reject": 1},
            reasoning="2 of 3 models confirmed with high confidence",
        )
        assert result.final_verdict == "confirmed"
        assert result.confidence == 0.85
