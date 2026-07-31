"""Tests for Next-Gen AI Agents."""

from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

import pytest

from codeverify_agents import (
    BehaviorChange,
    ChangeType,
    CounterexampleToTest,
    InvariantSpec,
    NaturalLanguageInvariantsAgent,
    ParsedConstraint,
    SemanticDiffAgent,
    SemanticDiffResult,
    Z3Assertion,
)
from codeverify_agents import (
    TestFramework as PublicFramework,
)
from codeverify_agents import (
    TestGeneratorAgent as PublicTestGenerator,
)
from codeverify_agents import (
    TestSuite as PublicTestSuite,
)
from codeverify_agents.base import BaseAgent
from codeverify_agents.model_arbitrator import (
    ArbitrationResult,
    ArbitrationVote,
    CompetingModelArbitrator,
    ModelProfile,
    ModelSpecialization,
    VotingMethod,
)
from codeverify_agents.multi_model_consensus import (
    ModelProvider,
)
from codeverify_agents.nl_invariants import (
    InvariantType,
    NaturalLanguageInvariant,
    NaturalLanguageInvariantAgent,
    ValueConstraint,
    Z3Compiler,
)
from codeverify_agents.semantic_diff import (
    RiskLevel,
    SemanticDiff,
    SemanticNode,
)
from codeverify_agents.team_learning import (
    OrgHealthReport,
    TeamLearningAgent,
    TrainingRecommendation,
    TrendDirection,
)
from codeverify_agents.test_generator import (
    Counterexample,
    GeneratedTest,
    Language,
)
from codeverify_agents.test_generator import (
    TestFramework as Framework,
)
from codeverify_agents.test_generator import (
    TestGenerationResult as GenerationResult,
)
from codeverify_agents.test_generator import (
    TestGeneratorAgent as RegressionTestGenerator,
)
from codeverify_core.models import (
    CodeLocation,
    Finding,
    FindingCategory,
    FindingSeverity,
    VerificationType,
)

# ============================================
# Feature 2: AI Regression Test Generator Tests
# ============================================


class TestTestFramework:
    """Tests for TestFramework enum."""

    def test_all_frameworks_exist(self):
        """All expected frameworks exist."""
        public_frameworks = {member.name: member.value for member in PublicFramework}
        implementation_frameworks = {member.name: member.value for member in Framework}

        assert public_frameworks == implementation_frameworks
        assert public_frameworks["PYTEST"] == "pytest"
        assert public_frameworks["UNITTEST"] == "unittest"
        assert public_frameworks["JEST"] == "jest"
        assert public_frameworks["VITEST"] == "vitest"
        assert public_frameworks["GO_TEST"] == "go_test"


class TestCounterexampleToTest:
    """Tests for the intentional CounterexampleToTest public alias."""

    def test_create_counterexample(self):
        """Alias constructs the current Counterexample data model."""
        values = {
            "variables": {"a": 10, "b": 0},
            "expected_behavior": "raise ZeroDivisionError",
            "actual_behavior": "division attempted with zero",
            "verification_type": "division_by_zero",
        }
        ce = CounterexampleToTest(**values)
        implementation = Counterexample(**values)

        assert asdict(ce) == asdict(implementation) == values
        assert ce.variables == {"a": 10, "b": 0}
        assert ce.actual_behavior == "division attempted with zero"


class TestGeneratedTest:
    """Tests for GeneratedTest dataclass."""

    def test_create_test(self):
        """Generated tests carry executable code and source metadata."""
        counterexample = Counterexample(
            variables={"a": 10, "b": 0},
            expected_behavior="raise ZeroDivisionError",
            verification_type="division_by_zero",
        )
        test = GeneratedTest(
            name="divide_division_by_zero",
            description="Regression test for division by zero",
            code=(
                "def test_divide_division_by_zero():\n"
                "    with pytest.raises(ZeroDivisionError):\n"
                "        divide(10, 0)"
            ),
            language=Language.PYTHON,
            framework=Framework.PYTEST,
            file_name="test_math.py",
            target_function="divide",
            counterexample=counterexample,
        )

        assert "pytest.raises" in test.code
        assert test.framework == Framework.PYTEST
        assert test.target_function == "divide"


class TestTestGeneratorAgent:
    """Tests for TestGeneratorAgent."""

    def test_create_agent(self):
        """The public export preserves the generator's default behavior."""
        agent = PublicTestGenerator()
        implementation = RegressionTestGenerator()
        public_defaults = {
            language.value: framework.value
            for language, framework in agent.default_frameworks.items()
        }
        implementation_defaults = {
            language.value: framework.value
            for language, framework in implementation.default_frameworks.items()
        }

        assert public_defaults == implementation_defaults
        assert public_defaults["python"] == "pytest"
        assert public_defaults["go"] == "go_test"
        assert callable(agent.analyze)

    @pytest.mark.asyncio
    async def test_generate_test_name(self):
        """Public analysis derives a descriptive name from the counterexample."""
        agent = RegressionTestGenerator()

        result = await agent.analyze(
            "def process_data(data):\n    return data.value\n",
            {
                "file_path": "processor.py",
                "language": "python",
                "verification_results": {
                    "results": [
                        {
                            "satisfiable": True,
                            "counterexample": {"data": "null"},
                            "message": "Null safety violation",
                            "target_function": "process_data",
                        }
                    ]
                },
            },
        )

        assert result.success is True
        assert result.data["tests"][0]["name"] == "process_data_null_safety"
        assert result.data["tests"][0]["target_function"] == "process_data"

    @pytest.mark.asyncio
    async def test_select_framework(self):
        """Public analysis selects the configured default for each language."""
        agent = RegressionTestGenerator()
        cases = [
            ("python", "module.py", "pytest"),
            ("typescript", "module.ts", "jest"),
            ("go", "module.go", "go_test"),
        ]

        for language, file_path, expected_framework in cases:
            result = await agent.analyze(
                "def target(value):\n    return value\n",
                {
                    "file_path": file_path,
                    "language": language,
                    "verification_results": {
                        "findings": [
                            {
                                "counterexample": {"value": "null"},
                                "title": "Null input",
                                "target_function": "target",
                            }
                        ]
                    },
                },
            )

            assert result.success is True
            assert result.data["tests"][0]["framework"] == expected_framework

    @pytest.mark.asyncio
    async def test_generate_pytest_template(self):
        """Public analysis emits a concrete pytest regression test."""
        agent = RegressionTestGenerator()

        result = await agent.analyze(
            "def divide(a, b):\n    return a / b\n",
            {
                "file_path": "math.py",
                "language": "python",
                "module_name": "math",
                "verification_results": {
                    "results": [
                        {
                            "satisfiable": True,
                            "counterexample": {"a": 10, "b": 0},
                            "message": "Division by zero",
                            "target_function": "divide",
                        }
                    ]
                },
            },
        )

        generated = result.data["tests"][0]
        assert generated["file_name"] == "test_math.py"
        assert generated["imports"] == ["import pytest", "from math import *"]
        assert "def test_divide_division_by_zero():" in generated["code"]
        assert "with pytest.raises(ZeroDivisionError):" in generated["code"]
        assert "divide(a, b)" in generated["code"]


class TestTestSuite:
    """Tests for the intentional TestSuite public alias."""

    def test_create_suite(self):
        """Alias constructs the current TestGenerationResult model."""
        suite = PublicTestSuite(
            tests=[],
            coverage_delta=0.15,
            suggestions=["Add a boundary case"],
        )
        implementation = GenerationResult(
            tests=[],
            coverage_delta=0.15,
            suggestions=["Add a boundary case"],
        )

        assert asdict(suite) == asdict(implementation)
        assert suite.coverage_delta == 0.15
        assert suite.suggestions == ["Add a boundary case"]

    def test_suite_with_tests(self):
        """Suite can contain tests."""
        first_counterexample = Counterexample(
            variables={"value": "null"},
            expected_behavior="Null input",
        )
        second_counterexample = Counterexample(
            variables={"index": -1},
            expected_behavior="Bounds violation",
        )
        tests = [
            GeneratedTest(
                name="first_null_safety",
                description="First regression",
                code="def test_first_null_safety(): pass",
                language=Language.PYTHON,
                framework=Framework.PYTEST,
                file_name="test_first.py",
                target_function="first",
                counterexample=first_counterexample,
            ),
            GeneratedTest(
                name="second_bounds_violation",
                description="Second regression",
                code="def test_second_bounds_violation(): pass",
                language=Language.PYTHON,
                framework=Framework.PYTEST,
                file_name="test_second.py",
                target_function="second",
                counterexample=second_counterexample,
            ),
        ]

        suite = PublicTestSuite(tests=tests)

        assert len(suite.tests) == 2
        assert [test.target_function for test in suite.tests] == ["first", "second"]


# ============================================
# Feature 5: Natural Language Invariant Specs Tests
# ============================================


class TestParsedConstraint:
    """Tests for ParsedConstraint dataclass."""

    def test_create_constraint(self):
        """Current constraints store a type plus optional parameters."""
        constraint = ParsedConstraint(
            variable="x",
            constraint_type=ValueConstraint.POSITIVE,
            parameters={},
            original_text="x must be positive",
        )

        assert constraint.variable == "x"
        assert constraint.constraint_type == ValueConstraint.POSITIVE
        assert constraint.parameters == {}


class TestZ3Assertion:
    """Tests for the intentional Z3Assertion compiler alias."""

    def test_create_assertion(self):
        """Alias compiles current ParsedConstraint instances."""
        assert Z3Assertion is Z3Compiler

        compiler = Z3Assertion()
        z3_code, smtlib = compiler.compile(
            [
                ParsedConstraint(
                    variable="x",
                    constraint_type=ValueConstraint.POSITIVE,
                    original_text="x must be positive",
                )
            ]
        )

        assert "x = Int('x')" in z3_code
        assert "solver.add(x > 0)" in z3_code
        assert "(assert (> x 0))" in smtlib


class TestNaturalLanguageInvariantsAgent:
    """Tests for NaturalLanguageInvariantsAgent."""

    def test_create_agent(self):
        """The public pluralized name aliases the current BaseAgent."""
        assert NaturalLanguageInvariantsAgent is NaturalLanguageInvariantAgent

        agent = NaturalLanguageInvariantsAgent()

        assert isinstance(agent, BaseAgent)

    @pytest.mark.asyncio
    async def test_parse_positive_constraint(self):
        """Public analysis parses and compiles a positive constraint."""
        agent = NaturalLanguageInvariantsAgent()

        result = await agent.analyze(
            "",
            {
                "invariant_text": "x must be positive",
                "scope": "calculate",
                "invariant_type": "precondition",
            },
        )

        assert result.success is True
        assert result.data["constraints"] == [
            {
                "variable": "x",
                "type": str(ValueConstraint.POSITIVE),
                "parameters": {},
                "original": "x must be positive",
            }
        ]
        assert "solver.add(x > 0)" in result.data["z3_code"]

    @pytest.mark.asyncio
    async def test_parse_non_negative_constraint(self):
        """Public analysis parses non-negative constraints."""
        agent = NaturalLanguageInvariantsAgent()

        result = await agent.analyze(
            "",
            {"invariant_text": "count must be non-negative"},
        )

        assert result.success is True
        assert result.data["constraints"][0]["type"] == str(ValueConstraint.NON_NEGATIVE)
        assert "solver.add(count >= 0)" in result.data["z3_code"]

    @pytest.mark.asyncio
    async def test_parse_range_constraint(self):
        """Public analysis retains numeric range bounds."""
        agent = NaturalLanguageInvariantsAgent()

        result = await agent.analyze(
            "",
            {"invariant_text": "x must be between 0 and 100"},
        )

        assert result.success is True
        assert result.data["constraints"][0]["type"] == "range"
        assert result.data["constraints"][0]["parameters"] == {
            "min": 0.0,
            "max": 100.0,
        }
        assert "solver.add(And(x >= 0.0, x <= 100.0))" in result.data["z3_code"]

    @pytest.mark.asyncio
    async def test_parse_not_null_constraint(self):
        """Public analysis parses a not-null constraint."""
        agent = NaturalLanguageInvariantsAgent()

        result = await agent.analyze(
            "",
            {"invariant_text": "name must not be null"},
        )

        assert result.success is True
        assert result.data["constraints"][0]["type"] == str(ValueConstraint.NOT_NULL)
        assert "solver.add(name != None)" in result.data["z3_code"]

    @pytest.mark.asyncio
    async def test_generate_z3_from_constraint(self):
        """The public compiler API returns both Z3 and SMT-LIB forms."""
        agent = NaturalLanguageInvariantsAgent()

        result = await agent.compile_invariant(
            text="balance must be positive",
            scope="withdraw",
            invariant_type=InvariantType.PRECONDITION,
            variable_types={"balance": "Real"},
            use_llm_fallback=False,
        )

        assert result.success is True
        assert "balance = Real('balance')" in result.z3_code
        assert "solver.add(balance > 0)" in result.z3_code
        assert "(declare-const balance Real)" in result.smtlib_formula


class TestInvariantSpec:
    """Tests for the intentional InvariantSpec public alias."""

    def test_create_spec(self):
        """Alias constructs the current NaturalLanguageInvariant model."""
        assert InvariantSpec is NaturalLanguageInvariant

        constraint = ParsedConstraint(
            variable="balance",
            constraint_type=ValueConstraint.POSITIVE,
            original_text="balance must be positive",
        )
        spec = InvariantSpec(
            id="positive-balance",
            text="balance must be positive",
            invariant_type=InvariantType.PRECONDITION,
            scope="withdraw",
            parsed_constraints=[constraint],
            z3_formula="balance > 0",
            smtlib_formula="(assert (> balance 0))",
            confidence=0.9,
        )

        assert spec.id == "positive-balance"
        assert spec.scope == "withdraw"
        assert spec.parsed_constraints == [constraint]


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
        """Behavior changes identify an affected semantic node and risk."""
        node = SemanticNode(
            id="math.py:calculate",
            name="calculate",
            node_type="function",
            file_path="math.py",
            line_start=1,
            line_end=2,
            signature="def calculate(a, b, c)",
        )
        change = BehaviorChange(
            id="sig-calculate",
            change_type=ChangeType.SIGNATURE_CHANGE,
            description="Signature of calculate changed",
            before_behavior="def calculate(a, b)",
            after_behavior="def calculate(a, b, c)",
            affected_node=node,
            risk_level=RiskLevel.HIGH,
            evidence=["Added required parameter c"],
        )

        assert change.change_type == ChangeType.SIGNATURE_CHANGE
        assert change.affected_node.id == "math.py:calculate"
        assert change.risk_level == RiskLevel.HIGH


class TestSemanticDiffAgent:
    """Tests for SemanticDiffAgent."""

    def test_create_agent(self):
        """SemanticDiffAgent implements the standard BaseAgent interface."""
        agent = SemanticDiffAgent()

        assert isinstance(agent, BaseAgent)

    @pytest.mark.asyncio
    async def test_detect_signature_change(self):
        """Public analysis reports a concrete signature change."""
        agent = SemanticDiffAgent()

        old_code = "def greet(name):\n    return f'Hello {name}'"
        new_code = "def greet(name, title=''):\n    return f'Hello {title} {name}'"

        result = await agent.analyze(
            new_code,
            {
                "before_code": old_code,
                "file_path": "greeting.py",
                "language": "python",
                "base_commit": "base123",
                "head_commit": "head456",
            },
        )

        assert result.success is True
        assert result.data["nodes_modified"] == 1
        signature_change = next(
            change
            for change in result.data["behavior_changes"]
            if change["type"] == "signature_change"
        )
        assert signature_change["risk"] == "high"
        assert signature_change["before"] == "def greet(name)"
        assert signature_change["after"] == "def greet(name, title='')"

    @pytest.mark.asyncio
    async def test_generate_mermaid(self):
        """Public analysis includes a Mermaid visualization."""
        agent = SemanticDiffAgent()

        result = await agent.analyze(
            "def added():\n    return 1\n",
            {
                "before_code": "",
                "file_path": "example.py",
                "language": "python",
            },
        )

        mermaid = result.data["visualization"]["mermaid"]

        assert result.data["nodes_added"] == 1
        assert "graph LR" in mermaid
        assert "example_py_added[added]:::added" in mermaid

    @pytest.mark.asyncio
    async def test_generate_dot(self):
        """Public analysis includes a GraphViz DOT visualization."""
        agent = SemanticDiffAgent()

        result = await agent.analyze(
            "def added():\n    return 1\n",
            {
                "before_code": "",
                "file_path": "example.py",
                "language": "python",
            },
        )

        dot = result.data["visualization"]["dot"]

        assert dot.startswith("digraph SemanticDiff {")
        assert 'example_py_added [label="added", fillcolor="#4CAF50"];' in dot


class TestSemanticDiffResult:
    """Tests for SemanticDiffResult dataclass."""

    def test_create_result(self):
        """Result serializes the current SemanticDiff summary."""
        diff = SemanticDiff(
            base_commit="base",
            head_commit="head",
            nodes_added=[],
            nodes_removed=[],
            nodes_modified=[],
            behavior_changes=[],
            call_graph_changes=[],
            summary={"total_changes": 0},
        )
        result = SemanticDiffResult(
            diff=diff,
            risk_score=0.0,
            summary_text="No behavioral changes",
            recommendations=["No regression tests required"],
        )

        assert result.to_dict() == {
            "diff": {"total_changes": 0},
            "risk_score": 0.0,
            "summary_text": "No behavioral changes",
            "recommendations": ["No regression tests required"],
        }


# ============================================
# Feature 8: Team Learning Mode Tests
# ============================================


class TestTeamLearningAgent:
    """Tests for TeamLearningAgent."""

    def test_create_agent(self):
        """A new agent starts with no recorded occurrences."""
        agent = TeamLearningAgent()

        assert agent.aggregator._occurrences == []
        assert agent.identify_systemic_patterns() == []

    def test_configure_teams(self):
        """Can configure team mappings."""
        agent = TeamLearningAgent()

        agent.configure_teams(
            {
                "alice": "frontend",
                "bob": "backend",
            }
        )

        # Mapping should be set
        assert agent.aggregator._team_mapping.get("alice") == "frontend"

    def test_record_findings(self):
        """Records current Finding models with team and location metadata."""
        agent = TeamLearningAgent()
        agent.configure_teams({"alice": "backend"})

        findings = [
            Finding(
                id=uuid4(),
                title="Null reference",
                description="Object may be None before dereference",
                category=FindingCategory.NULL_SAFETY,
                severity=FindingSeverity.HIGH,
                location=CodeLocation(file_path="src/main.py", line_start=10),
                confidence=0.9,
                verification_type=VerificationType.PATTERN,
            )
        ]

        agent.record_findings(findings, "my-repo", "alice")

        assert len(agent.aggregator._occurrences) == 1
        occurrence = agent.aggregator._occurrences[0]
        assert occurrence.file_path == "src/main.py"
        assert occurrence.repository == "my-repo"
        assert occurrence.team == "backend"
        assert occurrence.category == FindingCategory.NULL_SAFETY

    def test_identify_patterns(self):
        """Repeated null findings become a concrete systemic pattern."""
        agent = TeamLearningAgent()
        agent.configure_teams({"dev1": "backend"})

        for i in range(10):
            findings = [
                Finding(
                    id=uuid4(),
                    title="Null pointer exception possible",
                    description="Value may be null before use",
                    category=FindingCategory.NULL_SAFETY,
                    severity=FindingSeverity.HIGH,
                    location=CodeLocation(file_path="src/main.py", line_start=i + 1),
                    confidence=0.85,
                    verification_type=VerificationType.PATTERN,
                )
            ]
            agent.record_findings(findings, "repo", "dev1")

        patterns = agent.identify_systemic_patterns(min_occurrences=5)

        assert len(patterns) == 1
        pattern = patterns[0]
        assert pattern.pattern_id == "null_reference"
        assert len(pattern.occurrences) == 10
        assert pattern.affected_teams == {"backend"}
        assert pattern.affected_repos == {"repo"}


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
            trend_vs_last_period=TrendDirection.IMPROVING,
            top_improving_teams=["frontend"],
            teams_needing_attention=["backend"],
        )
        assert report.total_findings == 100
        assert report.trend_vs_last_period == TrendDirection.IMPROVING


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
