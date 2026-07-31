"""Tests for all 10 next-gen v0.6.0 features."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

# =============================================================================
# Feature 1: Verified Auto-Fix with Test Generation
# =============================================================================


class TestAutofixValidation:
    """Tests for the FixValidator, RegressionChecker, BatchFixProcessor, etc."""

    def test_fix_validation_config_defaults(self):
        from codeverify_core.autofix_validation import FixValidationConfig

        config = FixValidationConfig()
        assert config.max_validation_attempts == 3
        assert config.regression_check_enabled is True
        assert config.auto_rollback is True
        assert config.batch_size == 10

    def test_fix_validation_status_enum(self):
        from codeverify_core.autofix_validation import FixValidationStatus

        assert FixValidationStatus.PENDING == "pending"
        assert FixValidationStatus.PASSED == "passed"
        assert FixValidationStatus.REGRESSION_DETECTED == "regression_detected"

    def test_fix_validator_valid_fix(self):
        from codeverify_core.autofix_validation import FixValidationStatus, FixValidator

        validator = FixValidator()
        original = "x = 1 / 0"
        fixed = "x = 1 / 1 if 1 != 0 else 0"
        issue = "division by zero at line 1"
        result = validator.validate_fix(original, fixed, issue, language="python")
        assert result.status in (FixValidationStatus.PASSED, FixValidationStatus.FAILED)
        assert result.attempt_count >= 1
        assert result.validation_time_ms >= 0

    def test_fix_validator_syntax_error(self):
        from codeverify_core.autofix_validation import FixValidationStatus, FixValidator

        validator = FixValidator()
        result = validator.validate_fix("x = 1", "x = (", "test issue at line 1")
        assert result.status == FixValidationStatus.FAILED
        assert not result.issue_resolved

    def test_regression_checker(self):
        from codeverify_core.autofix_validation import RegressionChecker

        checker = RegressionChecker()
        results = checker.check_regressions("def f(): return 1", "def f(): return 2")
        assert isinstance(results, list)

    def test_batch_fix_processor(self):
        from codeverify_core.autofix_validation import (
            BatchFixProcessor,
            BatchFixStrategy,
        )

        processor = BatchFixProcessor()
        fixes = [
            {
                "original_code": "x = 1/0",
                "fixed_code": "x = 0",
                "issue": "division by zero",
                "language": "python",
            },
        ]
        result = processor.process_batch(fixes, strategy=BatchFixStrategy.SEQUENTIAL)
        assert result.total_fixes == 1
        assert result.total_time_ms >= 0

    def test_pr_description_generator(self):
        from codeverify_core.autofix_validation import (
            BatchFixProcessor,
            BatchFixStrategy,
            PRDescriptionGenerator,
        )

        processor = BatchFixProcessor()
        batch_result = processor.process_batch(
            [
                {
                    "original_code": "x = 1/0",
                    "fixed_code": "x = 0",
                    "issue": "div by zero",
                    "language": "python",
                }
            ],
            strategy=BatchFixStrategy.SEQUENTIAL,
        )
        gen = PRDescriptionGenerator()
        pr = gen.generate(batch_result)
        assert pr.title
        assert pr.body
        assert isinstance(pr.labels, list)

    def test_fix_validation_result_to_dict(self):
        from codeverify_core.autofix_validation import FixValidationResult, FixValidationStatus

        result = FixValidationResult(
            fix_id="fix-1",
            status=FixValidationStatus.PASSED,
            issue_resolved=True,
            regressions=[],
            validation_time_ms=50.0,
            attempt_count=1,
            original_code="x = 1",
            fixed_code="x = 2",
        )
        d = result.to_dict()
        assert d["fix_id"] == "fix-1"
        assert d["status"] == "passed"
        assert d["issue_resolved"] is True


# =============================================================================
# Feature 2: Continuous Learning from Production
# =============================================================================


class TestProductionLearning:
    """Tests for the production learning engine."""

    def test_incident_severity_enum(self):
        from codeverify_core.production_learning import IncidentSeverity

        assert IncidentSeverity.CRITICAL == "critical"
        assert IncidentSeverity.LOW == "low"

    def test_incident_collector_record_and_get(self):
        from codeverify_core.production_learning import (
            IncidentCollector,
            IncidentSeverity,
            ProductionIncident,
        )

        collector = IncidentCollector()
        incident = ProductionIncident(
            id="inc-1",
            timestamp=datetime.now(UTC),
            service="api",
            error_type="NullPointerException",
            stack_trace="at line 42",
            severity=IncidentSeverity.HIGH,
            commit_sha="abc123",
            file_path="src/main.py",
            function_name="process",
            root_cause=None,
        )
        collector.record_incident(incident)
        incidents = collector.get_incidents()
        assert len(incidents) >= 1
        assert incidents[0].id == "inc-1"

    def test_incident_correlation(self):
        from codeverify_core.production_learning import (
            IncidentCollector,
            IncidentSeverity,
            ProductionIncident,
        )

        collector = IncidentCollector()
        incident = ProductionIncident(
            id="inc-2",
            timestamp=datetime.now(UTC),
            service="worker",
            error_type="IndexError",
            stack_trace="index out of range",
            severity=IncidentSeverity.MEDIUM,
            commit_sha="def456",
            file_path="src/worker.py",
            function_name=None,
            root_cause=None,
        )
        correlation = collector.correlate_with_commits(incident)
        # May or may not find correlation, but should return valid type
        assert correlation is None or correlation.incident_id == "inc-2"

    def test_threshold_tuner(self):
        from codeverify_core.production_learning import ThresholdTuner

        tuner = ThresholdTuner()
        threshold = tuner.analyze_threshold("rule-1", [], [])
        assert threshold.rule_id == "rule-1"
        assert 0 <= threshold.current_value <= 1

    def test_pattern_extractor(self):
        from codeverify_core.production_learning import PatternExtractor

        extractor = PatternExtractor()
        patterns = extractor.extract_patterns([])
        assert isinstance(patterns, list)

    def test_ab_test_manager(self):
        from codeverify_core.production_learning import ABTestManager, ABTestStatus

        manager = ABTestManager()
        test = manager.create_test(
            name="Test rule v2",
            control_rule={"threshold": 0.5},
            variant_rule={"threshold": 0.7},
        )
        assert test.status == ABTestStatus.DRAFT
        assert test.name == "Test rule v2"
        manager.start_test(test.id)
        started = manager.evaluate_test(test.id)
        assert started.status == ABTestStatus.RUNNING

    def test_production_learning_engine(self):
        from codeverify_core.production_learning import ProductionLearningEngine

        engine = ProductionLearningEngine()
        report = engine.run_learning_cycle()
        assert report.incidents_analyzed >= 0
        assert isinstance(report.recommendations, list)

    def test_production_incident_to_dict(self):
        from codeverify_core.production_learning import (
            IncidentSeverity,
            ProductionIncident,
        )

        incident = ProductionIncident(
            id="inc-3",
            timestamp=datetime.now(UTC),
            service="api",
            error_type="ValueError",
            stack_trace="line 10",
            severity=IncidentSeverity.LOW,
            commit_sha=None,
            file_path=None,
            function_name=None,
            root_cause=None,
        )
        d = incident.to_dict()
        assert d["id"] == "inc-3"
        assert d["severity"] == "low"


# =============================================================================
# Feature 3: Real-Time Pair Programming
# =============================================================================


class TestRealTimePairProgramming:
    """Tests for the pair programming engine."""

    def test_suggestion_type_enum(self):
        from codeverify_core.realtime_pair_programming import SuggestionType

        assert SuggestionType.BUG_FIX == "bug_fix"
        assert SuggestionType.SECURITY == "security"

    def test_code_change_event(self):
        from codeverify_core.realtime_pair_programming import CodeChangeEvent

        event = CodeChangeEvent(
            file_path="src/main.py",
            changed_lines=[1, 2, 3],
            content="def foo():\n    pass",
            timestamp=time.time(),
            cursor_position=(2, 4),
        )
        assert event.language == "python"

    def test_incremental_analyzer(self):
        from codeverify_core.realtime_pair_programming import (
            CodeChangeEvent,
            IncrementalAnalyzer,
        )

        analyzer = IncrementalAnalyzer()
        event = CodeChangeEvent(
            file_path="test.py",
            changed_lines=[1],
            content="def foo(x):\n    return x / 0\n",
            timestamp=time.time(),
            cursor_position=(2, 0),
        )
        result = analyzer.analyze_change(event)
        assert result.file_path == "test.py"
        assert result.analysis_time_ms >= 0

    def test_smart_debouncer(self):
        from codeverify_core.realtime_pair_programming import (
            CodeChangeEvent,
            DebounceConfig,
            SmartDebouncer,
        )

        debouncer = SmartDebouncer(DebounceConfig(typing_pause_ms=100))
        event = CodeChangeEvent(
            file_path="test.py",
            changed_lines=[1],
            content="x = 1",
            timestamp=time.time(),
            cursor_position=(1, 5),
        )
        # First event should generally not trigger immediately
        result = debouncer.should_trigger(event)
        assert isinstance(result, bool)

    def test_suggestion_engine(self):
        from codeverify_core.realtime_pair_programming import (
            AnalysisScope,
            IncrementalAnalysisResult,
            SuggestionEngine,
            UserPreferences,
        )

        engine = SuggestionEngine()
        analysis = IncrementalAnalysisResult(
            file_path="test.py",
            changed_functions=["foo"],
            findings=[{"type": "bug", "message": "div by zero", "line": 2, "severity": "high"}],
            analysis_time_ms=10.0,
            cache_hit=False,
            scope=AnalysisScope.FUNCTION,
        )
        prefs = UserPreferences(user_id="user-1")
        suggestions = engine.generate_suggestions(analysis, prefs)
        assert isinstance(suggestions, list)

    def test_personalization_engine(self):
        from codeverify_core.realtime_pair_programming import (
            FeedbackAction,
            PersonalizationEngine,
        )

        engine = PersonalizationEngine()
        prefs = engine.get_preferences("user-1")
        assert prefs.user_id == "user-1"
        engine.record_feedback("user-1", "sugg-1", FeedbackAction.ACCEPTED)

    def test_pair_session(self):
        from codeverify_core.realtime_pair_programming import (
            CodeChangeEvent,
            RealTimePairSession,
        )

        session = RealTimePairSession(session_id="sess-1", user_id="user-1")
        event = CodeChangeEvent(
            file_path="test.py",
            changed_lines=[1],
            content="x = 1",
            timestamp=time.time(),
            cursor_position=(1, 5),
        )
        suggestions = session.on_code_change(event)
        assert isinstance(suggestions, list)
        metrics = session.get_metrics()
        assert metrics.session_id == "sess-1"
        final = session.close()
        assert final.session_id == "sess-1"


# =============================================================================
# Feature 4: Cross-Language Contract Verification
# =============================================================================


class TestCrossLanguageContracts:
    """Tests for cross-language contract verification."""

    def test_contract_language_enum(self):
        from codeverify_core.cross_language_contracts import ContractLanguage

        assert ContractLanguage.PYTHON == "python"
        assert ContractLanguage.TYPESCRIPT == "typescript"

    def test_universal_type(self):
        from codeverify_core.cross_language_contracts import (
            ContractLanguage,
            UniversalType,
        )

        t = UniversalType(
            name="user_id",
            language=ContractLanguage.PYTHON,
            native_type="int",
        )
        assert not t.nullable
        d = t.to_dict()
        assert d["name"] == "user_id"

    def test_type_mapper(self):
        from codeverify_core.cross_language_contracts import (
            ContractLanguage,
            TypeMapper,
            UniversalType,
        )

        mapper = TypeMapper()
        source = UniversalType(
            name="count",
            language=ContractLanguage.PYTHON,
            native_type="int",
        )
        mapping = mapper.map_type(source, ContractLanguage.TYPESCRIPT)
        assert mapping.source_lang == ContractLanguage.PYTHON
        assert mapping.target_lang == ContractLanguage.TYPESCRIPT

    def test_contract_extractor_python(self):
        from codeverify_core.cross_language_contracts import (
            ContractExtractor,
            ContractLanguage,
        )

        extractor = ContractExtractor()
        code = """
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> str:
    return f"Hello {name}"
"""
        contracts = extractor.extract_contracts(code, ContractLanguage.PYTHON, "math.py")
        assert len(contracts) >= 2
        assert any(c.name == "add" for c in contracts)

    def test_contract_extractor_typescript(self):
        from codeverify_core.cross_language_contracts import (
            ContractExtractor,
            ContractLanguage,
        )

        extractor = ContractExtractor()
        code = """
function add(a: number, b: number): number {
    return a + b;
}

export function greet(name: string): string {
    return `Hello ${name}`;
}
"""
        contracts = extractor.extract_contracts(code, ContractLanguage.TYPESCRIPT, "math.ts")
        assert len(contracts) >= 2

    def test_cross_language_verifier(self):
        from codeverify_core.cross_language_contracts import (
            ContractEndpoint,
            ContractLanguage,
            CrossLanguageVerifier,
            UniversalType,
        )

        verifier = CrossLanguageVerifier()
        py_endpoint = ContractEndpoint(
            name="get_user",
            language=ContractLanguage.PYTHON,
            file_path="api.py",
            line=10,
            parameters={
                "user_id": UniversalType(
                    name="user_id", language=ContractLanguage.PYTHON, native_type="int"
                )
            },
            return_type=UniversalType(
                name="result", language=ContractLanguage.PYTHON, native_type="str"
            ),
        )
        ts_endpoint = ContractEndpoint(
            name="get_user",
            language=ContractLanguage.TYPESCRIPT,
            file_path="api.ts",
            line=5,
            parameters={
                "user_id": UniversalType(
                    name="user_id", language=ContractLanguage.TYPESCRIPT, native_type="number"
                )
            },
            return_type=UniversalType(
                name="result", language=ContractLanguage.TYPESCRIPT, native_type="string"
            ),
        )
        report = verifier.verify_contracts([py_endpoint], [ts_endpoint])
        assert report.contracts_checked >= 1
        assert report.total_endpoints >= 1

    def test_stub_generation(self):
        from codeverify_core.cross_language_contracts import (
            ContractEndpoint,
            ContractLanguage,
            CrossLanguageVerifier,
            UniversalType,
        )

        verifier = CrossLanguageVerifier()
        endpoint = ContractEndpoint(
            name="add",
            language=ContractLanguage.PYTHON,
            file_path="math.py",
            line=1,
            parameters={
                "a": UniversalType(name="a", language=ContractLanguage.PYTHON, native_type="int"),
                "b": UniversalType(name="b", language=ContractLanguage.PYTHON, native_type="int"),
            },
            return_type=UniversalType(
                name="result", language=ContractLanguage.PYTHON, native_type="int"
            ),
        )
        stub = verifier.generate_interface_stubs([endpoint], ContractLanguage.TYPESCRIPT)
        assert isinstance(stub, str)
        assert len(stub) > 0


# =============================================================================
# Feature 5: Supply Chain Risk Scoring
# =============================================================================


class TestSupplyChainRisk:
    """Tests for supply chain risk scoring."""

    def test_cve_severity_enum(self):
        from codeverify_core.supply_chain_risk import CVESeverity

        assert CVESeverity.CRITICAL == "critical"

    def test_risk_scorer(self):
        from codeverify_core.supply_chain_risk import DependencyRiskProfile, RiskScorer

        scorer = RiskScorer()
        profile = DependencyRiskProfile(
            package_name="lodash",
            version="4.17.21",
            ecosystem="npm",
        )
        score = scorer.score_dependency(profile)
        assert 0 <= score <= 100

    def test_risk_scorer_typosquat(self):
        from codeverify_core.supply_chain_risk import RiskScorer

        scorer = RiskScorer()
        # lodahs is close to lodash - transposition = 2 ops in basic levenshtein
        dist = scorer._levenshtein_distance("lodash", "lodahs")
        assert dist <= 2

    def test_sbom_generator(self):
        from codeverify_core.supply_chain_risk import (
            SBOMComponent,
            SBOMFormat,
            SBOMGenerator,
        )

        generator = SBOMGenerator()
        components = [
            SBOMComponent(
                name="requests",
                version="2.31.0",
                ecosystem="pypi",
                purl="pkg:pypi/requests@2.31.0",
                licenses=["Apache-2.0"],
                supplier="PSF",
            ),
        ]
        sbom = generator.generate("test-project", components, SBOMFormat.CYCLONEDX)
        assert sbom.project_name == "test-project"
        assert len(sbom.components) == 1
        json_output = generator.export_json(sbom)
        assert "test-project" in json_output

    def test_supply_chain_risk_analyzer(self):
        from codeverify_core.supply_chain_risk import SupplyChainRiskAnalyzer

        analyzer = SupplyChainRiskAnalyzer()
        deps = [
            {"name": "requests", "version": "2.31.0", "ecosystem": "pypi"},
            {"name": "flask", "version": "3.0.0", "ecosystem": "pypi"},
        ]
        report = analyzer.analyze("test-project", deps)
        assert report.project_name == "test-project"
        assert report.total_dependencies == 2
        assert 0 <= report.risk_score <= 100

    def test_cve_correlator(self):
        from codeverify_core.supply_chain_risk import CVECorrelator

        correlator = CVECorrelator()
        cves = correlator.correlate("requests", "2.31.0", "pypi")
        assert isinstance(cves, list)

    def test_supply_chain_risk_report_to_dict(self):
        from codeverify_core.supply_chain_risk import SupplyChainRiskAnalyzer

        analyzer = SupplyChainRiskAnalyzer()
        report = analyzer.analyze("proj", [{"name": "x", "version": "1.0", "ecosystem": "npm"}])
        d = report.to_dict()
        assert d["project_name"] == "proj"


# =============================================================================
# Feature 6: Multi-Repository Impact Analysis
# =============================================================================


class TestMultiRepoImpact:
    """Tests for multi-repository impact analysis."""

    def test_index_status_enum(self):
        from codeverify_core.multi_repo_impact import IndexStatus

        assert IndexStatus.INDEXED == "indexed"
        assert IndexStatus.STALE == "stale"

    def test_org_repository_indexer(self):
        from codeverify_core.multi_repo_impact import OrgRepositoryIndexer

        indexer = OrgRepositoryIndexer()
        code_files = {
            "main.py": "import os\nfrom utils import helper\ndef process(): pass\n",
            "utils.py": "def helper(): pass\n",
        }
        index = indexer.index_repository("my-repo", "my-org", code_files)
        assert index.repo_name == "my-repo"
        assert index.file_count == 2

    def test_org_dependency_graph(self):
        from codeverify_core.multi_repo_impact import OrgRepositoryIndexer

        indexer = OrgRepositoryIndexer()
        repo1 = indexer.index_repository("lib-a", "org", {"lib.py": "def compute(): pass\n"})
        repo2 = indexer.index_repository(
            "app-b", "org", {"app.py": "from lib_a import compute\ndef main(): compute()\n"}
        )
        graph = indexer.build_org_graph([repo1, repo2])
        assert graph.total_repos == 2
        assert graph.indexed_repos == 2

    def test_blast_radius_calculator(self):
        from codeverify_core.multi_repo_impact import (
            BlastRadiusCalculator,
            OrgRepositoryIndexer,
        )

        indexer = OrgRepositoryIndexer()
        repo1 = indexer.index_repository("lib", "org", {"lib.py": "def helper(): pass\n"})
        repo2 = indexer.index_repository(
            "app", "org", {"app.py": "from lib import helper\ndef main(): helper()\n"}
        )
        graph = indexer.build_org_graph([repo1, repo2])
        calc = BlastRadiusCalculator()
        blast = calc.calculate(graph, "lib", ["helper"])
        assert blast.change_repo == "lib"
        assert isinstance(blast.affected_repos, list)

    def test_team_notifier(self):
        from codeverify_core.multi_repo_impact import (
            BlastRadiusResult,
            TeamNotifier,
        )

        notifier = TeamNotifier()
        blast = BlastRadiusResult(
            change_repo="lib",
            change_description="Updated helper function",
            change_scope="function",
            affected_repos=["app-a", "app-b"],
            affected_teams=["team-frontend"],
            affected_services=["api"],
            risk_score=0.7,
            direct_impacts=2,
            transitive_impacts=0,
            breaking_changes=[],
        )
        notifications = notifier.generate_notifications(
            blast, {"app-a": "team-a", "app-b": "team-b"}
        )
        assert len(notifications) >= 1

    def test_migration_planner(self):
        from codeverify_core.multi_repo_impact import (
            BlastRadiusResult,
            MigrationPlanner,
        )

        planner = MigrationPlanner()
        blast = BlastRadiusResult(
            change_repo="lib",
            change_description="Breaking API change",
            change_scope="api",
            affected_repos=["app-1", "app-2"],
            affected_teams=[],
            affected_services=[],
            risk_score=0.8,
            direct_impacts=2,
            transitive_impacts=0,
            breaking_changes=[{"symbol": "create_user", "type": "removed"}],
        )
        plan = planner.create_plan(blast, "Migrate to v2 API")
        assert plan.name  # Just check it has a name
        assert len(plan.stages) >= 1


# =============================================================================
# Feature 7: AI-Powered Refactoring Suggestions
# =============================================================================


class TestRefactoringEngine:
    """Tests for the refactoring engine."""

    def test_smell_type_enum(self):
        from codeverify_core.refactoring_engine import SmellType

        assert SmellType.GOD_CLASS == "god_class"
        assert SmellType.LONG_METHOD == "long_method"

    def test_complexity_analyzer(self):
        from codeverify_core.refactoring_engine import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer()
        code = """
def process(x):
    if x > 0:
        if x > 10:
            for i in range(x):
                if i % 2 == 0:
                    print(i)
        else:
            print(x)
    else:
        print("negative")
"""
        metrics = analyzer.analyze(code)
        assert metrics.cyclomatic_complexity >= 1
        assert metrics.nesting_depth >= 1
        assert metrics.lines_of_code > 0

    def test_code_smell_detector(self):
        from codeverify_core.refactoring_engine import CodeSmellDetector

        detector = CodeSmellDetector()
        # A "long method" - lots of lines
        long_func = "def long_func():\n" + "\n".join([f"    x{i} = {i}" for i in range(50)])
        smells = detector.detect_smells(long_func, "main.py")
        assert isinstance(smells, list)

    def test_code_smell_to_dict(self):
        from codeverify_core.refactoring_engine import CodeSmell, SmellType

        smell = CodeSmell(
            id="smell-1",
            smell_type=SmellType.LONG_METHOD,
            file_path="main.py",
            line_start=1,
            line_end=100,
            description="Method too long",
            severity=0.8,
            confidence=0.9,
        )
        d = smell.to_dict()
        assert d["id"] == "smell-1"
        assert d["smell_type"] == "long_method"

    def test_refactoring_planner(self):
        from codeverify_core.refactoring_engine import (
            CodeSmell,
            RefactoringPlanner,
            SmellType,
        )

        planner = RefactoringPlanner()
        smells = [
            CodeSmell(
                id="s1",
                smell_type=SmellType.LONG_METHOD,
                file_path="main.py",
                line_start=1,
                line_end=60,
                description="Long method",
                severity=0.7,
                confidence=0.9,
            )
        ]
        code = "def long_func():\n" + "\n".join([f"    x{i} = {i}" for i in range(50)])
        plans = planner.create_plan(smells, code, "python")
        assert isinstance(plans, list)

    def test_refactoring_engine_analyze(self):
        from codeverify_core.refactoring_engine import RefactoringEngine

        engine = RefactoringEngine()
        files = {
            "main.py": "def simple(): return 1\n",
            "utils.py": "def helper(x): return x + 1\n",
        }
        report = engine.analyze_project(files)
        assert report.project_path  # Not empty
        assert isinstance(report.smells_detected, list)
        assert report.technical_debt_score >= 0

    def test_calculate_technical_debt(self):
        from codeverify_core.refactoring_engine import ComplexityMetrics, RefactoringEngine

        engine = RefactoringEngine()
        metrics = ComplexityMetrics(
            cyclomatic_complexity=20,
            cognitive_complexity=15,
            nesting_depth=5,
            lines_of_code=500,
            parameter_count=10,
            dependency_count=8,
            coupling_score=0.8,
            cohesion_score=0.3,
        )
        debt = engine.calculate_technical_debt(metrics)
        assert debt > 0


# =============================================================================
# Feature 8: Compliance-as-Code Framework
# =============================================================================


class TestComplianceAsCode:
    """Tests for the compliance-as-code framework."""

    def test_framework_type_enum(self):
        from codeverify_core.compliance_as_code import ComplianceFrameworkType

        assert ComplianceFrameworkType.SOC2_TYPE_II.value == "SOC2_TYPE_II"
        assert ComplianceFrameworkType.HIPAA.value == "HIPAA"

    def test_framework_mapper_soc2(self):
        from codeverify_core.compliance_as_code import (
            ComplianceFrameworkType,
            FrameworkMapper,
        )

        mapper = FrameworkMapper()
        controls = mapper.get_controls(ComplianceFrameworkType.SOC2_TYPE_II)
        assert len(controls) >= 1
        assert any("CC" in c.control_id for c in controls)

    def test_framework_mapper_hipaa(self):
        from codeverify_core.compliance_as_code import (
            ComplianceFrameworkType,
            FrameworkMapper,
        )

        mapper = FrameworkMapper()
        controls = mapper.get_controls(ComplianceFrameworkType.HIPAA)
        assert len(controls) >= 1

    def test_evidence_vault(self):
        from codeverify_core.compliance_as_code import (
            EvidenceArtifact,
            EvidenceType,
            EvidenceVault,
        )

        vault = EvidenceVault()
        evidence = EvidenceArtifact(
            id="ev-1",
            evidence_type=EvidenceType.VERIFICATION_REPORT,
            title="Test report",
            description="A verification report",
            content_hash="",
            collected_at=datetime.now(UTC),
            control_ids=["CC5.1"],
        )
        stored_id = vault.store_evidence(evidence)
        assert stored_id == "ev-1"
        retrieved = vault.retrieve_evidence("ev-1")
        assert retrieved is not None
        assert retrieved.title == "Test report"

    def test_evidence_vault_integrity(self):
        from codeverify_core.compliance_as_code import (
            EvidenceArtifact,
            EvidenceType,
            EvidenceVault,
        )

        vault = EvidenceVault()
        evidence = EvidenceArtifact(
            id="ev-2",
            evidence_type=EvidenceType.SCAN_RESULT,
            title="Scan result",
            description="Test scan result content for integrity check",
            content_hash="",
            collected_at=datetime.now(UTC),
            control_ids=["CC6.1"],
        )
        vault.store_evidence(evidence)
        stored = vault.retrieve_evidence("ev-2")
        assert stored is not None
        assert stored.content_hash
        assert vault.verify_integrity("ev-2") is True

        stored.title = "Tampered scan result"
        assert vault.verify_integrity("ev-2") is False

    def test_attestation_engine(self):
        from codeverify_core.compliance_as_code import (
            AttestationEngine,
            AttestationLevel,
        )

        engine = AttestationEngine(secret=b"test-attestation-secret")
        attestation = engine.create_attestation(
            control_id="CC5.1",
            attester="admin@test.com",
            statement="Control is effective",
            evidence_ids=["ev-1"],
            level=AttestationLevel.AUTOMATED,
        )
        assert attestation.control_id == "CC5.1"
        assert attestation.valid
        assert engine.verify_attestation(attestation) is True

    def test_attestation_engine_requires_secret_for_signing(self, monkeypatch):
        from codeverify_core.compliance_as_code import AttestationEngine

        monkeypatch.delenv("CODEVERIFY_ATTESTATION_SECRET", raising=False)
        engine = AttestationEngine()

        with pytest.raises(RuntimeError, match="CODEVERIFY_ATTESTATION_SECRET"):
            engine.create_attestation(
                control_id="CC5.1",
                attester="admin@test.com",
                statement="Control is effective",
                evidence_ids=["ev-1"],
            )

    def test_compliance_report_generator(self):
        from codeverify_core.compliance_as_code import (
            ComplianceFrameworkType,
            ComplianceReportGenerator,
            ControlAssessment,
        )

        generator = ComplianceReportGenerator()
        assessments = [
            ControlAssessment(
                control_id="CC5.1",
                framework=ComplianceFrameworkType.SOC2_TYPE_II,
                status="passing",
                evidence_count=3,
                attestation_count=1,
                gaps=[],
                last_assessed=datetime.now(UTC),
            ),
        ]
        report = generator.generate_report(
            "test-org",
            ComplianceFrameworkType.SOC2_TYPE_II,
            assessments,
            datetime.now(UTC),
            datetime.now(UTC),
        )
        assert report.organization == "test-org"
        assert report.passing_controls >= 1

    def test_compliance_as_code_engine(self):
        from codeverify_core.compliance_as_code import (
            ComplianceAsCodeEngine,
            ComplianceFrameworkType,
        )

        engine = ComplianceAsCodeEngine(attestation_secret=b"test-attestation-secret")
        report = engine.assess_compliance(
            "test-org",
            ComplianceFrameworkType.SOC2_TYPE_II,
            [
                {
                    "rule_id": "risk_assessment_check",
                    "passed": True,
                    "file": "main.py",
                }
            ],
        )
        assert report.organization == "test-org"
        assert report.total_controls >= 1
        assert report.passing_controls >= 1


# =============================================================================
# Feature 9: Proof Marketplace & Reuse
# =============================================================================


class TestProofMarketplaceV2:
    """Tests for the proof marketplace v2."""

    def test_proof_category_enum(self):
        from codeverify_core.proof_marketplace_v2 import ProofCategory

        assert ProofCategory.NULL_SAFETY == "null_safety"
        assert ProofCategory.SECURITY == "security"

    def test_proof_storage(self):
        from codeverify_core.proof_marketplace_v2 import (
            ProofCategory,
            ProofContent,
            ProofMetadata,
            ProofStorage,
        )

        storage = ProofStorage()
        metadata = ProofMetadata(
            id="proof-1",
            name="Null Safety Check",
            description="Ensures no null pointer dereference",
            category=ProofCategory.NULL_SAFETY,
            author_id="user-1",
            language="python",
            framework=None,
            tags=["null", "safety"],
            created_at=datetime.now(UTC),
        )
        content = ProofContent(
            proof_id="proof-1",
            z3_expression="x != None",
            natural_language="x must not be None",
            code_pattern="if x is None: raise",
            test_cases=[{"input": None, "expected": "raise"}],
            applicable_languages=["python"],
        )
        stored_id = storage.store_proof(metadata, content)
        assert stored_id == "proof-1"
        result = storage.get_proof("proof-1")
        assert result is not None
        assert result[0].name == "Null Safety Check"

    def test_proof_search_engine(self):
        from codeverify_core.proof_marketplace_v2 import (
            ProofCategory,
            ProofContent,
            ProofMetadata,
            ProofSearchEngine,
            ProofStorage,
            SearchQuery,
        )

        storage = ProofStorage()
        search_engine = ProofSearchEngine()
        search_engine._storage = storage

        # Add some proofs
        for i in range(3):
            meta = ProofMetadata(
                id=f"search-proof-{i}",
                name=f"Proof {i}",
                description=f"Test proof number {i} for bounds checking",
                category=ProofCategory.BOUNDS_CHECK,
                author_id="user-1",
                language="python",
                framework=None,
                tags=["bounds", "array"],
                created_at=datetime.now(UTC),
            )
            content = ProofContent(
                proof_id=f"search-proof-{i}",
                z3_expression="index >= 0 && index < len",
                natural_language="Index within bounds",
                code_pattern="arr[index]",
                test_cases=[],
                applicable_languages=["python"],
            )
            storage.store_proof(meta, content)

        query = SearchQuery(query="bounds")
        result = search_engine.search(query)
        assert isinstance(result.proofs, list)

    def test_gamification_engine(self):
        from codeverify_core.proof_marketplace_v2 import (
            ContributionType,
            GamificationEngine,
        )

        engine = GamificationEngine()
        points = engine.award_contribution("user-1", ContributionType.PROOF_SUBMITTED)
        assert points > 0
        profile = engine.get_profile("user-1")
        assert profile.user_id == "user-1"
        assert profile.reputation_points > 0

    def test_gamification_leaderboard(self):
        from codeverify_core.proof_marketplace_v2 import (
            ContributionType,
            GamificationEngine,
        )

        engine = GamificationEngine()
        engine.award_contribution("user-a", ContributionType.PROOF_SUBMITTED)
        engine.award_contribution("user-b", ContributionType.PROOF_REVIEWED)
        engine.award_contribution("user-a", ContributionType.PROOF_SUBMITTED)
        leaderboard = engine.get_leaderboard(top_n=10)
        assert isinstance(leaderboard, list)

    def test_proof_quality_manager(self):
        from codeverify_core.proof_marketplace_v2 import (
            ProofCategory,
            ProofContent,
            ProofMetadata,
            ProofQualityManager,
            ProofReview,
            ProofStorage,
            QualityTier,
        )

        storage = ProofStorage()
        meta = ProofMetadata(
            id="quality-proof",
            name="Quality Test",
            description="For quality testing",
            category=ProofCategory.CORRECTNESS,
            author_id="user-1",
            language="python",
            framework=None,
            tags=[],
            created_at=datetime.now(UTC),
        )
        content = ProofContent(
            proof_id="quality-proof",
            z3_expression="x > 0",
            natural_language="x is positive",
            code_pattern="assert x > 0",
            test_cases=[],
            applicable_languages=["python"],
        )
        storage.store_proof(meta, content)

        manager = ProofQualityManager()
        manager._storage = storage
        review = ProofReview(
            id="review-1",
            proof_id="quality-proof",
            reviewer_id="reviewer-1",
            rating=5,
            comment="Excellent proof",
            quality_assessment=QualityTier.EXPERT_REVIEWED,
            issues_found=[],
            created_at=datetime.now(UTC),
        )
        manager.submit_review(review)
        metrics = manager.get_quality_metrics("quality-proof")
        assert metrics.proof_id == "quality-proof"

    def test_marketplace_v2(self):
        from codeverify_core.proof_marketplace_v2 import ProofMarketplaceV2

        marketplace = ProofMarketplaceV2()
        stats = marketplace.get_community_stats()
        assert isinstance(stats, dict)
        assert "total_proofs" in stats


# =============================================================================
# Feature 10: Verification Performance Profiler
# =============================================================================


class TestVerificationProfiler:
    """Tests for the verification performance profiler."""

    def test_profile_stage_enum(self):
        from codeverify_core.verification_profiler import ProfileStage

        assert ProfileStage.PARSING == "parsing"
        assert ProfileStage.Z3_SOLVING == "z3_solving"

    def test_verification_instrumenter(self):
        from codeverify_core.verification_profiler import (
            ProfileStage,
            VerificationInstrumenter,
        )

        instrumenter = VerificationInstrumenter()
        profile_id = instrumenter.start_profiling("test_func", "test.py", "python")
        assert profile_id

        instrumenter.record_stage(profile_id, ProfileStage.PARSING, duration_ms=10.0, memory_mb=5.0)
        instrumenter.record_stage(
            profile_id, ProfileStage.Z3_SOLVING, duration_ms=50.0, memory_mb=10.0
        )

        profile = instrumenter.end_profiling(profile_id)
        assert profile.function_name == "test_func"
        assert len(profile.stages) == 2
        assert profile.total_time_ms >= 0

    def test_bottleneck_detector(self):
        from codeverify_core.verification_profiler import (
            BottleneckDetector,
            FunctionProfile,
            ProfileStage,
            StageProfile,
        )

        detector = BottleneckDetector(timeout_threshold_ms=100)
        profile = FunctionProfile(
            function_name="slow_func",
            file_path="test.py",
            language="python",
            total_time_ms=200.0,
            stages=[
                StageProfile(
                    stage=ProfileStage.Z3_SOLVING,
                    duration_ms=180.0,
                    memory_mb=50.0,
                    cpu_percent=95.0,
                    constraint_count=500,
                ),
            ],
            complexity_score=45.0,
            line_count=100,
            bottlenecks=[],
            verified=True,
            timestamp=datetime.now(UTC),
        )
        bottlenecks = detector.detect_bottlenecks(profile)
        assert isinstance(bottlenecks, list)
        assert len(bottlenecks) >= 1  # Should detect timeout

    def test_optimization_advisor(self):
        from codeverify_core.verification_profiler import (
            BottleneckInfo,
            BottleneckType,
            FunctionProfile,
            OptimizationAdvisor,
            ProfileStage,
            StageProfile,
        )

        advisor = OptimizationAdvisor()
        bottleneck = BottleneckInfo(
            bottleneck_type=BottleneckType.TIMEOUT,
            stage=ProfileStage.Z3_SOLVING,
            description="Z3 solver timed out",
            severity=0.9,
            suggested_fix="Reduce constraint count",
            estimated_speedup=0.5,
        )
        profile = FunctionProfile(
            function_name="func",
            file_path="test.py",
            language="python",
            total_time_ms=5000.0,
            stages=[
                StageProfile(
                    stage=ProfileStage.Z3_SOLVING,
                    duration_ms=4500.0,
                    memory_mb=100.0,
                    cpu_percent=90.0,
                )
            ],
            complexity_score=30.0,
            line_count=50,
            bottlenecks=[bottleneck],
            verified=False,
            timestamp=datetime.now(UTC),
        )
        recommendations = advisor.recommend([bottleneck], profile)
        assert len(recommendations) >= 1

    def test_budget_allocator(self):
        from codeverify_core.verification_profiler import BudgetAllocator

        allocator = BudgetAllocator(total_budget_ms=10000)
        files = [
            {"file_path": "main.py", "language": "python", "line_count": 100},
            {"file_path": "utils.py", "language": "python", "line_count": 50},
        ]
        allocations = allocator.allocate(files, [])
        assert len(allocations) == 2
        assert all(a.allocated_time_ms > 0 for a in allocations)

    def test_verification_profiler_report(self):
        from codeverify_core.verification_profiler import (
            FunctionProfile,
            ProfileStage,
            StageProfile,
            VerificationProfiler,
        )

        profiler = VerificationProfiler()
        profiles = [
            FunctionProfile(
                function_name=f"func_{i}",
                file_path=f"file_{i}.py",
                language="python",
                total_time_ms=100.0 * (i + 1),
                stages=[
                    StageProfile(
                        stage=ProfileStage.PARSING,
                        duration_ms=10.0,
                        memory_mb=5.0,
                        cpu_percent=10.0,
                    )
                ],
                complexity_score=10.0 * i,
                line_count=20 * (i + 1),
                bottlenecks=[],
                verified=True,
                timestamp=datetime.now(UTC),
            )
            for i in range(5)
        ]
        report = profiler.generate_report("test-project", profiles)
        assert report.project_name == "test-project"
        assert report.total_functions_profiled == 5
        assert report.avg_verification_time_ms > 0

    def test_performance_trends(self):
        from codeverify_core.verification_profiler import (
            FunctionProfile,
            ProfileStage,
            StageProfile,
            VerificationProfiler,
        )

        profiler = VerificationProfiler()
        profiles = [
            FunctionProfile(
                function_name="func",
                file_path="test.py",
                language="python",
                total_time_ms=100.0,
                stages=[
                    StageProfile(
                        stage=ProfileStage.TOTAL,
                        duration_ms=100.0,
                        memory_mb=10.0,
                        cpu_percent=50.0,
                    )
                ],
                complexity_score=10.0,
                line_count=20,
                bottlenecks=[],
                verified=True,
                timestamp=datetime.now(UTC),
            )
        ]
        trends = profiler.get_trends(profiles, periods=3)
        assert isinstance(trends, list)

    def test_profile_to_dict(self):
        from codeverify_core.verification_profiler import (
            FunctionProfile,
            ProfileStage,
            StageProfile,
        )

        profile = FunctionProfile(
            function_name="func",
            file_path="test.py",
            language="python",
            total_time_ms=50.0,
            stages=[
                StageProfile(
                    stage=ProfileStage.PARSING,
                    duration_ms=50.0,
                    memory_mb=5.0,
                    cpu_percent=10.0,
                )
            ],
            complexity_score=5.0,
            line_count=10,
            bottlenecks=[],
            verified=True,
            timestamp=datetime.now(UTC),
        )
        d = profile.to_dict()
        assert d["function_name"] == "func"
        assert d["total_time_ms"] == 50.0
