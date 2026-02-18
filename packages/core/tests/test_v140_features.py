"""Tests for v1.4.0 next-gen features.

Covers all 10 next-gen features:
1. Agentic Review Orchestrator
2. Verification-Aware Code Generation
3. Privacy-Preserving Federated Verification
4. Live Verification Debugger
5. AI Drift & Regression Monitor
6. Spec-First Development Workflow
7. Multi-Language Polyglot Bridge
8. Organizational Learning Engine
9. Verification Cost Optimizer (Smart Router)
10. Embeddable Verification Widget
"""

import pytest


# --- Feature 1: Agentic Review Orchestrator ---


class TestAgenticOrchestrator:
    def test_planner_creates_tasks(self):
        from codeverify_core.agentic_orchestrator import PlannerAgent, PRContext

        planner = PlannerAgent()
        ctx = PRContext(
            pr_id="123", repo="myorg/myrepo",
            changed_files=[{"path": "app.py"}, {"path": "utils.ts"}],
        )
        plan = planner.create_plan(ctx)
        assert plan.task_count >= 2  # semantic + security at minimum
        assert plan.total_estimated_cost_cents > 0

    def test_budget_constraint(self):
        from codeverify_core.agentic_orchestrator import PlannerAgent, PRContext, TaskStatus

        planner = PlannerAgent()
        ctx = PRContext(
            changed_files=[{"path": f"file{i}.py"} for i in range(20)],
        )
        plan = planner.create_plan(ctx, budget_cents=5.0)
        skipped = [t for t in plan.tasks if t.status == TaskStatus.SKIPPED]
        active = [t for t in plan.tasks if t.status != TaskStatus.SKIPPED]
        total_active_cost = sum(t.estimated_cost_cents for t in active)
        assert total_active_cost <= 5.0

    def test_full_review(self):
        from codeverify_core.agentic_orchestrator import (
            AgenticReviewOrchestrator, PRContext,
        )

        orch = AgenticReviewOrchestrator(budget_cents=100.0)
        ctx = PRContext(
            pr_id="42", repo="org/repo",
            changed_files=[{"path": "main.py"}, {"path": "auth.py"}],
        )
        result = orch.review(ctx)
        assert result.tasks_completed >= 2
        assert len(result.execution_trace) >= 3  # plan + executes + resolve

    def test_conflict_resolution(self):
        from codeverify_core.agentic_orchestrator import (
            AgentFinding, ConflictResolver, ConflictStrategy, TaskType,
        )

        resolver = ConflictResolver()
        f1 = AgentFinding(agent_type=TaskType.SEMANTIC_ANALYSIS, file_path="a.py",
                          line=10, severity="high", category="null", confidence=0.9)
        f2 = AgentFinding(agent_type=TaskType.SECURITY_SCAN, file_path="a.py",
                          line=10, severity="medium", category="null", confidence=0.7)
        resolved, conflicts = resolver.resolve([f1, f2], ConflictStrategy.CONFIDENCE_WEIGHTED)
        assert len(resolved) <= 2

    def test_circuit_breaker(self):
        from codeverify_core.agentic_orchestrator import CircuitBreaker, CircuitState, TaskType

        cb = CircuitBreaker(failure_threshold=2)
        assert cb.is_available(TaskType.SEMANTIC_ANALYSIS) is True
        cb.record_failure(TaskType.SEMANTIC_ANALYSIS)
        cb.record_failure(TaskType.SEMANTIC_ANALYSIS)
        assert cb.get_state(TaskType.SEMANTIC_ANALYSIS) == CircuitState.OPEN
        assert cb.is_available(TaskType.SEMANTIC_ANALYSIS) is False

    def test_security_label_boosts_priority(self):
        from codeverify_core.agentic_orchestrator import PlannerAgent, PRContext, TaskPriority, TaskType

        planner = PlannerAgent()
        ctx = PRContext(
            changed_files=[{"path": "app.py"}], labels=["security"],
        )
        plan = planner.create_plan(ctx)
        sec_tasks = [t for t in plan.tasks if t.task_type == TaskType.SECURITY_SCAN]
        assert any(t.priority == TaskPriority.CRITICAL for t in sec_tasks)


# --- Feature 2: Verification-Aware Code Generation ---


class TestVerifiedCodegen:
    def test_counterexample_to_constraints(self):
        from codeverify_core.verified_codegen_loop import ConstraintTranslator, Counterexample

        translator = ConstraintTranslator()
        ce = Counterexample(
            check_type="null_safety",
            variable_assignments={"x": "None"},
            constraint_violated="x must not be null",
        )
        constraints = translator.translate(ce)
        assert len(constraints) >= 1
        assert "None" in constraints[0].natural_language or "null" in constraints[0].natural_language

    def test_generate_verify_loop_success(self):
        from codeverify_core.verified_codegen_loop import (
            Counterexample, VerificationAwareCodeGenService,
        )

        svc = VerificationAwareCodeGenService()
        ce = Counterexample(
            check_type="division_by_zero",
            variable_assignments={"b": 0},
            constraint_violated="b must not be zero",
        )
        result = svc.generate_verified_fix("result = a / b", ce)
        assert result.success is True
        assert result.certificate is not None
        assert result.certificate.proof_strength.value == "full"
        assert result.iterations_used <= result.max_iterations

    def test_proof_certificate(self):
        from codeverify_core.verified_codegen_loop import Counterexample, VerificationAwareCodeGenService

        svc = VerificationAwareCodeGenService()
        ce = Counterexample(check_type="null_safety", variable_assignments={"obj": "None"})
        result = svc.generate_verified_fix("result = obj.method()", ce)
        if result.success and result.certificate:
            assert len(result.certificate.content_hash) == 16
            assert len(result.certificate.constraints_verified) > 0

    def test_cost_tracking(self):
        from codeverify_core.verified_codegen_loop import Counterexample, VerificationAwareCodeGenService

        svc = VerificationAwareCodeGenService()
        ce = Counterexample(check_type="division_by_zero", variable_assignments={"b": 0})
        result = svc.generate_verified_fix("x / b", ce)
        assert result.total_tokens > 0
        assert result.total_cost_cents > 0

    def test_all_attempts_tracked(self):
        from codeverify_core.verified_codegen_loop import Counterexample, VerificationAwareCodeGenService

        svc = VerificationAwareCodeGenService(max_iterations=3)
        ce = Counterexample(check_type="division_by_zero", variable_assignments={"b": 0})
        result = svc.generate_verified_fix("a / b", ce)
        assert len(result.all_attempts) >= 1


# --- Feature 3: Privacy-Preserving Federated Verification ---


class TestFederatedVerification:
    def test_privacy_budget(self):
        from codeverify_core.federated_verification import PrivacyBudget

        budget = PrivacyBudget(total_epsilon=1.0)
        assert budget.remaining_epsilon == 1.0
        assert budget.consume(0.3) is True
        assert round(budget.remaining_epsilon, 1) == 0.7
        assert budget.queries_made == 1

    def test_laplace_noise(self):
        from codeverify_core.federated_verification import LaplaceMechanism

        mech = LaplaceMechanism()
        noisy = mech.add_noise(100.0, 1.0, 0.1)
        assert noisy != 100.0  # Should have noise added
        assert isinstance(noisy, float)

    def test_federated_round(self):
        from codeverify_core.federated_verification import FederatedVerificationService

        svc = FederatedVerificationService()
        for org in ["org_a", "org_b", "org_c"]:
            svc.register_org(org, epsilon_budget=5.0)

        svc.start_round()
        for org in ["org_a", "org_b", "org_c"]:
            findings = [
                {"category": "null_safety", "confidence": 0.9},
                {"category": "null_safety", "confidence": 0.8},
                {"category": "bounds", "confidence": 0.7},
            ]
            patterns = svc.contribute(org, findings)
            assert len(patterns) > 0

        aggregated = svc.complete_round()
        assert len(aggregated) >= 1  # null_safety should aggregate

    def test_privacy_status(self):
        from codeverify_core.federated_verification import FederatedVerificationService

        svc = FederatedVerificationService()
        svc.register_org("test_org", epsilon_budget=2.0)
        status = svc.get_privacy_status("test_org")
        assert status["epsilon_total"] == 2.0
        assert status["epsilon_remaining"] == 2.0

    def test_pattern_adoption(self):
        from codeverify_core.federated_verification import FederatedVerificationService

        svc = FederatedVerificationService()
        for org in ["a", "b", "c"]:
            svc.register_org(org, 5.0)
        svc.start_round()
        for org in ["a", "b", "c"]:
            svc.contribute(org, [{"category": "xss", "confidence": 0.9}])
        patterns = svc.complete_round()
        if patterns:
            assert svc.adopt_pattern("a", patterns[0].id) is True


# --- Feature 4: Live Verification Debugger ---


class TestLiveDebugger:
    def test_create_session(self):
        from codeverify_core.live_debugger import LiveVerificationDebuggerService

        svc = LiveVerificationDebuggerService()
        session = svc.create_session(
            "Test Proof",
            ["x >= 0", "y != 0", "x < 100"],
            {"x": 5, "y": 3},
        )
        assert session.total_steps > 0
        assert session.share_url.startswith("https://")

    def test_step_through(self):
        from codeverify_core.live_debugger import LiveVerificationDebuggerService

        svc = LiveVerificationDebuggerService()
        session = svc.create_session("Test", ["a > 0"], {"a": 5})
        step = svc.step_forward(session.id)
        assert step is not None
        assert step.step_number > 0

    def test_variable_override(self):
        from codeverify_core.live_debugger import LiveVerificationDebuggerService, NodeStatus

        svc = LiveVerificationDebuggerService()
        session = svc.create_session("Test", ["x >= 0"], {"x": 5})
        assert session.root_node.children[0].status == NodeStatus.SATISFIED

        updated = svc.override_variable(session.id, "x", -1)
        assert updated.root_node.children[0].status == NodeStatus.VIOLATED

    def test_export_mermaid(self):
        from codeverify_core.live_debugger import LiveVerificationDebuggerService

        svc = LiveVerificationDebuggerService()
        session = svc.create_session("Test", ["x >= 0", "y != 0"], {"x": 5, "y": 3})
        from codeverify_core.live_debugger import ExportFormat
        mermaid = svc.export(session.id, ExportFormat.MERMAID)
        assert "graph TD" in mermaid

    def test_proof_summary(self):
        from codeverify_core.live_debugger import LiveVerificationDebuggerService

        svc = LiveVerificationDebuggerService()
        session = svc.create_session("Test", ["x >= 0", "y != 0"], {"x": 5, "y": 3})
        summary = svc.get_summary(session.id)
        assert summary.total_nodes >= 3  # root + 2 children
        assert summary.constraint_count == 2

    def test_share_token_lookup(self):
        from codeverify_core.live_debugger import LiveVerificationDebuggerService

        svc = LiveVerificationDebuggerService()
        session = svc.create_session("Shared", ["a > 0"], {"a": 1})
        found = svc.get_session_by_token(session.share_token)
        assert found is not None
        assert found.id == session.id


# --- Feature 5: AI Drift & Regression Monitor ---


class TestDriftMonitor:
    def test_fingerprint_extraction(self):
        from codeverify_core.drift_monitor import FingerprintExtractor

        extractor = FingerprintExtractor()
        code = "def hello(name: str):\n    return f'Hello {name}'\n\ndef add(a, b):\n    return a + b\n"
        fps = extractor.extract("app.py", code)
        assert len(fps) == 2
        assert fps[0].function_name == "hello"
        assert fps[0].content_hash != ""

    def test_drift_detection(self):
        from codeverify_core.drift_monitor import DriftMonitorService

        svc = DriftMonitorService()
        old_code = {"app.py": "def calc(x):\n    return x * 2\n"}
        new_code = {"app.py": "def calc(x):\n    return x * 3\n    # changed behavior\n"}

        svc.set_baseline("repo", old_code)
        report = svc.scan("repo", new_code, commit_sha="abc123")
        assert report.functions_checked >= 1
        assert report.functions_drifted >= 1
        assert len(report.alerts) >= 1

    def test_invariant_monitoring(self):
        from codeverify_core.drift_monitor import DriftMonitorService, VerifiedInvariant

        svc = DriftMonitorService()
        svc.register_invariant("repo", VerifiedInvariant(
            function_name="validate_input",
            file_path="app.py",
            invariant_text="input must be validated",
        ))
        # Function removed → invariant violated
        report = svc.scan("repo", {"app.py": "def other_func(): pass\n"})
        inv_alerts = [a for a in report.alerts if a.drift_type.value == "invariant_violation"]
        assert len(inv_alerts) >= 1

    def test_no_drift_on_identical_code(self):
        from codeverify_core.drift_monitor import DriftMonitorService

        svc = DriftMonitorService()
        code = {"app.py": "def stable(x):\n    return x + 1\n"}
        svc.set_baseline("repo", code)
        report = svc.scan("repo", code)
        assert report.functions_drifted == 0

    def test_signature_change_detection(self):
        from codeverify_core.drift_monitor import DriftMonitorService, DriftType

        svc = DriftMonitorService()
        old = {"app.py": "def func(a, b):\n    return a + b\n"}
        new = {"app.py": "def func(a, b, c):\n    return a + b + c\n"}
        svc.set_baseline("repo", old)
        report = svc.scan("repo", new)
        sig_alerts = [a for a in report.alerts if a.drift_type == DriftType.SIGNATURE_CHANGE]
        assert len(sig_alerts) >= 1


# --- Feature 6: Spec-First Development Workflow ---


class TestSpecFirst:
    def test_spec_parsing(self):
        from codeverify_core.spec_first import SpecFirstService, SpecType

        svc = SpecFirstService()
        spec_content = """source: app.py
function divide:
  @requires divisor must not be zero
  @ensures result is defined
function get_user:
  @requires user_id must not be none
"""
        sf = svc.load_spec("app.spec.cv", spec_content)
        assert len(sf.specs) == 3
        assert sf.specs[0].spec_type == SpecType.PRECONDITION
        assert sf.specs[0].target_function == "divide"

    def test_spec_compilation(self):
        from codeverify_core.spec_first import SpecCompiler, SpecStatus, Specification, SpecType

        compiler = SpecCompiler()
        spec = Specification(
            spec_type=SpecType.PRECONDITION,
            natural_language="x must be positive",
            variables=["x"],
        )
        compiled = compiler.compile(spec)
        assert compiled.status == SpecStatus.COMPILED
        assert "(assert" in compiled.z3_assertion

    def test_spec_verification(self):
        from codeverify_core.spec_first import SpecFirstService

        svc = SpecFirstService()
        svc.load_spec("t.spec.cv", "source: t.py\nfunction calc:\n  @requires x must not be none\n")
        code = "def calc(x):\n    if x is not None:\n        return x * 2\n"
        results = svc.verify_code("t.spec.cv", code)
        assert len(results) >= 1
        assert results[0].passed is True

    def test_spec_auto_generation(self):
        from codeverify_core.spec_first import SpecAutoGenerator

        gen = SpecAutoGenerator()
        code = "def divide(a: int, b: int) -> float:\n    return a / b\n"
        generated = gen.generate("divide", code)
        assert len(generated.specs) >= 1
        assert generated.confidence > 0

    def test_spec_coverage(self):
        from codeverify_core.spec_first import SpecFirstService

        svc = SpecFirstService()
        svc.load_spec("t.spec.cv", "function add:\n  @requires x must be positive\n")
        code = "def add(x): return x + 1\ndef sub(x): return x - 1\n"
        cov = svc.get_coverage(code)
        assert cov.total_functions == 2
        assert cov.functions_with_specs == 1
        assert cov.coverage_percent == 50.0


# --- Feature 7: Multi-Language Polyglot Bridge ---


class TestPolyglotBridge:
    def test_python_contract_extraction(self):
        from codeverify_core.polyglot_bridge import BridgeLanguage, ContractExtractor

        extractor = ContractExtractor()
        code = "def get_user(user_id: int) -> str:\n    pass\n"
        contracts = extractor.extract("api", code, BridgeLanguage.PYTHON)
        assert len(contracts) == 1
        assert contracts[0].endpoint == "get_user"
        assert contracts[0].parameters[0].type_name == "int"

    def test_typescript_contract_extraction(self):
        from codeverify_core.polyglot_bridge import BridgeLanguage, ContractExtractor

        extractor = ContractExtractor()
        code = "export function getUser(userId: number): string { return ''; }\n"
        contracts = extractor.extract("frontend", code, BridgeLanguage.TYPESCRIPT)
        assert len(contracts) == 1
        assert contracts[0].parameters[0].type_name == "number"

    def test_type_compatibility(self):
        from codeverify_core.polyglot_bridge import TypeChecker, TypeCompatibility

        checker = TypeChecker()
        assert checker.check("str", "string") == TypeCompatibility.COMPATIBLE
        assert checker.check("int", "number") == TypeCompatibility.COMPATIBLE
        assert checker.check("str", "int") == TypeCompatibility.INCOMPATIBLE
        assert checker.check("int", "float") == TypeCompatibility.COERCIBLE

    def test_cross_language_verification(self):
        from codeverify_core.polyglot_bridge import BridgeLanguage, PolyglotBridgeService

        svc = PolyglotBridgeService()
        py_code = "def get_user(user_id: int) -> str:\n    pass\n"
        ts_code = "export function get_user(user_id: number): string { return ''; }\n"

        svc.register_service("api", py_code, BridgeLanguage.PYTHON)
        svc.register_service("frontend", ts_code, BridgeLanguage.TYPESCRIPT,
                             depends_on=["api"])

        report = svc.verify_boundary("api", "frontend")
        assert report.pairs_verified >= 1
        assert report.compatible_pairs >= 1

    def test_type_mismatch_detection(self):
        from codeverify_core.polyglot_bridge import BridgeLanguage, MismatchSeverity, PolyglotBridgeService

        svc = PolyglotBridgeService()
        py_code = "def process(data: str) -> int:\n    pass\n"
        ts_code = "export function process(data: boolean): string { return ''; }\n"

        svc.register_service("svc_a", py_code, BridgeLanguage.PYTHON)
        svc.register_service("svc_b", ts_code, BridgeLanguage.TYPESCRIPT)

        report = svc.verify_boundary("svc_a", "svc_b")
        breaking = [m for m in report.mismatches if m.severity == MismatchSeverity.BREAKING]
        assert len(breaking) >= 1

    def test_service_graph(self):
        from codeverify_core.polyglot_bridge import BridgeLanguage, PolyglotBridgeService

        svc = PolyglotBridgeService()
        svc.register_service("api", "def main(): pass", BridgeLanguage.PYTHON)
        svc.register_service("web", "function main() {}", BridgeLanguage.TYPESCRIPT,
                             depends_on=["api"])
        graph = svc.get_service_graph()
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1


# --- Feature 8: Organizational Learning Engine ---


class TestOrgLearning:
    def test_feedback_recording(self):
        from codeverify_core.org_learning import FindingFeedback, FeedbackType, OrgLearningService

        svc = OrgLearningService()
        svc.record_feedback(FindingFeedback(
            org_id="org1", finding_category="null_safety",
            finding_severity="high", rule_id="NS001",
            feedback_type=FeedbackType.ACCEPTED,
        ))
        profile = svc.get_profile("org1")
        assert profile.total_feedback == 1
        assert profile.rule_performance["NS001"].accepted == 1

    def test_false_positive_classifier(self):
        from codeverify_core.org_learning import FPClassifier

        clf = FPClassifier()
        features = {"historical_fp_rate": 0.8, "acceptance_rate": 0.1}
        prob = clf.predict_fp_probability(features)
        assert 0 <= prob <= 1

    def test_severity_calibration(self):
        from codeverify_core.org_learning import (
            FindingFeedback, FeedbackType, OrgLearningService,
        )

        svc = OrgLearningService()
        for _ in range(10):
            svc.record_feedback(FindingFeedback(
                org_id="org1", rule_id="R1", finding_severity="high",
                feedback_type=FeedbackType.FALSE_POSITIVE,
            ))
        profile = svc.train("org1")
        assert "R1" in profile.suppressed_rules  # 100% FP rate → suppressed

    def test_quality_prediction(self):
        from codeverify_core.org_learning import OrgLearningService, PredictionOutcome

        svc = OrgLearningService()
        pred = svc.predict_quality("auth.py", change_size=300)
        assert pred.risk_score > 0
        assert pred.outcome in list(PredictionOutcome)

    def test_training_updates_profile(self):
        from codeverify_core.org_learning import (
            FindingFeedback, FeedbackType, OrgLearningService,
        )

        svc = OrgLearningService()
        for i in range(5):
            svc.record_feedback(FindingFeedback(
                org_id="org2", rule_id="R2", finding_severity="medium",
                feedback_type=FeedbackType.ACCEPTED,
            ))
        profile = svc.train("org2")
        assert profile.last_trained is not None
        assert len(profile.severity_calibrations) >= 1


# --- Feature 9: Verification Cost Optimizer (Smart Router) ---


class TestCostOptimizer:
    def test_risk_scoring(self):
        from codeverify_core.smart_router import RiskBucket, RiskScorer

        scorer = RiskScorer()
        high_risk = scorer.score("auth/login.py", change_lines=200, is_new_file=True, author_commits=2)
        assert high_risk.risk_score > 0.5

        low_risk = scorer.score("tests/test_utils.py", change_lines=5, author_commits=100)
        assert low_risk.risk_score < high_risk.risk_score

    def test_depth_routing(self):
        from codeverify_core.smart_router import CostOptimizerService, VerificationDepth

        svc = CostOptimizerService(default_budget_cents=50.0)
        files = [
            {"path": "auth/crypto.py", "change_lines": 100, "is_new": True, "author_commits": 3},
            {"path": "tests/test_basic.py", "change_lines": 5, "author_commits": 100},
            {"path": "utils/helpers.py", "change_lines": 20, "author_commits": 50},
        ]
        decision = svc.optimize(files)
        assert len(decision.file_routes) == 3
        assert decision.total_estimated_cost_cents <= 50.0
        # High-risk file should get deeper analysis
        crypto_route = next(r for r in decision.file_routes if "crypto" in r.file_path)
        test_route = next(r for r in decision.file_routes if "test" in r.file_path)
        depth_order = {VerificationDepth.PATTERN: 0, VerificationDepth.STATIC: 1,
                       VerificationDepth.AI: 2, VerificationDepth.FORMAL: 3}
        assert depth_order[crypto_route.recommended_depth] >= depth_order[test_route.recommended_depth]

    def test_savings_estimation(self):
        from codeverify_core.smart_router import CostOptimizerService

        svc = CostOptimizerService()
        files = [{"path": f"file{i}.py", "change_lines": 10} for i in range(5)]
        decision = svc.optimize(files)
        savings = svc.estimate_savings(decision)
        assert savings["savings_percent"] > 0

    def test_budget_enforcement(self):
        from codeverify_core.smart_router import CostOptimizerService

        svc = CostOptimizerService()
        files = [{"path": f"f{i}.py", "change_lines": 100, "is_new": True, "author_commits": 1}
                 for i in range(50)]
        decision = svc.optimize(files, budget_cents=10.0)
        assert decision.total_estimated_cost_cents <= 10.0

    def test_routing_stats(self):
        from codeverify_core.smart_router import CostOptimizerService

        svc = CostOptimizerService()
        for _ in range(3):
            svc.optimize([{"path": "app.py", "change_lines": 50}])
        stats = svc.get_stats()
        assert stats.total_files_routed == 3


# --- Feature 10: Embeddable Verification Widget ---


class TestEmbedWidget:
    def test_widget_creation(self):
        from codeverify_core.embed_widget import EmbeddableWidgetService, WidgetType

        svc = EmbeddableWidgetService()
        config = svc.create_widget(widget_type=WidgetType.BADGE, repo="org/repo")
        assert config.id != ""
        assert len(config.token) == 16

    def test_badge_rendering_svg(self):
        from codeverify_core.embed_widget import BadgeStatus, EmbeddableWidgetService, WidgetType

        svc = EmbeddableWidgetService()
        config = svc.create_widget(WidgetType.BADGE, "repo")
        svc.set_badge_data(config.id, BadgeStatus.PASSING, "3 checks passed")
        svg = svc.render_badge(config.id)
        assert "<svg" in svg
        assert "CodeVerify" in svg

    def test_badge_rendering_html(self):
        from codeverify_core.embed_widget import BadgeStatus, EmbeddableWidgetService, WidgetType

        svc = EmbeddableWidgetService()
        config = svc.create_widget(WidgetType.BADGE, "repo")
        svc.set_badge_data(config.id, BadgeStatus.FAILING, "2 issues")
        html = svc.render_badge_html(config.id)
        assert "CodeVerify" in html
        assert "#e05d44" in html  # failing color

    def test_embed_code_generation(self):
        from codeverify_core.embed_widget import EmbedFormat, EmbeddableWidgetService, WidgetType

        svc = EmbeddableWidgetService()
        config = svc.create_widget(WidgetType.BADGE, "repo")
        iframe = svc.get_embed_code(config.id, EmbedFormat.IFRAME)
        assert "<iframe" in iframe
        react = svc.get_embed_code(config.id, EmbedFormat.REACT)
        assert "CodeVerifyWidget" in react
        wc = svc.get_embed_code(config.id, EmbedFormat.WEB_COMPONENT)
        assert "codeverify-widget" in wc
        md = svc.get_embed_code(config.id, EmbedFormat.MARKDOWN)
        assert "![CodeVerify]" in md

    def test_trust_score_data(self):
        from codeverify_core.embed_widget import EmbeddableWidgetService, WidgetType

        svc = EmbeddableWidgetService()
        config = svc.create_widget(WidgetType.TRUST_SCORE, "repo")
        data = svc.set_trust_score_data(config.id, score=85.5, risk_level="low")
        assert data.score == 85.5

    def test_finding_summary_data(self):
        from codeverify_core.embed_widget import EmbeddableWidgetService, WidgetType

        svc = EmbeddableWidgetService()
        config = svc.create_widget(WidgetType.FINDING_SUMMARY, "repo")
        data = svc.set_finding_data(config.id, critical=1, high=3, medium=5, low=10)
        assert data.total == 19

    def test_widget_data_json(self):
        from codeverify_core.embed_widget import BadgeStatus, EmbeddableWidgetService, WidgetType

        svc = EmbeddableWidgetService()
        config = svc.create_widget(WidgetType.BADGE, "repo")
        svc.set_badge_data(config.id, BadgeStatus.PASSING)
        json_str = svc.get_widget_data_json(config.id)
        assert "passing" in json_str
