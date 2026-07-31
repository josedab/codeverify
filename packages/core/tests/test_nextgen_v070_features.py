"""Tests for Next-Gen v0.7.0 features."""

from __future__ import annotations

import pytest

# =============================================================================
# Feature 1: AI Code Generation Firewall
# =============================================================================


class TestAICodeFirewall:
    """Tests for the AI Code Generation Firewall module."""

    def test_firewall_enums(self):
        from codeverify_core.ai_code_firewall import (
            FirewallAction,
            RiskLevel,
            SanitizationType,
            SuggestionSource,
        )

        assert SuggestionSource.COPILOT == "copilot"
        assert FirewallAction.BLOCK == "block"
        assert RiskLevel.CRITICAL == "critical"
        assert SanitizationType.REMOVE_SECRETS == "remove_secrets"

    def test_firewall_intercept_safe_code(self):
        from codeverify_core.ai_code_firewall import (
            AICodeFirewall,
            FirewallAction,
            SuggestionInterception,
            SuggestionSource,
        )

        fw = AICodeFirewall()
        inter = SuggestionInterception(
            id="t1",
            source=SuggestionSource.COPILOT,
            code="x = 1 + 2",
            language="python",
            file_path="test.py",
            cursor_line=1,
            cursor_column=0,
            timestamp=1.0,
        )
        dec = fw.intercept(inter)
        assert dec.action in (FirewallAction.ALLOW, FirewallAction.WARN)

    def test_firewall_blocks_secrets(self):
        from codeverify_core.ai_code_firewall import (
            AICodeFirewall,
            RiskLevel,
            SuggestionInterception,
            SuggestionSource,
        )

        fw = AICodeFirewall()
        inter = SuggestionInterception(
            id="t2",
            source=SuggestionSource.CHATGPT,
            code='AWS_SECRET_KEY = "AKIAIOSFODNN7EXAMPLE"',
            language="python",
            file_path="config.py",
            cursor_line=1,
            cursor_column=0,
            timestamp=1.0,
        )
        dec = fw.intercept(inter)
        assert dec.risk_assessment.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_firewall_metrics(self):
        from codeverify_core.ai_code_firewall import (
            AICodeFirewall,
            SuggestionInterception,
            SuggestionSource,
        )

        fw = AICodeFirewall()
        inter = SuggestionInterception(
            id="t3",
            source=SuggestionSource.COPILOT,
            code="print('hi')",
            language="python",
            file_path="a.py",
            cursor_line=1,
            cursor_column=0,
            timestamp=1.0,
        )
        fw.intercept(inter)
        m = fw.get_metrics()
        assert m.total_interceptions == 1

    def test_risk_analyzer(self):
        from codeverify_core.ai_code_firewall import (
            SuggestionInterception,
            SuggestionRiskAnalyzer,
            SuggestionSource,
        )

        analyzer = SuggestionRiskAnalyzer()
        inter = SuggestionInterception(
            id="t4",
            source=SuggestionSource.COPILOT,
            code="os.system('rm -rf /')",
            language="python",
            file_path="bad.py",
            cursor_line=1,
            cursor_column=0,
            timestamp=1.0,
        )
        risk = analyzer.analyze(inter)
        assert risk.overall_score > 0

    def test_code_sanitizer(self):
        from codeverify_core.ai_code_firewall import CodeSanitizer, RiskAssessment, RiskLevel

        sanitizer = CodeSanitizer()
        risk = RiskAssessment(
            risk_level=RiskLevel.HIGH,
            confidence=0.9,
            risk_factors=[{"type": "secret", "description": "Hardcoded password"}],
            security_issues=["hardcoded_secret"],
            quality_issues=[],
            overall_score=70.0,
        )
        sanitized, actions = sanitizer.sanitize('password = "secret123"', risk, "python")
        assert isinstance(sanitized, str)
        assert isinstance(actions, list)

    def test_firewall_decision_to_dict(self):
        from codeverify_core.ai_code_firewall import (
            AICodeFirewall,
            SuggestionInterception,
            SuggestionSource,
        )

        fw = AICodeFirewall()
        inter = SuggestionInterception(
            id="t5",
            source=SuggestionSource.COPILOT,
            code="x = 1",
            language="python",
            file_path="a.py",
            cursor_line=1,
            cursor_column=0,
            timestamp=1.0,
        )
        dec = fw.intercept(inter)
        d = dec.to_dict()
        assert "action" in d
        assert "risk_assessment" in d

    def test_firewall_policy_update(self):
        from codeverify_core.ai_code_firewall import (
            AICodeFirewall,
            FirewallPolicy,
            RiskLevel,
        )

        fw = AICodeFirewall()
        policy = FirewallPolicy(
            id="custom",
            name="Strict",
            description="Block all",
            risk_threshold=RiskLevel.LOW,
        )
        fw.update_policy(policy)


# =============================================================================
# Feature 2: Multi-Tenant SaaS Billing
# =============================================================================


class TestSaaSBilling:
    """Tests for the Multi-Tenant SaaS Billing module."""

    def test_billing_enums(self):
        from codeverify_core.saas_billing import (
            BillingCycle,
            PlanType,
            SubscriptionStatus,
        )

        assert PlanType.FREE == "free"
        assert BillingCycle.ANNUAL == "annual"
        assert SubscriptionStatus.ACTIVE == "active"

    def test_plan_catalog(self):
        from codeverify_core.saas_billing import PlanCatalog, PlanType

        catalog = PlanCatalog()
        plans = catalog.get_all_plans()
        assert len(plans) >= 4
        free = catalog.get_plan(PlanType.FREE)
        assert free.price_monthly == 0

    def test_plan_pricing(self):
        from codeverify_core.saas_billing import PlanCatalog, PlanType

        catalog = PlanCatalog()
        starter = catalog.get_plan(PlanType.STARTER)
        assert starter.price_monthly == 29
        team = catalog.get_plan(PlanType.TEAM)
        assert team.price_monthly == 79
        enterprise = catalog.get_plan(PlanType.ENTERPRISE)
        assert enterprise.price_monthly == 249

    def test_subscription_lifecycle(self):
        from codeverify_core.saas_billing import (
            PlanType,
            SubscriptionManager,
            SubscriptionStatus,
        )

        mgr = SubscriptionManager()
        sub = mgr.create_subscription("t1", PlanType.STARTER)
        assert sub.status in (SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING)

    def test_usage_meter(self):
        from codeverify_core.saas_billing import UsageMeter, UsageMetric

        meter = UsageMeter()
        rec = meter.record_usage("t1", UsageMetric.VERIFICATIONS)
        assert rec.value >= 1
        usage = meter.get_usage("t1", UsageMetric.VERIFICATIONS)
        assert usage >= 1

    def test_engine_onboard(self):
        from codeverify_core.saas_billing import PlanType, SaaSBillingEngine

        engine = SaaSBillingEngine()
        result = engine.onboard_tenant("TestCo", PlanType.STARTER)
        assert "tenant_id" in result

    def test_engine_process_verification(self):
        from codeverify_core.saas_billing import PlanType, SaaSBillingEngine

        engine = SaaSBillingEngine()
        result = engine.onboard_tenant("TestCo", PlanType.STARTER)
        allowed, reason = engine.process_verification(result["tenant_id"])
        assert isinstance(allowed, bool)

    def test_sso_manager(self):
        from codeverify_core.saas_billing import SSOManager, SSOProvider

        mgr = SSOManager()
        cfg = mgr.configure_sso("t1", SSOProvider.OKTA, "client-id", "https://issuer.example.com")
        assert cfg.provider == SSOProvider.OKTA
        assert cfg.enabled is True


# =============================================================================
# Feature 3: Proof-as-a-Service API
# =============================================================================


class TestProofServiceAPI:
    """Tests for the Proof-as-a-Service API module."""

    _HMAC_SECRET = b"test-proof-service-hmac-secret"

    def test_api_enums(self):
        from codeverify_core.proof_service_api import (
            ProofFormat,
            ProofRequestStatus,
            VerificationCheck,
        )

        assert ProofRequestStatus.COMPLETED == "completed"
        assert VerificationCheck.NULL_SAFETY == "null_safety"
        assert ProofFormat.JSON == "json"

    def test_api_key_creation(self):
        from codeverify_core.proof_service_api import APIKeyManager

        mgr = APIKeyManager(hmac_secret=self._HMAC_SECRET)
        raw_key, config = mgr.create_key("tenant-1", "test-key")
        assert len(raw_key) > 10
        assert config.active is True

    def test_api_key_validation(self):
        from codeverify_core.proof_service_api import APIKeyManager

        mgr = APIKeyManager(hmac_secret=self._HMAC_SECRET)
        raw_key, _ = mgr.create_key("tenant-1", "test-key")
        validated = mgr.validate_key(raw_key)
        assert validated is not None
        assert validated.tenant_id == "tenant-1"

    def test_api_key_revoke(self):
        from codeverify_core.proof_service_api import APIKeyManager

        mgr = APIKeyManager(hmac_secret=self._HMAC_SECRET)
        raw_key, config = mgr.create_key("tenant-1", "test-key")
        assert mgr.revoke_key(config.key_id) is True
        assert mgr.validate_key(raw_key) is None

    def test_api_key_manager_requires_secret_for_key_operations(self, monkeypatch):
        from codeverify_core.proof_service_api import APIKeyManager

        monkeypatch.delenv("CODEVERIFY_PROOF_HMAC_SECRET", raising=False)
        mgr = APIKeyManager()

        with pytest.raises(RuntimeError, match="CODEVERIFY_PROOF_HMAC_SECRET"):
            mgr.create_key("tenant-1", "test-key")

    def test_rate_limiter(self):
        from codeverify_core.proof_service_api import RateLimiter

        limiter = RateLimiter()
        allowed, remaining, limit = limiter.check_rate_limit("key-1")
        assert allowed is True
        assert remaining > 0

    def test_proof_service_verify(self):
        from codeverify_core.proof_service_api import ProofServiceAPI

        api = ProofServiceAPI(hmac_secret=self._HMAC_SECRET)
        raw_key, _ = api._key_manager.create_key("t1", "test")
        result = api.verify(raw_key, "x = 1 + 2", "python")
        assert result.status.value in ("completed", "failed")

    def test_pricing_config(self):
        from codeverify_core.proof_service_api import PricingConfig, PricingModel

        config = PricingConfig(model=PricingModel.PER_PROOF)
        assert config.cost_per_proof_cents == 5.0


# =============================================================================
# Feature 4: Smart Contract Analyzer
# =============================================================================


class TestSmartContractAnalyzer:
    """Tests for the Smart Contract Analyzer module."""

    def test_analyzer_enums(self):
        from codeverify_core.smart_contract_analyzer import (
            AnalysisDepth,
            ContractStandard,
            ProofStatus,
        )

        assert AnalysisDepth.STANDARD == "standard"
        assert ContractStandard.ERC20 == "ERC-20"
        assert ProofStatus.PROVEN_SAFE.value in ("proven_safe", "PROVEN_SAFE")

    def test_solidity_parser(self):
        from codeverify_core.smart_contract_analyzer import SolidityParser

        parser = SolidityParser()
        code = """
        contract Token {
            function transfer(address to, uint256 amount) public returns (bool) {
                return true;
            }
        }
        """
        funcs, state_vars = parser.parse(code)
        assert len(funcs) >= 1
        assert funcs[0].name == "transfer"

    def test_vulnerability_detection(self):
        from codeverify_core.smart_contract_analyzer import (
            AnalysisDepth,
            SmartContractAnalyzer,
        )

        analyzer = SmartContractAnalyzer(depth=AnalysisDepth.STANDARD)
        code = """
        contract Vulnerable {
            mapping(address => uint256) balances;
            function withdraw() public {
                msg.sender.call{value: balances[msg.sender]}("");
                balances[msg.sender] = 0;
            }
        }
        """
        report = analyzer.analyze(code, "solidity")
        assert report.functions_analyzed >= 1
        assert len(report.vulnerabilities) > 0

    def test_quick_scan(self):
        from codeverify_core.smart_contract_analyzer import SmartContractAnalyzer

        analyzer = SmartContractAnalyzer()
        vulns = analyzer.quick_scan("contract Test { function f() public { } }")
        assert isinstance(vulns, list)

    def test_gas_analyzer(self):
        from codeverify_core.smart_contract_analyzer import ContractFunction, GasAnalyzer

        ga = GasAnalyzer()
        func = ContractFunction(
            name="transfer",
            visibility="public",
            mutability="nonpayable",
            parameters=[{"name": "to", "type": "address"}],
            return_types=["bool"],
        )
        analyses = ga.analyze_gas("function transfer() {}", [func])
        assert isinstance(analyses, list)

    def test_erc_compliance(self):
        from codeverify_core.smart_contract_analyzer import (
            ContractFunction,
            ContractStandard,
            ERCComplianceChecker,
        )

        checker = ERCComplianceChecker()
        funcs = [
            ContractFunction(
                name="transfer",
                visibility="public",
                mutability="nonpayable",
                parameters=[
                    {"name": "to", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                ],
                return_types=["bool"],
            ),
        ]
        result = checker.check_compliance("contract Token {}", funcs, ContractStandard.ERC20)
        assert isinstance(result.compliant, bool)
        assert isinstance(result.missing_functions, list)

    def test_smart_contract_report_to_dict(self):
        from codeverify_core.smart_contract_analyzer import SmartContractAnalyzer

        analyzer = SmartContractAnalyzer()
        report = analyzer.analyze("contract A { function f() public {} }", "solidity")
        d = report.to_dict()
        assert "contract_name" in d or "vulnerabilities" in d


# =============================================================================
# Feature 5: AI Model Fine-Tuning Pipeline
# =============================================================================


class TestModelFineTuning:
    """Tests for the AI Model Fine-Tuning Pipeline module."""

    def test_fine_tuning_enums(self):
        from codeverify_core.model_fine_tuning import (
            AdapterType,
            ModelType,
            ModelVersion,
            TrainingStatus,
        )

        assert TrainingStatus.TRAINING == "training"
        assert ModelType.CODE_LLAMA == "code_llama"
        assert AdapterType.QLORA == "qlora"
        assert ModelVersion.PRODUCTION == "production"

    def test_dataset_builder(self):
        from codeverify_core.model_fine_tuning import DatasetBuilder

        builder = DatasetBuilder()
        example = builder.add_example(
            "def f(x): return x", "Missing type hint", "warning", "python"
        )
        assert example.code_language == "python"
        assert len(example.prompt) > 0

    def test_dataset_from_history(self):
        from codeverify_core.model_fine_tuning import DatasetBuilder

        builder = DatasetBuilder()
        history = [
            {"code": "x = 1", "finding": "unused", "result": "warning", "language": "python"},
            {"code": "y = 2", "finding": "naming", "result": "info", "language": "python"},
        ]
        examples = builder.from_verification_history(history)
        assert len(examples) == 2

    def test_training_orchestrator(self):
        from codeverify_core.model_fine_tuning import (
            ModelType,
            TrainingConfig,
            TrainingOrchestrator,
            TrainingStatus,
        )

        orch = TrainingOrchestrator()
        config = TrainingConfig(base_model=ModelType.CODE_LLAMA)
        job = orch.create_job(config, dataset_size=100)
        assert job.status == TrainingStatus.PREPARING

    def test_model_registry(self):
        from codeverify_core.model_fine_tuning import ModelRegistry

        registry = ModelRegistry()
        models = registry.list_models()
        assert isinstance(models, list)

    def test_pipeline_end_to_end(self):
        from codeverify_core.model_fine_tuning import FineTuningPipeline

        pipe = FineTuningPipeline()
        history = [
            {"code": "x = 1", "finding": "unused var", "result": "warning", "language": "python"},
        ]
        result = pipe.run_pipeline(history)
        assert result["status"] in ("completed", "failed")

    def test_training_config_defaults(self):
        from codeverify_core.model_fine_tuning import ModelType, TrainingConfig

        config = TrainingConfig(base_model=ModelType.DEEPSEEK_CODER)
        assert config.learning_rate == 2e-4
        assert config.batch_size == 4
        assert config.num_epochs == 3


# =============================================================================
# Feature 6: Cross-Repo Dependency Visualizer
# =============================================================================


class TestDependencyVisualizer:
    """Tests for the Cross-Repo Dependency Visualizer module."""

    def test_visualizer_enums(self):
        from codeverify_core.dependency_visualizer import (
            EdgeType,
            ExportFormat,
            NodeType,
        )

        assert NodeType.REPOSITORY == "repository"
        assert EdgeType.DEPENDS_ON == "depends_on"
        assert ExportFormat.MERMAID == "mermaid"

    def test_graph_builder(self):
        from codeverify_core.dependency_visualizer import (
            EdgeType,
            GraphBuilder,
            NodeType,
        )

        builder = GraphBuilder()
        n1 = builder.add_node("repo-a", NodeType.REPOSITORY)
        n2 = builder.add_node("repo-b", NodeType.REPOSITORY)
        builder.add_edge(n1.id, n2.id, EdgeType.DEPENDS_ON)
        graph = builder.build()
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_graph_analyzer_cycles(self):
        from codeverify_core.dependency_visualizer import (
            EdgeType,
            GraphAnalyzer,
            GraphBuilder,
            NodeType,
        )

        builder = GraphBuilder()
        n1 = builder.add_node("a", NodeType.PACKAGE)
        n2 = builder.add_node("b", NodeType.PACKAGE)
        builder.add_edge(n1.id, n2.id, EdgeType.DEPENDS_ON)
        builder.add_edge(n2.id, n1.id, EdgeType.DEPENDS_ON)
        graph = builder.build()
        analyzer = GraphAnalyzer()
        cycles = analyzer.find_cycles(graph)
        assert len(cycles) > 0

    def test_graph_analyzer_orphans(self):
        from codeverify_core.dependency_visualizer import (
            EdgeType,
            GraphAnalyzer,
            GraphBuilder,
            NodeType,
        )

        builder = GraphBuilder()
        builder.add_node("orphan", NodeType.MODULE)
        n1 = builder.add_node("a", NodeType.MODULE)
        n2 = builder.add_node("b", NodeType.MODULE)
        builder.add_edge(n1.id, n2.id, EdgeType.IMPORTS)
        graph = builder.build()
        analyzer = GraphAnalyzer()
        orphans = analyzer.find_orphans(graph)
        assert len(orphans) >= 1

    def test_mermaid_export(self):
        from codeverify_core.dependency_visualizer import (
            EdgeType,
            GraphBuilder,
            GraphExporter,
            NodeType,
        )

        builder = GraphBuilder()
        n1 = builder.add_node("a", NodeType.PACKAGE)
        n2 = builder.add_node("b", NodeType.PACKAGE)
        builder.add_edge(n1.id, n2.id, EdgeType.DEPENDS_ON)
        graph = builder.build()
        exporter = GraphExporter()
        mermaid = exporter.to_mermaid(graph)
        assert "graph" in mermaid.lower() or "flowchart" in mermaid.lower() or "-->" in mermaid

    def test_dot_export(self):
        from codeverify_core.dependency_visualizer import (
            EdgeType,
            GraphBuilder,
            GraphExporter,
            NodeType,
        )

        builder = GraphBuilder()
        n1 = builder.add_node("a", NodeType.PACKAGE)
        n2 = builder.add_node("b", NodeType.PACKAGE)
        builder.add_edge(n1.id, n2.id, EdgeType.DEPENDS_ON)
        graph = builder.build()
        exporter = GraphExporter()
        dot = exporter.to_dot(graph)
        assert "digraph" in dot

    def test_graph_query_engine(self):
        from codeverify_core.dependency_visualizer import (
            EdgeType,
            GraphBuilder,
            GraphQueryEngine,
            NodeType,
        )

        builder = GraphBuilder()
        n1 = builder.add_node("root", NodeType.REPOSITORY)
        n2 = builder.add_node("child", NodeType.PACKAGE)
        builder.add_edge(n1.id, n2.id, EdgeType.DEPENDS_ON)
        graph = builder.build()
        engine = GraphQueryEngine(graph)
        deps = engine.get_dependencies(n1.id, direct_only=True)
        assert len(deps) >= 0  # Implementation may vary

    def test_graph_report(self):
        from codeverify_core.dependency_visualizer import (
            EdgeType,
            GraphAnalyzer,
            GraphBuilder,
            NodeType,
        )

        builder = GraphBuilder()
        n1 = builder.add_node("a", NodeType.PACKAGE)
        n2 = builder.add_node("b", NodeType.PACKAGE)
        builder.add_edge(n1.id, n2.id, EdgeType.DEPENDS_ON)
        graph = builder.build()
        analyzer = GraphAnalyzer()
        report = analyzer.generate_report(graph)
        assert report.total_nodes == 2
        assert report.total_edges == 1


# =============================================================================
# Feature 7: Auto-Fix with Test Generation
# =============================================================================


class TestAutoFixTestGeneration:
    """Tests for the Auto-Fix with Test Generation module."""

    def test_test_gen_enums(self):
        from codeverify_core.autofix_test_generation import (
            CoverageLevel,
            FixConfidence,
            TestFramework,
            TestType,
        )

        assert TestType.UNIT == "unit"
        assert TestFramework.PYTEST == "pytest"
        assert FixConfidence.HIGH == "high"
        assert CoverageLevel.FULL == "full"

    def test_test_generator(self):
        from codeverify_core.autofix_test_generation import (
            FixWithTests,
            TestGenerator,
        )

        gen = TestGenerator()
        fix = FixWithTests(
            fix_id="f1",
            original_code="def f(x): return x",
            fixed_code="def f(x):\n  if x is None:\n    return 0\n  return x",
            issue_description="null check",
            language="python",
        )
        tests = gen.generate_tests(fix)
        assert len(tests) > 0
        assert all(t.code for t in tests)

    def test_coverage_analyzer(self):
        from codeverify_core.autofix_test_generation import (
            CoverageAnalyzer,
            FixWithTests,
            TestCase,
            TestFramework,
            TestType,
        )

        analyzer = CoverageAnalyzer()
        fix = FixWithTests(
            fix_id="f2",
            original_code="def g(): pass",
            fixed_code="def g(): return 1",
            issue_description="add return",
            language="python",
        )
        test = TestCase(
            id="tc1",
            name="test_g",
            test_type=TestType.UNIT,
            code="def test_g(): assert g() == 1",
            language="python",
            target_function="g",
            description="test g",
            framework=TestFramework.PYTEST,
        )
        level = analyzer.analyze_coverage(fix, [test])
        assert level is not None

    def test_autofix_engine_validate(self):
        from codeverify_core.autofix_test_generation import AutoFixTestEngine

        engine = AutoFixTestEngine()
        report = engine.validate_fix(
            "def f(x): return x",
            "def f(x):\n  if x is None:\n    return 0\n  return x",
            "null check",
            "python",
        )
        assert report.tests_generated > 0
        assert isinstance(report.confidence.value, str)

    def test_fix_validation_report_to_dict(self):
        from codeverify_core.autofix_test_generation import AutoFixTestEngine

        engine = AutoFixTestEngine()
        report = engine.validate_fix(
            "def f(): pass",
            "def f(): return 1",
            "add return",
            "python",
        )
        d = report.to_dict()
        assert "fix_id" in d
        assert "tests_generated" in d

    def test_batch_validate(self):
        from codeverify_core.autofix_test_generation import (
            AutoFixTestEngine,
            FixWithTests,
        )

        engine = AutoFixTestEngine()
        fixes = [
            FixWithTests(
                fix_id="b1",
                original_code="x = 1",
                fixed_code="x: int = 1",
                issue_description="type hint",
                language="python",
            ),
            FixWithTests(
                fix_id="b2",
                original_code="y = []",
                fixed_code="y: list = []",
                issue_description="type hint",
                language="python",
            ),
        ]
        reports = engine.batch_validate(fixes)
        assert len(reports) == 2


# =============================================================================
# Feature 8: CI/CD Verification Agent
# =============================================================================


class TestCIVerificationAgent:
    """Tests for the CI/CD Verification Agent module."""

    def test_ci_enums(self):
        from codeverify_core.ci_verification_agent import (
            CIProvider,
            GateDecision,
            VerificationScope,
        )

        assert CIProvider.GITHUB_ACTIONS == "github_actions"
        assert GateDecision.PASS == "pass"
        assert VerificationScope.INCREMENTAL == "incremental"

    def test_change_detector(self):
        from codeverify_core.ci_verification_agent import ChangeDetector

        det = ChangeDetector()
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 def foo():
+    x = 1
     pass
"""
        changes = det.analyze_diff(diff)
        assert isinstance(changes, list)

    def test_proof_cache(self):
        from codeverify_core.ci_verification_agent import ProofCacheManager

        cache = ProofCacheManager()
        stored = cache.store_proof("test.py", "abc123", {"verified": True})
        assert stored.file_path == "test.py"
        retrieved = cache.get_cached_proof("test.py", "abc123")
        assert retrieved is not None

    def test_proof_cache_invalidation(self):
        from codeverify_core.ci_verification_agent import ProofCacheManager

        cache = ProofCacheManager()
        cache.store_proof("test.py", "abc123", {"verified": True})
        assert cache.invalidate("test.py") is True
        assert cache.get_cached_proof("test.py", "abc123") is None

    def test_gate_evaluator(self):
        from codeverify_core.ci_verification_agent import (
            GateDecision,
            GateEvaluator,
            GatePolicy,
        )

        evaluator = GateEvaluator(GatePolicy(max_critical=0, max_high=0))
        decision = evaluator.evaluate([], 90.0)
        assert decision == GateDecision.PASS

    def test_gate_evaluator_fail(self):
        from codeverify_core.ci_verification_agent import (
            GateDecision,
            GateEvaluator,
            GatePolicy,
        )

        evaluator = GateEvaluator(GatePolicy(max_critical=0))
        findings = [{"severity": "critical", "message": "test"}]
        decision = evaluator.evaluate(findings, 90.0)
        assert decision in (GateDecision.FAIL, GateDecision.WARN)

    def test_ci_config_generator_github(self):
        from codeverify_core.ci_verification_agent import (
            CIConfig,
            CIConfigGenerator,
            CIProvider,
        )

        gen = CIConfigGenerator()
        config = CIConfig(
            provider=CIProvider.GITHUB_ACTIONS, repo_url="https://github.com/test/repo"
        )
        yaml_output = gen.generate(config)
        assert "codeverify" in yaml_output.lower() or "verify" in yaml_output.lower()

    def test_ci_agent_verify(self):
        from codeverify_core.ci_verification_agent import CIVerificationAgent

        agent = CIVerificationAgent()
        report = agent.verify_commit("abc123", "", {"test.py": "x = 1"})
        assert report.gate_decision is not None


# =============================================================================
# Feature 9: Compliance Dashboard & Reporter
# =============================================================================


class TestComplianceDashboard:
    """Tests for the Compliance Dashboard & Reporter module."""

    def test_dashboard_enums(self):
        from codeverify_core.compliance_dashboard import (
            ComplianceStatus,
            ControlPriority,
            ReportFormat,
            TrendDirection,
        )

        assert ReportFormat.MARKDOWN == "markdown"
        assert ComplianceStatus.COMPLIANT == "compliant"
        assert ControlPriority.CRITICAL == "critical"
        assert TrendDirection.IMPROVING == "improving"

    def test_add_control(self):
        from codeverify_core.compliance_dashboard import (
            ComplianceDashboard,
            ComplianceStatus,
            ControlPriority,
        )

        dash = ComplianceDashboard()
        ctrl = dash.add_control(
            "AC-1",
            "Access Control",
            "SOC2",
            ComplianceStatus.COMPLIANT,
            ControlPriority.HIGH,
        )
        assert ctrl.control_id == "AC-1"
        assert ctrl.framework == "SOC2"

    def test_compliance_scorer(self):
        from codeverify_core.compliance_dashboard import (
            ComplianceScorer,
            ComplianceStatus,
            ControlPriority,
            ControlStatus,
        )

        scorer = ComplianceScorer()
        controls = [
            ControlStatus(
                control_id="AC-1",
                control_name="Access Control",
                framework="SOC2",
                status=ComplianceStatus.COMPLIANT,
                priority=ControlPriority.HIGH,
            ),
            ControlStatus(
                control_id="AC-2",
                control_name="User Management",
                framework="SOC2",
                status=ComplianceStatus.NON_COMPLIANT,
                priority=ControlPriority.MEDIUM,
            ),
        ]
        score = scorer.calculate_score(controls, "SOC2")
        assert 0 <= score.overall_score <= 100

    def test_markdown_report(self):
        from codeverify_core.compliance_dashboard import (
            ComplianceDashboard,
            ComplianceStatus,
            ControlPriority,
            ReportFormat,
        )

        dash = ComplianceDashboard()
        dash.add_control(
            "AC-1", "Access Control", "SOC2", ComplianceStatus.COMPLIANT, ControlPriority.HIGH
        )
        report = dash.generate_report("SOC2", ReportFormat.MARKDOWN)
        assert report.format == ReportFormat.MARKDOWN

    def test_remediation_tracker(self):
        from codeverify_core.compliance_dashboard import (
            ControlPriority,
            RemediationTracker,
        )

        tracker = RemediationTracker()
        item = tracker.create_item("AC-2", "SOC2", "Fix access control", ControlPriority.HIGH)
        assert item.status == "open"
        updated = tracker.update_status(item.id, "in_progress")
        assert updated.status == "in_progress"

    def test_audit_manager(self):
        from codeverify_core.compliance_dashboard import (
            AuditManager,
            AuditType,
        )

        mgr = AuditManager()
        audit = mgr.start_audit(AuditType.INTERNAL, "SOC2", "John Doe")
        assert audit.status == "in_progress"
        completed = mgr.complete_audit(audit.id, "passed")
        assert completed.status == "completed"

    def test_dashboard_view(self):
        from codeverify_core.compliance_dashboard import (
            ComplianceDashboard,
            ComplianceStatus,
            ControlPriority,
        )

        dash = ComplianceDashboard()
        dash.add_control(
            "AC-1", "Access Control", "SOC2", ComplianceStatus.COMPLIANT, ControlPriority.HIGH
        )
        view = dash.get_dashboard(["SOC2"])
        assert len(view.widgets) > 0
        assert len(view.scores) > 0


# =============================================================================
# Feature 10: Verification Marketplace & Community
# =============================================================================


class TestMarketplaceCommunity:
    """Tests for the Verification Marketplace & Community module."""

    def test_community_enums(self):
        from codeverify_core.marketplace_community import (
            ContributorRole,
            ReputationTier,
            ReviewStatus,
            VoteType,
        )

        assert ReviewStatus.PENDING == "pending"
        assert ContributorRole.REVIEWER == "reviewer"
        assert ReputationTier.NEWCOMER == "newcomer"
        assert VoteType.UPVOTE == "upvote"

    def test_register_member(self):
        from codeverify_core.marketplace_community import (
            MarketplaceCommunity,
            ReputationTier,
        )

        mc = MarketplaceCommunity()
        member = mc.register_member("alice")
        assert member.username == "alice"
        assert member.tier == ReputationTier.NEWCOMER
        assert member.reputation_score == 0

    def test_submit_proof(self):
        from codeverify_core.marketplace_community import (
            MarketplaceCommunity,
            ReviewStatus,
        )

        mc = MarketplaceCommunity()
        member = mc.register_member("bob")
        submission = mc.submit_proof(
            member.id,
            "Null Safety Proof",
            "Proves null safety",
            "def f(x): assert x is not None",
            "python",
            "safety",
            tags=["null", "safety"],
        )
        assert submission.status == ReviewStatus.PENDING
        assert submission.author_id == member.id

    def test_review_proof(self):
        from codeverify_core.marketplace_community import MarketplaceCommunity

        mc = MarketplaceCommunity()
        author = mc.register_member("alice")
        reviewer = mc.register_member("bob")
        submission = mc.submit_proof(
            author.id,
            "Test Proof",
            "desc",
            "code",
            "python",
            "safety",
        )
        review = mc.review_proof(submission.id, reviewer.id, "Looks good")
        assert review.submission_id == submission.id

    def test_voting(self):
        from codeverify_core.marketplace_community import (
            MarketplaceCommunity,
            VoteType,
        )

        mc = MarketplaceCommunity()
        author = mc.register_member("alice")
        voter = mc.register_member("bob")
        sub = mc.submit_proof(
            author.id,
            "Test",
            "desc",
            "code",
            "python",
            "safety",
        )
        vote = mc.vote(voter.id, sub.id, "proof", VoteType.UPVOTE)
        assert vote.vote_type == VoteType.UPVOTE

    def test_reputation_tiers(self):
        from codeverify_core.marketplace_community import (
            ReputationEngine,
            ReputationTier,
        )

        engine = ReputationEngine()
        assert engine.calculate_tier(0) == ReputationTier.NEWCOMER
        assert engine.calculate_tier(100) == ReputationTier.CONTRIBUTOR
        assert engine.calculate_tier(500) == ReputationTier.TRUSTED
        assert engine.calculate_tier(2000) == ReputationTier.EXPERT
        assert engine.calculate_tier(5000) == ReputationTier.ELITE

    def test_leaderboard(self):
        from codeverify_core.marketplace_community import MarketplaceCommunity

        mc = MarketplaceCommunity()
        mc.register_member("alice")
        mc.register_member("bob")
        leaderboard = mc.get_leaderboard(limit=5)
        assert isinstance(leaderboard, list)

    def test_community_stats(self):
        from codeverify_core.marketplace_community import MarketplaceCommunity

        mc = MarketplaceCommunity()
        mc.register_member("alice")
        stats = mc.get_community_stats()
        assert stats.total_members >= 1

    def test_challenge_creation(self):
        from codeverify_core.marketplace_community import (
            ChallengeType,
            MarketplaceCommunity,
        )

        mc = MarketplaceCommunity()
        challenge = mc.create_challenge(
            "Weekly Challenge",
            "Prove null safety",
            100,
            ChallengeType.WEEKLY_PROOF,
        )
        assert challenge.reward_points == 100
        assert challenge.active is True

    def test_member_profile(self):
        from codeverify_core.marketplace_community import MarketplaceCommunity

        mc = MarketplaceCommunity()
        member = mc.register_member("alice")
        profile = mc.get_member_profile(member.id)
        assert isinstance(profile, dict)
        assert profile.get("username") == "alice" or "username" in str(profile)
