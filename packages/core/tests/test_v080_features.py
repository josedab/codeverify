"""Tests for all 10 next-gen v0.8.0 features."""

import pytest


# =============================================================================
# Feature 1: Language Expansion Engine
# =============================================================================


class TestLanguageExpansionEngine:
    """Tests for pluggable language adapter framework with Rust support."""

    def test_rust_in_supported_languages(self):
        from codeverify_core.language_support import SupportedLanguage
        assert SupportedLanguage.RUST == "rust"

    def test_rust_language_config(self):
        from codeverify_core.language_support import LANGUAGE_REGISTRY, SupportedLanguage
        cfg = LANGUAGE_REGISTRY[SupportedLanguage.RUST]
        assert cfg.file_extensions == [".rs"]
        assert cfg.type_system == "static"
        assert "i32" in cfg.integer_types
        assert cfg.supports_null_safety is True

    def test_rust_function_parsing(self):
        from codeverify_core.language_support import LanguageParser, SupportedLanguage
        parser = LanguageParser()
        code = 'pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}'
        funcs = parser.parse_functions(code, SupportedLanguage.RUST)
        assert len(funcs) >= 1
        assert funcs[0]["name"] == "add"

    def test_rust_import_parsing(self):
        from codeverify_core.language_support import LanguageParser, SupportedLanguage
        parser = LanguageParser()
        code = 'use std::collections::HashMap;\nuse std::io;\n'
        imports = parser.parse_imports(code, SupportedLanguage.RUST)
        assert len(imports) == 2
        assert "std::collections::HashMap" in imports

    def test_rust_detect_language(self):
        from codeverify_core.language_support import detect_language, SupportedLanguage
        assert detect_language("main.rs") == SupportedLanguage.RUST
        assert detect_language("lib.rs") == SupportedLanguage.RUST

    def test_rust_rules_registered(self):
        from codeverify_core.language_support import (
            LanguageRuleRegistry, SupportedLanguage, reset_language_registry,
        )
        reset_language_registry()
        registry = LanguageRuleRegistry()
        rust_rules = registry.get_rules(SupportedLanguage.RUST)
        assert len(rust_rules) >= 5
        rule_ids = [r.id for r in rust_rules]
        assert "rust_unwrap_used" in rule_ids
        assert "rust_unsafe_block" in rule_ids
        assert "rust_panic_in_lib" in rule_ids

    def test_rust_advanced_analysis(self):
        from codeverify_core.language_support import AdvancedLanguageAnalyzer, SupportedLanguage
        analyzer = AdvancedLanguageAnalyzer()
        code = 'fn main() {\n    let x = foo().unwrap();\n    let y = bar().unwrap();\n    let z = baz().unwrap();\n    let w = qux().unwrap();\n}'
        findings = analyzer.analyze(code, SupportedLanguage.RUST)
        rule_ids = [f["rule_id"] for f in findings]
        assert "rust_excessive_unwrap" in rule_ids

    def test_rust_unsafe_documentation_check(self):
        from codeverify_core.language_support import AdvancedLanguageAnalyzer, SupportedLanguage
        analyzer = AdvancedLanguageAnalyzer()
        code = 'fn main() {\n    unsafe {\n        raw_ptr();\n    }\n}'
        findings = analyzer.analyze(code, SupportedLanguage.RUST)
        rule_ids = [f["rule_id"] for f in findings]
        assert "rust_undocumented_unsafe" in rule_ids

    def test_rust_error_handling_check(self):
        from codeverify_core.language_support import Z3ConstraintGenerator, SupportedLanguage
        gen = Z3ConstraintGenerator()
        code = 'fn read_file() -> Result<String, Error> {\n    let data = fs::read_to_string("f")?;\n    Ok(data)\n}'
        result = gen.generate_error_handling_check(code, SupportedLanguage.RUST)
        assert "Rust error handling" in result
        assert "handled" in result

    def test_language_adapter_registry(self):
        from codeverify_core.language_adapter import (
            LanguageAdapterRegistry, reset_adapter_registry,
        )
        from codeverify_core.language_support import SupportedLanguage
        reset_adapter_registry()
        registry = LanguageAdapterRegistry()
        langs = registry.supported_languages()
        assert SupportedLanguage.RUST in langs
        assert SupportedLanguage.PYTHON in langs
        assert len(langs) == 5

    def test_adapter_full_analysis(self):
        from codeverify_core.language_adapter import get_adapter_registry, reset_adapter_registry
        from codeverify_core.language_support import SupportedLanguage
        reset_adapter_registry()
        registry = get_adapter_registry()
        code = 'def hello(name: str) -> str:\n    return f"Hello {name}"\n'
        result = registry.analyze_file(code, SupportedLanguage.PYTHON)
        assert result is not None
        assert len(result.functions) >= 1
        assert result.prompt_context != ""

    def test_rust_adapter_prompt(self):
        from codeverify_core.language_adapter import RustAdapter
        adapter = RustAdapter()
        prompt = adapter.agent_prompt_template("fn main() {}", "test context")
        assert "Rust" in prompt
        assert "unsafe" in prompt.lower()


# =============================================================================
# Feature 2: Zero-Config Cloud SaaS
# =============================================================================


class TestCloudSaaS:
    """Tests for multi-tenant SaaS support."""

    def test_create_tenant(self):
        from codeverify_core.cloud_saas import TenantManager, TenantTier, TenantStatus
        mgr = TenantManager()
        tenant = mgr.create_tenant("Acme Corp", "admin@acme.com")
        assert tenant.name == "Acme Corp"
        assert tenant.tier == TenantTier.FREE
        assert tenant.status == TenantStatus.TRIAL
        assert tenant.trial_ends_at is not None

    def test_tenant_limits_by_tier(self):
        from codeverify_core.cloud_saas import TenantLimits, TenantTier
        free = TenantLimits.for_tier(TenantTier.FREE)
        assert free.analyses_per_month == 100
        assert free.repos_limit == 3
        enterprise = TenantLimits.for_tier(TenantTier.ENTERPRISE)
        assert enterprise.analyses_per_month > 100000

    def test_upgrade_tier(self):
        from codeverify_core.cloud_saas import TenantManager, TenantTier, TenantStatus
        mgr = TenantManager()
        tenant = mgr.create_tenant("Acme Corp", "admin@acme.com")
        upgraded = mgr.upgrade_tier(tenant.id, TenantTier.TEAM)
        assert upgraded is not None
        assert upgraded.tier == TenantTier.TEAM
        assert upgraded.status == TenantStatus.ACTIVE
        assert upgraded.limits.analyses_per_month == 5000

    def test_add_user_respects_limits(self):
        from codeverify_core.cloud_saas import TenantManager, TenantTier
        mgr = TenantManager()
        tenant = mgr.create_tenant("Solo", "solo@test.com", tier=TenantTier.FREE)
        user1 = mgr.add_user(tenant.id, "user1@test.com", "User 1")
        assert user1 is not None
        user2 = mgr.add_user(tenant.id, "user2@test.com", "User 2")
        assert user2 is None  # Free tier: 1 member limit

    def test_usage_quota(self):
        from codeverify_core.cloud_saas import TenantManager, TenantTier
        mgr = TenantManager()
        tenant = mgr.create_tenant("Test", "t@t.com", tier=TenantTier.FREE)
        assert mgr.check_quota(tenant.id) is True
        for _ in range(100):
            mgr.record_usage(tenant.id)
        assert mgr.check_quota(tenant.id) is False

    def test_oauth_flow(self):
        from codeverify_core.cloud_saas import OAuthManager, AuthProvider
        oauth = OAuthManager(client_id="test-id", redirect_uri="http://localhost/callback")
        result = oauth.generate_auth_url(AuthProvider.GITHUB)
        assert "github.com" in result["url"]
        state = result["state"]
        token = oauth.validate_callback(state, "test-code")
        assert token is not None
        assert token.provider == AuthProvider.GITHUB

    def test_suspend_tenant(self):
        from codeverify_core.cloud_saas import TenantManager, TenantStatus
        mgr = TenantManager()
        tenant = mgr.create_tenant("Bad Tenant", "bad@t.com")
        assert mgr.suspend_tenant(tenant.id, "TOS violation") is True
        assert mgr.get_tenant(tenant.id).status == TenantStatus.SUSPENDED


# =============================================================================
# Feature 3: Incremental Verification Cache (already exists, test it)
# =============================================================================


class TestIncrementalCache:
    """Tests for the existing verification cache with AST fingerprinting."""

    def test_cache_fingerprint(self):
        from codeverify_core.verification_cache import ASTFingerprinter
        fp = ASTFingerprinter()
        hash1 = fp.fingerprint_function("def f(x): return x", "f", "python")
        hash2 = fp.fingerprint_function("def f(x): return x", "f", "python")
        hash3 = fp.fingerprint_function("def f(x): return x + 1", "f", "python")
        assert hash1 == hash2
        assert hash1 != hash3

    def test_cache_store_and_retrieve(self):
        from codeverify_core.verification_cache import VerificationCache, CacheConfig
        cache = VerificationCache(CacheConfig(enabled=True))
        cache.put("fp-abc", "test_func", "test.py", "null_safety", {"verified": True}, 50.0)
        result = cache.get("fp-abc", "null_safety")
        assert result is not None
        assert result.result["verified"] is True

    def test_cache_miss(self):
        from codeverify_core.verification_cache import VerificationCache, CacheConfig
        cache = VerificationCache(CacheConfig(enabled=True))
        result = cache.get("nonexistent", "null_safety")
        assert result is None

    def test_cache_stats(self):
        from codeverify_core.verification_cache import VerificationCache, CacheConfig
        cache = VerificationCache(CacheConfig(enabled=True))
        cache.put("fp-1", "f1", "a.py", "null_safety", {"ok": True}, 10.0)
        cache.get("fp-1", "null_safety")  # hit
        cache.get("fp-2", "null_safety")  # miss
        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1


# =============================================================================
# Feature 4: LLM Cost Optimizer with Local Models
# =============================================================================


class TestLLMCostOptimizer:
    """Tests for LLM cost optimization and routing."""

    def test_complexity_classification(self):
        from codeverify_core.llm_cost_optimizer import ComplexityClassifier, CheckComplexity
        classifier = ComplexityClassifier()
        assert classifier.classify("x = 1", "null_safety") == CheckComplexity.TRIVIAL
        long_code = "\n".join([f"line_{i} = {i}" for i in range(50)])
        assert classifier.classify(long_code, "semantic") in (CheckComplexity.MODERATE, CheckComplexity.COMPLEX)

    def test_routing_local_for_simple(self):
        from codeverify_core.llm_cost_optimizer import (
            CostOptimizer, ModelEndpoint, ModelProvider, RoutingStrategy
        )
        opt = CostOptimizer(strategy=RoutingStrategy.COST_OPTIMIZED)
        opt.add_endpoint(ModelEndpoint(
            provider=ModelProvider.OLLAMA, model_name="codellama",
            cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
        ))
        opt.add_endpoint(ModelEndpoint(
            provider=ModelProvider.OPENAI, model_name="gpt-4",
            cost_per_1k_input_tokens=0.03, cost_per_1k_output_tokens=0.06,
        ))
        decision = opt.route("x = 1", "null_safety")
        assert decision.endpoint.provider == ModelProvider.OLLAMA
        assert decision.estimated_cost == 0.0

    def test_routing_cloud_for_complex(self):
        from codeverify_core.llm_cost_optimizer import (
            CostOptimizer, ModelEndpoint, ModelProvider, RoutingStrategy
        )
        opt = CostOptimizer(strategy=RoutingStrategy.COST_OPTIMIZED)
        opt.add_endpoint(ModelEndpoint(
            provider=ModelProvider.OLLAMA, model_name="codellama",
        ))
        opt.add_endpoint(ModelEndpoint(
            provider=ModelProvider.OPENAI, model_name="gpt-4",
            cost_per_1k_input_tokens=0.03, cost_per_1k_output_tokens=0.06,
        ))
        long_code = "\n".join([f"async def func_{i}(): await something()" for i in range(50)])
        decision = opt.route(long_code, "semantic")
        assert decision.endpoint.provider == ModelProvider.OPENAI

    def test_cost_recording(self):
        from codeverify_core.llm_cost_optimizer import CostOptimizer, ModelEndpoint, ModelProvider
        opt = CostOptimizer()
        opt.add_endpoint(ModelEndpoint(
            provider=ModelProvider.OPENAI, model_name="gpt-4",
            cost_per_1k_input_tokens=0.03, cost_per_1k_output_tokens=0.06,
        ))
        record = opt.record_call(ModelProvider.OPENAI, "gpt-4", 1000, 500, 1500.0)
        assert record.cost_usd == pytest.approx(0.03 + 0.03, abs=0.01)

    def test_budget_tracking(self):
        from codeverify_core.llm_cost_optimizer import CostOptimizer, CostBudget
        budget = CostBudget(monthly_limit_usd=10.0)
        opt = CostOptimizer(budget=budget)
        assert budget.is_exhausted is False
        budget.current_spend_usd = 10.0
        assert budget.is_exhausted is True
        assert budget.usage_pct == 100.0

    def test_cost_report(self):
        from codeverify_core.llm_cost_optimizer import CostOptimizer, ModelEndpoint, ModelProvider
        opt = CostOptimizer()
        opt.add_endpoint(ModelEndpoint(
            provider=ModelProvider.OPENAI, model_name="gpt-4",
            cost_per_1k_input_tokens=0.03, cost_per_1k_output_tokens=0.06,
        ))
        opt.record_call(ModelProvider.OPENAI, "gpt-4", 500, 200, 800.0)
        report = opt.get_report()
        assert report.total_calls == 1
        assert report.cloud_calls == 1


# =============================================================================
# Feature 5: Interactive Proof Explorer
# =============================================================================


class TestProofExplorer:
    """Tests for proof trace serialization and exploration."""

    def test_create_trace_verified(self):
        from codeverify_core.proof_explorer import ProofTraceSerializer, ProofOutcome
        serializer = ProofTraceSerializer()
        smt = '(declare-const x Int)\n(assert (>= x 0))\n(check-sat)'
        result = {"satisfiable": False, "counterexample": None, "proof_time_ms": 5.0}
        trace = serializer.create_trace("add", "math.py", "null_safety", smt, result)
        assert trace.outcome == ProofOutcome.VERIFIED
        assert trace.step_count >= 3  # declare + assert + check + result

    def test_create_trace_counterexample(self):
        from codeverify_core.proof_explorer import ProofTraceSerializer, ProofOutcome
        serializer = ProofTraceSerializer()
        smt = '(declare-const x Int)\n(assert (< x 0))\n(check-sat)'
        result = {"satisfiable": True, "counterexample": {"x": "-1"}, "proof_time_ms": 3.0}
        trace = serializer.create_trace("div", "math.py", "div_zero", smt, result)
        assert trace.outcome == ProofOutcome.COUNTEREXAMPLE_FOUND
        assert trace.counterexample is not None
        assert trace.counterexample.variables["x"] == "-1"

    def test_proof_explorer_store_and_retrieve(self):
        from codeverify_core.proof_explorer import ProofExplorer, ProofOutcome
        explorer = ProofExplorer()
        smt = '(declare-const x Int)\n(assert (>= x 0))\n(check-sat)'
        trace = explorer.create_and_store_trace(
            "func", "test.py", "bounds", smt,
            {"satisfiable": False, "proof_time_ms": 2.0},
        )
        assert explorer.get_trace(trace.id) is not None
        assert explorer.get_step(trace.id, 0) is not None

    def test_export_mermaid(self):
        from codeverify_core.proof_explorer import ProofExplorer
        explorer = ProofExplorer()
        smt = '(declare-const x Int)\n(assert (>= x 0))\n(check-sat)'
        trace = explorer.create_and_store_trace(
            "func", "t.py", "bounds", smt,
            {"satisfiable": False, "proof_time_ms": 1.0},
        )
        mermaid = explorer.export_mermaid(trace.id)
        assert mermaid is not None
        assert "flowchart TD" in mermaid

    def test_export_json(self):
        from codeverify_core.proof_explorer import ProofExplorer
        explorer = ProofExplorer()
        smt = '(declare-const x Int)\n(check-sat)'
        trace = explorer.create_and_store_trace(
            "f", "a.py", "null", smt,
            {"satisfiable": False, "proof_time_ms": 0.5},
        )
        data = explorer.export_json(trace.id)
        assert data is not None
        assert data["function"] == "f"
        assert "steps" in data

    def test_step_explanations(self):
        from codeverify_core.proof_explorer import ProofTraceSerializer
        serializer = ProofTraceSerializer()
        smt = '(declare-const x Int)\n(declare-const x_is_null Bool)\n(assert (not x_is_null))\n(check-sat)'
        trace = serializer.create_trace("func", "f.py", "null_safety", smt,
            {"satisfiable": False, "proof_time_ms": 1.0})
        # Check that at least one step has a meaningful explanation
        explanations = [s.explanation for s in trace.steps if s.explanation]
        assert len(explanations) > 0


# =============================================================================
# Feature 6: Auto-Fix Pipeline with Verification Loop
# =============================================================================


class TestAutoFixPipeline:
    """Tests for LLM-powered fix generation with Z3 re-verification."""

    def test_generate_null_fix(self):
        from codeverify_core.autofix_loop import FixGenerator, Finding
        gen = FixGenerator()
        finding = Finding(rule_id="null_safety", message="Variable 'x' may be None", line=2)
        code = "def foo(x):\n    return x.strip()\n"
        fixed = gen.generate_fix(finding, code)
        assert "None" in fixed
        assert fixed != code

    def test_verify_fix_passes(self):
        from codeverify_core.autofix_loop import FixVerifier
        verifier = FixVerifier()
        result = verifier.verify_fix("x = 1", "x = 1\nassert x > 0")
        assert result["passed"] is True

    def test_verify_fix_no_change(self):
        from codeverify_core.autofix_loop import FixVerifier
        verifier = FixVerifier()
        result = verifier.verify_fix("x = 1", "x = 1")
        assert result["passed"] is False

    def test_pipeline_generates_verified_fix(self):
        from codeverify_core.autofix_loop import AutoFixPipeline, Finding, FixStatus
        pipeline = AutoFixPipeline(max_iterations=3)
        finding = Finding(rule_id="null_safety", message="'value' is None", line=2)
        code = "def process(value):\n    return value.upper()\n"
        result = pipeline.fix_finding(finding, code)
        assert result.status == FixStatus.VERIFIED
        assert result.attempt_count >= 1
        assert result.proof_hash != ""

    def test_pipeline_batch_fix(self):
        from codeverify_core.autofix_loop import AutoFixPipeline, Finding
        pipeline = AutoFixPipeline()
        findings = [
            Finding(rule_id="null_safety", message="'x' is None", line=2),
            Finding(rule_id="division_by_zero", message="Division by 'y'", line=3),
        ]
        code = "def calc(x, y):\n    a = x.strip()\n    b = 10 / y\n"
        results = pipeline.fix_batch(findings, code)
        assert len(results) == 2

    def test_pipeline_stats(self):
        from codeverify_core.autofix_loop import AutoFixPipeline, Finding
        pipeline = AutoFixPipeline()
        finding = Finding(rule_id="null_safety", message="'x' is None", line=1)
        pipeline.fix_finding(finding, "x.strip()")
        stats = pipeline.get_stats()
        assert stats["total_fixes"] == 1


# =============================================================================
# Feature 7: CI/CD Native Actions & Plugins
# =============================================================================


class TestCICDActions:
    """Tests for CI/CD configuration generation and quality gates."""

    def test_github_actions_config(self):
        from codeverify_core.cicd_actions import CIConfigGenerator, CIPlatform
        gen = CIConfigGenerator()
        config = gen.generate(CIPlatform.GITHUB_ACTIONS, ["python", "rust"])
        assert "codeverify/action@v1" in config
        assert "python, rust" in config

    def test_gitlab_ci_config(self):
        from codeverify_core.cicd_actions import CIConfigGenerator, CIPlatform
        gen = CIConfigGenerator()
        config = gen.generate(CIPlatform.GITLAB_CI)
        assert "codeverify scan" in config
        assert "sast:" in config

    def test_jenkins_config(self):
        from codeverify_core.cicd_actions import CIConfigGenerator, CIPlatform
        gen = CIConfigGenerator()
        config = gen.generate(CIPlatform.JENKINS)
        assert "pipeline" in config
        assert "codeverify" in config

    def test_quality_gate_pass(self):
        from codeverify_core.cicd_actions import QualityGateEvaluator, QualityGateConfig
        gate = QualityGateEvaluator()
        findings = [{"severity": "low"}, {"severity": "low"}]
        result = gate.evaluate(findings, QualityGateConfig(max_low=5))
        assert result["result"] == "warn"

    def test_quality_gate_fail(self):
        from codeverify_core.cicd_actions import QualityGateEvaluator, QualityGateConfig
        gate = QualityGateEvaluator()
        findings = [{"severity": "critical"}]
        result = gate.evaluate(findings, QualityGateConfig(max_critical=0))
        assert result["result"] == "fail"

    def test_sarif_report(self):
        from codeverify_core.cicd_actions import SARIFReport, SARIFResult
        report = SARIFReport()
        report.results.append(SARIFResult(
            rule_id="null_safety", message="Null issue", level="warning",
            file_path="test.py", start_line=5,
        ))
        sarif = report.to_dict()
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"][0]["results"]) == 1

    def test_action_runner(self):
        from codeverify_core.cicd_actions import CICDActionRunner
        runner = CICDActionRunner()
        findings = [
            {"rule_id": "test", "message": "test", "severity": "low", "file_path": "a.py", "line": 1},
        ]
        result = runner.run_analysis(findings)
        assert "sarif" in result
        assert result["gate"]["result"] in ("pass", "warn")


# =============================================================================
# Feature 8: Organization Intelligence Dashboard
# =============================================================================


class TestOrgDashboard:
    """Tests for organization metrics and dashboards."""

    def test_metrics_aggregator(self):
        from codeverify_core.org_dashboard import MetricsAggregator
        agg = MetricsAggregator()
        agg.ingest_finding({"repo": "api", "severity": "critical", "category": "security", "rule_id": "sql_injection"})
        agg.ingest_finding({"repo": "api", "severity": "high", "category": "null_safety", "rule_id": "null_deref"})
        agg.ingest_analysis({"repo": "api"})
        metrics = agg.compute_repo_metrics("api")
        assert metrics.total_findings == 2
        assert metrics.findings_by_severity["critical"] == 1

    def test_team_metrics(self):
        from codeverify_core.org_dashboard import MetricsAggregator
        agg = MetricsAggregator()
        for _ in range(5):
            agg.ingest_finding({"repo": "web", "severity": "medium", "rule_id": "style"})
        agg.ingest_analysis({"repo": "web"})
        team = agg.compute_team_metrics("t1", "Frontend", ["web"])
        assert team.total_findings == 5
        assert team.medium_findings == 5

    def test_roi_calculation(self):
        from codeverify_core.org_dashboard import MetricsAggregator
        agg = MetricsAggregator()
        agg.ingest_finding({"repo": "api", "severity": "critical", "category": "security", "rule_id": "sqli"})
        agg.ingest_finding({"repo": "api", "severity": "low", "category": "style", "rule_id": "naming"})
        roi = agg.compute_roi()
        assert roi.bugs_caught_pre_production == 2
        assert roi.estimated_cost_avoided_usd > 0
        assert roi.security_vulns_prevented == 1

    def test_risk_heatmap(self):
        from codeverify_core.org_dashboard import MetricsAggregator, RiskLevel
        agg = MetricsAggregator()
        agg.ingest_finding({"repo": "api", "severity": "critical", "rule_id": "sqli"})
        heatmap = agg.compute_risk_heatmap([{"repo": "api", "team": "backend"}])
        assert len(heatmap) == 1
        assert heatmap[0].risk_level == RiskLevel.CRITICAL

    def test_full_dashboard(self):
        from codeverify_core.org_dashboard import OrgDashboard
        dash = OrgDashboard(org_name="Acme")
        dash.ingest_finding({"repo": "api", "severity": "high", "rule_id": "null"})
        dash.ingest_analysis({"repo": "api"})
        data = dash.generate_dashboard(
            teams=[{"id": "t1", "name": "Backend", "repos": ["api"]}],
        )
        assert data.org_name == "Acme"
        assert len(data.team_metrics) == 1
        assert data.roi.bugs_caught_pre_production == 1

    def test_export_csv(self):
        from codeverify_core.org_dashboard import OrgDashboard
        dash = OrgDashboard(org_name="Test")
        dash.ingest_finding({"repo": "web", "severity": "low", "rule_id": "x"})
        data = dash.generate_dashboard()
        csv = dash.export_csv(data)
        assert "org_name,Test" in csv
        assert "bugs_caught" in csv


# =============================================================================
# Feature 9: Verification-Aware Code Generation
# =============================================================================


class TestVerificationCodeGen:
    """Tests for verification-enriched code generation."""

    def test_context_extraction(self):
        from codeverify_core.verification_codegen import ContextExtractor
        extractor = ContextExtractor()
        code = 'def process(name: str, count: int) -> list:\n    """Process items."""\n    return [name] * count\n'
        ctx = extractor.extract(code, 2, "python")
        assert ctx.function_name == "process"
        assert len(ctx.parameters) == 2
        assert ctx.return_type == "list"

    def test_assertion_generation(self):
        from codeverify_core.verification_codegen import AssertionGenerator, GenerationContext
        gen = AssertionGenerator()
        ctx = GenerationContext(
            function_name="process",
            parameters=[
                {"name": "index", "type": "int"},
                {"name": "name", "type": "str"},
            ],
            return_type="str",
            language="python",
        )
        assertions = gen.generate(ctx)
        assert len(assertions) >= 2  # bounds check for index + postcondition
        types = [a.assertion_type.value for a in assertions]
        assert "bounds_check" in types

    def test_inline_assertion_to_code(self):
        from codeverify_core.verification_codegen import InlineAssertion, AssertionType
        assertion = InlineAssertion(
            assertion_type=AssertionType.PRECONDITION,
            expression="x is not None",
            natural_language="x must not be None",
            language="python",
        )
        code = assertion.to_code()
        assert "assert x is not None" in code

    def test_suggestion_ranking(self):
        from codeverify_core.verification_codegen import SuggestionRanker, GenerationContext
        ranker = SuggestionRanker()
        ctx = GenerationContext(function_name="test", language="python")
        suggestions = [
            "def test():\n    return eval('1+1')\n",
            "def test():\n    try:\n        return 1 + 1\n    except Exception:\n        return 0\n",
        ]
        ranked = ranker.rank_suggestions(suggestions, ctx)
        assert len(ranked) == 2
        # The safe version should rank higher
        assert ranked[0].verification_score > ranked[1].verification_score

    def test_verification_aware_codegen_prompt(self):
        from codeverify_core.verification_codegen import VerificationAwareCodeGen
        gen = VerificationAwareCodeGen()
        code = 'def divide(a: int, b: int) -> float:\n    """Divide a by b."""\n    return a / b\n'
        prompt = gen.enrich_prompt(code, 2, "python")
        assert "divide" in prompt
        assert "constraints" in prompt.lower()

    def test_rust_assertion(self):
        from codeverify_core.verification_codegen import InlineAssertion, AssertionType
        assertion = InlineAssertion(
            assertion_type=AssertionType.PRECONDITION,
            expression="x > 0",
            natural_language="x must be positive",
            language="rust",
        )
        assert "debug_assert!" in assertion.to_code()


# =============================================================================
# Feature 10: Plugin Marketplace & SDK
# =============================================================================


class TestPluginMarketplace:
    """Tests for plugin SDK and registry."""

    def test_create_manifest(self):
        from codeverify_core.plugin_marketplace import PluginSDK, PluginType
        manifest = PluginSDK.create_manifest(
            name="rust-analyzer",
            version="1.0.0",
            description="Rust language support",
            plugin_type=PluginType.LANGUAGE_ADAPTER,
            author="Test Author",
        )
        assert manifest.id == "rust-analyzer@1.0.0"
        assert manifest.plugin_type == PluginType.LANGUAGE_ADAPTER

    def test_validate_manifest(self):
        from codeverify_core.plugin_marketplace import PluginSDK, PluginManifest, PluginType
        valid = PluginManifest(name="test", version="1.0", description="desc", plugin_type=PluginType.RULE)
        assert len(PluginSDK.validate_manifest(valid)) == 0
        invalid = PluginManifest(name="", version="", description="", plugin_type=PluginType.RULE)
        errors = PluginSDK.validate_manifest(invalid)
        assert len(errors) >= 3

    def test_publish_and_search(self):
        from codeverify_core.plugin_marketplace import PluginRegistry, PluginManifest, PluginType
        registry = PluginRegistry()
        manifest = PluginManifest(
            name="security-rules",
            version="1.0.0",
            description="Advanced security rules for OWASP",
            plugin_type=PluginType.RULE,
            keywords=["security", "owasp"],
        )
        entry = registry.publish(manifest)
        assert entry.name == "security-rules"

        results = registry.search("security")
        assert results.total_count == 1
        assert results.plugins[0].name == "security-rules"

    def test_install_plugin(self):
        from codeverify_core.plugin_marketplace import PluginRegistry, PluginManifest, PluginType
        registry = PluginRegistry()
        manifest = PluginManifest(name="test-plugin", version="1.0.0", description="test", plugin_type=PluginType.AGENT)
        registry.publish(manifest)
        entry = registry.install("test-plugin")
        assert entry is not None
        assert entry.downloads == 1

    def test_add_review(self):
        from codeverify_core.plugin_marketplace import PluginRegistry, PluginManifest, PluginType, PluginReview
        registry = PluginRegistry()
        manifest = PluginManifest(name="good-plugin", version="1.0.0", description="great", plugin_type=PluginType.RULE)
        registry.publish(manifest)
        review = PluginReview(plugin_name="good-plugin", reviewer="user1", rating=5, comment="Excellent!")
        assert registry.add_review("good-plugin", review) is True
        entry = registry.get("good-plugin")
        assert entry.rating == 5.0

    def test_search_by_type(self):
        from codeverify_core.plugin_marketplace import PluginRegistry, PluginManifest, PluginType
        registry = PluginRegistry()
        registry.publish(PluginManifest(name="p1", version="1.0", description="d1", plugin_type=PluginType.RULE))
        registry.publish(PluginManifest(name="p2", version="1.0", description="d2", plugin_type=PluginType.AGENT))
        rules = registry.list_by_type(PluginType.RULE)
        assert len(rules) == 1

    def test_deprecate_plugin(self):
        from codeverify_core.plugin_marketplace import PluginRegistry, PluginManifest, PluginType, PluginStatus
        registry = PluginRegistry()
        registry.publish(PluginManifest(name="old", version="1.0.0", description="old", plugin_type=PluginType.RULE))
        assert registry.deprecate("old", "1.0.0") is True
        entry = registry.get("old", "1.0.0")
        assert entry.status == PluginStatus.DEPRECATED

    def test_registry_stats(self):
        from codeverify_core.plugin_marketplace import PluginRegistry, PluginManifest, PluginType
        registry = PluginRegistry()
        registry.publish(PluginManifest(name="a", version="1.0", description="a", plugin_type=PluginType.RULE))
        registry.publish(PluginManifest(name="b", version="1.0", description="b", plugin_type=PluginType.AGENT))
        stats = registry.stats()
        assert stats["total_plugins"] == 2
        assert stats["by_type"]["rule"] == 1
        assert stats["by_type"]["agent"] == 1
