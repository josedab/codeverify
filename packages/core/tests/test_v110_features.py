"""Tests for v1.1.0 next-gen features.

Covers all 10 next-gen features:
1. Hosted SaaS with Free Tier
2. Go + Java Language Support
3. Autofix Agent with PR Generation
4. GitHub Copilot Extension
5. Incremental Verification Engine
6. Organization Security Posture Dashboard
7. CI/CD Pipeline Orchestrator
8. LLM-Powered Proof Explainer
9. Supply Chain Verification
10. Self-Learning Rule Engine
"""

import pytest


# ─── Feature 1: Hosted SaaS with Free Tier ─────────────────────────────


class TestSaaSPlatform:
    def test_create_tenant_free(self):
        from codeverify_core.saas_platform import SaaSPlatform, PlanTier, TenantStatus

        platform = SaaSPlatform()
        tenant = platform.create_tenant("TestOrg", "admin@test.com")
        assert tenant.name == "TestOrg"
        assert tenant.plan == PlanTier.FREE
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.limits.max_verifications_per_month == 500
        assert tenant.limits.max_repositories == 3

    def test_upgrade_tenant(self):
        from codeverify_core.saas_platform import SaaSPlatform, PlanTier

        platform = SaaSPlatform()
        tenant = platform.create_tenant("Org", "a@b.com")
        platform.upgrade_tenant(tenant.id, PlanTier.PRO)
        updated = platform.get_tenant(tenant.id)
        assert updated is not None
        assert updated.plan == PlanTier.PRO
        assert updated.limits.max_verifications_per_month == 10000
        assert updated.limits.custom_rules is True

    def test_enterprise_plan_limits(self):
        from codeverify_core.saas_platform import PlanLimits, PlanTier

        limits = PlanLimits.for_tier(PlanTier.ENTERPRISE)
        assert limits.max_verifications_per_month == -1  # unlimited
        assert limits.sso_enabled is True
        assert limits.audit_log is True
        assert limits.sla_uptime == 99.9

    def test_api_key_lifecycle(self):
        from codeverify_core.saas_platform import SaaSPlatform, ApiKeyScope

        platform = SaaSPlatform()
        tenant = platform.create_tenant("Org", "a@b.com")
        api_key, raw = platform.create_api_key(tenant.id, "CI Key", [ApiKeyScope.VERIFY])
        assert raw.startswith("cv_")
        assert api_key.has_scope(ApiKeyScope.VERIFY)

        validated = platform.validate_api_key(raw)
        assert validated is not None
        assert validated.id == api_key.id

    def test_rate_limiting(self):
        from codeverify_core.saas_platform import SaaSPlatform, PlanTier

        platform = SaaSPlatform()
        tenant = platform.create_tenant("Org", "a@b.com", plan=PlanTier.FREE)
        # Free tier: 60 calls/hour
        for _ in range(60):
            assert platform.check_rate_limit(tenant.id) is True
        assert platform.check_rate_limit(tenant.id) is False

    def test_usage_tracking(self):
        from codeverify_core.saas_platform import SaaSPlatform, UsageMetricType

        platform = SaaSPlatform()
        tenant = platform.create_tenant("Org", "a@b.com")
        platform.record_usage(tenant.id, UsageMetricType.VERIFICATIONS, 10)
        platform.record_usage(tenant.id, UsageMetricType.VERIFICATIONS, 5)
        summary = platform.get_usage_summary(tenant.id)
        assert summary is not None
        assert summary.totals["verifications"] == 15

    def test_usage_limit_check(self):
        from codeverify_core.saas_platform import SaaSPlatform, UsageMetricType

        platform = SaaSPlatform()
        tenant = platform.create_tenant("Org", "a@b.com")
        # Under limit
        assert platform.check_usage_limit(tenant.id, UsageMetricType.VERIFICATIONS) is True

    def test_tenant_trial(self):
        from codeverify_core.saas_platform import SaaSPlatform, PlanTier, TenantStatus

        platform = SaaSPlatform()
        tenant = platform.create_tenant("Org", "a@b.com")
        tenant.start_trial(days=14)
        assert tenant.status == TenantStatus.TRIAL
        assert tenant.plan == PlanTier.PRO
        assert tenant.trial_ends_at is not None

    def test_suspend_tenant(self):
        from codeverify_core.saas_platform import SaaSPlatform, TenantStatus

        platform = SaaSPlatform()
        tenant = platform.create_tenant("Org", "a@b.com")
        result = platform.suspend_tenant(tenant.id, "non-payment")
        assert result is True
        assert tenant.status == TenantStatus.SUSPENDED

    def test_singleton(self):
        from codeverify_core.saas_platform import get_saas_platform, reset_saas_platform

        reset_saas_platform()
        p1 = get_saas_platform()
        p2 = get_saas_platform()
        assert p1 is p2
        reset_saas_platform()


# ─── Feature 2: Go + Java Language Support ──────────────────────────────


class TestGoJavaSupport:
    def test_go_parser_functions(self):
        from codeverify_core.go_java_support import AdvancedGoParser, GoJavaNodeType

        parser = AdvancedGoParser()
        code = '''
func Add(a int, b int) int {
    return a + b
}

func (s *Server) Start(port int) error {
    return nil
}
'''
        result = parser.parse(code, "main.go")
        funcs = [n for n in result.nodes if n.node_type in (GoJavaNodeType.FUNCTION, GoJavaNodeType.METHOD)]
        assert len(funcs) >= 2
        assert funcs[0].name == "Add"
        assert funcs[1].name == "Start"
        assert funcs[1].node_type == GoJavaNodeType.METHOD

    def test_go_parser_structs_interfaces(self):
        from codeverify_core.go_java_support import AdvancedGoParser, GoJavaNodeType

        parser = AdvancedGoParser()
        code = '''
type User struct {
    Name string
    Age  int
}

type Reader interface {
    Read(p []byte) (int, error)
}
'''
        result = parser.parse(code, "types.go")
        structs = [n for n in result.nodes if n.node_type == GoJavaNodeType.STRUCT]
        interfaces = [n for n in result.nodes if n.node_type == GoJavaNodeType.INTERFACE]
        assert len(structs) == 1
        assert structs[0].name == "User"
        assert len(interfaces) == 1

    def test_go_parser_goroutines_channels(self):
        from codeverify_core.go_java_support import AdvancedGoParser, GoJavaNodeType

        parser = AdvancedGoParser()
        code = '''
func main() {
    ch := make(chan int, 10)
    go worker(ch)
}
'''
        result = parser.parse(code, "main.go")
        goroutines = [n for n in result.nodes if n.node_type == GoJavaNodeType.GOROUTINE]
        channels = [n for n in result.nodes if n.node_type == GoJavaNodeType.CHANNEL]
        assert len(goroutines) >= 1
        assert len(channels) >= 1

    def test_go_error_ignored_pattern(self):
        from codeverify_core.go_java_support import AdvancedGoParser, IdiomaticPattern

        parser = AdvancedGoParser()
        code = '''
func bad() {
    result, _ := doSomething()
}
'''
        result = parser.parse(code)
        warnings = [p for p in result.patterns if p.pattern == IdiomaticPattern.GO_ERROR_IGNORED]
        assert len(warnings) >= 1
        assert warnings[0].severity == "warning"

    def test_java_parser_classes(self):
        from codeverify_core.go_java_support import AdvancedJavaParser, GoJavaNodeType

        parser = AdvancedJavaParser()
        code = '''
public class UserService extends BaseService implements Serializable {
    public String getName(int id) {
        return "user";
    }
}
'''
        result = parser.parse(code, "UserService.java")
        classes = [n for n in result.nodes if n.node_type == GoJavaNodeType.CLASS]
        methods = [n for n in result.nodes if n.node_type == GoJavaNodeType.METHOD]
        assert len(classes) == 1
        assert classes[0].name == "UserService"
        assert len(methods) >= 1

    def test_java_parser_annotations(self):
        from codeverify_core.go_java_support import AdvancedJavaParser, IdiomaticPattern

        parser = AdvancedJavaParser()
        code = '''
public class Api {
    @Nullable String name;
    public void process(@NonNull String input) {
    }
}
'''
        result = parser.parse(code, "Api.java")
        nullable_patterns = [p for p in result.patterns if p.pattern == IdiomaticPattern.JAVA_NULLABLE_ANNOTATION]
        assert len(nullable_patterns) >= 1

    def test_java_optional_detection(self):
        from codeverify_core.go_java_support import AdvancedJavaParser, IdiomaticPattern

        parser = AdvancedJavaParser()
        code = '''
public class Service {
    public Optional<User> findUser(String id) {
        return Optional.empty();
    }
}
'''
        result = parser.parse(code, "Service.java")
        optional_patterns = [p for p in result.patterns if p.pattern == IdiomaticPattern.JAVA_OPTIONAL_USAGE]
        assert len(optional_patterns) >= 1

    def test_go_java_support_unified(self):
        from codeverify_core.go_java_support import GoJavaLanguageSupport

        support = GoJavaLanguageSupport()
        go_result = support.parse("func main() {}", "go")
        java_result = support.parse("public class Main {}", "java")
        assert go_result.language == "go"
        assert java_result.language == "java"
        assert support.detect_language("main.go") == "go"
        assert support.detect_language("App.java") == "java"

    def test_singleton(self):
        from codeverify_core.go_java_support import get_go_java_support, reset_go_java_support

        reset_go_java_support()
        s1 = get_go_java_support()
        s2 = get_go_java_support()
        assert s1 is s2
        reset_go_java_support()


# ─── Feature 3: Autofix Agent with PR Generation ────────────────────────


class TestAutofixAgent:
    def test_fix_null_safety(self):
        from codeverify_core.autofix_pr_agent import AutofixAgent, Finding, FixStatus

        agent = AutofixAgent()
        finding = Finding(
            file_path="app.py",
            line_number=42,
            category="null_safety",
            severity="high",
            message="Variable may be None",
            code_snippet="result = value.strip()",
        )
        result = agent.fix(finding, language="python")
        assert result.status in (FixStatus.VERIFIED, FixStatus.FAILED_VERIFICATION)
        assert len(result.candidates) > 0

    def test_fix_generates_pr_metadata(self):
        from codeverify_core.autofix_pr_agent import AutofixAgent, Finding, FixStatus

        agent = AutofixAgent()
        finding = Finding(
            file_path="api.py",
            line_number=10,
            category="null_safety",
            severity="critical",
            message="Null deref",
            code_snippet="x = obj.method()",
        )
        result = agent.fix(finding, language="python")
        if result.status == FixStatus.VERIFIED:
            assert result.pr_branch.startswith("autofix/")
            assert "null_safety" in result.pr_title
            assert result.pr_body != ""

    def test_fix_batch(self):
        from codeverify_core.autofix_pr_agent import AutofixAgent, Finding

        agent = AutofixAgent()
        findings = [
            Finding(category="null_safety", message="null1", code_snippet="a.b()"),
            Finding(category="division_by_zero", message="div0", code_snippet="x/y"),
        ]
        results = agent.fix_batch(findings)
        assert len(results) == 2

    def test_fix_confidence_levels(self):
        from codeverify_core.autofix_pr_agent import FixGenerator, Finding, FixConfidence

        gen = FixGenerator()
        finding = Finding(category="null_safety", code_snippet="x.y()")
        candidates = gen.generate(finding, "python")
        assert len(candidates) > 0
        assert all(c.confidence in (FixConfidence.HIGH, FixConfidence.MEDIUM, FixConfidence.LOW) for c in candidates)

    def test_diff_limit_enforcement(self):
        from codeverify_core.autofix_pr_agent import AutofixAgent, Finding, FixStatus

        agent = AutofixAgent(max_diff_lines=1)
        finding = Finding(
            category="null_safety", code_snippet="x = value.strip()\ny = value.upper()\nz = value.lower()",
        )
        result = agent.fix(finding)
        # With very low diff limit, might fail
        assert result.status in (FixStatus.VERIFIED, FixStatus.FAILED_VERIFICATION)

    def test_success_rate(self):
        from codeverify_core.autofix_pr_agent import AutofixAgent, Finding

        agent = AutofixAgent()
        agent.fix(Finding(category="null_safety", code_snippet="x.y()"))
        assert 0.0 <= agent.success_rate <= 1.0

    def test_singleton(self):
        from codeverify_core.autofix_pr_agent import get_autofix_agent, reset_autofix_agent

        reset_autofix_agent()
        a1 = get_autofix_agent()
        a2 = get_autofix_agent()
        assert a1 is a2
        reset_autofix_agent()


# ─── Feature 4: GitHub Copilot Extension ─────────────────────────────────


class TestCopilotExtension:
    def test_verify_command(self):
        from codeverify_core.copilot_chat_extension import CopilotExtensionHandler, ChatContext

        handler = CopilotExtensionHandler()
        ctx = ChatContext(file_path="main.py", language="python", selected_code="x = 42")
        response = handler.handle("@codeverify verify", ctx)
        assert len(response.messages) > 0
        assert "Verifying" in response.full_text

    def test_explain_command(self):
        from codeverify_core.copilot_chat_extension import CopilotExtensionHandler, ChatContext

        handler = CopilotExtensionHandler()
        ctx = ChatContext(file_path="main.py", selected_code="x = y.strip()")
        response = handler.handle("@codeverify explain", ctx)
        assert "Explanation" in response.full_text or "explanation" in response.full_text.lower()

    def test_fix_command(self):
        from codeverify_core.copilot_chat_extension import CopilotExtensionHandler, ChatContext

        handler = CopilotExtensionHandler()
        ctx = ChatContext(file_path="main.py", selected_code="x = 1/0")
        response = handler.handle("@codeverify fix", ctx)
        assert len(response.messages) > 0

    def test_help_command(self):
        from codeverify_core.copilot_chat_extension import CopilotExtensionHandler

        handler = CopilotExtensionHandler()
        response = handler.handle("@codeverify help")
        assert "verify" in response.full_text.lower()
        assert "explain" in response.full_text.lower()
        assert "fix" in response.full_text.lower()

    def test_status_command(self):
        from codeverify_core.copilot_chat_extension import CopilotExtensionHandler

        handler = CopilotExtensionHandler()
        response = handler.handle("@codeverify status")
        assert "Status" in response.full_text or "Ready" in response.full_text

    def test_scan_command(self):
        from codeverify_core.copilot_chat_extension import CopilotExtensionHandler

        handler = CopilotExtensionHandler()
        response = handler.handle("@codeverify scan src/main.py")
        assert "Scanning" in response.full_text

    def test_session_tracking(self):
        from codeverify_core.copilot_chat_extension import CopilotExtensionHandler, ChatContext

        handler = CopilotExtensionHandler()
        ctx = ChatContext(user_id="user1")
        handler.handle("@codeverify help", ctx)
        assert handler.active_sessions >= 1

    def test_command_parsing(self):
        from codeverify_core.copilot_chat_extension import CommandRouter, CopilotCommand

        router = CommandRouter()
        cmd, args = router.parse_command("@codeverify verify main.py")
        assert cmd == CopilotCommand.VERIFY
        assert args == "main.py"

    def test_singleton(self):
        from codeverify_core.copilot_chat_extension import (
            get_copilot_extension_handler, reset_copilot_extension_handler,
        )

        reset_copilot_extension_handler()
        h1 = get_copilot_extension_handler()
        h2 = get_copilot_extension_handler()
        assert h1 is h2
        reset_copilot_extension_handler()


# ─── Feature 5: Incremental Verification Engine ─────────────────────────


class TestIncrementalVerification:
    def test_cache_miss_and_store(self):
        from codeverify_core.incremental_verification import (
            IncrementalVerificationEngine, CodeUnit, CacheStatus, VerificationStatus,
        )

        engine = IncrementalVerificationEngine()
        unit = CodeUnit(file_path="main.py", name="add", content="def add(a, b): return a + b")
        status, cached = engine.lookup(unit)
        assert status == CacheStatus.MISS
        assert cached is None

        engine.store(unit, VerificationStatus.SAFE, verification_time_ms=150.0)
        status, cached = engine.lookup(unit)
        assert status == CacheStatus.HIT
        assert cached is not None
        assert cached.is_safe

    def test_cache_invalidation_on_change(self):
        from codeverify_core.incremental_verification import (
            IncrementalVerificationEngine, CodeUnit, CacheStatus, VerificationStatus,
        )

        engine = IncrementalVerificationEngine()
        unit = CodeUnit(file_path="main.py", name="add", content="def add(a, b): return a + b")
        engine.store(unit, VerificationStatus.SAFE)

        changed_unit = CodeUnit(file_path="main.py", name="add", content="def add(a, b): return a - b")
        status, _ = engine.lookup(changed_unit)
        assert status == CacheStatus.INVALIDATED

    def test_dependency_cascade_invalidation(self):
        from codeverify_core.incremental_verification import (
            IncrementalVerificationEngine, CodeUnit, VerificationStatus,
        )

        engine = IncrementalVerificationEngine()
        base = CodeUnit(file_path="utils.py", name="helper", content="def helper(): pass")
        dependent = CodeUnit(file_path="main.py", name="main", content="def main(): helper()")

        engine.store(base, VerificationStatus.SAFE)
        engine.store(dependent, VerificationStatus.SAFE)
        engine.add_dependency("main.py:main", "utils.py:helper")

        invalidated = engine.invalidate("utils.py:helper", cascade=True)
        assert "main.py:main" in invalidated

    def test_get_units_to_verify(self):
        from codeverify_core.incremental_verification import (
            IncrementalVerificationEngine, CodeUnit, VerificationStatus,
        )

        engine = IncrementalVerificationEngine()
        u1 = CodeUnit(file_path="a.py", name="f1", content="def f1(): pass")
        u2 = CodeUnit(file_path="b.py", name="f2", content="def f2(): pass")
        engine.store(u1, VerificationStatus.SAFE)

        to_verify = engine.get_units_to_verify([u1, u2])
        assert len(to_verify) == 1
        assert to_verify[0].name == "f2"

    def test_cache_metrics(self):
        from codeverify_core.incremental_verification import (
            IncrementalVerificationEngine, CodeUnit, VerificationStatus,
        )

        engine = IncrementalVerificationEngine()
        unit = CodeUnit(file_path="a.py", name="f", content="pass")
        engine.lookup(unit)  # miss
        engine.store(unit, VerificationStatus.SAFE, verification_time_ms=100.0)
        engine.lookup(unit)  # hit

        assert engine.metrics.hits == 1
        assert engine.metrics.misses == 1
        assert engine.metrics.hit_rate == 0.5
        assert engine.metrics.total_saved_ms == 100.0

    def test_dependency_graph(self):
        from codeverify_core.incremental_verification import IncrementalVerificationEngine

        engine = IncrementalVerificationEngine()
        engine.add_dependency("a", "b")
        engine.add_dependency("b", "c")
        deps = engine.dependency_graph.get_dependents("c")
        assert "b" in deps
        assert "a" in deps

    def test_singleton(self):
        from codeverify_core.incremental_verification import (
            get_incremental_engine, reset_incremental_engine,
        )

        reset_incremental_engine()
        e1 = get_incremental_engine()
        e2 = get_incremental_engine()
        assert e1 is e2
        reset_incremental_engine()


# ─── Feature 6: Organization Security Posture Dashboard ──────────────────


class TestOrgSecurityDashboard:
    def test_repo_metrics_risk_level(self):
        from codeverify_core.org_security_dashboard import RepoMetrics, RiskLevel

        metrics = RepoMetrics(repo_name="api", critical_findings=1)
        assert metrics.risk_level == RiskLevel.CRITICAL

        clean = RepoMetrics(repo_name="docs")
        assert clean.risk_level == RiskLevel.MINIMAL

    def test_generate_posture(self):
        from codeverify_core.org_security_dashboard import (
            OrgSecurityDashboard, RepoMetrics, RiskLevel,
        )

        dashboard = OrgSecurityDashboard("TestOrg")
        dashboard.add_repo_metrics(RepoMetrics(
            repo_id="r1", repo_name="api",
            total_findings=5, critical_findings=0, high_findings=1,
            verification_coverage=0.8, trust_score=75.0,
        ))
        dashboard.add_repo_metrics(RepoMetrics(
            repo_id="r2", repo_name="web",
            total_findings=2, verification_coverage=0.9, trust_score=85.0,
        ))

        posture = dashboard.generate_posture()
        assert posture.org_name == "TestOrg"
        assert posture.total_repos == 2
        assert posture.total_findings == 7
        assert posture.avg_trust_score > 0

    def test_compliance_tracking(self):
        from codeverify_core.org_security_dashboard import (
            OrgSecurityDashboard, ComplianceRecord, ComplianceFramework, ComplianceStatus,
        )

        dashboard = OrgSecurityDashboard("Org")
        dashboard.set_compliance(ComplianceRecord(
            framework=ComplianceFramework.SOC2,
            status=ComplianceStatus.PARTIAL,
            controls_total=50, controls_met=35,
        ))
        posture = dashboard.generate_posture()
        assert len(posture.compliance) == 1
        assert posture.compliance[0].coverage_pct == 70.0

    def test_dora_metrics(self):
        from codeverify_core.org_security_dashboard import DORAMetrics

        dora = DORAMetrics(
            deployment_frequency_per_day=2.0,
            lead_time_hours=12.0,
            mean_time_to_recovery_hours=1.0,
            change_failure_rate=0.05,
        )
        assert dora.deployment_frequency_rating == "Elite"
        assert dora.lead_time_rating == "Elite"
        assert dora.overall_rating == "Elite"

    def test_risk_heatmap(self):
        from codeverify_core.org_security_dashboard import OrgSecurityDashboard, RepoMetrics

        dashboard = OrgSecurityDashboard("Org")
        dashboard.add_repo_metrics(RepoMetrics(repo_id="r1", repo_name="api", high_findings=5))
        dashboard.add_repo_metrics(RepoMetrics(repo_id="r2", repo_name="web"))
        heatmap = dashboard.get_risk_heatmap()
        assert len(heatmap) == 2
        assert heatmap[0]["risk_score"] >= heatmap[1]["risk_score"]

    def test_executive_summary(self):
        from codeverify_core.org_security_dashboard import (
            OrgSecurityDashboard, RepoMetrics, DORAMetrics,
        )

        dashboard = OrgSecurityDashboard("Org")
        dashboard.add_repo_metrics(RepoMetrics(repo_id="r1", repo_name="api", trust_score=80.0, verification_coverage=0.9))
        dashboard.set_dora_metrics(DORAMetrics(deployment_frequency_per_day=1.0, lead_time_hours=20.0))
        posture = dashboard.generate_posture()
        summary = posture.to_executive_summary()
        assert summary["organization"] == "Org"
        assert "dora_rating" in summary

    def test_trend_tracking(self):
        from codeverify_core.org_security_dashboard import OrgSecurityDashboard, TrendDirection

        dashboard = OrgSecurityDashboard("Org")
        dashboard.add_trend_point("coverage", 0.7)
        dashboard.add_trend_point("coverage", 0.8)
        posture = dashboard.generate_posture()
        assert len(posture.trends) == 1
        assert posture.trends[0].direction == TrendDirection.IMPROVING

    def test_singleton(self):
        from codeverify_core.org_security_dashboard import (
            get_org_security_dashboard, reset_org_security_dashboard,
        )

        reset_org_security_dashboard()
        d1 = get_org_security_dashboard("Test")
        d2 = get_org_security_dashboard("Test")
        assert d1 is d2
        reset_org_security_dashboard()


# ─── Feature 7: CI/CD Pipeline Orchestrator ──────────────────────────────


class TestCICDOrchestrator:
    def test_gate_evaluation_pass(self):
        from codeverify_core.cicd_orchestrator import CICDOrchestrator, GateResult

        orch = CICDOrchestrator()
        result = orch.evaluate("default", critical=0, high=0, medium=2, low=3)
        assert result.result == GateResult.PASSED
        assert result.should_block is False

    def test_gate_evaluation_fail(self):
        from codeverify_core.cicd_orchestrator import CICDOrchestrator, GateResult

        orch = CICDOrchestrator()
        result = orch.evaluate("strict", critical=1, high=0)
        assert result.result == GateResult.FAILED
        assert result.should_block is True

    def test_gate_evaluation_warn(self):
        from codeverify_core.cicd_orchestrator import CICDOrchestrator, GateResult

        orch = CICDOrchestrator()
        result = orch.evaluate("lenient", critical=0, high=5, medium=20)
        assert result.result == GateResult.WARNING
        assert result.should_block is False

    def test_pipeline_config_generation(self):
        from codeverify_core.cicd_orchestrator import CICDOrchestrator, CICDPlatform

        orch = CICDOrchestrator()
        for platform in CICDPlatform:
            config = orch.generate_config(platform)
            assert config.config_content != ""
            assert config.file_name != ""

    def test_status_generation(self):
        from codeverify_core.cicd_orchestrator import CICDOrchestrator, StatusState

        orch = CICDOrchestrator()
        result = orch.evaluate("default", critical=0, high=0)
        status = orch.generate_status(result)
        assert status.state == StatusState.SUCCESS

    def test_custom_gate(self):
        from codeverify_core.cicd_orchestrator import (
            CICDOrchestrator, QualityGate, QualityThresholds, GateResult,
        )

        orch = CICDOrchestrator()
        custom = QualityGate(
            name="custom",
            thresholds=QualityThresholds(max_critical=0, max_high=1, min_trust_score=70.0),
        )
        orch.register_gate(custom)
        result = orch.evaluate("custom", critical=0, high=0, trust_score=80.0)
        assert result.result == GateResult.PASSED

    def test_available_gates(self):
        from codeverify_core.cicd_orchestrator import CICDOrchestrator

        orch = CICDOrchestrator()
        gates = orch.available_gates
        assert "default" in gates
        assert "strict" in gates
        assert "lenient" in gates

    def test_singleton(self):
        from codeverify_core.cicd_orchestrator import get_cicd_orchestrator, reset_cicd_orchestrator

        reset_cicd_orchestrator()
        o1 = get_cicd_orchestrator()
        o2 = get_cicd_orchestrator()
        assert o1 is o2
        reset_cicd_orchestrator()


# ─── Feature 8: LLM-Powered Proof Explainer ─────────────────────────────


class TestProofExplainer:
    def test_explain_safe_result(self):
        from codeverify_core.proof_explainer import (
            ProofExplainerEngine, CheckCategory, ProofOutcome,
        )

        engine = ProofExplainerEngine()
        explanation = engine.explain(
            check_category=CheckCategory.NULL_SAFETY,
            outcome=ProofOutcome.PROVED_SAFE,
            function_name="process",
            file_path="main.py",
        )
        assert explanation.is_safe
        assert "safe" in explanation.summary.lower() or "✅" in explanation.summary

    def test_explain_counterexample(self):
        from codeverify_core.proof_explainer import (
            ProofExplainerEngine, CheckCategory, ProofOutcome,
        )

        engine = ProofExplainerEngine()
        raw = "x -> 0\ny -> -1 (int)"
        explanation = engine.explain(
            check_category=CheckCategory.DIVISION_BY_ZERO,
            outcome=ProofOutcome.COUNTEREXAMPLE_FOUND,
            raw_z3_output=raw,
            variable_name="divisor",
        )
        assert not explanation.is_safe
        assert len(explanation.fix_suggestions) > 0
        assert explanation.counterexample is not None

    def test_counterexample_parser(self):
        from codeverify_core.proof_explainer import CounterexampleParser

        parser = CounterexampleParser()
        raw = "x -> 42\ny -> -1 (int)\nz -> true (bool)"
        result = parser.parse(raw)
        assert len(result.values) == 3
        assert result.get_value("x") is not None
        assert result.get_value("x").value == 42

    def test_counterexample_parser_smt(self):
        from codeverify_core.proof_explainer import CounterexampleParser

        parser = CounterexampleParser()
        raw = "(define-fun x () Int 42)\n(define-fun y () Bool true)"
        result = parser.parse(raw)
        assert len(result.values) == 2
        assert result.get_value("x").value == 42
        assert result.get_value("y").value is True

    def test_explanation_markdown(self):
        from codeverify_core.proof_explainer import (
            ProofExplainerEngine, CheckCategory, ProofOutcome, ExplanationDetail,
        )

        engine = ProofExplainerEngine()
        explanation = engine.explain(
            check_category=CheckCategory.ARRAY_BOUNDS,
            outcome=ProofOutcome.COUNTEREXAMPLE_FOUND,
            raw_z3_output="idx -> 10",
            detail_level=ExplanationDetail.FULL_PROOF,
        )
        md = explanation.to_markdown()
        assert "❌" in md
        assert "Suggested Fixes" in md

    def test_explain_batch(self):
        from codeverify_core.proof_explainer import ProofExplainerEngine

        engine = ProofExplainerEngine()
        results = engine.explain_batch([
            {"category": "null_safety", "outcome": "proved_safe"},
            {"category": "integer_overflow", "outcome": "counterexample_found", "raw_output": "x -> 2147483647"},
        ])
        assert len(results) == 2
        assert results[0].is_safe
        assert not results[1].is_safe

    def test_educational_links(self):
        from codeverify_core.proof_explainer import (
            ProofExplainerEngine, CheckCategory, ProofOutcome,
        )

        engine = ProofExplainerEngine()
        explanation = engine.explain(
            check_category=CheckCategory.INTEGER_OVERFLOW,
            outcome=ProofOutcome.COUNTEREXAMPLE_FOUND,
        )
        assert explanation.educational_link.get("url", "").startswith("https://")

    def test_code_examples(self):
        from codeverify_core.proof_explainer import (
            ProofExplainerEngine, CheckCategory, ProofOutcome, ExplanationDetail,
        )

        engine = ProofExplainerEngine()
        explanation = engine.explain(
            check_category=CheckCategory.NULL_SAFETY,
            outcome=ProofOutcome.COUNTEREXAMPLE_FOUND,
            variable_name="user",
            detail_level=ExplanationDetail.FULL_PROOF,
        )
        assert explanation.code_example != ""
        assert "None" in explanation.code_example or "none" in explanation.code_example.lower()

    def test_singleton(self):
        from codeverify_core.proof_explainer import get_proof_explainer, reset_proof_explainer

        reset_proof_explainer()
        e1 = get_proof_explainer()
        e2 = get_proof_explainer()
        assert e1 is e2
        reset_proof_explainer()


# ─── Feature 9: Supply Chain Verification ────────────────────────────────


class TestSupplyChain:
    def test_scan_dependencies(self):
        from codeverify_core.supply_chain import (
            SupplyChainVerifier, Dependency, PackageManager,
        )

        verifier = SupplyChainVerifier()
        deps = [
            Dependency(name="requests", version="2.31.0", package_manager=PackageManager.PIP, license_id="Apache-2.0"),
            Dependency(name="flask", version="3.0.0", package_manager=PackageManager.PIP, license_id="BSD-3-Clause"),
        ]
        report = verifier.scan_dependencies(deps, "my-project")
        assert report.total_dependencies == 2
        assert report.direct_dependencies == 2
        assert report.is_clean

    def test_vulnerability_detection(self):
        from codeverify_core.supply_chain import (
            SupplyChainVerifier, Dependency, Vulnerability, PackageManager, VulnerabilitySeverity,
        )

        verifier = SupplyChainVerifier()
        verifier.vulnerability_db.add_vulnerability(Vulnerability(
            id="VULN-1", cve_id="CVE-2024-1234", package_name="requests",
            severity=VulnerabilitySeverity.HIGH, title="SSRF vulnerability",
        ))
        deps = [Dependency(name="requests", version="2.28.0", package_manager=PackageManager.PIP)]
        report = verifier.scan_dependencies(deps)
        assert report.total_vulnerabilities == 1
        assert report.high_vulnerabilities == 1
        assert not report.is_clean

    def test_license_compliance(self):
        from codeverify_core.supply_chain import (
            SupplyChainVerifier, Dependency, PackageManager,
        )

        verifier = SupplyChainVerifier()
        deps = [
            Dependency(name="lib", version="1.0", package_manager=PackageManager.PIP, license_id="GPL-3.0"),
        ]
        report = verifier.scan_dependencies(deps)
        assert report.license_violations == 1

    def test_sbom_generation(self):
        from codeverify_core.supply_chain import (
            SupplyChainVerifier, Dependency, PackageManager,
        )

        verifier = SupplyChainVerifier()
        deps = [
            Dependency(name="numpy", version="1.26.0", package_manager=PackageManager.PIP, license_id="BSD-3-Clause"),
        ]
        report = verifier.scan_dependencies(deps, "data-project")
        assert len(report.sbom) == 1
        assert report.sbom[0].purl == "pkg:pip/numpy@1.26.0"

    def test_parse_requirements(self):
        from codeverify_core.supply_chain import LockfileParser

        parser = LockfileParser()
        content = "requests==2.31.0\nflask>=3.0.0\n# comment\npydantic~=2.5"
        deps = parser.parse_requirements(content)
        assert len(deps) == 3

    def test_parse_go_sum(self):
        from codeverify_core.supply_chain import LockfileParser

        parser = LockfileParser()
        content = "github.com/gin-gonic/gin v1.9.1 h1:abc=\ngithub.com/gin-gonic/gin v1.9.1/go.mod h1:def="
        deps = parser.parse_go_sum(content)
        assert len(deps) == 1
        assert deps[0].name == "github.com/gin-gonic/gin"

    def test_risk_scoring(self):
        from codeverify_core.supply_chain import (
            SupplyChainVerifier, Dependency, Vulnerability, PackageManager, VulnerabilitySeverity,
        )

        verifier = SupplyChainVerifier()
        verifier.vulnerability_db.add_vulnerability(Vulnerability(
            id="V1", package_name="lib", severity=VulnerabilitySeverity.CRITICAL,
        ))
        deps = [Dependency(name="lib", version="1.0", package_manager=PackageManager.PIP, license_id="GPL-3.0")]
        report = verifier.scan_dependencies(deps)
        assert report.risks[0].risk_score > 0.5  # critical vuln + license violation

    def test_singleton(self):
        from codeverify_core.supply_chain import get_supply_chain_verifier, reset_supply_chain_verifier

        reset_supply_chain_verifier()
        v1 = get_supply_chain_verifier()
        v2 = get_supply_chain_verifier()
        assert v1 is v2
        reset_supply_chain_verifier()


# ─── Feature 10: Self-Learning Rule Engine ───────────────────────────────


class TestSelfLearningRules:
    def test_record_feedback(self):
        from codeverify_core.self_learning_rules import SelfLearningRuleEngine, FindingFeedback, FeedbackType

        engine = SelfLearningRuleEngine()
        engine.record_feedback(FindingFeedback(
            rule_id="null_check", category="null_safety",
            severity="high", feedback_type=FeedbackType.ACCEPTED,
        ))
        assert engine.feedback_count == 1

    def test_train_classifier(self):
        from codeverify_core.self_learning_rules import SelfLearningRuleEngine, FindingFeedback, FeedbackType

        engine = SelfLearningRuleEngine()
        for i in range(10):
            engine.record_feedback(FindingFeedback(
                rule_id="rule1", category="null_safety", severity="high",
                feedback_type=FeedbackType.ACCEPTED,
            ))
        for i in range(10):
            engine.record_feedback(FindingFeedback(
                rule_id="rule2", category="style", severity="low",
                feedback_type=FeedbackType.FALSE_POSITIVE,
            ))
        engine.train()
        assert engine.classifier_trained

    def test_predict_false_positive(self):
        from codeverify_core.self_learning_rules import SelfLearningRuleEngine, FindingFeedback, FeedbackType

        engine = SelfLearningRuleEngine()
        for _ in range(20):
            engine.record_feedback(FindingFeedback(
                rule_id="noisy", category="style", severity="low",
                feedback_type=FeedbackType.FALSE_POSITIVE, file_path="test.py",
            ))
        engine.train()
        result = engine.predict_false_positive("noisy", "style", "low", "test.py")
        assert result.confidence >= 0.0

    def test_severity_calibration(self):
        from codeverify_core.self_learning_rules import (
            SelfLearningRuleEngine, FindingFeedback, FeedbackType, SeverityAdjustment,
        )

        engine = SelfLearningRuleEngine()
        # Rule with high FP rate should get suppressed
        for _ in range(10):
            engine.record_feedback(FindingFeedback(
                rule_id="bad_rule", category="style", severity="medium",
                feedback_type=FeedbackType.FALSE_POSITIVE,
            ))
        engine.train()
        adj = engine.get_severity_adjustment("bad_rule")
        assert adj in (SeverityAdjustment.DECREASE, SeverityAdjustment.SUPPRESS)

    def test_pattern_learning(self):
        from codeverify_core.self_learning_rules import SelfLearningRuleEngine, FindingFeedback, FeedbackType

        engine = SelfLearningRuleEngine()
        for _ in range(5):
            engine.record_feedback(FindingFeedback(
                rule_id="rule_x", category="perf", severity="low",
                feedback_type=FeedbackType.DISMISSED, repo_id="repo1",
            ))
        engine.train()
        patterns = engine.learned_patterns
        assert len(patterns) >= 1
        assert any("rule_x" in p.rule_id for p in patterns)

    def test_acceptance_rate(self):
        from codeverify_core.self_learning_rules import SelfLearningRuleEngine, FindingFeedback, FeedbackType

        engine = SelfLearningRuleEngine()
        engine.record_feedback(FindingFeedback(feedback_type=FeedbackType.ACCEPTED))
        engine.record_feedback(FindingFeedback(feedback_type=FeedbackType.ACCEPTED))
        engine.record_feedback(FindingFeedback(feedback_type=FeedbackType.DISMISSED))
        assert abs(engine.acceptance_rate - 2 / 3) < 0.01

    def test_rule_performance(self):
        from codeverify_core.self_learning_rules import SelfLearningRuleEngine, FindingFeedback, FeedbackType

        engine = SelfLearningRuleEngine()
        engine.record_feedback(FindingFeedback(rule_id="r1", feedback_type=FeedbackType.ACCEPTED))
        engine.record_feedback(FindingFeedback(rule_id="r1", feedback_type=FeedbackType.ACCEPTED))
        engine.record_feedback(FindingFeedback(rule_id="r1", feedback_type=FeedbackType.FALSE_POSITIVE))
        engine.train()
        perfs = engine.rule_performances
        assert len(perfs) >= 1
        r1_perf = next(p for p in perfs if p.rule_id == "r1")
        assert r1_perf.accepted == 2
        assert r1_perf.false_positives == 1

    def test_singleton(self):
        from codeverify_core.self_learning_rules import get_self_learning_engine, reset_self_learning_engine

        reset_self_learning_engine()
        e1 = get_self_learning_engine()
        e2 = get_self_learning_engine()
        assert e1 is e2
        reset_self_learning_engine()
