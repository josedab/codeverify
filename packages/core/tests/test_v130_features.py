"""Tests for v1.3.0 next-gen features.

Covers all 10 next-gen features:
1. GitHub Marketplace & One-Click Install
2. Streaming IDE Verification (Language Server)
3. Verification Insights API (GraphQL)
4. AI Autofix with Verified Patches
5. Organization Security Posture Score
6. Copilot Extension (Chat + Agent)
7. Multi-Tenant Hosted SaaS Platform
8. Proof Artifact Marketplace
9. Compliance-as-Code Engine
10. Performance & Cost Dashboard
"""


# --- Feature 1: GitHub Marketplace & One-Click Install ---


class TestMarketplaceListing:
    def test_plan_limits(self):
        from codeverify_core.marketplace_listing import MarketplacePlan, PlanLimits

        free = PlanLimits.for_plan(MarketplacePlan.FREE)
        assert free.verifications_per_month == 50
        assert free.max_repos == 3
        assert free.private_repos is False

        enterprise = PlanLimits.for_plan(MarketplacePlan.ENTERPRISE)
        assert enterprise.verifications_per_month == -1
        assert enterprise.private_repos is True

    def test_config_auto_generator(self):
        from codeverify_core.marketplace_listing import ConfigAutoGenerator

        gen = ConfigAutoGenerator()
        files = ["src/main.py", "src/utils.ts", "go.mod", "Makefile"]
        languages = gen.detect_languages(files)
        assert "python" in languages
        assert "typescript" in languages

        frameworks = gen.detect_frameworks(files)
        assert "go-module" in frameworks
        assert "make" in frameworks

    def test_onboarding_config_generation(self):
        from codeverify_core.marketplace_listing import ConfigAutoGenerator

        gen = ConfigAutoGenerator()
        config = gen.generate_config("myrepo", ["app.py", "test.py", "utils.py"])
        assert "python" in config.config_yaml
        assert config.detected_languages == ["python"]
        assert "codeverify-action" in config.workflow_yaml

    def test_installation_lifecycle(self):
        from codeverify_core.marketplace_listing import (
            InstallationStatus,
            MarketplacePlan,
            MarketplaceService,
            OnboardingPhase,
        )

        svc = MarketplaceService()
        inst = svc.handle_installation(
            github_installation_id=12345,
            account_login="testorg",
            repos=["testorg/repo1"],
            plan=MarketplacePlan.PRO,
        )
        assert inst.status == InstallationStatus.ACTIVE
        assert inst.plan == MarketplacePlan.PRO

        config = svc.one_click_onboard(inst.id, ["main.py", "utils.py"])
        assert config.detected_languages == ["python"]

        updated = svc.get_installation(inst.id)
        assert updated.onboarding_phase == OnboardingPhase.BASELINE_SCAN_QUEUED

        svc.complete_onboarding(inst.id)
        updated = svc.get_installation(inst.id)
        assert updated.onboarding_phase == OnboardingPhase.ONBOARDING_COMPLETE

    def test_usage_metering(self):
        from codeverify_core.marketplace_listing import (
            MarketplacePlan,
            PlanLimits,
            UsageMeter,
            UsageMetric,
        )

        meter = UsageMeter()
        meter.record_usage("inst1", UsageMetric.VERIFICATIONS, 5)
        usage = meter.get_usage("inst1")
        assert usage.metrics["verifications"] == 5

        limits = PlanLimits.for_plan(MarketplacePlan.FREE)
        assert meter.check_limit("inst1", UsageMetric.VERIFICATIONS, limits) is True
        assert meter.get_remaining("inst1", UsageMetric.VERIFICATIONS, limits) == 45

    def test_analytics(self):
        from codeverify_core.marketplace_listing import MarketplaceService

        svc = MarketplaceService()
        svc.handle_installation(1, "org1", repos=["r1", "r2"])
        svc.handle_installation(2, "org2", repos=["r3"])
        analytics = svc.get_analytics()
        assert analytics["active_installations"] == 2
        assert analytics["total_repos"] == 3


# --- Feature 2: Streaming IDE Verification ---


class TestStreamingIDEVerification:
    def test_code_block_parsing(self):
        from codeverify_core.streaming_ide import CodeBlockParser

        parser = CodeBlockParser()
        code = "def foo():\n    return 1\n\ndef bar(x):\n    return x + 1\n"
        blocks = parser.parse_blocks("test.py", code, "python")
        assert len(blocks) == 2
        assert blocks[0].name == "foo"
        assert blocks[1].name == "bar"

    def test_content_hash_caching(self):
        from codeverify_core.streaming_ide import CodeBlock

        block = CodeBlock(content="def foo(): pass")
        h1 = block.compute_hash()
        assert len(h1) == 16

        block2 = CodeBlock(content="def foo(): pass")
        h2 = block2.compute_hash()
        assert h1 == h2

    def test_verification_cache(self):
        from codeverify_core.streaming_ide import (
            BlockVerificationResult,
            IncrementalVerificationCache,
            ProofStatus,
        )

        cache = IncrementalVerificationCache()
        result = BlockVerificationResult(
            block_id="b1", status=ProofStatus.VERIFIED, content_hash="abc123"
        )
        cache.put("abc123", result)
        cached = cache.get("abc123")
        assert cached is not None
        assert cached.cached is True
        assert cache.stats["hits"] == 1

    def test_streaming_verification(self):
        from codeverify_core.streaming_ide import (
            StreamingIDEVerificationService,
        )

        svc = StreamingIDEVerificationService()
        svc.open_session("test.py")
        code = "def safe():\n    return 42\n\ndef risky():\n    eval('print(1)')\n"
        result = svc.verify_file("test.py", code, "python")
        assert len(result.blocks) == 2
        assert result.diagnostic_count >= 1  # eval() should trigger

    def test_dependency_tracking(self):
        from codeverify_core.streaming_ide import StreamingIDEVerificationService

        svc = StreamingIDEVerificationService()
        svc.register_dependency("b.py", "a.py")
        svc.register_dependency("c.py", "a.py")
        affected = svc.get_affected_files("a.py")
        assert "b.py" in affected
        assert "c.py" in affected

    def test_incremental_cache_hit(self):
        from codeverify_core.streaming_ide import StreamingIDEVerificationService

        svc = StreamingIDEVerificationService()
        code = "def hello():\n    return 'world'\n"
        svc.verify_file("test.py", code)
        r2 = svc.verify_file("test.py", code)
        assert r2.cache_hit_rate == 1.0


# --- Feature 3: Verification Insights API (GraphQL) ---


class TestGraphQLInsightsAPI:
    def test_api_key_lifecycle(self):
        from codeverify_core.graphql_insights import (
            ApiKeyScope,
            GraphQLInsightsService,
        )

        svc = GraphQLInsightsService()
        key, raw = svc.create_api_key("test-key", "user1", [ApiKeyScope.READ])
        assert raw.startswith("cv_")
        assert key.is_active is True

        authed = svc.authenticate(raw)
        assert authed is not None
        assert authed.id == key.id

        svc.revoke_api_key(key.id)
        assert svc.authenticate(raw) is None

    def test_rate_limiting(self):
        from codeverify_core.graphql_insights import (
            RateLimitConfig,
            RateLimiter,
            RateLimitTier,
        )

        limiter = RateLimiter()
        config = RateLimitConfig.for_tier(RateLimitTier.FREE)
        assert config.requests_per_minute == 10

        for _ in range(10):
            allowed, _ = limiter.check_and_consume("key1", config)
            assert allowed is True

        allowed, reason = limiter.check_and_consume("key1", config)
        assert allowed is False
        assert "per-minute" in reason

    def test_query_analysis(self):
        from codeverify_core.graphql_insights import QueryAnalyzer, QueryType

        analyzer = QueryAnalyzer()
        query = "{ analysis(id: 1) { id status findings { severity } } }"
        parsed = analyzer.analyze(query)
        assert parsed.query_type == QueryType.ANALYSIS
        assert parsed.depth >= 2

    def test_execute_query_auth(self):
        from codeverify_core.graphql_insights import GraphQLInsightsService

        svc = GraphQLInsightsService()
        resp = svc.execute_query("invalid_key", "{ analysis { id } }")
        assert resp.has_errors
        assert resp.errors[0]["extensions"]["code"] == "UNAUTHENTICATED"

    def test_webhook_subscription(self):
        from codeverify_core.graphql_insights import (
            GraphQLInsightsService,
            WebhookEvent,
        )

        svc = GraphQLInsightsService()
        wh = svc.create_webhook(
            "user1", "https://example.com/hook", [WebhookEvent.ANALYSIS_COMPLETED]
        )
        assert wh.is_active is True

        deliveries = svc.deliver_webhook(WebhookEvent.ANALYSIS_COMPLETED, {"analysis_id": "123"})
        assert len(deliveries) == 1
        assert deliveries[0].delivered is True

    def test_execute_query_success(self):
        from codeverify_core.graphql_insights import ApiKeyScope, GraphQLInsightsService

        svc = GraphQLInsightsService()
        _, raw = svc.create_api_key("key", "u1", [ApiKeyScope.READ])
        resp = svc.execute_query(raw, "{ analysis(id: 1) { id } }", {"id": "1"})
        assert not resp.has_errors


# --- Feature 4: AI Autofix with Verified Patches ---


class TestAutofixVerifiedPatches:
    def test_fix_generation(self):
        from codeverify_core.autofix_verified_patches import (
            AutofixVerifiedService,
            FindingCategory,
            FixableFinding,
            FixStatus,
        )

        svc = AutofixVerifiedService()
        finding = FixableFinding(
            file_path="app.py",
            line=10,
            category=FindingCategory.NULL_SAFETY,
            message="Potential null dereference",
            code_snippet="result = obj.method()",
        )
        fixes = svc.generate_fix(finding)
        assert len(fixes) > 0
        assert fixes[0].status == FixStatus.VERIFIED

    def test_fix_verification(self):
        from codeverify_core.autofix_verified_patches import (
            FindingCategory,
            FixCandidate,
            FixVerifier,
        )

        verifier = FixVerifier()
        fix = FixCandidate(
            original_code="result = a / b",
            fixed_code="result = a / b if b != 0 else 0",
        )
        result = verifier.verify(fix, FindingCategory.DIVISION_BY_ZERO)
        assert result.passed is True
        assert "zero_guard" in result.checks_passed

    def test_pr_suggestion_generation(self):
        from codeverify_core.autofix_verified_patches import (
            AutofixVerifiedService,
            FindingCategory,
            FixableFinding,
            FixCandidate,
            FixConfidence,
        )

        svc = AutofixVerifiedService()
        finding = FixableFinding(
            file_path="app.py",
            line=5,
            message="Division by zero",
            category=FindingCategory.DIVISION_BY_ZERO,
        )
        fix = FixCandidate(
            fixed_code="x / y if y != 0 else 0",
            description="Add zero guard",
            confidence=FixConfidence.HIGH,
            verification_details="All checks passed",
        )
        suggestion = svc.create_pr_suggestion(finding, fix)
        assert "CodeVerify Autofix" in suggestion.comment_body
        assert suggestion.file_path == "app.py"

    def test_batch_fix(self):
        from codeverify_core.autofix_verified_patches import (
            AutofixVerifiedService,
            FindingCategory,
            FixableFinding,
        )

        svc = AutofixVerifiedService()
        findings = [
            FixableFinding(
                file_path="a.py",
                line=1,
                category=FindingCategory.NULL_SAFETY,
                code_snippet="result = obj.method()",
            ),
            FixableFinding(
                file_path="b.py",
                line=5,
                category=FindingCategory.DIVISION_BY_ZERO,
                code_snippet="result = a / b",
            ),
        ]
        result = svc.batch_fix(findings)
        assert result.total_findings == 2
        assert result.fixes_generated > 0

    def test_safety_guardrails(self):
        from codeverify_core.autofix_verified_patches import (
            AutofixVerifiedService,
            FindingCategory,
            FixableFinding,
            SafetyGuardrails,
        )

        guardrails = SafetyGuardrails(
            allowed_categories=[FindingCategory.NULL_SAFETY],
        )
        svc = AutofixVerifiedService(guardrails=guardrails)
        finding = FixableFinding(category=FindingCategory.SECURITY)
        fixes = svc.generate_fix(finding)
        assert len(fixes) == 0  # SECURITY not in allowed categories


# --- Feature 5: Organization Security Posture Score ---


class TestOrgSecurityPosture:
    def test_posture_calculation(self):
        from codeverify_core.org_posture import (
            OrgSecurityPostureService,
            RepositoryMetrics,
        )

        svc = OrgSecurityPostureService(org_name="TestOrg")
        svc.set_repo_metrics(
            RepositoryMetrics(
                repo_id="r1",
                repo_name="api",
                verification_coverage=0.8,
                fix_rate=0.9,
                critical_findings=0,
                high_findings=1,
                is_compliant=True,
            )
        )
        svc.set_repo_metrics(
            RepositoryMetrics(
                repo_id="r2",
                repo_name="web",
                verification_coverage=0.6,
                fix_rate=0.7,
                critical_findings=0,
                high_findings=0,
                is_compliant=True,
            )
        )
        score = svc.calculate_posture()
        assert 0 <= score.overall_score <= 100
        assert score.coverage_score > 0
        assert score.compliance_score == 100.0

    def test_dora_metrics(self):
        from codeverify_core.org_posture import DORAMetricLevel, DORAMetrics

        dora = DORAMetrics(
            deployment_frequency_per_day=2.0,
            lead_time_hours=12.0,
            mean_time_to_restore_hours=0.5,
            change_failure_rate=0.03,
        )
        assert dora.deployment_level == DORAMetricLevel.ELITE
        assert dora.lead_time_level == DORAMetricLevel.ELITE
        assert dora.mttr_level == DORAMetricLevel.ELITE
        assert dora.overall_level == DORAMetricLevel.ELITE

    def test_heatmap(self):
        from codeverify_core.org_posture import (
            OrgSecurityPostureService,
            RepositoryMetrics,
        )

        svc = OrgSecurityPostureService()
        svc.set_repo_metrics(
            RepositoryMetrics(
                repo_id="r1",
                repo_name="critical-repo",
                critical_findings=3,
            )
        )
        heatmap = svc.get_heatmap()
        assert len(heatmap) == 1
        assert heatmap[0].risk_level.value == "critical"

    def test_trend_detection(self):
        from codeverify_core.org_posture import PostureScore, PostureTrend, TrendDetector

        detector = TrendDetector()
        history = [
            PostureScore(overall_score=60.0),
            PostureScore(overall_score=62.0),
            PostureScore(overall_score=65.0),
            PostureScore(overall_score=70.0),
            PostureScore(overall_score=75.0),
        ]
        trend = detector.detect(history)
        assert trend == PostureTrend.IMPROVING

    def test_executive_digest(self):
        from codeverify_core.org_posture import (
            OrgSecurityPostureService,
            RepositoryMetrics,
        )

        svc = OrgSecurityPostureService(org_name="ACME Corp")
        svc.set_repo_metrics(
            RepositoryMetrics(
                repo_id="r1",
                repo_name="api",
                verification_coverage=0.9,
                fix_rate=0.95,
                is_compliant=True,
            )
        )
        digest = svc.generate_digest()
        assert "ACME Corp" in digest.summary_markdown
        assert digest.posture_score is not None


# --- Feature 6: Copilot Extension (Chat + Agent) ---


class TestCopilotExtension:
    def test_command_routing(self):
        from codeverify_core.copilot_chat_agent import CommandRouter, CopilotCommand

        router = CommandRouter()
        cmd, args = router.parse_command("/verify my code")
        assert cmd == CopilotCommand.VERIFY
        assert args == "my code"

        cmd, args = router.parse_command("just a question")
        assert cmd is None

    def test_verify_command(self):
        from codeverify_core.copilot_chat_agent import (
            ChatContext,
            CommandHandler,
        )

        handler = CommandHandler()
        ctx = ChatContext(
            file_path="app.py",
            selected_code="x = eval(input())",
            language="python",
        )
        resp = handler.handle_verify(ctx, "")
        assert "issue" in resp.content.lower()
        assert resp.metadata.get("issues_found", 0) > 0

    def test_help_command(self):
        from codeverify_core.copilot_chat_agent import CommandRouter

        router = CommandRouter()
        help_text = router.get_help_text()
        assert "/verify" in help_text
        assert "/fix" in help_text

    def test_session_management(self):
        from codeverify_core.copilot_chat_agent import (
            CopilotExtensionService,
            SessionState,
        )

        svc = CopilotExtensionService()
        session = svc.create_session()
        assert session.state == SessionState.ACTIVE

        resp = svc.process_message(session.id, "/help")
        assert "/verify" in resp.content

        session = svc.get_session(session.id)
        assert len(session.messages) == 2  # user + assistant

        svc.close_session(session.id)
        session = svc.get_session(session.id)
        assert session.state == SessionState.CLOSED

    def test_trust_score_command(self):
        from codeverify_core.copilot_chat_agent import (
            ChatContext,
            CommandHandler,
        )

        handler = CommandHandler()
        ctx = ChatContext(selected_code="x = 1\ny = 2\n")
        resp = handler.handle_trust_score(ctx, "")
        assert "Trust Score" in resp.content
        assert "score" in resp.metadata

    def test_fix_with_code_action(self):
        from codeverify_core.copilot_chat_agent import ChatContext, CommandHandler

        handler = CommandHandler()
        ctx = ChatContext(
            file_path="app.py",
            selected_code="result = eval(user_input)",
            cursor_line=5,
        )
        resp = handler.handle_fix(ctx, "")
        assert len(resp.code_actions) > 0
        assert "literal_eval" in resp.code_actions[0].new_text


# --- Feature 7: Multi-Tenant Hosted SaaS Platform ---


class TestHostedSaaS:
    def test_tenant_provisioning(self):
        from codeverify_core.hosted_saas import (
            HostedSaaSService,
            SaaSPlan,
            TenantStatus,
        )

        svc = HostedSaaSService()
        tenant = svc.create_tenant("Test Org", "admin@test.com", SaaSPlan.PRO)
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.plan == SaaSPlan.PRO

    def test_plan_config(self):
        from codeverify_core.hosted_saas import FeatureFlag, PlanConfig, SaaSPlan

        free = PlanConfig.for_plan(SaaSPlan.FREE)
        assert free.price_monthly_cents == 0
        assert FeatureFlag.SSO not in free.features

        enterprise = PlanConfig.for_plan(SaaSPlan.ENTERPRISE)
        assert FeatureFlag.SSO in enterprise.features
        assert enterprise.max_repos == -1

    def test_feature_gates(self):
        from codeverify_core.hosted_saas import (
            FeatureFlag,
            HostedSaaSService,
            SaaSPlan,
        )

        svc = HostedSaaSService()
        tenant = svc.create_tenant("Free Org", "user@free.com", SaaSPlan.FREE)
        assert svc.is_feature_enabled(tenant.id, FeatureFlag.FORMAL_VERIFICATION) is True
        assert svc.is_feature_enabled(tenant.id, FeatureFlag.SSO) is False

    def test_usage_metering(self):
        from codeverify_core.hosted_saas import HostedSaaSService, SaaSPlan

        svc = HostedSaaSService()
        tenant = svc.create_tenant("Org", "a@b.com", SaaSPlan.FREE)
        assert svc.record_usage(tenant.id, "verifications", 5) is True
        usage = svc.get_usage(tenant.id)
        assert usage.verifications == 5

    def test_invoice_generation(self):
        from codeverify_core.hosted_saas import HostedSaaSService, SaaSPlan

        svc = HostedSaaSService()
        tenant = svc.create_tenant("Pro Org", "a@b.com", SaaSPlan.PRO)
        invoice = svc.generate_invoice(tenant.id)
        assert invoice is not None
        assert invoice.amount_cents == 4900  # $49/month

    def test_tenant_lifecycle(self):
        from codeverify_core.hosted_saas import (
            HostedSaaSService,
            TenantStatus,
        )

        svc = HostedSaaSService()
        tenant = svc.create_tenant("Org", "a@b.com")
        svc.suspend_tenant(tenant.id, "non-payment")
        t = svc.get_tenant(tenant.id)
        assert t.status == TenantStatus.SUSPENDED

        svc.reactivate_tenant(tenant.id)
        t = svc.get_tenant(tenant.id)
        assert t.status == TenantStatus.ACTIVE

    def test_plan_change(self):
        from codeverify_core.hosted_saas import HostedSaaSService, SaaSPlan

        svc = HostedSaaSService()
        tenant = svc.create_tenant("Org", "a@b.com", SaaSPlan.FREE)
        svc.change_plan(tenant.id, SaaSPlan.PRO)
        t = svc.get_tenant(tenant.id)
        assert t.plan == SaaSPlan.PRO


# --- Feature 8: Proof Artifact Marketplace ---


class TestProofArtifactMarketplace:
    def test_artifact_submission(self):
        from codeverify_core.proof_artifact_marketplace import (
            ArtifactStatus,
            ProofArtifactMarketplaceService,
            ProofCategory,
            ProofLanguage,
        )

        svc = ProofArtifactMarketplaceService()
        artifact = svc.submit_artifact(
            title="Null Check Pattern",
            description="Z3 null safety proof template",
            category=ProofCategory.NULL_SAFETY,
            language=ProofLanguage.PYTHON,
            z3_constraints="x != None",
            author_id="user123",
        )
        assert artifact.status == ArtifactStatus.SUBMITTED
        assert artifact.is_anonymized is True

    def test_anonymization(self):
        from codeverify_core.proof_artifact_marketplace import (
            ProofAnonymizer,
            ProofArtifact,
        )

        anon = ProofAnonymizer()
        artifact = ProofArtifact(
            z3_constraints="x != null",
            pattern_code="email = user@company.com",
            author_id="john_doe",
        )
        result = anon.anonymize(artifact)
        assert "user@company.com" not in result.pattern_code
        assert result.is_anonymized is True
        assert result.author_id != "john_doe"

    def test_voting(self):
        from codeverify_core.proof_artifact_marketplace import (
            ProofArtifactMarketplaceService,
            ProofCategory,
            ProofLanguage,
        )

        svc = ProofArtifactMarketplaceService()
        artifact = svc.submit_artifact(
            "Test",
            "desc",
            ProofCategory.BOUNDS_CHECK,
            ProofLanguage.UNIVERSAL,
            z3_constraints="i >= 0 && i < len",
        )
        svc.publish_artifact(artifact.id)

        svc.vote(artifact.id, "user1", upvote=True)
        svc.vote(artifact.id, "user2", upvote=True)
        svc.vote(artifact.id, "user3", upvote=False)

        a = svc.download(artifact.id)
        assert a.upvotes == 2
        assert a.downvotes == 1

    def test_search(self):
        from codeverify_core.proof_artifact_marketplace import (
            ProofArtifactMarketplaceService,
            ProofCategory,
            ProofLanguage,
        )

        svc = ProofArtifactMarketplaceService()
        a1 = svc.submit_artifact(
            "Null Check",
            "null safety",
            ProofCategory.NULL_SAFETY,
            ProofLanguage.PYTHON,
            "x != None",
        )
        a2 = svc.submit_artifact(
            "Bounds", "array bounds", ProofCategory.BOUNDS_CHECK, ProofLanguage.PYTHON, "i < len"
        )
        svc.publish_artifact(a1.id)
        svc.publish_artifact(a2.id)

        results = svc.search(category=ProofCategory.NULL_SAFETY)
        assert len(results) == 1
        assert results[0].title == "Null Check"

    def test_proof_reuse(self):
        from codeverify_core.proof_artifact_marketplace import (
            ProofArtifactMarketplaceService,
            ProofCategory,
            ProofLanguage,
        )

        svc = ProofArtifactMarketplaceService()
        a = svc.submit_artifact(
            "Division Guard",
            "div zero",
            ProofCategory.DIVISION_ZERO,
            ProofLanguage.PYTHON,
            z3_constraints="b != 0",
            pattern_code="result = a / b if b != 0 else default",
        )
        svc.publish_artifact(a.id)

        matches = svc.find_reusable_proof(
            "result = x / y if y != 0 else 0",
            ProofCategory.DIVISION_ZERO,
            ProofLanguage.PYTHON,
        )
        assert len(matches) > 0


# --- Feature 9: Compliance-as-Code Engine ---


class TestComplianceEngine:
    def test_nl_query_parsing(self):
        from codeverify_core.compliance_engine import (
            ComplianceFramework,
            NLQueryParser,
            QueryIntent,
        )

        parser = NLQueryParser()
        intent, framework = parser.parse("Does this code encrypt PII data for HIPAA?")
        assert intent == QueryIntent.ENCRYPTION
        assert framework == ComplianceFramework.HIPAA

    def test_compliance_check_library(self):
        from codeverify_core.compliance_engine import (
            ComplianceCheckLibrary,
            ComplianceFramework,
        )

        lib = ComplianceCheckLibrary()
        soc2_checks = lib.get_checks(framework=ComplianceFramework.SOC2)
        assert len(soc2_checks) >= 3

        hipaa_checks = lib.get_checks(framework=ComplianceFramework.HIPAA)
        assert len(hipaa_checks) >= 2

    def test_codebase_scanning(self):
        from codeverify_core.compliance_engine import (
            CheckStatus,
            CodebaseScanner,
            ComplianceCheck,
            ComplianceFramework,
        )

        scanner = CodebaseScanner()
        check = ComplianceCheck(
            framework=ComplianceFramework.SOC2,
            control_id="CC7.2",
            title="Audit Logging",
            code_patterns=["structlog", "logger.info"],
            anti_patterns=["print("],
        )
        files = {
            "app.py": "import structlog\nlogger = structlog.get_logger()\nlogger.info('started')\n",
        }
        result = scanner.scan(check, files)
        assert result.status == CheckStatus.PASS
        assert len(result.evidence) > 0

    def test_nl_compliance_query(self):
        from codeverify_core.compliance_engine import ComplianceAsCodeService

        svc = ComplianceAsCodeService()
        files = {
            "auth.py": "from bcrypt import hashpw\ndef login(password):\n    hashed = hashpw(password)\n",
        }
        result = svc.query("Does this service use proper authentication?", files)
        assert result.intent is not None
        assert len(result.results) > 0
        assert "Compliance Query Result" in result.answer

    def test_framework_audit(self):
        from codeverify_core.compliance_engine import (
            ComplianceAsCodeService,
            ComplianceFramework,
        )

        svc = ComplianceAsCodeService()
        files = {
            "app.py": "import structlog\nlogger = structlog.get_logger()\n",
            "auth.py": "from bcrypt import hashpw\n",
        }
        report = svc.run_framework_audit(ComplianceFramework.SOC2, "myrepo", files)
        assert report.framework == ComplianceFramework.SOC2
        assert len(report.results) >= 3
        assert 0 <= report.pass_rate <= 1.0
        assert report.summary_markdown != ""

    def test_gap_detection(self):
        from codeverify_core.compliance_engine import (
            CheckStatus,
            CodebaseScanner,
            ComplianceCheck,
            ComplianceFramework,
        )

        scanner = CodebaseScanner()
        check = ComplianceCheck(
            framework=ComplianceFramework.PCI_DSS,
            control_id="3.4",
            title="Cardholder Data Encryption",
            code_patterns=["tokenize", "encrypt_card"],
            anti_patterns=["card_number ="],
        )
        files = {"payment.py": "card_number = request.form['card']\n"}
        result = scanner.scan(check, files)
        assert result.status == CheckStatus.FAIL
        assert len(result.gaps) > 0


# --- Feature 10: Performance & Cost Dashboard ---


class TestPerformanceCostDashboard:
    def test_token_cost_calculation(self):
        from codeverify_core.perf_cost_dashboard import CostCalculator

        calc = CostCalculator()
        cost = calc.calculate_token_cost("gpt-4", 1000, 500)
        assert cost > 0
        expected = 1.0 * 3.0 + 0.5 * 6.0  # 3 + 3 = 6.0 cents
        assert abs(cost - expected) < 0.01

    def test_usage_recording(self):
        from codeverify_core.perf_cost_dashboard import (
            ModelProvider,
            PerformanceCostDashboardService,
        )

        svc = PerformanceCostDashboardService()
        record = svc.record_token_usage(
            ModelProvider.OPENAI,
            "gpt-4",
            "semantic_analysis",
            input_tokens=500,
            output_tokens=200,
            latency_ms=1200,
        )
        assert record.total_tokens == 700
        assert record.cost_cents > 0

    def test_solver_metrics(self):
        from codeverify_core.perf_cost_dashboard import PerformanceCostDashboardService

        svc = PerformanceCostDashboardService()
        metric = svc.record_solver_metric(
            "null_safety", solve_time_ms=150, constraint_count=25, result="unsat"
        )
        assert metric.solve_time_ms == 150
        assert metric.result == "unsat"

    def test_dashboard_data(self):
        from codeverify_core.perf_cost_dashboard import (
            ModelProvider,
            PerformanceCostDashboardService,
        )

        svc = PerformanceCostDashboardService()
        svc.record_token_usage(ModelProvider.OPENAI, "gpt-4", "analysis", 1000, 500)
        svc.record_solver_metric("null_safety", 100)
        svc.record_review_cost("review-1", finding_count=3)

        data = svc.get_dashboard_data()
        assert data.total_cost_cents > 0
        assert data.review_count == 1
        assert data.roi is not None
        assert data.roi.total_findings == 3

    def test_budget_alerts(self):
        from codeverify_core.perf_cost_dashboard import (
            BudgetConfig,
            ModelProvider,
            PerformanceCostDashboardService,
        )

        budget = BudgetConfig(
            monthly_budget_cents=100,  # $1 budget
            warning_threshold=0.5,
            critical_threshold=0.9,
        )
        svc = PerformanceCostDashboardService(budget=budget)

        # Record enough usage to trigger warning
        for _ in range(5):
            svc.record_token_usage(ModelProvider.OPENAI, "gpt-4", "test", 5000, 2000)

        alerts = svc.get_alerts()
        assert len(alerts) > 0

    def test_optimization_recommendations(self):
        from codeverify_core.perf_cost_dashboard import (
            ModelProvider,
            PerformanceCostDashboardService,
        )

        svc = PerformanceCostDashboardService()
        # Generate enough data for optimization engine
        for _i in range(60):
            svc.record_token_usage(
                ModelProvider.OPENAI,
                "gpt-4",
                "semantic_analysis",
                input_tokens=2000,
                output_tokens=1000,
            )

        data = svc.get_dashboard_data()
        assert len(data.recommendations) > 0

    def test_roi_calculation(self):
        from codeverify_core.perf_cost_dashboard import (
            ModelProvider,
            PerformanceCostDashboardService,
        )

        svc = PerformanceCostDashboardService()
        svc.record_token_usage(ModelProvider.OPENAI, "gpt-4", "analysis", 1000, 500)
        svc.record_review_cost("r1", finding_count=5)
        svc.record_review_cost("r2", finding_count=3)

        data = svc.get_dashboard_data()
        assert data.roi.total_reviews == 2
        assert data.roi.total_findings == 8
        assert data.roi.manual_review_hours_saved == 1.0
