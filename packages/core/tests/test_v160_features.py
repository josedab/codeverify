"""Tests for v1.6.0 next-gen features."""

import json


class TestAgentMarketplace:
    def test_submit_and_publish(self):
        from codeverify_core.agent_marketplace import (
            AgentManifest,
            AgentMarketplaceService,
            PublishStatus,
        )

        svc = AgentMarketplaceService()
        manifest = AgentManifest(name="SQL Injection Scanner", author="alice", author_id="a1")
        agent = svc.submit_agent(manifest)
        assert agent.status == PublishStatus.SUBMITTED
        svc.review_agent(agent.id, "reviewer1", approved=True)
        svc.publish_agent(agent.id)
        assert svc.get_agent(agent.id).status == PublishStatus.PUBLISHED

    def test_install_and_usage(self):
        from codeverify_core.agent_marketplace import AgentManifest, AgentMarketplaceService

        svc = AgentMarketplaceService()
        m = AgentManifest(name="Test Agent")
        a = svc.submit_agent(m)
        svc.review_agent(a.id, "r1", True)
        svc.publish_agent(a.id)
        inst = svc.install_agent(a.id, "org1")
        assert inst is not None
        svc.record_usage(a.id, "org1")
        assert svc.get_agent(a.id).usage_count == 1

    def test_search_and_rating(self):
        from codeverify_core.agent_marketplace import (
            AgentCategory,
            AgentManifest,
            AgentMarketplaceService,
        )

        svc = AgentMarketplaceService()
        m = AgentManifest(name="Security Scanner", category=AgentCategory.SECURITY, tags=["owasp"])
        a = svc.submit_agent(m)
        svc.review_agent(a.id, "r", True)
        svc.publish_agent(a.id)
        svc.rate_agent(a.id, 4.5)
        results = svc.search("security")
        assert len(results) >= 1
        assert results[0].rating == 4.5

    def test_revenue_share(self):
        from codeverify_core.agent_marketplace import (
            AgentManifest,
            AgentMarketplaceService,
            PricingModel,
        )

        svc = AgentMarketplaceService()
        m = AgentManifest(
            name="Paid Agent", pricing=PricingModel.PAID, price_cents_per_use=10, author_id="a1"
        )
        a = svc.submit_agent(m)
        svc.review_agent(a.id, "r", True)
        svc.publish_agent(a.id)
        svc.install_agent(a.id, "org1")
        for _ in range(5):
            svc.record_usage(a.id, "org1")
        rev = svc.get_revenue_share(a.id)
        assert rev.total_revenue_cents == 50
        assert rev.author_share_cents == 35  # 70%


class TestVerificationTelemetry:
    def test_submit_and_benchmark(self):
        from codeverify_core.verification_telemetry import VerificationTelemetryService

        svc = VerificationTelemetryService()
        for i in range(6):
            svc.submit_telemetry(
                f"org{i}", {"false_positive_rate": 0.1 + i * 0.05, "detection_rate": 0.8 - i * 0.05}
            )
        results = svc.benchmark("org0")
        assert len(results) >= 1
        assert results[0].percentile > 0

    def test_quarterly_report(self):
        from codeverify_core.verification_telemetry import VerificationTelemetryService

        svc = VerificationTelemetryService()
        for i in range(6):
            svc.submit_telemetry(f"org{i}", {"detection_rate": 0.8, "total_verifications": 100})
        report = svc.generate_report("Q1-2026")
        assert report.participating_orgs >= 6


class TestVerificationProtocol:
    def test_capabilities(self):
        from codeverify_core.verification_protocol import VerificationProtocolServer

        server = VerificationProtocolServer()
        caps = server.get_capabilities()
        assert "python" in caps.supported_languages
        assert caps.supports_proofs is True

    def test_verify_clean_code(self):
        from codeverify_core.verification_protocol import (
            VerificationProtocolServer,
            VerifyRequest,
            VerifyStatus,
        )

        server = VerificationProtocolServer()
        req = VerifyRequest(files=[{"path": "app.py", "content": "def add(a, b): return a + b"}])
        resp = server.verify(req)
        assert resp.status == VerifyStatus.VERIFIED

    def test_verify_buggy_code(self):
        from codeverify_core.verification_protocol import (
            VerificationProtocolServer,
            VerifyRequest,
            VerifyStatus,
        )

        server = VerificationProtocolServer()
        req = VerifyRequest(
            files=[{"path": "app.py", "content": "x = eval(input())"}], include_proofs=True
        )
        resp = server.verify(req)
        assert resp.status == VerifyStatus.FAILED
        assert len(resp.findings) >= 1
        assert len(resp.proofs) >= 1
        assert resp.proofs[0].signature != ""

    def test_proof_signing(self):
        from codeverify_core.verification_protocol import ProofCert, VerifyStatus

        cert = ProofCert(status=VerifyStatus.VERIFIED, content_hash="abc123")
        sig = cert.sign("secret-key")
        assert len(sig) == 16


class TestDefectHeatmap:
    def test_prediction(self):
        from codeverify_core.defect_heatmap import DefectHeatmapService, FileHistory, HeatmapLevel

        svc = DefectHeatmapService()
        histories = [
            FileHistory(
                file_path="auth.py",
                total_findings=20,
                critical_findings=5,
                churn_rate=0.8,
                complexity=25.0,
            ),
            FileHistory(
                file_path="utils.py",
                total_findings=1,
                critical_findings=0,
                churn_rate=0.05,
                complexity=3.0,
                last_finding_days_ago=90,
            ),
        ]
        heatmap = svc.generate_heatmap("repo", histories)
        assert heatmap.total_files == 2
        assert heatmap.predictions[0].file_path == "auth.py"
        assert heatmap.predictions[0].heatmap_level in (HeatmapLevel.CRITICAL, HeatmapLevel.HIGH)

    def test_ticket_suggestions(self):
        from codeverify_core.defect_heatmap import DefectHeatmapService, FileHistory

        svc = DefectHeatmapService()
        heatmap = svc.generate_heatmap(
            "r",
            [
                FileHistory(
                    file_path="danger.py",
                    total_findings=30,
                    critical_findings=10,
                    churn_rate=0.9,
                    complexity=30.0,
                )
            ],
        )
        tickets = svc.suggest_tickets(heatmap)
        assert len(tickets) >= 1
        assert "danger.py" in tickets[0].file_path


class TestSelfHealing:
    def test_incident_diagnosis_and_fix(self):
        from codeverify_core.self_healing import HealingStatus, RuntimeIncident, SelfHealingService

        svc = SelfHealingService()
        incident = RuntimeIncident(
            function_name="get_user",
            file_path="api.py",
            error_type="TypeError",
            error_message="NoneType",
            variable_state={"user": None},
        )
        action = svc.report_incident(incident)
        assert action.status in (
            HealingStatus.PR_CREATED,
            HealingStatus.GENERATING_FIX,
            HealingStatus.VERIFYING_FIX,
        )
        assert action.diagnosis is not None
        assert action.fixed_code != ""

    def test_auto_fix_mode(self):
        from codeverify_core.self_healing import (
            AutonomyLevel,
            HealingConfig,
            HealingStatus,
            RuntimeIncident,
            SelfHealingService,
        )

        config = HealingConfig(
            autonomy=AutonomyLevel.AUTO_FIX, min_confidence=0.5, auto_merge_confidence=0.5
        )
        svc = SelfHealingService(config=config)
        incident = RuntimeIncident(
            function_name="div", error_type="ZeroDivisionError", variable_state={"b": 0}
        )
        action = svc.report_incident(incident)
        assert action.status in (HealingStatus.APPLIED, HealingStatus.PR_CREATED)

    def test_stats(self):
        from codeverify_core.self_healing import RuntimeIncident, SelfHealingService

        svc = SelfHealingService()
        svc.report_incident(RuntimeIncident(error_type="TypeError", variable_state={"x": None}))
        stats = svc.get_stats()
        assert stats["total_incidents"] == 1


class TestVerificationCICD:
    def test_pipeline_creation(self):
        from codeverify_core.verification_cicd import VerificationCICDService

        svc = VerificationCICDService()
        yml = "name: My Pipeline\npolicy: strict\nmin_coverage: 0.9"
        pipeline = svc.create_pipeline(yml, repo="org/repo", commit_sha="abc")
        assert pipeline.name == "My Pipeline"
        assert len(pipeline.stages) >= 3

    def test_pipeline_execution_clean(self):
        from codeverify_core.verification_cicd import PipelineStatus, VerificationCICDService

        svc = VerificationCICDService()
        pipeline = svc.create_pipeline("name: Test")
        result = svc.run_pipeline(pipeline, {"app.py": "def safe(): return 1"})
        assert result.status == PipelineStatus.PASSED

    def test_pipeline_execution_fails(self):
        from codeverify_core.verification_cicd import PipelineStatus, VerificationCICDService

        svc = VerificationCICDService()
        pipeline = svc.create_pipeline("name: Test")
        result = svc.run_pipeline(pipeline, {"app.py": "x = eval('bad')"})
        assert result.status == PipelineStatus.FAILED


class TestCodeEvolution:
    def test_timeline_building(self):
        from codeverify_core.evolution_timeline import CodeEvolutionService

        svc = CodeEvolutionService()
        events = [
            {"type": "created", "commit": "aaa", "author": "alice", "message": "initial"},
            {"type": "modified", "commit": "bbb", "author": "bob"},
            {"type": "proof_passed", "commit": "ccc", "author": "alice"},
        ]
        timeline = svc.record_events("get_user", "api.py", events)
        assert timeline.total_proof_passes == 1
        assert timeline.stability_score > 0

    def test_mermaid_rendering(self):
        from codeverify_core.evolution_timeline import CodeEvolutionService

        svc = CodeEvolutionService()
        svc.record_events("fn", "f.py", [{"type": "created"}, {"type": "proof_passed"}])
        mermaid = svc.render_mermaid("fn", "f.py")
        assert "timeline" in mermaid

    def test_repo_evolution(self):
        from codeverify_core.evolution_timeline import CodeEvolutionService

        svc = CodeEvolutionService()
        svc.record_events(
            "f1", "a.py", [{"type": "modified"}, {"type": "modified"}, {"type": "modified"}]
        )
        svc.record_events("f2", "b.py", [{"type": "proof_passed"}])
        evo = svc.get_repo_evolution("repo")
        assert evo.most_modified == "f1"


class TestCreditSystem:
    def test_award_and_balance(self):
        from codeverify_core.credit_system import CreditSource, VerificationCreditService

        svc = VerificationCreditService()
        svc.register_org("org1", "Test Org")
        svc.award_credits("org1", CreditSource.VERIFICATION_COVERAGE, 100)
        assert svc.get_balance("org1") == 100

    def test_rule_evaluation(self):
        from codeverify_core.credit_system import VerificationCreditService

        svc = VerificationCreditService()
        svc.register_org("org1")
        awarded = svc.evaluate_rules("org1", {"verification_coverage": 0.95, "fix_rate": 0.92})
        assert len(awarded) >= 2  # 80%+ and 95%+ coverage + fix rate

    def test_redemption(self):
        from codeverify_core.credit_system import (
            CreditSource,
            RedemptionType,
            VerificationCreditService,
        )

        svc = VerificationCreditService()
        svc.register_org("org1")
        svc.award_credits("org1", CreditSource.VERIFICATION_COVERAGE, 600)
        tx = svc.redeem("org1", RedemptionType.FEATURE_UNLOCK)
        assert tx is not None
        assert svc.get_balance("org1") == 100  # 600 - 500

    def test_leaderboard(self):
        from codeverify_core.credit_system import CreditSource, VerificationCreditService

        svc = VerificationCreditService()
        for name in ["org_a", "org_b", "org_c"]:
            svc.register_org(name, name)
        svc.award_credits("org_a", CreditSource.FIX_RATE, 300)
        svc.award_credits("org_b", CreditSource.FIX_RATE, 100)
        lb = svc.get_leaderboard()
        assert lb[0].org_id == "org_a"
        assert lb[0].rank == 1


class TestMultiModal:
    def test_terraform_verification(self):
        from codeverify_core.multimodal_verify import MultiModalVerificationService

        svc = MultiModalVerificationService()
        result = svc.verify_file(
            "main.tf",
            'resource "aws_security_group" {\n  ingress {\n    cidr_blocks = ["0.0.0.0/0"]\n  }\n}',
        )
        assert not result.passed
        assert any("0.0.0.0/0" in f.message for f in result.findings)

    def test_migration_verification(self):
        from codeverify_core.multimodal_verify import MultiModalVerificationService

        svc = MultiModalVerificationService()
        result = svc.verify_file("migration_001.sql", "ALTER TABLE users DROP COLUMN email;")
        assert not result.passed
        assert any("DROP COLUMN" in f.message for f in result.findings)

    def test_api_contract_verification(self):
        from codeverify_core.multimodal_verify import MultiModalVerificationService

        svc = MultiModalVerificationService()
        spec = json.dumps({"openapi": "3.0.0", "paths": {"/users": {"get": {}}}})
        result = svc.verify_file("openapi.json", spec)
        assert any("responses" in f.message for f in result.findings)

    def test_config_secrets(self):
        from codeverify_core.multimodal_verify import MultiModalVerificationService

        svc = MultiModalVerificationService()
        result = svc.verify_file("app.env", "password=hunter2\napi_key=sk-abc123")
        assert not result.passed
        assert len(result.findings) >= 2


class TestVerificationSearch:
    def test_index_and_search(self):
        from codeverify_core.verification_search import (
            CodeEntity,
            VerificationSearchService,
            VerificationStatus,
        )

        svc = VerificationSearchService()
        svc.index_entity(
            CodeEntity(
                file_path="auth/login.py",
                function_name="login",
                verification_status=VerificationStatus.UNVERIFIED,
                trust_score=0.3,
                categories=["null_safety"],
            )
        )
        svc.index_entity(
            CodeEntity(
                file_path="utils/helpers.py",
                function_name="add",
                verification_status=VerificationStatus.VERIFIED,
                trust_score=0.95,
            )
        )
        results = svc.search("unverified functions in auth")
        assert results.total_count >= 1
        assert results.hits[0].entity.function_name == "login"

    def test_structured_search(self):
        from codeverify_core.verification_search import (
            CodeEntity,
            VerificationSearchService,
            VerificationStatus,
        )

        svc = VerificationSearchService()
        svc.index_entity(
            CodeEntity(
                file_path="a.py",
                verification_status=VerificationStatus.FAILING,
                critical_findings=3,
            )
        )
        svc.index_entity(
            CodeEntity(file_path="b.py", verification_status=VerificationStatus.VERIFIED)
        )
        results = svc.search_structured({"verification_status": "failing", "severity": "critical"})
        assert results.total_count == 1

    def test_nl_query_parsing(self):
        from codeverify_core.verification_search import QueryParser

        parser = QueryParser()
        q = parser.parse("unverified functions in auth/")
        assert q.filters.get("verification_status") == "unverified"
        assert "auth" in str(q.filters.get("file_path_contains", ""))
