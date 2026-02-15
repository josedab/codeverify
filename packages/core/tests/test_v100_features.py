"""Tests for v1.0.0 features.

Covers all 10 next-gen features:
1. AI Code Insurance Underwriting Platform
2. Cross-Repository Security Graph
3. Differential Privacy-Preserving Proof Marketplace
4. Blockchain-Verified Code Provenance
5. Natural Language Compliance Query Engine
6. AI Model Bias & Fairness Verification
7. IDE Copilot Undo with Proof Preservation
8. Predictive Code Quality Forecasting
9. Verification-Driven Code Generation
10. Real-Time Collaborative Verification Sessions
"""

import pytest


# ─── Feature 1: AI Code Insurance Underwriting Platform ─────────────────


class TestInsuranceUnderwriting:
    def test_risk_profile_scoring(self):
        from codeverify_core.insurance_underwriting import RiskProfile, RiskTier

        profile = RiskProfile(
            repo_id="repo-1",
            trust_score=80.0,
            verification_coverage=0.7,
            lines_of_code=50000,
            ai_generated_ratio=0.3,
            critical_findings=0,
        )
        assert 0.0 <= profile.composite_risk_score <= 1.0
        assert profile.risk_tier in (RiskTier.MINIMAL, RiskTier.LOW, RiskTier.MODERATE)

    def test_risk_profile_high_risk(self):
        from codeverify_core.insurance_underwriting import RiskProfile, RiskTier

        profile = RiskProfile(
            repo_id="repo-risky",
            trust_score=20.0,
            verification_coverage=0.1,
            ai_generated_ratio=0.8,
            critical_findings=5,
            historical_incidents=3,
        )
        assert profile.composite_risk_score > 0.5
        assert profile.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL)

    def test_premium_calculation(self):
        from codeverify_core.insurance_underwriting import (
            PremiumCalculator, RiskProfile,
        )

        calc = PremiumCalculator()
        profile = RiskProfile(
            repo_id="repo-1",
            trust_score=75.0,
            verification_coverage=0.85,
            lines_of_code=10000,
        )
        result = calc.calculate(profile)
        assert result.final_monthly_premium > 0
        assert result.coverage_discount > 0  # High coverage discount

    def test_policy_lifecycle(self):
        from codeverify_core.insurance_underwriting import (
            InsuranceUnderwriter, RiskProfile, PolicyStatus,
        )

        uw = InsuranceUnderwriter()
        uw.assess_risk(RiskProfile(repo_id="repo-a", trust_score=70.0, lines_of_code=5000))

        policy = uw.create_policy(
            holder_org="TestOrg",
            repo_ids=["repo-a"],
            coverage_limit=50000.0,
            deductible=2500.0,
        )
        assert policy.status == PolicyStatus.ACTIVE
        assert policy.monthly_premium > 0
        assert policy.is_active

    def test_claim_submission_and_validation(self):
        from codeverify_core.insurance_underwriting import (
            InsuranceUnderwriter, RiskProfile, CoverageType, ClaimStatus,
        )

        uw = InsuranceUnderwriter()
        uw.assess_risk(RiskProfile(repo_id="repo-b", trust_score=60.0, lines_of_code=3000))

        policy = uw.create_policy(holder_org="ClaimOrg", repo_ids=["repo-b"])
        claim = uw.submit_claim(
            policy_id=policy.id,
            repo_id="repo-b",
            description="Logic error in payment processing",
            claimed_amount=10000.0,
            coverage_type=CoverageType.LOGIC_ERROR,
            evidence_proof_ids=["proof-abc-123-long-hash-value-here"],
        )
        assert claim.status == ClaimStatus.SUBMITTED

        processed = uw.process_claim(
            claim.id,
            proof_hashes=["abcdef1234567890abcdef1234567890ab"],
        )
        assert processed.status == ClaimStatus.VALIDATED
        assert processed.paid_amount > 0

    def test_claim_rejection_outside_coverage(self):
        from codeverify_core.insurance_underwriting import (
            InsuranceUnderwriter, RiskProfile, ClaimStatus, ClaimRejectionReason,
        )

        uw = InsuranceUnderwriter()
        uw.assess_risk(RiskProfile(repo_id="repo-c", lines_of_code=1000))
        policy = uw.create_policy(holder_org="Org", repo_ids=["repo-c"])

        claim = uw.submit_claim(
            policy_id=policy.id,
            repo_id="repo-NOT-covered",
            description="Bug",
            claimed_amount=5000.0,
            evidence_proof_ids=["proof-1"],
        )
        processed = uw.process_claim(claim.id)
        assert processed.status == ClaimStatus.REJECTED
        assert processed.rejection_reason == ClaimRejectionReason.OUTSIDE_COVERAGE

    def test_portfolio_summary(self):
        from codeverify_core.insurance_underwriting import (
            InsuranceUnderwriter, RiskProfile,
        )

        uw = InsuranceUnderwriter()
        uw.assess_risk(RiskProfile(repo_id="r1", lines_of_code=5000))
        uw.create_policy(holder_org="Org1", repo_ids=["r1"])
        summary = uw.get_portfolio_summary()
        assert summary["total_policies"] == 1
        assert summary["active_policies"] == 1

    def test_singleton(self):
        from codeverify_core.insurance_underwriting import (
            get_insurance_underwriter, reset_insurance_underwriter,
        )

        reset_insurance_underwriter()
        uw1 = get_insurance_underwriter()
        uw2 = get_insurance_underwriter()
        assert uw1 is uw2
        reset_insurance_underwriter()


# ─── Feature 2: Cross-Repository Security Graph ─────────────────


class TestSecurityGraph:
    def test_add_nodes_and_edges(self):
        from codeverify_core.cross_repo_security_graph import (
            SecurityKnowledgeGraph,
        )

        graph = SecurityKnowledgeGraph()
        graph.add_repository("repo-1", "MyApp")
        graph.add_dependency("repo-1", "lodash", "4.17.21")
        assert graph.node_count == 2
        assert graph.edge_count == 1

    def test_vulnerability_registration(self):
        from codeverify_core.cross_repo_security_graph import (
            SecurityKnowledgeGraph, VulnerabilityRecord, VulnSeverity,
        )

        graph = SecurityKnowledgeGraph()
        graph.add_repository("repo-1", "MyApp")
        graph.add_dependency("repo-1", "lodash", "4.17.21")

        vuln = VulnerabilityRecord(
            cve_id="CVE-2024-1234",
            title="Prototype Pollution in lodash",
            severity=VulnSeverity.HIGH,
            cvss_score=7.5,
            affected_packages=["lodash"],
        )
        graph.register_vulnerability(vuln)
        assert graph.node_count == 3  # repo + pkg + vuln

    def test_blast_radius(self):
        from codeverify_core.cross_repo_security_graph import (
            SecurityKnowledgeGraph, VulnerabilityRecord, VulnSeverity,
        )

        graph = SecurityKnowledgeGraph()
        graph.add_repository("repo-1", "App1")
        graph.add_repository("repo-2", "App2")
        graph.add_dependency("repo-1", "shared-lib", "1.0.0")
        graph.add_dependency("repo-2", "shared-lib", "1.0.0")

        vuln = VulnerabilityRecord(
            id="vuln-1",
            title="Vuln in shared-lib",
            severity=VulnSeverity.CRITICAL,
            affected_packages=["shared-lib"],
        )
        graph.register_vulnerability(vuln)

        radius = graph.compute_blast_radius("vuln-1")
        assert len(radius.affected_repos) >= 2
        assert radius.risk_score > 0

    def test_org_risk_summary(self):
        from codeverify_core.cross_repo_security_graph import (
            SecurityKnowledgeGraph, VulnerabilityRecord, VulnSeverity,
        )

        graph = SecurityKnowledgeGraph()
        graph.add_repository("repo-1", "App1")
        vuln = VulnerabilityRecord(
            title="Test vuln", severity=VulnSeverity.CRITICAL,
        )
        graph.register_vulnerability(vuln)

        summary = graph.get_org_risk_summary()
        assert summary["total_repos"] == 1
        assert summary["critical_vulnerabilities"] == 1

    def test_singleton(self):
        from codeverify_core.cross_repo_security_graph import (
            get_security_graph, reset_security_graph,
        )

        reset_security_graph()
        g1 = get_security_graph()
        g2 = get_security_graph()
        assert g1 is g2
        reset_security_graph()


# ─── Feature 3: Privacy-Preserving Proof Marketplace ─────────────────


class TestPrivacyPreservingProofs:
    def test_proof_anonymization(self):
        from codeverify_core.privacy_preserving_proofs import (
            ProofAnonymizer, ProofCategory,
        )

        anon = ProofAnonymizer()
        proof = anon.anonymize(
            "var_x > 0 AND func_validate(var_input)",
            ProofCategory.NULL_SAFETY,
        )
        assert proof.original_hash != ""
        assert proof.contributor_pseudonym != ""
        assert "var_x" not in proof.pattern_description

    def test_marketplace_contribute_and_search(self):
        from codeverify_core.privacy_preserving_proofs import (
            PrivacyPreservingProofMarketplace, ProofCategory,
        )

        mp = PrivacyPreservingProofMarketplace(epsilon_budget=10.0)
        proof = mp.contribute_proof(
            "x >= 0 AND x < array_length",
            ProofCategory.BOUNDS_CHECK,
            language="python",
        )
        assert proof is not None
        assert mp.proof_count == 1

        results = mp.search_proofs(category=ProofCategory.BOUNDS_CHECK)
        assert len(results) == 1

    def test_privacy_budget_exhaustion(self):
        from codeverify_core.privacy_preserving_proofs import (
            PrivacyPreservingProofMarketplace, ProofCategory,
        )

        mp = PrivacyPreservingProofMarketplace(epsilon_budget=2.0)
        mp.contribute_proof("proof1", ProofCategory.SECURITY, epsilon=1.5)
        result = mp.contribute_proof("proof2", ProofCategory.SECURITY, epsilon=1.0)
        assert result is None  # Budget exhausted

    def test_federated_aggregation(self):
        from codeverify_core.privacy_preserving_proofs import (
            PrivacyPreservingProofMarketplace,
        )

        mp = PrivacyPreservingProofMarketplace()
        mp.submit_federated_update({"null_check": 0.8, "bounds": 0.6})
        mp.submit_federated_update({"null_check": 0.9, "bounds": 0.7})
        update = mp.run_federated_round(epsilon=1.0)
        assert update.participant_count == 2
        assert len(update.pattern_weights) >= 2

    def test_upvote_and_download(self):
        from codeverify_core.privacy_preserving_proofs import (
            PrivacyPreservingProofMarketplace, ProofCategory,
        )

        mp = PrivacyPreservingProofMarketplace()
        proof = mp.contribute_proof("test proof", ProofCategory.CUSTOM)
        assert proof is not None
        mp.upvote_proof(proof.id)
        downloaded = mp.download_proof(proof.id)
        assert downloaded is not None
        assert downloaded.upvotes == 1
        assert downloaded.downloads == 1

    def test_singleton(self):
        from codeverify_core.privacy_preserving_proofs import (
            get_privacy_marketplace, reset_privacy_marketplace,
        )

        reset_privacy_marketplace()
        m1 = get_privacy_marketplace()
        m2 = get_privacy_marketplace()
        assert m1 is m2
        reset_privacy_marketplace()


# ─── Feature 4: Blockchain-Verified Code Provenance ─────────────────


class TestBlockchainProvenance:
    def test_create_attestation(self):
        from codeverify_core.blockchain_provenance import (
            BlockchainProvenanceEngine, AttestationStatus,
        )

        engine = BlockchainProvenanceEngine()
        att = engine.create_attestation(
            repo_id="repo-1",
            commit_sha="abc123def456",
            proof_data='{"verified": true, "constraints": 5}',
        )
        assert att.status == AttestationStatus.CONFIRMED
        assert att.tx_hash != ""
        assert att.block_number > 0
        assert att.gas_used > 0

    def test_verify_attestation(self):
        from codeverify_core.blockchain_provenance import BlockchainProvenanceEngine

        engine = BlockchainProvenanceEngine()
        att = engine.create_attestation("repo-1", "abc123", "proof data")
        assert engine.verify_attestation(att.id) is True
        assert engine.verify_attestation("nonexistent") is False

    def test_issue_badge(self):
        from codeverify_core.blockchain_provenance import (
            BlockchainProvenanceEngine, BadgeLevel,
        )

        engine = BlockchainProvenanceEngine()
        badge = engine.issue_badge(
            repo_id="repo-1",
            release_tag="v1.0.0",
            trust_score=92.0,
            verification_coverage=0.85,
        )
        assert badge.level == BadgeLevel.PLATINUM
        assert badge.token_id == 1

    def test_badge_levels(self):
        from codeverify_core.blockchain_provenance import (
            VerificationBadge, BadgeLevel,
        )

        assert VerificationBadge.level_from_score(96) == BadgeLevel.DIAMOND
        assert VerificationBadge.level_from_score(87) == BadgeLevel.PLATINUM
        assert VerificationBadge.level_from_score(72) == BadgeLevel.GOLD
        assert VerificationBadge.level_from_score(55) == BadgeLevel.SILVER
        assert VerificationBadge.level_from_score(30) == BadgeLevel.BRONZE

    def test_content_address(self):
        from codeverify_core.blockchain_provenance import ContentAddress

        ca = ContentAddress.from_content("hello world")
        assert ca.cid.startswith("Qm")
        assert ca.content_hash != ""
        assert ca.size_bytes == 11

    def test_revoke_attestation(self):
        from codeverify_core.blockchain_provenance import (
            BlockchainProvenanceEngine, AttestationStatus,
        )

        engine = BlockchainProvenanceEngine()
        att = engine.create_attestation("repo-1", "abc123", "proof")
        assert engine.revoke_attestation(att.id) is True
        assert engine._attestations[att.id].status == AttestationStatus.REVOKED

    def test_provenance_summary(self):
        from codeverify_core.blockchain_provenance import BlockchainProvenanceEngine

        engine = BlockchainProvenanceEngine()
        engine.create_attestation("r1", "sha1", "proof1")
        engine.create_attestation("r2", "sha2", "proof2")
        engine.issue_badge("r1", "v1.0", trust_score=85.0)

        summary = engine.get_provenance_summary()
        assert summary["confirmed_attestations"] == 2
        assert summary["total_badges"] == 1
        assert summary["block_height"] >= 3

    def test_singleton(self):
        from codeverify_core.blockchain_provenance import (
            get_blockchain_provenance, reset_blockchain_provenance,
        )

        reset_blockchain_provenance()
        e1 = get_blockchain_provenance()
        e2 = get_blockchain_provenance()
        assert e1 is e2
        reset_blockchain_provenance()


# ─── Feature 5: Natural Language Compliance Query Engine ─────────────────


class TestNLComplianceEngine:
    def test_query_parsing(self):
        from codeverify_core.nl_compliance_engine import NLQueryParser, QueryType

        parser = NLQueryParser()
        query = parser.parse("Show me all database queries that access PII without encryption")
        assert query.query_type == QueryType.DATA_PROTECTION
        assert query.confidence > 0.4

    def test_standard_detection(self):
        from codeverify_core.nl_compliance_engine import (
            NLQueryParser, ComplianceStandard,
        )

        parser = NLQueryParser()
        query = parser.parse("Check HIPAA compliance for health data access")
        assert query.detected_standard == ComplianceStandard.HIPAA

    def test_execute_query_with_codebase(self):
        from codeverify_core.nl_compliance_engine import ComplianceQueryExecutor

        executor = ComplianceQueryExecutor()
        files = {
            "auth.py": "def check_auth(user):\n    if not user.has_permission:\n        raise PermissionError\n",
            "data.py": "def get_data():\n    # TODO: add access control\n    return db.query()\n",
        }
        result = executor.execute_query(
            "Verify authorization checks before data access", files,
        )
        assert result.confidence > 0
        assert len(result.evidence) > 0

    def test_execute_template(self):
        from codeverify_core.nl_compliance_engine import ComplianceQueryExecutor

        executor = ComplianceQueryExecutor()
        result = executor.execute_template("auth-check")
        assert result.query_id != ""

    def test_list_templates(self):
        from codeverify_core.nl_compliance_engine import (
            ComplianceQueryExecutor, ComplianceStandard,
        )

        executor = ComplianceQueryExecutor()
        templates = executor.list_templates()
        assert len(templates) >= 5

        soc2 = executor.list_templates(standard=ComplianceStandard.SOC2)
        assert all(t.standard == ComplianceStandard.SOC2 for t in soc2)

    def test_violations_detected(self):
        from codeverify_core.nl_compliance_engine import ComplianceQueryExecutor

        executor = ComplianceQueryExecutor()
        files = {
            "handler.py": "def handle_input(data):\n    # FIXME: need input validation\n    return process(data)\n",
        }
        result = executor.execute_query("Find input validation issues", files)
        assert len(result.violations) > 0

    def test_singleton(self):
        from codeverify_core.nl_compliance_engine import (
            get_compliance_query_engine, reset_compliance_query_engine,
        )

        reset_compliance_query_engine()
        e1 = get_compliance_query_engine()
        e2 = get_compliance_query_engine()
        assert e1 is e2
        reset_compliance_query_engine()


# ─── Feature 6: AI Model Bias & Fairness Verification ─────────────────


class TestFairnessVerification:
    def _make_groups(self):
        from codeverify_core.fairness_verification import GroupMetrics

        return [
            GroupMetrics(
                group_name="group_a", total_count=100,
                positive_count=60, true_positive=50,
                false_positive=10, true_negative=30, false_negative=10,
            ),
            GroupMetrics(
                group_name="group_b", total_count=100,
                positive_count=40, true_positive=30,
                false_positive=10, true_negative=50, false_negative=10,
            ),
        ]

    def test_demographic_parity(self):
        from codeverify_core.fairness_verification import BiasDetector

        detector = BiasDetector()
        groups = self._make_groups()
        result = detector.check_demographic_parity(groups, threshold=0.8)
        assert result.disparity_ratio > 0
        # 40/60 = 0.667, below 0.8 threshold
        assert result.passed is False

    def test_equal_opportunity(self):
        from codeverify_core.fairness_verification import BiasDetector

        detector = BiasDetector()
        groups = self._make_groups()
        result = detector.check_equal_opportunity(groups)
        assert result.disparity_ratio > 0
        assert result.metric.value == "equal_opportunity"

    def test_equalized_odds(self):
        from codeverify_core.fairness_verification import BiasDetector

        detector = BiasDetector()
        groups = self._make_groups()
        result = detector.check_equalized_odds(groups)
        assert result.metric.value == "equalized_odds"

    def test_remediation_suggestions(self):
        from codeverify_core.fairness_verification import (
            BiasDetector, RemediationAdvisor,
        )

        detector = BiasDetector()
        groups = self._make_groups()
        result = detector.check_demographic_parity(groups, threshold=0.8)
        advisor = RemediationAdvisor()
        suggestions = advisor.suggest(result)
        assert len(suggestions) > 0

    def test_full_fairness_report(self):
        from codeverify_core.fairness_verification import (
            FairnessVerifier, ComplianceStatus,
        )

        verifier = FairnessVerifier()
        groups = self._make_groups()
        report = verifier.verify_fairness("test-model", groups)
        assert len(report.results) >= 2
        assert report.overall_compliance in (
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.PARTIALLY_COMPLIANT,
            ComplianceStatus.NON_COMPLIANT,
        )
        assert report.to_dict()["model_name"] == "test-model"

    def test_singleton(self):
        from codeverify_core.fairness_verification import (
            get_fairness_verifier, reset_fairness_verifier,
        )

        reset_fairness_verifier()
        v1 = get_fairness_verifier()
        v2 = get_fairness_verifier()
        assert v1 is v2
        reset_fairness_verifier()


# ─── Feature 7: IDE Copilot Undo with Proof Preservation ─────────────────


class TestCopilotUndo:
    def test_create_save_point(self):
        from codeverify_core.copilot_undo import (
            CopilotUndoManager, CodeDiff, ProofSnapshot, VerificationState,
        )

        mgr = CopilotUndoManager()
        diff = CodeDiff(file_path="main.py", before="x = 1", after="x = compute()")
        proof = ProofSnapshot(trust_score=85.0, verification_state=VerificationState.VERIFIED)
        sp = mgr.create_save_point(diff, proof)
        assert sp.id != ""
        assert mgr.save_point_count == 1
        assert sp.proof_snapshot.trust_score == 85.0

    def test_rollback(self):
        from codeverify_core.copilot_undo import (
            CopilotUndoManager, CodeDiff, SavePointStatus,
        )

        mgr = CopilotUndoManager()
        sp1 = mgr.create_save_point(CodeDiff(file_path="a.py", before="a", after="b"))
        sp2 = mgr.create_save_point(CodeDiff(file_path="a.py", before="b", after="c"))
        sp3 = mgr.create_save_point(CodeDiff(file_path="a.py", before="c", after="d"))

        result = mgr.rollback(sp1.id)
        assert result is not None
        assert result.id == sp1.id
        assert mgr.rollback_count == 1

        # sp2 and sp3 should be rolled back
        assert mgr._save_points[sp2.id].status == SavePointStatus.ROLLED_BACK
        assert mgr._save_points[sp3.id].status == SavePointStatus.ROLLED_BACK

    def test_trust_trend(self):
        from codeverify_core.copilot_undo import (
            CopilotUndoManager, CodeDiff, ProofSnapshot,
        )

        mgr = CopilotUndoManager()
        for score in [60, 65, 70, 75, 80]:
            mgr.create_save_point(
                CodeDiff(file_path="x.py", before="", after=""),
                ProofSnapshot(trust_score=float(score)),
            )
        assert mgr.trust_trend.direction == "improving"
        assert mgr.trust_trend.current == 80.0

    def test_branch_comparison(self):
        from codeverify_core.copilot_undo import (
            CopilotUndoManager, CodeDiff, ProofSnapshot,
        )

        mgr = CopilotUndoManager()
        sp1 = mgr.create_save_point(CodeDiff(file_path="x.py", before="", after=""))
        branch_a = mgr.create_branch(
            sp1.id,
            CodeDiff(file_path="x.py", before="", after="option A"),
            "Option A",
            ProofSnapshot(trust_score=80.0),
        )
        branch_b = mgr.create_branch(
            sp1.id,
            CodeDiff(file_path="x.py", before="", after="option B"),
            "Option B",
            ProofSnapshot(trust_score=90.0),
        )
        assert branch_a is not None
        assert branch_b is not None
        comparison = mgr.compare_branches(branch_a.id, branch_b.id)
        assert comparison["recommendation"] == "Option B"

    def test_history(self):
        from codeverify_core.copilot_undo import CopilotUndoManager, CodeDiff

        mgr = CopilotUndoManager()
        for i in range(5):
            mgr.create_save_point(CodeDiff(file_path=f"f{i}.py", before="", after=""))
        history = mgr.get_history(limit=3)
        assert len(history) == 3

    def test_singleton(self):
        from codeverify_core.copilot_undo import (
            get_copilot_undo_manager, reset_copilot_undo_manager,
        )

        reset_copilot_undo_manager()
        m1 = get_copilot_undo_manager()
        m2 = get_copilot_undo_manager()
        assert m1 is m2
        reset_copilot_undo_manager()


# ─── Feature 8: Predictive Code Quality Forecasting ─────────────────


class TestQualityForecasting:
    def test_trend_analysis_stable(self):
        from codeverify_core.quality_forecasting import (
            TrendCalculator, MetricType, TrendDirection,
        )

        calc = TrendCalculator()
        values = [10.0, 10.1, 9.9, 10.0, 10.1, 10.0]
        trend = calc.analyze(values, MetricType.FINDINGS)
        assert trend.direction == TrendDirection.STABLE
        assert trend.data_points == 6

    def test_trend_analysis_declining(self):
        from codeverify_core.quality_forecasting import (
            TrendCalculator, MetricType, TrendDirection,
        )

        calc = TrendCalculator()
        values = [5.0, 8.0, 12.0, 15.0, 20.0, 25.0]
        trend = calc.analyze(values, MetricType.FINDINGS)
        # Increasing findings = declining quality
        assert trend.direction == TrendDirection.DECLINING
        assert trend.slope > 0

    def test_trend_analysis_improving(self):
        from codeverify_core.quality_forecasting import (
            TrendCalculator, MetricType, TrendDirection,
        )

        calc = TrendCalculator()
        values = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
        trend = calc.analyze(values, MetricType.TRUST_SCORE)
        assert trend.direction == TrendDirection.IMPROVING
        assert trend.slope > 0

    def test_anomaly_detection(self):
        from codeverify_core.quality_forecasting import AnomalyDetector, MetricType

        detector = AnomalyDetector()
        values = [10.0, 11.0, 10.5, 10.0, 11.0, 10.5, 50.0]  # Last is anomaly
        alerts = detector.detect(values, MetricType.FINDINGS)
        assert len(alerts) > 0
        assert alerts[0].severity.value in ("warning", "critical")

    def test_full_forecast(self):
        from codeverify_core.quality_forecasting import QualityForecaster, MetricType

        forecaster = QualityForecaster()
        for i in range(10):
            forecaster.record_metric(MetricType.FINDINGS, float(10 + i), "my-repo")
            forecaster.record_metric(MetricType.TRUST_SCORE, float(70 - i), "my-repo")

        forecast = forecaster.generate_forecast("my-repo")
        assert len(forecast.trends) > 0
        assert forecast.executive_summary != ""
        assert len(forecast.scenarios) > 0

    def test_singleton(self):
        from codeverify_core.quality_forecasting import (
            get_quality_forecaster, reset_quality_forecaster,
        )

        reset_quality_forecaster()
        f1 = get_quality_forecaster()
        f2 = get_quality_forecaster()
        assert f1 is f2
        reset_quality_forecaster()


# ─── Feature 9: Verification-Driven Code Generation ─────────────────


class TestVerifiedCodeGen:
    def test_basic_generation(self):
        from codeverify_core.verified_codegen import (
            VerifiedCodeGenerator, CodeSpec, SpecLanguage, GenerationStatus,
        )

        gen = VerifiedCodeGenerator()
        spec = CodeSpec(
            natural_language="Create a calculator function",
            target_language=SpecLanguage.PYTHON,
        )
        result = gen.generate(spec)
        assert result.status == GenerationStatus.COMPLETED
        assert len(result.candidates) > 0
        assert result.best_candidate is not None

    def test_constraint_checking(self):
        from codeverify_core.verified_codegen import (
            ConstraintChecker, ConstraintType, FormalConstraint, SpecLanguage,
        )

        checker = ConstraintChecker()
        safe_code = "result = x + y\nreturn result"
        constraints = [
            FormalConstraint(
                constraint_type=ConstraintType.SECURITY,
                description="No unsafe code execution",
            ),
        ]
        passed, total, ce = checker.check(safe_code, constraints, SpecLanguage.PYTHON)
        assert passed == 1
        assert total == 1

    def test_security_constraint_fails(self):
        from codeverify_core.verified_codegen import (
            ConstraintChecker, ConstraintType, FormalConstraint, SpecLanguage,
        )

        checker = ConstraintChecker()
        unsafe_code = "result = eval(user_input)"
        constraints = [
            FormalConstraint(
                constraint_type=ConstraintType.SECURITY,
                description="No eval",
            ),
        ]
        passed, total, ce = checker.check(unsafe_code, constraints, SpecLanguage.PYTHON)
        assert passed == 0
        assert len(ce) > 0

    def test_generation_with_constraints(self):
        from codeverify_core.verified_codegen import (
            VerifiedCodeGenerator, CodeSpec, SpecLanguage,
            FormalConstraint, ConstraintType,
        )

        gen = VerifiedCodeGenerator()
        spec = CodeSpec(
            natural_language="Create a safe data processor",
            target_language=SpecLanguage.PYTHON,
            constraints=[
                FormalConstraint(
                    constraint_type=ConstraintType.SECURITY,
                    description="No code execution",
                ),
            ],
            max_candidates=3,
        )
        result = gen.generate(spec)
        assert result.best_candidate is not None
        assert result.best_candidate.constraints_total > 0

    def test_code_spec_serialization(self):
        from codeverify_core.verified_codegen import CodeSpec, SpecLanguage

        spec = CodeSpec(
            natural_language="Build API endpoint",
            target_language=SpecLanguage.TYPESCRIPT,
        )
        d = spec.to_dict()
        assert d["language"] == "typescript"

    def test_singleton(self):
        from codeverify_core.verified_codegen import (
            get_verified_codegen, reset_verified_codegen,
        )

        reset_verified_codegen()
        g1 = get_verified_codegen()
        g2 = get_verified_codegen()
        assert g1 is g2
        reset_verified_codegen()


# ─── Feature 10: Real-Time Collaborative Verification Sessions ─────────────────


class TestCollaborativeVerification:
    def test_create_session(self):
        from codeverify_core.collaborative_verification_sessions import (
            CollaborativeVerificationSession, SessionPhase,
        )

        session = CollaborativeVerificationSession(host_name="Alice")
        assert session.phase == SessionPhase.LOBBY
        assert len(session.participants) == 1
        assert session.participants[0].name == "Alice"

    def test_join_and_leave(self):
        from codeverify_core.collaborative_verification_sessions import (
            CollaborativeVerificationSession, ParticipantRole,
        )

        session = CollaborativeVerificationSession(host_name="Alice")
        bob = session.join("Bob", ParticipantRole.EDITOR)
        assert bob is not None
        assert len(session.participants) == 2

        session.leave(bob.id)
        assert len(session.active_participants) == 1

    def test_session_lifecycle(self):
        from codeverify_core.collaborative_verification_sessions import (
            CollaborativeVerificationSession, SessionPhase,
        )

        session = CollaborativeVerificationSession(host_name="Host")
        session.start()
        assert session.phase == SessionPhase.ACTIVE
        session.pause()
        assert session.phase == SessionPhase.PAUSED
        stats = session.end()
        assert session.phase == SessionPhase.ENDED
        assert stats.duration_seconds >= 0

    def test_code_changes_and_verification(self):
        from codeverify_core.collaborative_verification_sessions import (
            CollaborativeVerificationSession, LineStatus,
        )

        session = CollaborativeVerificationSession(host_name="Dev")
        session.start()
        host = session.participants[0]

        session.submit_code_change(host.id, "main.py", "x = 42")
        assert session.stats.total_edits == 1

        lv = session.update_line_verification("main.py", 1, LineStatus.VERIFIED)
        assert lv.status == LineStatus.VERIFIED
        assert session.stats.lines_verified == 1

    def test_chat_messages(self):
        from codeverify_core.collaborative_verification_sessions import (
            CollaborativeVerificationSession,
        )

        session = CollaborativeVerificationSession(host_name="Alice")
        bob = session.join("Bob")
        assert bob is not None

        msg = session.send_chat(bob.id, "Found an issue on line 42!")
        assert msg is not None
        assert msg.content == "Found an issue on line 42!"
        assert session.stats.messages_sent == 1

    def test_recording(self):
        from codeverify_core.collaborative_verification_sessions import (
            CollaborativeVerificationSession,
        )

        session = CollaborativeVerificationSession(host_name="Host")
        rec = session.start_recording()
        assert rec.session_id == session.id

        session.start()
        host = session.participants[0]
        session.submit_code_change(host.id, "test.py", "code")
        session.send_chat(host.id, "Testing recording")

        assert len(rec.events) >= 2  # system messages + code + chat

    def test_max_participants(self):
        from codeverify_core.collaborative_verification_sessions import (
            CollaborativeVerificationSession,
        )

        session = CollaborativeVerificationSession(host_name="Host")
        for i in range(9):  # host + 9 = 10 (max)
            session.join(f"User{i}")
        extra = session.join("OneMore")
        assert extra is None  # Max reached

    def test_session_manager(self):
        from codeverify_core.collaborative_verification_sessions import (
            CollaborativeSessionManager,
        )

        mgr = CollaborativeSessionManager()
        s1 = mgr.create_session("Alice")
        s2 = mgr.create_session("Bob")
        assert mgr.total_sessions == 2
        assert len(mgr.list_active_sessions()) == 2

        mgr.end_session(s1.id)
        assert len(mgr.list_active_sessions()) == 1

    def test_singleton(self):
        from codeverify_core.collaborative_verification_sessions import (
            get_collab_session_manager, reset_collab_session_manager,
        )

        reset_collab_session_manager()
        m1 = get_collab_session_manager()
        m2 = get_collab_session_manager()
        assert m1 is m2
        reset_collab_session_manager()
