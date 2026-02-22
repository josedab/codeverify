"""Integration tests: Full pipeline webhook → analysis → results.

Tests the complete CodeVerify pipeline:
  GitHub webhook → PR context → agentic orchestrator → Z3 verification
  → conflict resolution → findings → PR comment generation.

These tests use real codeverify_core modules (no external services needed).
"""

import pytest


class TestFullPipelineIntegration:
    """End-to-end pipeline: PR → plan → dispatch → verify → resolve → explain."""

    def test_pr_to_findings_pipeline(self):
        """Simulate: PR opened → orchestrator creates plan → agents run → findings produced."""
        from codeverify_core.agentic_orchestrator import AgenticReviewOrchestrator, PRContext

        ctx = PRContext(
            pr_id="42", repo="acme/api",
            changed_files=[
                {"path": "auth.py"},
                {"path": "utils.py"},
            ],
            labels=["feature"],
        )
        orch = AgenticReviewOrchestrator(budget_cents=100.0)
        result = orch.review(ctx)

        assert result.tasks_completed >= 2
        assert len(result.execution_trace) >= 3
        assert result.total_cost_cents >= 0

    def test_findings_to_explanation_pipeline(self):
        """Simulate: findings → NL proof explanation → PR comment."""
        from codeverify_core.nl_proof_explanation import (
            ExplanationContext,
            NLProofExplanationService,
        )

        svc = NLProofExplanationService()
        contexts = [
            ExplanationContext(
                check_type="null_safety", function_name="get_user",
                file_path="api.py", line=42,
                variable_assignments={"user": None}, severity="high",
            ),
            ExplanationContext(
                check_type="division_by_zero", function_name="calculate",
                file_path="math.py", line=15,
                variable_assignments={"divisor": 0}, severity="critical",
            ),
        ]
        comment = svc.generate_pr_comment(contexts)
        assert "CodeVerify Analysis" in comment
        assert "null" in comment.lower() or "None" in comment

    def test_verification_protocol_end_to_end(self):
        """Simulate: external client → protocol server → verification → proof cert."""
        from codeverify_core.protocol_client import VerificationClient

        client = VerificationClient()  # local mode
        result = client.verify(
            "def process(x):\n    return eval(x)\n",
            language="python",
            file_path="handler.py",
        )
        assert result.status == "failed"
        assert result.finding_count >= 1
        assert any("eval" in f.message.lower() or "injection" in f.message.lower() for f in result.findings)
        assert len(result.proofs) >= 1

    def test_verified_codegen_loop_integration(self):
        """Simulate: counterexample → generate fix → verify → proof certificate."""
        from codeverify_core.verified_codegen_loop import (
            Counterexample,
            VerificationAwareCodeGenService,
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

    def test_drift_then_replay_pipeline(self):
        """Simulate: set baseline → change code → detect drift → record → replay."""
        from codeverify_core.drift_monitor import DriftMonitorService

        svc = DriftMonitorService()
        old = {"app.py": "def calc(x):\n    return x * 2\n"}
        new = {"app.py": "def calc(x, y):\n    return x * y\n"}

        svc.set_baseline("repo", old)
        report = svc.scan("repo", new, commit_sha="abc")
        assert report.functions_drifted >= 1

        # Now replay
        from codeverify_core.verification_replay_regression import VerificationReplayService

        replay_svc = VerificationReplayService()
        session = replay_svc.record_session("repo", "v1", [
            {"check_type": "null_safety", "function_name": "calc",
             "file_path": "app.py", "result": "pass", "code": old["app.py"]},
        ], old)
        replay = replay_svc.replay_session(session.id, new, "v2")
        assert replay.total_checks >= 1

    def test_multimodal_then_search_pipeline(self):
        """Simulate: verify Terraform → index results → search for findings."""
        from codeverify_core.multimodal_verify import MultiModalVerificationService

        mm_svc = MultiModalVerificationService()
        result = mm_svc.verify_file("main.tf", 'cidr_blocks = ["0.0.0.0/0"]')
        assert not result.passed

        from codeverify_core.verification_search import (
            CodeEntity,
            VerificationSearchService,
            VerificationStatus,
        )

        search_svc = VerificationSearchService()
        search_svc.index_entity(CodeEntity(
            file_path="main.tf", verification_status=VerificationStatus.FAILING,
            critical_findings=len(result.findings),
        ))
        results = search_svc.search_structured({"verification_status": "failing"})
        assert results.total_count >= 1

    def test_self_healing_pipeline(self):
        """Simulate: runtime incident → diagnose → generate fix → verify."""
        from codeverify_core.self_healing import RuntimeIncident, SelfHealingService

        svc = SelfHealingService()
        incident = RuntimeIncident(
            function_name="get_user", file_path="api.py",
            error_type="TypeError", error_message="'NoneType' has no attribute 'name'",
            variable_state={"user": None},
        )
        action = svc.report_incident(incident)
        assert action.diagnosis is not None
        assert action.fixed_code != ""
        assert "None" in action.fixed_code or "is not" in action.fixed_code

    def test_compliance_then_review_assignment(self):
        """Simulate: compliance scan → risk classification → reviewer assignment."""
        from codeverify_core.compliance_engine import ComplianceAsCodeService, ComplianceFramework

        comp_svc = ComplianceAsCodeService()
        files = {"auth.py": "import bcrypt\npassword = bcrypt.hashpw(pwd, salt)\n"}
        report = comp_svc.run_framework_audit(ComplianceFramework.SOC2, "repo", files)

        from codeverify_core.review_assignments import ExpertiseArea, Reviewer, ReviewAssignmentService

        rev_svc = ReviewAssignmentService()
        rev_svc.register_reviewer(Reviewer(
            id="r1", name="Alice Security",
            expertise=[ExpertiseArea.SECURITY], seniority="senior",
        ))
        rev_svc.register_reviewer(Reviewer(
            id="r2", name="Bob General",
            expertise=[ExpertiseArea.GENERAL], seniority="mid",
        ))

        has_failures = any(r.status.value == "fail" for r in report.results)
        assignments = rev_svc.assign_reviewer(
            "PR-1", high=2 if has_failures else 0, has_security=True,
        )
        assert len(assignments) >= 1

    def test_credit_system_with_telemetry(self):
        """Simulate: submit telemetry → evaluate credits → benchmark."""
        from codeverify_core.verification_telemetry import VerificationTelemetryService

        tel_svc = VerificationTelemetryService()
        for i in range(6):
            tel_svc.submit_telemetry(f"org{i}", {
                "verification_coverage": 0.7 + i * 0.05,
                "false_positive_rate": 0.15 - i * 0.02,
            })

        from codeverify_core.credit_system import VerificationCreditService

        credit_svc = VerificationCreditService()
        credit_svc.register_org("org0", "Org Zero")
        awarded = credit_svc.evaluate_rules("org0", {"verification_coverage": 0.95, "fix_rate": 0.92})
        assert len(awarded) >= 1
        assert credit_svc.get_balance("org0") > 0

    def test_cicd_pipeline_with_verification(self):
        """Simulate: .verify.yml → create pipeline → execute → check result."""
        from codeverify_core.verification_cicd import PipelineStatus, VerificationCICDService

        svc = VerificationCICDService()
        yml = "name: CI Pipeline\npolicy: strict"
        pipeline = svc.create_pipeline(yml, repo="org/repo", commit_sha="abc123")
        result = svc.run_pipeline(pipeline, {"app.py": "def safe(): return 42\n"})
        assert result.status == PipelineStatus.PASSED
        assert result.duration_ms >= 0
