"""End-to-end characterizations spanning current CodeVerify packages."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest


class TestAIFingerprintingWithReachability:
    """AI fingerprinting results can accompany formal reachability evidence."""

    @pytest.fixture
    def sample_code(self):
        return """
def process_user_input(user_data):
    # This function validates each input step before returning.
    if user_data is not None:
        if isinstance(user_data, str):
            if len(user_data) > 0:
                return user_data.strip()
    return ""

def vulnerable_function(query):
    sql = f"SELECT * FROM users WHERE id = {query}"
    return execute_query(sql)

def main():
    data = process_user_input(get_input())
    return vulnerable_function(data)
"""

    @pytest.mark.asyncio
    async def test_fingerprint_and_analyze_reachability(self, sample_code):
        """Fingerprint metadata and a reachable path share one report."""
        from codeverify_agents import AgentResult, AIFingerprintAgent
        from codeverify_verifier.reachability import (
            CallGraphBuilder,
            ReachabilityAnalyzer,
            ReachabilityStatus,
            create_cve_vulnerability,
        )

        fingerprint = await AIFingerprintAgent().analyze(
            sample_code,
            {"file_path": "test.py", "language": "python"},
        )
        assert isinstance(fingerprint, AgentResult)
        assert fingerprint.success is True
        assert 0 <= fingerprint.data["confidence"] <= 1

        nodes, edges = CallGraphBuilder().build_from_code(sample_code, "test.py")
        assert {"test.py::main", "test.py::vulnerable_function"}.issubset(nodes)
        assert any(
            edge.source == "test.py::main" and edge.target == "test.py::vulnerable_function"
            for edge in edges
        )

        vulnerability = create_cve_vulnerability(
            cve_id="CVE-2024-1234",
            title="SQL injection in user lookup",
            affected_package="application",
            affected_functions=["vulnerable_function"],
            severity="critical",
            cvss_score=9.8,
        )
        reachability = ReachabilityAnalyzer().analyze(
            {"test.py": sample_code},
            [vulnerability],
            entry_points=["main"],
        )
        result = reachability.results[0]

        assert result.status == ReachabilityStatus.CONDITIONAL
        assert result.reachable_paths == [["main", "vulnerable_function"]]
        assert result.entry_points == ["main"]
        assert result.conditions == ["Conditional call at line 5"]

        report = {
            "fingerprint": fingerprint.data,
            "vulnerability": result.to_dict(),
        }
        assert report["fingerprint"]["detected_model"]
        assert report["vulnerability"]["vulnerability"]["cve_id"] == "CVE-2024-1234"


class TestSBOMWithVerificationAttestations:
    """Formal verification evidence flows into exported SBOM metadata."""

    @pytest.fixture
    def sample_dependencies(self):
        return [
            {"name": "requests", "version": "2.28.0", "ecosystem": "pypi"},
            {"name": "flask", "version": "2.3.0", "ecosystem": "pypi"},
            {"name": "sqlalchemy", "version": "2.0.0", "ecosystem": "pypi"},
        ]

    def test_generate_sbom_with_verification(self, sample_dependencies):
        """A proven condition is embedded in a signed-build SBOM export."""
        from codeverify_core.sbom import (
            SBOMFormat,
            SBOMGenerator,
            SLSAAttestationGenerator,
            SLSALevel,
            VerifiedSBOMExporter,
        )
        from codeverify_verifier import Z3Verifier

        proof = Z3Verifier().check_division_by_zero(
            divisor_var="denominator",
            divisor_range=(1, 100),
        )
        assert proof["satisfiable"] is False

        now = datetime.now(UTC)
        provenance = SLSAAttestationGenerator(builder_id="codeverify-ci").generate(
            source_uri="https://github.com/test/repo",
            source_commit="abc123",
            build_started=now,
            build_finished=now,
            entry_point="pytest",
            materials=[{"uri": "pkg:pypi/pytest@9.0.2"}],
            level=SLSALevel.LEVEL_3,
        )
        sbom = SBOMGenerator(author_name="CodeVerify").generate(
            project_name="test-project",
            dependencies=sample_dependencies,
            verification_results={
                "verification_type": "formal",
                "passed": True,
                "conditions_checked": 1,
                "conditions_passed": 1,
                "findings": [],
            },
            slsa_provenance=provenance,
        )
        exported = VerifiedSBOMExporter().export(
            sbom,
            format=SBOMFormat.CYCLONEDX,
            sign=False,
        )

        assert len(sbom.components) == 3
        assert sbom.slsa_provenance is provenance
        assert sbom.verification_attestation.verification_passed is True
        assert exported["format"] == "cyclonedx"
        assert len(exported["sbom"]["components"]) == 3
        assert exported["verification_badge"] == {
            "passed": True,
            "type": "formal",
            "findings": 0,
            "critical": 0,
        }


class TestAgenticAutoFixWithRuntimeProbes:
    """Agentic fix orchestration can feed a production runtime guard."""

    @pytest.fixture
    def buggy_code(self):
        return """
def normalize(value):
    return value.strip()
"""

    @pytest.mark.asyncio
    async def test_autofix_and_generate_probes(self, buggy_code):
        """A verified fix result and its precondition use current contracts."""
        from codeverify_agents import (
            AgenticAutoFix,
            AgentResult,
            FixStatus,
            GeneratedFix,
        )
        from codeverify_verifier.runtime_probes import ProbeGenerator, RuntimeMonitor

        fixed_code = """
def normalize(value):
    if value is None:
        raise ValueError("value is required")
    return value.strip()
"""
        candidate = GeneratedFix(
            id="fix-1",
            finding_id="finding-1",
            status=FixStatus.PENDING,
            original_code=buggy_code,
            fixed_code=fixed_code,
            diff="-    return value.strip()\n+    if value is None: ...",
            explanation="Guard the dereference with an explicit null check.",
            confidence=0.9,
        )
        autofix = AgenticAutoFix()
        autofix._generator.generate_fix = AsyncMock(return_value=[candidate])

        result = await autofix.analyze(
            buggy_code,
            {
                "language": "python",
                "findings": [
                    {
                        "id": "finding-1",
                        "title": "Null dereference",
                        "description": "value may be None",
                        "category": "null_safety",
                        "severity": "high",
                        "file_path": "normalize.py",
                        "line_start": 2,
                        "line_end": 2,
                        "code_snippet": "return value.strip()",
                    }
                ],
            },
        )

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.data["fixes_ready"] == 1
        assert result.data["fixes"][0]["status"] == "ready_for_pr"
        assert result.data["fixes"][0]["verification_result"]["verified"] is True

        RuntimeMonitor.reset()
        monitor = RuntimeMonitor.get_instance()
        spec = ProbeGenerator().from_z3_spec(
            z3_spec="value is not None",
            function_name="normalize",
            parameters=["value"],
        )
        monitor.register_spec(spec)

        assert monitor.check_spec(spec.id, value="text") is True
        assert monitor.check_spec(spec.id, value=None) is False
        assert monitor.get_stats()["total_violations"] == 1
        RuntimeMonitor.reset()


class TestCodebaseIntelligenceWithROI:
    """Historical bug intelligence can be valued by the ROI dashboard."""

    @pytest.fixture
    def sample_findings(self):
        return [
            {"severity": "critical", "type": "sql_injection", "file": "auth.py"},
            {"severity": "high", "type": "xss", "file": "views.py"},
            {"severity": "medium", "type": "null_pointer", "file": "utils.py"},
        ]

    def test_intelligence_feeds_roi(self, sample_findings):
        """Bug correlations retain enough metadata for ROI ingestion."""
        from codeverify_agents import CodebaseIntelligenceEngine
        from codeverify_core.roi_dashboard import ROIDashboard

        intelligence = CodebaseIntelligenceEngine()
        for index, finding in enumerate(sample_findings, start=1):
            intelligence.bug_tracker.record_bug(
                file_path=finding["file"],
                bug_id=f"BUG-{index}",
                bug_title=finding["type"],
                introduced_commit=f"commit-{index}",
                pattern_id=finding["type"],
                severity=finding["severity"],
            )

        dashboard = ROIDashboard()
        bugs_found = [
            {
                "severity": bug.severity,
                "title": bug.bug_title,
                "description": f"Historical issue in {bug.file_path}",
                "file_path": bug.file_path,
                "finding_type": bug.pattern_id,
            }
            for bug in intelligence.bug_tracker.bugs
        ]
        recorded = dashboard.record_pr_analysis(
            repository="owner/repo",
            pr_number=42,
            lines_of_code=120,
            input_tokens=4_000,
            output_tokens=1_000,
            z3_seconds=10.5,
            bugs_found=bugs_found,
        )
        metrics = dashboard.calculate_roi()

        assert recorded["bugs_found"] == 3
        assert metrics.bugs_caught == 3
        assert metrics.bugs_by_severity == {"critical": 1, "high": 1, "medium": 1}
        assert metrics.estimated_cost_avoided > metrics.total_cost_usd


class TestIntentTraceabilityWithUniversalGit:
    """Git webhooks and issue intent combine into traceability evidence."""

    @pytest.fixture
    def mock_ticket(self):
        return {
            "id": "PROJ-123",
            "title": "Add rate limiting to API",
            "description": (
                "Implement rate limiting for /api/users.\n"
                "- Limit requests to 100 per minute\n"
                "- Return 429 when exceeded"
            ),
            "acceptance_criteria": [
                "Rate limit should be 100 req/min",
                "Return 429 status when exceeded",
            ],
        }

    @pytest.fixture
    def mock_diff(self):
        return """
diff --git a/api/routes.py b/api/routes.py
+++ b/api/routes.py
+def get_users():
+    limiter.check(max_requests=100)
+    return users_list()
"""

    def test_traceability_with_webhook_processing(self, mock_ticket, mock_diff):
        """Parsed PR metadata is joined with current intent-alignment models."""
        from codeverify_agents import (
            AlignmentChecker,
            CodeChangeAnalyzer,
            IntentExtractor,
            IssueDetails,
            IssueProvider,
        )
        from codeverify_core.universal_git import (
            GitCredentials,
            GitHubAdapter,
            GitProvider,
            WebhookEventType,
        )

        webhook_payload = {
            "action": "opened",
            "pull_request": {
                "id": 1,
                "number": 42,
                "title": "PROJ-123: Add rate limiting",
                "body": f"Implements {mock_ticket['id']}",
                "head": {"ref": "feature/rate-limiting"},
                "base": {"ref": "main"},
                "user": {"login": "developer"},
                "state": "open",
                "html_url": "https://github.com/org/repo/pull/42",
                "diff_url": "https://github.com/org/repo/pull/42.diff",
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:00:00Z",
            },
            "repository": {
                "name": "repo",
                "owner": {"login": "org"},
                "clone_url": "https://github.com/org/repo.git",
                "default_branch": "main",
            },
            "sender": {"login": "developer"},
        }
        payload = GitHubAdapter(GitCredentials(provider=GitProvider.GITHUB)).parse_webhook(
            {"X-GitHub-Event": "pull_request"}, webhook_payload
        )

        issue = IssueDetails(
            id=mock_ticket["id"],
            provider=IssueProvider.JIRA,
            key=mock_ticket["id"],
            title=mock_ticket["title"],
            description=mock_ticket["description"],
            issue_type="feature",
            status="open",
            acceptance_criteria=mock_ticket["acceptance_criteria"],
        )
        intent = IntentExtractor().extract_intent(issue)
        changes = CodeChangeAnalyzer().analyze_diff(
            mock_diff,
            [
                {
                    "filename": "api/routes.py",
                    "status": "modified",
                    "additions": 3,
                    "deletions": 0,
                }
            ],
        )
        alignment_score, findings = AlignmentChecker().check_alignment(intent, changes)

        assert payload.event_type == WebhookEventType.PULL_REQUEST
        assert payload.pull_request.number == 42
        assert payload.repository.owner == "org"
        assert intent.acceptance_criteria == mock_ticket["acceptance_criteria"]
        assert changes.functions_modified == ["get_users"]
        assert changes.detected_scope == intent.change_scope
        assert alignment_score > 0.5
        assert all(finding.type != "scope_mismatch" for finding in findings)


class TestCounterexamplePlaygroundWithVerification:
    """A Z3 model can be explored and exported without external services."""

    @pytest.fixture
    def z3_counterexample(self):
        return """sat
(model
  (define-fun x () Int -5)
  (define-fun y () Int 0)
  (define-fun arr_len () Int 3)
  (define-fun idx () Int 10)
)"""

    @pytest.fixture
    def source_with_bug(self):
        return """
def calculate(x, y):
    assert x > 0, "x must be positive"
    return x / y
"""

    def test_playground_from_verification_failure(self, z3_counterexample, source_with_bug):
        """A parsed counterexample remains navigable across export formats."""
        from codeverify_verifier.counterexample_playground import (
            PlaygroundEngine,
            Z3ModelParser,
        )

        counterexample = Z3ModelParser().parse(z3_counterexample, source_with_bug)
        assert counterexample.variables["x"].value == -5
        assert counterexample.variables["y"].value == 0

        engine = PlaygroundEngine()
        session = engine.create_session(
            z3_output=z3_counterexample,
            source_code=source_with_bug,
            function_name="calculate",
        )
        engine.step_forward(session.session_id)
        assert engine.modify_value(session.session_id, "x", 10) is True

        html = engine.export_html(session.session_id)
        mermaid = engine.export_mermaid(session.session_id)
        share_link = engine.generate_share_link(session.session_id)

        assert session.modified_values["x"] == 10
        assert "<!DOCTYPE html>" in html
        assert "flowchart TD" in mermaid
        assert share_link == f"/playground/{session.session_id}"


class TestFullPipelineIntegration:
    """A PR can flow from webhook intake to verification and reporting."""

    @pytest.fixture
    def pr_payload(self):
        return {
            "action": "opened",
            "pull_request": {
                "id": 12345,
                "number": 100,
                "title": "PROJ-456: Fix SQL injection vulnerability",
                "body": "This PR addresses unsafe user lookup",
                "head": {"ref": "fix/sql-injection"},
                "base": {"ref": "main"},
                "user": {"login": "security-dev"},
                "state": "open",
                "html_url": "https://github.com/org/repo/pull/100",
                "diff_url": "https://github.com/org/repo/pull/100.diff",
                "created_at": "2024-01-20T14:00:00Z",
                "updated_at": "2024-01-20T14:00:00Z",
            },
            "repository": {
                "name": "secure-app",
                "owner": {"login": "org"},
                "clone_url": "https://github.com/org/secure-app.git",
                "default_branch": "main",
            },
            "sender": {"login": "security-dev"},
        }

    @pytest.fixture
    def vulnerable_code(self):
        return """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)

def main():
    return get_user(read_user_id())
"""

    @pytest.mark.asyncio
    async def test_full_verification_pipeline(self, pr_payload, vulnerable_code):
        """Current public APIs produce one coherent, evidence-backed report."""
        from codeverify_agents import AIFingerprintAgent
        from codeverify_core.roi_dashboard import ROIDashboard
        from codeverify_core.rules import RuleBuilder, RuleEvaluator
        from codeverify_core.sbom import SBOMGenerator, VerifiedSBOMExporter
        from codeverify_core.universal_git import GitCredentials, GitHubAdapter, GitProvider
        from codeverify_verifier.reachability import (
            ReachabilityAnalyzer,
            ReachabilityStatus,
            create_cve_vulnerability,
        )

        webhook = GitHubAdapter(GitCredentials(provider=GitProvider.GITHUB)).parse_webhook(
            {"X-GitHub-Event": "pull_request"}, pr_payload
        )
        fingerprint = await AIFingerprintAgent().analyze(
            vulnerable_code,
            {"file_path": "user_service.py", "language": "python"},
        )

        sql_rule = (
            RuleBuilder()
            .name("No interpolated SQL")
            .description("Reject interpolated SELECT statements")
            .severity("critical")
            .pattern(r'f"SELECT[^"]*\{')
            .action("Use a parameterized query")
            .for_languages("python")
            .build()
        )
        violations = RuleEvaluator([sql_rule]).evaluate(
            vulnerable_code,
            "user_service.py",
            "python",
        )

        vulnerability = create_cve_vulnerability(
            "CVE-2024-SQL-001",
            "SQL injection in user lookup",
            "secure-app",
            ["get_user"],
            severity="critical",
            cvss_score=9.8,
        )
        reachability = (
            ReachabilityAnalyzer()
            .analyze(
                {"user_service.py": vulnerable_code},
                [vulnerability],
                entry_points=["main"],
            )
            .results[0]
        )

        dashboard = ROIDashboard()
        dashboard.record_pr_analysis(
            repository="org/secure-app",
            pr_number=webhook.pull_request.number,
            lines_of_code=len(vulnerable_code.splitlines()),
            input_tokens=3_000,
            output_tokens=750,
            z3_seconds=5.2,
            bugs_found=[
                {
                    "severity": "critical",
                    "title": violations[0]["message"],
                    "description": "Interpolated SQL is reachable from main.",
                    "file_path": violations[0]["file_path"],
                    "finding_type": "sql_injection",
                }
            ],
        )
        roi = dashboard.calculate_roi()

        sbom = SBOMGenerator(author_name="CodeVerify").generate(
            project_name="secure-app",
            dependencies=[{"name": "sqlalchemy", "version": "2.0.0", "ecosystem": "pypi"}],
            verification_results={
                "verification_type": "hybrid",
                "passed": False,
                "conditions_checked": 2,
                "conditions_passed": 0,
                "findings": [{"severity": "critical"}],
            },
        )
        sbom_export = VerifiedSBOMExporter().export(sbom, sign=False)

        final_report = {
            "pr_number": webhook.pull_request.number,
            "fingerprint": fingerprint.data,
            "violations": violations,
            "reachability": reachability.to_dict(),
            "roi": roi.to_dict(),
            "sbom": sbom_export,
            "verdict": "NEEDS_REVIEW",
        }

        assert final_report["pr_number"] == 100
        assert fingerprint.success is True
        assert len(final_report["violations"]) == 1
        assert reachability.status == ReachabilityStatus.REACHABLE
        assert roi.bugs_by_severity == {"critical": 1}
        assert sbom_export["verification_badge"]["passed"] is False
        assert final_report["verdict"] == "NEEDS_REVIEW"


class TestCrossFeatureDataFlow:
    """Focused data-flow checks between independently versioned packages."""

    def test_bug_flows_from_intelligence_to_roi(self):
        """A BugCorrelation maps losslessly into ROI bug accounting."""
        from codeverify_agents import BugTracker
        from codeverify_core.roi_dashboard import ROIDashboard

        tracker = BugTracker()
        bug = tracker.record_bug(
            file_path="auth.py",
            bug_id="BUG-001",
            bug_title="Authentication bypass",
            introduced_commit="bad-commit",
            pattern_id="authentication_bypass",
            severity="critical",
        )

        dashboard = ROIDashboard()
        dashboard.record_pr_analysis(
            repository="owner/repo",
            pr_number=7,
            lines_of_code=25,
            input_tokens=500,
            output_tokens=100,
            z3_seconds=0.2,
            bugs_found=[
                {
                    "severity": bug.severity,
                    "title": bug.bug_title,
                    "description": f"Tracked as {bug.bug_id}",
                    "file_path": bug.file_path,
                    "finding_type": bug.pattern_id,
                }
            ],
        )
        metrics = dashboard.calculate_roi()

        assert tracker.get_bugs_for_file("auth.py") == [bug]
        assert metrics.bugs_caught == 1
        assert metrics.bugs_by_severity["critical"] == 1
        assert metrics.estimated_cost_avoided == 150_000

    def test_verification_creates_runtime_spec(self):
        """A formal constraint becomes an executable runtime precondition."""
        from codeverify_verifier.runtime_probes import ProbeGenerator, RuntimeMonitor

        RuntimeMonitor.reset()
        generator = ProbeGenerator()
        spec = generator.from_z3_spec(
            z3_spec="len(password) >= 8",
            function_name="validate_password",
            parameters=["password"],
        )
        monitor = RuntimeMonitor.get_instance()
        monitor.register_spec(spec)

        assert monitor.check_spec(spec.id, password="correct-horse") is True
        assert monitor.check_spec(spec.id, password="short") is False
        assert "len(password) >= 8" in generator.generate_python_decorator(spec)
        assert monitor.get_violations(spec_id=spec.id)[0].function_name == "validate_password"
        RuntimeMonitor.reset()

    def test_counterexample_from_autofix_verification(self):
        """A verification counterexample remains consumable by the playground."""
        from codeverify_verifier.counterexample_playground import PlaygroundEngine

        z3_output = """sat
(model
  (define-fun input_length () Int 0)
  (define-fun max_length () Int 100)
)"""
        session = PlaygroundEngine().create_session(z3_output)

        assert "input_length" in session.counterexample.variables
        assert session.counterexample.variables["input_length"].value == 0
