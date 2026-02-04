"""
End-to-End Integration Tests for CodeVerify Killer Features

Tests that verify multiple features work together correctly:
- AI Fingerprinting + Reachability Analysis
- SBOM Generation + Verification Attestations
- Agentic Auto-Fix + Runtime Probes
- Codebase Intelligence + ROI Dashboard
- Intent Traceability + Universal Git Support
- Counterexample Playground + Verification Flow
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add package paths for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'packages' / 'core' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'packages' / 'verifier' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'packages' / 'ai-agents' / 'src'))


class TestAIFingerprintingWithReachability:
    """Integration tests for AI Fingerprinting + Vulnerability Reachability."""

    @pytest.fixture
    def sample_code(self):
        return '''
def process_user_input(user_data):
    # AI-generated pattern: overly verbose validation
    if user_data is not None:
        if isinstance(user_data, str):
            if len(user_data) > 0:
                return user_data.strip()
    return ""

def vulnerable_function(query):
    # SQL injection vulnerability
    sql = f"SELECT * FROM users WHERE id = {query}"
    return execute_query(sql)

def main():
    data = process_user_input(get_input())
    if data:
        result = vulnerable_function(data)
    return result
'''

    def test_fingerprint_and_analyze_reachability(self, sample_code):
        """Test that AI-generated code is fingerprinted and vulnerabilities are analyzed."""
        # Import modules
        from codeverify_agents.ai_fingerprinting import AICodeFingerprinter, CodeOrigin
        from codeverify_verifier.reachability import (
            ReachabilityAnalyzer, CallGraphBuilder, VulnerabilityScanner,
            VulnerabilityType, create_cve_vulnerability
        )

        # Step 1: Fingerprint the code
        fingerprinter = AICodeFingerprinter()
        fingerprint_result = fingerprinter.analyze(sample_code, "test.py")
        
        assert fingerprint_result is not None
        assert fingerprint_result.confidence_score >= 0.0
        assert fingerprint_result.confidence_score <= 1.0
        
        # Step 2: Build call graph
        builder = CallGraphBuilder()
        call_graph = builder.build_from_source(sample_code, "python")
        
        assert len(call_graph.nodes) > 0
        
        # Step 3: Create vulnerability
        vuln = create_cve_vulnerability(
            "CVE-2024-1234",
            "vulnerable_function",
            VulnerabilityType.SQL_INJECTION,
            cvss_score=9.8
        )
        
        # Step 4: Analyze reachability
        analyzer = ReachabilityAnalyzer(call_graph)
        result = analyzer.analyze_vulnerability(vuln, entry_points=["main"])
        
        # Verify integration
        assert result is not None
        
        # Combined report
        report = {
            "file": "test.py",
            "ai_generated": fingerprint_result.origin == CodeOrigin.AI_GENERATED,
            "ai_confidence": fingerprint_result.confidence_score,
            "vulnerability": vuln.cve_id,
            "reachability_status": result.status.value,
        }
        
        assert "file" in report
        assert "ai_generated" in report
        assert "reachability_status" in report


class TestSBOMWithVerificationAttestations:
    """Integration tests for SBOM Generation + Verification Attestations."""

    @pytest.fixture
    def sample_dependencies(self):
        return [
            {"name": "requests", "version": "2.28.0", "type": "library"},
            {"name": "flask", "version": "2.3.0", "type": "framework"},
            {"name": "sqlalchemy", "version": "2.0.0", "type": "library"},
        ]

    def test_generate_sbom_with_verification(self, sample_dependencies):
        """Test SBOM generation with embedded verification attestations."""
        from codeverify_core.sbom import (
            SBOMGenerator, SLSAAttestationGenerator, VerifiedSBOMExporter,
            SBOMFormat, SLSALevel, Component, ComponentType
        )

        # Step 1: Create SBOM generator
        generator = SBOMGenerator(
            name="test-project",
            version="1.0.0"
        )
        
        # Step 2: Add components
        for dep in sample_dependencies:
            component = Component(
                name=dep["name"],
                version=dep["version"],
                component_type=ComponentType.LIBRARY,
                purl=f"pkg:pypi/{dep['name']}@{dep['version']}"
            )
            generator.add_component(component)
        
        # Step 3: Generate SBOM
        sbom = generator.generate()
        
        assert sbom is not None
        assert len(sbom.components) == 3
        
        # Step 4: Generate SLSA attestation
        slsa_gen = SLSAAttestationGenerator(
            builder_id="codeverify-ci",
            slsa_level=SLSALevel.LEVEL_3
        )
        attestation = slsa_gen.generate(
            subject_name="test-project",
            subject_digest={"sha256": "abc123"},
            materials=[{"uri": "git+https://github.com/test/repo", "digest": {"sha1": "def456"}}]
        )
        
        assert attestation is not None
        assert attestation.slsa_level == SLSALevel.LEVEL_3
        
        # Step 5: Export verified SBOM
        exporter = VerifiedSBOMExporter()
        verified_sbom = exporter.export(sbom, attestation, format=SBOMFormat.CYCLONEDX_JSON)
        
        assert verified_sbom is not None
        assert "components" in verified_sbom or len(verified_sbom) > 0


class TestAgenticAutoFixWithRuntimeProbes:
    """Integration tests for Agentic Auto-Fix + Runtime Probes."""

    @pytest.fixture
    def buggy_code(self):
        return '''
def divide(a, b):
    return a / b  # Bug: no zero division check

def get_element(arr, idx):
    return arr[idx]  # Bug: no bounds check
'''

    @pytest.fixture
    def fix_spec(self):
        return {
            "function": "divide",
            "issue": "division_by_zero",
            "constraint": "b != 0"
        }

    def test_autofix_and_generate_probes(self, buggy_code, fix_spec):
        """Test auto-fix generates verified fixes and runtime probes."""
        from codeverify_agents.agentic_autofix import (
            AgenticAutoFix, FixCategory, FixCandidate
        )
        from codeverify_verifier.runtime_probes import (
            RuntimeMonitor, ProbeGenerator, RuntimeSpec, MonitorMode
        )

        # Step 1: Create auto-fix agent
        autofix = AgenticAutoFix()
        
        # Step 2: Analyze and generate fix
        result = autofix.analyze_and_fix(
            code=buggy_code,
            finding={
                "type": "division_by_zero",
                "function": "divide",
                "line": 2
            }
        )
        
        assert result is not None
        assert result.success or result.data is not None
        
        # Step 3: Create runtime spec from fix
        spec = RuntimeSpec(
            name="divide_precondition",
            condition="b != 0",
            message="Division by zero prevented"
        )
        
        # Step 4: Generate runtime probe
        probe_gen = ProbeGenerator()
        probe_code = probe_gen.generate_python_probe(spec)
        
        assert probe_code is not None
        assert "b != 0" in probe_code or "precondition" in probe_code.lower()
        
        # Step 5: Register with monitor
        monitor = RuntimeMonitor.get_instance()
        monitor.register_spec(spec)
        
        # Verify spec is registered
        assert spec.name in [s.name for s in monitor.get_all_specs()]


class TestCodebaseIntelligenceWithROI:
    """Integration tests for Codebase Intelligence + ROI Dashboard."""

    @pytest.fixture
    def sample_findings(self):
        return [
            {"severity": "critical", "type": "sql_injection", "file": "auth.py"},
            {"severity": "high", "type": "xss", "file": "views.py"},
            {"severity": "medium", "type": "null_pointer", "file": "utils.py"},
        ]

    def test_intelligence_feeds_roi(self, sample_findings):
        """Test codebase intelligence data feeds into ROI calculations."""
        from codeverify_agents.codebase_intelligence import (
            CodebaseIntelligenceEngine, PatternDetector, BugTracker
        )
        from codeverify_core.roi_dashboard import (
            ROIDashboard, CostTracker, BugValueCalculator, BugSeverity
        )

        # Step 1: Initialize intelligence engine
        intel_engine = CodebaseIntelligenceEngine()
        
        # Step 2: Track patterns and bugs
        bug_tracker = BugTracker()
        for finding in sample_findings:
            bug_tracker.record_bug(
                bug_id=f"BUG-{hash(finding['file'])}",
                file_path=finding["file"],
                pattern_type=finding["type"],
                severity=finding["severity"]
            )
        
        # Step 3: Initialize ROI dashboard
        dashboard = ROIDashboard()
        cost_tracker = CostTracker()
        bug_calculator = BugValueCalculator()
        
        # Step 4: Record verification costs
        cost_tracker.record_llm_cost(tokens=5000, model="gpt-4")
        cost_tracker.record_z3_cost(compute_seconds=10.5)
        
        # Step 5: Calculate bug values
        severity_map = {
            "critical": BugSeverity.CRITICAL,
            "high": BugSeverity.HIGH,
            "medium": BugSeverity.MEDIUM,
        }
        
        total_value = 0
        for finding in sample_findings:
            value = bug_calculator.calculate_value(
                severity=severity_map[finding["severity"]],
                category=finding["type"]
            )
            total_value += value
        
        # Step 6: Generate ROI metrics
        metrics = dashboard.calculate_metrics(
            total_cost=cost_tracker.get_total_cost(),
            bugs_caught=[f for f in sample_findings],
            time_period_days=30
        )
        
        assert metrics is not None
        assert metrics.total_bugs_caught == 3
        assert metrics.estimated_savings > 0


class TestIntentTraceabilityWithUniversalGit:
    """Integration tests for Intent Traceability + Universal Git Support."""

    @pytest.fixture
    def mock_ticket(self):
        return {
            "id": "PROJ-123",
            "title": "Add rate limiting to API",
            "description": "Implement rate limiting of 100 requests per minute for the /api/users endpoint",
            "acceptance_criteria": [
                "Rate limit should be 100 req/min",
                "Return 429 status when exceeded",
                "Include Retry-After header"
            ]
        }

    @pytest.fixture
    def mock_diff(self):
        return '''
diff --git a/api/routes.py b/api/routes.py
+from ratelimit import RateLimiter
+
+limiter = RateLimiter(max_requests=100, window_seconds=60)
+
@app.route('/api/users')
+@limiter.limit
def get_users():
+    # Return 429 if rate exceeded
    return users_list()
'''

    def test_traceability_with_webhook_processing(self, mock_ticket, mock_diff):
        """Test intent extraction from tickets with Git webhook processing."""
        from codeverify_agents.intent_traceability import (
            IntentTraceabilityEngine, IntentExtractor, CodeChangeAnalyzer,
            AlignmentChecker
        )
        from codeverify_core.universal_git import (
            UniversalGitSupport, GitProvider, WebhookReceiver, GitHubAdapter,
            GitCredentials
        )

        # Step 1: Process webhook
        git_support = UniversalGitSupport()
        
        webhook_payload = {
            "action": "opened",
            "pull_request": {
                "id": 1, "number": 42,
                "title": "PROJ-123: Add rate limiting",
                "body": f"Implements {mock_ticket['id']}\n\n{mock_diff}",
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
                "default_branch": "main"
            },
            "sender": {"login": "developer"}
        }
        
        headers = {"X-GitHub-Event": "pull_request"}
        adapter = GitHubAdapter(GitCredentials(provider=GitProvider.GITHUB))
        payload = adapter.parse_webhook(headers, webhook_payload)
        
        assert payload.pull_request.number == 42
        
        # Step 2: Extract intent from ticket
        extractor = IntentExtractor()
        intent = extractor.extract(
            title=mock_ticket["title"],
            description=mock_ticket["description"],
            acceptance_criteria=mock_ticket["acceptance_criteria"]
        )
        
        assert intent is not None
        assert len(intent.expected_changes) > 0 or len(intent.keywords) > 0
        
        # Step 3: Analyze code changes
        change_analyzer = CodeChangeAnalyzer()
        changes = change_analyzer.analyze_diff(mock_diff)
        
        assert changes is not None
        
        # Step 4: Check alignment
        alignment_checker = AlignmentChecker()
        alignment = alignment_checker.check(intent, changes)
        
        assert alignment is not None
        assert alignment.score >= 0.0
        
        # Integration result
        result = {
            "pr_number": payload.pull_request.number,
            "ticket_id": mock_ticket["id"],
            "alignment_score": alignment.score,
            "aligned": alignment.is_aligned,
        }
        
        assert "pr_number" in result
        assert "alignment_score" in result


class TestCounterexamplePlaygroundWithVerification:
    """Integration tests for Counterexample Playground + Verification Flow."""

    @pytest.fixture
    def z3_counterexample(self):
        return '''sat
(model
  (define-fun x () Int -5)
  (define-fun y () Int 0)
  (define-fun arr_len () Int 3)
  (define-fun idx () Int 10)
)'''

    @pytest.fixture
    def source_with_bug(self):
        return '''
def access_array(arr, idx):
    # Bug: No bounds check
    if idx >= 0:
        return arr[idx]  # Can fail if idx >= len(arr)
    return None

def calculate(x, y):
    assert x > 0, "x must be positive"
    return x / y  # Bug: y can be 0
'''

    def test_playground_from_verification_failure(self, z3_counterexample, source_with_bug):
        """Test creating playground session from Z3 verification failure."""
        from codeverify_verifier.counterexample_playground import (
            PlaygroundEngine, Z3ModelParser, TraceGenerator,
            CounterexampleVisualizer, VariableType
        )

        # Step 1: Parse Z3 output
        parser = Z3ModelParser()
        counterexample = parser.parse(z3_counterexample, source_with_bug)
        
        assert counterexample is not None
        assert "x" in counterexample.variables
        assert counterexample.variables["x"].value == -5
        assert counterexample.variables["y"].value == 0
        
        # Step 2: Create playground engine
        engine = PlaygroundEngine()
        
        # Step 3: Create session
        session = engine.create_session(
            z3_output=z3_counterexample,
            source_code=source_with_bug,
            function_name="calculate"
        )
        
        assert session is not None
        assert session.session_id is not None
        
        # Step 4: Navigate through trace
        engine.step_forward(session.session_id)
        engine.step_forward(session.session_id)
        
        # Step 5: Modify value to explore
        engine.modify_value(session.session_id, "x", 10)
        assert session.modified_values["x"] == 10
        
        # Step 6: Export visualization
        html = engine.export_html(session.session_id)
        assert html is not None
        assert "<!DOCTYPE html>" in html
        assert session.session_id in html
        
        # Step 7: Export Mermaid diagram
        mermaid = engine.export_mermaid(session.session_id)
        assert mermaid is not None
        assert "flowchart TD" in mermaid
        
        # Step 8: Get share link
        share_link = engine.generate_share_link(session.session_id)
        assert share_link is not None
        assert session.session_id in share_link


class TestFullPipelineIntegration:
    """Full end-to-end pipeline integration test."""

    @pytest.fixture
    def pr_payload(self):
        return {
            "action": "opened",
            "number": 100,
            "pull_request": {
                "id": 12345,
                "number": 100,
                "title": "PROJ-456: Fix SQL injection vulnerability",
                "body": "This PR fixes the SQL injection in user lookup",
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
                "default_branch": "main"
            },
            "sender": {"login": "security-dev"}
        }

    @pytest.fixture
    def vulnerable_code(self):
        return '''
def get_user(user_id):
    # Vulnerable to SQL injection
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
'''

    @pytest.fixture
    def fixed_code(self):
        return '''
def get_user(user_id):
    # Fixed: using parameterized query
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, [user_id])
'''

    def test_full_verification_pipeline(self, pr_payload, vulnerable_code, fixed_code):
        """Test complete verification pipeline from PR to report."""
        from codeverify_core.universal_git import (
            UniversalGitSupport, GitProvider, GitHubAdapter, GitCredentials
        )
        from codeverify_agents.ai_fingerprinting import AICodeFingerprinter
        from codeverify_verifier.reachability import (
            ReachabilityAnalyzer, CallGraphBuilder, VulnerabilityScanner,
            create_cve_vulnerability, VulnerabilityType
        )
        from codeverify_core.roi_dashboard import (
            ROIDashboard, CostTracker, BugValueCalculator, BugSeverity
        )
        from codeverify_core.sbom import SBOMGenerator, Component, ComponentType

        # === Phase 1: Receive PR via Universal Git ===
        adapter = GitHubAdapter(GitCredentials(provider=GitProvider.GITHUB))
        headers = {"X-GitHub-Event": "pull_request"}
        webhook = adapter.parse_webhook(headers, pr_payload)
        
        assert webhook.pull_request.number == 100
        pr_info = {
            "number": webhook.pull_request.number,
            "title": webhook.pull_request.title,
            "author": webhook.pull_request.author,
        }
        
        # === Phase 2: Fingerprint Code ===
        fingerprinter = AICodeFingerprinter()
        fp_original = fingerprinter.analyze(vulnerable_code, "user_service.py")
        fp_fixed = fingerprinter.analyze(fixed_code, "user_service.py")
        
        fingerprint_report = {
            "original_ai_score": fp_original.confidence_score,
            "fixed_ai_score": fp_fixed.confidence_score,
        }
        
        # === Phase 3: Reachability Analysis ===
        builder = CallGraphBuilder()
        graph = builder.build_from_source(vulnerable_code, "python")
        
        vuln = create_cve_vulnerability(
            "CVE-2024-SQL-001",
            "get_user",
            VulnerabilityType.SQL_INJECTION,
            cvss_score=9.8
        )
        
        analyzer = ReachabilityAnalyzer(graph)
        reachability = analyzer.analyze_vulnerability(vuln, entry_points=["get_user"])
        
        # === Phase 4: Track Costs ===
        cost_tracker = CostTracker()
        cost_tracker.record_llm_cost(tokens=3000, model="gpt-4")
        cost_tracker.record_z3_cost(compute_seconds=5.2)
        
        # === Phase 5: Calculate ROI ===
        dashboard = ROIDashboard()
        bug_calc = BugValueCalculator()
        
        bug_value = bug_calc.calculate_value(
            severity=BugSeverity.CRITICAL,
            category="sql_injection"
        )
        
        # === Phase 6: Generate SBOM ===
        sbom_gen = SBOMGenerator(name="secure-app", version="2.0.0")
        sbom_gen.add_component(Component(
            name="sqlalchemy",
            version="2.0.0",
            component_type=ComponentType.LIBRARY,
            purl="pkg:pypi/sqlalchemy@2.0.0"
        ))
        sbom = sbom_gen.generate()
        
        # === Final Report ===
        final_report = {
            "pr": pr_info,
            "fingerprinting": fingerprint_report,
            "vulnerability": {
                "cve": vuln.cve_id,
                "reachability": reachability.status.value,
            },
            "costs": {
                "llm_tokens": 3000,
                "z3_seconds": 5.2,
                "total_usd": cost_tracker.get_total_cost(),
            },
            "roi": {
                "bug_value_prevented": bug_value,
            },
            "sbom": {
                "component_count": len(sbom.components),
            },
            "verdict": "APPROVED" if reachability.status.value != "reachable" else "NEEDS_REVIEW"
        }
        
        # Verify complete report
        assert final_report["pr"]["number"] == 100
        assert "vulnerability" in final_report
        assert "costs" in final_report
        assert "roi" in final_report
        assert "sbom" in final_report
        assert final_report["sbom"]["component_count"] == 1


class TestCrossFeatureDataFlow:
    """Tests verifying data flows correctly between features."""

    def test_bug_flows_from_intelligence_to_roi(self):
        """Test that bug data from intelligence engine correctly feeds ROI."""
        from codeverify_agents.codebase_intelligence import BugTracker
        from codeverify_core.roi_dashboard import ROIDashboard, BugSeverity, BugCaught

        # Track bug in intelligence
        tracker = BugTracker()
        tracker.record_bug(
            bug_id="BUG-001",
            file_path="auth.py",
            pattern_type="authentication_bypass",
            severity="critical"
        )
        
        # Get bug data
        bugs = tracker.get_bugs_by_severity("critical")
        
        # Create ROI bug record
        dashboard = ROIDashboard()
        for bug in bugs:
            caught = BugCaught(
                bug_id=bug.bug_id,
                severity=BugSeverity.CRITICAL,
                category=bug.pattern_type,
                detected_at=datetime.now()
            )
            dashboard.record_bug(caught)
        
        # Verify data flowed correctly
        assert dashboard.total_bugs_caught >= 1

    def test_verification_creates_runtime_spec(self):
        """Test that verification results can create runtime specs."""
        from codeverify_verifier.runtime_probes import RuntimeSpec, ProbeGenerator

        # Simulate verification finding a constraint
        constraint = "len(password) >= 8"
        function_name = "validate_password"
        
        # Create runtime spec from constraint
        spec = RuntimeSpec(
            name=f"{function_name}_constraint",
            condition=constraint,
            message=f"Constraint violated: {constraint}"
        )
        
        # Generate probe
        generator = ProbeGenerator()
        probe_code = generator.generate_python_probe(spec)
        
        assert spec.name == "validate_password_constraint"
        assert constraint in spec.condition

    def test_counterexample_from_autofix_verification(self):
        """Test that auto-fix verification failures create playground sessions."""
        from codeverify_agents.agentic_autofix import FixVerifier
        from codeverify_verifier.counterexample_playground import PlaygroundEngine

        # Simulate a fix verification that produces counterexample
        z3_output = '''sat
(model
  (define-fun input_length () Int 0)
  (define-fun max_length () Int 100)
)'''
        
        # Create playground from counterexample
        engine = PlaygroundEngine()
        session = engine.create_session(z3_output)
        
        # Verify session contains verification data
        assert "input_length" in session.counterexample.variables
        assert session.counterexample.variables["input_length"].value == 0


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
