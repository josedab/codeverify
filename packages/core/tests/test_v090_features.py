"""Tests for v0.9.0 features.

Covers all 10 next-gen features:
1. Verification-First Code Completion
2. Multi-Repo Verification Graph
3. Verification-Driven Test Generation
4. Adversarial Testing Copilot
5. Live Verification During Code Review
6. AI Code Audit Trail & Provenance
7. Explainable AI Verification Reports
8. Semantic Code Clone Detector
9. Verification-as-Code Infrastructure
10. Verification Budget Marketplace
"""

import time

import pytest


# ─── Feature 1: Verification-First Code Completion ─────────────────


class TestCompletionVerifier:
    def test_verify_safe_completion(self):
        from codeverify_core.verification_completion import (
            CompletionCandidate, CompletionVerifier, VerificationStatus,
        )
        verifier = CompletionVerifier()
        candidate = CompletionCandidate(
            text="result = x + y\nreturn result",
            language="python",
        )
        result = verifier.verify_candidate(candidate)
        assert result.status == VerificationStatus.VERIFIED
        assert result.checks_passed > 0

    def test_verify_unsafe_completion(self):
        from codeverify_core.verification_completion import (
            CompletionCandidate, CompletionVerifier, VerificationStatus,
        )
        verifier = CompletionVerifier()
        candidate = CompletionCandidate(
            text="eval(user_input)",
            language="python",
        )
        result = verifier.verify_candidate(candidate)
        assert result.status in (VerificationStatus.PARTIAL, VerificationStatus.FAILED)
        assert len(result.issues) > 0

    def test_verify_and_rank(self):
        from codeverify_core.verification_completion import (
            CompletionCandidate, CompletionVerifier,
        )
        verifier = CompletionVerifier()
        candidates = [
            CompletionCandidate(text="eval(x)", language="python"),
            CompletionCandidate(text="result = x + 1\nreturn result", language="python"),
            CompletionCandidate(text="None.method()", language="python"),
        ]
        results = verifier.verify_and_rank(candidates)
        assert len(results) == 3
        # Verified ones should be ranked higher
        assert results[0].score >= results[-1].score

    def test_cache_hit(self):
        from codeverify_core.verification_completion import (
            CompletionCandidate, CompletionVerifier,
        )
        verifier = CompletionVerifier()
        candidate = CompletionCandidate(text="x = 42", language="python")
        r1 = verifier.verify_candidate(candidate)
        r2 = verifier.verify_candidate(candidate)
        assert r2.cached is True
        assert verifier.stats.cache_hits >= 1

    def test_completion_middleware(self):
        from codeverify_core.verification_completion import (
            CompletionCandidate, CompletionMiddleware,
        )
        middleware = CompletionMiddleware()
        candidates = [
            CompletionCandidate(text="return x + y", language="python"),
        ]
        results = middleware.process_completions(candidates)
        assert len(results) == 1
        assert middleware.intercepted_count == 1

    def test_middleware_disabled(self):
        from codeverify_core.verification_completion import (
            CompletionCandidate, CompletionMiddleware, VerificationStatus,
        )
        middleware = CompletionMiddleware(enabled=False)
        candidates = [CompletionCandidate(text="eval(x)", language="python")]
        results = middleware.process_completions(candidates)
        assert results[0].status == VerificationStatus.UNVERIFIED

    def test_format_label(self):
        from codeverify_core.verification_completion import (
            CompletionCandidate, CompletionMiddleware,
        )
        middleware = CompletionMiddleware()
        candidates = [CompletionCandidate(text="result = x + 1", language="python")]
        results = middleware.process_completions(candidates)
        label = middleware.format_completion_label(results[0])
        assert "✅" in label or "⚠️" in label or "❓" in label

    def test_rust_rules(self):
        from codeverify_core.verification_completion import (
            CompletionCandidate, CompletionVerifier,
        )
        verifier = CompletionVerifier()
        candidate = CompletionCandidate(
            text="let val = result.unwrap();",
            language="rust",
        )
        result = verifier.verify_candidate(candidate)
        assert len(result.issues) > 0

    def test_singleton(self):
        from codeverify_core.verification_completion import (
            get_completion_verifier, reset_completion_verifier,
        )
        v1 = get_completion_verifier()
        v2 = get_completion_verifier()
        assert v1 is v2
        reset_completion_verifier()
        v3 = get_completion_verifier()
        assert v3 is not v1


# ─── Feature 2: Multi-Repo Verification Graph ──────────────────────


class TestMultiRepoGraph:
    def test_add_services(self):
        from codeverify_core.multi_repo_graph import (
            ServiceContract, VerificationGraph,
        )
        graph = VerificationGraph()
        graph.add_service(ServiceContract(service_name="auth"))
        graph.add_service(ServiceContract(service_name="billing"))
        assert len(graph.services) == 2

    def test_detect_breaking_change(self):
        from codeverify_core.multi_repo_graph import (
            ContractCompatibilityChecker, ContractField, Endpoint,
            FieldType, ServiceContract,
        )
        checker = ContractCompatibilityChecker()
        consumer = ServiceContract(
            service_name="frontend",
            endpoints=[Endpoint(
                path="/api/user", method="GET",
                response_fields=[ContractField(name="email", required=True)],
            )],
        )
        provider = ServiceContract(
            service_name="backend",
            endpoints=[Endpoint(
                path="/api/user", method="GET",
                response_fields=[],  # Email field missing!
            )],
        )
        breaks = checker.check_compatibility(consumer, provider)
        assert len(breaks) > 0
        assert breaks[0].field_name == "email"

    def test_no_breaking_change(self):
        from codeverify_core.multi_repo_graph import (
            ContractCompatibilityChecker, ContractField, Endpoint,
            ServiceContract,
        )
        checker = ContractCompatibilityChecker()
        consumer = ServiceContract(
            service_name="frontend",
            endpoints=[Endpoint(
                path="/api/user", method="GET",
                response_fields=[ContractField(name="id", required=True)],
            )],
        )
        provider = ServiceContract(
            service_name="backend",
            endpoints=[Endpoint(
                path="/api/user", method="GET",
                response_fields=[ContractField(name="id", required=True)],
            )],
        )
        breaks = checker.check_compatibility(consumer, provider)
        assert len(breaks) == 0

    def test_blast_radius(self):
        from codeverify_core.multi_repo_graph import (
            ServiceContract, ServiceDependency, VerificationGraph,
        )
        graph = VerificationGraph()
        graph.add_service(ServiceContract(service_name="db"))
        graph.add_service(ServiceContract(service_name="api"))
        graph.add_service(ServiceContract(service_name="web"))
        graph.add_dependency(ServiceDependency(consumer="api", provider="db"))
        graph.add_dependency(ServiceDependency(consumer="web", provider="api"))
        radius = graph.blast_radius("db")
        assert "api" in radius.directly_affected
        assert "web" in radius.transitively_affected

    def test_mermaid_export(self):
        from codeverify_core.multi_repo_graph import (
            ServiceContract, ServiceDependency, VerificationGraph,
        )
        graph = VerificationGraph()
        graph.add_service(ServiceContract(service_name="a"))
        graph.add_service(ServiceContract(service_name="b"))
        graph.add_dependency(ServiceDependency(consumer="a", provider="b"))
        mermaid = graph.to_mermaid()
        assert "graph LR" in mermaid
        assert "a" in mermaid and "b" in mermaid

    def test_contract_extractor_openapi(self):
        from codeverify_core.multi_repo_graph import ContractExtractor
        extractor = ContractExtractor()
        spec = {
            "info": {"version": "2.0.0"},
            "paths": {
                "/users": {
                    "get": {
                        "summary": "List users",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "properties": {
                                                "id": {"type": "integer"},
                                                "name": {"type": "string"},
                                            },
                                            "required": ["id"],
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        contract = extractor.extract_openapi(spec, "user-service")
        assert contract.version == "2.0.0"
        assert len(contract.endpoints) == 1
        assert len(contract.endpoints[0].response_fields) == 2

    def test_version_breaking_change(self):
        from codeverify_core.multi_repo_graph import (
            ContractCompatibilityChecker, ContractField, Endpoint,
            ServiceContract,
        )
        checker = ContractCompatibilityChecker()
        v1 = ServiceContract(
            service_name="api", version="1.0",
            endpoints=[Endpoint(
                path="/data", method="GET",
                response_fields=[ContractField(name="value", required=True)],
            )],
        )
        v2 = ServiceContract(
            service_name="api", version="2.0",
            endpoints=[Endpoint(
                path="/data", method="GET",
                response_fields=[],  # Field removed!
            )],
        )
        breaks = checker.check_version_compatibility(v1, v2)
        assert any(b.field_name == "value" for b in breaks)

    def test_singleton(self):
        from codeverify_core.multi_repo_graph import (
            get_verification_graph, reset_verification_graph,
        )
        g1 = get_verification_graph()
        g2 = get_verification_graph()
        assert g1 is g2
        reset_verification_graph()


# ─── Feature 3: Verification-Driven Test Generation ─────────────────


class TestTestGeneration:
    SAMPLE_CODE = '''
def calculate_discount(price, quantity):
    if price <= 0:
        raise ValueError("Price must be positive")
    if quantity > 100:
        return price * 0.8
    if quantity > 10:
        return price * 0.9
    return price
'''

    def test_path_discovery(self):
        from codeverify_core.test_generation import SymbolicPathDiscoverer
        discoverer = SymbolicPathDiscoverer()
        paths = discoverer.discover_paths(self.SAMPLE_CODE, "calculate_discount")
        assert len(paths) >= 2  # At least true/false branches

    def test_test_generation(self):
        from codeverify_core.test_generation import TestSuiteGenerator
        gen = TestSuiteGenerator()
        tests = gen.generate_tests(self.SAMPLE_CODE, "calculate_discount")
        assert len(tests) >= 1
        for test in tests:
            assert test.name.startswith("test_calculate_discount")

    def test_pytest_output(self):
        from codeverify_core.test_generation import TestSuiteGenerator
        gen = TestSuiteGenerator()
        suite = gen.generate_suite(self.SAMPLE_CODE, ["calculate_discount"])
        assert "def test_" in suite
        assert "import pytest" in suite

    def test_jest_output(self):
        from codeverify_core.test_generation import TestSuiteGenerator, TestFramework
        gen = TestSuiteGenerator(framework=TestFramework.JEST)
        tests = gen.generate_tests(self.SAMPLE_CODE, "calculate_discount")
        if tests:
            jest = tests[0].to_jest()
            assert "test(" in jest

    def test_mutation_testing(self):
        from codeverify_core.test_generation import MutationTester
        tester = MutationTester()
        mutants = tester.generate_mutants(self.SAMPLE_CODE)
        assert len(mutants) > 0
        assert any("arithmetic" in m.mutation_type or "comparison" in m.mutation_type for m in mutants)

    def test_mutation_report(self):
        from codeverify_core.test_generation import TestSuiteGenerator
        gen = TestSuiteGenerator()
        tests = gen.generate_tests(self.SAMPLE_CODE, "calculate_discount")
        report = gen.mutation_test(self.SAMPLE_CODE, tests)
        assert report.total_mutants > 0
        assert 0 <= report.kill_rate <= 1.0
        assert 0 <= report.quality_score <= 10.0

    def test_path_test_name(self):
        from codeverify_core.test_generation import CodePath, PathCondition
        path = CodePath(
            function_name="foo",
            conditions=[PathCondition(condition="x > 0")],
        )
        assert "test_foo" in path.test_name

    def test_singleton(self):
        from codeverify_core.test_generation import get_test_generator, reset_test_generator
        g1 = get_test_generator()
        g2 = get_test_generator()
        assert g1 is g2
        reset_test_generator()


# ─── Feature 4: Adversarial Testing Copilot ─────────────────────────


class TestAdversarialTesting:
    def test_vulnerability_scanner(self):
        from codeverify_core.adversarial_testing import VulnerabilityScanner
        scanner = VulnerabilityScanner()
        source = 'result = eval(user_input)\nos.system(cmd)'
        exploits = scanner.scan(source, "test.py")
        assert len(exploits) >= 2
        assert any("eval" in e.title.lower() for e in exploits)

    def test_attack_vector_generation(self):
        from codeverify_core.adversarial_testing import AttackVectorGenerator
        gen = AttackVectorGenerator()
        vectors = gen.generate_vectors("process", ["user_input"])
        assert len(vectors) > 0
        categories = {v.category for v in vectors}
        assert len(categories) >= 3

    def test_adversarial_tester(self):
        from codeverify_core.adversarial_testing import AdversarialTester
        tester = AdversarialTester()
        source = 'def process(x):\n    return eval(x)'
        report = tester.test_function(source, "process", "test.py")
        assert report.total_exploits > 0
        assert report.risk_score > 0

    def test_exploit_cvss(self):
        from codeverify_core.adversarial_testing import Exploit, ExploitSeverity
        exploit = Exploit(severity=ExploitSeverity.CRITICAL)
        assert exploit.cvss_score == 0.0  # default; scanner sets this
        assert "AV:N" in exploit.cvss_vector

    def test_remediation_report(self):
        from codeverify_core.adversarial_testing import AdversarialTester
        tester = AdversarialTester()
        source = 'password = "secret123"'
        report = tester.test_file(source, "config.py")
        md = tester.generate_remediation_report(report)
        assert "Adversarial Testing Report" in md

    def test_file_scan(self):
        from codeverify_core.adversarial_testing import AdversarialTester
        tester = AdversarialTester()
        report = tester.test_file("x = 1 + 2", "safe.py")
        assert report.vectors_tested > 0

    def test_report_dict(self):
        from codeverify_core.adversarial_testing import AdversarialReport
        report = AdversarialReport(target_file="test.py")
        d = report.to_dict()
        assert d["target_file"] == "test.py"
        assert d["risk_score"] == 0.0

    def test_singleton(self):
        from codeverify_core.adversarial_testing import (
            get_adversarial_tester, reset_adversarial_tester,
        )
        t1 = get_adversarial_tester()
        t2 = get_adversarial_tester()
        assert t1 is t2
        reset_adversarial_tester()


# ─── Feature 5: Live Verification During Code Review ────────────────


class TestLiveReview:
    def test_nl_assertion_compile(self):
        from codeverify_core.live_review import (
            NLAssertion, NLAssertionCompiler, AssertionStatus,
        )
        compiler = NLAssertionCompiler()
        assertion = NLAssertion(text="x should never be null")
        result = compiler.compile(assertion)
        assert result.status == AssertionStatus.VERIFIED
        assert "None" in result.constraint

    def test_nl_positive_assertion(self):
        from codeverify_core.live_review import NLAssertion, NLAssertionCompiler
        compiler = NLAssertionCompiler()
        result = compiler.compile(NLAssertion(text="count must be positive"))
        assert "> 0" in result.constraint

    def test_nl_comparison_assertion(self):
        from codeverify_core.live_review import NLAssertion, NLAssertionCompiler
        compiler = NLAssertionCompiler()
        result = compiler.compile(NLAssertion(text="start should be less than end"))
        assert "<" in result.constraint

    def test_session_add_assertion(self):
        from codeverify_core.live_review import LiveReviewSession
        session = LiveReviewSession(pr_id="PR-42")
        compiled = session.add_assertion("x must be positive", author="alice")
        assert session.assertion_count == 1
        assert compiled.constraint

    def test_voting(self):
        from codeverify_core.live_review import LiveReviewSession, VoteType
        session = LiveReviewSession(pr_id="PR-42")
        consensus = session.vote("finding-1", "alice", VoteType.AGREE)
        consensus = session.vote("finding-1", "bob", VoteType.AGREE)
        consensus = session.vote("finding-1", "carol", VoteType.FALSE_POSITIVE)
        assert consensus.agree_count == 2
        assert consensus.consensus == VoteType.AGREE
        assert consensus.confidence > 0.5

    def test_vote_replacement(self):
        from codeverify_core.live_review import LiveReviewSession, VoteType
        session = LiveReviewSession(pr_id="PR-42")
        session.vote("f-1", "alice", VoteType.AGREE)
        session.vote("f-1", "alice", VoteType.DISAGREE)  # Changed vote
        consensus = session.get_consensus("f-1")
        assert consensus.disagree_count == 1
        assert consensus.agree_count == 0

    def test_session_manager(self):
        from codeverify_core.live_review import LiveReviewManager
        manager = LiveReviewManager(max_sessions=3)
        s1 = manager.create_session("PR-1")
        s2 = manager.create_session("PR-2")
        assert manager.active_sessions == 2
        found = manager.get_session_by_pr("PR-1")
        assert found is s1

    def test_session_history(self):
        from codeverify_core.live_review import LiveReviewSession
        session = LiveReviewSession(pr_id="PR-1")
        session.add_assertion("x must be positive")
        history = session.get_history()
        assert len(history) == 1
        assert history[0]["action"] == "assertion_added"

    def test_singleton(self):
        from codeverify_core.live_review import (
            get_live_review_manager, reset_live_review_manager,
        )
        m1 = get_live_review_manager()
        m2 = get_live_review_manager()
        assert m1 is m2
        reset_live_review_manager()


# ─── Feature 6: AI Code Audit Trail & Provenance ────────────────────


class TestAuditTrail:
    def test_ai_detection_human_code(self):
        from codeverify_core.audit_trail import AICodeDetector
        detector = AICodeDetector()
        # Short simple code
        fp = detector.analyze("x = 1")
        assert fp.ai_probability < 0.8

    def test_ai_detection_ai_code(self):
        from codeverify_core.audit_trail import AICodeDetector
        detector = AICodeDetector()
        # AI-typical code: docstrings, consistent naming, good comments
        ai_code = '''
"""Module for processing data."""

import json
from typing import Any


def process_data(data: Any) -> dict:
    """Process the input data and return results.

    Args:
        data: The input data to process.

    Returns:
        A dictionary containing processed results.
    """
    result = {}
    value = data.get("value", None)
    output = json.dumps(result)
    temp = len(output)
    item = {"processed": True}
    response = {"status": "ok", "data": item}
    return response
'''
        fp = detector.analyze(ai_code)
        assert fp.ai_probability > 0.0
        assert len(fp.signals) >= 1

    def test_provenance_tracking(self):
        from codeverify_core.audit_trail import ProvenanceTracker, ProvenanceSource
        tracker = ProvenanceTracker()
        record = tracker.track(
            source="x = 1\ny = 2",
            file_path="test.py",
            author="alice",
        )
        assert record.source == ProvenanceSource.HUMAN
        assert record.content_hash

    def test_provenance_ai_model(self):
        from codeverify_core.audit_trail import ProvenanceTracker, ProvenanceSource
        tracker = ProvenanceTracker()
        record = tracker.track(
            source="def process(): pass",
            file_path="test.py",
            ai_model="gpt-4",
        )
        assert record.source == ProvenanceSource.AI_CHATGPT

    def test_hash_chain_integrity(self):
        from codeverify_core.audit_trail import ProvenanceTracker
        tracker = ProvenanceTracker()
        tracker.track("code1", "a.py")
        tracker.track("code2", "b.py")
        tracker.track("code3", "c.py")
        assert tracker.verify_chain() is True

    def test_compliance_report(self):
        from codeverify_core.audit_trail import (
            ProvenanceTracker, ComplianceFramework,
        )
        tracker = ProvenanceTracker()
        tracker.track("x = 1", "a.py", ai_model="copilot")
        tracker.track("y = 2", "b.py")
        report = tracker.generate_report(ComplianceFramework.EU_AI_ACT)
        assert report.total_files == 2
        assert report.ai_percentage > 0
        assert report.risk_assessment.get("status") is not None

    def test_aibom_export(self):
        from codeverify_core.audit_trail import (
            ProvenanceTracker, ComplianceFramework,
        )
        tracker = ProvenanceTracker()
        tracker.track("ai_code", "gen.py", ai_model="claude")
        report = tracker.generate_report(ComplianceFramework.SOX)
        aibom = report.to_aibom()
        assert aibom["aibomVersion"] == "1.0"
        assert len(aibom["components"]) > 0

    def test_singleton(self):
        from codeverify_core.audit_trail import (
            get_provenance_tracker, reset_provenance_tracker,
        )
        t1 = get_provenance_tracker()
        t2 = get_provenance_tracker()
        assert t1 is t2
        reset_provenance_tracker()


# ─── Feature 7: Explainable AI Verification Reports ─────────────────


class TestExplainableReports:
    def test_explain_null_safety(self):
        from codeverify_core.explainable_reports import (
            FindingExplainer, VerificationFinding, VerificationOutcome,
        )
        explainer = FindingExplainer()
        finding = VerificationFinding(
            check_type="null_safety",
            outcome=VerificationOutcome.UNSAFE,
            function_name="process",
            line=42,
            severity="high",
        )
        explained = explainer.explain(finding)
        assert "Null Safety" in explained.title
        assert explained.explanation
        assert explained.fix_suggestion

    def test_explain_division_by_zero(self):
        from codeverify_core.explainable_reports import (
            FindingExplainer, VerificationFinding, ExplanationLevel,
        )
        explainer = FindingExplainer()
        finding = VerificationFinding(check_type="division_by_zero", severity="medium")
        explained = explainer.explain(finding, ExplanationLevel.BEGINNER)
        assert "zero" in explained.explanation.lower()
        assert explained.analogy

    def test_explain_with_counterexample(self):
        from codeverify_core.explainable_reports import (
            FindingExplainer, VerificationFinding, CounterExample,
        )
        explainer = FindingExplainer()
        finding = VerificationFinding(
            check_type="array_bounds",
            severity="high",
            counterexample=CounterExample(
                variables={"index": 10, "array_len": 5},
                execution_path=["Set index=10", "Check bounds", "Access array[10]"],
            ),
        )
        explained = explainer.explain(finding)
        assert "Array" in explained.title
        assert "graph TD" in explained.visual_diagram

    def test_markdown_output(self):
        from codeverify_core.explainable_reports import (
            ExplainableReport, VerificationFinding,
        )
        report = ExplainableReport()
        report.add_finding(VerificationFinding(
            check_type="null_safety", severity="high",
        ))
        md = report.to_markdown()
        assert "# Verification Report" in md

    def test_batch_explain(self):
        from codeverify_core.explainable_reports import (
            FindingExplainer, VerificationFinding,
        )
        explainer = FindingExplainer()
        findings = [
            VerificationFinding(check_type="null_safety", severity="high"),
            VerificationFinding(check_type="division_by_zero", severity="medium"),
        ]
        explained = explainer.explain_batch(findings)
        assert len(explained) == 2

    def test_expert_level(self):
        from codeverify_core.explainable_reports import (
            FindingExplainer, VerificationFinding, ExplanationLevel,
        )
        explainer = FindingExplainer()
        finding = VerificationFinding(check_type="integer_overflow", severity="high")
        explained = explainer.explain(finding, ExplanationLevel.EXPERT)
        assert "Z3" in explained.explanation

    def test_similar_cves(self):
        from codeverify_core.explainable_reports import (
            FindingExplainer, VerificationFinding,
        )
        explainer = FindingExplainer()
        finding = VerificationFinding(check_type="array_bounds", severity="high")
        explained = explainer.explain(finding)
        assert len(explained.similar_cves) > 0

    def test_singleton(self):
        from codeverify_core.explainable_reports import (
            get_finding_explainer, reset_finding_explainer,
        )
        e1 = get_finding_explainer()
        e2 = get_finding_explainer()
        assert e1 is e2
        reset_finding_explainer()


# ─── Feature 8: Semantic Code Clone Detector ─────────────────────────


class TestCloneDetector:
    CLONE_SOURCE = '''
def add_numbers(a, b):
    result = a + b
    return result

def sum_values(x, y):
    total = x + y
    return total

def multiply(a, b):
    return a * b
'''

    def test_function_extraction(self):
        from codeverify_core.clone_detector import FunctionExtractor
        extractor = FunctionExtractor()
        funcs = extractor.extract(self.CLONE_SOURCE, "test.py")
        assert len(funcs) == 3
        names = {f.name for f in funcs}
        assert "add_numbers" in names
        assert "sum_values" in names

    def test_detect_clones(self):
        from codeverify_core.clone_detector import CloneDetector
        detector = CloneDetector()
        report = detector.detect_clones({"test.py": self.CLONE_SOURCE})
        assert report.total_functions >= 2
        # add_numbers and sum_values should be detected as clones
        assert report.total_clones >= 1

    def test_clone_cluster(self):
        from codeverify_core.clone_detector import CloneDetector
        detector = CloneDetector()
        report = detector.detect_clones({"test.py": self.CLONE_SOURCE})
        if report.clusters:
            cluster = report.clusters[0]
            assert cluster.size >= 2
            assert cluster.dedup_savings > 0

    def test_refactoring_suggestion(self):
        from codeverify_core.clone_detector import CloneDetector
        detector = CloneDetector()
        report = detector.detect_clones({"test.py": self.CLONE_SOURCE})
        if report.suggestions:
            suggestion = report.suggestions[0]
            assert suggestion.new_function_name.startswith("shared_")
            assert suggestion.loc_saved > 0

    def test_structural_hash(self):
        from codeverify_core.clone_detector import FunctionExtractor
        extractor = FunctionExtractor()
        funcs = extractor.extract(self.CLONE_SOURCE, "test.py")
        add_func = next(f for f in funcs if f.name == "add_numbers")
        sum_func = next(f for f in funcs if f.name == "sum_values")
        # Same structure, different names → same structural hash
        assert add_func.structural_hash == sum_func.structural_hash

    def test_report_dict(self):
        from codeverify_core.clone_detector import CloneDetector
        detector = CloneDetector()
        report = detector.detect_clones({"test.py": self.CLONE_SOURCE})
        d = report.to_dict()
        assert "total_functions" in d
        assert "duplication_percentage" in d

    def test_multi_file(self):
        from codeverify_core.clone_detector import CloneDetector
        detector = CloneDetector()
        files = {
            "a.py": "def foo(x):\n    result = x + 1\n    return result\n",
            "b.py": "def bar(y):\n    result = y + 1\n    return result\n",
        }
        report = detector.detect_clones(files)
        assert report.total_functions == 2

    def test_singleton(self):
        from codeverify_core.clone_detector import (
            get_clone_detector, reset_clone_detector,
        )
        d1 = get_clone_detector()
        d2 = get_clone_detector()
        assert d1 is d2
        reset_clone_detector()


# ─── Feature 9: Verification-as-Code Infrastructure ─────────────────


class TestVerificationAsCode:
    def test_load_soc2_template(self):
        from codeverify_core.verification_as_code import PolicyEngine
        engine = PolicyEngine()
        module = engine.load_template("soc2")
        assert module is not None
        assert module.name == "soc2"
        assert len(module.rules) >= 3

    def test_load_hipaa_template(self):
        from codeverify_core.verification_as_code import PolicyEngine
        engine = PolicyEngine()
        module = engine.load_template("hipaa")
        assert module is not None
        assert any("phi" in r.id for r in module.rules)

    def test_dsl_parsing(self):
        from codeverify_core.verification_as_code import PolicyDSLParser
        parser = PolicyDSLParser()
        dsl = '''
module "custom" {
  version = "2.0.0"
  scope = "team"

  rule "no-eval" {
    check = "input_validation"
    severity = "block"
    message = "No eval allowed"
    tags = ["security"]
  }

  variable "max_complexity" {
    default = 15
  }
}
'''
        modules = parser.parse(dsl)
        assert len(modules) == 1
        assert modules[0].name == "custom"
        assert modules[0].version == "2.0.0"
        assert len(modules[0].rules) == 1
        assert modules[0].variables.get("max_complexity") == 15

    def test_policy_evaluation(self):
        from codeverify_core.verification_as_code import PolicyEngine
        engine = PolicyEngine()
        engine.load_template("soc2")
        source = 'password = "hunter2"\nresult = md5(data)'
        violations = engine.evaluate(source, "config.py")
        assert len(violations) > 0

    def test_drift_detection_in_sync(self):
        from codeverify_core.verification_as_code import (
            PolicyEngine, PolicyModule, PolicyRule, PolicySeverity, DriftStatus,
        )
        engine = PolicyEngine()
        central = PolicyModule(name="test", rules=[
            PolicyRule(id="r1", check="null_safety", severity=PolicySeverity.BLOCK),
        ])
        local = PolicyModule(name="test", rules=[
            PolicyRule(id="r1", check="null_safety", severity=PolicySeverity.BLOCK),
        ])
        report = engine.check_drift(central, local)
        assert report.status == DriftStatus.IN_SYNC

    def test_drift_detection_drifted(self):
        from codeverify_core.verification_as_code import (
            PolicyEngine, PolicyModule, PolicyRule, PolicySeverity, DriftStatus,
        )
        engine = PolicyEngine()
        central = PolicyModule(name="test", rules=[
            PolicyRule(id="r1", check="null_safety", severity=PolicySeverity.BLOCK),
        ])
        local = PolicyModule(name="test", rules=[
            PolicyRule(id="r1", check="null_safety", severity=PolicySeverity.WARN),  # Changed!
        ])
        report = engine.check_drift(central, local)
        assert report.status == DriftStatus.DRIFTED
        assert len(report.differences) > 0

    def test_module_fingerprint(self):
        from codeverify_core.verification_as_code import PolicyModule, PolicyRule
        m1 = PolicyModule(name="test", rules=[PolicyRule(id="r1")])
        m2 = PolicyModule(name="test", rules=[PolicyRule(id="r1")])
        assert m1.fingerprint() == m2.fingerprint()

    def test_singleton(self):
        from codeverify_core.verification_as_code import (
            get_policy_engine, reset_policy_engine,
        )
        e1 = get_policy_engine()
        e2 = get_policy_engine()
        assert e1 is e2
        reset_policy_engine()


# ─── Feature 10: Verification Budget Marketplace ────────────────────


class TestBudgetMarketplace:
    def test_wallet_operations(self):
        from codeverify_core.budget_marketplace import (
            VerificationMarketplace, TransactionType,
        )
        market = VerificationMarketplace()
        wallet = market.get_or_create_wallet("team-a")
        market.grant_credits("team-a", 1000)
        assert wallet.balance == 1000
        assert wallet.available == 1000

    def test_credit_usage(self):
        from codeverify_core.budget_marketplace import (
            VerificationMarketplace, VerificationCredit,
        )
        market = VerificationMarketplace()
        market.grant_credits("team-a", 100)
        credit = VerificationCredit(amount=1.0, complexity_factor=2.0)
        success = market.use_credits("team-a", credit)
        assert success is True
        wallet = market.get_or_create_wallet("team-a")
        assert wallet.balance == 98.0

    def test_credit_insufficient(self):
        from codeverify_core.budget_marketplace import (
            VerificationMarketplace, VerificationCredit,
        )
        market = VerificationMarketplace()
        market.grant_credits("team-a", 1)
        credit = VerificationCredit(amount=1.0, complexity_factor=10.0)
        success = market.use_credits("team-a", credit)
        assert success is False

    def test_order_matching(self):
        from codeverify_core.budget_marketplace import (
            VerificationMarketplace, OrderSide, OrderStatus,
        )
        market = VerificationMarketplace()
        market.grant_credits("seller", 100)
        market.grant_credits("buyer", 0)

        sell = market.place_order("seller", OrderSide.SELL, 50, 0.01)
        buy = market.place_order("buyer", OrderSide.BUY, 50, 0.01)

        assert sell.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
        assert buy.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)

        trades = market.get_trades()
        assert len(trades) >= 1
        assert trades[0].credits == 50

    def test_no_match_price_gap(self):
        from codeverify_core.budget_marketplace import (
            VerificationMarketplace, OrderSide, OrderStatus,
        )
        market = VerificationMarketplace()
        market.grant_credits("seller", 100)

        sell = market.place_order("seller", OrderSide.SELL, 50, 0.10)
        buy = market.place_order("buyer", OrderSide.BUY, 50, 0.01)

        assert sell.status == OrderStatus.OPEN
        assert buy.status == OrderStatus.OPEN

    def test_cancel_order(self):
        from codeverify_core.budget_marketplace import (
            VerificationMarketplace, OrderSide,
        )
        market = VerificationMarketplace()
        market.grant_credits("team", 100)
        order = market.place_order("team", OrderSide.SELL, 50, 0.01)
        wallet = market.get_or_create_wallet("team")
        assert wallet.reserved == 50
        market.cancel_order(order.id)
        assert wallet.reserved == 0

    def test_market_stats(self):
        from codeverify_core.budget_marketplace import (
            VerificationMarketplace, OrderSide,
        )
        market = VerificationMarketplace()
        market.grant_credits("s", 100)
        market.place_order("s", OrderSide.SELL, 10, 0.05)
        market.place_order("b", OrderSide.BUY, 10, 0.05)
        stats = market.get_market_stats()
        assert stats.trades_count >= 1
        assert stats.total_volume > 0

    def test_capacity_lease(self):
        from codeverify_core.budget_marketplace import VerificationMarketplace
        market = VerificationMarketplace()
        lease = market.create_lease("provider", "consumer", 100, 0.005)
        assert lease.active is True
        ended = market.end_lease(lease.id)
        assert ended is not None
        assert ended.active is False

    def test_credit_from_verification(self):
        from codeverify_core.budget_marketplace import VerificationCredit
        credit = VerificationCredit.from_verification(constraints=20, loc=100)
        assert credit.complexity_factor > 1.0

    def test_singleton(self):
        from codeverify_core.budget_marketplace import (
            get_marketplace, reset_marketplace,
        )
        m1 = get_marketplace()
        m2 = get_marketplace()
        assert m1 is m2
        reset_marketplace()
