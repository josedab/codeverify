"""Tests for v1.5.0 next-gen features.

Covers all 10 features:
1. Runtime Verification Bridge
2. Verification-Guided Fuzzing
3. Intent-Preserving Refactoring
4. Context-Window Verification
5. Verification Replay & Regression
6. Natural Language Proof Explanation
7. Verification-Aware Review Assignments
8. Multi-Repository Invariant Propagation
9. Proof-Based Documentation Generation
10. Gamified Developer Security Training
"""


class TestRuntimeBridge:
    def test_assertion_translation(self):
        from codeverify_core.runtime_bridge import AssertionLanguage, AssertionTranslator

        t = AssertionTranslator()
        assertions = t.translate(
            "null_safety", "process", "app.py", ["user"], AssertionLanguage.PYTHON
        )
        assert len(assertions) == 1
        assert "is not None" in assertions[0].assertion_code

    def test_typescript_assertions(self):
        from codeverify_core.runtime_bridge import AssertionLanguage, AssertionTranslator

        t = AssertionTranslator()
        assertions = t.translate(
            "division_by_zero", "calc", "app.ts", ["x"], AssertionLanguage.TYPESCRIPT
        )
        assert len(assertions) == 1
        assert "=== 0" in assertions[0].assertion_code

    def test_violation_capture(self):
        from codeverify_core.runtime_bridge import RuntimeVerificationBridgeService

        svc = RuntimeVerificationBridgeService()
        assertions = svc.instrument_function("null_safety", "get_user", "api.py", ["user_id"])
        assert len(assertions) >= 1
        v = svc.report_violation(assertions[0].id, {"user_id": None}, "user_id is None")
        assert v is not None
        assert v.severity.value in ("critical", "high", "medium")

    def test_feedback_loop(self):
        from codeverify_core.runtime_bridge import RuntimeVerificationBridgeService

        svc = RuntimeVerificationBridgeService()
        a = svc.instrument_function("division_by_zero", "calc", "m.py", ["b"])
        svc.report_violation(a[0].id, {"b": 0})
        svc.report_pass(a[0].id)
        stats = svc.get_stats()
        assert stats.confirmed_bugs >= 1
        assert stats.false_positives_detected >= 1

    def test_decorator_generation(self):
        from codeverify_core.runtime_bridge import RuntimeVerificationBridgeService

        svc = RuntimeVerificationBridgeService()
        svc.instrument_function("null_safety", "fn", "f.py", ["x"])
        code = svc.generate_instrumented_code("fn", "f.py")
        assert "codeverify_check" in code
        assert "assert" in code


class TestGuidedFuzzing:
    def test_input_generation(self):
        from codeverify_core.guided_fuzzing import InputGenerator, InputType

        gen = InputGenerator()
        inputs = gen.from_counterexample({"b": 0, "a": 10})
        assert len(inputs) == 2
        zero_input = next(i for i in inputs if i.variable_name == "b")
        assert zero_input.value == 0
        assert zero_input.input_type == InputType.INTEGER

    def test_mutation_generation(self):
        from codeverify_core.guided_fuzzing import InputGenerator

        gen = InputGenerator()
        base = gen.from_counterexample({"x": 5})
        mutations = gen.generate_mutations(base, rounds=3)
        assert len(mutations) == 4  # base + 3 mutations
        values = [m[0].value for m in mutations]
        assert len(set(values)) > 1  # mutations are different

    def test_fuzz_campaign_confirmed(self):
        from codeverify_core.guided_fuzzing import FuzzResult, VerificationGuidedFuzzingService

        svc = VerificationGuidedFuzzingService()
        campaign = svc.fuzz_counterexample("divide", "math.py", "division_by_zero", {"b": 0})
        assert campaign.total_confirmed >= 1
        assert svc.classify_finding(campaign) == FuzzResult.CONFIRMED_BUG

    def test_fuzz_campaign_false_positive(self):
        from codeverify_core.guided_fuzzing import VerificationGuidedFuzzingService

        svc = VerificationGuidedFuzzingService()
        campaign = svc.fuzz_counterexample("divide", "math.py", "division_by_zero", {"b": 5})
        assert campaign.total_false_positives >= 1

    def test_pytest_code_generation(self):
        from codeverify_core.guided_fuzzing import FuzzInput, InputType, TestCodeGenerator

        gen = TestCodeGenerator()
        inputs = [FuzzInput(variable_name="b", value=0, input_type=InputType.INTEGER)]
        code = gen.generate_pytest("divide", inputs, "division_by_zero")
        assert "def test_divide_fuzz" in code
        assert "ZeroDivisionError" in code


class TestIntentRefactoring:
    def test_contract_extraction(self):
        from codeverify_core.intent_refactoring import ContractExtractor

        ex = ContractExtractor()
        code = "def add(a: int, b: int) -> int:\n    assert a >= 0\n    return a + b\n"
        contract = ex.extract("add", code)
        assert contract.parameters == ["a", "b"]
        assert contract.return_type == "int"
        assert len(contract.preconditions) >= 1

    def test_equivalence_check_passes(self):
        from codeverify_core.intent_refactoring import (
            EquivalenceResult,
            IntentPreservingRefactoringService,
        )

        svc = IntentPreservingRefactoringService()
        before = "def add(a: int, b: int) -> int:\n    return a + b\n"
        after = "def add(a: int, b: int) -> int:\n    result = a + b\n    return result\n"
        v = svc.verify_refactoring("add", before, after)
        assert v.is_safe is True
        assert v.equivalence.result == EquivalenceResult.EQUIVALENT

    def test_equivalence_check_fails(self):
        from codeverify_core.intent_refactoring import IntentPreservingRefactoringService

        svc = IntentPreservingRefactoringService()
        before = "def calc(a, b) -> int:\n    return a + b\n"
        after = "def calc(a, b, c) -> str:\n    raise ValueError\n    return str(a + b + c)\n"
        v = svc.verify_refactoring("calc", before, after)
        assert v.is_safe is False
        assert len(v.warnings) >= 1

    def test_safety_score(self):
        from codeverify_core.intent_refactoring import IntentPreservingRefactoringService

        svc = IntentPreservingRefactoringService()
        before = "def f(x): return x\n"
        after = "def f(x): return x\n"
        v = svc.verify_refactoring("f", before, after)
        assert v.safety_score >= 0.5


class TestContextWindowVerify:
    def test_consistency_check(self):
        from codeverify_core.context_window_verify import (
            ContextSnippet,
            ContextWindowVerificationService,
        )

        svc = ContextWindowVerificationService()
        snippets = [
            ContextSnippet(content="def foo(x: int): return x", token_count=20),
            ContextSnippet(content="def bar(y: str): return y", token_count=20),
        ]
        window = svc.verify_context(snippets)
        assert window.quality in ("excellent", "good", "fair", "poor", "unusable")
        assert window.quality_score > 0

    def test_truncation_detection(self):
        from codeverify_core.context_window_verify import (
            ContextSnippet,
            ContextWindowVerificationService,
        )

        svc = ContextWindowVerificationService(model="gpt-4")
        snippets = [ContextSnippet(content="def foo(", token_count=7800)]
        window = svc.verify_context(snippets)
        truncation_issues = [i for i in window.issues if "truncat" in i.message.lower()]
        assert len(truncation_issues) >= 1

    def test_optimization(self):
        from codeverify_core.context_window_verify import (
            ContextSnippet,
            ContextWindowVerificationService,
        )

        svc = ContextWindowVerificationService()
        snippets = [
            ContextSnippet(content="important code", token_count=100, relevance_score=0.9),
            ContextSnippet(content="less important", token_count=100, relevance_score=0.3),
            ContextSnippet(content="critical code", token_count=100, relevance_score=0.95),
        ]
        window, opt = svc.optimize_context(snippets)
        assert opt.selected_snippets <= len(snippets)

    def test_duplicate_detection(self):
        from codeverify_core.context_window_verify import (
            ConsistencyIssueType,
            ContextSnippet,
            ContextWindowVerificationService,
        )

        svc = ContextWindowVerificationService()
        snippets = [
            ContextSnippet(content="def helper(x): return x", token_count=20),
            ContextSnippet(content="def helper(y): return y * 2", token_count=20),
        ]
        window = svc.verify_context(snippets)
        dups = [
            i for i in window.issues if i.issue_type == ConsistencyIssueType.DUPLICATE_DEFINITION
        ]
        assert len(dups) >= 1


class TestVerificationReplay:
    def test_session_recording(self):
        from codeverify_core.verification_replay_regression import VerificationReplayService

        svc = VerificationReplayService()
        session = svc.record_session(
            "repo",
            "abc123",
            [
                {
                    "check_type": "null_safety",
                    "function_name": "get_user",
                    "file_path": "api.py",
                    "result": "pass",
                    "code": "if x is not None: pass",
                },
            ],
            {"api.py": "if x is not None: pass"},
        )
        assert len(session.snapshots) == 1
        assert session.commit_sha == "abc123"

    def test_replay_no_regression(self):
        from codeverify_core.verification_replay_regression import (
            VerificationReplayService,
        )

        svc = VerificationReplayService()
        session = svc.record_session(
            "repo",
            "v1",
            [
                {
                    "check_type": "null_safety",
                    "function_name": "fn",
                    "file_path": "a.py",
                    "result": "pass",
                    "code": "x",
                },
            ],
            {"a.py": "x"},
        )
        report = svc.replay_session(session.id, {"a.py": "x"}, "v2")
        assert report.regressions == 0

    def test_replay_detects_regression(self):
        from codeverify_core.verification_replay_regression import VerificationReplayService

        svc = VerificationReplayService()
        session = svc.record_session(
            "repo",
            "v1",
            [
                {
                    "check_type": "null_safety",
                    "function_name": "fn",
                    "file_path": "a.py",
                    "result": "pass",
                    "code": "if x is not None: x",
                },
            ],
            {"a.py": "if x is not None: x"},
        )
        report = svc.replay_session(session.id, {"a.py": "x = None\nx.foo()"}, "v2")
        assert report.regressions >= 1

    def test_trend_tracking(self):
        from codeverify_core.verification_replay_regression import VerificationReplayService

        svc = VerificationReplayService()
        s = svc.record_session(
            "r",
            "v1",
            [
                {
                    "check_type": "null_safety",
                    "function_name": "f",
                    "file_path": "a.py",
                    "result": "pass",
                    "code": "c",
                }
            ],
            {"a.py": "c"},
        )
        svc.replay_session(s.id, {"a.py": "c"})
        trends = svc.get_trend()
        assert len(trends) >= 1


class TestNLProofExplanation:
    def test_counterexample_explanation(self):
        from codeverify_core.nl_proof_explanation import (
            DetailLevel,
            ExplanationContext,
            NLProofExplanationService,
        )

        svc = NLProofExplanationService()
        ctx = ExplanationContext(
            check_type="division_by_zero",
            function_name="divide",
            file_path="math.py",
            line=10,
            variable_assignments={"b": 0},
            severity="critical",
        )
        exp = svc.explain(ctx, DetailLevel.STANDARD)
        assert "zero" in exp.narrative.lower() or "Zero" in exp.narrative
        assert "divide" in exp.narrative

    def test_proof_success_explanation(self):
        from codeverify_core.nl_proof_explanation import (
            ExplanationContext,
            NLProofExplanationService,
        )

        svc = NLProofExplanationService()
        ctx = ExplanationContext(
            check_type="null_safety",
            function_name="safe_fn",
            file_path="a.py",
        )
        exp = svc.explain(ctx)
        assert "✅" in exp.narrative

    def test_pr_comment_generation(self):
        from codeverify_core.nl_proof_explanation import (
            ExplanationContext,
            NLProofExplanationService,
        )

        svc = NLProofExplanationService()
        contexts = [
            ExplanationContext(
                check_type="null_safety",
                function_name="f1",
                file_path="a.py",
                variable_assignments={"x": None},
            ),
            ExplanationContext(check_type="division_by_zero", function_name="f2", file_path="b.py"),
        ]
        comment = svc.generate_pr_comment(contexts)
        assert "CodeVerify Analysis" in comment
        assert "Issue" in comment or "Passed" in comment

    def test_detail_levels(self):
        from codeverify_core.nl_proof_explanation import (
            DetailLevel,
            ExplanationContext,
            NLProofExplanationService,
        )

        svc = NLProofExplanationService()
        ctx = ExplanationContext(
            check_type="null_safety",
            function_name="fn",
            file_path="a.py",
            line=5,
            variable_assignments={"x": None},
        )
        brief = svc.explain(ctx, DetailLevel.BRIEF)
        detailed = svc.explain(ctx, DetailLevel.DETAILED)
        assert len(detailed.narrative) > len(brief.narrative)


class TestReviewAssignments:
    def test_risk_classification(self):
        from codeverify_core.review_assignments import PRRiskLevel, RiskClassifier

        clf = RiskClassifier()
        profile = clf.classify(critical=2, has_security=True)
        assert profile.risk_level == PRRiskLevel.CRITICAL

    def test_reviewer_matching(self):
        from codeverify_core.review_assignments import (
            ExpertiseArea,
            ReviewAssignmentService,
            Reviewer,
        )

        svc = ReviewAssignmentService()
        svc.register_reviewer(
            Reviewer(id="r1", name="Alice", expertise=[ExpertiseArea.SECURITY], seniority="senior")
        )
        svc.register_reviewer(
            Reviewer(id="r2", name="Bob", expertise=[ExpertiseArea.GENERAL], seniority="mid")
        )
        assignments = svc.assign_reviewer("PR-1", critical=1, has_security=True)
        assert len(assignments) >= 1
        assert assignments[0].reviewer_name == "Alice"

    def test_load_balancing(self):
        from codeverify_core.review_assignments import (
            ExpertiseArea,
            ReviewAssignmentService,
            Reviewer,
        )

        svc = ReviewAssignmentService()
        svc.register_reviewer(
            Reviewer(
                id="r1", name="A", expertise=[ExpertiseArea.GENERAL], current_load=4, max_load=5
            )
        )
        svc.register_reviewer(
            Reviewer(
                id="r2", name="B", expertise=[ExpertiseArea.GENERAL], current_load=0, max_load=5
            )
        )
        assignments = svc.assign_reviewer("PR-2", medium=3, max_reviewers=1)
        assert assignments[0].reviewer_name == "B"  # lower load

    def test_stats(self):
        from codeverify_core.review_assignments import (
            ExpertiseArea,
            ReviewAssignmentService,
            Reviewer,
        )

        svc = ReviewAssignmentService()
        svc.register_reviewer(Reviewer(id="r1", name="A", expertise=[ExpertiseArea.GENERAL]))
        svc.assign_reviewer("PR-1", medium=1)
        stats = svc.get_stats()
        assert stats.total_assignments >= 1


class TestInvariantPropagation:
    def test_register_and_propagate(self):
        from codeverify_core.invariant_propagation import (
            InvariantPropagationService,
        )

        svc = InvariantPropagationService()
        inv = svc.register_invariant(
            name="null_check_all_inputs",
            description="All inputs must be null-checked",
            z3_assertion="(assert (not (= x null)))",
            source_repo="api",
            target_repos=["api", "web"],
            check_type="null_safety",
        )
        report = svc.propagate(
            inv.id,
            {
                "api": {"handler.py": "if x is not None:\n    process(x)"},
                "web": {"view.py": "x = None\nresult = x.method()"},  # uses None without guard
            },
        )
        assert report.compliant_repos >= 1
        assert report.violated_repos >= 1

    def test_org_scope_propagation(self):
        from codeverify_core.invariant_propagation import (
            InvariantPropagationService,
            InvariantScope,
        )

        svc = InvariantPropagationService()
        svc.register_invariant(
            name="encrypt_passwords",
            description="Passwords must be encrypted",
            z3_assertion="",
            source_repo="auth",
            scope=InvariantScope.ORG,
            check_type="encryption",
        )
        reports = svc.propagate_all(
            {
                "auth": {"auth.py": "hashed = bcrypt.hash(password)"},
                "admin": {"admin.py": "password = request.form['password']"},
            }
        )
        assert len(reports) >= 1
        assert any(r.violated_repos > 0 for r in reports)

    def test_governance_summary(self):
        from codeverify_core.invariant_propagation import (
            InvariantPropagationService,
        )

        svc = InvariantPropagationService()
        svc.register_invariant(
            name="test",
            description="test",
            z3_assertion="",
            source_repo="r",
            target_repos=["r"],
            check_type="null_safety",
        )
        svc.propagate_all({"r": {"f.py": "if x is not None: pass"}})
        summary = svc.get_governance_summary()
        assert summary.total_invariants >= 1


class TestProofDocs:
    def test_doc_generation(self):
        from codeverify_core.proof_docs import DocFormat, ProofBasedDocService

        svc = ProofBasedDocService()
        doc = svc.generate(
            "API Reference",
            [
                (
                    "get_user",
                    "def get_user(user_id: int) -> dict:\n    if user_id is not None:\n        return {}\n",
                ),
            ],
            fmt=DocFormat.MARKDOWN,
        )
        assert "get_user" in doc.content
        assert "Verified Properties" in doc.content

    def test_html_format(self):
        from codeverify_core.proof_docs import DocFormat, ProofBasedDocService

        svc = ProofBasedDocService()
        doc = svc.generate("API", [("fn", "def fn(x): return x\n")], fmt=DocFormat.HTML)
        assert "<html>" in doc.content

    def test_freshness_check(self):
        from codeverify_core.proof_docs import FreshnessStatus, ProofBasedDocService

        svc = ProofBasedDocService()
        code = "def fn(x): return x\n"
        doc = svc.generate("API", [("fn", code)])
        assert svc.check_freshness(doc.id, code) == FreshnessStatus.FRESH
        assert svc.check_freshness(doc.id, "def fn(x): return x + 1\n") == FreshnessStatus.STALE

    def test_proof_references(self):
        from codeverify_core.proof_docs import ProofBasedDocService

        svc = ProofBasedDocService()
        doc = svc.generate(
            "API",
            [("fn", "def fn(x): return x\n")],
            specs={"fn": [{"type": "precondition", "description": "x > 0", "proof_id": "P-42"}]},
        )
        section = doc.sections[0]
        assert "P-42" in section.proof_references


class TestGamifiedTraining:
    def test_weakness_analysis(self):
        from codeverify_core.gamified_training import GamifiedTrainingService

        svc = GamifiedTrainingService()
        areas = svc.analyze_weaknesses(
            "dev1",
            [
                {"category": "null_safety"},
                {"category": "null_safety"},
                {"category": "injection"},
            ],
        )
        assert areas[0].category == "null_safety"
        assert areas[0].finding_count == 2

    def test_curriculum_generation(self):
        from codeverify_core.gamified_training import GamifiedTrainingService

        svc = GamifiedTrainingService()
        svc.analyze_weaknesses("dev1", [{"category": "null_safety"}, {"category": "injection"}])
        lessons = svc.get_curriculum("dev1")
        assert len(lessons) >= 1
        assert any("null" in lesson.title.lower() for lesson in lessons)

    def test_challenge_completion(self):
        from codeverify_core.gamified_training import GamifiedTrainingService

        svc = GamifiedTrainingService()
        svc.analyze_weaknesses("dev1", [{"category": "null_safety"}])
        passed, feedback = svc.submit_challenge(
            "dev1", "test", "if user is not None:\n    greet(user)"
        )
        assert passed is True
        assert "✅" in feedback
        progress = svc.get_progress("dev1")
        assert progress.total_points >= 10

    def test_badges_earned(self):
        from codeverify_core.gamified_training import BadgeType, GamifiedTrainingService

        svc = GamifiedTrainingService()
        svc.analyze_weaknesses("dev1", [{"category": "null_safety"}])
        svc.submit_challenge("dev1", "c1", "if x is not None: pass")
        progress = svc.get_progress("dev1")
        assert BadgeType.FIRST_FIX in progress.badges

    def test_leaderboard(self):
        from codeverify_core.gamified_training import GamifiedTrainingService

        svc = GamifiedTrainingService()
        for dev in ["dev1", "dev2"]:
            svc.analyze_weaknesses(dev, [{"category": "null_safety"}])
            svc.submit_challenge(dev, "c", "if x is not None: pass")
        lb = svc.get_leaderboard()
        assert len(lb.entries) == 2
        assert lb.total_challenges_completed >= 2

    def test_skill_level_progression(self):
        from codeverify_core.gamified_training import GamifiedTrainingService, SkillLevel

        svc = GamifiedTrainingService()
        svc.analyze_weaknesses(
            "dev1",
            [
                {"category": "null_safety"},
                {"category": "division_by_zero"},
                {"category": "injection"},
            ],
        )
        for _ in range(4):
            svc.submit_challenge("dev1", "c", "if x is not None: pass")
        progress = svc.get_progress("dev1")
        assert progress.skill_level in (SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED)
