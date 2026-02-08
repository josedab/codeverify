"""Tests for Verified Autofix module."""

import asyncio

import pytest

from codeverify_core.verified_autofix import (
    AutofixConfig,
    AutofixPipeline,
    CodeIssue,
    DifferentialVerifier,
    FixAttemptStatus,
    FixCache,
    FixGenerator,
    GeneratedPatch,
    VerificationProof,
    VerifiedFix,
    get_autofix_pipeline,
    reset_autofix_pipeline,
)


class TestFixAttemptStatus:
    """Tests for FixAttemptStatus enum."""

    def test_all_statuses_exist(self):
        """All expected statuses exist."""
        assert FixAttemptStatus.PENDING.value == "pending"
        assert FixAttemptStatus.GENERATING.value == "generating"
        assert FixAttemptStatus.VERIFYING.value == "verifying"
        assert FixAttemptStatus.VERIFIED.value == "verified"
        assert FixAttemptStatus.FAILED.value == "failed"
        assert FixAttemptStatus.REJECTED.value == "rejected"


class TestAutofixConfig:
    """Tests for AutofixConfig dataclass."""

    def test_default_values(self):
        """Default config values."""
        config = AutofixConfig()
        assert config.max_attempts == 3
        assert config.verify_fix is True
        assert config.timeout_seconds == 30
        assert "python" in config.allowed_languages


class TestCodeIssue:
    """Tests for CodeIssue dataclass."""

    def test_creation(self):
        """Can create a CodeIssue."""
        issue = CodeIssue(
            file_path="src/main.py",
            line=10,
            message="Use of eval()",
            severity="error",
            category="security",
            code_snippet="result = eval(user_input)",
        )
        assert issue.file_path == "src/main.py"
        assert issue.line == 10
        assert issue.category == "security"


class TestGeneratedPatch:
    """Tests for GeneratedPatch dataclass."""

    def test_creation_with_confidence(self):
        """Can create a GeneratedPatch with confidence."""
        patch = GeneratedPatch(
            original_code="eval(x)",
            fixed_code="ast.literal_eval(x)",
            diff_text="-eval(x)\n+ast.literal_eval(x)",
            explanation="Replaced eval with ast.literal_eval.",
            confidence=0.95,
        )
        assert patch.confidence == 0.95
        assert patch.fixed_code == "ast.literal_eval(x)"


class TestVerificationProof:
    """Tests for VerificationProof dataclass."""

    def test_all_bool_fields(self):
        """Can create proof with all boolean fields."""
        proof = VerificationProof(
            issue_resolved=True,
            no_new_bugs=True,
            behavior_preserved=True,
            proof_details="All checks passed.",
            verification_time_ms=1.5,
        )
        assert proof.issue_resolved is True
        assert proof.no_new_bugs is True
        assert proof.behavior_preserved is True
        assert proof.verification_time_ms > 0


class TestFixGenerator:
    """Tests for FixGenerator."""

    def _make_issue(self, snippet: str, category: str = "security") -> CodeIssue:
        return CodeIssue(
            file_path="src/test.py",
            line=1,
            message="issue",
            severity="error",
            category=category,
            code_snippet=snippet,
        )

    def test_eval_to_literal_eval(self):
        """Replaces eval() with ast.literal_eval()."""
        gen = FixGenerator()
        issue = self._make_issue("result = eval(user_input)")
        patch = gen.generate_fix(issue, "")
        assert "ast.literal_eval(" in patch.fixed_code
        assert patch.confidence > 0.9

    def test_bare_except_to_exception(self):
        """Replaces bare except with except Exception."""
        gen = FixGenerator()
        issue = self._make_issue("except:", category="error_handling")
        patch = gen.generate_fix(issue, "")
        assert "except Exception:" in patch.fixed_code
        assert patch.confidence > 0.9

    def test_equality_none_to_is_none(self):
        """Replaces == None with is None."""
        gen = FixGenerator()
        issue = self._make_issue("if x == None:")
        patch = gen.generate_fix(issue, "")
        assert "is None" in patch.fixed_code
        assert patch.confidence > 0.9


class TestDifferentialVerifier:
    """Tests for DifferentialVerifier."""

    def test_verify_fix_returns_valid_proof(self):
        """verify_fix returns a VerificationProof."""
        verifier = DifferentialVerifier()
        issue = CodeIssue(
            file_path="f.py",
            line=1,
            message="eval usage",
            severity="error",
            category="security",
            code_snippet="eval(x)",
        )
        patch = GeneratedPatch(
            original_code="eval(x)",
            fixed_code="ast.literal_eval(x)",
            diff_text="-eval(x)\n+ast.literal_eval(x)",
            explanation="safe",
            confidence=0.95,
        )
        proof = verifier.verify_fix(issue, "eval(x)", patch)
        assert isinstance(proof, VerificationProof)
        assert proof.issue_resolved is True
        assert proof.verification_time_ms >= 0


class TestAutofixPipeline:
    """Tests for AutofixPipeline."""

    def _make_issue(self, snippet: str) -> CodeIssue:
        return CodeIssue(
            file_path="src/test.py",
            line=1,
            message="eval usage",
            severity="error",
            category="security",
            code_snippet=snippet,
        )

    def test_fix_issue_returns_verified_fix(self):
        """fix_issue returns a VerifiedFix."""
        pipeline = AutofixPipeline()
        issue = self._make_issue("eval(user_input)")
        result = asyncio.get_event_loop().run_until_complete(
            pipeline.fix_issue(issue, "")
        )
        assert isinstance(result, VerifiedFix)
        assert result.status == FixAttemptStatus.VERIFIED

    def test_fix_batch_handles_multiple_issues(self):
        """fix_batch processes multiple issues."""
        pipeline = AutofixPipeline()
        issues = [
            self._make_issue("eval(a)"),
            self._make_issue("eval(b)"),
        ]
        results = asyncio.get_event_loop().run_until_complete(
            pipeline.fix_batch(issues, "")
        )
        assert len(results) == 2
        assert all(isinstance(r, VerifiedFix) for r in results)


class TestFixCache:
    """Tests for FixCache."""

    def test_store_and_retrieve(self):
        """Stores and retrieves cached fixes."""
        cache = FixCache()
        issue = CodeIssue(
            file_path="f.py",
            line=1,
            message="msg",
            severity="error",
            category="security",
            code_snippet="eval(x)",
        )
        fix = VerifiedFix(
            issue=issue,
            patch=GeneratedPatch(
                original_code="eval(x)",
                fixed_code="ast.literal_eval(x)",
                diff_text="",
                explanation="",
                confidence=0.95,
            ),
            proof=None,
            status=FixAttemptStatus.VERIFIED,
            attempts=1,
        )
        assert cache.lookup(issue) is None
        cache.store(issue, fix)
        assert cache.lookup(issue) is not None
        assert cache.size == 1


class TestAutofixSingletons:
    """Tests for module-level singletons."""

    def test_get_and_reset_autofix_pipeline(self):
        """get/reset autofix pipeline singletons."""
        reset_autofix_pipeline()
        p1 = get_autofix_pipeline()
        p2 = get_autofix_pipeline()
        assert p1 is p2
        reset_autofix_pipeline()
        p3 = get_autofix_pipeline()
        assert p3 is not p1
        reset_autofix_pipeline()
