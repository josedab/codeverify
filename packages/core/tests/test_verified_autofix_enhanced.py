"""Tests for interactive fix suggestions and multi-language autofix."""

from codeverify_core.verified_autofix import (
    CodeIssue,
    FixAttemptStatus,
    GeneratedPatch,
    GitHubSuggestedChange,
    MultiLanguageFixGenerator,
    SuggestedChangeGenerator,
    VerificationProof,
    VerifiedFix,
)


def _make_issue(code: str, category: str = "security") -> CodeIssue:
    return CodeIssue(
        file_path="src/app.py",
        line=10,
        message="Test issue",
        severity="high",
        category=category,
        code_snippet=code,
    )


def _make_verified_fix(issue: CodeIssue, patch: GeneratedPatch) -> VerifiedFix:
    return VerifiedFix(
        issue=issue,
        patch=patch,
        proof=VerificationProof(
            issue_resolved=True,
            no_new_bugs=True,
            behavior_preserved=True,
            proof_details="Verified by Z3",
            verification_time_ms=10.0,
        ),
        status=FixAttemptStatus.VERIFIED,
        attempts=1,
    )


class TestGitHubSuggestedChange:
    def test_to_review_comment(self):
        sc = GitHubSuggestedChange(
            file_path="src/app.py",
            start_line=10,
            end_line=10,
            original_code="eval(input())",
            suggested_code="ast.literal_eval(input())",
            comment_body="Replace eval with literal_eval",
            confidence=0.95,
            category="security",
        )
        comment = sc.to_review_comment()
        assert comment["path"] == "src/app.py"
        assert "```suggestion" in comment["body"]
        assert "ast.literal_eval" in comment["body"]
        assert "95%" in comment["body"]

    def test_multiline_suggestion(self):
        sc = GitHubSuggestedChange(
            file_path="f.py",
            start_line=5,
            end_line=8,
            original_code="old",
            suggested_code="new",
            comment_body="Fix",
            confidence=0.9,
            category="bug",
        )
        comment = sc.to_review_comment()
        assert comment["start_line"] == 5
        assert comment["line"] == 8


class TestSuggestedChangeGenerator:
    def test_from_verified_fix(self):
        issue = _make_issue("eval(x)")
        patch = GeneratedPatch(
            original_code="eval(x)",
            fixed_code="ast.literal_eval(x)",
            diff_text="-eval(x)\n+ast.literal_eval(x)",
            explanation="Replace eval",
            confidence=0.95,
        )
        fix = _make_verified_fix(issue, patch)

        gen = SuggestedChangeGenerator()
        suggestions = gen.from_verified_fix(fix)
        assert len(suggestions) == 1
        assert suggestions[0].suggested_code == "ast.literal_eval(x)"

    def test_skips_unverified_fix(self):
        issue = _make_issue("eval(x)")
        patch = GeneratedPatch("eval(x)", "eval(x)", "", "No fix", 0.0)
        fix = VerifiedFix(
            issue=issue,
            patch=patch,
            proof=None,
            status=FixAttemptStatus.FAILED,
            attempts=1,
        )
        gen = SuggestedChangeGenerator()
        assert gen.from_verified_fix(fix) == []

    def test_min_confidence_filter(self):
        fixes = []
        for conf in [0.5, 0.85, 0.95]:
            issue = _make_issue(f"code_{conf}")
            patch = GeneratedPatch(f"old_{conf}", f"new_{conf}", "diff", "fix", conf)
            fixes.append(_make_verified_fix(issue, patch))

        gen = SuggestedChangeGenerator()
        suggestions = gen.from_verified_fixes(fixes, min_confidence=0.8)
        assert len(suggestions) == 2

    def test_format_pr_review(self):
        issue = _make_issue("eval(x)")
        patch = GeneratedPatch("eval(x)", "safe(x)", "-eval\n+safe", "fix", 0.95)
        fix = _make_verified_fix(issue, patch)

        gen = SuggestedChangeGenerator()
        suggestions = gen.from_verified_fix(fix)
        review = gen.format_pr_review(suggestions)
        assert review["event"] == "COMMENT"
        assert "1" in review["body"]
        assert len(review["comments"]) == 1


class TestMultiLanguageFixGenerator:
    def test_go_error_fix(self):
        gen = MultiLanguageFixGenerator()
        issue = _make_issue("result, _ := doWork(ctx)", "error_handling")
        patch = gen.generate_fix(issue, "", language="go")
        assert "err" in patch.fixed_code
        assert patch.confidence > 0

    def test_java_string_equals_fix(self):
        gen = MultiLanguageFixGenerator()
        issue = _make_issue('if (name == "admin")', "bug")
        patch = gen.generate_fix(issue, "", language="java")
        assert ".equals(" in patch.fixed_code

    def test_typescript_any_fix(self):
        gen = MultiLanguageFixGenerator()
        issue = _make_issue("data: any", "type_safety")
        patch = gen.generate_fix(issue, "", language="typescript")
        assert "unknown" in patch.fixed_code

    def test_fallback_to_python_patterns(self):
        gen = MultiLanguageFixGenerator()
        issue = _make_issue("eval(user_input)", "security")
        patch = gen.generate_fix(issue, "", language="python")
        assert "literal_eval" in patch.fixed_code

    def test_go_nil_map_fix(self):
        gen = MultiLanguageFixGenerator()
        issue = _make_issue("var cache map[string]int", "null_safety")
        patch = gen.generate_fix(issue, "", language="go")
        assert "make(" in patch.fixed_code
