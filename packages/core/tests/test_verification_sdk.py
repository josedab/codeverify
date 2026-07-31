"""Tests for the Verification-as-Code SDK."""

import pytest

from codeverify_core.verification_sdk import (
    RuleContext,
    RuleRegistry,
    RuleResult,
    Severity,
    VerificationPipeline,
    check,
    verification_rule,
)

# =============================================================================
# Example Rules for Testing
# =============================================================================


@verification_rule(
    id="no-eval",
    severity="critical",
    languages=["python"],
    category="security",
    description="Forbid use of eval()",
)
def no_eval(ctx: RuleContext) -> "RuleResult":
    if ctx.has_pattern(r"\beval\s*\("):
        return ctx.fail(
            "eval() is forbidden", line=ctx.find_line("eval("), fix="Use ast.literal_eval()"
        )
    return ctx.pass_rule()


@verification_rule(
    id="no-exec",
    severity="high",
    languages=["python"],
    category="security",
)
def no_exec(ctx: RuleContext) -> "RuleResult":
    if ctx.has_pattern(r"\bexec\s*\("):
        return ctx.fail("exec() is forbidden", line=ctx.find_line("exec("))
    return ctx.pass_rule()


@check(category="style", severity="low")
def no_print(ctx: RuleContext) -> "RuleResult":
    if "print(" in ctx.code:
        return ctx.fail("Remove print() statements", line=ctx.find_line("print("))
    return ctx.pass_rule()


@verification_rule(
    id="ts-no-any", severity="medium", languages=["typescript"], category="type_safety"
)
def ts_no_any(ctx: RuleContext) -> "RuleResult":
    if ctx.has_pattern(r":\s*any\b"):
        return ctx.fail("Avoid 'any' type", line=ctx.find_line(": any"))
    return ctx.pass_rule()


# =============================================================================
# Tests
# =============================================================================


class TestRuleDecorators:
    def test_verification_rule_metadata(self):
        meta = no_eval.__rule_metadata__
        assert meta.id == "no-eval"
        assert meta.severity == Severity.CRITICAL
        assert "python" in meta.languages
        assert meta.category == "security"

    def test_check_decorator_metadata(self):
        meta = no_print.__rule_metadata__
        assert meta.id == "no_print"
        assert meta.severity == Severity.LOW
        assert meta.category == "style"

    def test_rule_execution(self):
        ctx = RuleContext(code="result = eval(input())", language="python")
        result = no_eval(ctx)
        assert result.passed is False
        assert "forbidden" in result.message
        assert result.line > 0

    def test_rule_passes_clean_code(self):
        ctx = RuleContext(code="result = safe_parse(input())", language="python")
        result = no_eval(ctx)
        assert result.passed is True


class TestRuleContext:
    def test_find_line(self):
        ctx = RuleContext(code="line1\nline2\neval(x)\nline4")
        assert ctx.find_line("eval(") == 3

    def test_find_all_lines(self):
        ctx = RuleContext(code="eval(a)\nsafe()\neval(b)")
        assert ctx.find_all_lines("eval(") == [1, 3]

    def test_has_pattern(self):
        ctx = RuleContext(code="x = eval(input())")
        assert ctx.has_pattern(r"\beval\s*\(") is True
        assert ctx.has_pattern(r"\bexec\s*\(") is False


class TestVerificationPipeline:
    def test_basic_pipeline(self):
        pipeline = VerificationPipeline("test")
        pipeline.add(no_eval).add(no_exec)
        assert pipeline.rule_count == 2

        result = pipeline.run("x = eval(input())", language="python")
        assert result.rules_run == 2
        assert result.rules_failed == 1
        assert result.passed is False
        assert result.findings[0]["rule_id"] == "no-eval"

    def test_all_pass(self):
        pipeline = VerificationPipeline("test")
        pipeline.add(no_eval).add(no_exec)
        result = pipeline.run("x = safe(input())", language="python")
        assert result.passed is True
        assert result.rules_passed == 2

    def test_language_filtering(self):
        pipeline = VerificationPipeline("test")
        pipeline.add(no_eval)  # python only
        pipeline.add(ts_no_any)  # typescript only

        result = pipeline.run("x: any = 1", language="typescript")
        assert result.rules_run == 1
        assert result.rules_skipped == 1
        assert result.rules_failed == 1

    def test_fail_fast(self):
        pipeline = VerificationPipeline("test", fail_fast=True)
        pipeline.add(no_eval).add(no_exec)
        result = pipeline.run("eval(exec('x'))", language="python")
        assert result.rules_failed == 1  # stops after first failure

    def test_compose_pipelines(self):
        p1 = VerificationPipeline("security")
        p1.add(no_eval)
        p2 = VerificationPipeline("style")
        p2.add(no_print)

        combined = p1.compose(p2)
        assert combined.rule_count == 2
        assert "security+style" in combined.name

    def test_summary_output(self):
        pipeline = VerificationPipeline("test")
        pipeline.add(no_eval)
        result = pipeline.run("safe()", language="python")
        summary = result.summary()
        assert "[PASS]" in summary
        assert "test" in summary


class TestRuleRegistry:
    def test_register_and_list(self):
        registry = RuleRegistry()
        registry.register(no_eval)
        registry.register(no_exec)
        registry.register(ts_no_any)

        assert registry.count == 3
        rules = registry.list_rules()
        ids = {r["id"] for r in rules}
        assert "no-eval" in ids
        assert "ts-no-any" in ids

    def test_get_by_category(self):
        registry = RuleRegistry()
        registry.register(no_eval)
        registry.register(no_exec)
        registry.register(no_print)

        security = registry.get_by_category("security")
        assert len(security) == 2

    def test_get_by_language(self):
        registry = RuleRegistry()
        registry.register(no_eval)
        registry.register(ts_no_any)

        py_rules = registry.get_by_language("python")
        assert len(py_rules) == 1

    def test_to_pipeline(self):
        registry = RuleRegistry()
        registry.register(no_eval)
        registry.register(no_exec)
        registry.register(no_print)

        pipeline = registry.to_pipeline(
            "sec-only",
            categories=["security"],
        )
        assert pipeline.rule_count == 2

    def test_to_pipeline_with_severity_filter(self):
        registry = RuleRegistry()
        registry.register(no_eval)  # critical
        registry.register(no_exec)  # high
        registry.register(no_print)  # low

        pipeline = registry.to_pipeline("high+", severity_min="high")
        assert pipeline.rule_count == 2  # no_eval (critical) + no_exec (high)

    def test_register_undecorated_fails(self):
        registry = RuleRegistry()
        with pytest.raises(ValueError, match="decorated"):
            registry.register(lambda _ctx: None)

    def test_get_rule(self):
        registry = RuleRegistry()
        registry.register(no_eval)
        assert registry.get("no-eval") is not None
        assert registry.get("nonexistent") is None
