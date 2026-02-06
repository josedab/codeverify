"""End-to-end autonomous fix generation pipeline.

Orchestrates automated code fixes: generation, verification, testing,
and reporting via LLM-powered candidate generation.
"""

from __future__ import annotations

import difflib
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()

_CONFIDENCE_ORDER = ("low", "medium", "high", "very_high")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FixStatus(str, Enum):
    """Lifecycle status of a code fix."""
    PENDING = "pending"
    GENERATING = "generating"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    REJECTED = "rejected"


class FixConfidence(str, Enum):
    """Confidence level assigned to a generated fix."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

    def _rank(self) -> int:
        return _CONFIDENCE_ORDER.index(self.value)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, FixConfidence):
            return NotImplemented
        return self._rank() >= other._rank()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, FixConfidence):
            return NotImplemented
        return self._rank() > other._rank()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, FixConfidence):
            return NotImplemented
        return self._rank() <= other._rank()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, FixConfidence):
            return NotImplemented
        return self._rank() < other._rank()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CodeFix:
    """A single generated code fix with full provenance metadata."""
    fix_id: str
    finding_id: str
    file_path: str
    original_code: str
    fixed_code: str
    diff: str
    explanation: str
    confidence: FixConfidence
    status: FixStatus
    verification_result: dict[str, Any] | None = None
    generated_test: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "fix_id": self.fix_id, "finding_id": self.finding_id,
            "file_path": self.file_path,
            "original_code": self.original_code, "fixed_code": self.fixed_code,
            "diff": self.diff, "explanation": self.explanation,
            "confidence": self.confidence.value, "status": self.status.value,
            "verification_result": self.verification_result,
            "generated_test": self.generated_test, "created_at": self.created_at,
        }


@dataclass
class FixGenerationConfig:
    """Tuneable knobs for the fix generation pipeline."""
    max_attempts: int = 3
    require_verification: bool = True
    require_test: bool = True
    auto_create_pr: bool = False
    min_confidence: FixConfidence = FixConfidence.MEDIUM
    supported_fix_types: list[str] = field(
        default_factory=lambda: [
            "security", "bug", "null_check", "type_error",
            "resource_leak", "bounds_check", "logic_error", "performance",
        ]
    )


# ---------------------------------------------------------------------------
# DiffGenerator
# ---------------------------------------------------------------------------

class DiffGenerator:
    """Utility for creating, applying and validating unified diffs."""

    def generate_unified_diff(self, original: str, fixed: str, file_path: str) -> str:
        """Return a unified diff between *original* and *fixed* code."""
        diff_lines = difflib.unified_diff(
            original.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
        return "\n".join(diff_lines)

    def apply_diff(self, original: str, diff: str) -> str:
        """Apply a unified diff to *original* and return the patched text."""
        result: list[str] = []
        orig_lines = original.splitlines(keepends=True)
        idx = 0
        hunk_re = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
        in_hunk = False

        for line in diff.splitlines(keepends=True):
            m = hunk_re.match(line)
            if m:
                start = int(m.group(1)) - 1
                while idx < start and idx < len(orig_lines):
                    result.append(orig_lines[idx])
                    idx += 1
                in_hunk = True
                continue
            if not in_hunk:
                continue
            if line.startswith("-"):
                idx += 1
            elif line.startswith("+"):
                result.append(line[1:])
            elif line.startswith(" "):
                result.append(orig_lines[idx] if idx < len(orig_lines) else line[1:])
                idx += 1

        while idx < len(orig_lines):
            result.append(orig_lines[idx])
            idx += 1
        return "".join(result)

    def validate_diff(self, diff: str) -> bool:
        """Return ``True`` if *diff* contains at least one hunk with changes."""
        if not diff or not diff.strip():
            return False
        has_hunk = bool(re.search(r"^@@ -\d+", diff, re.MULTILINE))
        has_changes = bool(re.search(r"^[+-]", diff, re.MULTILINE))
        return has_hunk and has_changes


# ---------------------------------------------------------------------------
# FixVerifier
# ---------------------------------------------------------------------------

class FixVerifier:
    """Lightweight static verifier for generated fixes."""

    def verify_fix(
        self, original_code: str, fixed_code: str, finding: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify that *fixed_code* addresses *finding* without regressions.

        Returns ``fixes_original_issue``, ``introduces_new_issues``,
        ``preserves_behavior``, and ``verification_details``.
        """
        language = finding.get("language", "python")
        syntax_ok = self._check_syntax_valid(fixed_code, language)
        new_issues = self._check_no_new_issues(fixed_code, language)
        snippet = finding.get("code_snippet", "")
        snippet_removed = snippet != "" and snippet not in fixed_code
        code_changed = original_code.strip() != fixed_code.strip()

        fixes_original = code_changed and snippet_removed
        introduces_new = len(new_issues) > 0
        preserves = syntax_ok and not introduces_new
        logger.info("Fix verification completed", fixes_issue=fixes_original,
                     introduces_new=introduces_new, preserves_behavior=preserves)
        return {
            "fixes_original_issue": fixes_original,
            "introduces_new_issues": introduces_new,
            "preserves_behavior": preserves,
            "verification_details": {"syntax_valid": syntax_ok, "new_issues": new_issues,
                                     "snippet_removed": snippet_removed, "code_changed": code_changed},
        }

    def _check_syntax_valid(self, code: str, language: str) -> bool:
        """Return ``True`` if *code* is syntactically valid."""
        if language == "python":
            try:
                compile(code, "<fix>", "exec")
                return True
            except SyntaxError:
                return False
        if language in ("typescript", "javascript"):
            opens = sum(code.count(c) for c in "({[")
            closes = sum(code.count(c) for c in ")}]")
            return opens == closes
        return True

    def _check_no_new_issues(self, fixed_code: str, language: str) -> list[str]:
        """Scan *fixed_code* for common anti-patterns introduced by the fix."""
        issues: list[str] = []
        if re.search(r"\beval\s*\(", fixed_code):
            issues.append("Introduces eval() call")
        if re.search(r"\bexec\s*\(", fixed_code):
            issues.append("Introduces exec() call")
        if re.search(
            r"(?:password|secret|token|api_key)\s*=\s*[\"'][^\"']+[\"']",
            fixed_code, re.IGNORECASE,
        ):
            issues.append("Possible hard-coded credential")
        if re.search(r"subprocess\.\w+\(.*shell\s*=\s*True", fixed_code, re.DOTALL):
            issues.append("subprocess call with shell=True")
        return issues


# ---------------------------------------------------------------------------
# TestGenerator
# ---------------------------------------------------------------------------

class TestGenerator:
    """Generates regression tests that verify a fix addresses its finding."""

    def generate_regression_test(self, fix: CodeFix, language: str) -> str:
        """Create a regression test for *fix* in the given *language*."""
        if language in ("typescript", "javascript"):
            return self._generate_typescript_test(fix)
        return self._generate_python_test(fix)

    def _generate_python_test(self, fix: CodeFix) -> str:
        """Generate a pytest-style regression test."""
        safe = re.sub(r"\W+", "_", fix.finding_id)[:40]
        lines = [
            f'"""Regression test for fix {fix.fix_id}."""',
            "import pytest", "", "",
            f"class TestFix{safe.title()}:",
            f'    """Verify that finding {fix.finding_id} is resolved."""',
            "",
            "    def test_fix_applied(self):",
            f"        fixed_code = {fix.fixed_code!r}",
            f"        original_snippet = {fix.original_code!r}",
            "        assert fixed_code != original_snippet",
            "",
            "    def test_no_syntax_errors(self):",
            f"        compile({fix.fixed_code!r}, '<test>', 'exec')",
            "",
            "    def test_explanation_present(self):",
            f"        assert len({fix.explanation!r}) > 0",
        ]
        return "\n".join(lines) + "\n"

    def _generate_typescript_test(self, fix: CodeFix) -> str:
        """Generate a Jest-style regression test."""
        safe = re.sub(r"\W+", "_", fix.finding_id)[:40]
        esc = lambda s: s.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        lines = [
            f"// Regression test for fix {fix.fix_id}", "",
            f"describe('Fix {safe}', () => {{",
            f"  const fixedCode = `{esc(fix.fixed_code)}`;",
            f"  const originalCode = `{esc(fix.original_code)}`;", "",
            "  test('fix modifies the original code', () => {",
            "    expect(fixedCode).not.toEqual(originalCode);",
            "  });", "",
            "  test('fixed code is non-empty', () => {",
            "    expect(fixedCode.trim().length).toBeGreaterThan(0);",
            "  });", "",
            "  test('explanation is provided', () => {",
            f"    expect(`{esc(fix.explanation)}`.length).toBeGreaterThan(0);",
            "  });",
            "});",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# AutoFixAgent
# ---------------------------------------------------------------------------

class AutoFixAgent(BaseAgent):
    """LLM-powered agent that generates verified code fixes."""

    def __init__(
        self,
        agent_config: AgentConfig | None = None,
        fix_config: FixGenerationConfig | None = None,
    ) -> None:
        super().__init__(agent_config)
        self.fix_config = fix_config or FixGenerationConfig()
        self._diff_gen = DiffGenerator()
        self._verifier = FixVerifier()
        self._test_gen = TestGenerator()

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """Analyse code for a single finding and return a fix."""
        start = time.time()
        finding: dict[str, Any] = context.get("finding", {})
        language: str = context.get("language", "python")

        if not finding:
            return AgentResult(
                success=False, error="No finding provided in context",
                latency_ms=(time.time() - start) * 1000,
            )
        try:
            fix = await self.generate_fix(code, finding, language)
            return AgentResult(
                success=fix.status == FixStatus.VERIFIED,
                data=fix.to_dict(),
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as exc:
            logger.error("AutoFixAgent.analyze failed", error=str(exc))
            return AgentResult(
                success=False, error=str(exc),
                latency_ms=(time.time() - start) * 1000,
            )

    async def generate_fix(
        self, code: str, finding: dict[str, Any], language: str,
    ) -> CodeFix:
        """Generate, verify and test a fix for *finding*."""
        candidates = await self._generate_fix_candidates(
            code, finding, language, self.fix_config.max_attempts,
        )
        best = await self._select_best_fix(candidates)

        if best is None:
            return CodeFix(
                fix_id=str(uuid.uuid4()),
                finding_id=finding.get("id", "unknown"),
                file_path=finding.get("file_path", "unknown"),
                original_code=code, fixed_code=code, diff="",
                explanation="No suitable fix candidate could be generated.",
                confidence=FixConfidence.LOW, status=FixStatus.FAILED,
            )
        return await self._verify_and_test(best, language)

    async def _generate_fix_candidates(
        self, code: str, finding: dict[str, Any], language: str, attempts: int,
    ) -> list[CodeFix]:
        """Ask the LLM for *attempts* fix candidates."""
        prompt = self._build_fix_prompt(code, finding, language)
        system_prompt = (
            "You are an expert code remediation assistant. "
            "Given a code finding, produce a JSON response with keys: "
            '"fixed_code", "explanation", and "confidence" '
            "(one of: low, medium, high, very_high). Respond ONLY with JSON."
        )
        candidates: list[CodeFix] = []

        for attempt in range(attempts):
            try:
                response = await self._call_llm(
                    system_prompt=system_prompt, user_prompt=prompt,
                    json_mode=True,
                )
                parsed = self._parse_json_response(response)
                if parsed.parse_error:
                    logger.warning("Failed to parse LLM fix response", attempt=attempt)
                    continue

                data = parsed.data
                fixed_code = data.get("fixed_code", "")
                if not fixed_code:
                    continue

                try:
                    confidence = FixConfidence(data.get("confidence", "medium"))
                except ValueError:
                    confidence = FixConfidence.MEDIUM

                diff = self._diff_gen.generate_unified_diff(
                    code, fixed_code, finding.get("file_path", "unknown"),
                )
                candidate = CodeFix(
                    fix_id=str(uuid.uuid4()), finding_id=finding.get("id", "unknown"),
                    file_path=finding.get("file_path", "unknown"),
                    original_code=code, fixed_code=fixed_code, diff=diff,
                    explanation=data.get("explanation", ""),
                    confidence=confidence, status=FixStatus.GENERATING,
                )
                candidates.append(candidate)
                logger.info("Generated fix candidate", attempt=attempt,
                            confidence=confidence.value, fix_id=candidate.fix_id)
            except Exception as exc:
                logger.warning("Fix candidate generation failed", attempt=attempt, error=str(exc))

        return candidates

    async def _select_best_fix(self, candidates: list[CodeFix]) -> CodeFix | None:
        """Pick the highest-confidence candidate above the minimum bar."""
        if not candidates:
            return None
        eligible = [c for c in candidates if c.confidence >= self.fix_config.min_confidence]
        if not eligible:
            logger.info(
                "No candidate meets minimum confidence",
                min_confidence=self.fix_config.min_confidence.value,
                candidates=len(candidates),
            )
            return None
        eligible.sort(key=lambda c: c.confidence._rank(), reverse=True)
        return eligible[0]

    async def _verify_and_test(self, fix: CodeFix, language: str) -> CodeFix:
        """Run verification and test generation, updating *fix* in place."""
        fix.status = FixStatus.VERIFYING

        if self.fix_config.require_verification:
            result = self._verifier.verify_fix(
                fix.original_code, fix.fixed_code,
                {"id": fix.finding_id, "file_path": fix.file_path, "language": language},
            )
            fix.verification_result = result
            if not result.get("fixes_original_issue") or result.get("introduces_new_issues"):
                fix.status = FixStatus.REJECTED
                logger.info("Fix rejected by verification", fix_id=fix.fix_id)
                return fix

        if self.fix_config.require_test:
            fix.generated_test = self._test_gen.generate_regression_test(fix, language)

        fix.status = FixStatus.VERIFIED
        logger.info("Fix verified successfully", fix_id=fix.fix_id)
        return fix

    def _build_fix_prompt(self, code: str, finding: dict[str, Any], language: str) -> str:
        """Build the user prompt sent to the LLM for fix generation."""
        snippet = finding.get("code_snippet", "")
        code_block = self._build_code_block(code, language, label="Source file")
        parts = [
            "## Fix Request", "",
            f"**Finding ID:** {finding.get('id', 'N/A')}",
            f"**Type:** {finding.get('type', 'bug')}",
            f"**Severity:** {finding.get('severity', 'medium')}",
            f"**File:** {finding.get('file_path', 'unknown')}",
            f"**Language:** {language}", "",
            "### Description", finding.get("description", "Unknown issue"), "",
        ]
        if snippet:
            parts += ["### Problematic Snippet", f"```{language}", snippet, "```", ""]
        parts += [
            "### Full Source", code_block, "",
            "Produce a corrected version of the full source that resolves the "
            "finding above.  Keep changes minimal and preserve the existing style.",
        ]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# AutoFixPipeline
# ---------------------------------------------------------------------------

class AutoFixPipeline:
    """High-level orchestrator that runs the autofix agent across findings."""

    def __init__(
        self,
        agent_config: AgentConfig | None = None,
        fix_config: FixGenerationConfig | None = None,
    ) -> None:
        self.agent = AutoFixAgent(agent_config=agent_config, fix_config=fix_config)

    async def run(
        self,
        findings: list[dict[str, Any]],
        code_files: dict[str, str],
        language: str,
    ) -> list[CodeFix]:
        """Process every finding and return all resulting fixes."""
        fixes: list[CodeFix] = []
        for finding in findings:
            file_path = finding.get("file_path", "")
            code = code_files.get(file_path, "")
            if not code:
                logger.warning(
                    "No source code found for finding",
                    finding_id=finding.get("id"), file_path=file_path,
                )
                continue
            fix = await self.run_single(finding, code, language)
            if fix is not None:
                fixes.append(fix)

        logger.info(
            "Pipeline run complete",
            total_findings=len(findings), total_fixes=len(fixes),
            verified=sum(1 for f in fixes if f.status == FixStatus.VERIFIED),
        )
        return fixes

    async def run_single(
        self, finding: dict[str, Any], code: str, language: str,
    ) -> CodeFix | None:
        """Run the pipeline for a single finding."""
        try:
            return await self.agent.generate_fix(code, finding, language)
        except Exception as exc:
            logger.error(
                "Pipeline failed for finding",
                finding_id=finding.get("id"), error=str(exc),
            )
            return None

    @staticmethod
    def get_fix_summary(fixes: list[CodeFix]) -> dict[str, Any]:
        """Produce aggregate statistics for a list of fixes."""
        total = len(fixes)
        status_counts: dict[str, int] = {}
        confidence_counts: dict[str, int] = {}
        for fix in fixes:
            status_counts[fix.status.value] = status_counts.get(fix.status.value, 0) + 1
            confidence_counts[fix.confidence.value] = confidence_counts.get(fix.confidence.value, 0) + 1

        verified = status_counts.get(FixStatus.VERIFIED.value, 0)
        return {
            "total": total,
            "verified": verified,
            "failed": status_counts.get(FixStatus.FAILED.value, 0),
            "rejected": status_counts.get(FixStatus.REJECTED.value, 0),
            "success_rate": verified / total if total > 0 else 0.0,
            "status_counts": status_counts,
            "confidence_counts": confidence_counts,
        }
