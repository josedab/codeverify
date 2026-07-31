"""LLM-Powered Proof Explanation Agent.

AI agent that uses LLMs to generate human-readable explanations of
Z3 proofs and counterexamples, falling back to the template-based
system in codeverify_core.nl_proof_explanation when LLMs are unavailable.

Features:
- LLM-based counterexample narrative generation
- Context-aware explanations with source code understanding
- Multi-detail-level output (brief, standard, detailed)
- Automatic fallback to template system
- Explanation quality scoring
- Token usage tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class ExplanationMode(str, Enum):
    """Mode of explanation generation."""

    LLM = "llm"
    TEMPLATE = "template"
    HYBRID = "hybrid"


@dataclass
class ExplanationRequest:
    """Request to explain a proof or counterexample."""

    check_type: str = ""
    function_name: str = ""
    file_path: str = ""
    line: int = 0
    code_snippet: str = ""
    variable_assignments: dict[str, Any] = field(default_factory=dict)
    constraint_violated: str = ""
    z3_output: str = ""
    severity: str = "medium"
    detail_level: str = "standard"  # brief, standard, detailed


@dataclass
class ExplanationResponse:
    """Response with the generated explanation."""

    title: str = ""
    narrative: str = ""
    fix_suggestion: str = ""
    confidence: float = 0.0
    mode_used: ExplanationMode = ExplanationMode.TEMPLATE
    tokens_used: int = 0
    latency_ms: float = 0.0


class ProofExplanationAgent(BaseAgent):
    """AI agent for generating proof explanations using LLMs.

    Falls back to template-based explanations from
    codeverify_core.nl_proof_explanation when LLM is unavailable.
    """

    SYSTEM_PROMPT = """You are a code verification expert who explains formal verification
results to software developers. Given a Z3 SMT solver counterexample or proof result,
generate a clear, actionable explanation in plain English.

Rules:
1. Start with a one-sentence summary of the issue
2. Explain what the counterexample means in terms of the original code
3. Show the concrete input values that trigger the bug
4. Suggest a specific fix with a code example
5. Use the developer's language (Python, TypeScript, etc.) for examples
6. Be concise — aim for 3-5 sentences for standard detail level"""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config)
        self._mode = ExplanationMode.HYBRID
        self._template_fallback = self._init_template_system()

    async def analyze(self, code: str, **kwargs: Any) -> AgentResult:
        """Analyze a proof/counterexample and generate explanation."""
        request = ExplanationRequest(
            check_type=kwargs.get("check_type", ""),
            function_name=kwargs.get("function_name", ""),
            file_path=kwargs.get("file_path", ""),
            line=kwargs.get("line", 0),
            code_snippet=code,
            variable_assignments=kwargs.get("variable_assignments", {}),
            constraint_violated=kwargs.get("constraint_violated", ""),
            z3_output=kwargs.get("z3_output", ""),
            severity=kwargs.get("severity", "medium"),
            detail_level=kwargs.get("detail_level", "standard"),
        )

        response = await self.explain(request)
        return AgentResult(
            success=True,
            data={
                "title": response.title,
                "narrative": response.narrative,
                "fix_suggestion": response.fix_suggestion,
                "mode_used": response.mode_used.value,
            },
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
        )

    async def explain(self, request: ExplanationRequest) -> ExplanationResponse:
        """Generate an explanation, trying LLM first with template fallback."""
        start = time.time()

        if self._mode in (ExplanationMode.LLM, ExplanationMode.HYBRID):
            try:
                response = await self._explain_with_llm(request)
                if response.narrative:
                    response.latency_ms = (time.time() - start) * 1000
                    return response
            except Exception as e:
                logger.warning("llm_explanation_failed", error=str(e))

        # Fallback to templates
        response = self._explain_with_templates(request)
        response.latency_ms = (time.time() - start) * 1000
        return response

    async def _explain_with_llm(self, request: ExplanationRequest) -> ExplanationResponse:
        """Generate explanation using LLM."""
        assignments_str = ", ".join(f"{k} = {v}" for k, v in request.variable_assignments.items())
        user_prompt = (
            f"Explain this {request.check_type} verification finding:\n\n"
            f"Function: {request.function_name} in {request.file_path}:{request.line}\n"
            f"Severity: {request.severity}\n"
            f"Counterexample: {assignments_str}\n"
            f"Constraint violated: {request.constraint_violated}\n\n"
            f"Code:\n```\n{request.code_snippet}\n```\n\n"
            f"Detail level: {request.detail_level}\n"
            f"Provide: 1) Summary 2) Explanation 3) Fix suggestion"
        )

        try:
            result = await self._call_openai(self.SYSTEM_PROMPT, user_prompt, json_mode=True)
            content = result.get("content", {})
            tokens = result.get("tokens", 0)

            if isinstance(content, dict):
                return ExplanationResponse(
                    title=content.get(
                        "summary", f"{request.check_type} in {request.function_name}"
                    ),
                    narrative=content.get("explanation", ""),
                    fix_suggestion=content.get("fix_suggestion", ""),
                    confidence=0.9,
                    mode_used=ExplanationMode.LLM,
                    tokens_used=tokens,
                )
        except Exception:
            pass

        return ExplanationResponse(mode_used=ExplanationMode.LLM)

    def _explain_with_templates(self, request: ExplanationRequest) -> ExplanationResponse:
        """Generate explanation using the template system."""
        if not self._template_fallback:
            return self._basic_template_explanation(request)

        try:
            from codeverify_core.nl_proof_explanation import (
                DetailLevel,
                ExplanationContext,
            )

            detail_map = {
                "brief": DetailLevel.BRIEF,
                "standard": DetailLevel.STANDARD,
                "detailed": DetailLevel.DETAILED,
            }
            detail = detail_map.get(request.detail_level, DetailLevel.STANDARD)

            ctx = ExplanationContext(
                check_type=request.check_type,
                function_name=request.function_name,
                file_path=request.file_path,
                line=request.line,
                code_snippet=request.code_snippet,
                variable_assignments=request.variable_assignments,
                constraint_violated=request.constraint_violated,
                z3_output=request.z3_output,
                severity=request.severity,
            )
            explanation = self._template_fallback.explain(ctx, detail)
            return ExplanationResponse(
                title=explanation.title,
                narrative=explanation.narrative,
                fix_suggestion=explanation.fix_suggestion,
                confidence=0.7,
                mode_used=ExplanationMode.TEMPLATE,
            )
        except Exception as e:
            logger.warning("template_explanation_failed", error=str(e))
            return self._basic_template_explanation(request)

    def _basic_template_explanation(self, request: ExplanationRequest) -> ExplanationResponse:
        """Minimal fallback when both LLM and template system fail."""
        var_str = ", ".join(f"{k}={v}" for k, v in request.variable_assignments.items())
        narrative = (
            f"{request.check_type} violation in `{request.function_name}` "
            f"at {request.file_path}:{request.line}. "
            f"Counterexample: {var_str or 'none'}. "
            f"Severity: {request.severity}."
        )
        return ExplanationResponse(
            title=f"{request.check_type} in {request.function_name}",
            narrative=narrative,
            confidence=0.5,
            mode_used=ExplanationMode.TEMPLATE,
        )

    def _init_template_system(self) -> Any:
        """Initialize the template-based fallback."""
        try:
            from codeverify_core.nl_proof_explanation import NLProofExplanationService

            return NLProofExplanationService()
        except ImportError:
            logger.info("template_system_unavailable")
            return None

    def set_mode(self, mode: ExplanationMode) -> None:
        self._mode = mode
