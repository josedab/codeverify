"""Semantic Analysis Agent - Understands code intent and behavior."""

import time
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent, CodeContext

logger = structlog.get_logger()


SEMANTIC_SYSTEM_PROMPT = """You are an expert code analyst specializing in understanding code semantics and behavior. Your task is to analyze code changes and identify:

1. **Function Contracts**: Implicit preconditions, postconditions, and invariants
2. **Intent Analysis**: What the code is trying to accomplish
3. **Behavioral Changes**: How the code change affects program behavior
4. **Edge Cases**: Potential edge cases and corner conditions
5. **Assumptions**: Implicit assumptions the code makes

For each function or code block, provide:
- A natural language description of its purpose
- Preconditions (what must be true before execution)
- Postconditions (what is guaranteed after execution)
- Potential issues or concerns

Respond in JSON format with the following structure:
{
  "summary": "Brief summary of the code change",
  "functions": [
    {
      "name": "function_name",
      "purpose": "What this function does",
      "preconditions": ["List of preconditions"],
      "postconditions": ["List of postconditions"],
      "assumptions": ["List of assumptions"],
      "edge_cases": ["Potential edge cases"],
      "concerns": ["Any concerns or potential issues"]
    }
  ],
  "behavioral_changes": ["List of behavioral changes if this is a diff"],
  "verification_hints": ["Hints for formal verification"]
}"""


class SemanticAgent(BaseAgent):
    """
    Agent for semantic code analysis using LLMs.

    This agent understands code intent, extracts function contracts,
    and identifies potential issues based on semantic understanding.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize semantic agent."""
        super().__init__(config)
        # Prefer GPT-4 for semantic understanding
        if config is None:
            self.config.openai_model = "gpt-4-turbo-preview"

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Analyze code semantics.

        Args:
            code: The code to analyze
            context: Additional context including:
                - file_path: Path to the file
                - language: Programming language
                - diff: Git diff if available
                - related_code: Related code for context

        Returns:
            AgentResult with semantic analysis
        """
        start_time = time.time()

        # Use CodeContext for clean parameter handling
        ctx = CodeContext.from_dict(code, context)
        user_prompt = self._build_prompt(ctx)

        try:
            response = await self._call_llm(
                system_prompt=SEMANTIC_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_mode=True,
            )

            # Use common JSON parsing helper
            parsed = self._parse_json_response(response)
            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                "Semantic analysis completed",
                file_path=ctx.file_path,
                functions_found=len(parsed.data.get("functions", [])),
                parse_error=parsed.parse_error,
                latency_ms=elapsed_ms,
            )

            return AgentResult(
                success=True,
                data=parsed.data,
                tokens_used=response.get("tokens", 0),
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("Semantic analysis failed", error=str(e), file_path=ctx.file_path)
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _build_prompt(self, ctx: CodeContext) -> str:
        """Build the analysis prompt using CodeContext."""
        prompt_parts = [
            f"Analyze the following {ctx.language} code from `{ctx.file_path}`:",
            "",
            self._build_code_block(ctx.code, ctx.language),
        ]

        if ctx.diff:
            prompt_parts.extend([
                "",
                "This is a code change. Here is the diff:",
                "",
                self._build_code_block(ctx.diff, "diff"),
            ])

        if ctx.related_code:
            prompt_parts.extend([
                "",
                "Related code for context:",
                "",
                self._build_code_block(ctx.related_code, ctx.language),
            ])

        prompt_parts.extend([
            "",
            "Provide a detailed semantic analysis including function contracts, "
            "preconditions, postconditions, and potential issues.",
        ])

        return "\n".join(prompt_parts)

    async def extract_verification_conditions(
        self,
        code: str,
        language: str,
    ) -> dict[str, Any]:
        """
        Extract verification conditions from code for formal verification.

        This is a specialized analysis that produces conditions
        that can be checked by the Z3 verifier.
        """
        system_prompt = """You are a formal verification expert. Analyze the code and extract verification conditions that can be checked by an SMT solver.

For each verification condition, provide:
1. A unique identifier
2. The type of check (null_safety, bounds_check, overflow_check, etc.)
3. The condition in a form that can be translated to SMT-LIB
4. The source location

Respond in JSON format:
{
  "conditions": [
    {
      "id": "vc_1",
      "type": "overflow_check",
      "description": "Check if multiplication can overflow",
      "expression": "a * b",
      "variables": {"a": {"type": "int32", "range": [0, 1000]}, "b": {"type": "int32", "range": [0, 1000]}},
      "location": {"line": 42, "column": 10}
    }
  ]
}"""

        user_prompt = f"""Extract formal verification conditions from this {language} code:

{self._build_code_block(code, language)}

Focus on:
- Integer overflow checks
- Array bounds checks
- Null/undefined checks
- Division by zero checks
- Type safety checks"""

        try:
            response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_mode=True,
            )
            parsed = self._parse_json_response(response)
            if parsed.parse_error:
                return {"conditions": [], "error": "Failed to parse response"}
            return parsed.data
        except Exception as e:
            logger.error("Failed to extract verification conditions", error=str(e))
            return {"conditions": [], "error": str(e)}
