"""Synthesis Agent - Combines results and generates fixes."""

import json
import time
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent, CodeContext

logger = structlog.get_logger()


SYNTHESIS_SYSTEM_PROMPT = """You are an expert at synthesizing code analysis results and generating clear, actionable feedback. Your task is to:

1. **Consolidate Findings**: Merge and deduplicate findings from multiple analysis sources
2. **Prioritize Issues**: Rank issues by severity and impact
3. **Generate Fixes**: Create concrete, working code fixes for each issue
4. **Explain Clearly**: Write explanations that help developers understand and learn

When generating fixes:
- Provide complete, syntactically correct code
- Preserve the original code style and formatting
- Include comments explaining the fix
- Offer alternative approaches when applicable

Respond in JSON format:
{
  "summary": {
    "total_issues": 5,
    "critical": 1,
    "high": 2,
    "medium": 1,
    "low": 1,
    "pass": false,
    "recommendation": "Fix critical and high severity issues before merging"
  },
  "findings": [
    {
      "id": "finding_1",
      "severity": "critical",
      "category": "security",
      "title": "SQL Injection Vulnerability",
      "description": "User input is directly used in SQL query without sanitization",
      "location": {
        "file": "db.py",
        "line_start": 42,
        "line_end": 42
      },
      "explanation": "This code concatenates user input directly into a SQL query, allowing attackers to inject malicious SQL...",
      "evidence": {
        "type": "ai",
        "confidence": 0.95,
        "source": "security_agent"
      },
      "fix": {
        "description": "Use parameterized queries to prevent SQL injection",
        "before": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
        "after": "query = \"SELECT * FROM users WHERE id = ?\"\ncursor.execute(query, (user_id,))",
        "diff": "@@ -42 +42,2 @@\n-query = f\"SELECT * FROM users WHERE id = {user_id}\"\n+query = \"SELECT * FROM users WHERE id = ?\"\n+cursor.execute(query, (user_id,))"
      }
    }
  ],
  "github_comment": "## CodeVerify Analysis\\n\\n### Summary\\n..."
}"""


class SynthesisAgent(BaseAgent):
    """
    Agent for synthesizing analysis results and generating fixes.

    This agent combines results from semantic analysis, formal verification,
    and security scanning to produce a coherent report with actionable fixes.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize synthesis agent."""
        super().__init__(config)
        # Use GPT-4.1 or similar for clear explanations
        if config is None:
            self.config.openai_model = "gpt-4-turbo-preview"

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Synthesize analysis results.

        Args:
            code: The original code
            context: Context including:
                - semantic_results: Results from semantic analysis
                - verification_results: Results from formal verification
                - security_results: Results from security analysis
                - pr_info: PR metadata

        Returns:
            AgentResult with synthesized findings and fixes
        """
        return await self.synthesize(
            code=code,
            semantic_results=context.get("semantic_results", {}),
            verification_results=context.get("verification_results", {}),
            security_results=context.get("security_results", {}),
            pr_info=context.get("pr_info", {}),
        )

    async def synthesize(
        self,
        code: str,
        semantic_results: dict[str, Any],
        verification_results: dict[str, Any],
        security_results: dict[str, Any],
        pr_info: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Synthesize all analysis results into a coherent report.

        Args:
            code: The analyzed code
            semantic_results: Results from semantic analysis agent
            verification_results: Results from Z3 verifier
            security_results: Results from security agent
            pr_info: Optional PR metadata

        Returns:
            AgentResult with consolidated findings and generated fixes
        """
        start_time = time.time()

        user_prompt = self._build_prompt(
            code, semantic_results, verification_results, security_results, pr_info
        )

        try:
            response = await self._call_llm(
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_mode=True,
            )

            # Use common JSON parsing helper
            parsed = self._parse_json_response(response)
            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                "Synthesis completed",
                total_findings=parsed.data.get("summary", {}).get("total_issues", 0),
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
            logger.error("Synthesis failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _build_prompt(
        self,
        code: str,
        semantic_results: dict[str, Any],
        verification_results: dict[str, Any],
        security_results: dict[str, Any],
        pr_info: dict[str, Any] | None,
    ) -> str:
        """Build the synthesis prompt."""
        prompt_parts = [
            "Synthesize the following analysis results and generate a consolidated report with fixes.",
            "",
            "## Original Code",
            "```",
            code[:5000] if len(code) > 5000 else code,  # Limit code length
            "```",
            "",
        ]

        if pr_info:
            prompt_parts.extend(
                [
                    "## PR Information",
                    f"- Title: {pr_info.get('title', 'N/A')}",
                    f"- Files Changed: {pr_info.get('files_changed', 'N/A')}",
                    "",
                ]
            )

        if semantic_results:
            prompt_parts.extend(
                [
                    "## Semantic Analysis Results",
                    "```json",
                    json.dumps(semantic_results, indent=2)[:3000],
                    "```",
                    "",
                ]
            )

        if verification_results:
            prompt_parts.extend(
                [
                    "## Formal Verification Results",
                    "```json",
                    json.dumps(verification_results, indent=2)[:3000],
                    "```",
                    "",
                ]
            )

        if security_results:
            prompt_parts.extend(
                [
                    "## Security Analysis Results",
                    "```json",
                    json.dumps(security_results, indent=2)[:3000],
                    "```",
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "## Instructions",
                "1. Consolidate all findings, removing duplicates",
                "2. Prioritize by severity (critical > high > medium > low)",
                "3. Generate concrete code fixes for each issue",
                "4. Create a GitHub-compatible summary comment",
                "5. Determine if this PR should pass or fail based on findings",
            ]
        )

        return "\n".join(prompt_parts)

    def generate_github_comment(self, synthesis_result: dict[str, Any]) -> str:
        """
        Generate a GitHub PR comment from synthesis results.

        Args:
            synthesis_result: The synthesized analysis result

        Returns:
            Markdown-formatted GitHub comment
        """
        summary = synthesis_result.get("summary", {})
        findings = synthesis_result.get("findings", [])

        lines = [
            "## 🔍 CodeVerify Analysis",
            "",
            "### Summary",
            "",
            f"| Total Issues | Critical | High | Medium | Low |",
            f"|:---:|:---:|:---:|:---:|:---:|",
            f"| {summary.get('total_issues', 0)} | {summary.get('critical', 0)} | "
            f"{summary.get('high', 0)} | {summary.get('medium', 0)} | {summary.get('low', 0)} |",
            "",
        ]

        if summary.get("pass", True):
            lines.append("✅ **Status: Passed**")
        else:
            lines.append("❌ **Status: Issues Found**")

        if summary.get("recommendation"):
            lines.extend(["", f"**Recommendation:** {summary['recommendation']}"])

        if findings:
            lines.extend(["", "### Findings", ""])

            for finding in findings[:10]:  # Limit to 10 findings in comment
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🔵",
                }.get(finding.get("severity", "low"), "⚪")

                lines.extend(
                    [
                        f"<details>",
                        f"<summary>{severity_emoji} <b>{finding.get('title', 'Issue')}</b> "
                        f"({finding.get('location', {}).get('file', '')}:"
                        f"{finding.get('location', {}).get('line_start', '')})</summary>",
                        "",
                        finding.get("description", ""),
                        "",
                    ]
                )

                if fix := finding.get("fix"):
                    lines.extend(
                        [
                            "**Suggested Fix:**",
                            "```suggestion",
                            fix.get("after", ""),
                            "```",
                            "",
                        ]
                    )

                lines.extend(["</details>", ""])

        lines.extend(
            [
                "---",
                "*Powered by [CodeVerify](https://codeverify.dev) - "
                "AI-powered code review with formal verification*",
            ]
        )

        return "\n".join(lines)
