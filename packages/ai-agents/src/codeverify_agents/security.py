"""Security Analysis Agent - Identifies security vulnerabilities."""

import time
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent, CodeContext

logger = structlog.get_logger()


SECURITY_SYSTEM_PROMPT = """You are an expert application security engineer specializing in secure code review. Your task is to analyze code for security vulnerabilities, with particular attention to patterns common in AI-generated code.

Analyze for:

1. **OWASP Top 10** vulnerabilities:
   - Injection (SQL, Command, XSS, etc.)
   - Broken Authentication
   - Sensitive Data Exposure
   - XML External Entities (XXE)
   - Broken Access Control
   - Security Misconfiguration
   - Cross-Site Scripting (XSS)
   - Insecure Deserialization
   - Using Components with Known Vulnerabilities
   - Insufficient Logging & Monitoring

2. **AI-Specific Patterns**:
   - Prompt injection vulnerabilities
   - Insecure default configurations
   - Missing input validation
   - Overly permissive error handling
   - Hardcoded secrets or credentials
   - Insecure randomness
   - Missing authentication/authorization checks

3. **Language-Specific Issues**:
   - Python: pickle deserialization, eval/exec usage, path traversal
   - JavaScript/TypeScript: prototype pollution, unsafe DOM operations, eval usage
   - SQL: injection through string concatenation

For each vulnerability found, provide:
- Severity (critical, high, medium, low)
- CWE ID if applicable
- Clear description of the issue
- Exact location in code
- Recommended fix

Respond in JSON format:
{
  "vulnerabilities": [
    {
      "id": "vuln_1",
      "severity": "high",
      "category": "injection",
      "cwe_id": "CWE-89",
      "title": "SQL Injection vulnerability",
      "description": "User input is directly concatenated into SQL query",
      "location": {"file": "db.py", "line": 42, "column": 10},
      "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
      "fix_suggestion": "Use parameterized queries instead",
      "fix_code": "cursor.execute(\"SELECT * FROM users WHERE id = ?\", (user_id,))",
      "confidence": 0.95
    }
  ],
  "secrets_detected": [
    {
      "type": "api_key",
      "location": {"line": 10},
      "pattern": "sk-..."
    }
  ],
  "security_score": 65,
  "summary": "Found 2 high severity issues requiring immediate attention"
}"""


class SecurityAgent(BaseAgent):
    """
    Agent for security vulnerability analysis.

    This agent specializes in finding security issues,
    particularly patterns common in AI-generated code.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize security agent."""
        super().__init__(config)
        # Prefer Claude for security reasoning
        if config is None:
            self.config.provider = "anthropic"
            self.config.anthropic_model = "claude-3-sonnet-20240229"

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Analyze code for security vulnerabilities.

        Args:
            code: The code to analyze
            context: Additional context including:
                - file_path: Path to the file
                - language: Programming language
                - framework: Framework being used (e.g., "fastapi", "express")
                - is_ai_generated: Whether code is AI-generated

        Returns:
            AgentResult with security findings
        """
        start_time = time.time()

        # Use CodeContext for clean parameter handling
        ctx = CodeContext.from_dict(code, context)
        user_prompt = self._build_prompt(ctx)

        try:
            response = await self._call_llm(
                system_prompt=SECURITY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_mode=True,
            )

            # Use common JSON parsing helper
            parsed = self._parse_json_response(response)
            if parsed.parse_error:
                parsed.data["vulnerabilities"] = []

            elapsed_ms = (time.time() - start_time) * 1000
            vuln_count = len(parsed.data.get("vulnerabilities", []))

            logger.info(
                "Security analysis completed",
                file_path=ctx.file_path,
                vulnerabilities_found=vuln_count,
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
            logger.error("Security analysis failed", error=str(e), file_path=ctx.file_path)
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _build_prompt(self, ctx: CodeContext) -> str:
        """Build the security analysis prompt using CodeContext."""
        prompt_parts = [
            f"Perform a security analysis on this {ctx.language} code from `{ctx.file_path}`:",
        ]

        if ctx.framework:
            prompt_parts.append(f"Framework: {ctx.framework}")

        if ctx.is_ai_generated:
            prompt_parts.append(
                "Note: This code may be AI-generated. Pay special attention to "
                "common AI code generation security pitfalls."
            )

        prompt_parts.extend([
            "",
            self._build_code_block(ctx.code, ctx.language),
            "",
            "Identify all security vulnerabilities, secrets, and security concerns.",
        ])

        return "\n".join(prompt_parts)

    async def scan_for_secrets(self, code: str) -> list[dict[str, Any]]:
        """
        Quick scan for hardcoded secrets using patterns.

        This is a fast, pattern-based scan that doesn't require LLM calls.
        """
        import re

        secrets = []

        # Common secret patterns
        patterns = [
            (r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]([^'\"]+)['\"]", "api_key"),
            (r"(?i)(secret[_-]?key|secretkey)\s*[=:]\s*['\"]([^'\"]+)['\"]", "secret_key"),
            (r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]([^'\"]+)['\"]", "password"),
            (r"(?i)(token|auth[_-]?token)\s*[=:]\s*['\"]([^'\"]+)['\"]", "token"),
            (r"sk-[a-zA-Z0-9]{20,}", "openai_key"),
            (r"ghp_[a-zA-Z0-9]{36}", "github_token"),
            (r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", "private_key"),
            (r"(?i)aws[_-]?access[_-]?key[_-]?id\s*[=:]\s*['\"]?([A-Z0-9]{20})", "aws_key"),
        ]

        lines = code.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern, secret_type in patterns:
                if re.search(pattern, line):
                    secrets.append(
                        {
                            "type": secret_type,
                            "line": line_num,
                            "severity": "critical" if secret_type == "private_key" else "high",
                        }
                    )

        return secrets
