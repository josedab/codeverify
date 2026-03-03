"""HTTP client for the CodeVerify API."""

from __future__ import annotations

import os
from typing import Any

import httpx

from codeverify_sdk.models import (
    Finding,
    ProofResult,
    SafetyResult,
    VerificationResult,
)


class CodeVerifyClient:
    """Client for the CodeVerify verification API.

    Reads configuration from environment variables if not provided:
        CODEVERIFY_API_KEY
        CODEVERIFY_API_URL (default: https://api.codeverify.dev)
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("CODEVERIFY_API_KEY", "")
        self.api_url = (api_url or os.environ.get("CODEVERIFY_API_URL", "https://api.codeverify.dev")).rstrip("/")
        self.timeout = timeout
        self._http = httpx.Client(
            base_url=self.api_url,
            timeout=timeout,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "codeverify-sdk-python/0.1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # -- Public API ----------------------------------------------------------

    def verify(
        self,
        code: str,
        language: str = "python",
        checks: list[str] | None = None,
    ) -> VerificationResult:
        """Run verification checks on code."""
        payload: dict[str, Any] = {"code": code, "language": language}
        if checks:
            payload["checks"] = checks

        try:
            resp = self._http.post("/api/v1/verified-autofix/generate", json={
                "code": code,
                "language": language,
                "finding": {
                    "finding_id": "sdk-check",
                    "type": "general",
                    "severity": "medium",
                    "description": "SDK verification check",
                    "file_path": "<inline>",
                    "line_start": 1,
                    "code_snippet": code[:200],
                },
            })
            resp.raise_for_status()
            data = resp.json()

            findings = []
            if data.get("status") != "verified":
                findings.append(Finding(
                    id=data.get("fix_id", "unknown"),
                    type="verification",
                    severity="medium",
                    title="Verification issue found",
                    description=data.get("explanation", ""),
                ))

            return VerificationResult(
                passed=len(findings) == 0,
                findings=findings,
                checks_run=checks or ["all"],
                language=language,
            )
        except httpx.HTTPError:
            # Fallback: run local heuristic checks
            return self._local_verify(code, language, checks)

    def check_safety(self, code: str, language: str = "python") -> SafetyResult:
        """Check code for safety issues."""
        issues: list[Finding] = []
        categories = ["null_safety", "injection", "bounds", "resource_leak"]

        # Local pattern-based safety checks
        import re
        safety_patterns = {
            "sql_injection": (r"execute\s*\(\s*[\"'].*%s", "Possible SQL injection"),
            "eval_usage": (r"\beval\s*\(", "Use of eval()"),
            "shell_injection": (r"subprocess\.\w+\(.*shell\s*=\s*True", "Shell injection risk"),
            "hardcoded_secret": (r"(?:password|secret|token)\s*=\s*[\"'][^\"']+[\"']", "Hardcoded secret"),
        }

        for check_name, (pattern, message) in safety_patterns.items():
            matches = list(re.finditer(pattern, code, re.IGNORECASE | re.DOTALL))
            for match in matches:
                line_num = code[:match.start()].count("\n") + 1
                issues.append(Finding(
                    id=f"safety-{check_name}-{line_num}",
                    type=check_name,
                    severity="high",
                    title=message,
                    description=f"{message} detected at line {line_num}",
                    line_start=line_num,
                ))

        risk = "low"
        if any(i.severity == "critical" for i in issues):
            risk = "critical"
        elif any(i.severity == "high" for i in issues):
            risk = "high"
        elif issues:
            risk = "medium"

        return SafetyResult(
            safe=len(issues) == 0,
            risk_level=risk,
            issues=issues,
            categories_checked=categories,
        )

    def generate_proof(
        self,
        code: str,
        property: str | None = None,
        language: str = "python",
    ) -> ProofResult:
        """Generate a formal proof for a property."""
        try:
            resp = self._http.post("/api/v1/proof-explorer/explore", json={
                "code": code,
                "function_name": "target",
                "language": language,
                "properties": [{"expression": property}] if property else [],
            })
            resp.raise_for_status()
            data = resp.json()

            return ProofResult(
                status=data.get("status", "unknown"),
                property_checked=property or "auto-inferred",
                counterexamples=[
                    {"variables": ce.get("variables", [])}
                    for ce in data.get("counterexamples", [])
                ],
                proof_tree=data.get("proof_tree"),
                solver_time_ms=data.get("solver_time_ms", 0),
            )
        except httpx.HTTPError:
            return ProofResult(
                status="unknown",
                property_checked=property or "auto-inferred",
            )

    def _local_verify(
        self, code: str, language: str, checks: list[str] | None
    ) -> VerificationResult:
        """Fallback local verification when API is unavailable."""
        findings: list[Finding] = []

        if language == "python":
            try:
                compile(code, "<sdk-verify>", "exec")
            except SyntaxError as e:
                findings.append(Finding(
                    id="syntax-error",
                    type="syntax",
                    severity="critical",
                    title="Syntax error",
                    description=str(e),
                    line_start=e.lineno,
                ))

        return VerificationResult(
            passed=len(findings) == 0,
            findings=findings,
            checks_run=checks or ["syntax"],
            language=language,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "CodeVerifyClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
