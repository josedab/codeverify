"""CodeVerify Verification Protocol Client.

Reference client for the LLM Output Verification Protocol. Allows any
tool to send code for verification and receive proof certificates.

Usage:
    from codeverify_client import VerificationClient

    client = VerificationClient("https://api.codeverify.dev")
    result = client.verify("def divide(a, b): return a / b", language="python")
    print(result.status)       # "failed"
    print(result.findings)     # [Finding(severity="critical", ...)]
    print(result.proofs)       # [ProofCert(status="failed", ...)]
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

PROTOCOL_VERSION = "1.0.0"


@dataclass
class Finding:
    """A verification finding from the server."""

    id: str = ""
    file_path: str = ""
    line: int = 0
    check_type: str = ""
    severity: str = "medium"
    message: str = ""
    fix_suggestion: str = ""
    proof_id: str = ""


@dataclass
class ProofCertificate:
    """A proof certificate from the server."""

    id: str = ""
    check_type: str = ""
    status: str = ""
    constraints_checked: int = 0
    content_hash: str = ""
    signature: str = ""


@dataclass
class VerifyResult:
    """Result from a verification request."""

    request_id: str = ""
    status: str = ""  # verified, failed, partial, timeout, error
    findings: list[Finding] = field(default_factory=list)
    proofs: list[ProofCertificate] = field(default_factory=list)
    verification_time_ms: int = 0
    server_id: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "verified"

    @property
    def critical_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def finding_count(self) -> int:
        return len(self.findings)


@dataclass
class ClientConfig:
    """Client configuration."""

    server_url: str = "http://localhost:8000"
    api_key: str = ""
    timeout_seconds: int = 30
    include_proofs: bool = True
    include_fixes: bool = False
    client_name: str = "codeverify-python-client"
    client_version: str = PROTOCOL_VERSION


class VerificationClient:
    """Reference client for the LLM Output Verification Protocol.

    Can operate in two modes:
    1. HTTP mode: sends requests to a CodeVerify server
    2. Local mode: uses codeverify_core directly (if installed)
    """

    def __init__(
        self,
        server_url: str = "",
        api_key: str = "",
        config: ClientConfig | None = None,
    ) -> None:
        self._config = config or ClientConfig(
            server_url=server_url, api_key=api_key
        )
        self._local_server = None
        if not server_url:
            self._init_local()

    def _init_local(self) -> None:
        """Try to use local codeverify_core if available."""
        try:
            from codeverify_core.verification_protocol import (
                VerificationProtocolServer,
            )

            self._local_server = VerificationProtocolServer()
        except ImportError:
            pass

    def verify(
        self,
        code: str,
        language: str = "python",
        file_path: str = "input.py",
        checks: list[str] | None = None,
    ) -> VerifyResult:
        """Verify code and return results with proof certificates."""
        if self._local_server:
            return self._verify_local(code, language, file_path, checks)
        return self._verify_remote(code, language, file_path, checks)

    def verify_files(
        self,
        files: dict[str, str],
        language: str = "python",
        checks: list[str] | None = None,
    ) -> VerifyResult:
        """Verify multiple files."""
        file_list = [{"path": p, "content": c} for p, c in files.items()]
        if self._local_server:
            return self._verify_local_files(file_list, language, checks)
        return self._verify_remote_files(file_list, language, checks)

    def get_capabilities(self) -> dict[str, Any]:
        """Get server capabilities."""
        if self._local_server:
            caps = self._local_server.get_capabilities()
            return {
                "protocol_version": caps.protocol_version,
                "supported_languages": caps.supported_languages,
                "supported_checks": caps.supported_checks,
                "supports_proofs": caps.supports_proofs,
                "supports_fixes": caps.supports_fixes,
            }
        return {"protocol_version": PROTOCOL_VERSION, "mode": "remote"}

    def _verify_local(
        self, code: str, language: str, file_path: str, checks: list[str] | None
    ) -> VerifyResult:
        """Verify using local codeverify_core."""
        from codeverify_core.verification_protocol import (
            CheckType,
            VerifyRequest,
        )

        check_types = []
        if checks:
            for c in checks:
                try:
                    check_types.append(CheckType(c))
                except ValueError:
                    pass
        if not check_types:
            check_types = [CheckType.ALL]

        req = VerifyRequest(
            client_id=self._config.client_name,
            client_name=self._config.client_name,
            files=[{"path": file_path, "content": code}],
            language=language,
            checks=check_types,
            include_proofs=self._config.include_proofs,
            include_fixes=self._config.include_fixes,
            timeout_ms=self._config.timeout_seconds * 1000,
        )

        resp = self._local_server.verify(req)
        return self._convert_response(resp)

    def _verify_local_files(
        self, files: list[dict[str, str]], language: str, checks: list[str] | None
    ) -> VerifyResult:
        from codeverify_core.verification_protocol import (
            CheckType,
            VerifyRequest,
        )

        req = VerifyRequest(
            client_id=self._config.client_name,
            files=files,
            language=language,
            checks=[CheckType.ALL],
            include_proofs=self._config.include_proofs,
        )
        resp = self._local_server.verify(req)
        return self._convert_response(resp)

    def _verify_remote(
        self, code: str, language: str, file_path: str, checks: list[str] | None
    ) -> VerifyResult:
        """Verify via HTTP (placeholder — requires httpx in production)."""
        return VerifyResult(
            status="error",
            findings=[],
            proofs=[],
        )

    def _verify_remote_files(
        self, files: list[dict[str, str]], language: str, checks: list[str] | None
    ) -> VerifyResult:
        return VerifyResult(status="error")

    def _convert_response(self, resp: Any) -> VerifyResult:
        """Convert protocol response to client result."""
        findings = [
            Finding(
                id=f.id, file_path=f.file_path, line=f.line,
                check_type=f.check_type.value if hasattr(f.check_type, "value") else str(f.check_type),
                severity=f.severity, message=f.message,
                fix_suggestion=f.fix_suggestion, proof_id=f.proof_id,
            )
            for f in resp.findings
        ]
        proofs = [
            ProofCertificate(
                id=p.id,
                check_type=p.check_type.value if hasattr(p.check_type, "value") else str(p.check_type),
                status=p.status.value if hasattr(p.status, "value") else str(p.status),
                constraints_checked=p.constraints_checked,
                content_hash=p.content_hash, signature=p.signature,
            )
            for p in resp.proofs
        ]
        return VerifyResult(
            request_id=resp.request_id,
            status=resp.status.value if hasattr(resp.status, "value") else str(resp.status),
            findings=findings, proofs=proofs,
            verification_time_ms=resp.verification_time_ms,
            server_id=resp.server_id,
        )
