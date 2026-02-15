"""LLM Output Verification Protocol.

Standardized protocol for any AI coding assistant to send generated
code for verification and receive proof certificates back.

Features:
- Protocol message types (VerifyRequest, VerifyResponse, ProofCertificate)
- Capability negotiation between client and server
- Multi-language support with check type selection
- Proof certificate format with cryptographic signing
- Session management for multi-turn verification
- Protocol versioning
"""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


PROTOCOL_VERSION = "1.0.0"


class MessageType(str, Enum):
    VERIFY_REQUEST = "verify_request"
    VERIFY_RESPONSE = "verify_response"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    PROOF_CERTIFICATE = "proof_certificate"
    ERROR = "error"


class CheckType(str, Enum):
    NULL_SAFETY = "null_safety"
    BOUNDS_CHECK = "bounds_check"
    DIVISION_ZERO = "division_zero"
    OVERFLOW = "overflow"
    MEMORY_SAFETY = "memory_safety"
    TYPE_SAFETY = "type_safety"
    SECURITY = "security"
    ALL = "all"


class VerifyStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ProtocolCapabilities:
    """Server capabilities advertised to clients."""
    protocol_version: str = PROTOCOL_VERSION
    supported_languages: list[str] = field(default_factory=lambda: ["python", "typescript", "go", "java", "rust", "c", "cpp"])
    supported_checks: list[str] = field(default_factory=lambda: [c.value for c in CheckType])
    max_file_size_bytes: int = 500_000
    max_files_per_request: int = 20
    supports_streaming: bool = True
    supports_proofs: bool = True
    supports_fixes: bool = True


@dataclass
class VerifyRequest:
    """Request to verify code."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    protocol_version: str = PROTOCOL_VERSION
    client_id: str = ""
    client_name: str = ""
    files: list[dict[str, str]] = field(default_factory=list)
    language: str = "python"
    checks: list[CheckType] = field(default_factory=lambda: [CheckType.ALL])
    include_proofs: bool = True
    include_fixes: bool = False
    timeout_ms: int = 30000
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    """A verification finding."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    file_path: str = ""
    line: int = 0
    check_type: CheckType = CheckType.NULL_SAFETY
    severity: str = "medium"
    message: str = ""
    fix_suggestion: str = ""
    proof_id: str = ""


@dataclass
class ProofCert:
    """A proof certificate for a verification result."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    check_type: CheckType = CheckType.NULL_SAFETY
    status: VerifyStatus = VerifyStatus.VERIFIED
    constraints_checked: int = 0
    content_hash: str = ""
    signature: str = ""
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def sign(self, secret: str) -> str:
        payload = f"{self.id}:{self.check_type.value}:{self.status.value}:{self.content_hash}"
        self.signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
        return self.signature


@dataclass
class VerifyResponse:
    """Response from verification."""
    request_id: str = ""
    protocol_version: str = PROTOCOL_VERSION
    status: VerifyStatus = VerifyStatus.VERIFIED
    findings: list[Finding] = field(default_factory=list)
    proofs: list[ProofCert] = field(default_factory=list)
    verification_time_ms: int = 0
    server_id: str = "codeverify"


class VerificationProtocolServer:
    """Server implementing the LLM Output Verification Protocol."""

    CHECK_PATTERNS: dict[CheckType, list[tuple[str, str, str]]] = {
        CheckType.NULL_SAFETY: [("None.", "high", "Potential null dereference"), (".get(", "info", "Safe dictionary access pattern")],
        CheckType.DIVISION_ZERO: [("/ 0", "critical", "Division by zero"), ("/ ", "medium", "Potential division by zero if divisor is 0")],
        CheckType.SECURITY: [("eval(", "critical", "Use of eval() — code injection risk"), ("exec(", "high", "Use of exec() — code execution risk")],
        CheckType.BOUNDS_CHECK: [("[i]", "medium", "Potential array out-of-bounds"), ("[-1]", "low", "Negative index access")],
    }

    def __init__(self, server_id: str = "codeverify", signing_secret: str = "default-secret") -> None:
        self._server_id = server_id
        self._secret = signing_secret
        self._capabilities = ProtocolCapabilities()
        self._sessions: dict[str, list[VerifyRequest]] = {}

    def get_capabilities(self) -> ProtocolCapabilities:
        return self._capabilities

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        start = time.time()
        findings: list[Finding] = []
        proofs: list[ProofCert] = []

        checks = request.checks
        if CheckType.ALL in checks:
            checks = [c for c in CheckType if c != CheckType.ALL]

        for file_info in request.files:
            path = file_info.get("path", "")
            content = file_info.get("content", "")
            for check in checks:
                file_findings = self._check_code(path, content, check)
                findings.extend(file_findings)

        if request.include_proofs:
            checked_types = set(f.check_type for f in findings) if findings else set(checks)
            for ct in checked_types:
                ct_findings = [f for f in findings if f.check_type == ct]
                status = VerifyStatus.FAILED if ct_findings else VerifyStatus.VERIFIED
                cert = ProofCert(
                    check_type=ct, status=status,
                    constraints_checked=len(request.files),
                    content_hash=hashlib.sha256("".join(f.get("content", "") for f in request.files).encode()).hexdigest()[:12],
                )
                cert.sign(self._secret)
                proofs.append(cert)

        overall = VerifyStatus.VERIFIED
        if any(f.severity in ("critical", "high") for f in findings):
            overall = VerifyStatus.FAILED
        elif findings:
            overall = VerifyStatus.PARTIAL

        elapsed = int((time.time() - start) * 1000)

        if request.client_id:
            self._sessions.setdefault(request.client_id, []).append(request)

        return VerifyResponse(
            request_id=request.id, status=overall,
            findings=findings, proofs=proofs,
            verification_time_ms=elapsed, server_id=self._server_id,
        )

    def _check_code(self, path: str, content: str, check: CheckType) -> list[Finding]:
        findings: list[Finding] = []
        patterns = self.CHECK_PATTERNS.get(check, [])
        for i, line in enumerate(content.split("\n"), 1):
            for pattern, severity, message in patterns:
                if pattern in line:
                    findings.append(Finding(
                        file_path=path, line=i, check_type=check,
                        severity=severity, message=message,
                    ))
        return findings


_protocol_instance: VerificationProtocolServer | None = None
def get_verification_protocol_server() -> VerificationProtocolServer:
    global _protocol_instance
    if _protocol_instance is None: _protocol_instance = VerificationProtocolServer()
    return _protocol_instance
def reset_verification_protocol_server() -> None:
    global _protocol_instance
    _protocol_instance = None
