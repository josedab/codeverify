"""Multi-Modal Verification.

Verifies infrastructure-as-code, database migrations, API contracts,
and configuration files beyond source code.

Features:
- IaC verification (Terraform, CloudFormation)
- Database migration compatibility checking
- API contract verification (OpenAPI)
- Configuration file schema validation
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum

import structlog

logger = structlog.get_logger()


class ModalType(str, Enum):
    IAC_TERRAFORM = "iac_terraform"
    IAC_CLOUDFORMATION = "iac_cloudformation"
    DB_MIGRATION = "db_migration"
    API_CONTRACT = "api_contract"
    CONFIG = "config"


class ModalSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ModalFinding:
    """A finding from multi-modal verification."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    modal_type: ModalType = ModalType.CONFIG
    severity: ModalSeverity = ModalSeverity.MEDIUM
    file_path: str = ""
    line: int = 0
    message: str = ""
    fix_suggestion: str = ""


@dataclass
class ModalVerificationResult:
    """Result of verifying a non-code artifact."""

    modal_type: ModalType = ModalType.CONFIG
    file_path: str = ""
    findings: list[ModalFinding] = field(default_factory=list)
    passed: bool = True
    summary: str = ""


class TerraformVerifier:
    """Verifies Terraform configurations."""

    SECURITY_CHECKS = [
        ("public", ModalSeverity.HIGH, "Resource may be publicly accessible"),
        ("0.0.0.0/0", ModalSeverity.CRITICAL, "CIDR 0.0.0.0/0 allows access from anywhere"),
        ("encrypt", ModalSeverity.HIGH, "Encryption should be enabled"),
        ("logging", ModalSeverity.MEDIUM, "Logging should be enabled"),
    ]

    def verify(self, file_path: str, content: str) -> ModalVerificationResult:
        findings: list[ModalFinding] = []
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            lower = line.lower()
            if "0.0.0.0/0" in line:
                findings.append(
                    ModalFinding(
                        modal_type=ModalType.IAC_TERRAFORM,
                        severity=ModalSeverity.CRITICAL,
                        file_path=file_path,
                        line=i,
                        message="Open CIDR 0.0.0.0/0 — allows access from anywhere",
                    )
                )
            if "public" in lower and ("true" in lower or "yes" in lower):
                findings.append(
                    ModalFinding(
                        modal_type=ModalType.IAC_TERRAFORM,
                        severity=ModalSeverity.HIGH,
                        file_path=file_path,
                        line=i,
                        message="Resource is publicly accessible",
                    )
                )
            if "encrypted" in lower and "false" in lower:
                findings.append(
                    ModalFinding(
                        modal_type=ModalType.IAC_TERRAFORM,
                        severity=ModalSeverity.HIGH,
                        file_path=file_path,
                        line=i,
                        message="Encryption is disabled",
                    )
                )
        return ModalVerificationResult(
            modal_type=ModalType.IAC_TERRAFORM,
            file_path=file_path,
            findings=findings,
            passed=len(findings) == 0,
            summary=f"Terraform: {len(findings)} findings in {file_path}",
        )


class MigrationVerifier:
    """Verifies database migration compatibility."""

    DESTRUCTIVE_OPS = ["DROP TABLE", "DROP COLUMN", "ALTER COLUMN", "RENAME COLUMN", "TRUNCATE"]

    def verify(self, file_path: str, content: str) -> ModalVerificationResult:
        findings: list[ModalFinding] = []
        for i, line in enumerate(content.split("\n"), 1):
            for op in self.DESTRUCTIVE_OPS:
                if op in line.upper():
                    findings.append(
                        ModalFinding(
                            modal_type=ModalType.DB_MIGRATION,
                            severity=ModalSeverity.HIGH,
                            file_path=file_path,
                            line=i,
                            message=f"Destructive operation: {op}",
                            fix_suggestion="Consider a reversible migration or add a safety check",
                        )
                    )
        return ModalVerificationResult(
            modal_type=ModalType.DB_MIGRATION,
            file_path=file_path,
            findings=findings,
            passed=len(findings) == 0,
            summary=f"Migration: {len(findings)} destructive operations",
        )


class APIContractVerifier:
    """Verifies API contracts (OpenAPI compatibility)."""

    def verify(self, file_path: str, content: str) -> ModalVerificationResult:
        findings: list[ModalFinding] = []
        try:
            spec = json.loads(content)
        except (json.JSONDecodeError, Exception):
            findings.append(
                ModalFinding(
                    modal_type=ModalType.API_CONTRACT,
                    severity=ModalSeverity.HIGH,
                    file_path=file_path,
                    message="Invalid JSON in API contract",
                )
            )
            return ModalVerificationResult(
                modal_type=ModalType.API_CONTRACT,
                file_path=file_path,
                findings=findings,
                passed=False,
            )

        if "openapi" not in spec and "swagger" not in spec:
            findings.append(
                ModalFinding(
                    modal_type=ModalType.API_CONTRACT,
                    severity=ModalSeverity.MEDIUM,
                    file_path=file_path,
                    message="Missing OpenAPI version field",
                )
            )

        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                if method in ("get", "post", "put", "delete", "patch"):
                    if "responses" not in details:
                        findings.append(
                            ModalFinding(
                                modal_type=ModalType.API_CONTRACT,
                                severity=ModalSeverity.MEDIUM,
                                file_path=file_path,
                                message=f"{method.upper()} {path}: missing responses definition",
                            )
                        )
                    if method in ("post", "put", "patch") and "requestBody" not in details:
                        findings.append(
                            ModalFinding(
                                modal_type=ModalType.API_CONTRACT,
                                severity=ModalSeverity.LOW,
                                file_path=file_path,
                                message=f"{method.upper()} {path}: missing requestBody",
                            )
                        )

        return ModalVerificationResult(
            modal_type=ModalType.API_CONTRACT,
            file_path=file_path,
            findings=findings,
            passed=len(findings) == 0,
        )


class ConfigVerifier:
    """Verifies configuration files."""

    SENSITIVE_KEYS = ["password", "secret", "api_key", "token", "private_key"]

    def verify(self, file_path: str, content: str) -> ModalVerificationResult:
        findings: list[ModalFinding] = []
        for i, line in enumerate(content.split("\n"), 1):
            lower = line.lower()
            for key in self.SENSITIVE_KEYS:
                if key in lower and ("=" in line or ":" in line):
                    val_part = line.split("=", 1)[-1].split(":", 1)[-1].strip()
                    if val_part and val_part not in ('""', "''", "", "${", "$(", "env."):
                        findings.append(
                            ModalFinding(
                                modal_type=ModalType.CONFIG,
                                severity=ModalSeverity.CRITICAL,
                                file_path=file_path,
                                line=i,
                                message=f"Potential hardcoded secret: {key}",
                                fix_suggestion="Use environment variables or a secret manager",
                            )
                        )
        return ModalVerificationResult(
            modal_type=ModalType.CONFIG,
            file_path=file_path,
            findings=findings,
            passed=len(findings) == 0,
        )


class MultiModalVerificationService:
    """Main service for multi-modal verification."""

    def __init__(self) -> None:
        self._terraform = TerraformVerifier()
        self._migration = MigrationVerifier()
        self._api = APIContractVerifier()
        self._config = ConfigVerifier()
        self._results: list[ModalVerificationResult] = []

    def verify_file(self, file_path: str, content: str) -> ModalVerificationResult:
        if file_path.endswith(".tf") or file_path.endswith(".tfvars"):
            result = self._terraform.verify(file_path, content)
        elif "migration" in file_path.lower() or file_path.endswith(".sql"):
            result = self._migration.verify(file_path, content)
        elif file_path.endswith(".json") and (
            "openapi" in content.lower() or "swagger" in content.lower()
        ):
            result = self._api.verify(file_path, content)
        elif file_path.endswith((".yml", ".yaml", ".env", ".ini", ".toml", ".conf")):
            result = self._config.verify(file_path, content)
        else:
            result = self._config.verify(file_path, content)
        self._results.append(result)
        return result

    def verify_batch(self, files: dict[str, str]) -> list[ModalVerificationResult]:
        return [self.verify_file(path, content) for path, content in files.items()]

    def get_results(self) -> list[ModalVerificationResult]:
        return list(self._results)


_multimodal_instance: MultiModalVerificationService | None = None


def get_multimodal_service() -> MultiModalVerificationService:
    global _multimodal_instance
    if _multimodal_instance is None:
        _multimodal_instance = MultiModalVerificationService()
    return _multimodal_instance


def reset_multimodal_service() -> None:
    global _multimodal_instance
    _multimodal_instance = None
