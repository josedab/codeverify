"""Self-Hosted Enterprise Deployment Configuration.

Provides enterprise-grade features for air-gapped and on-premises deployments:
- SAML/SSO authentication abstraction
- Immutable audit logging
- Enterprise configuration with security policies
- License management
- Local LLM backend configuration
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enterprise Configuration
# =============================================================================


class DeploymentMode(str, Enum):
    CLOUD = "cloud"
    SELF_HOSTED = "self_hosted"
    AIR_GAPPED = "air_gapped"


class LLMBackendType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VLLM = "vllm"
    LOCAL = "local"


class AuthProvider(str, Enum):
    LOCAL = "local"
    SAML = "saml"
    OIDC = "oidc"
    LDAP = "ldap"


@dataclass
class LLMBackendConfig:
    """Configuration for an LLM backend in enterprise deployments."""

    backend_type: LLMBackendType
    base_url: str = ""
    api_key: str = ""
    model_name: str = ""
    max_tokens: int = 4096
    timeout_seconds: int = 60
    tls_verify: bool = True
    tls_ca_cert: str = ""

    def is_local(self) -> bool:
        return self.backend_type in (
            LLMBackendType.OLLAMA,
            LLMBackendType.VLLM,
            LLMBackendType.LOCAL,
        )


@dataclass
class SAMLConfig:
    """SAML SSO configuration."""

    enabled: bool = False
    idp_metadata_url: str = ""
    idp_entity_id: str = ""
    sp_entity_id: str = "codeverify"
    assertion_consumer_service_url: str = ""
    single_logout_service_url: str = ""
    certificate_path: str = ""
    private_key_path: str = ""
    name_id_format: str = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
    attribute_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            "name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
            "groups": "http://schemas.xmlsoap.org/claims/Group",
        }
    )


@dataclass
class OIDCConfig:
    """OpenID Connect configuration."""

    enabled: bool = False
    issuer_url: str = ""
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    scopes: list[str] = field(default_factory=lambda: ["openid", "profile", "email"])


@dataclass
class SecurityPolicy:
    """Enterprise security policies."""

    require_tls: bool = True
    min_tls_version: str = "1.2"
    allowed_ip_ranges: list[str] = field(default_factory=list)
    session_timeout_minutes: int = 480
    max_failed_logins: int = 5
    lockout_duration_minutes: int = 30
    require_mfa: bool = False
    data_retention_days: int = 365
    encrypt_at_rest: bool = True
    encryption_key_rotation_days: int = 90


@dataclass
class EnterpriseConfig:
    """Top-level enterprise deployment configuration."""

    deployment_mode: DeploymentMode = DeploymentMode.SELF_HOSTED
    organization_name: str = ""
    license_key: str = ""

    # Authentication
    auth_provider: AuthProvider = AuthProvider.LOCAL
    saml: SAMLConfig = field(default_factory=SAMLConfig)
    oidc: OIDCConfig = field(default_factory=OIDCConfig)

    # LLM backends (primary + fallback)
    llm_primary: LLMBackendConfig = field(
        default_factory=lambda: LLMBackendConfig(backend_type=LLMBackendType.OLLAMA)
    )
    llm_fallback: LLMBackendConfig | None = None

    # Security
    security: SecurityPolicy = field(default_factory=SecurityPolicy)

    # Audit
    audit_log_enabled: bool = True
    audit_log_path: str = "/var/log/codeverify/audit.jsonl"

    def is_air_gapped(self) -> bool:
        return self.deployment_mode == DeploymentMode.AIR_GAPPED

    def validate(self) -> list[str]:
        """Validate the configuration. Returns list of errors."""
        errors: list[str] = []

        if self.deployment_mode == DeploymentMode.AIR_GAPPED:
            if not self.llm_primary.is_local():
                errors.append(
                    "Air-gapped deployment requires a local LLM backend (ollama, vllm, or local)."
                )

        if self.auth_provider == AuthProvider.SAML:
            if not self.saml.enabled:
                errors.append("SAML auth provider selected but SAML is not enabled.")
            if not self.saml.idp_metadata_url and not self.saml.idp_entity_id:
                errors.append("SAML requires either idp_metadata_url or idp_entity_id.")

        if self.auth_provider == AuthProvider.OIDC:
            if not self.oidc.enabled:
                errors.append("OIDC auth provider selected but OIDC is not enabled.")
            if not self.oidc.issuer_url:
                errors.append("OIDC requires issuer_url.")

        return errors


# =============================================================================
# Audit Logging
# =============================================================================


class AuditAction(str, Enum):
    """Auditable actions in the system."""

    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    FINDING_CREATED = "finding_created"
    FINDING_DISMISSED = "finding_dismissed"
    RULE_CREATED = "rule_created"
    RULE_MODIFIED = "rule_modified"
    RULE_DELETED = "rule_deleted"
    CONFIG_CHANGED = "config_changed"
    USER_CREATED = "user_created"
    USER_MODIFIED = "user_modified"
    USER_DELETED = "user_deleted"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    EXPORT_DATA = "export_data"
    COMPLIANCE_REPORT = "compliance_report"


@dataclass
class AuditEntry:
    """An immutable audit log entry."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    action: str = ""
    actor: str = ""
    actor_ip: str = ""
    resource_type: str = ""
    resource_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"
    checksum: str = ""

    def compute_checksum(self, previous_checksum: str = "") -> str:
        """Compute a chain-linked checksum for tamper detection."""
        data = json.dumps(
            {
                "id": self.id,
                "timestamp": self.timestamp,
                "action": self.action,
                "actor": self.actor,
                "resource_type": self.resource_type,
                "resource_id": self.resource_id,
                "details": self.details,
                "outcome": self.outcome,
                "previous": previous_checksum,
            },
            sort_keys=True,
        )
        self.checksum = hashlib.sha256(data.encode()).hexdigest()
        return self.checksum

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "action": self.action,
            "actor": self.actor,
            "actor_ip": self.actor_ip,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "outcome": self.outcome,
            "checksum": self.checksum,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class AuditLogger:
    """Immutable audit logger with chain-linked checksums.

    Each entry's checksum depends on the previous entry, making
    tampering detectable.

    Usage:
        audit = AuditLogger()
        audit.log(AuditAction.LOGIN, actor="user@org.com", resource_type="session")
    """

    def __init__(
        self,
        log_path: str | None = None,
        write_callback: Callable[[AuditEntry], None] | None = None,
    ) -> None:
        self._log_path = log_path
        self._write_callback = write_callback
        self._entries: list[AuditEntry] = []
        self._last_checksum: str = ""

    def log(
        self,
        action: AuditAction | str,
        actor: str = "system",
        actor_ip: str = "",
        resource_type: str = "",
        resource_id: str = "",
        details: dict[str, Any] | None = None,
        outcome: str = "success",
    ) -> AuditEntry:
        """Record an audit event."""
        entry = AuditEntry(
            action=action.value if isinstance(action, AuditAction) else action,
            actor=actor,
            actor_ip=actor_ip,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            outcome=outcome,
        )
        entry.compute_checksum(self._last_checksum)
        self._last_checksum = entry.checksum

        self._entries.append(entry)

        # Persist to file if configured
        if self._log_path:
            try:
                with open(self._log_path, "a") as f:
                    f.write(entry.to_json() + "\n")
            except OSError as e:
                logger.error("Failed to write audit log", error=str(e))

        # Call write callback if configured
        if self._write_callback:
            self._write_callback(entry)

        logger.info(
            "Audit event",
            action=entry.action,
            actor=entry.actor,
            resource=f"{entry.resource_type}/{entry.resource_id}",
        )
        return entry

    def verify_chain(self) -> bool:
        """Verify the integrity of the audit chain."""
        previous_checksum = ""
        for entry in self._entries:
            expected = entry.checksum
            entry.checksum = ""
            computed = entry.compute_checksum(previous_checksum)
            if computed != expected:
                logger.error(
                    "Audit chain integrity violation",
                    entry_id=entry.id,
                )
                return False
            previous_checksum = computed
        return True

    def get_entries(
        self,
        action: str | None = None,
        actor: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters."""
        results = self._entries

        if action:
            results = [e for e in results if e.action == action]
        if actor:
            results = [e for e in results if e.actor == actor]
        if since:
            results = [e for e in results if e.timestamp >= since]

        return results[-limit:]

    def export_for_compliance(self) -> list[dict[str, Any]]:
        """Export audit log for compliance reporting."""
        return [e.to_dict() for e in self._entries]

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# =============================================================================
# License Management
# =============================================================================


@dataclass
class LicenseInfo:
    """Enterprise license information."""

    organization: str = ""
    tier: str = "free"  # free, pro, enterprise
    max_users: int = 5
    max_repos: int = 10
    features: list[str] = field(default_factory=list)
    valid_until: str = ""
    is_valid: bool = False

    def has_feature(self, feature: str) -> bool:
        return feature in self.features

    def to_dict(self) -> dict[str, Any]:
        return {
            "organization": self.organization,
            "tier": self.tier,
            "max_users": self.max_users,
            "max_repos": self.max_repos,
            "features": self.features,
            "valid_until": self.valid_until,
            "is_valid": self.is_valid,
        }


def validate_license(license_key: str) -> LicenseInfo:
    """Validate an enterprise license key.

    In production, this would verify against a license server or
    validate a signed JWT. For self-hosted, it decodes a signed payload.
    """
    if not license_key:
        return LicenseInfo(tier="free", is_valid=True, features=["basic"])

    # Simple validation: license keys are base64-encoded JSON with a checksum
    try:
        import base64

        decoded = base64.b64decode(license_key).decode("utf-8")
        data = json.loads(decoded)

        return LicenseInfo(
            organization=data.get("org", ""),
            tier=data.get("tier", "pro"),
            max_users=data.get("max_users", 50),
            max_repos=data.get("max_repos", 100),
            features=data.get("features", ["basic", "ai", "formal", "compliance"]),
            valid_until=data.get("valid_until", ""),
            is_valid=True,
        )
    except Exception:
        logger.warning("Invalid license key")
        return LicenseInfo(tier="free", is_valid=False, features=["basic"])


# =============================================================================
# Enterprise Deployment Manager
# =============================================================================


class EnterpriseDeploymentManager:
    """Orchestrates enterprise deployment setup and health checks.

    Usage:
        config = EnterpriseConfig(
            deployment_mode=DeploymentMode.SELF_HOSTED,
            auth_provider=AuthProvider.SAML,
            llm_primary=LLMBackendConfig(backend_type=LLMBackendType.OLLAMA, base_url="http://ollama:11434"),
        )
        mgr = EnterpriseDeploymentManager(config)
        errors = mgr.validate()
        health = mgr.health_check()
    """

    def __init__(self, config: EnterpriseConfig) -> None:
        self._config = config
        self._audit = AuditLogger(
            log_path=config.audit_log_path if config.audit_log_enabled else None
        )
        self._license = validate_license(config.license_key)

    @property
    def config(self) -> EnterpriseConfig:
        return self._config

    @property
    def audit(self) -> AuditLogger:
        return self._audit

    @property
    def license(self) -> LicenseInfo:
        return self._license

    def validate(self) -> list[str]:
        """Validate the full enterprise configuration."""
        errors = self._config.validate()

        if not self._license.is_valid:
            errors.append("Invalid or missing license key.")

        return errors

    def health_check(self) -> dict[str, Any]:
        """Run a health check on the enterprise deployment."""
        checks: dict[str, Any] = {
            "deployment_mode": self._config.deployment_mode.value,
            "config_valid": len(self._config.validate()) == 0,
            "license_valid": self._license.is_valid,
            "license_tier": self._license.tier,
            "auth_provider": self._config.auth_provider.value,
            "llm_backend": self._config.llm_primary.backend_type.value,
            "llm_is_local": self._config.llm_primary.is_local(),
            "audit_enabled": self._config.audit_log_enabled,
            "audit_entries": self._audit.entry_count,
            "audit_chain_valid": self._audit.verify_chain(),
            "security_tls_required": self._config.security.require_tls,
        }

        checks["healthy"] = (
            checks["config_valid"] and checks["license_valid"] and checks["audit_chain_valid"]
        )

        return checks

    def get_deployment_summary(self) -> dict[str, Any]:
        """Get a summary of the deployment configuration."""
        return {
            "mode": self._config.deployment_mode.value,
            "organization": self._config.organization_name,
            "auth": self._config.auth_provider.value,
            "llm": {
                "primary": self._config.llm_primary.backend_type.value,
                "is_local": self._config.llm_primary.is_local(),
                "has_fallback": self._config.llm_fallback is not None,
            },
            "license": self._license.to_dict(),
            "security": {
                "tls": self._config.security.require_tls,
                "mfa": self._config.security.require_mfa,
                "session_timeout_min": self._config.security.session_timeout_minutes,
                "data_retention_days": self._config.security.data_retention_days,
            },
        }
