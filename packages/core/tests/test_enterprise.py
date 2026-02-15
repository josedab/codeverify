"""Tests for enterprise deployment features."""

import json
import tempfile

from codeverify_core.enterprise import (
    AuditAction,
    AuditEntry,
    AuditLogger,
    AuthProvider,
    DeploymentMode,
    EnterpriseConfig,
    EnterpriseDeploymentManager,
    LLMBackendConfig,
    LLMBackendType,
    SAMLConfig,
    validate_license,
)


class TestEnterpriseConfig:
    def test_default_config(self):
        config = EnterpriseConfig()
        assert config.deployment_mode == DeploymentMode.SELF_HOSTED
        assert config.auth_provider == AuthProvider.LOCAL
        assert config.llm_primary.is_local()

    def test_air_gapped_requires_local_llm(self):
        config = EnterpriseConfig(
            deployment_mode=DeploymentMode.AIR_GAPPED,
            llm_primary=LLMBackendConfig(backend_type=LLMBackendType.OPENAI),
        )
        errors = config.validate()
        assert any("local LLM" in e for e in errors)

    def test_air_gapped_with_ollama_valid(self):
        config = EnterpriseConfig(
            deployment_mode=DeploymentMode.AIR_GAPPED,
            llm_primary=LLMBackendConfig(backend_type=LLMBackendType.OLLAMA),
        )
        errors = config.validate()
        assert not any("local LLM" in e for e in errors)

    def test_saml_requires_idp(self):
        config = EnterpriseConfig(
            auth_provider=AuthProvider.SAML,
            saml=SAMLConfig(enabled=True),
        )
        errors = config.validate()
        assert any("idp_metadata_url" in e for e in errors)

    def test_saml_with_idp_valid(self):
        config = EnterpriseConfig(
            auth_provider=AuthProvider.SAML,
            saml=SAMLConfig(
                enabled=True,
                idp_metadata_url="https://idp.corp.com/metadata",
            ),
        )
        errors = config.validate()
        assert not any("SAML" in e for e in errors)


class TestAuditLogger:
    def test_basic_logging(self):
        audit = AuditLogger()
        entry = audit.log(AuditAction.LOGIN, actor="user@test.com")
        assert entry.action == "login"
        assert entry.actor == "user@test.com"
        assert entry.checksum

    def test_chain_integrity(self):
        audit = AuditLogger()
        audit.log(AuditAction.LOGIN, actor="a@test.com")
        audit.log(AuditAction.ANALYSIS_STARTED, actor="a@test.com")
        audit.log(AuditAction.ANALYSIS_COMPLETED, actor="a@test.com")
        assert audit.verify_chain() is True

    def test_file_persistence(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        audit = AuditLogger(log_path=path)
        audit.log(AuditAction.LOGIN, actor="user@test.com")
        audit.log(AuditAction.CONFIG_CHANGED, actor="admin@test.com")

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["action"] == "login"

    def test_query_entries(self):
        audit = AuditLogger()
        audit.log(AuditAction.LOGIN, actor="alice")
        audit.log(AuditAction.LOGIN, actor="bob")
        audit.log(AuditAction.ANALYSIS_STARTED, actor="alice")

        results = audit.get_entries(action="login")
        assert len(results) == 2

        results = audit.get_entries(actor="alice")
        assert len(results) == 2

    def test_export_for_compliance(self):
        audit = AuditLogger()
        audit.log(AuditAction.LOGIN, actor="user")
        export = audit.export_for_compliance()
        assert len(export) == 1
        assert "checksum" in export[0]


class TestAuditEntry:
    def test_checksum_chaining(self):
        e1 = AuditEntry(action="login", actor="a")
        c1 = e1.compute_checksum("")

        e2 = AuditEntry(action="logout", actor="a")
        c2 = e2.compute_checksum(c1)

        assert c1 != c2
        assert len(c1) == 64  # SHA-256 hex

    def test_to_json(self):
        entry = AuditEntry(action="login", actor="user")
        entry.compute_checksum("")
        data = json.loads(entry.to_json())
        assert data["action"] == "login"
        assert data["checksum"]


class TestLicenseValidation:
    def test_empty_key_returns_free(self):
        info = validate_license("")
        assert info.tier == "free"
        assert info.is_valid is True

    def test_valid_base64_key(self):
        import base64

        payload = json.dumps(
            {
                "org": "TestCorp",
                "tier": "enterprise",
                "max_users": 500,
                "features": ["basic", "ai", "formal", "compliance", "sso"],
            }
        )
        key = base64.b64encode(payload.encode()).decode()
        info = validate_license(key)
        assert info.tier == "enterprise"
        assert info.organization == "TestCorp"
        assert info.has_feature("sso")
        assert info.max_users == 500

    def test_invalid_key(self):
        info = validate_license("not-valid-base64!!!")
        assert info.is_valid is False


class TestEnterpriseDeploymentManager:
    def test_health_check_default(self):
        config = EnterpriseConfig()
        mgr = EnterpriseDeploymentManager(config)
        health = mgr.health_check()
        assert health["deployment_mode"] == "self_hosted"
        assert health["llm_is_local"] is True
        assert health["audit_chain_valid"] is True

    def test_deployment_summary(self):
        config = EnterpriseConfig(
            organization_name="TestCorp",
            auth_provider=AuthProvider.SAML,
        )
        mgr = EnterpriseDeploymentManager(config)
        summary = mgr.get_deployment_summary()
        assert summary["organization"] == "TestCorp"
        assert summary["auth"] == "saml"
        assert summary["llm"]["is_local"] is True

    def test_audit_via_manager(self):
        config = EnterpriseConfig(audit_log_enabled=False)
        mgr = EnterpriseDeploymentManager(config)
        mgr.audit.log(AuditAction.LOGIN, actor="admin")
        assert mgr.audit.entry_count == 1
