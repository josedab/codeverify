"""Tests for SBOM + SLSA Provenance Integration."""

import json
from datetime import datetime

import pytest

from codeverify_core.sbom import (
    Component,
    ComponentHash,
    ComponentType,
    ExternalReference,
    LicenseType,
    SBOM,
    SBOMFormat,
    SBOMGenerator,
    SLSAAttestationGenerator,
    SLSALevel,
    SLSAProvenance,
    VerificationAttestation,
    VerifiedSBOMExporter,
    Vulnerability,
)


class TestComponent:
    """Test Component model."""
    
    def test_to_cyclonedx(self):
        """Test CycloneDX format conversion."""
        component = Component(
            name="requests",
            version="2.31.0",
            type=ComponentType.LIBRARY,
            purl="pkg:pypi/requests@2.31.0",
            license=LicenseType.APACHE_2,
            supplier="PSF",
            hashes=[ComponentHash(algorithm="SHA-256", value="abc123")],
        )
        
        cdx = component.to_cyclonedx()
        
        assert cdx["name"] == "requests"
        assert cdx["version"] == "2.31.0"
        assert cdx["type"] == "library"
        assert cdx["purl"] == "pkg:pypi/requests@2.31.0"
        assert "hashes" in cdx
        assert cdx["hashes"][0]["alg"] == "SHA-256"
    
    def test_to_spdx(self):
        """Test SPDX format conversion."""
        component = Component(
            name="lodash",
            version="4.17.21",
            purl="pkg:npm/lodash@4.17.21",
            license=LicenseType.MIT,
        )
        
        spdx = component.to_spdx()
        
        assert spdx["name"] == "lodash"
        assert spdx["versionInfo"] == "4.17.21"
        assert spdx["licenseConcluded"] == "MIT"
        assert "SPDXRef-Package" in spdx["SPDXID"]


class TestSBOM:
    """Test SBOM model."""
    
    def test_to_cyclonedx(self):
        """Test CycloneDX export."""
        sbom = SBOM(
            serial_number="test-123",
            name="test-project",
            author_name="Test Author",
            components=[
                Component(name="dep1", version="1.0.0"),
                Component(name="dep2", version="2.0.0"),
            ],
            dependencies={"dep1": ["dep2"]},
        )
        
        cdx = sbom.to_cyclonedx()
        
        assert cdx["bomFormat"] == "CycloneDX"
        assert cdx["specVersion"] == "1.5"
        assert len(cdx["components"]) == 2
        assert len(cdx["dependencies"]) == 1
    
    def test_to_spdx(self):
        """Test SPDX export."""
        sbom = SBOM(
            serial_number="test-456",
            name="test-project",
            components=[
                Component(name="dep1", version="1.0.0"),
            ],
        )
        
        spdx = sbom.to_spdx()
        
        assert spdx["spdxVersion"] == "SPDX-2.3"
        assert spdx["name"] == "test-project"
        assert len(spdx["packages"]) == 1
    
    def test_to_json(self):
        """Test JSON export."""
        sbom = SBOM(
            serial_number="test-789",
            name="test-project",
            components=[Component(name="dep1", version="1.0.0")],
        )
        
        json_str = sbom.to_json(SBOMFormat.CYCLONEDX)
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["bomFormat"] == "CycloneDX"
    
    def test_with_verification_attestation(self):
        """Test SBOM with verification attestation."""
        attestation = VerificationAttestation(
            verification_type="formal",
            verification_passed=True,
            conditions_checked=10,
            conditions_passed=10,
            findings_count=0,
            critical_findings=0,
        )
        
        sbom = SBOM(
            serial_number="test-attest",
            name="verified-project",
            verification_attestation=attestation,
        )
        
        cdx = sbom.to_cyclonedx()
        
        # Attestation should be in properties
        props = cdx["metadata"].get("properties", [])
        assert any("codeverify:verification" in p.get("name", "") for p in props)


class TestSBOMGenerator:
    """Test SBOM generation."""
    
    def test_generate_basic(self):
        """Test basic SBOM generation."""
        generator = SBOMGenerator(author_name="Test Author")
        
        sbom = generator.generate(
            project_name="my-project",
            dependencies=[
                {"name": "requests", "version": "2.31.0", "ecosystem": "pypi"},
                {"name": "flask", "version": "3.0.0", "ecosystem": "pypi"},
            ],
        )
        
        assert sbom.name == "my-project"
        assert len(sbom.components) == 2
        assert sbom.components[0].purl == "pkg:pypi/requests@2.31.0"
    
    def test_generate_from_requirements(self):
        """Test generation from requirements.txt."""
        generator = SBOMGenerator()
        
        requirements = """
requests==2.31.0
flask>=3.0.0
pydantic~=2.5.0
# comment
python-dotenv
"""
        
        sbom = generator.generate_from_requirements("test-project", requirements)
        
        assert len(sbom.components) >= 3
        assert any(c.name == "requests" for c in sbom.components)
        assert any(c.name == "flask" for c in sbom.components)
    
    def test_generate_from_package_json(self):
        """Test generation from package.json."""
        generator = SBOMGenerator()
        
        package_json = """
{
    "name": "my-app",
    "version": "1.0.0",
    "dependencies": {
        "express": "^4.18.0",
        "lodash": "~4.17.21"
    },
    "devDependencies": {
        "jest": "^29.0.0"
    }
}
"""
        
        sbom = generator.generate_from_package_json("my-app", package_json)
        
        assert len(sbom.components) == 3
        assert any(c.name == "express" for c in sbom.components)
        assert any(c.name == "jest" for c in sbom.components)
    
    def test_generate_with_verification_results(self):
        """Test generation with verification results."""
        generator = SBOMGenerator()
        
        verification_results = {
            "verification_type": "hybrid",
            "passed": True,
            "conditions_checked": 50,
            "conditions_passed": 48,
            "findings": [
                {"severity": "medium", "title": "Minor issue"},
            ],
        }
        
        sbom = generator.generate(
            project_name="verified-project",
            dependencies=[{"name": "dep1", "version": "1.0.0"}],
            verification_results=verification_results,
        )
        
        assert sbom.verification_attestation is not None
        assert sbom.verification_attestation.verification_passed is True
        assert sbom.verification_attestation.conditions_checked == 50
        assert sbom.verification_attestation.critical_findings == 0


class TestSLSAAttestationGenerator:
    """Test SLSA attestation generation."""
    
    def test_generate_attestation(self):
        """Test generating SLSA attestation."""
        generator = SLSAAttestationGenerator(
            builder_id="https://codeverify.io/builder/v1"
        )
        
        now = datetime.utcnow()
        
        provenance = generator.generate(
            source_uri="https://github.com/org/repo",
            source_commit="abc123def456",
            build_started=now,
            build_finished=now,
            entry_point="make build",
            parameters={"target": "release"},
            level=SLSALevel.LEVEL_2,
        )
        
        assert provenance.builder_id == "https://codeverify.io/builder/v1"
        assert provenance.source_uri == "https://github.com/org/repo"
        assert provenance.slsa_level == SLSALevel.LEVEL_2
    
    def test_provenance_to_dict(self):
        """Test SLSA provenance serialization."""
        now = datetime.utcnow()
        
        provenance = SLSAProvenance(
            build_type="https://codeverify.io/build/v1",
            builder_id="test-builder",
            invocation_id="inv-123",
            build_started_on=now,
            build_finished_on=now,
            source_uri="https://github.com/test/repo",
            source_digest={"sha256": "abc123"},
        )
        
        data = provenance.to_dict()
        
        assert data["_type"] == "https://in-toto.io/Statement/v1"
        assert data["predicateType"] == "https://slsa.dev/provenance/v1"
        assert "buildDefinition" in data["predicate"]


class TestVerifiedSBOMExporter:
    """Test SBOM export with signatures."""
    
    def test_export_without_signing(self):
        """Test export without signing."""
        sbom = SBOM(
            serial_number="test-export",
            name="export-test",
            components=[Component(name="dep1", version="1.0.0")],
        )
        
        exporter = VerifiedSBOMExporter()
        result = exporter.export(sbom, sign=False)
        
        assert "sbom" in result
        assert result["format"] == "cyclonedx"
        assert "signature" not in result
    
    def test_export_with_signing(self):
        """Test export with signing."""
        sbom = SBOM(
            serial_number="test-signed",
            name="signed-test",
            components=[Component(name="dep1", version="1.0.0")],
        )
        
        exporter = VerifiedSBOMExporter(signing_key="test-key")
        result = exporter.export(sbom, sign=True)
        
        assert "sbom" in result
        assert "signature" in result
        assert result["signature"]["algorithm"] == "sha256"
    
    def test_export_with_verification_badge(self):
        """Test export includes verification badge."""
        attestation = VerificationAttestation(
            verification_type="formal",
            verification_passed=True,
            conditions_checked=10,
            conditions_passed=10,
            findings_count=2,
            critical_findings=0,
        )
        
        sbom = SBOM(
            serial_number="test-badge",
            name="badge-test",
            verification_attestation=attestation,
        )
        
        exporter = VerifiedSBOMExporter()
        result = exporter.export(sbom)
        
        assert "verification_badge" in result
        assert result["verification_badge"]["passed"] is True
        assert result["verification_badge"]["findings"] == 2


class TestVerificationAttestation:
    """Test verification attestation model."""
    
    def test_to_dict(self):
        """Test attestation serialization."""
        attestation = VerificationAttestation(
            verification_type="hybrid",
            verification_passed=True,
            conditions_checked=100,
            conditions_passed=95,
            findings_count=5,
            critical_findings=1,
            verification_hash="abc123",
        )
        
        data = attestation.to_dict()
        
        assert data["verificationType"] == "hybrid"
        assert data["passed"] is True
        assert data["conditionsChecked"] == 100
        assert data["conditionsPassed"] == 95
        assert data["findingsCount"] == 5
        assert data["criticalFindings"] == 1


class TestSBOMVulnerability:
    """Test vulnerability in SBOM context."""
    
    def test_vulnerability_tracking(self):
        """Test tracking vulnerabilities in components."""
        vuln = Vulnerability(
            id="CVE-2024-0001",
            source="NVD",
            severity="critical",
            description="Remote code execution",
            fixed_version="2.0.1",
        )
        
        component = Component(
            name="vulnerable-lib",
            version="2.0.0",
            vulnerabilities=[vuln],
        )
        
        assert len(component.vulnerabilities) == 1
        assert component.vulnerabilities[0].id == "CVE-2024-0001"
        assert component.vulnerabilities[0].fixed_version == "2.0.1"
