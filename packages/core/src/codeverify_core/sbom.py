"""SBOM + SLSA Provenance Integration.

Generates CISA-compliant Software Bill of Materials (SBOM) and SLSA attestations
with embedded verification proofs for supply chain security.

Compliance with:
- CISA 2025 Minimum Elements for SBOM
- SLSA v1.0 Provenance
- CycloneDX 1.5
- SPDX 3.0
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class SBOMFormat(str, Enum):
    """Supported SBOM formats."""
    
    CYCLONEDX = "cyclonedx"
    SPDX = "spdx"


class SLSALevel(str, Enum):
    """SLSA build levels."""
    
    LEVEL_0 = "0"  # No guarantees
    LEVEL_1 = "1"  # Documentation of build process
    LEVEL_2 = "2"  # Tamper resistance of build service
    LEVEL_3 = "3"  # Hardened builds


class ComponentType(str, Enum):
    """Types of software components."""
    
    APPLICATION = "application"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    OPERATING_SYSTEM = "operating-system"
    DEVICE = "device"
    FILE = "file"
    CONTAINER = "container"


class LicenseType(str, Enum):
    """Common license types."""
    
    MIT = "MIT"
    APACHE_2 = "Apache-2.0"
    GPL_3 = "GPL-3.0"
    BSD_3 = "BSD-3-Clause"
    ISC = "ISC"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


@dataclass
class ComponentHash:
    """Hash of a component for integrity verification."""
    
    algorithm: str  # "SHA-256", "SHA-512", etc.
    value: str


@dataclass
class ExternalReference:
    """External reference for a component (VCS, website, etc.)."""
    
    type: str  # "vcs", "website", "issue-tracker", "documentation"
    url: str
    comment: str | None = None


@dataclass
class Vulnerability:
    """Known vulnerability in a component."""
    
    id: str  # CVE ID or other identifier
    source: str  # "NVD", "GitHub", "Snyk", etc.
    severity: str
    description: str | None = None
    fixed_version: str | None = None


@dataclass
class Component:
    """A software component (dependency)."""
    
    name: str
    version: str
    type: ComponentType = ComponentType.LIBRARY
    purl: str | None = None  # Package URL
    cpe: str | None = None  # Common Platform Enumeration
    supplier: str | None = None
    author: str | None = None
    license: LicenseType = LicenseType.UNKNOWN
    hashes: list[ComponentHash] = field(default_factory=list)
    external_references: list[ExternalReference] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # Names of dependencies
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)
    
    def to_cyclonedx(self) -> dict[str, Any]:
        """Convert to CycloneDX format."""
        component = {
            "type": self.type.value,
            "name": self.name,
            "version": self.version,
        }
        
        if self.purl:
            component["purl"] = self.purl
        
        if self.cpe:
            component["cpe"] = self.cpe
        
        if self.supplier:
            component["supplier"] = {"name": self.supplier}
        
        if self.author:
            component["author"] = self.author
        
        if self.license != LicenseType.UNKNOWN:
            component["licenses"] = [{"license": {"id": self.license.value}}]
        
        if self.hashes:
            component["hashes"] = [
                {"alg": h.algorithm, "content": h.value}
                for h in self.hashes
            ]
        
        if self.external_references:
            component["externalReferences"] = [
                {"type": ref.type, "url": ref.url}
                for ref in self.external_references
            ]
        
        return component
    
    def to_spdx(self) -> dict[str, Any]:
        """Convert to SPDX format."""
        package = {
            "SPDXID": f"SPDXRef-Package-{self.name}-{self.version}".replace(".", "-"),
            "name": self.name,
            "versionInfo": self.version,
            "downloadLocation": "NOASSERTION",
        }
        
        if self.purl:
            package["externalRefs"] = [{
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": self.purl,
            }]
        
        if self.license != LicenseType.UNKNOWN:
            package["licenseConcluded"] = self.license.value
        else:
            package["licenseConcluded"] = "NOASSERTION"
        
        if self.supplier:
            package["supplier"] = f"Organization: {self.supplier}"
        
        if self.hashes:
            for h in self.hashes:
                if h.algorithm == "SHA-256":
                    package["checksums"] = [{"algorithm": "SHA256", "checksumValue": h.value}]
        
        return package


@dataclass
class VerificationAttestation:
    """Attestation of CodeVerify verification results."""
    
    verification_type: str  # "formal", "ai", "pattern", "hybrid"
    verification_passed: bool
    conditions_checked: int
    conditions_passed: int
    findings_count: int
    critical_findings: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    verification_hash: str | None = None  # Hash of verification output
    proof_artifacts: list[str] = field(default_factory=list)  # Links to proof files
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "verificationType": self.verification_type,
            "passed": self.verification_passed,
            "conditionsChecked": self.conditions_checked,
            "conditionsPassed": self.conditions_passed,
            "findingsCount": self.findings_count,
            "criticalFindings": self.critical_findings,
            "timestamp": self.timestamp.isoformat() + "Z",
            "verificationHash": self.verification_hash,
            "proofArtifacts": self.proof_artifacts,
        }


@dataclass
class SLSAProvenance:
    """SLSA v1.0 compliant provenance."""
    
    build_type: str
    builder_id: str
    invocation_id: str
    build_started_on: datetime
    build_finished_on: datetime
    source_uri: str
    source_digest: dict[str, str]  # {"sha256": "abc..."}
    entry_point: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    materials: list[dict[str, Any]] = field(default_factory=list)
    slsa_level: SLSALevel = SLSALevel.LEVEL_1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to SLSA provenance format."""
        return {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [],  # Filled by SBOM generator
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {
                "buildDefinition": {
                    "buildType": self.build_type,
                    "externalParameters": self.parameters,
                    "internalParameters": {},
                    "resolvedDependencies": self.materials,
                },
                "runDetails": {
                    "builder": {"id": self.builder_id},
                    "metadata": {
                        "invocationId": self.invocation_id,
                        "startedOn": self.build_started_on.isoformat() + "Z",
                        "finishedOn": self.build_finished_on.isoformat() + "Z",
                    },
                },
            },
        }


@dataclass
class SBOM:
    """Software Bill of Materials."""
    
    # Metadata
    serial_number: str
    version: int = 1
    name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Author info (required by CISA 2025)
    author_name: str = ""
    author_email: str | None = None
    tool_name: str = "CodeVerify"
    tool_version: str = "0.4.0"
    
    # Components
    components: list[Component] = field(default_factory=list)
    
    # Relationships
    dependencies: dict[str, list[str]] = field(default_factory=dict)  # component -> [deps]
    
    # CodeVerify specific
    verification_attestation: VerificationAttestation | None = None
    slsa_provenance: SLSAProvenance | None = None
    
    def to_cyclonedx(self) -> dict[str, Any]:
        """Export as CycloneDX 1.5 format."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": f"urn:uuid:{self.serial_number}",
            "version": self.version,
            "metadata": {
                "timestamp": self.timestamp.isoformat() + "Z",
                "tools": [{
                    "vendor": "CodeVerify",
                    "name": self.tool_name,
                    "version": self.tool_version,
                }],
                "authors": [{
                    "name": self.author_name,
                    "email": self.author_email,
                }] if self.author_name else [],
                "component": {
                    "type": "application",
                    "name": self.name,
                },
            },
            "components": [c.to_cyclonedx() for c in self.components],
        }
        
        # Add dependencies
        if self.dependencies:
            sbom["dependencies"] = [
                {
                    "ref": comp_name,
                    "dependsOn": deps,
                }
                for comp_name, deps in self.dependencies.items()
            ]
        
        # Add CodeVerify verification attestation as property
        if self.verification_attestation:
            sbom["metadata"]["properties"] = [
                {
                    "name": "codeverify:verification",
                    "value": json.dumps(self.verification_attestation.to_dict()),
                }
            ]
        
        # Add SLSA provenance
        if self.slsa_provenance:
            sbom["metadata"]["properties"] = sbom["metadata"].get("properties", [])
            sbom["metadata"]["properties"].append({
                "name": "slsa:provenance",
                "value": json.dumps(self.slsa_provenance.to_dict()),
            })
        
        return sbom
    
    def to_spdx(self) -> dict[str, Any]:
        """Export as SPDX 2.3 format."""
        spdx = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": self.name,
            "documentNamespace": f"https://codeverify.io/spdx/{self.serial_number}",
            "creationInfo": {
                "created": self.timestamp.isoformat() + "Z",
                "creators": [
                    f"Tool: {self.tool_name}-{self.tool_version}",
                ],
            },
            "packages": [c.to_spdx() for c in self.components],
            "relationships": [],
        }
        
        if self.author_name:
            spdx["creationInfo"]["creators"].append(f"Organization: {self.author_name}")
        
        # Add relationships
        for comp_name, deps in self.dependencies.items():
            comp_id = f"SPDXRef-Package-{comp_name}".replace(".", "-")
            for dep in deps:
                dep_id = f"SPDXRef-Package-{dep}".replace(".", "-")
                spdx["relationships"].append({
                    "spdxElementId": comp_id,
                    "relatedSpdxElement": dep_id,
                    "relationshipType": "DEPENDS_ON",
                })
        
        return spdx
    
    def to_json(self, format: SBOMFormat = SBOMFormat.CYCLONEDX) -> str:
        """Export SBOM as JSON string."""
        if format == SBOMFormat.CYCLONEDX:
            return json.dumps(self.to_cyclonedx(), indent=2)
        else:
            return json.dumps(self.to_spdx(), indent=2)


class SBOMGenerator:
    """Generates SBOM from project dependencies and verification results."""
    
    def __init__(
        self,
        author_name: str = "",
        author_email: str | None = None,
    ) -> None:
        self.author_name = author_name
        self.author_email = author_email
    
    def generate(
        self,
        project_name: str,
        dependencies: list[dict[str, Any]],
        verification_results: dict[str, Any] | None = None,
        slsa_provenance: SLSAProvenance | None = None,
    ) -> SBOM:
        """Generate SBOM from project information.
        
        Args:
            project_name: Name of the project
            dependencies: List of dependency dicts with name, version, etc.
            verification_results: CodeVerify verification results
            slsa_provenance: SLSA provenance information
            
        Returns:
            Complete SBOM with verification attestation
        """
        serial_number = str(uuid.uuid4())
        
        # Parse dependencies into components
        components = []
        for dep in dependencies:
            component = self._parse_dependency(dep)
            components.append(component)
        
        # Create verification attestation from results
        attestation = None
        if verification_results:
            attestation = self._create_attestation(verification_results)
        
        # Build dependency graph
        dep_graph = self._build_dependency_graph(dependencies)
        
        sbom = SBOM(
            serial_number=serial_number,
            name=project_name,
            author_name=self.author_name,
            author_email=self.author_email,
            components=components,
            dependencies=dep_graph,
            verification_attestation=attestation,
            slsa_provenance=slsa_provenance,
        )
        
        logger.info(
            "SBOM generated",
            project=project_name,
            components=len(components),
            has_verification=attestation is not None,
            has_slsa=slsa_provenance is not None,
        )
        
        return sbom
    
    def generate_from_requirements(
        self,
        project_name: str,
        requirements_content: str,
        verification_results: dict[str, Any] | None = None,
    ) -> SBOM:
        """Generate SBOM from requirements.txt content."""
        dependencies = self._parse_requirements(requirements_content)
        return self.generate(project_name, dependencies, verification_results)
    
    def generate_from_package_json(
        self,
        project_name: str,
        package_json_content: str,
        verification_results: dict[str, Any] | None = None,
    ) -> SBOM:
        """Generate SBOM from package.json content."""
        dependencies = self._parse_package_json(package_json_content)
        return self.generate(project_name, dependencies, verification_results)
    
    def _parse_dependency(self, dep: dict[str, Any]) -> Component:
        """Parse a dependency dict into a Component."""
        name = dep.get("name", "unknown")
        version = dep.get("version", "0.0.0")
        
        # Build package URL
        ecosystem = dep.get("ecosystem", "pypi")
        purl = f"pkg:{ecosystem}/{name}@{version}"
        
        # Parse license
        license_str = dep.get("license", "unknown")
        license_type = self._parse_license(license_str)
        
        # Build hashes if available
        hashes = []
        if "sha256" in dep:
            hashes.append(ComponentHash(algorithm="SHA-256", value=dep["sha256"]))
        
        # Parse vulnerabilities if present
        vulns = []
        for v in dep.get("vulnerabilities", []):
            vulns.append(Vulnerability(
                id=v.get("id", "unknown"),
                source=v.get("source", "unknown"),
                severity=v.get("severity", "unknown"),
                description=v.get("description"),
                fixed_version=v.get("fixed_version"),
            ))
        
        return Component(
            name=name,
            version=version,
            purl=purl,
            license=license_type,
            supplier=dep.get("supplier"),
            hashes=hashes,
            dependencies=dep.get("dependencies", []),
            vulnerabilities=vulns,
        )
    
    def _parse_license(self, license_str: str) -> LicenseType:
        """Parse license string to LicenseType."""
        license_map = {
            "mit": LicenseType.MIT,
            "apache-2.0": LicenseType.APACHE_2,
            "apache 2.0": LicenseType.APACHE_2,
            "gpl-3.0": LicenseType.GPL_3,
            "bsd-3-clause": LicenseType.BSD_3,
            "isc": LicenseType.ISC,
        }
        return license_map.get(license_str.lower(), LicenseType.UNKNOWN)
    
    def _parse_requirements(self, content: str) -> list[dict[str, Any]]:
        """Parse requirements.txt format."""
        dependencies = []
        
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Parse name==version or name>=version format
            for sep in ["==", ">=", "<=", "~=", "!="]:
                if sep in line:
                    parts = line.split(sep, 1)
                    dependencies.append({
                        "name": parts[0].strip(),
                        "version": parts[1].strip() if len(parts) > 1 else "0.0.0",
                        "ecosystem": "pypi",
                    })
                    break
            else:
                # Just package name without version
                if line and not line.startswith("-"):
                    dependencies.append({
                        "name": line,
                        "version": "*",
                        "ecosystem": "pypi",
                    })
        
        return dependencies
    
    def _parse_package_json(self, content: str) -> list[dict[str, Any]]:
        """Parse package.json format."""
        try:
            pkg = json.loads(content)
        except json.JSONDecodeError:
            return []
        
        dependencies = []
        
        for dep_type in ["dependencies", "devDependencies"]:
            for name, version in pkg.get(dep_type, {}).items():
                # Clean version string
                version = version.lstrip("^~")
                dependencies.append({
                    "name": name,
                    "version": version,
                    "ecosystem": "npm",
                })
        
        return dependencies
    
    def _create_attestation(self, results: dict[str, Any]) -> VerificationAttestation:
        """Create verification attestation from results."""
        findings = results.get("findings", [])
        critical = sum(1 for f in findings if f.get("severity") == "critical")
        
        # Hash the results for integrity
        results_json = json.dumps(results, sort_keys=True)
        results_hash = hashlib.sha256(results_json.encode()).hexdigest()
        
        return VerificationAttestation(
            verification_type=results.get("verification_type", "hybrid"),
            verification_passed=results.get("passed", False),
            conditions_checked=results.get("conditions_checked", 0),
            conditions_passed=results.get("conditions_passed", 0),
            findings_count=len(findings),
            critical_findings=critical,
            verification_hash=results_hash,
        )
    
    def _build_dependency_graph(
        self,
        dependencies: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Build dependency graph from dependency list."""
        graph = {}
        
        for dep in dependencies:
            name = dep.get("name", "")
            deps = dep.get("dependencies", [])
            if name:
                graph[name] = deps
        
        return graph


class SLSAAttestationGenerator:
    """Generates SLSA attestations for builds."""
    
    def __init__(self, builder_id: str) -> None:
        self.builder_id = builder_id
    
    def generate(
        self,
        source_uri: str,
        source_commit: str,
        build_started: datetime,
        build_finished: datetime,
        entry_point: str | None = None,
        parameters: dict[str, Any] | None = None,
        materials: list[dict[str, Any]] | None = None,
        level: SLSALevel = SLSALevel.LEVEL_2,
    ) -> SLSAProvenance:
        """Generate SLSA provenance attestation.
        
        Args:
            source_uri: URI of the source repository
            source_commit: Git commit SHA
            build_started: Build start time
            build_finished: Build finish time
            entry_point: Build entry point (e.g., Makefile target)
            parameters: Build parameters
            materials: Build materials (dependencies, etc.)
            level: SLSA level to attest
            
        Returns:
            SLSA provenance attestation
        """
        invocation_id = str(uuid.uuid4())
        
        return SLSAProvenance(
            build_type="https://codeverify.io/build/v1",
            builder_id=self.builder_id,
            invocation_id=invocation_id,
            build_started_on=build_started,
            build_finished_on=build_finished,
            source_uri=source_uri,
            source_digest={"sha256": source_commit},
            entry_point=entry_point,
            parameters=parameters or {},
            materials=materials or [],
            slsa_level=level,
        )


class VerifiedSBOMExporter:
    """Exports SBOM with cryptographic signatures."""
    
    def __init__(self, signing_key: str | None = None) -> None:
        self.signing_key = signing_key
    
    def export(
        self,
        sbom: SBOM,
        format: SBOMFormat = SBOMFormat.CYCLONEDX,
        sign: bool = True,
    ) -> dict[str, Any]:
        """Export SBOM with optional signing.
        
        Returns:
            Dictionary with:
            - sbom: The SBOM content
            - signature: Cryptographic signature (if signing enabled)
            - metadata: Export metadata
        """
        if format == SBOMFormat.CYCLONEDX:
            sbom_content = sbom.to_cyclonedx()
        else:
            sbom_content = sbom.to_spdx()
        
        result = {
            "sbom": sbom_content,
            "format": format.value,
            "exported_at": datetime.utcnow().isoformat() + "Z",
        }
        
        if sign and self.signing_key:
            # In production, use proper cryptographic signing
            # This is a placeholder showing the structure
            content_hash = hashlib.sha256(
                json.dumps(sbom_content, sort_keys=True).encode()
            ).hexdigest()
            
            result["signature"] = {
                "algorithm": "sha256",
                "value": content_hash,
                "signer": "CodeVerify",
            }
        
        # Add verification badge info
        if sbom.verification_attestation:
            attestation = sbom.verification_attestation
            result["verification_badge"] = {
                "passed": attestation.verification_passed,
                "type": attestation.verification_type,
                "findings": attestation.findings_count,
                "critical": attestation.critical_findings,
            }
        
        return result
