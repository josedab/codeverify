"""Supply Chain Risk Scoring.

Comprehensive risk scoring for supply chain dependencies with CVE correlation,
SBOM generation, and a multi-factor scoring model.

Key features:
1. CVE Correlation: Match dependencies against known vulnerabilities
2. Risk Scoring: Multi-factor scoring combining CVSS, maintenance, and typosquatting signals
3. SBOM Generation: Produce CycloneDX and SPDX documents with risk annotations
4. Remediation Planning: Actionable upgrade paths ranked by risk reduction
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enumerations
# =============================================================================


class CVESeverity(str, Enum):
    """CVSS v3 severity rating."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class ExploitMaturity(str, Enum):
    """Exploit code maturity from CVSS temporal metrics."""

    UNPROVEN = "unproven"
    PROOF_OF_CONCEPT = "proof_of_concept"
    FUNCTIONAL = "functional"
    HIGH = "high"
    NOT_DEFINED = "not_defined"


class RemediationLevel(str, Enum):
    """Available remediation level for a vulnerability."""

    OFFICIAL_FIX = "official_fix"
    TEMPORARY_FIX = "temporary_fix"
    WORKAROUND = "workaround"
    UNAVAILABLE = "unavailable"


class SBOMFormat(str, Enum):
    """Supported SBOM output formats."""

    SPDX = "spdx"
    CYCLONEDX = "cyclonedx"
    SWID = "swid"


class RiskCategory(str, Enum):
    """Category of supply-chain risk."""

    VULNERABILITY = "vulnerability"
    LICENSE = "license"
    MAINTENANCE = "maintenance"
    QUALITY = "quality"
    TYPOSQUATTING = "typosquatting"


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class CVERecord:
    """A single CVE entry correlated to a dependency."""

    cve_id: str
    severity: CVESeverity
    cvss_score: float
    description: str
    affected_versions: list[str]
    fixed_versions: list[str]
    exploit_maturity: ExploitMaturity
    published_date: datetime
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cve_id": self.cve_id,
            "severity": self.severity.value,
            "cvss_score": self.cvss_score,
            "description": self.description,
            "affected_versions": self.affected_versions,
            "fixed_versions": self.fixed_versions,
            "exploit_maturity": self.exploit_maturity.value,
            "published_date": self.published_date.isoformat(),
            "references": self.references,
        }


@dataclass
class DependencyRiskProfile:
    """Aggregated risk profile for a single dependency."""

    package_name: str
    version: str
    ecosystem: str
    cve_records: list[CVERecord] = field(default_factory=list)
    risk_score: float = 0.0
    risk_category: RiskCategory = RiskCategory.VULNERABILITY
    maintenance_score: float = 0.0
    popularity_score: float = 0.0
    license_risk: str = "low"
    typosquat_risk: float = 0.0
    transitive_risk: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_name": self.package_name,
            "version": self.version,
            "ecosystem": self.ecosystem,
            "cve_records": [c.to_dict() for c in self.cve_records],
            "risk_score": round(self.risk_score, 2),
            "risk_category": self.risk_category.value,
            "maintenance_score": round(self.maintenance_score, 2),
            "popularity_score": round(self.popularity_score, 2),
            "license_risk": self.license_risk,
            "typosquat_risk": round(self.typosquat_risk, 2),
            "transitive_risk": round(self.transitive_risk, 2),
        }


@dataclass
class SBOMComponent:
    """A component entry inside an SBOM document."""

    name: str
    version: str
    ecosystem: str
    purl: str
    licenses: list[str]
    supplier: str | None
    checksums: dict[str, str] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    risk_profile: DependencyRiskProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "ecosystem": self.ecosystem,
            "purl": self.purl,
            "licenses": self.licenses,
            "supplier": self.supplier,
            "checksums": self.checksums,
            "dependencies": self.dependencies,
        }
        if self.risk_profile is not None:
            result["risk_profile"] = self.risk_profile.to_dict()
        return result


@dataclass
class SBOMDocument:
    """A complete SBOM document."""

    format: SBOMFormat
    version: str
    created_at: datetime
    project_name: str
    components: list[SBOMComponent]
    total_risk_score: float
    high_risk_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "project_name": self.project_name,
            "components": [c.to_dict() for c in self.components],
            "total_risk_score": round(self.total_risk_score, 2),
            "high_risk_count": self.high_risk_count,
        }


@dataclass
class SupplyChainRiskReport:
    """Top-level report produced by the supply chain risk analyzer."""

    project_name: str
    scan_date: datetime
    total_dependencies: int
    direct_dependencies: int
    risk_score: float
    critical_cves: int
    high_cves: int
    dependency_profiles: list[DependencyRiskProfile]
    sbom: SBOMDocument | None = None
    recommendations: list[str] = field(default_factory=list)
    risk_trend: str = "stable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "scan_date": self.scan_date.isoformat(),
            "total_dependencies": self.total_dependencies,
            "direct_dependencies": self.direct_dependencies,
            "risk_score": round(self.risk_score, 2),
            "critical_cves": self.critical_cves,
            "high_cves": self.high_cves,
            "dependency_profiles": [p.to_dict() for p in self.dependency_profiles],
            "sbom": self.sbom.to_dict() if self.sbom else None,
            "recommendations": self.recommendations,
            "risk_trend": self.risk_trend,
        }


# =============================================================================
# CVE Correlator
# =============================================================================


class CVECorrelator:
    """Correlate packages against a local CVE knowledge base."""

    _CVE_DATABASE: dict[str, list[dict[str, Any]]] = {
        "npm:lodash": [
            {
                "cve_id": "CVE-2021-23337",
                "severity": CVESeverity.HIGH,
                "cvss_score": 7.2,
                "description": "Prototype pollution in lodash",
                "affected_versions": ["<4.17.21"],
                "fixed_versions": ["4.17.21"],
                "exploit_maturity": ExploitMaturity.FUNCTIONAL,
                "published_date": "2021-02-15",
            }
        ],
        "npm:minimist": [
            {
                "cve_id": "CVE-2021-44906",
                "severity": CVESeverity.CRITICAL,
                "cvss_score": 9.8,
                "description": "Prototype pollution in minimist",
                "affected_versions": ["<1.2.6"],
                "fixed_versions": ["1.2.6"],
                "exploit_maturity": ExploitMaturity.PROOF_OF_CONCEPT,
                "published_date": "2022-03-17",
            }
        ],
        "pypi:requests": [
            {
                "cve_id": "CVE-2023-32681",
                "severity": CVESeverity.MEDIUM,
                "cvss_score": 6.1,
                "description": "Unintended leak of Proxy-Authorization header",
                "affected_versions": ["<2.31.0"],
                "fixed_versions": ["2.31.0"],
                "exploit_maturity": ExploitMaturity.UNPROVEN,
                "published_date": "2023-05-26",
            }
        ],
        "npm:express": [
            {
                "cve_id": "CVE-2024-29041",
                "severity": CVESeverity.MEDIUM,
                "cvss_score": 6.1,
                "description": "Open redirect via malformed URLs",
                "affected_versions": ["<4.19.2"],
                "fixed_versions": ["4.19.2"],
                "exploit_maturity": ExploitMaturity.PROOF_OF_CONCEPT,
                "published_date": "2024-03-25",
            }
        ],
    }

    def __init__(self) -> None:
        self._lookup_count: int = 0
        self._hit_count: int = 0
        logger.info("cve_correlator_initialized", known_packages=len(self._CVE_DATABASE))

    def correlate(self, package_name: str, version: str, ecosystem: str) -> list[CVERecord]:
        """Return CVEs that affect *package_name@version* in *ecosystem*."""
        self._lookup_count += 1
        matched: list[CVERecord] = []
        for cve in self._get_known_cves(package_name, ecosystem):
            if self._check_version_affected(version, cve.affected_versions):
                matched.append(cve)
        if matched:
            self._hit_count += len(matched)
            logger.warning(
                "cves_matched", package=package_name, version=version, count=len(matched)
            )
        return matched

    def _check_version_affected(self, version: str, affected_versions: list[str]) -> bool:
        """Check whether *version* falls within any affected range."""
        for constraint in affected_versions:
            match = re.match(r"^(<|<=|>=|>|==)(.+)$", constraint)
            if not match:
                continue
            op, target = match.group(1), match.group(2)
            cmp = self._compare_versions(version, target)
            if op == "<" and cmp < 0:
                return True
            if op == "<=" and cmp <= 0:
                return True
            if op == ">=" and cmp >= 0:
                return True
            if op == ">" and cmp > 0:
                return True
            if op == "==" and cmp == 0:
                return True
        return False

    def _get_known_cves(self, package_name: str, ecosystem: str) -> list[CVERecord]:
        """Materialise raw CVE entries into ``CVERecord`` objects."""
        entries = self._CVE_DATABASE.get(f"{ecosystem}:{package_name}", [])
        return [
            CVERecord(
                cve_id=e["cve_id"],
                severity=e["severity"],
                cvss_score=e["cvss_score"],
                description=e["description"],
                affected_versions=e["affected_versions"],
                fixed_versions=e["fixed_versions"],
                exploit_maturity=e["exploit_maturity"],
                published_date=datetime.fromisoformat(e["published_date"]),
            )
            for e in entries
        ]

    def get_cve_database_stats(self) -> dict[str, Any]:
        """Return statistics about the local CVE database."""
        return {
            "total_packages": len(self._CVE_DATABASE),
            "total_cves": sum(len(v) for v in self._CVE_DATABASE.values()),
            "lookups": self._lookup_count,
            "hits": self._hit_count,
        }

    @staticmethod
    def _compare_versions(a: str, b: str) -> int:
        """Compare two semver-like version strings."""

        def _parts(v: str) -> list[int]:
            return [int(x) for x in re.findall(r"\d+", v)]

        pa, pb = _parts(a), _parts(b)
        for x, y in zip(pa, pb, strict=False):
            if x != y:
                return x - y
        return len(pa) - len(pb)


# =============================================================================
# Risk Scorer
# =============================================================================


class RiskScorer:
    """Multi-factor risk scoring engine.

    Combines vulnerability severity, maintenance health, popularity, licence
    compliance and typosquatting likelihood into a normalised 0-100 score.
    """

    VULN_WEIGHT: float = 0.45
    MAINTENANCE_WEIGHT: float = 0.15
    LICENSE_WEIGHT: float = 0.10
    TYPOSQUAT_WEIGHT: float = 0.20
    TRANSITIVE_WEIGHT: float = 0.10

    KNOWN_PACKAGES: dict[str, list[str]] = {
        "npm": ["lodash", "express", "react", "webpack", "axios", "chalk", "moment", "debug"],
        "pypi": [
            "requests",
            "numpy",
            "pandas",
            "flask",
            "django",
            "boto3",
            "urllib3",
            "setuptools",
        ],
    }

    def __init__(self) -> None:
        logger.info("risk_scorer_initialized")

    def score_dependency(self, profile: DependencyRiskProfile) -> float:
        """Compute a composite risk score (0-100) for a single dependency."""
        vuln = self._vulnerability_score(profile.cve_records)
        maint = self._maintenance_score(profile)
        typo = self._typosquat_score(profile.package_name, profile.ecosystem)
        lic = self._license_score(profile.license_risk)

        composite = (
            vuln * self.VULN_WEIGHT
            + maint * self.MAINTENANCE_WEIGHT
            + lic * self.LICENSE_WEIGHT
            + typo * self.TYPOSQUAT_WEIGHT
            + profile.transitive_risk * self.TRANSITIVE_WEIGHT
        )
        score = min(max(composite, 0.0), 100.0)

        scores = {
            RiskCategory.VULNERABILITY: vuln,
            RiskCategory.MAINTENANCE: maint,
            RiskCategory.LICENSE: lic,
            RiskCategory.TYPOSQUATTING: typo,
        }
        profile.risk_category = max(scores, key=scores.get)  # type: ignore[arg-type]
        profile.risk_score = round(score, 2)
        profile.typosquat_risk = round(typo, 2)
        profile.maintenance_score = round(maint, 2)
        logger.debug("dependency_scored", package=profile.package_name, score=profile.risk_score)
        return profile.risk_score

    def score_project(self, profiles: list[DependencyRiskProfile]) -> float:
        """Compute an aggregate project-level risk score."""
        if not profiles:
            return 0.0
        for p in profiles:
            self.score_dependency(p)
        transitive = self._calculate_transitive_risk(profiles)
        for p in profiles:
            p.transitive_risk = round(transitive, 2)
        max_score = max(p.risk_score for p in profiles)
        avg_score = sum(p.risk_score for p in profiles) / len(profiles)
        return round(min(0.6 * max_score + 0.4 * avg_score, 100.0), 2)

    def _vulnerability_score(self, cves: list[CVERecord]) -> float:
        """Derive a 0-100 score from CVE records using CVSS base scores."""
        if not cves:
            return 0.0
        severity_w = {
            CVESeverity.CRITICAL: 1.0,
            CVESeverity.HIGH: 0.75,
            CVESeverity.MEDIUM: 0.4,
            CVESeverity.LOW: 0.15,
            CVESeverity.NONE: 0.0,
        }
        maturity_m = {
            ExploitMaturity.HIGH: 1.3,
            ExploitMaturity.FUNCTIONAL: 1.2,
            ExploitMaturity.PROOF_OF_CONCEPT: 1.0,
            ExploitMaturity.UNPROVEN: 0.8,
            ExploitMaturity.NOT_DEFINED: 0.9,
        }
        total = sum(
            (c.cvss_score / 10.0)
            * 100
            * severity_w.get(c.severity, 0.5)
            * maturity_m.get(c.exploit_maturity, 1.0)
            for c in cves
        )
        return min(total, 100.0)

    def _maintenance_score(self, profile: DependencyRiskProfile) -> float:
        """Estimate maintenance risk from popularity."""
        if profile.popularity_score >= 80:
            return max(10.0, profile.maintenance_score)
        if profile.popularity_score >= 50:
            return max(30.0, profile.maintenance_score)
        if profile.popularity_score >= 20:
            return max(50.0, profile.maintenance_score)
        return max(70.0, profile.maintenance_score)

    def _typosquat_score(self, name: str, ecosystem: str) -> float:
        """Detect potential typosquatting via Levenshtein distance to known packages."""
        known = self.KNOWN_PACKAGES.get(ecosystem, [])
        if name in known:
            return 0.0
        min_dist = min((self._levenshtein_distance(name, k) for k in known), default=float("inf"))
        if min_dist <= 1:
            return 95.0
        if min_dist == 2:
            return 70.0
        if min_dist == 3:
            return 35.0
        return 0.0

    def _calculate_transitive_risk(self, profiles: list[DependencyRiskProfile]) -> float:
        """Estimate transitive risk as 30% of average direct risk."""
        if not profiles:
            return 0.0
        return round(sum(p.risk_score for p in profiles) / len(profiles) * 0.3, 2)

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute the Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return RiskScorer._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                curr_row.append(
                    min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (0 if c1 == c2 else 1))
                )
            prev_row = curr_row
        return prev_row[-1]

    @staticmethod
    def _license_score(license_risk: str) -> float:
        """Map licence risk labels to a numeric score."""
        return {"low": 0.0, "medium": 35.0, "high": 70.0, "critical": 100.0}.get(license_risk, 50.0)


# =============================================================================
# SBOM Generator
# =============================================================================


class SBOMGenerator:
    """Generate SBOM documents in CycloneDX or SPDX format."""

    _CYCLONEDX_SPEC = "1.5"
    _SPDX_SPEC = "SPDX-2.3"
    _PURL_TYPES: dict[str, str] = {
        "npm": "npm",
        "pypi": "pypi",
        "maven": "maven",
        "cargo": "cargo",
        "go": "golang",
        "nuget": "nuget",
    }

    def __init__(self) -> None:
        logger.info("sbom_generator_initialized")

    def generate(
        self,
        project_name: str,
        components: list[SBOMComponent],
        format: SBOMFormat = SBOMFormat.CYCLONEDX,
    ) -> SBOMDocument:
        """Build an ``SBOMDocument`` from a list of components."""
        spec = self._CYCLONEDX_SPEC if format == SBOMFormat.CYCLONEDX else self._SPDX_SPEC
        high_risk = sum(
            1 for c in components if c.risk_profile and c.risk_profile.risk_score >= 70.0
        )
        total_risk = sum(c.risk_profile.risk_score for c in components if c.risk_profile)
        sbom = SBOMDocument(
            format=format,
            version=spec,
            created_at=datetime.utcnow(),
            project_name=project_name,
            components=components,
            total_risk_score=round(total_risk, 2),
            high_risk_count=high_risk,
        )
        logger.info(
            "sbom_generated", project=project_name, format=format.value, components=len(components)
        )
        return sbom

    def _generate_purl(self, name: str, version: str, ecosystem: str) -> str:
        """Build a Package URL (purl) per the purl-spec."""
        purl_type = self._PURL_TYPES.get(ecosystem, ecosystem)
        return f"pkg:{purl_type}/{name}@{version}"

    def export_json(self, sbom: SBOMDocument) -> str:
        """Serialise an SBOM document to JSON."""
        if sbom.format == SBOMFormat.CYCLONEDX:
            return self._export_cyclonedx_json(sbom)
        return self._export_spdx_json(sbom)

    def export_xml(self, sbom: SBOMDocument) -> str:
        """Serialise an SBOM document to minimal XML."""
        if sbom.format == SBOMFormat.CYCLONEDX:
            return self._export_cyclonedx_xml(sbom)
        return self._export_spdx_xml(sbom)

    def _export_cyclonedx_json(self, sbom: SBOMDocument) -> str:
        doc: dict[str, Any] = {
            "bomFormat": "CycloneDX",
            "specVersion": sbom.version,
            "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "version": 1,
            "metadata": {
                "timestamp": sbom.created_at.isoformat(),
                "component": {"type": "application", "name": sbom.project_name},
            },
            "components": [],
        }
        for comp in sbom.components:
            entry: dict[str, Any] = {
                "type": "library",
                "name": comp.name,
                "version": comp.version,
                "purl": comp.purl,
            }
            if comp.licenses:
                entry["licenses"] = [{"license": {"id": lic}} for lic in comp.licenses]
            if comp.supplier:
                entry["supplier"] = {"name": comp.supplier}
            if comp.checksums:
                entry["hashes"] = [
                    {"alg": alg, "content": val} for alg, val in comp.checksums.items()
                ]
            doc["components"].append(entry)
        return json.dumps(doc, indent=2)

    def _export_spdx_json(self, sbom: SBOMDocument) -> str:
        doc: dict[str, Any] = {
            "spdxVersion": sbom.version,
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": sbom.project_name,
            "documentNamespace": f"https://spdx.org/spdxdocs/{sbom.project_name}-{uuid.uuid4()}",
            "creationInfo": {
                "created": sbom.created_at.isoformat(),
                "creators": ["Tool: codeverify-supply-chain-risk"],
            },
            "packages": [],
        }
        for comp in sbom.components:
            pkg: dict[str, Any] = {
                "SPDXID": f"SPDXRef-Package-{comp.name}-{comp.version}".replace(".", "-"),
                "name": comp.name,
                "versionInfo": comp.version,
                "downloadLocation": "NOASSERTION",
                "licenseConcluded": comp.licenses[0] if comp.licenses else "NOASSERTION",
            }
            if comp.purl:
                pkg["externalRefs"] = [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": comp.purl,
                    }
                ]
            if comp.supplier:
                pkg["supplier"] = f"Organization: {comp.supplier}"
            doc["packages"].append(pkg)
        return json.dumps(doc, indent=2)

    def _export_cyclonedx_xml(self, sbom: SBOMDocument) -> str:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<bom xmlns="http://cyclonedx.org/schema/bom/{sbom.version}" version="1">',
            "  <components>",
        ]
        for comp in sbom.components:
            lines.extend(
                [
                    '    <component type="library">',
                    f"      <name>{comp.name}</name>",
                    f"      <version>{comp.version}</version>",
                    f"      <purl>{comp.purl}</purl>",
                    "    </component>",
                ]
            )
        lines += ["  </components>", "</bom>"]
        return "\n".join(lines)

    def _export_spdx_xml(self, sbom: SBOMDocument) -> str:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<SpdxDocument xmlns="https://spdx.org/rdf/terms">',
            f"  <name>{sbom.project_name}</name>",
        ]
        for comp in sbom.components:
            lines.extend(
                [
                    "  <package>",
                    f"    <name>{comp.name}</name>",
                    f"    <versionInfo>{comp.version}</versionInfo>",
                    "  </package>",
                ]
            )
        lines.append("</SpdxDocument>")
        return "\n".join(lines)


# =============================================================================
# Supply Chain Risk Analyzer (orchestrator)
# =============================================================================


class SupplyChainRiskAnalyzer:
    """High-level orchestrator that ties correlation, scoring and SBOM generation.

    Usage::

        analyzer = SupplyChainRiskAnalyzer()
        report = analyzer.analyze("my-project", [
            {"name": "lodash", "version": "4.17.19", "ecosystem": "npm"},
        ])
    """

    def __init__(self) -> None:
        self._correlator = CVECorrelator()
        self._scorer = RiskScorer()
        self._sbom_gen = SBOMGenerator()
        logger.info("supply_chain_risk_analyzer_initialized")

    def analyze(
        self, project_name: str, dependencies: list[dict[str, Any]]
    ) -> SupplyChainRiskReport:
        """Run a full supply-chain risk analysis."""
        profiles: list[DependencyRiskProfile] = []
        for dep in dependencies:
            name, version = dep["name"], dep["version"]
            ecosystem = dep.get("ecosystem", "npm")
            cves = self._correlator.correlate(name, version, ecosystem)
            profiles.append(
                DependencyRiskProfile(
                    package_name=name,
                    version=version,
                    ecosystem=ecosystem,
                    cve_records=cves,
                    license_risk=dep.get("license_risk", "low"),
                    popularity_score=float(dep.get("popularity_score", 50)),
                )
            )

        project_score = self._scorer.score_project(profiles)
        critical = sum(
            1 for p in profiles for c in p.cve_records if c.severity == CVESeverity.CRITICAL
        )
        high = sum(1 for p in profiles for c in p.cve_records if c.severity == CVESeverity.HIGH)

        report = SupplyChainRiskReport(
            project_name=project_name,
            scan_date=datetime.utcnow(),
            total_dependencies=len(dependencies),
            direct_dependencies=sum(1 for d in dependencies if d.get("direct", True)),
            risk_score=project_score,
            critical_cves=critical,
            high_cves=high,
            dependency_profiles=profiles,
            recommendations=self._build_recommendations(profiles),
        )
        logger.info("supply_chain_analysis_complete", project=project_name, score=project_score)
        return report

    def generate_sbom(
        self,
        project_name: str,
        dependencies: list[dict[str, Any]],
        format: SBOMFormat = SBOMFormat.CYCLONEDX,
    ) -> SBOMDocument:
        """Generate an SBOM for the given dependencies."""
        components: list[SBOMComponent] = []
        for dep in dependencies:
            name, version = dep["name"], dep["version"]
            ecosystem = dep.get("ecosystem", "npm")
            purl = self._sbom_gen._generate_purl(name, version, ecosystem)
            cves = self._correlator.correlate(name, version, ecosystem)
            profile = DependencyRiskProfile(
                package_name=name, version=version, ecosystem=ecosystem, cve_records=cves
            )
            self._scorer.score_dependency(profile)
            components.append(
                SBOMComponent(
                    name=name,
                    version=version,
                    ecosystem=ecosystem,
                    purl=purl,
                    licenses=dep.get("licenses", []),
                    supplier=dep.get("supplier"),
                    checksums=dep.get("checksums", {}),
                    dependencies=dep.get("dependencies", []),
                    risk_profile=profile,
                )
            )
        return self._sbom_gen.generate(project_name, components, format)

    def get_remediation_plan(self, report: SupplyChainRiskReport) -> list[dict[str, Any]]:
        """Produce a prioritised remediation plan from a risk report."""
        plan: list[dict[str, Any]] = []
        for profile in sorted(report.dependency_profiles, key=lambda p: p.risk_score, reverse=True):
            if profile.risk_score < 10.0:
                continue
            actions: list[str] = []
            for cve in profile.cve_records:
                if cve.fixed_versions:
                    actions.append(
                        f"Upgrade {profile.package_name} to {cve.fixed_versions[0]} "
                        f"(fixes {cve.cve_id}, CVSS {cve.cvss_score})"
                    )
            if profile.typosquat_risk >= 70.0:
                actions.append(
                    f"Verify {profile.package_name} is not a typosquat (risk: {profile.typosquat_risk}%)"
                )
            if profile.license_risk in ("high", "critical"):
                actions.append(
                    f"Review licence compliance for {profile.package_name} (risk: {profile.license_risk})"
                )
            if actions:
                plan.append(
                    {
                        "package": profile.package_name,
                        "version": profile.version,
                        "risk_score": profile.risk_score,
                        "risk_category": profile.risk_category.value,
                        "actions": actions,
                    }
                )
        logger.info("remediation_plan_generated", items=len(plan))
        return plan

    def compare_reports(
        self, old_report: SupplyChainRiskReport, new_report: SupplyChainRiskReport
    ) -> dict[str, Any]:
        """Compare two risk reports and summarise the delta."""
        old_pkgs = {p.package_name: p for p in old_report.dependency_profiles}
        new_pkgs = {p.package_name: p for p in new_report.dependency_profiles}

        added = [n for n in new_pkgs if n not in old_pkgs]
        removed = [o for o in old_pkgs if o not in new_pkgs]
        changed: list[dict[str, Any]] = []
        for name in set(old_pkgs) & set(new_pkgs):
            old_p, new_p = old_pkgs[name], new_pkgs[name]
            if old_p.risk_score != new_p.risk_score or old_p.version != new_p.version:
                changed.append(
                    {
                        "package": name,
                        "old_version": old_p.version,
                        "new_version": new_p.version,
                        "old_score": old_p.risk_score,
                        "new_score": new_p.risk_score,
                        "delta": round(new_p.risk_score - old_p.risk_score, 2),
                    }
                )

        score_delta = round(new_report.risk_score - old_report.risk_score, 2)
        trend = "worsening" if score_delta > 5 else ("improving" if score_delta < -5 else "stable")

        logger.info(
            "reports_compared",
            old_score=old_report.risk_score,
            new_score=new_report.risk_score,
            trend=trend,
        )
        return {
            "old_score": old_report.risk_score,
            "new_score": new_report.risk_score,
            "score_delta": score_delta,
            "trend": trend,
            "dependencies_added": added,
            "dependencies_removed": removed,
            "dependencies_changed": changed,
            "new_critical_cves": new_report.critical_cves - old_report.critical_cves,
            "new_high_cves": new_report.high_cves - old_report.high_cves,
        }

    def _build_recommendations(self, profiles: list[DependencyRiskProfile]) -> list[str]:
        """Generate human-readable recommendations from scored profiles."""
        recs: list[str] = []
        critical_cves = [
            (p, c) for p in profiles for c in p.cve_records if c.severity == CVESeverity.CRITICAL
        ]
        if critical_cves:
            recs.append(
                f"URGENT: {len(critical_cves)} critical CVE(s) found — upgrade affected packages immediately."
            )
        typo_suspects = [p for p in profiles if p.typosquat_risk >= 70.0]
        if typo_suspects:
            names = ", ".join(p.package_name for p in typo_suspects)
            recs.append(
                f"Potential typosquatting detected for: {names}. Verify package authenticity before use."
            )
        high_risk = [p for p in profiles if p.risk_score >= 70.0]
        if high_risk:
            recs.append(
                f"{len(high_risk)} high-risk dependency(ies). Consider pinning versions and enabling alerts."
            )
        license_issues = [p for p in profiles if p.license_risk in ("high", "critical")]
        if license_issues:
            recs.append(
                "Licence compliance issues found. Review high-risk licences against your organisation's policy."
            )
        if not recs:
            recs.append("No significant supply-chain risks detected.")
        return recs
