"""
Dependency Vulnerability Scanner

Deep integration with dependency scanning to identify vulnerable packages
and their transitive dependencies. Includes CVE database integration,
dependency graph analysis, and upgrade path suggestions.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4


class VulnerabilitySeverity(str, Enum):
    """Severity levels for vulnerabilities."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class PackageEcosystem(str, Enum):
    """Package ecosystems."""
    NPM = "npm"
    PYPI = "pypi"
    MAVEN = "maven"
    NUGET = "nuget"
    RUBYGEMS = "rubygems"
    GO = "go"
    CARGO = "cargo"
    COMPOSER = "composer"


class DependencyType(str, Enum):
    """Types of dependencies."""
    DIRECT = "direct"
    TRANSITIVE = "transitive"
    DEV = "dev"
    OPTIONAL = "optional"


class UpgradeRisk(str, Enum):
    """Risk level of an upgrade."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BREAKING = "breaking"


@dataclass
class CVE:
    """Common Vulnerabilities and Exposures record."""
    id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    cvss_score: float
    cvss_vector: Optional[str]
    cwe_ids: List[str]
    published_date: datetime
    modified_date: datetime
    references: List[str]
    affected_versions: List[str]
    fixed_versions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "cvss_score": self.cvss_score,
            "cvss_vector": self.cvss_vector,
            "cwe_ids": self.cwe_ids,
            "published_date": self.published_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
            "references": self.references[:5],
            "affected_versions": self.affected_versions,
            "fixed_versions": self.fixed_versions,
        }


@dataclass
class Package:
    """A software package/dependency."""
    name: str
    version: str
    ecosystem: PackageEcosystem
    dependency_type: DependencyType
    source_file: str
    parent: Optional[str] = None
    license: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "ecosystem": self.ecosystem.value,
            "dependency_type": self.dependency_type.value,
            "source_file": self.source_file,
            "parent": self.parent,
            "license": self.license,
        }


@dataclass
class VulnerablePackage:
    """A package with known vulnerabilities."""
    package: Package
    vulnerabilities: List[CVE]
    highest_severity: VulnerabilitySeverity
    total_cves: int
    exploit_available: bool
    in_kev: bool  # Known Exploited Vulnerabilities catalog

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "package": self.package.to_dict(),
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "highest_severity": self.highest_severity.value,
            "total_cves": self.total_cves,
            "exploit_available": self.exploit_available,
            "in_kev": self.in_kev,
        }


@dataclass
class UpgradePath:
    """A suggested upgrade path for a package."""
    package_name: str
    current_version: str
    target_version: str
    intermediate_versions: List[str]
    risk: UpgradeRisk
    breaking_changes: List[str]
    vulnerabilities_fixed: int
    release_date: Optional[datetime]
    changelog_url: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "package_name": self.package_name,
            "current_version": self.current_version,
            "target_version": self.target_version,
            "intermediate_versions": self.intermediate_versions,
            "risk": self.risk.value,
            "breaking_changes": self.breaking_changes,
            "vulnerabilities_fixed": self.vulnerabilities_fixed,
            "release_date": self.release_date.isoformat() if self.release_date else None,
            "changelog_url": self.changelog_url,
        }


@dataclass
class DependencyNode:
    """Node in the dependency graph."""
    package: Package
    children: List[str] = field(default_factory=list)  # Package names
    depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "package": self.package.to_dict(),
            "children": self.children,
            "depth": self.depth,
        }


@dataclass
class ScanResult:
    """Result of a dependency scan."""
    id: str
    project_name: str
    ecosystem: PackageEcosystem
    scanned_at: datetime
    total_packages: int
    vulnerable_packages: List[VulnerablePackage]
    upgrade_paths: List[UpgradePath]
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    dependency_graph: Dict[str, DependencyNode]
    scan_duration_ms: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "project_name": self.project_name,
            "ecosystem": self.ecosystem.value,
            "scanned_at": self.scanned_at.isoformat(),
            "total_packages": self.total_packages,
            "vulnerable_packages": [v.to_dict() for v in self.vulnerable_packages],
            "upgrade_paths": [u.to_dict() for u in self.upgrade_paths],
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "dependency_graph": {k: v.to_dict() for k, v in self.dependency_graph.items()},
            "scan_duration_ms": self.scan_duration_ms,
        }


class CVEDatabase:
    """Mock CVE database for vulnerability lookups."""

    # Sample vulnerability data for common packages
    KNOWN_VULNERABILITIES: Dict[str, List[Dict[str, Any]]] = {
        "lodash": [
            {
                "id": "CVE-2021-23337",
                "title": "Command Injection in lodash",
                "description": "lodash before 4.17.21 is vulnerable to Command Injection",
                "severity": "high",
                "cvss": 7.2,
                "affected": "<4.17.21",
                "fixed": "4.17.21",
            },
            {
                "id": "CVE-2020-8203",
                "title": "Prototype Pollution in lodash",
                "description": "Prototype pollution in lodash before 4.17.19",
                "severity": "high",
                "cvss": 7.4,
                "affected": "<4.17.19",
                "fixed": "4.17.19",
            },
        ],
        "requests": [
            {
                "id": "CVE-2023-32681",
                "title": "Unintended Request Proxy",
                "description": "Requests before 2.31.0 is vulnerable to unintended request proxy",
                "severity": "medium",
                "cvss": 5.9,
                "affected": "<2.31.0",
                "fixed": "2.31.0",
            },
        ],
        "axios": [
            {
                "id": "CVE-2021-3749",
                "title": "Server-Side Request Forgery",
                "description": "SSRF vulnerability in axios before 0.21.2",
                "severity": "high",
                "cvss": 7.5,
                "affected": "<0.21.2",
                "fixed": "0.21.2",
            },
        ],
        "express": [
            {
                "id": "CVE-2022-24999",
                "title": "Open Redirect",
                "description": "Open redirect vulnerability in Express before 4.17.3",
                "severity": "medium",
                "cvss": 6.1,
                "affected": "<4.17.3",
                "fixed": "4.17.3",
            },
        ],
        "django": [
            {
                "id": "CVE-2023-36053",
                "title": "Potential ReDoS",
                "description": "Django before 4.2.3 has a potential ReDoS vulnerability",
                "severity": "medium",
                "cvss": 5.3,
                "affected": "<4.2.3",
                "fixed": "4.2.3",
            },
        ],
        "flask": [
            {
                "id": "CVE-2023-30861",
                "title": "Improper Session Handling",
                "description": "Flask before 2.3.2 is vulnerable to improper session handling",
                "severity": "high",
                "cvss": 7.5,
                "affected": "<2.3.2",
                "fixed": "2.3.2",
            },
        ],
    }

    def lookup(self, package_name: str, version: str) -> List[CVE]:
        """Look up vulnerabilities for a package version."""
        vulns = self.KNOWN_VULNERABILITIES.get(package_name.lower(), [])
        applicable = []

        for vuln in vulns:
            if self._is_version_affected(version, vuln.get("affected", "")):
                severity = VulnerabilitySeverity(vuln["severity"])
                cve = CVE(
                    id=vuln["id"],
                    title=vuln["title"],
                    description=vuln["description"],
                    severity=severity,
                    cvss_score=vuln["cvss"],
                    cvss_vector=None,
                    cwe_ids=[],
                    published_date=datetime.now() - timedelta(days=180),
                    modified_date=datetime.now() - timedelta(days=30),
                    references=[],
                    affected_versions=[vuln["affected"]],
                    fixed_versions=[vuln["fixed"]],
                )
                applicable.append(cve)

        return applicable

    def _is_version_affected(self, version: str, affected_spec: str) -> bool:
        """Check if a version is affected by a vulnerability."""
        if not affected_spec:
            return False

        # Simple version comparison (in production, use proper semver)
        if affected_spec.startswith("<"):
            target = affected_spec[1:]
            return self._compare_versions(version, target) < 0
        elif affected_spec.startswith("<="):
            target = affected_spec[2:]
            return self._compare_versions(version, target) <= 0
        elif affected_spec.startswith(">="):
            target = affected_spec[2:]
            return self._compare_versions(version, target) >= 0
        elif affected_spec.startswith(">"):
            target = affected_spec[1:]
            return self._compare_versions(version, target) > 0

        return version == affected_spec

    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings."""
        def normalize(v: str) -> List[int]:
            return [int(x) for x in re.sub(r'[^0-9.]', '', v).split('.') if x]

        v1_parts = normalize(v1)
        v2_parts = normalize(v2)

        for i in range(max(len(v1_parts), len(v2_parts))):
            p1 = v1_parts[i] if i < len(v1_parts) else 0
            p2 = v2_parts[i] if i < len(v2_parts) else 0
            if p1 < p2:
                return -1
            elif p1 > p2:
                return 1

        return 0


class DependencyParser:
    """Parses dependency files to extract packages."""

    def parse(
        self,
        file_path: str,
        content: str,
    ) -> Tuple[PackageEcosystem, List[Package]]:
        """Parse a dependency file and extract packages."""
        if "package.json" in file_path:
            return PackageEcosystem.NPM, self._parse_npm(content, file_path)
        elif "requirements" in file_path or file_path.endswith(".txt"):
            return PackageEcosystem.PYPI, self._parse_pip(content, file_path)
        elif "pyproject.toml" in file_path:
            return PackageEcosystem.PYPI, self._parse_pyproject(content, file_path)
        elif "Gemfile" in file_path:
            return PackageEcosystem.RUBYGEMS, self._parse_gemfile(content, file_path)
        elif "go.mod" in file_path:
            return PackageEcosystem.GO, self._parse_gomod(content, file_path)
        elif "Cargo.toml" in file_path:
            return PackageEcosystem.CARGO, self._parse_cargo(content, file_path)
        elif "pom.xml" in file_path:
            return PackageEcosystem.MAVEN, self._parse_maven(content, file_path)

        return PackageEcosystem.NPM, []

    def _parse_npm(self, content: str, file_path: str) -> List[Package]:
        """Parse package.json."""
        packages: List[Package] = []

        try:
            import json
            data = json.loads(content)

            # Parse dependencies
            for name, version in data.get("dependencies", {}).items():
                packages.append(Package(
                    name=name,
                    version=self._clean_version(version),
                    ecosystem=PackageEcosystem.NPM,
                    dependency_type=DependencyType.DIRECT,
                    source_file=file_path,
                ))

            # Parse devDependencies
            for name, version in data.get("devDependencies", {}).items():
                packages.append(Package(
                    name=name,
                    version=self._clean_version(version),
                    ecosystem=PackageEcosystem.NPM,
                    dependency_type=DependencyType.DEV,
                    source_file=file_path,
                ))

        except Exception:
            pass

        return packages

    def _parse_pip(self, content: str, file_path: str) -> List[Package]:
        """Parse requirements.txt."""
        packages: List[Package] = []

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            # Parse package==version or package>=version format
            match = re.match(r"([a-zA-Z0-9_-]+)(?:[=<>!]+)([0-9.]+)?", line)
            if match:
                name = match.group(1)
                version = match.group(2) or "unknown"
                packages.append(Package(
                    name=name,
                    version=version,
                    ecosystem=PackageEcosystem.PYPI,
                    dependency_type=DependencyType.DIRECT,
                    source_file=file_path,
                ))

        return packages

    def _parse_pyproject(self, content: str, file_path: str) -> List[Package]:
        """Parse pyproject.toml (simplified)."""
        packages: List[Package] = []

        # Simple regex-based parsing
        deps_match = re.search(r'dependencies\s*=\s*\[([^\]]+)\]', content, re.MULTILINE)
        if deps_match:
            deps_content = deps_match.group(1)
            for dep in re.findall(r'"([^"]+)"', deps_content):
                match = re.match(r"([a-zA-Z0-9_-]+)(?:[=<>!]+)([0-9.]+)?", dep)
                if match:
                    packages.append(Package(
                        name=match.group(1),
                        version=match.group(2) or "unknown",
                        ecosystem=PackageEcosystem.PYPI,
                        dependency_type=DependencyType.DIRECT,
                        source_file=file_path,
                    ))

        return packages

    def _parse_gemfile(self, content: str, file_path: str) -> List[Package]:
        """Parse Gemfile."""
        packages: List[Package] = []

        for match in re.finditer(r"gem\s+['\"]([^'\"]+)['\"](?:,\s*['\"]([^'\"]+)['\"])?", content):
            name = match.group(1)
            version = match.group(2) or "unknown"
            packages.append(Package(
                name=name,
                version=self._clean_version(version),
                ecosystem=PackageEcosystem.RUBYGEMS,
                dependency_type=DependencyType.DIRECT,
                source_file=file_path,
            ))

        return packages

    def _parse_gomod(self, content: str, file_path: str) -> List[Package]:
        """Parse go.mod."""
        packages: List[Package] = []

        for line in content.split("\n"):
            match = re.match(r"\s*(\S+)\s+v([0-9.]+)", line)
            if match:
                packages.append(Package(
                    name=match.group(1),
                    version=match.group(2),
                    ecosystem=PackageEcosystem.GO,
                    dependency_type=DependencyType.DIRECT,
                    source_file=file_path,
                ))

        return packages

    def _parse_cargo(self, content: str, file_path: str) -> List[Package]:
        """Parse Cargo.toml."""
        packages: List[Package] = []

        for match in re.finditer(r'(\w+)\s*=\s*["\']([0-9.]+)["\']', content):
            packages.append(Package(
                name=match.group(1),
                version=match.group(2),
                ecosystem=PackageEcosystem.CARGO,
                dependency_type=DependencyType.DIRECT,
                source_file=file_path,
            ))

        return packages

    def _parse_maven(self, content: str, file_path: str) -> List[Package]:
        """Parse pom.xml."""
        packages: List[Package] = []

        # Simple regex-based parsing
        for match in re.finditer(
            r"<dependency>.*?<groupId>([^<]+)</groupId>.*?<artifactId>([^<]+)</artifactId>.*?<version>([^<]+)</version>",
            content,
            re.DOTALL,
        ):
            packages.append(Package(
                name=f"{match.group(1)}:{match.group(2)}",
                version=match.group(3),
                ecosystem=PackageEcosystem.MAVEN,
                dependency_type=DependencyType.DIRECT,
                source_file=file_path,
            ))

        return packages

    def _clean_version(self, version: str) -> str:
        """Clean version string."""
        # Remove ^ ~ = < > prefixes
        return re.sub(r'^[\^~=<>]+', '', version)


class UpgradeAdvisor:
    """Suggests upgrade paths for vulnerable packages."""

    def suggest_upgrades(
        self,
        vulnerable_packages: List[VulnerablePackage],
    ) -> List[UpgradePath]:
        """Suggest upgrade paths for vulnerable packages."""
        upgrades: List[UpgradePath] = []

        for vuln_pkg in vulnerable_packages:
            pkg = vuln_pkg.package

            # Find the best fixed version
            fixed_versions = set()
            for cve in vuln_pkg.vulnerabilities:
                fixed_versions.update(cve.fixed_versions)

            if not fixed_versions:
                continue

            # Get the minimum fixed version
            target_version = max(fixed_versions)

            # Determine upgrade risk
            risk = self._assess_risk(pkg.version, target_version)

            # Check for breaking changes (simplified)
            breaking_changes = self._check_breaking_changes(pkg, target_version)

            upgrade = UpgradePath(
                package_name=pkg.name,
                current_version=pkg.version,
                target_version=target_version,
                intermediate_versions=[],
                risk=risk,
                breaking_changes=breaking_changes,
                vulnerabilities_fixed=len(vuln_pkg.vulnerabilities),
                release_date=datetime.now() - timedelta(days=30),
                changelog_url=f"https://github.com/{pkg.name}/releases/tag/v{target_version}",
            )

            upgrades.append(upgrade)

        return upgrades

    def _assess_risk(self, current: str, target: str) -> UpgradeRisk:
        """Assess risk of an upgrade."""
        current_parts = [int(x) for x in re.findall(r'\d+', current)]
        target_parts = [int(x) for x in re.findall(r'\d+', target)]

        if len(current_parts) >= 1 and len(target_parts) >= 1:
            # Major version change
            if target_parts[0] > current_parts[0]:
                return UpgradeRisk.BREAKING

        if len(current_parts) >= 2 and len(target_parts) >= 2:
            # Minor version change
            if target_parts[0] == current_parts[0] and target_parts[1] > current_parts[1]:
                return UpgradeRisk.MEDIUM

        return UpgradeRisk.LOW

    def _check_breaking_changes(self, pkg: Package, target_version: str) -> List[str]:
        """Check for breaking changes (simplified)."""
        breaking: List[str] = []

        # In production, this would query changelogs/release notes
        current_parts = [int(x) for x in re.findall(r'\d+', pkg.version)]
        target_parts = [int(x) for x in re.findall(r'\d+', target_version)]

        if len(current_parts) >= 1 and len(target_parts) >= 1:
            if target_parts[0] > current_parts[0]:
                breaking.append(f"Major version upgrade from {current_parts[0]} to {target_parts[0]}")

        return breaking


class DependencyVulnerabilityScanner:
    """Main scanner for dependency vulnerabilities."""

    def __init__(self):
        self.cve_db = CVEDatabase()
        self.parser = DependencyParser()
        self.advisor = UpgradeAdvisor()

        self.scan_results: Dict[str, ScanResult] = {}

    async def scan(
        self,
        project_name: str,
        files: Dict[str, str],  # file_path -> content
    ) -> ScanResult:
        """Scan dependencies for vulnerabilities."""
        start_time = datetime.now()

        all_packages: List[Package] = []
        ecosystem = PackageEcosystem.NPM  # Default

        # Parse all dependency files
        for file_path, content in files.items():
            parsed_ecosystem, packages = self.parser.parse(file_path, content)
            if packages:
                ecosystem = parsed_ecosystem
                all_packages.extend(packages)

        # Build dependency graph
        dep_graph = self._build_dependency_graph(all_packages)

        # Find vulnerabilities
        vulnerable_packages: List[VulnerablePackage] = []
        severity_counts: Dict[str, int] = defaultdict(int)

        for pkg in all_packages:
            cves = self.cve_db.lookup(pkg.name, pkg.version)

            if cves:
                highest_sev = max(cves, key=lambda c: self._severity_rank(c.severity))

                vuln_pkg = VulnerablePackage(
                    package=pkg,
                    vulnerabilities=cves,
                    highest_severity=highest_sev.severity,
                    total_cves=len(cves),
                    exploit_available=False,  # Would check exploit DB
                    in_kev=False,  # Would check CISA KEV
                )

                vulnerable_packages.append(vuln_pkg)
                severity_counts[highest_sev.severity.value] += 1

        # Get upgrade suggestions
        upgrade_paths = self.advisor.suggest_upgrades(vulnerable_packages)

        # Calculate duration
        duration = int((datetime.now() - start_time).total_seconds() * 1000)

        result = ScanResult(
            id=str(uuid4()),
            project_name=project_name,
            ecosystem=ecosystem,
            scanned_at=datetime.now(),
            total_packages=len(all_packages),
            vulnerable_packages=vulnerable_packages,
            upgrade_paths=upgrade_paths,
            critical_count=severity_counts.get("critical", 0),
            high_count=severity_counts.get("high", 0),
            medium_count=severity_counts.get("medium", 0),
            low_count=severity_counts.get("low", 0),
            dependency_graph=dep_graph,
            scan_duration_ms=duration,
        )

        self.scan_results[result.id] = result
        return result

    def _build_dependency_graph(
        self,
        packages: List[Package],
    ) -> Dict[str, DependencyNode]:
        """Build a dependency graph."""
        graph: Dict[str, DependencyNode] = {}

        for pkg in packages:
            key = f"{pkg.name}@{pkg.version}"
            node = DependencyNode(
                package=pkg,
                depth=0 if pkg.dependency_type == DependencyType.DIRECT else 1,
            )

            if pkg.parent:
                parent_key = pkg.parent
                if parent_key in graph:
                    graph[parent_key].children.append(key)

            graph[key] = node

        return graph

    def _severity_rank(self, severity: VulnerabilitySeverity) -> int:
        """Get numeric rank for severity."""
        ranks = {
            VulnerabilitySeverity.CRITICAL: 4,
            VulnerabilitySeverity.HIGH: 3,
            VulnerabilitySeverity.MEDIUM: 2,
            VulnerabilitySeverity.LOW: 1,
            VulnerabilitySeverity.INFORMATIONAL: 0,
        }
        return ranks.get(severity, 0)

    def get_result(self, result_id: str) -> Optional[ScanResult]:
        """Get a scan result by ID."""
        return self.scan_results.get(result_id)

    def get_vulnerable_packages(
        self,
        result_id: str,
        min_severity: Optional[str] = None,
    ) -> List[VulnerablePackage]:
        """Get vulnerable packages from a scan result."""
        result = self.scan_results.get(result_id)
        if not result:
            return []

        if not min_severity:
            return result.vulnerable_packages

        min_rank = self._severity_rank(VulnerabilitySeverity(min_severity))

        return [
            v for v in result.vulnerable_packages
            if self._severity_rank(v.highest_severity) >= min_rank
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get scanner statistics."""
        total_vulns = 0
        severity_totals: Dict[str, int] = defaultdict(int)

        for result in self.scan_results.values():
            total_vulns += len(result.vulnerable_packages)
            severity_totals["critical"] += result.critical_count
            severity_totals["high"] += result.high_count
            severity_totals["medium"] += result.medium_count
            severity_totals["low"] += result.low_count

        return {
            "total_scans": len(self.scan_results),
            "total_vulnerabilities": total_vulns,
            "severity_breakdown": dict(severity_totals),
            "ecosystems_scanned": list(set(r.ecosystem.value for r in self.scan_results.values())),
        }
