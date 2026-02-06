"""Supply Chain Verification - Verify third-party dependencies.

This module extends verification to third-party dependencies, using differential
analysis to detect malicious changes in package updates.

Key features:
1. Dependency Extraction: Parse and resolve dependency trees
2. Differential Analysis: Compare package versions for behavioral changes
3. Malicious Pattern Detection: Identify typosquatting, code injection
4. Lockfile Verification: Verify integrity of dependency locks
"""

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

import structlog

logger = structlog.get_logger()


class PackageEcosystem(str, Enum):
    """Package ecosystem."""

    NPM = "npm"
    PYPI = "pypi"
    MAVEN = "maven"
    CARGO = "cargo"
    GO = "go"
    NUGET = "nuget"


class RiskLevel(str, Enum):
    """Risk level for a dependency."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class ThreatType(str, Enum):
    """Type of supply chain threat."""

    TYPOSQUATTING = "typosquatting"
    DEPENDENCY_CONFUSION = "dependency_confusion"
    MALICIOUS_UPDATE = "malicious_update"
    COMPROMISED_MAINTAINER = "compromised_maintainer"
    CODE_INJECTION = "code_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    BACKDOOR = "backdoor"
    CRYPTO_MINER = "crypto_miner"
    KNOWN_VULNERABILITY = "known_vulnerability"
    LICENSE_VIOLATION = "license_violation"
    UNMAINTAINED = "unmaintained"


@dataclass
class PackageInfo:
    """Information about a package."""

    name: str
    version: str
    ecosystem: PackageEcosystem
    dependencies: list["PackageInfo"] = field(default_factory=list)
    dev_dependency: bool = False
    checksum: str | None = None
    source_url: str | None = None
    homepage: str | None = None
    license: str | None = None
    maintainers: list[str] = field(default_factory=list)
    last_updated: datetime | None = None
    download_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "ecosystem": self.ecosystem.value,
            "dependencies": [d.name for d in self.dependencies],
            "dev_dependency": self.dev_dependency,
            "checksum": self.checksum,
            "license": self.license,
        }


@dataclass
class SupplyChainThreat:
    """A detected supply chain threat."""

    id: str
    threat_type: ThreatType
    package: PackageInfo
    risk_level: RiskLevel
    title: str
    description: str
    evidence: list[str]
    remediation: str
    cve_ids: list[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    false_positive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "threat_type": self.threat_type.value,
            "package": self.package.to_dict(),
            "risk_level": self.risk_level.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "remediation": self.remediation,
            "cve_ids": self.cve_ids,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class DiffResult:
    """Result of differential analysis between package versions."""

    old_version: str
    new_version: str
    files_added: list[str]
    files_removed: list[str]
    files_modified: list[str]
    code_changes: dict[str, str]  # file -> diff
    behavioral_changes: list[str]
    risk_indicators: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "files_added": self.files_added,
            "files_removed": self.files_removed,
            "files_modified": self.files_modified,
            "behavioral_changes": self.behavioral_changes,
            "risk_indicators": self.risk_indicators,
        }


@dataclass
class VerificationResult:
    """Result of supply chain verification."""

    success: bool
    threats: list[SupplyChainThreat]
    packages_scanned: int
    risk_summary: dict[str, int]  # risk_level -> count
    lockfile_valid: bool
    recommendations: list[str]
    scan_duration_ms: float = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "threats": [t.to_dict() for t in self.threats],
            "packages_scanned": self.packages_scanned,
            "risk_summary": self.risk_summary,
            "lockfile_valid": self.lockfile_valid,
            "recommendations": self.recommendations,
            "scan_duration_ms": self.scan_duration_ms,
            "error": self.error,
        }


class DependencyParser(ABC):
    """Abstract base class for dependency parsers."""

    @abstractmethod
    def parse(self, content: str) -> list[PackageInfo]:
        """Parse dependencies from manifest file."""
        pass

    @abstractmethod
    def parse_lockfile(self, content: str) -> list[PackageInfo]:
        """Parse dependencies from lockfile."""
        pass


class NpmDependencyParser(DependencyParser):
    """Parser for npm/package.json dependencies."""

    def parse(self, content: str) -> list[PackageInfo]:
        """Parse package.json."""
        packages = []

        try:
            data = json.loads(content)

            # Regular dependencies
            for name, version in data.get("dependencies", {}).items():
                packages.append(PackageInfo(
                    name=name,
                    version=self._normalize_version(version),
                    ecosystem=PackageEcosystem.NPM,
                ))

            # Dev dependencies
            for name, version in data.get("devDependencies", {}).items():
                packages.append(PackageInfo(
                    name=name,
                    version=self._normalize_version(version),
                    ecosystem=PackageEcosystem.NPM,
                    dev_dependency=True,
                ))

        except json.JSONDecodeError as e:
            logger.error("Failed to parse package.json", error=str(e))

        return packages

    def parse_lockfile(self, content: str) -> list[PackageInfo]:
        """Parse package-lock.json."""
        packages = []

        try:
            data = json.loads(content)

            # npm v7+ format (packages)
            if "packages" in data:
                for path, info in data["packages"].items():
                    if path == "":  # Root package
                        continue
                    name = path.split("node_modules/")[-1]
                    packages.append(PackageInfo(
                        name=name,
                        version=info.get("version", ""),
                        ecosystem=PackageEcosystem.NPM,
                        checksum=info.get("integrity"),
                    ))

            # Legacy format (dependencies)
            elif "dependencies" in data:
                packages.extend(self._parse_legacy_lockfile(data["dependencies"]))

        except json.JSONDecodeError as e:
            logger.error("Failed to parse package-lock.json", error=str(e))

        return packages

    def _parse_legacy_lockfile(
        self,
        deps: dict[str, Any],
        result: list[PackageInfo] | None = None,
    ) -> list[PackageInfo]:
        """Parse legacy lockfile format recursively."""
        if result is None:
            result = []

        for name, info in deps.items():
            result.append(PackageInfo(
                name=name,
                version=info.get("version", ""),
                ecosystem=PackageEcosystem.NPM,
                checksum=info.get("integrity"),
            ))

            # Recurse into nested dependencies
            if "dependencies" in info:
                self._parse_legacy_lockfile(info["dependencies"], result)

        return result

    def _normalize_version(self, version: str) -> str:
        """Normalize version string."""
        # Remove semver prefixes
        return version.lstrip("^~>=<")


class PypiDependencyParser(DependencyParser):
    """Parser for PyPI/requirements.txt and pyproject.toml dependencies."""

    def parse(self, content: str) -> list[PackageInfo]:
        """Parse requirements.txt or pyproject.toml."""
        packages = []

        # Try pyproject.toml first
        if "[project]" in content or "[tool.poetry" in content:
            return self._parse_pyproject(content)

        # Parse requirements.txt
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            # Parse requirement spec
            match = re.match(r"([a-zA-Z0-9_-]+)([>=<~!]+)?([0-9.*]+)?", line)
            if match:
                name = match.group(1)
                version = match.group(3) or "*"
                packages.append(PackageInfo(
                    name=name,
                    version=version,
                    ecosystem=PackageEcosystem.PYPI,
                ))

        return packages

    def _parse_pyproject(self, content: str) -> list[PackageInfo]:
        """Parse pyproject.toml."""
        packages = []

        # Simple regex-based parsing (would use toml library in production)
        deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if deps_match:
            deps_str = deps_match.group(1)
            for match in re.finditer(r'"([^"]+)"', deps_str):
                spec = match.group(1)
                parts = re.split(r"[>=<~!]", spec, maxsplit=1)
                name = parts[0].strip()
                version = parts[1].strip() if len(parts) > 1 else "*"
                packages.append(PackageInfo(
                    name=name,
                    version=version,
                    ecosystem=PackageEcosystem.PYPI,
                ))

        return packages

    def parse_lockfile(self, content: str) -> list[PackageInfo]:
        """Parse poetry.lock or pip lock."""
        packages = []

        # Simple poetry.lock parsing
        current_package = None
        for line in content.split("\n"):
            if line.startswith("[[package]]"):
                current_package = {}
            elif current_package is not None:
                if line.startswith("name = "):
                    current_package["name"] = line.split("=")[1].strip().strip('"')
                elif line.startswith("version = "):
                    current_package["version"] = line.split("=")[1].strip().strip('"')
                    packages.append(PackageInfo(
                        name=current_package.get("name", ""),
                        version=current_package.get("version", ""),
                        ecosystem=PackageEcosystem.PYPI,
                    ))
                    current_package = None

        return packages


class ThreatDetector:
    """Detects supply chain threats in dependencies."""

    # Known malicious package patterns
    TYPOSQUAT_PATTERNS = [
        (r"lodash", ["lodash", "1odash", "iodash", "lodahs"]),
        (r"express", ["express", "expresss", "expres"]),
        (r"requests", ["requests", "request", "requets"]),
        (r"numpy", ["numpy", "numppy", "nunpy"]),
    ]

    # Suspicious code patterns
    MALICIOUS_PATTERNS = [
        (r"eval\s*\(", "Dynamic code execution with eval"),
        (r"exec\s*\(", "Dynamic code execution with exec"),
        (r"os\.system", "System command execution"),
        (r"subprocess\.(?:run|call|Popen)", "Subprocess execution"),
        (r"base64\.(?:b64decode|decode)", "Base64 decoding (potential obfuscation)"),
        (r"socket\.(?:connect|send)", "Network socket operations"),
        (r"urllib\.request|requests\.(?:get|post)", "Network requests"),
        (r"\\x[0-9a-fA-F]{2}", "Hex-encoded strings (potential obfuscation)"),
        (r"\\u[0-9a-fA-F]{4}", "Unicode-encoded strings"),
        (r"crypto|encrypt|decrypt", "Cryptographic operations"),
        (r"btc|bitcoin|eth|ethereum|wallet", "Cryptocurrency references"),
    ]

    # Data exfiltration patterns
    EXFIL_PATTERNS = [
        (r"\.env", "Environment variable access"),
        (r"password|secret|api.?key|token", "Credential access"),
        (r"ssh|\.pem|private.?key", "SSH/private key access"),
        (r"aws.?access|aws.?secret", "AWS credential access"),
    ]

    def __init__(self) -> None:
        self._known_malicious: set[str] = set()
        self._known_vulnerabilities: dict[str, list[str]] = {}

    def detect_threats(
        self,
        packages: list[PackageInfo],
        code_samples: dict[str, str] | None = None,
    ) -> list[SupplyChainThreat]:
        """Detect threats in a list of packages."""
        threats = []
        threat_id = 0

        for package in packages:
            # Check for typosquatting
            typosquat = self._check_typosquatting(package)
            if typosquat:
                threat_id += 1
                threats.append(SupplyChainThreat(
                    id=f"threat-{threat_id}",
                    threat_type=ThreatType.TYPOSQUATTING,
                    package=package,
                    risk_level=RiskLevel.HIGH,
                    title=f"Potential typosquatting: {package.name}",
                    description=f"Package name '{package.name}' is similar to popular package '{typosquat}'",
                    evidence=[f"Similar to: {typosquat}"],
                    remediation=f"Verify this is the intended package, not a typosquat of '{typosquat}'",
                ))

            # Check known malicious
            if package.name.lower() in self._known_malicious:
                threat_id += 1
                threats.append(SupplyChainThreat(
                    id=f"threat-{threat_id}",
                    threat_type=ThreatType.MALICIOUS_UPDATE,
                    package=package,
                    risk_level=RiskLevel.CRITICAL,
                    title=f"Known malicious package: {package.name}",
                    description="This package has been flagged as malicious",
                    evidence=["Listed in known malicious package database"],
                    remediation="Remove this package immediately",
                ))

            # Check known vulnerabilities
            vulns = self._check_vulnerabilities(package)
            for vuln in vulns:
                threat_id += 1
                threats.append(SupplyChainThreat(
                    id=f"threat-{threat_id}",
                    threat_type=ThreatType.KNOWN_VULNERABILITY,
                    package=package,
                    risk_level=RiskLevel.HIGH,
                    title=f"Known vulnerability in {package.name}@{package.version}",
                    description=vuln.get("description", "Security vulnerability detected"),
                    evidence=[f"CVE: {vuln.get('cve', 'N/A')}"],
                    remediation=f"Upgrade to version {vuln.get('fixed_version', 'latest')}",
                    cve_ids=[vuln.get("cve")] if vuln.get("cve") else [],
                ))

        # Check code samples for malicious patterns
        if code_samples:
            code_threats = self._analyze_code(code_samples)
            for threat in code_threats:
                threat_id += 1
                threat.id = f"threat-{threat_id}"
                threats.append(threat)

        return threats

    def _check_typosquatting(self, package: PackageInfo) -> str | None:
        """Check if package name is potential typosquatting."""
        name_lower = package.name.lower()

        for popular, variants in self.TYPOSQUAT_PATTERNS:
            # Skip if it's the actual popular package
            if name_lower == popular:
                continue

            # Check Levenshtein-like similarity
            if self._is_similar(name_lower, popular):
                return popular

            # Check known typosquat variants
            if name_lower in variants and name_lower != popular:
                return popular

        return None

    def _is_similar(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """Check if two names are similar (potential typosquat)."""
        if name1 == name2:
            return False

        # Simple edit distance check
        if abs(len(name1) - len(name2)) > 2:
            return False

        # Count matching characters
        matches = sum(1 for a, b in zip(name1, name2) if a == b)
        similarity = matches / max(len(name1), len(name2))

        return similarity >= threshold

    def _check_vulnerabilities(self, package: PackageInfo) -> list[dict[str, Any]]:
        """Check package against vulnerability database."""
        key = f"{package.ecosystem.value}:{package.name}"
        vulns = self._known_vulnerabilities.get(key, [])

        # Filter by version
        matching = []
        for vuln in vulns:
            affected = vuln.get("affected_versions", [])
            if package.version in affected or "*" in affected:
                matching.append(vuln)

        return matching

    def _analyze_code(
        self,
        code_samples: dict[str, str],
    ) -> list[SupplyChainThreat]:
        """Analyze code for malicious patterns."""
        threats = []

        for file_path, code in code_samples.items():
            # Check malicious patterns
            for pattern, description in self.MALICIOUS_PATTERNS:
                matches = re.findall(pattern, code, re.IGNORECASE)
                if matches:
                    threats.append(SupplyChainThreat(
                        id="",  # Will be set by caller
                        threat_type=ThreatType.CODE_INJECTION,
                        package=PackageInfo(
                            name=file_path,
                            version="N/A",
                            ecosystem=PackageEcosystem.NPM,  # Will be corrected
                        ),
                        risk_level=RiskLevel.MEDIUM,
                        title=f"Suspicious code pattern: {description}",
                        description=f"Found {len(matches)} occurrence(s) of suspicious pattern",
                        evidence=[f"File: {file_path}", f"Pattern: {pattern}"],
                        remediation="Review the code to ensure it's not malicious",
                    ))

            # Check exfiltration patterns
            for pattern, description in self.EXFIL_PATTERNS:
                if re.search(pattern, code, re.IGNORECASE):
                    threats.append(SupplyChainThreat(
                        id="",
                        threat_type=ThreatType.DATA_EXFILTRATION,
                        package=PackageInfo(
                            name=file_path,
                            version="N/A",
                            ecosystem=PackageEcosystem.NPM,
                        ),
                        risk_level=RiskLevel.HIGH,
                        title=f"Potential data access: {description}",
                        description="Code accesses potentially sensitive data",
                        evidence=[f"File: {file_path}", f"Pattern: {description}"],
                        remediation="Verify this data access is legitimate",
                    ))

        return threats

    def add_known_malicious(self, package_names: list[str]) -> None:
        """Add packages to known malicious list."""
        self._known_malicious.update(name.lower() for name in package_names)

    def add_vulnerability(
        self,
        ecosystem: PackageEcosystem,
        package_name: str,
        vuln_info: dict[str, Any],
    ) -> None:
        """Add a known vulnerability."""
        key = f"{ecosystem.value}:{package_name}"
        if key not in self._known_vulnerabilities:
            self._known_vulnerabilities[key] = []
        self._known_vulnerabilities[key].append(vuln_info)


class LockfileVerifier:
    """Verifies lockfile integrity."""

    def verify_integrity(
        self,
        lockfile_content: str,
        manifest_content: str,
        ecosystem: PackageEcosystem,
    ) -> tuple[bool, list[str]]:
        """Verify lockfile matches manifest and has valid checksums."""
        issues = []

        # Get appropriate parser
        if ecosystem == PackageEcosystem.NPM:
            parser = NpmDependencyParser()
        elif ecosystem == PackageEcosystem.PYPI:
            parser = PypiDependencyParser()
        else:
            return True, ["Verification not implemented for this ecosystem"]

        # Parse both files
        manifest_deps = parser.parse(manifest_content)
        lockfile_deps = parser.parse_lockfile(lockfile_content)

        manifest_names = {d.name for d in manifest_deps}
        lockfile_names = {d.name for d in lockfile_deps}

        # Check for missing in lockfile
        missing = manifest_names - lockfile_names
        if missing:
            issues.append(f"Dependencies missing from lockfile: {', '.join(missing)}")

        # Check for extra in lockfile (not necessarily an issue)
        # extra = lockfile_names - manifest_names

        # Check checksums
        for dep in lockfile_deps:
            if dep.checksum is None:
                issues.append(f"Missing checksum for {dep.name}@{dep.version}")
            elif not self._validate_checksum_format(dep.checksum):
                issues.append(f"Invalid checksum format for {dep.name}@{dep.version}")

        return len(issues) == 0, issues

    def _validate_checksum_format(self, checksum: str) -> bool:
        """Validate checksum format (e.g., sha512-...)."""
        if checksum.startswith("sha512-") or checksum.startswith("sha256-"):
            return len(checksum) > 10
        return False


class SupplyChainVerifier:
    """Main interface for supply chain verification.

    Usage:
        verifier = SupplyChainVerifier()

        # Verify from manifest files
        result = await verifier.verify_project("/path/to/project")

        # Or verify specific packages
        result = await verifier.verify_packages(packages)
    """

    def __init__(self) -> None:
        self._threat_detector = ThreatDetector()
        self._lockfile_verifier = LockfileVerifier()
        self._parsers: dict[PackageEcosystem, DependencyParser] = {
            PackageEcosystem.NPM: NpmDependencyParser(),
            PackageEcosystem.PYPI: PypiDependencyParser(),
        }

    async def verify_project(
        self,
        project_path: str,
        include_dev: bool = True,
    ) -> VerificationResult:
        """Verify all dependencies in a project."""
        import time
        start_time = time.time()

        path = Path(project_path)
        all_packages: list[PackageInfo] = []
        lockfile_valid = True
        issues: list[str] = []

        # Check for npm
        package_json = path / "package.json"
        if package_json.exists():
            packages = self._parsers[PackageEcosystem.NPM].parse(
                package_json.read_text()
            )
            all_packages.extend(packages)

            # Verify lockfile
            package_lock = path / "package-lock.json"
            if package_lock.exists():
                valid, lock_issues = self._lockfile_verifier.verify_integrity(
                    package_lock.read_text(),
                    package_json.read_text(),
                    PackageEcosystem.NPM,
                )
                lockfile_valid = lockfile_valid and valid
                issues.extend(lock_issues)

        # Check for Python
        requirements = path / "requirements.txt"
        if requirements.exists():
            packages = self._parsers[PackageEcosystem.PYPI].parse(
                requirements.read_text()
            )
            all_packages.extend(packages)

        pyproject = path / "pyproject.toml"
        if pyproject.exists():
            packages = self._parsers[PackageEcosystem.PYPI].parse(
                pyproject.read_text()
            )
            all_packages.extend(packages)

        # Filter dev dependencies if requested
        if not include_dev:
            all_packages = [p for p in all_packages if not p.dev_dependency]

        # Detect threats
        threats = self._threat_detector.detect_threats(all_packages)

        # Generate risk summary
        risk_summary = {level.value: 0 for level in RiskLevel}
        for threat in threats:
            risk_summary[threat.risk_level.value] += 1

        # Generate recommendations
        recommendations = self._generate_recommendations(threats, issues)

        elapsed_ms = (time.time() - start_time) * 1000

        return VerificationResult(
            success=len([t for t in threats if t.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)]) == 0,
            threats=threats,
            packages_scanned=len(all_packages),
            risk_summary=risk_summary,
            lockfile_valid=lockfile_valid,
            recommendations=recommendations,
            scan_duration_ms=elapsed_ms,
        )

    async def verify_packages(
        self,
        packages: list[PackageInfo],
    ) -> VerificationResult:
        """Verify a specific list of packages."""
        import time
        start_time = time.time()

        threats = self._threat_detector.detect_threats(packages)

        risk_summary = {level.value: 0 for level in RiskLevel}
        for threat in threats:
            risk_summary[threat.risk_level.value] += 1

        recommendations = self._generate_recommendations(threats, [])

        elapsed_ms = (time.time() - start_time) * 1000

        return VerificationResult(
            success=len([t for t in threats if t.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)]) == 0,
            threats=threats,
            packages_scanned=len(packages),
            risk_summary=risk_summary,
            lockfile_valid=True,
            recommendations=recommendations,
            scan_duration_ms=elapsed_ms,
        )

    async def analyze_update(
        self,
        package: PackageInfo,
        new_version: str,
        old_code: str | None = None,
        new_code: str | None = None,
    ) -> DiffResult:
        """Analyze a package update for suspicious changes."""
        files_added = []
        files_removed = []
        files_modified = []
        behavioral_changes = []
        risk_indicators = []

        if old_code and new_code:
            # Compare code
            old_lines = set(old_code.split("\n"))
            new_lines = set(new_code.split("\n"))

            added_lines = new_lines - old_lines
            removed_lines = old_lines - new_lines

            # Check for suspicious additions
            for line in added_lines:
                for pattern, desc in ThreatDetector.MALICIOUS_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        risk_indicators.append(f"New suspicious pattern: {desc}")
                        behavioral_changes.append(f"Added: {line[:100]}...")

        return DiffResult(
            old_version=package.version,
            new_version=new_version,
            files_added=files_added,
            files_removed=files_removed,
            files_modified=files_modified,
            code_changes={},
            behavioral_changes=behavioral_changes,
            risk_indicators=risk_indicators,
        )

    def _generate_recommendations(
        self,
        threats: list[SupplyChainThreat],
        issues: list[str],
    ) -> list[str]:
        """Generate recommendations based on findings."""
        recommendations = []

        # Critical threats
        critical = [t for t in threats if t.risk_level == RiskLevel.CRITICAL]
        if critical:
            recommendations.append(
                f"URGENT: Remove {len(critical)} critical threat(s) immediately"
            )

        # High threats
        high = [t for t in threats if t.risk_level == RiskLevel.HIGH]
        if high:
            recommendations.append(
                f"Review and address {len(high)} high-risk finding(s)"
            )

        # Lockfile issues
        if issues:
            recommendations.append("Fix lockfile integrity issues to ensure reproducible builds")

        # General
        if not threats and not issues:
            recommendations.append("No immediate threats detected. Continue monitoring.")

        return recommendations

    def add_known_malicious_packages(self, packages: list[str]) -> None:
        """Add packages to the known malicious list."""
        self._threat_detector.add_known_malicious(packages)

    def add_vulnerability_data(
        self,
        ecosystem: PackageEcosystem,
        package_name: str,
        vulnerability: dict[str, Any],
    ) -> None:
        """Add vulnerability data for a package."""
        self._threat_detector.add_vulnerability(ecosystem, package_name, vulnerability)
