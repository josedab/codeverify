"""Tests for Supply Chain Verification module."""

import asyncio
import json

from codeverify_core.supply_chain_verification import (
    LockfileVerifier,
    NpmDependencyParser,
    PackageEcosystem,
    PackageInfo,
    PypiDependencyParser,
    RiskLevel,
    SupplyChainThreat,
    SupplyChainVerifier,
    ThreatDetector,
    ThreatType,
)


class TestPackageEcosystem:
    """Tests for PackageEcosystem enum."""

    def test_all_ecosystems_exist(self):
        """All expected ecosystems exist."""
        assert PackageEcosystem.NPM.value == "npm"
        assert PackageEcosystem.PYPI.value == "pypi"
        assert PackageEcosystem.MAVEN.value == "maven"
        assert PackageEcosystem.CARGO.value == "cargo"
        assert PackageEcosystem.GO.value == "go"
        assert PackageEcosystem.NUGET.value == "nuget"


class TestThreatType:
    """Tests for ThreatType enum."""

    def test_all_threat_types_exist(self):
        """All expected threat types exist."""
        assert ThreatType.TYPOSQUATTING.value == "typosquatting"
        assert ThreatType.DEPENDENCY_CONFUSION.value == "dependency_confusion"
        assert ThreatType.MALICIOUS_UPDATE.value == "malicious_update"
        assert ThreatType.CODE_INJECTION.value == "code_injection"
        assert ThreatType.DATA_EXFILTRATION.value == "data_exfiltration"
        assert ThreatType.BACKDOOR.value == "backdoor"
        assert ThreatType.KNOWN_VULNERABILITY.value == "known_vulnerability"


class TestPackageInfo:
    """Tests for PackageInfo dataclass."""

    def test_creation(self):
        """Can create a PackageInfo."""
        pkg = PackageInfo(
            name="requests",
            version="2.31.0",
            ecosystem=PackageEcosystem.PYPI,
        )
        assert pkg.name == "requests"
        assert pkg.version == "2.31.0"
        assert pkg.dev_dependency is False
        assert pkg.dependencies == []

    def test_to_dict(self):
        """to_dict returns expected keys."""
        pkg = PackageInfo(name="lodash", version="4.17.21", ecosystem=PackageEcosystem.NPM)
        d = pkg.to_dict()
        assert d["name"] == "lodash"
        assert d["ecosystem"] == "npm"


class TestSupplyChainThreat:
    """Tests for SupplyChainThreat dataclass."""

    def test_creation(self):
        """Can create a SupplyChainThreat."""
        pkg = PackageInfo(name="evil-pkg", version="0.1.0", ecosystem=PackageEcosystem.NPM)
        threat = SupplyChainThreat(
            id="t1",
            threat_type=ThreatType.TYPOSQUATTING,
            package=pkg,
            risk_level=RiskLevel.HIGH,
            title="Typosquat detected",
            description="Looks like lodash",
            evidence=["Similar to: lodash"],
            remediation="Remove package",
        )
        assert threat.id == "t1"
        assert threat.threat_type == ThreatType.TYPOSQUATTING
        assert threat.false_positive is False


class TestThreatDetector:
    """Tests for ThreatDetector."""

    def test_detects_typosquatting(self):
        """Detects typosquatting patterns."""
        detector = ThreatDetector()
        pkg = PackageInfo(name="1odash", version="1.0.0", ecosystem=PackageEcosystem.NPM)
        threats = detector.detect_threats([pkg])
        typo_threats = [t for t in threats if t.threat_type == ThreatType.TYPOSQUATTING]
        assert len(typo_threats) >= 1

    def test_no_threat_for_legitimate_package(self):
        """No typosquat for actual popular package."""
        detector = ThreatDetector()
        pkg = PackageInfo(name="lodash", version="4.17.21", ecosystem=PackageEcosystem.NPM)
        threats = detector.detect_threats([pkg])
        typo_threats = [t for t in threats if t.threat_type == ThreatType.TYPOSQUATTING]
        assert len(typo_threats) == 0


class TestNpmDependencyParser:
    """Tests for NpmDependencyParser."""

    def test_parse_package_json(self):
        """Parses dependencies from package.json content."""
        parser = NpmDependencyParser()
        content = json.dumps(
            {
                "name": "my-app",
                "dependencies": {
                    "express": "^4.18.0",
                    "lodash": "~4.17.21",
                },
                "devDependencies": {
                    "jest": "^29.0.0",
                },
            }
        )
        packages = parser.parse(content)
        assert len(packages) == 3

        prod = [p for p in packages if not p.dev_dependency]
        dev = [p for p in packages if p.dev_dependency]
        assert len(prod) == 2
        assert len(dev) == 1
        assert all(p.ecosystem == PackageEcosystem.NPM for p in packages)


class TestPypiDependencyParser:
    """Tests for PypiDependencyParser."""

    def test_parse_requirements_txt(self):
        """Parses requirements.txt content."""
        parser = PypiDependencyParser()
        content = "requests==2.31.0\nflask>=2.0.0\n# comment\nnumpy\n"
        packages = parser.parse(content)
        assert len(packages) == 3
        names = [p.name for p in packages]
        assert "requests" in names
        assert "flask" in names
        assert "numpy" in names
        assert all(p.ecosystem == PackageEcosystem.PYPI for p in packages)


class TestLockfileVerifier:
    """Tests for LockfileVerifier."""

    def test_basic_verification(self):
        """Verifies lockfile against manifest."""
        verifier = LockfileVerifier()
        manifest = json.dumps(
            {
                "dependencies": {"express": "^4.18.0"},
            }
        )
        lockfile = json.dumps(
            {
                "packages": {
                    "": {},
                    "node_modules/express": {
                        "version": "4.18.2",
                        "integrity": "sha512-abcdefghijk",
                    },
                },
            }
        )
        valid, issues = verifier.verify_integrity(lockfile, manifest, PackageEcosystem.NPM)
        assert isinstance(valid, bool)
        assert isinstance(issues, list)


class TestSupplyChainVerifier:
    """Tests for SupplyChainVerifier orchestration."""

    def test_verify_packages(self):
        """verify_packages returns a VerificationResult."""
        verifier = SupplyChainVerifier()
        packages = [
            PackageInfo(name="lodash", version="4.17.21", ecosystem=PackageEcosystem.NPM),
            PackageInfo(name="requests", version="2.31.0", ecosystem=PackageEcosystem.PYPI),
        ]
        result = asyncio.get_event_loop().run_until_complete(verifier.verify_packages(packages))
        assert result.success is True
        assert result.packages_scanned == 2
        assert "risk_summary" in result.to_dict()


class TestSupplyChainEdgeCases:
    """Edge case tests for supply chain verification."""

    def test_empty_dependencies(self):
        """Handles empty dependency list."""
        detector = ThreatDetector()
        threats = detector.detect_threats([])
        assert threats == []

    def test_unknown_ecosystem_lockfile(self):
        """LockfileVerifier handles unsupported ecosystem."""
        verifier = LockfileVerifier()
        valid, issues = verifier.verify_integrity("{}", "{}", PackageEcosystem.MAVEN)
        assert valid is True
        assert len(issues) == 1
