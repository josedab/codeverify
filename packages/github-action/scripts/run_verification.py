#!/usr/bin/env python3
"""
CodeVerify GitHub Action - Verification Runner

Runs tier-based verification with support for:
- Free tier: Pattern-based static analysis
- Pro tier: AI-powered semantic analysis
- Enterprise tier: Full Z3 SMT solver + AI verification
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


class VerificationTier(str, Enum):
    """Verification tier levels."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class Severity(str, Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(str, Enum):
    """Categories of verification issues."""

    SECURITY = "security"
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    SUPPLY_CHAIN = "supply_chain"


@dataclass
class VerificationIssue:
    """A verification issue found during analysis."""

    id: str
    title: str
    description: str
    severity: Severity
    category: IssueCategory
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 1
    column_end: int = 1
    rule_id: str = ""
    fix_suggestion: str | None = None
    proof_available: bool = False

    def to_sarif_result(self) -> dict[str, Any]:
        """Convert to SARIF result format."""
        result = {
            "ruleId": self.rule_id or f"codeverify/{self.category.value}/{self.id}",
            "level": self._severity_to_sarif_level(),
            "message": {
                "text": f"{self.title}\n\n{self.description}"
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": self.file_path,
                        "uriBaseId": "%SRCROOT%"
                    },
                    "region": {
                        "startLine": self.line_start,
                        "endLine": self.line_end,
                        "startColumn": self.column_start,
                        "endColumn": self.column_end
                    }
                }
            }]
        }

        if self.fix_suggestion:
            result["fixes"] = [{
                "description": {"text": "Suggested fix"},
                "artifactChanges": [{
                    "artifactLocation": {"uri": self.file_path},
                    "replacements": [{
                        "deletedRegion": {
                            "startLine": self.line_start,
                            "endLine": self.line_end
                        },
                        "insertedContent": {"text": self.fix_suggestion}
                    }]
                }]
            }]

        return result

    def _severity_to_sarif_level(self) -> str:
        """Map severity to SARIF level."""
        mapping = {
            Severity.CRITICAL: "error",
            Severity.HIGH: "error",
            Severity.MEDIUM: "warning",
            Severity.LOW: "note",
            Severity.INFO: "none"
        }
        return mapping[self.severity]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
            "rule_id": self.rule_id,
            "fix_suggestion": self.fix_suggestion,
            "proof_available": self.proof_available
        }


@dataclass
class VerificationResult:
    """Complete verification result."""

    status: str  # passed, failed, warning
    tier: VerificationTier
    issues: list[VerificationIssue] = field(default_factory=list)
    files_analyzed: int = 0
    duration_seconds: float = 0.0
    proofs_generated: int = 0
    supply_chain_issues: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.LOW)


class TierVerifier:
    """Base class for tier-specific verification."""

    def __init__(self, tier: VerificationTier):
        self.tier = tier
        self.issues: list[VerificationIssue] = []

    def verify_file(self, file_path: Path) -> list[VerificationIssue]:
        """Verify a single file. Override in subclasses."""
        raise NotImplementedError

    def verify_supply_chain(self) -> list[VerificationIssue]:
        """Verify supply chain dependencies."""
        return []


class FreeTierVerifier(TierVerifier):
    """Free tier: Pattern-based static analysis only."""

    DANGEROUS_PATTERNS = {
        ".py": [
            (r"eval\s*\(", "EVAL_USAGE", "Use of eval() is dangerous"),
            (r"exec\s*\(", "EXEC_USAGE", "Use of exec() is dangerous"),
            (r"subprocess\..*shell\s*=\s*True", "SHELL_INJECTION", "Shell=True enables command injection"),
            (r"pickle\.loads?\s*\(", "PICKLE_DESERIALIZE", "Pickle deserialization can execute arbitrary code"),
            (r"__import__\s*\(", "DYNAMIC_IMPORT", "Dynamic imports can load malicious modules"),
            (r"os\.system\s*\(", "OS_SYSTEM", "os.system() is vulnerable to command injection"),
        ],
        ".js": [
            (r"eval\s*\(", "EVAL_USAGE", "Use of eval() is dangerous"),
            (r"innerHTML\s*=", "XSS_INNERHTML", "innerHTML assignment can cause XSS"),
            (r"document\.write\s*\(", "XSS_DOCUMENT_WRITE", "document.write can cause XSS"),
            (r"new\s+Function\s*\(", "FUNCTION_CONSTRUCTOR", "Function constructor can execute arbitrary code"),
        ],
        ".ts": [
            (r"eval\s*\(", "EVAL_USAGE", "Use of eval() is dangerous"),
            (r"innerHTML\s*=", "XSS_INNERHTML", "innerHTML assignment can cause XSS"),
            (r"as\s+any", "ANY_CAST", "Casting to 'any' bypasses type safety"),
            (r"@ts-ignore", "TS_IGNORE", "ts-ignore suppresses type checking"),
        ]
    }

    def __init__(self):
        super().__init__(VerificationTier.FREE)

    def verify_file(self, file_path: Path) -> list[VerificationIssue]:
        """Run pattern-based verification."""
        import re

        issues = []
        suffix = file_path.suffix.lower()
        patterns = self.DANGEROUS_PATTERNS.get(suffix, [])

        if not patterns:
            return issues

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
        except (OSError, UnicodeDecodeError):
            return issues

        for line_num, line in enumerate(lines, start=1):
            for pattern, rule_id, description in patterns:
                if re.search(pattern, line):
                    issues.append(VerificationIssue(
                        id=f"pattern-{rule_id.lower()}-{line_num}",
                        title=f"Dangerous pattern: {rule_id}",
                        description=description,
                        severity=Severity.HIGH if "injection" in description.lower() or "xss" in description.lower() else Severity.MEDIUM,
                        category=IssueCategory.SECURITY,
                        file_path=str(file_path),
                        line_start=line_num,
                        line_end=line_num,
                        rule_id=f"codeverify/pattern/{rule_id}"
                    ))

        return issues


class ProTierVerifier(TierVerifier):
    """Pro tier: AI-powered semantic analysis."""

    def __init__(self, api_key: str | None = None):
        super().__init__(VerificationTier.PRO)
        self.api_key = api_key or os.environ.get("CODEVERIFY_API_KEY")
        self.free_verifier = FreeTierVerifier()

    def verify_file(self, file_path: Path) -> list[VerificationIssue]:
        """Run AI-powered verification."""
        # Start with pattern-based checks
        issues = self.free_verifier.verify_file(file_path)

        # Add AI-powered analysis if API key available
        if self.api_key:
            ai_issues = self._run_ai_analysis(file_path)
            issues.extend(ai_issues)

        return issues

    def _run_ai_analysis(self, file_path: Path) -> list[VerificationIssue]:
        """Run AI-powered semantic analysis."""
        try:
            # Import AI analysis module
            from codeverify_agents import VerificationAgent

            agent = VerificationAgent(api_key=self.api_key)
            result = agent.analyze_file(str(file_path))

            return [
                VerificationIssue(
                    id=f"ai-{issue['id']}",
                    title=issue["title"],
                    description=issue["description"],
                    severity=Severity(issue.get("severity", "medium")),
                    category=IssueCategory(issue.get("category", "security")),
                    file_path=str(file_path),
                    line_start=issue.get("line_start", 1),
                    line_end=issue.get("line_end", 1),
                    fix_suggestion=issue.get("fix_suggestion")
                )
                for issue in result.get("issues", [])
            ]
        except ImportError:
            log.warning("AI analysis unavailable - codeverify-agents not installed")
            return []
        except Exception as e:
            log.error("AI analysis failed", error=str(e))
            return []


class EnterpriseTierVerifier(TierVerifier):
    """Enterprise tier: Full Z3 SMT solver + AI verification."""

    def __init__(self, api_key: str | None = None):
        super().__init__(VerificationTier.ENTERPRISE)
        self.api_key = api_key or os.environ.get("CODEVERIFY_API_KEY")
        self.pro_verifier = ProTierVerifier(api_key)
        self.proofs_generated = 0

    def verify_file(self, file_path: Path) -> list[VerificationIssue]:
        """Run full Z3 + AI verification."""
        # Start with Pro tier checks
        issues = self.pro_verifier.verify_file(file_path)

        # Add Z3 formal verification
        z3_issues = self._run_z3_verification(file_path)
        issues.extend(z3_issues)

        return issues

    def _run_z3_verification(self, file_path: Path) -> list[VerificationIssue]:
        """Run Z3 SMT solver verification."""
        try:
            from codeverify_core import CodeVerifier

            verifier = CodeVerifier()
            result = verifier.verify_file(str(file_path))

            issues = []
            for violation in result.get("violations", []):
                issue = VerificationIssue(
                    id=f"z3-{violation['id']}",
                    title=violation["title"],
                    description=violation["description"],
                    severity=Severity(violation.get("severity", "high")),
                    category=IssueCategory.CORRECTNESS,
                    file_path=str(file_path),
                    line_start=violation.get("line_start", 1),
                    line_end=violation.get("line_end", 1),
                    proof_available=True
                )
                issues.append(issue)

            self.proofs_generated += result.get("proofs_generated", 0)
            return issues

        except ImportError:
            log.warning("Z3 verification unavailable - z3-solver not installed")
            return []
        except Exception as e:
            log.error("Z3 verification failed", error=str(e))
            return []

    def verify_supply_chain(self) -> list[VerificationIssue]:
        """Verify supply chain dependencies with full analysis."""
        try:
            from codeverify_core.supply_chain_verification import SupplyChainVerifier

            verifier = SupplyChainVerifier.get_verifier()
            result = verifier.verify_project(Path.cwd())

            issues = []
            for threat in result.threats:
                issues.append(VerificationIssue(
                    id=f"supply-chain-{threat.id}",
                    title=f"Supply Chain: {threat.threat_type}",
                    description=threat.description,
                    severity=Severity(threat.severity),
                    category=IssueCategory.SUPPLY_CHAIN,
                    file_path=threat.file_path or "requirements.txt",
                    line_start=threat.line_number or 1,
                    line_end=threat.line_number or 1
                ))

            return issues

        except ImportError:
            log.warning("Supply chain verification unavailable")
            return []
        except Exception as e:
            log.error("Supply chain verification failed", error=str(e))
            return []


class GitHubActionRunner:
    """Main runner for the GitHub Action."""

    def __init__(
        self,
        tier: str,
        paths: str,
        exclude_paths: str,
        fail_on: str,
        config_file: str | None,
        enable_supply_chain: bool,
        enable_sarif: bool,
        changed_files: str | None,
        output_dir: str
    ):
        self.tier = VerificationTier(tier)
        self.paths = [p.strip() for p in paths.split(",") if p.strip()]
        self.exclude_paths = [p.strip() for p in exclude_paths.split(",") if p.strip()]
        self.fail_on = Severity(fail_on) if fail_on != "never" else None
        self.config_file = Path(config_file) if config_file else None
        self.enable_supply_chain = enable_supply_chain
        self.enable_sarif = enable_sarif
        self.changed_files = set(changed_files.split(",")) if changed_files else None
        self.output_dir = Path(output_dir)

        # Create verifier based on tier
        api_key = os.environ.get("CODEVERIFY_API_KEY")
        if self.tier == VerificationTier.ENTERPRISE:
            self.verifier = EnterpriseTierVerifier(api_key)
        elif self.tier == VerificationTier.PRO:
            self.verifier = ProTierVerifier(api_key)
        else:
            self.verifier = FreeTierVerifier()

    def run(self) -> VerificationResult:
        """Run the verification."""
        import fnmatch
        import time

        start_time = time.time()
        all_issues: list[VerificationIssue] = []
        files_analyzed = 0

        # Find files to verify
        files_to_verify = self._find_files()

        log.info(
            "Starting verification",
            tier=self.tier.value,
            files=len(files_to_verify)
        )

        # Verify each file
        for file_path in files_to_verify:
            try:
                issues = self.verifier.verify_file(file_path)
                all_issues.extend(issues)
                files_analyzed += 1
            except Exception as e:
                log.error("Failed to verify file", file=str(file_path), error=str(e))

        # Run supply chain verification if enabled
        supply_chain_issues = 0
        if self.enable_supply_chain:
            sc_issues = self.verifier.verify_supply_chain()
            all_issues.extend(sc_issues)
            supply_chain_issues = len(sc_issues)

        duration = time.time() - start_time

        # Determine status
        status = self._determine_status(all_issues)

        result = VerificationResult(
            status=status,
            tier=self.tier,
            issues=all_issues,
            files_analyzed=files_analyzed,
            duration_seconds=duration,
            proofs_generated=getattr(self.verifier, "proofs_generated", 0),
            supply_chain_issues=supply_chain_issues
        )

        # Generate outputs
        self._generate_outputs(result)

        return result

    def _find_files(self) -> list[Path]:
        """Find files to verify based on patterns."""
        import fnmatch

        files = []
        workspace = Path.cwd()

        for pattern in self.paths:
            for file_path in workspace.rglob("*"):
                if not file_path.is_file():
                    continue

                rel_path = str(file_path.relative_to(workspace))

                # Check if matches include pattern
                if not fnmatch.fnmatch(rel_path, pattern):
                    continue

                # Check if matches exclude pattern
                excluded = False
                for exclude in self.exclude_paths:
                    if fnmatch.fnmatch(rel_path, exclude):
                        excluded = True
                        break

                if excluded:
                    continue

                # If changed_files is set, only include those
                if self.changed_files and rel_path not in self.changed_files:
                    continue

                files.append(file_path)

        return list(set(files))

    def _determine_status(self, issues: list[VerificationIssue]) -> str:
        """Determine verification status based on issues and fail_on threshold."""
        if not issues:
            return "passed"

        if self.fail_on is None:
            return "warning"

        severity_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
        threshold_index = severity_order.index(self.fail_on)

        for issue in issues:
            if issue.severity in severity_order[:threshold_index + 1]:
                return "failed"

        return "warning"

    def _generate_outputs(self, result: VerificationResult) -> None:
        """Generate output files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON report
        report = {
            "status": result.status,
            "tier": result.tier.value,
            "files_analyzed": result.files_analyzed,
            "duration_seconds": result.duration_seconds,
            "proofs_generated": result.proofs_generated,
            "summary": {
                "total_issues": len(result.issues),
                "critical": result.critical_count,
                "high": result.high_count,
                "medium": result.medium_count,
                "low": result.low_count
            },
            "issues": [i.to_dict() for i in result.issues]
        }

        report_path = self.output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2))

        # Generate SARIF if enabled
        if self.enable_sarif:
            sarif = self._generate_sarif(result)
            sarif_path = self.output_dir / "results.sarif"
            sarif_path.write_text(json.dumps(sarif, indent=2))

        # Generate PR comment markdown
        pr_comment = self._generate_pr_comment(result)
        comment_path = self.output_dir / "pr-comment.md"
        comment_path.write_text(pr_comment)

        # Set GitHub Action outputs
        self._set_github_outputs(result)

    def _generate_sarif(self, result: VerificationResult) -> dict[str, Any]:
        """Generate SARIF format report."""
        rules = {}
        results = []

        for issue in result.issues:
            rule_id = issue.rule_id or f"codeverify/{issue.category.value}/{issue.id}"

            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": issue.title,
                    "shortDescription": {"text": issue.title},
                    "fullDescription": {"text": issue.description},
                    "defaultConfiguration": {
                        "level": issue._severity_to_sarif_level()
                    },
                    "properties": {
                        "category": issue.category.value,
                        "security-severity": self._severity_to_score(issue.severity)
                    }
                }

            results.append(issue.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "CodeVerify",
                        "version": "1.0.0",
                        "informationUri": "https://codeverify.dev",
                        "rules": list(rules.values())
                    }
                },
                "results": results,
                "invocations": [{
                    "executionSuccessful": True,
                    "endTimeUtc": datetime.now(timezone.utc).isoformat()
                }]
            }]
        }

    def _severity_to_score(self, severity: Severity) -> str:
        """Map severity to CVSS-style score."""
        mapping = {
            Severity.CRITICAL: "9.0",
            Severity.HIGH: "7.0",
            Severity.MEDIUM: "4.0",
            Severity.LOW: "2.0",
            Severity.INFO: "0.0"
        }
        return mapping[severity]

    def _generate_pr_comment(self, result: VerificationResult) -> str:
        """Generate PR comment markdown."""
        status_emoji = {
            "passed": ":white_check_mark:",
            "failed": ":x:",
            "warning": ":warning:"
        }

        lines = [
            f"## CodeVerify Results {status_emoji.get(result.status, '')}",
            "",
            f"**Status:** {result.status.upper()}",
            f"**Tier:** {result.tier.value.title()}",
            f"**Files Analyzed:** {result.files_analyzed}",
            f"**Duration:** {result.duration_seconds:.2f}s",
            ""
        ]

        if result.proofs_generated > 0:
            lines.append(f"**Formal Proofs Generated:** {result.proofs_generated}")
            lines.append("")

        # Summary table
        lines.extend([
            "### Summary",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| :red_circle: Critical | {result.critical_count} |",
            f"| :orange_circle: High | {result.high_count} |",
            f"| :yellow_circle: Medium | {result.medium_count} |",
            f"| :white_circle: Low | {result.low_count} |",
            ""
        ])

        # Issues by file
        if result.issues:
            lines.extend([
                "### Issues Found",
                ""
            ])

            issues_by_file: dict[str, list[VerificationIssue]] = {}
            for issue in result.issues:
                issues_by_file.setdefault(issue.file_path, []).append(issue)

            for file_path, file_issues in sorted(issues_by_file.items()):
                lines.append(f"<details>")
                lines.append(f"<summary><b>{file_path}</b> ({len(file_issues)} issues)</summary>")
                lines.append("")

                for issue in file_issues:
                    severity_badge = {
                        Severity.CRITICAL: ":red_circle:",
                        Severity.HIGH: ":orange_circle:",
                        Severity.MEDIUM: ":yellow_circle:",
                        Severity.LOW: ":white_circle:",
                        Severity.INFO: ":blue_circle:"
                    }.get(issue.severity, "")

                    lines.append(f"- {severity_badge} **{issue.title}** (L{issue.line_start})")
                    lines.append(f"  - {issue.description}")
                    if issue.proof_available:
                        lines.append(f"  - :shield: Formally verified with Z3")
                    lines.append("")

                lines.append("</details>")
                lines.append("")

        # Footer
        lines.extend([
            "---",
            f"*Powered by [CodeVerify](https://codeverify.dev) - {result.tier.value.title()} Tier*"
        ])

        return "\n".join(lines)

    def _set_github_outputs(self, result: VerificationResult) -> None:
        """Set GitHub Action outputs."""
        github_output = os.environ.get("GITHUB_OUTPUT")
        if not github_output:
            return

        outputs = {
            "status": result.status,
            "issues_found": str(len(result.issues)),
            "critical_count": str(result.critical_count),
            "high_count": str(result.high_count),
            "sarif_file": str(self.output_dir / "results.sarif")
        }

        with open(github_output, "a") as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CodeVerify GitHub Action Runner")

    parser.add_argument("--tier", default="free", help="Verification tier")
    parser.add_argument("--paths", default="**/*.py,**/*.ts,**/*.js", help="Paths to verify")
    parser.add_argument("--exclude", default="", help="Paths to exclude")
    parser.add_argument("--fail-on", default="high", help="Severity threshold for failure")
    parser.add_argument("--config", default="", help="Config file path")
    parser.add_argument("--supply-chain", default="true", help="Enable supply chain checks")
    parser.add_argument("--sarif", default="true", help="Generate SARIF report")
    parser.add_argument("--changed-files", default="", help="Changed files only")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    runner = GitHubActionRunner(
        tier=args.tier,
        paths=args.paths,
        exclude_paths=args.exclude,
        fail_on=args.fail_on,
        config_file=args.config if args.config else None,
        enable_supply_chain=args.supply_chain.lower() == "true",
        enable_sarif=args.sarif.lower() == "true",
        changed_files=args.changed_files if args.changed_files else None,
        output_dir=args.output_dir
    )

    result = runner.run()

    log.info(
        "Verification complete",
        status=result.status,
        issues=len(result.issues),
        files=result.files_analyzed
    )

    # Exit with appropriate code
    if result.status == "failed":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
