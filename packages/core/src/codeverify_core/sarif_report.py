"""SARIF Report Generator.

Generates SARIF 2.1.0 compliant reports from CodeVerify verification
findings, suitable for upload to GitHub Advanced Security.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json"
TOOL_NAME = "CodeVerify"
TOOL_VERSION = "0.3.0"
TOOL_URI = "https://codeverify.dev"


SEVERITY_TO_SARIF_LEVEL: dict[str, str] = {
    "critical": "error",
    "high": "error",
    "medium": "warning",
    "low": "note",
    "info": "note",
}


def _finding_fingerprint(finding: dict[str, Any]) -> str:
    """Compute a stable fingerprint for a finding."""
    raw = (
        f"{finding.get('rule_id', '')}|"
        f"{finding.get('file_path', '')}|"
        f"{finding.get('line', 0)}|"
        f"{finding.get('message', '')}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class SarifRule:
    """A SARIF rule descriptor."""

    id: str
    name: str
    short_description: str
    full_description: str = ""
    help_uri: str = ""
    tags: list[str] = field(default_factory=list)
    default_level: str = "warning"


class SarifReportGenerator:
    """Generates SARIF 2.1.0 JSON from verification findings.

    Usage:
        gen = SarifReportGenerator()
        sarif_json = gen.generate(findings)
        with open("results.sarif", "w") as f:
            f.write(sarif_json)
    """

    def __init__(
        self,
        tool_name: str = TOOL_NAME,
        tool_version: str = TOOL_VERSION,
        tool_uri: str = TOOL_URI,
    ):
        self._tool_name = tool_name
        self._tool_version = tool_version
        self._tool_uri = tool_uri

    def generate(
        self,
        findings: list[dict[str, Any]],
        rules: list[SarifRule] | None = None,
    ) -> str:
        """Generate SARIF JSON from findings.

        Each finding dict should have: rule_id, message, file_path, line,
        severity, and optionally code_snippet, fix_suggestion, tags.
        """
        # Collect unique rules
        rule_map: dict[str, dict[str, Any]] = {}
        if rules:
            for r in rules:
                rule_map[r.id] = self._rule_to_sarif(r)

        results = []
        for finding in findings:
            rid = finding.get("rule_id", "unknown")
            if rid not in rule_map:
                rule_map[rid] = {
                    "id": rid,
                    "name": finding.get("rule_name", rid),
                    "shortDescription": {
                        "text": finding.get("rule_name", rid),
                    },
                    "defaultConfiguration": {
                        "level": SEVERITY_TO_SARIF_LEVEL.get(
                            finding.get("severity", "medium"), "warning"
                        ),
                    },
                }

            result = self._finding_to_result(finding)
            results.append(result)

        sarif = {
            "$schema": SARIF_SCHEMA,
            "version": SARIF_VERSION,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": self._tool_name,
                            "version": self._tool_version,
                            "informationUri": self._tool_uri,
                            "rules": list(rule_map.values()),
                        },
                    },
                    "results": results,
                },
            ],
        }

        return json.dumps(sarif, indent=2)

    def _rule_to_sarif(self, rule: SarifRule) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": rule.id,
            "name": rule.name,
            "shortDescription": {"text": rule.short_description},
            "defaultConfiguration": {"level": rule.default_level},
        }
        if rule.full_description:
            result["fullDescription"] = {"text": rule.full_description}
        if rule.help_uri:
            result["helpUri"] = rule.help_uri
        if rule.tags:
            result["properties"] = {"tags": rule.tags}
        return result

    def _finding_to_result(self, finding: dict[str, Any]) -> dict[str, Any]:
        level = SEVERITY_TO_SARIF_LEVEL.get(finding.get("severity", "medium"), "warning")

        result: dict[str, Any] = {
            "ruleId": finding.get("rule_id", "unknown"),
            "level": level,
            "message": {"text": finding.get("message", "")},
            "fingerprints": {
                "codeverify/v1": _finding_fingerprint(finding),
            },
        }

        # Location
        file_path = finding.get("file_path", "")
        line = finding.get("line", 1)
        if file_path:
            result["locations"] = [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": file_path},
                        "region": {
                            "startLine": max(1, line),
                        },
                    },
                },
            ]

        # Code snippet
        snippet = finding.get("code_snippet", "")
        if snippet and "locations" in result:
            result["locations"][0]["physicalLocation"]["region"]["snippet"] = {
                "text": snippet,
            }

        # Fix suggestion
        fix = finding.get("fix_suggestion", "")
        if fix:
            result["fixes"] = [
                {
                    "description": {"text": fix},
                },
            ]

        return result

    def generate_summary(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate a summary of findings by severity and rule."""
        by_severity: dict[str, int] = {}
        by_rule: dict[str, int] = {}
        for f in findings:
            sev = f.get("severity", "medium")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            rid = f.get("rule_id", "unknown")
            by_rule[rid] = by_rule.get(rid, 0) + 1

        return {
            "total": len(findings),
            "by_severity": by_severity,
            "by_rule": by_rule,
        }
