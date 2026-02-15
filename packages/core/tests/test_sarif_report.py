"""Tests for sarif_report module."""

from __future__ import annotations

import json

from codeverify_core.sarif_report import (
    SARIF_VERSION,
    SarifReportGenerator,
    SarifRule,
)

SAMPLE_FINDINGS = [
    {
        "rule_id": "null_safety",
        "rule_name": "Null Safety Check",
        "message": "Possible None dereference on 'user'",
        "file_path": "src/auth.py",
        "line": 42,
        "severity": "high",
        "code_snippet": "user.name",
        "fix_suggestion": "Add null check: if user is not None",
    },
    {
        "rule_id": "integer_overflow",
        "rule_name": "Integer Overflow",
        "message": "Potential integer overflow in multiplication",
        "file_path": "src/calc.py",
        "line": 15,
        "severity": "medium",
    },
    {
        "rule_id": "null_safety",
        "rule_name": "Null Safety Check",
        "message": "Possible None dereference on 'order'",
        "file_path": "src/orders.py",
        "line": 88,
        "severity": "critical",
    },
]


class TestSarifReportGenerator:
    def test_valid_sarif_structure(self):
        gen = SarifReportGenerator()
        sarif_json = gen.generate(SAMPLE_FINDINGS)
        data = json.loads(sarif_json)
        assert data["version"] == SARIF_VERSION
        assert "$schema" in data
        assert len(data["runs"]) == 1

    def test_tool_metadata(self):
        gen = SarifReportGenerator(tool_name="TestTool", tool_version="1.0.0")
        data = json.loads(gen.generate(SAMPLE_FINDINGS))
        driver = data["runs"][0]["tool"]["driver"]
        assert driver["name"] == "TestTool"
        assert driver["version"] == "1.0.0"

    def test_results_count(self):
        gen = SarifReportGenerator()
        data = json.loads(gen.generate(SAMPLE_FINDINGS))
        assert len(data["runs"][0]["results"]) == 3

    def test_severity_mapping(self):
        gen = SarifReportGenerator()
        data = json.loads(gen.generate(SAMPLE_FINDINGS))
        results = data["runs"][0]["results"]
        levels = [r["level"] for r in results]
        assert "error" in levels  # high → error
        assert "warning" in levels  # medium → warning

    def test_location_info(self):
        gen = SarifReportGenerator()
        data = json.loads(gen.generate(SAMPLE_FINDINGS))
        result = data["runs"][0]["results"][0]
        loc = result["locations"][0]["physicalLocation"]
        assert loc["artifactLocation"]["uri"] == "src/auth.py"
        assert loc["region"]["startLine"] == 42

    def test_code_snippet_included(self):
        gen = SarifReportGenerator()
        data = json.loads(gen.generate(SAMPLE_FINDINGS))
        result = data["runs"][0]["results"][0]
        snippet = result["locations"][0]["physicalLocation"]["region"]["snippet"]
        assert snippet["text"] == "user.name"

    def test_fix_suggestion_included(self):
        gen = SarifReportGenerator()
        data = json.loads(gen.generate(SAMPLE_FINDINGS))
        result = data["runs"][0]["results"][0]
        assert len(result["fixes"]) == 1
        assert "null check" in result["fixes"][0]["description"]["text"]

    def test_fingerprints_present(self):
        gen = SarifReportGenerator()
        data = json.loads(gen.generate(SAMPLE_FINDINGS))
        for result in data["runs"][0]["results"]:
            assert "codeverify/v1" in result["fingerprints"]

    def test_rules_collected(self):
        gen = SarifReportGenerator()
        data = json.loads(gen.generate(SAMPLE_FINDINGS))
        rules = data["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = [r["id"] for r in rules]
        assert "null_safety" in rule_ids
        assert "integer_overflow" in rule_ids

    def test_custom_rules(self):
        gen = SarifReportGenerator()
        rules = [
            SarifRule(
                id="custom_1",
                name="Custom Rule",
                short_description="A custom rule",
                full_description="This is a detailed description",
                help_uri="https://example.com/rules/custom_1",
                tags=["security", "custom"],
            ),
        ]
        findings = [
            {
                "rule_id": "custom_1",
                "message": "Issue",
                "severity": "low",
                "file_path": "a.py",
                "line": 1,
            }
        ]
        data = json.loads(gen.generate(findings, rules=rules))
        rule = data["runs"][0]["tool"]["driver"]["rules"][0]
        assert rule["id"] == "custom_1"
        assert rule["fullDescription"]["text"] == "This is a detailed description"
        assert rule["helpUri"] == "https://example.com/rules/custom_1"
        assert "security" in rule["properties"]["tags"]

    def test_empty_findings(self):
        gen = SarifReportGenerator()
        data = json.loads(gen.generate([]))
        assert len(data["runs"][0]["results"]) == 0

    def test_no_file_path(self):
        gen = SarifReportGenerator()
        findings = [{"rule_id": "r1", "message": "msg", "severity": "low"}]
        data = json.loads(gen.generate(findings))
        result = data["runs"][0]["results"][0]
        assert "locations" not in result

    def test_generate_summary(self):
        gen = SarifReportGenerator()
        summary = gen.generate_summary(SAMPLE_FINDINGS)
        assert summary["total"] == 3
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_rule"]["null_safety"] == 2

    def test_stable_fingerprints(self):
        gen = SarifReportGenerator()
        data1 = json.loads(gen.generate(SAMPLE_FINDINGS))
        data2 = json.loads(gen.generate(SAMPLE_FINDINGS))
        fp1 = data1["runs"][0]["results"][0]["fingerprints"]["codeverify/v1"]
        fp2 = data2["runs"][0]["results"][0]["fingerprints"]["codeverify/v1"]
        assert fp1 == fp2
