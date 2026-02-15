"""Tests for finding_deduplication module."""

from __future__ import annotations

from codeverify_core.finding_deduplication import (
    FindingDeduplicator,
    compute_context_hash,
    fingerprint_finding,
    normalize_code_snippet,
)


class TestNormalizeCodeSnippet:
    def test_strips_comments(self):
        assert "#" not in normalize_code_snippet("x = 1  # comment")
        assert "//" not in normalize_code_snippet("x = 1; // comment")

    def test_collapses_whitespace(self):
        assert normalize_code_snippet("a   =   b") == "a = b"

    def test_normalizes_strings(self):
        result = normalize_code_snippet('print("hello world")')
        assert '"s"' in result

    def test_normalizes_numbers(self):
        result = normalize_code_snippet("x = 42 + 3.14")
        assert "42" not in result
        assert "n" in result

    def test_empty_input(self):
        assert normalize_code_snippet("") == ""


class TestFingerprintFinding:
    def test_same_finding_same_fingerprint(self):
        fp1 = fingerprint_finding("rule1", "a.py", "x = None", 10)
        fp2 = fingerprint_finding("rule1", "a.py", "x = None", 10)
        assert fp1.fingerprint == fp2.fingerprint

    def test_different_rule_different_fingerprint(self):
        fp1 = fingerprint_finding("rule1", "a.py", "x = None", 10)
        fp2 = fingerprint_finding("rule2", "a.py", "x = None", 10)
        assert fp1.fingerprint != fp2.fingerprint

    def test_nearby_lines_same_fingerprint(self):
        fp1 = fingerprint_finding("rule1", "a.py", "x = None", 10)
        fp2 = fingerprint_finding("rule1", "a.py", "x = None", 11)
        assert fp1.fingerprint == fp2.fingerprint  # Same quantized region

    def test_different_file_different_fingerprint(self):
        fp1 = fingerprint_finding("rule1", "a.py", "x = None", 10)
        fp2 = fingerprint_finding("rule1", "b.py", "x = None", 10)
        assert fp1.fingerprint != fp2.fingerprint


class TestContextHash:
    def test_same_region(self):
        h1 = compute_context_hash("a.py", 10)
        h2 = compute_context_hash("a.py", 11)
        assert h1 == h2  # Same window

    def test_different_region(self):
        h1 = compute_context_hash("a.py", 1)
        h2 = compute_context_hash("a.py", 100)
        assert h1 != h2


class TestFindingDeduplicator:
    def test_deduplicates_identical_findings(self):
        dedup = FindingDeduplicator()
        findings = [
            {
                "rule_id": "r1",
                "file_path": "a.py",
                "message": "msg",
                "severity": "high",
                "code_snippet": "x=1",
                "line": 5,
            },
            {
                "rule_id": "r1",
                "file_path": "a.py",
                "message": "msg",
                "severity": "high",
                "code_snippet": "x=1",
                "line": 5,
            },
        ]
        result = dedup.process_findings(findings)
        assert len(result) == 1

    def test_different_findings_kept(self):
        dedup = FindingDeduplicator()
        findings = [
            {
                "rule_id": "r1",
                "file_path": "a.py",
                "message": "msg1",
                "severity": "high",
                "code_snippet": "x=1",
            },
            {
                "rule_id": "r2",
                "file_path": "b.py",
                "message": "msg2",
                "severity": "low",
                "code_snippet": "y=2",
            },
        ]
        result = dedup.process_findings(findings)
        assert len(result) == 2

    def test_suppressed_findings_excluded(self):
        dedup = FindingDeduplicator()
        findings = [
            {
                "rule_id": "r1",
                "file_path": "a.py",
                "message": "fp",
                "severity": "low",
                "code_snippet": "x=1",
            },
        ]
        result = dedup.process_findings(findings)
        fp_hash = result[0]["_fingerprint"]
        dedup.suppress(fp_hash, reason="false positive")
        result2 = dedup.process_findings(findings)
        assert len(result2) == 0

    def test_unsuppress(self):
        dedup = FindingDeduplicator()
        findings = [
            {
                "rule_id": "r1",
                "file_path": "a.py",
                "message": "msg",
                "severity": "low",
                "code_snippet": "x=1",
            },
        ]
        result = dedup.process_findings(findings)
        fp_hash = result[0]["_fingerprint"]
        dedup.suppress(fp_hash)
        dedup.unsuppress(fp_hash)
        result2 = dedup.process_findings(findings)
        assert len(result2) == 1

    def test_get_new_findings(self):
        dedup = FindingDeduplicator()
        batch1 = [
            {
                "rule_id": "r1",
                "file_path": "a.py",
                "message": "old",
                "severity": "low",
                "code_snippet": "x=1",
            }
        ]
        dedup.process_findings(batch1)
        batch2 = [
            {
                "rule_id": "r1",
                "file_path": "a.py",
                "message": "old",
                "severity": "low",
                "code_snippet": "x=1",
            },
            {
                "rule_id": "r2",
                "file_path": "b.py",
                "message": "new",
                "severity": "high",
                "code_snippet": "y=2",
            },
        ]
        new = dedup.get_new_findings(batch2)
        assert len(new) == 1
        assert new[0]["rule_id"] == "r2"

    def test_occurrence_count_increments(self):
        dedup = FindingDeduplicator()
        finding = {
            "rule_id": "r1",
            "file_path": "a.py",
            "message": "msg",
            "severity": "low",
            "code_snippet": "x=1",
        }
        result1 = dedup.process_findings([finding])
        fp = result1[0]["_fingerprint"]
        dedup.process_findings([finding])
        tracked = dedup.get_tracked(fp)
        assert tracked is not None
        assert tracked.occurrence_count == 2

    def test_total_tracked(self):
        dedup = FindingDeduplicator()
        findings = [
            {
                "rule_id": "r1",
                "file_path": "a.py",
                "message": "a",
                "severity": "low",
                "code_snippet": "x=1",
            },
            {
                "rule_id": "r2",
                "file_path": "b.py",
                "message": "b",
                "severity": "low",
                "code_snippet": "y=2",
            },
        ]
        dedup.process_findings(findings)
        assert dedup.total_tracked == 2

    def test_recurring_findings(self):
        dedup = FindingDeduplicator()
        finding = {
            "rule_id": "r1",
            "file_path": "a.py",
            "message": "msg",
            "severity": "low",
            "code_snippet": "x=1",
        }
        for _ in range(5):
            dedup.process_findings([finding])
        recurring = dedup.get_recurring_findings(min_occurrences=3)
        assert len(recurring) == 1
