"""Characterization tests for the internal analysis request models."""

import json

import pytest

from codeverify_api.routers.internal import AnalysisData, FindingData


def _worker_finding(
    *,
    severity: str = "medium",
    verification_type: str = "pattern",
) -> dict[str, object]:
    return {
        "category": "security",
        "severity": severity,
        "title": "Unsafe query",
        "description": "User input reaches a query.",
        "file_path": "src/db.py",
        "line_start": None,
        "line_end": None,
        "code_snippet": None,
        "fix_suggestion": None,
        "confidence": 0.8,
        "verification_type": verification_type,
        "verification_proof": None,
    }


def _worker_stage() -> dict[str, object]:
    return {
        "name": "semantic",
        "started_at": "2026-07-31T01:02:03.456789",
        "status": "completed",
        "completed_at": "2026-07-31T01:02:04.706789",
        "duration_ms": 1250.0,
        "result": {"issues_found": 1},
    }


def test_analysis_data_applies_current_optional_and_collection_defaults() -> None:
    data = AnalysisData.model_validate(
        {
            "repo_id": 123,
            "repo_full_name": "owner/repo",
            "pr_number": 42,
            "head_sha": "abcdef1234567890",
            "status": "completed",
            "started_at": "2026-07-31T01:02:03.456789",
            "completed_at": "2026-07-31T01:02:05.987654",
        }
    )

    assert data.model_dump() == {
        "repo_id": 123,
        "repo_full_name": "owner/repo",
        "pr_number": 42,
        "pr_title": None,
        "head_sha": "abcdef1234567890",
        "base_sha": None,
        "status": "completed",
        "started_at": "2026-07-31T01:02:03.456789",
        "completed_at": "2026-07-31T01:02:05.987654",
        "error_message": None,
        "findings": [],
        "stages": [],
        "summary": {},
    }


def test_analysis_data_accepts_current_worker_json_unchanged() -> None:
    payload = {
        "repo_id": 123,
        "repo_full_name": "owner/repo",
        "pr_number": 42,
        "pr_title": None,
        "head_sha": "abcdef1234567890",
        "base_sha": None,
        "status": "completed",
        "started_at": "2026-07-31T01:02:03.456789",
        "completed_at": "2026-07-31T01:02:05.987654",
        "error_message": None,
        "findings": [_worker_finding(severity="info")],
        "stages": [_worker_stage()],
        "summary": {
            "total_issues": 1,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "pass": True,
        },
    }

    data = AnalysisData.model_validate_json(json.dumps(payload))

    assert data.model_dump() == payload


def test_finding_data_uses_current_flat_field_spellings_and_null_defaults() -> None:
    data = FindingData.model_validate(
        {
            "category": "security",
            "severity": "medium",
            "title": "Unsafe query",
            "file_path": "src/db.py",
        }
    )

    assert data.model_dump() == {
        "category": "security",
        "severity": "medium",
        "title": "Unsafe query",
        "description": None,
        "file_path": "src/db.py",
        "line_start": None,
        "line_end": None,
        "code_snippet": None,
        "fix_suggestion": None,
        "fix_diff": None,
        "confidence": None,
        "verification_type": None,
        "verification_proof": None,
        "metadata": None,
    }


@pytest.mark.parametrize("severity", ["critical", "high", "medium", "low", "info"])
def test_finding_data_preserves_current_severity_spellings(severity: str) -> None:
    data = FindingData.model_validate(_worker_finding(severity=severity))

    assert data.severity == severity


@pytest.mark.parametrize("verification_type", ["formal", "ai", "pattern"])
def test_finding_data_preserves_current_verification_type_spellings(
    verification_type: str,
) -> None:
    data = FindingData.model_validate(
        _worker_finding(verification_type=verification_type),
    )

    assert data.verification_type == verification_type
