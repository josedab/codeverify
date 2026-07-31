"""Characterization tests for the worker analysis wire contract."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from codeverify_worker.tasks import analysis as analysis_module
from codeverify_worker.tasks.analysis import (
    AnalysisPipeline,
    AnalysisResult,
    Finding,
    store_analysis_results,
)


def _pipeline() -> AnalysisPipeline:
    return AnalysisPipeline(
        repo_full_name="owner/repo",
        pr_number=42,
        head_sha="abcdef1234567890",
        base_sha=None,
        installation_id=None,
    )


def _finding(**overrides: object) -> Finding:
    values = {
        "category": "security",
        "severity": "medium",
        "title": "Unsafe query",
        "description": "User input reaches a query.",
        "file_path": "src/db.py",
        "line_start": None,
        "line_end": None,
        "code_snippet": None,
        "fix_suggestion": None,
        "confidence": 0.8,
        "verification_type": "pattern",
    }
    values.update(overrides)
    return Finding(**values)


def test_finding_serializes_exact_flat_keys_and_null_default() -> None:
    payload = _pipeline()._finding_to_dict(_finding())

    assert payload == {
        "category": "security",
        "severity": "medium",
        "title": "Unsafe query",
        "description": "User input reaches a query.",
        "file_path": "src/db.py",
        "line_start": None,
        "line_end": None,
        "code_snippet": None,
        "fix_suggestion": None,
        "confidence": 0.8,
        "verification_type": "pattern",
        "verification_proof": None,
    }


@pytest.mark.parametrize("severity", ["critical", "high", "medium", "low", "info"])
def test_finding_preserves_current_severity_spellings(severity: str) -> None:
    payload = _pipeline()._finding_to_dict(_finding(severity=severity))

    assert payload["severity"] == severity


@pytest.mark.parametrize("verification_type", ["formal", "ai", "pattern"])
def test_finding_preserves_current_verification_type_spellings(
    verification_type: str,
) -> None:
    payload = _pipeline()._finding_to_dict(
        _finding(verification_type=verification_type),
    )

    assert payload["verification_type"] == verification_type


def test_info_finding_is_transported_without_an_info_summary_bucket() -> None:
    pipeline = _pipeline()
    pipeline.findings = [_finding(severity="info")]

    assert pipeline._finding_to_dict(pipeline.findings[0])["severity"] == "info"
    assert pipeline._calculate_summary() == {
        "total_issues": 1,
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "pass": True,
    }


@pytest.mark.asyncio
async def test_run_stage_serializes_exact_success_shape_and_timestamps() -> None:
    pipeline = _pipeline()
    stage = AsyncMock(return_value={"issues_found": 2})
    started_at = datetime(2026, 7, 31, 1, 2, 3, 456789)
    completed_at = datetime(2026, 7, 31, 1, 2, 4, 706789)

    with patch.object(analysis_module, "datetime") as datetime_mock:
        datetime_mock.utcnow.side_effect = [started_at, completed_at]
        await pipeline._run_stage("semantic", stage)

    assert pipeline.stages == [
        {
            "name": "semantic",
            "started_at": "2026-07-31T01:02:03.456789",
            "status": "completed",
            "completed_at": "2026-07-31T01:02:04.706789",
            "duration_ms": 1250.0,
            "result": {"issues_found": 2},
        }
    ]


@pytest.mark.asyncio
async def test_store_analysis_results_sends_exact_current_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    internal_api_key = "contract-test-key"
    monkeypatch.setenv("API_URL", "https://internal.example")
    monkeypatch.setenv("INTERNAL_API_KEY", internal_api_key)

    finding = _pipeline()._finding_to_dict(_finding(severity="info"))
    stage = {
        "name": "semantic",
        "started_at": "2026-07-31T01:02:03.456789",
        "status": "completed",
        "completed_at": "2026-07-31T01:02:04.706789",
        "duration_ms": 1250.0,
        "result": {"issues_found": 1},
    }
    result = AnalysisResult(
        analysis_id="owner/repo#42@abcdef12",
        status="completed",
        findings=[finding],
        stages=[stage],
        started_at=datetime(2026, 7, 31, 1, 2, 3, 456789),
        completed_at=datetime(2026, 7, 31, 1, 2, 5, 987654),
        summary={
            "total_issues": 1,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "pass": True,
        },
    )

    response = Mock(status_code=201)
    response.json.return_value = {"id": "persisted-analysis-id"}
    client = Mock()
    client.post = AsyncMock(return_value=response)
    client_context = MagicMock()
    client_context.__aenter__.return_value = client
    client_context.__aexit__.return_value = None

    with patch.object(
        analysis_module.httpx,
        "AsyncClient",
        return_value=client_context,
    ) as client_factory:
        await store_analysis_results(
            result=result,
            repo_id=123,
            repo_full_name="owner/repo",
            pr_number=42,
            pr_title=None,
            head_sha="abcdef1234567890",
            base_sha=None,
        )

    client_factory.assert_called_once_with(timeout=30.0)
    client.post.assert_awaited_once_with(
        "https://internal.example/internal/analyses",
        json={
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
            "findings": [finding],
            "stages": [stage],
            "summary": result.summary,
        },
        headers={
            "Authorization": f"Bearer {internal_api_key}",
            "Content-Type": "application/json",
        },
    )
