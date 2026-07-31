"""Characterization tests for the current analysis worker."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from codeverify_core.vcs import CheckConclusion, CheckStatus
from codeverify_worker.tasks import analysis as analysis_module
from codeverify_worker.tasks.analysis import (
    AnalysisPipeline,
    AnalysisResult,
    Finding,
    format_check_annotations,
    format_github_comment,
    post_results_to_github,
    store_analysis_results,
)


def _finding(**overrides):
    values = {
        "category": "security",
        "severity": "high",
        "title": "SQL injection",
        "description": "User-controlled input reaches a query.",
        "file_path": "src/db.py",
        "line_start": 12,
        "line_end": 12,
        "code_snippet": "cursor.execute(query)",
        "fix_suggestion": "cursor.execute(query, params)",
        "confidence": 0.95,
        "verification_type": "formal",
        "verification_proof": "counterexample",
    }
    values.update(overrides)
    return Finding(**values)


@pytest.fixture
def pipeline():
    return AnalysisPipeline(
        repo_full_name="owner/repo",
        pr_number=42,
        head_sha="abcdef1234567890",
        base_sha="1234567890abcdef",
        installation_id=99,
    )


@pytest.fixture
def analysis_result():
    finding = _finding()
    return AnalysisResult(
        analysis_id="owner/repo#42@abcdef12",
        status="completed",
        findings=[
            {
                "category": finding.category,
                "severity": finding.severity,
                "title": finding.title,
                "description": finding.description,
                "file_path": finding.file_path,
                "line_start": finding.line_start,
                "line_end": finding.line_end,
                "code_snippet": finding.code_snippet,
                "fix_suggestion": finding.fix_suggestion,
                "confidence": finding.confidence,
                "verification_type": finding.verification_type,
                "verification_proof": finding.verification_proof,
            }
        ],
        stages=[{"name": "fetch", "status": "completed", "result": {"files_changed": 1}}],
        started_at=datetime(2026, 7, 30, 12, 0, 0),
        completed_at=datetime(2026, 7, 30, 12, 0, 5),
        summary={
            "total_issues": 1,
            "critical": 0,
            "high": 1,
            "medium": 0,
            "low": 0,
            "pass": False,
        },
    )


class TestAnalysisPipeline:
    def test_initializes_from_repository_and_pull_request_inputs(self, pipeline):
        assert pipeline.repo_full_name == "owner/repo"
        assert pipeline.pr_number == 42
        assert pipeline.head_sha == "abcdef1234567890"
        assert pipeline.base_sha == "1234567890abcdef"
        assert pipeline.installation_id == 99
        assert pipeline.stages == []
        assert pipeline.findings == []
        assert pipeline.pr_files == []
        assert pipeline.pr_diff == ""
        assert pipeline.file_contents == {}

    @pytest.mark.asyncio
    async def test_run_preserves_current_stage_order_and_builds_result(self, pipeline):
        pipeline.findings = [_finding()]
        stage_specs = [
            ("fetch", "_fetch_pr_data", {"files_changed": 1}),
            ("parse", "_parse_code", {"files_parsed": 1}),
            ("semantic", "_semantic_analysis", {"issues_found": 0}),
            ("verify", "_formal_verification", {"proofs_attempted": 0}),
            ("security", "_security_analysis", {"vulnerabilities_found": 0}),
            ("synthesize", "_synthesize_results", {"total_findings": 1}),
        ]
        stage_mocks = {}
        for _, method_name, stage_result in stage_specs:
            stage_mock = AsyncMock(return_value=stage_result)
            setattr(pipeline, method_name, stage_mock)
            stage_mocks[method_name] = stage_mock

        result = await pipeline.run()

        assert result.analysis_id == "owner/repo#42@abcdef12"
        assert result.status == "completed"
        assert result.error is None
        assert [stage["name"] for stage in result.stages] == [
            "fetch",
            "parse",
            "semantic",
            "verify",
            "security",
            "synthesize",
        ]
        assert [stage["result"] for stage in result.stages] == [
            stage_result for _, _, stage_result in stage_specs
        ]
        assert all(stage["status"] == "completed" for stage in result.stages)
        assert result.summary == {
            "total_issues": 1,
            "critical": 0,
            "high": 1,
            "medium": 0,
            "low": 0,
            "pass": False,
        }
        assert result.findings == [pipeline._finding_to_dict(pipeline.findings[0])]
        assert result.started_at <= result.completed_at
        for stage_mock in stage_mocks.values():
            stage_mock.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_fetch_stage_summarizes_preloaded_pull_request_files(self, pipeline):
        pipeline.pr_files = [
            {"filename": "src/main.py", "additions": 8, "deletions": 2},
            {"filename": "src/service.py", "additions": 3},
        ]

        result = await pipeline._fetch_pr_data()

        assert result == {"files_changed": 2, "additions": 11, "deletions": 2}

    @pytest.mark.asyncio
    async def test_parse_stage_aggregates_parser_and_adapter_results(self, pipeline):
        pipeline.file_contents = {
            "src/main.py": "def first():\n    pass\n\ndef second():\n    pass\n",
            "src/lib.rs": "fn helper() {}",
            "README.md": "# project",
        }

        python_parser = Mock(language="python")
        python_parser.can_parse.side_effect = lambda path: path.endswith(".py")
        python_parser.parse.return_value = SimpleNamespace(
            functions=[SimpleNamespace(name="first"), SimpleNamespace(name="second")],
            classes=[SimpleNamespace(name="Service")],
        )

        other_parsers = [Mock() for _ in range(3)]
        for parser_mock in other_parsers:
            parser_mock.can_parse.return_value = False

        rust_language = SimpleNamespace(value="rust")
        adapter_registry = Mock()
        adapter_registry.analyze_file.return_value = SimpleNamespace(
            functions=[SimpleNamespace(name="helper")]
        )

        def detect_language(path):
            return rust_language if path.endswith(".rs") else None

        with (
            patch("codeverify_verifier.parsers.PythonParser", return_value=python_parser),
            patch(
                "codeverify_verifier.parsers.TypeScriptParser",
                return_value=other_parsers[0],
            ),
            patch("codeverify_verifier.parsers.GoParser", return_value=other_parsers[1]),
            patch("codeverify_verifier.parsers.JavaParser", return_value=other_parsers[2]),
            patch(
                "codeverify_core.language_adapter.get_adapter_registry",
                return_value=adapter_registry,
            ),
            patch(
                "codeverify_core.language_support.detect_language",
                side_effect=detect_language,
            ),
        ):
            result = await pipeline._parse_code()

        assert result == {
            "files_parsed": 3,
            "functions_found": 3,
            "classes_found": 1,
            "languages": {"python": 1, "rust": 1},
        }
        python_parser.parse.assert_called_once_with(
            pipeline.file_contents["src/main.py"],
            "src/main.py",
        )
        adapter_registry.analyze_file.assert_called_once_with(
            pipeline.file_contents["src/lib.rs"],
            rust_language,
        )

    @pytest.mark.asyncio
    async def test_semantic_stage_maps_mocked_llm_concerns_to_findings(self, pipeline):
        pipeline.file_contents = {"src/main.py": "def load():\n    return query(user_input)\n"}
        pipeline.pr_diff = "@@ -1 +1 @@"

        agent = Mock()
        agent.analyze = AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                data={
                    "functions": [
                        {
                            "name": "load",
                            "concerns": ["User input may reach the query unchanged."],
                        }
                    ]
                },
            )
        )
        detected_language = SimpleNamespace(value="python")
        adapter_registry = Mock()
        adapter_registry.get.return_value = None

        with (
            patch("codeverify_agents.semantic.SemanticAgent", return_value=agent),
            patch.object(pipeline, "_detect_language", return_value="python"),
            patch(
                "codeverify_core.language_adapter.get_adapter_registry",
                return_value=adapter_registry,
            ),
            patch(
                "codeverify_core.language_support.detect_language",
                return_value=detected_language,
            ),
        ):
            result = await pipeline._semantic_analysis()

        assert result == {"issues_found": 1}
        agent.analyze.assert_awaited_once_with(
            code=pipeline.file_contents["src/main.py"],
            context={
                "file_path": "src/main.py",
                "language": "python",
                "diff": "@@ -1 +1 @@",
                "adapter_context": "",
            },
        )
        assert pipeline.findings == [
            Finding(
                category="logic_error",
                severity="medium",
                title="Potential issue in load",
                description="User input may reach the query unchanged.",
                file_path="src/main.py",
                line_start=None,
                line_end=None,
                code_snippet=None,
                fix_suggestion=None,
                confidence=0.7,
                verification_type="ai",
            )
        ]

    @pytest.mark.asyncio
    async def test_formal_stage_uses_mocked_parser_and_verifier(self, pipeline):
        pipeline.file_contents = {
            "src/math.py": "def multiply(values, i):\n    return values[i] * 2\n"
        }
        parsed_function = SimpleNamespace(
            name="multiply",
            calls=["*"],
            body="return values[i] * 2",
            line_start=1,
            line_end=2,
        )
        parser = Mock()
        parser.can_parse.return_value = True
        parser.parse.return_value = SimpleNamespace(functions=[parsed_function])

        verifier = Mock()
        verifier.check_integer_overflow.return_value = {
            "satisfiable": True,
            "message": "Counterexample found.",
            "counterexample": {"value": 1_000_000},
        }
        verifier.check_array_bounds.return_value = {"satisfiable": False}

        with (
            patch("codeverify_verifier.Z3Verifier", return_value=verifier) as verifier_factory,
            patch(
                "codeverify_verifier.parsers.PythonParser",
                return_value=parser,
            ) as parser_factory,
        ):
            result = await pipeline._formal_verification()

        assert result == {
            "proofs_attempted": 2,
            "proofs_succeeded": 0,
            "issues_found": 1,
        }
        verifier_factory.assert_called_once_with(timeout_ms=30000)
        parser_factory.assert_called_once_with()
        verifier.check_integer_overflow.assert_called_once_with(
            var_name="multiply",
            operation="mul",
            operand1_range=(0, 1000000),
            operand2_range=(0, 1000000),
        )
        verifier.check_array_bounds.assert_called_once_with(
            index_var="i",
            index_range=None,
            array_length=100,
        )
        assert pipeline.findings[0].category == "overflow"
        assert pipeline.findings[0].severity == "high"
        assert pipeline.findings[0].verification_type == "formal"

    def test_summary_counts_severities_and_blocks_on_high_or_critical(self, pipeline):
        pipeline.findings = [
            _finding(severity="critical"),
            _finding(severity="high"),
            _finding(severity="medium"),
            _finding(severity="low"),
            _finding(severity="info"),
        ]

        assert pipeline._calculate_summary() == {
            "total_issues": 5,
            "critical": 1,
            "high": 1,
            "medium": 1,
            "low": 1,
            "pass": False,
        }

        pipeline.findings = [_finding(severity="medium"), _finding(severity="low")]

        assert pipeline._calculate_summary() == {
            "total_issues": 2,
            "critical": 0,
            "high": 0,
            "medium": 1,
            "low": 1,
            "pass": True,
        }

    @pytest.mark.asyncio
    async def test_run_returns_current_error_result_without_raising(self, pipeline):
        pipeline.findings = [_finding()]
        pipeline._fetch_pr_data = AsyncMock(side_effect=RuntimeError("GitHub unavailable"))
        pipeline._parse_code = AsyncMock()
        pipeline._semantic_analysis = AsyncMock()
        pipeline._formal_verification = AsyncMock()
        pipeline._security_analysis = AsyncMock()
        pipeline._synthesize_results = AsyncMock()

        result = await pipeline.run()

        assert result.analysis_id == "owner/repo#42@abcdef12"
        assert result.status == "failed"
        assert result.error == "GitHub unavailable"
        assert result.findings == []
        assert result.summary == {}
        assert result.stages == []
        pipeline._fetch_pr_data.assert_awaited_once_with()
        pipeline._parse_code.assert_not_awaited()
        pipeline._semantic_analysis.assert_not_awaited()
        pipeline._formal_verification.assert_not_awaited()
        pipeline._security_analysis.assert_not_awaited()
        pipeline._synthesize_results.assert_not_awaited()


class TestGitHubFormatting:
    def test_formats_pull_request_comment(self, analysis_result):
        expected = """## 🔍 CodeVerify Analysis

### Summary

| Total | Critical | High | Medium | Low |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0 | 1 | 0 | 0 |

❌ **Status: Issues Found**

### Findings

<details>
<summary>🟠 <b>SQL injection</b> (src/db.py:12)</summary>

User-controlled input reaches a query.

**Suggested fix:**
```
cursor.execute(query, params)
```

</details>


---
*Powered by [CodeVerify](https://codeverify.dev)*"""

        assert format_github_comment(analysis_result) == expected

    def test_formats_and_limits_check_annotations(self, analysis_result):
        extra_findings = [
            {
                **analysis_result.findings[0],
                "title": f"Issue {index}",
                "severity": "low",
            }
            for index in range(51)
        ]

        annotations = format_check_annotations([analysis_result.findings[0], *extra_findings])

        assert len(annotations) == 50
        assert annotations[0] == {
            "path": "src/db.py",
            "start_line": 12,
            "end_line": 12,
            "annotation_level": "failure",
            "title": "SQL injection",
            "message": "User-controlled input reaches a query.",
            "raw_details": (
                "Confidence: 95%\n"
                "Verification: formal\n"
                "\nSuggested fix:\n"
                "cursor.execute(query, params)"
            ),
        }
        assert annotations[1]["annotation_level"] == "notice"

    @pytest.mark.asyncio
    async def test_posts_formatted_results_through_mocked_github_client(
        self,
        analysis_result,
    ):
        github = Mock()
        github.create_check_run = AsyncMock(return_value=SimpleNamespace(id=321))
        github.update_check_run = AsyncMock()
        github.create_pull_request_comment = AsyncMock(return_value=SimpleNamespace(id=654))

        with patch.object(
            analysis_module,
            "_create_github_client",
            return_value=github,
        ) as github_factory:
            posted = await post_results_to_github(
                result=analysis_result,
                repo_full_name="owner/repo",
                pr_number=42,
                head_sha="abcdef1234567890",
                installation_id=99,
            )

        assert posted == {
            "check_run_id": 321,
            "comment_id": 654,
            "review_id": None,
            "annotations_count": 1,
        }
        github_factory.assert_called_once_with(99)

        create_call = github.create_check_run.await_args
        assert create_call.kwargs["repo_full_name"] == "owner/repo"
        assert create_call.kwargs["head_sha"] == "abcdef1234567890"
        assert create_call.kwargs["check_run"].status == CheckStatus.IN_PROGRESS

        update_call = github.update_check_run.await_args
        completed_check = update_call.kwargs["check_run"]
        assert update_call.kwargs["repo_full_name"] == "owner/repo"
        assert update_call.kwargs["check_run_id"] == 321
        assert completed_check.status == CheckStatus.COMPLETED
        assert completed_check.conclusion == CheckConclusion.FAILURE
        assert completed_check.title == "CodeVerify: 1 issues found"
        assert len(completed_check.annotations) == 1
        assert completed_check.annotations[0].path == "src/db.py"

        github.create_pull_request_comment.assert_awaited_once_with(
            repo_full_name="owner/repo",
            pr_number=42,
            body=format_github_comment(analysis_result),
        )


class TestPersistence:
    @pytest.mark.asyncio
    async def test_posts_exact_analysis_payload_to_internal_api(
        self,
        analysis_result,
        monkeypatch,
    ):
        monkeypatch.setenv("API_URL", "https://internal.example")
        monkeypatch.setenv("INTERNAL_API_KEY", "worker-secret")

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
            stored = await store_analysis_results(
                result=analysis_result,
                repo_id=123,
                repo_full_name="owner/repo",
                pr_number=42,
                pr_title="Improve verification",
                head_sha="abcdef1234567890",
                base_sha="1234567890abcdef",
            )

        assert stored == {
            "analysis_id": "persisted-analysis-id",
            "findings_count": 1,
            "stored": True,
        }
        client_factory.assert_called_once_with(timeout=30.0)
        client.post.assert_awaited_once_with(
            "https://internal.example/internal/analyses",
            json={
                "repo_id": 123,
                "repo_full_name": "owner/repo",
                "pr_number": 42,
                "pr_title": "Improve verification",
                "head_sha": "abcdef1234567890",
                "base_sha": "1234567890abcdef",
                "status": "completed",
                "started_at": "2026-07-30T12:00:00",
                "completed_at": "2026-07-30T12:00:05",
                "error_message": None,
                "findings": analysis_result.findings,
                "stages": analysis_result.stages,
                "summary": analysis_result.summary,
            },
            headers={
                "Authorization": "Bearer worker-secret",
                "Content-Type": "application/json",
            },
        )
