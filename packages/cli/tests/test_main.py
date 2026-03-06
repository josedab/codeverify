"""Unit tests for cli/main.py — CLI commands via CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from codeverify_cli.main import (
    _sarif_level,
    apply_fix,
    cli,
    get_staged_files,
    to_sarif,
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_dir(tmp_path):
    """Create a temp directory with a .codeverify.yml for tests."""
    config = tmp_path / ".codeverify.yml"
    config.write_text('version: "1"\nlanguages:\n  - python\n')
    return tmp_path


# ============================================================================
# Utility function tests
# ============================================================================


class TestSarifLevel:
    """Test _sarif_level conversion."""

    def test_critical_maps_to_error(self):
        assert _sarif_level("critical") == "error"

    def test_high_maps_to_error(self):
        assert _sarif_level("high") == "error"

    def test_medium_maps_to_warning(self):
        assert _sarif_level("medium") == "warning"

    def test_low_maps_to_note(self):
        assert _sarif_level("low") == "note"

    def test_info_maps_to_note(self):
        assert _sarif_level("info") == "note"

    def test_unknown_maps_to_note(self):
        assert _sarif_level("unknown") == "note"


class TestToSarif:
    """Test SARIF output generation."""

    def test_sarif_schema_structure(self):
        results = MagicMock()
        results.findings = []
        sarif = to_sarif(results)

        assert sarif["version"] == "2.1.0"
        assert "$schema" in sarif
        assert len(sarif["runs"]) == 1
        assert sarif["runs"][0]["tool"]["driver"]["name"] == "CodeVerify"

    def test_sarif_includes_findings(self):
        results = MagicMock()
        results.findings = [
            {
                "category": "null_safety",
                "severity": "high",
                "description": "Possible null dereference",
                "file_path": "src/main.py",
                "line_start": 42,
                "line_end": 42,
            }
        ]
        sarif = to_sarif(results)
        sarif_results = sarif["runs"][0]["results"]

        assert len(sarif_results) == 1
        assert sarif_results[0]["ruleId"] == "null_safety"
        assert sarif_results[0]["level"] == "error"
        assert sarif_results[0]["message"]["text"] == "Possible null dereference"

    def test_sarif_empty_findings(self):
        results = MagicMock()
        results.findings = []
        sarif = to_sarif(results)
        assert sarif["runs"][0]["results"] == []


class TestGetStagedFiles:
    """Test get_staged_files helper."""

    def test_returns_empty_on_no_git(self, tmp_path):
        result = get_staged_files(str(tmp_path))
        assert result == []

    def test_returns_file_paths(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="file1.py\nfile2.py\n")
            result = get_staged_files(str(tmp_path))
            assert len(result) == 2
            assert result[0] == Path(str(tmp_path)) / "file1.py"

    def test_returns_empty_on_subprocess_error(self, tmp_path):
        with patch("subprocess.run", side_effect=Exception("git not found")):
            result = get_staged_files(str(tmp_path))
            assert result == []


class TestApplyFix:
    """Test apply_fix helper."""

    def test_raises_for_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            apply_fix({"file_path": "/nonexistent/path.py", "line_start": 1})


# ============================================================================
# CLI command tests
# ============================================================================


class TestCliGroup:
    """Test the main cli group."""

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "CodeVerify" in result.output
        assert "analyze" in result.output


class TestInitCommand:
    """Test the 'init' command."""

    def test_creates_config_file(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "Created" in result.output
            assert Path(".codeverify.yml").exists()

    def test_does_not_overwrite_existing(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path(".codeverify.yml").write_text("existing")
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "already exists" in result.output
            assert Path(".codeverify.yml").read_text() == "existing"

    def test_force_overwrites(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path(".codeverify.yml").write_text("old")
            result = runner.invoke(cli, ["init", "--force"])
            assert result.exit_code == 0
            content = Path(".codeverify.yml").read_text()
            assert "version" in content
            assert content != "old"


class TestValidateCommand:
    """Test the 'validate' command."""

    def test_valid_config(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path(".codeverify.yml").write_text('version: "1"\nlanguages:\n  - python\n')
            result = runner.invoke(cli, ["validate"])
            assert result.exit_code == 0
            assert "valid" in result.output.lower()

    def test_missing_config(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["validate"])
            assert result.exit_code != 0

    def test_invalid_yaml(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path(".codeverify.yml").write_text("{{invalid yaml")
            result = runner.invoke(cli, ["validate"])
            assert result.exit_code != 0


class TestStatusCommand:
    """Test the 'status' command."""

    def test_shows_status(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["status"])
            assert result.exit_code == 0
            assert "CodeVerify Status" in result.output

    def test_shows_config_found(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path(".codeverify.yml").write_text("version: 1")
            result = runner.invoke(cli, ["status"])
            assert result.exit_code == 0
            assert "Configuration" in result.output


class TestAnalyzeCommand:
    """Test the 'analyze' command with mocked analyzer."""

    def test_analyze_json_format(self, runner, tmp_path):
        mock_results = MagicMock()
        mock_results.findings = []
        mock_results.to_dict.return_value = {"findings": [], "summary": {}}

        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("codeverify_cli.main.LocalAnalyzer") as MockAnalyzer,
            patch("codeverify_cli.main.asyncio") as mock_asyncio,
        ):
            Path(".codeverify.yml").write_text("version: 1")
            Path("test.py").write_text("x = 1")
            mock_asyncio.run.return_value = mock_results
            result = runner.invoke(cli, ["analyze", ".", "-f", "json"])

            assert result.exit_code == 0

    def test_analyze_sarif_format(self, runner, tmp_path):
        mock_results = MagicMock()
        mock_results.findings = []

        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("codeverify_cli.main.LocalAnalyzer") as MockAnalyzer,
            patch("codeverify_cli.main.asyncio") as mock_asyncio,
        ):
            Path(".codeverify.yml").write_text("version: 1")
            Path("test.py").write_text("x = 1")
            mock_asyncio.run.return_value = mock_results
            result = runner.invoke(cli, ["analyze", ".", "-f", "sarif"])

            assert result.exit_code == 0

    def test_analyze_severity_filter(self, runner, tmp_path):
        mock_results = MagicMock()
        mock_results.findings = [
            {"severity": "low", "title": "minor"},
            {"severity": "critical", "title": "major"},
        ]
        mock_results.to_dict.return_value = {"summary": {"critical": 1, "low": 1, "total": 2}}
        mock_results.files_analyzed = 1
        mock_results.functions_found = 0
        mock_results.classes_found = 0
        mock_results.duration_ms = 100.0

        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("codeverify_cli.main.LocalAnalyzer"),
            patch("codeverify_cli.main.asyncio") as mock_asyncio,
        ):
            Path(".codeverify.yml").write_text("version: 1")
            Path("test.py").write_text("x = 1")
            mock_asyncio.run.return_value = mock_results
            result = runner.invoke(cli, ["analyze", ".", "-s", "critical"])
            # Filtered to only critical
            assert result.exit_code != 0  # fail-on high default, critical found

    def test_analyze_fail_on_none(self, runner, tmp_path):
        mock_results = MagicMock()
        mock_results.findings = [{"severity": "critical", "title": "big"}]
        mock_results.to_dict.return_value = {"summary": {"critical": 1, "total": 1}}
        mock_results.files_analyzed = 1
        mock_results.functions_found = 0
        mock_results.classes_found = 0
        mock_results.duration_ms = 50.0

        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("codeverify_cli.main.LocalAnalyzer"),
            patch("codeverify_cli.main.asyncio") as mock_asyncio,
        ):
            Path(".codeverify.yml").write_text("version: 1")
            Path("test.py").write_text("x = 1")
            mock_asyncio.run.return_value = mock_results
            result = runner.invoke(cli, ["analyze", ".", "--fail-on", "none"])
            assert result.exit_code == 0
