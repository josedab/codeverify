"""Tests for proof coverage dashboard."""

from datetime import datetime

import pytest

from codeverify_core.proof_coverage import (
    CoverageTrend,
    DashboardData,
    FileCoverage,
    FunctionCoverage,
    LineCoverage,
    ProofCoverageCalculator,
    ProofCoverageDashboard,
    ProofStatus,
    RepositoryCoverage,
    VerificationCategory,
    get_proof_coverage_dashboard,
    reset_proof_coverage_dashboard,
)


class TestProofStatus:
    """Tests for ProofStatus enum."""

    def test_all_statuses(self):
        """Test all proof statuses exist."""
        assert ProofStatus.PROVEN.value == "proven"
        assert ProofStatus.DISPROVEN.value == "disproven"
        assert ProofStatus.TIMEOUT.value == "timeout"
        assert ProofStatus.UNKNOWN.value == "unknown"
        assert ProofStatus.NOT_ATTEMPTED.value == "not_attempted"


class TestVerificationCategory:
    """Tests for VerificationCategory enum."""

    def test_all_categories(self):
        """Test all verification categories exist."""
        assert VerificationCategory.NULL_SAFETY.value == "null_safety"
        assert VerificationCategory.BOUNDS_CHECK.value == "bounds_check"
        assert VerificationCategory.DIVISION.value == "division"
        assert VerificationCategory.TYPE_SAFETY.value == "type_safety"
        assert VerificationCategory.OVERFLOW.value == "overflow"


class TestLineCoverage:
    """Tests for LineCoverage dataclass."""

    def test_creation(self):
        """Test line coverage creation."""
        coverage = LineCoverage(
            line_number=42,
            proof_status=ProofStatus.PROVEN,
            categories=[VerificationCategory.NULL_SAFETY],
            last_verified=datetime.utcnow(),
        )

        assert coverage.line_number == 42
        assert coverage.proof_status == ProofStatus.PROVEN
        assert VerificationCategory.NULL_SAFETY in coverage.categories

    def test_default_values(self):
        """Test default line coverage values."""
        coverage = LineCoverage(
            line_number=1,
            proof_status=ProofStatus.NOT_ATTEMPTED,
        )

        assert coverage.categories == []
        assert coverage.last_verified is None


class TestFunctionCoverage:
    """Tests for FunctionCoverage dataclass."""

    def test_creation(self):
        """Test function coverage creation."""
        coverage = FunctionCoverage(
            function_name="process_payment",
            start_line=10,
            end_line=50,
            line_coverage=[],
            proven_lines=45,
        )

        assert coverage.function_name == "process_payment"
        assert coverage.proven_lines == 45

    def test_default_values(self):
        """Test default function coverage values."""
        coverage = FunctionCoverage(
            function_name="test",
            start_line=1,
            end_line=10,
        )

        assert coverage.proof_strength == 0.0
        assert coverage.line_coverage == []


class TestFileCoverage:
    """Tests for FileCoverage dataclass."""

    def test_creation(self):
        """Test file coverage creation."""
        coverage = FileCoverage(
            file_path="src/api/handlers.py",
            language="python",
            total_lines=100,
            executable_lines=85,
            covered_lines=70,
            proven_lines=60,
            coverage_percentage=82.4,
        )

        assert coverage.file_path == "src/api/handlers.py"
        assert coverage.total_lines == 100
        assert coverage.covered_lines == 70

    def test_coverage_percentage(self):
        """Test coverage percentage calculation."""
        coverage = FileCoverage(
            file_path="test.py",
            language="python",
            total_lines=100,
            executable_lines=80,
            covered_lines=60,
            proven_lines=60,
            coverage_percentage=75.0,
        )

        assert coverage.coverage_percentage == 75.0


class TestRepositoryCoverage:
    """Tests for RepositoryCoverage dataclass."""

    def test_creation(self):
        """Test repository coverage creation."""
        coverage = RepositoryCoverage(
            repository="myorg/myrepo",
            total_files=50,
            files_with_proofs=40,
            total_lines=5000,
            proven_lines=4000,
        )

        assert coverage.repository == "myorg/myrepo"
        assert coverage.files_with_proofs == 40

    def test_default_values(self):
        """Test default repository coverage values."""
        coverage = RepositoryCoverage(
            repository="test/repo",
        )

        assert coverage.total_files == 0
        assert coverage.file_coverage == 0.0
        assert coverage.branch == "main"


class TestCoverageTrend:
    """Tests for CoverageTrend dataclass."""

    def test_creation(self):
        """Test trend creation."""
        trend = CoverageTrend(
            date=datetime.utcnow(),
            line_coverage=75.5,
            function_coverage=80.0,
            proof_strength=0.9,
            proven_count=1500,
            disproven_count=10,
        )

        assert trend.line_coverage == 75.5
        assert trend.proven_count == 1500


class TestDashboardData:
    """Tests for DashboardData dataclass."""

    def test_creation(self):
        """Test dashboard data creation."""
        repo_cov = RepositoryCoverage(
            repository="test/repo",
            total_files=10,
            files_with_proofs=8,
            total_lines=1000,
            proven_lines=800,
        )
        data = DashboardData(
            repository="test/repo",
            current_coverage=repo_cov,
            total_proofs=100,
            passed_proofs=80,
        )

        assert data.current_coverage.proven_lines == 800
        assert data.trends == []


class TestProofCoverageCalculator:
    """Tests for ProofCoverageCalculator."""

    @pytest.fixture
    def calculator(self):
        return ProofCoverageCalculator()

    def test_calculate_line_coverage(self, calculator):
        """Test calculating line coverage."""
        code = """def foo():
    x = 1
    return x
"""
        verifications = [
            {"line": 2, "category": "null_safety", "proof_status": "proven"},
        ]

        coverage = calculator.calculate_line_coverage(code, verifications)

        assert len(coverage) > 0
        proven_lines = [c for c in coverage if c.proof_status == ProofStatus.PROVEN]
        assert len(proven_lines) >= 1

    def test_calculate_function_coverage(self, calculator):
        """Test calculating function coverage."""
        line_coverages = [
            LineCoverage(line_number=1, proof_status=ProofStatus.PROVEN),
            LineCoverage(line_number=2, proof_status=ProofStatus.PROVEN),
            LineCoverage(line_number=3, proof_status=ProofStatus.PROVEN),
            LineCoverage(line_number=5, proof_status=ProofStatus.NOT_ATTEMPTED),
            LineCoverage(line_number=6, proof_status=ProofStatus.NOT_ATTEMPTED),
            LineCoverage(line_number=7, proof_status=ProofStatus.NOT_ATTEMPTED),
        ]

        func_cov = calculator.calculate_function_coverage("foo", 1, 3, line_coverages)

        assert isinstance(func_cov, FunctionCoverage)
        assert func_cov.function_name == "foo"
        assert func_cov.proven_lines == 3

    def test_calculate_file_coverage(self, calculator):
        """Test calculating file coverage."""
        file_content = "line1\nline2\nline3\nline4\nline5"
        verifications = [
            {"line": 1, "proof_status": "proven"},
            {"line": 2, "proof_status": "proven"},
            {"line": 3, "proof_status": "proven"},
        ]

        coverage = calculator.calculate_file_coverage(
            "test.py",
            file_content,
            verifications,
        )

        assert coverage.file_path == "test.py"
        assert coverage.total_lines == 5
        assert coverage.proven_lines == 3

    def test_empty_verification_results(self, calculator):
        """Test with no verification results."""
        code = "x = 1\ny = 2\n"
        coverage = calculator.calculate_line_coverage(code, [])

        assert len(coverage) > 0
        for lc in coverage:
            assert lc.proof_status == ProofStatus.NOT_ATTEMPTED


class TestProofCoverageDashboard:
    """Tests for ProofCoverageDashboard."""

    @pytest.fixture
    def dashboard(self):
        return ProofCoverageDashboard()

    def test_generate_dashboard_data(self, dashboard):
        """Test generating full dashboard data."""
        files = [
            {
                "path": "src/test.py",
                "content": "x = 1\ny = 2\n",
                "language": "python",
                "functions": [],
            }
        ]
        verification_results = {
            "src/test.py": [
                {"line": 1, "proof_status": "proven", "category": "null_safety"},
            ]
        }

        data = dashboard.generate_dashboard_data("test/repo", files, verification_results)

        assert isinstance(data, DashboardData)
        assert data.repository == "test/repo"
        assert data.current_coverage is not None

    def test_record_coverage(self, dashboard):
        """Test recording coverage for trend tracking."""
        coverage = RepositoryCoverage(repository="test/repo")
        dashboard.record_coverage("test/repo", coverage)

        # Should not raise
        assert True

    def test_to_json(self, dashboard):
        """Test converting dashboard data to JSON."""
        repo_cov = RepositoryCoverage(repository="test/repo")
        data = DashboardData(
            repository="test/repo",
            current_coverage=repo_cov,
        )

        result = dashboard.to_json(data)

        assert isinstance(result, dict)
        assert result["repository"] == "test/repo"
        assert "summary" in result
        assert "coverage" in result


class TestGlobalDashboard:
    """Tests for global dashboard functions."""

    def teardown_method(self):
        reset_proof_coverage_dashboard()

    def test_get_dashboard_singleton(self):
        """Test singleton pattern."""
        dashboard1 = get_proof_coverage_dashboard()
        dashboard2 = get_proof_coverage_dashboard()
        assert dashboard1 is dashboard2

    def test_reset_dashboard(self):
        """Test dashboard reset."""
        dashboard1 = get_proof_coverage_dashboard()
        reset_proof_coverage_dashboard()
        dashboard2 = get_proof_coverage_dashboard()
        assert dashboard1 is not dashboard2
