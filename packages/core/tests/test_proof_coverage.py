"""Tests for proof coverage dashboard."""

import pytest
from datetime import datetime, timedelta
from codeverify_core.proof_coverage import (
    CoverageTrend,
    DashboardData,
    FileCoverage,
    FunctionCoverage,
    LineCoverage,
    ProofCoverageCalculator,
    ProofCoverageDashboard,
    ProofCoverageStatus,
    RepositoryCoverage,
    VerificationCategory,
    get_proof_coverage_dashboard,
    reset_proof_coverage_dashboard,
)


class TestProofCoverageStatus:
    """Tests for ProofCoverageStatus enum."""

    def test_all_statuses(self):
        """Test all coverage statuses exist."""
        assert ProofCoverageStatus.VERIFIED.value == "verified"
        assert ProofCoverageStatus.PARTIALLY_VERIFIED.value == "partially_verified"
        assert ProofCoverageStatus.UNVERIFIED.value == "unverified"
        assert ProofCoverageStatus.EXCLUDED.value == "excluded"


class TestVerificationCategory:
    """Tests for VerificationCategory enum."""

    def test_all_categories(self):
        """Test all verification categories exist."""
        assert VerificationCategory.NULL_SAFETY.value == "null_safety"
        assert VerificationCategory.BOUNDS_CHECK.value == "bounds_check"
        assert VerificationCategory.DIVISION_SAFETY.value == "division_safety"
        assert VerificationCategory.TYPE_SAFETY.value == "type_safety"
        assert VerificationCategory.MEMORY_SAFETY.value == "memory_safety"


class TestLineCoverage:
    """Tests for LineCoverage dataclass."""

    def test_creation(self):
        """Test line coverage creation."""
        coverage = LineCoverage(
            line_number=42,
            status=ProofCoverageStatus.VERIFIED,
            categories=[VerificationCategory.NULL_SAFETY],
            verification_time=datetime.utcnow(),
        )
        
        assert coverage.line_number == 42
        assert coverage.status == ProofCoverageStatus.VERIFIED
        assert VerificationCategory.NULL_SAFETY in coverage.categories

    def test_default_values(self):
        """Test default line coverage values."""
        coverage = LineCoverage(
            line_number=1,
            status=ProofCoverageStatus.UNVERIFIED,
        )
        
        assert coverage.categories == []
        assert coverage.verification_time is None


class TestFunctionCoverage:
    """Tests for FunctionCoverage dataclass."""

    def test_creation(self):
        """Test function coverage creation."""
        coverage = FunctionCoverage(
            name="process_payment",
            start_line=10,
            end_line=50,
            status=ProofCoverageStatus.VERIFIED,
            line_coverage=[],
            coverage_percentage=95.0,
        )
        
        assert coverage.name == "process_payment"
        assert coverage.coverage_percentage == 95.0

    def test_default_percentage(self):
        """Test default coverage percentage."""
        coverage = FunctionCoverage(
            name="test",
            start_line=1,
            end_line=10,
            status=ProofCoverageStatus.UNVERIFIED,
            line_coverage=[],
        )
        
        assert coverage.coverage_percentage == 0.0


class TestFileCoverage:
    """Tests for FileCoverage dataclass."""

    def test_creation(self):
        """Test file coverage creation."""
        coverage = FileCoverage(
            file_path="src/api/handlers.py",
            total_lines=100,
            verified_lines=85,
            functions=[],
            status=ProofCoverageStatus.PARTIALLY_VERIFIED,
        )
        
        assert coverage.file_path == "src/api/handlers.py"
        assert coverage.total_lines == 100
        assert coverage.verified_lines == 85

    def test_coverage_percentage(self):
        """Test coverage percentage calculation."""
        coverage = FileCoverage(
            file_path="test.py",
            total_lines=100,
            verified_lines=75,
            functions=[],
            status=ProofCoverageStatus.PARTIALLY_VERIFIED,
        )
        
        # Calculate percentage
        percentage = (coverage.verified_lines / coverage.total_lines) * 100
        assert percentage == 75.0


class TestRepositoryCoverage:
    """Tests for RepositoryCoverage dataclass."""

    def test_creation(self):
        """Test repository coverage creation."""
        coverage = RepositoryCoverage(
            repository="myorg/myrepo",
            total_files=50,
            verified_files=40,
            total_lines=5000,
            verified_lines=4000,
            files=[],
            overall_percentage=80.0,
        )
        
        assert coverage.repository == "myorg/myrepo"
        assert coverage.overall_percentage == 80.0

    def test_default_values(self):
        """Test default repository coverage values."""
        coverage = RepositoryCoverage(
            repository="test/repo",
            total_files=10,
            verified_files=5,
            total_lines=1000,
            verified_lines=500,
            files=[],
        )
        
        assert coverage.overall_percentage == 0.0
        assert coverage.last_updated is None


class TestCoverageTrend:
    """Tests for CoverageTrend dataclass."""

    def test_creation(self):
        """Test trend creation."""
        trend = CoverageTrend(
            date=datetime.utcnow(),
            coverage_percentage=75.5,
            verified_lines=1500,
            total_lines=2000,
        )
        
        assert trend.coverage_percentage == 75.5
        assert trend.verified_lines == 1500


class TestDashboardData:
    """Tests for DashboardData dataclass."""

    def test_creation(self):
        """Test dashboard data creation."""
        data = DashboardData(
            repository_coverage=RepositoryCoverage(
                repository="test/repo",
                total_files=10,
                verified_files=8,
                total_lines=1000,
                verified_lines=800,
                files=[],
                overall_percentage=80.0,
            ),
            trends=[],
            heatmap={},
        )
        
        assert data.repository_coverage.overall_percentage == 80.0
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
            {"line": 2, "category": "null_safety", "status": "verified"},
        ]
        
        coverage = calculator.calculate_line_coverage(code, verifications)
        
        assert len(coverage) > 0
        verified_lines = [c for c in coverage if c.status == ProofCoverageStatus.VERIFIED]
        assert len(verified_lines) >= 1

    def test_calculate_function_coverage(self, calculator):
        """Test calculating function coverage."""
        code = """def foo():
    x = 1
    return x

def bar():
    y = 2
    return y
"""
        line_coverage = [
            LineCoverage(line_number=1, status=ProofCoverageStatus.VERIFIED),
            LineCoverage(line_number=2, status=ProofCoverageStatus.VERIFIED),
            LineCoverage(line_number=3, status=ProofCoverageStatus.VERIFIED),
            LineCoverage(line_number=5, status=ProofCoverageStatus.UNVERIFIED),
            LineCoverage(line_number=6, status=ProofCoverageStatus.UNVERIFIED),
            LineCoverage(line_number=7, status=ProofCoverageStatus.UNVERIFIED),
        ]
        
        functions = calculator.calculate_function_coverage(code, line_coverage, "python")
        
        # Should find at least the functions
        assert isinstance(functions, list)

    def test_calculate_file_coverage(self, calculator):
        """Test calculating file coverage."""
        file_content = "line1\nline2\nline3\nline4\nline5"
        verifications = [
            {"line": 1, "status": "verified"},
            {"line": 2, "status": "verified"},
            {"line": 3, "status": "verified"},
        ]
        
        coverage = calculator.calculate_file_coverage(
            "test.py",
            file_content,
            verifications,
        )
        
        assert coverage.file_path == "test.py"
        assert coverage.total_lines == 5
        assert coverage.verified_lines == 3

    def test_calculate_repository_coverage(self, calculator):
        """Test calculating repository coverage."""
        file_coverages = [
            FileCoverage(
                file_path="file1.py",
                total_lines=100,
                verified_lines=80,
                functions=[],
                status=ProofCoverageStatus.PARTIALLY_VERIFIED,
            ),
            FileCoverage(
                file_path="file2.py",
                total_lines=50,
                verified_lines=50,
                functions=[],
                status=ProofCoverageStatus.VERIFIED,
            ),
        ]
        
        coverage = calculator.calculate_repository_coverage(
            "test/repo",
            file_coverages,
        )
        
        assert coverage.repository == "test/repo"
        assert coverage.total_files == 2
        assert coverage.verified_files >= 1
        assert coverage.total_lines == 150
        assert coverage.verified_lines == 130

    def test_get_coverage_status(self, calculator):
        """Test getting coverage status from percentage."""
        assert calculator.get_coverage_status(100) == ProofCoverageStatus.VERIFIED
        assert calculator.get_coverage_status(80) == ProofCoverageStatus.PARTIALLY_VERIFIED
        assert calculator.get_coverage_status(0) == ProofCoverageStatus.UNVERIFIED


class TestProofCoverageDashboard:
    """Tests for ProofCoverageDashboard."""

    @pytest.fixture
    def dashboard(self):
        return ProofCoverageDashboard()

    def test_add_verification_result(self, dashboard):
        """Test adding verification results."""
        dashboard.add_verification_result(
            repository="test/repo",
            file_path="src/test.py",
            line_number=10,
            category=VerificationCategory.NULL_SAFETY,
            status=ProofCoverageStatus.VERIFIED,
        )
        
        # Should be tracked
        stats = dashboard.get_repository_stats("test/repo")
        assert stats is not None

    def test_get_file_coverage(self, dashboard):
        """Test getting file coverage."""
        dashboard.add_verification_result(
            repository="test/repo",
            file_path="src/test.py",
            line_number=10,
            category=VerificationCategory.NULL_SAFETY,
            status=ProofCoverageStatus.VERIFIED,
        )
        
        coverage = dashboard.get_file_coverage("test/repo", "src/test.py")
        
        assert coverage is not None
        assert "src/test.py" in coverage.file_path

    def test_get_repository_stats(self, dashboard):
        """Test getting repository statistics."""
        # Add some verifications
        dashboard.add_verification_result(
            repository="test/repo",
            file_path="file1.py",
            line_number=1,
            category=VerificationCategory.NULL_SAFETY,
            status=ProofCoverageStatus.VERIFIED,
        )
        dashboard.add_verification_result(
            repository="test/repo",
            file_path="file2.py",
            line_number=1,
            category=VerificationCategory.BOUNDS_CHECK,
            status=ProofCoverageStatus.UNVERIFIED,
        )
        
        stats = dashboard.get_repository_stats("test/repo")
        
        assert "total_verifications" in stats
        assert "by_category" in stats
        assert "by_status" in stats

    def test_get_trends(self, dashboard):
        """Test getting coverage trends."""
        # Add verification at different times
        dashboard.add_verification_result(
            repository="test/repo",
            file_path="test.py",
            line_number=1,
            category=VerificationCategory.NULL_SAFETY,
            status=ProofCoverageStatus.VERIFIED,
        )
        
        trends = dashboard.get_trends("test/repo", days=30)
        
        assert isinstance(trends, list)

    def test_generate_heatmap(self, dashboard):
        """Test generating coverage heatmap."""
        dashboard.add_verification_result(
            repository="test/repo",
            file_path="src/api/handlers.py",
            line_number=10,
            category=VerificationCategory.NULL_SAFETY,
            status=ProofCoverageStatus.VERIFIED,
        )
        dashboard.add_verification_result(
            repository="test/repo",
            file_path="src/api/handlers.py",
            line_number=20,
            category=VerificationCategory.NULL_SAFETY,
            status=ProofCoverageStatus.UNVERIFIED,
        )
        
        heatmap = dashboard.generate_heatmap("test/repo")
        
        assert isinstance(heatmap, dict)
        # Should have structure by file/directory

    def test_get_dashboard_data(self, dashboard):
        """Test getting full dashboard data."""
        dashboard.add_verification_result(
            repository="test/repo",
            file_path="test.py",
            line_number=1,
            category=VerificationCategory.NULL_SAFETY,
            status=ProofCoverageStatus.VERIFIED,
        )
        
        data = dashboard.get_dashboard_data("test/repo")
        
        assert isinstance(data, DashboardData)
        assert data.repository_coverage is not None

    def test_export_report(self, dashboard):
        """Test exporting coverage report."""
        dashboard.add_verification_result(
            repository="test/repo",
            file_path="test.py",
            line_number=1,
            category=VerificationCategory.NULL_SAFETY,
            status=ProofCoverageStatus.VERIFIED,
        )
        
        report = dashboard.export_report("test/repo", format="json")
        
        assert isinstance(report, dict) or isinstance(report, str)


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
