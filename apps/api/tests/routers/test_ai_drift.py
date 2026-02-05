"""Tests for AI Drift Detection API endpoints."""

import pytest
from fastapi.testclient import TestClient

from codeverify_api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRecordSnapshot:
    """Tests for /api/v1/drift/snapshot endpoint."""

    def test_record_basic_snapshot(self, client):
        """Should record a basic AI code snapshot."""
        response = client.post(
            "/api/v1/drift/snapshot",
            json={
                "file_path": "src/utils.py",
                "code": "def add(a, b):\n    return a + b",
                "trust_score": 75.0,
                "ai_probability": 0.8,
                "findings": [],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "snapshot_id" in data
        assert "timestamp" in data
        assert "immediate_alerts" in data

    def test_record_snapshot_with_metadata(self, client):
        """Should record snapshot with all metadata."""
        response = client.post(
            "/api/v1/drift/snapshot",
            json={
                "file_path": "src/api.py",
                "code": "async def fetch_data(): pass",
                "trust_score": 60.0,
                "ai_probability": 0.9,
                "findings": [{"severity": "high", "title": "Test finding"}],
                "detected_model": "gpt-4",
                "complexity_score": 45.0,
                "security_score": 70.0,
                "test_coverage": 80.0,
                "documentation_score": 50.0,
                "was_reviewed": True,
                "review_depth": 0.8,
                "time_to_accept": 120.0,
                "author": "developer@example.com",
                "commit_hash": "abc123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "snapshot_id" in data

    def test_record_low_trust_generates_alert(self, client):
        """Should generate alert for low trust score."""
        response = client.post(
            "/api/v1/drift/snapshot",
            json={
                "file_path": "src/risky.py",
                "code": "eval(user_input)",
                "trust_score": 30.0,
                "ai_probability": 0.95,
                "findings": [{"severity": "critical", "title": "Unsafe eval"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should have alerts for low trust and critical findings
        assert len(data["immediate_alerts"]) > 0

    def test_record_snapshot_validates_input(self, client):
        """Should validate input parameters."""
        # Missing required fields
        response = client.post(
            "/api/v1/drift/snapshot",
            json={
                "file_path": "test.py",
            },
        )
        assert response.status_code == 422

        # Invalid trust score range
        response = client.post(
            "/api/v1/drift/snapshot",
            json={
                "file_path": "test.py",
                "code": "x = 1",
                "trust_score": 150.0,  # Invalid
                "ai_probability": 0.5,
                "findings": [],
            },
        )
        assert response.status_code == 422


class TestDriftReport:
    """Tests for /api/v1/drift/report endpoint."""

    def test_generate_report(self, client):
        """Should generate a drift report."""
        # First record some snapshots
        for i in range(3):
            client.post(
                "/api/v1/drift/snapshot",
                json={
                    "file_path": f"src/file_{i}.py",
                    "code": f"def func_{i}(): pass",
                    "trust_score": 70.0 - i * 5,
                    "ai_probability": 0.7 + i * 0.1,
                    "findings": [],
                },
            )

        # Generate report
        response = client.post(
            "/api/v1/drift/report",
            json={"days": 30},
        )
        assert response.status_code == 200
        data = response.json()

        assert "report_id" in data
        assert "health_score" in data
        assert "health_trend" in data
        assert "alerts" in data
        assert "current_metrics" in data
        assert "recommendations" in data

    def test_report_includes_metrics(self, client):
        """Should include aggregate metrics."""
        response = client.post(
            "/api/v1/drift/report",
            json={"days": 7},
        )
        assert response.status_code == 200
        data = response.json()

        metrics = data["current_metrics"]
        assert "avg_trust_score" in metrics
        assert "avg_ai_probability" in metrics
        assert "review_rate" in metrics
        assert "total_ai_snippets" in metrics

    def test_report_validates_days(self, client):
        """Should validate days parameter."""
        response = client.post(
            "/api/v1/drift/report",
            json={"days": 0},  # Invalid
        )
        assert response.status_code == 422

        response = client.post(
            "/api/v1/drift/report",
            json={"days": 400},  # Invalid (max 365)
        )
        assert response.status_code == 422


class TestBaseline:
    """Tests for /api/v1/drift/baseline endpoint."""

    def test_establish_baseline(self, client):
        """Should establish baseline metrics."""
        # Record some data first
        for i in range(5):
            client.post(
                "/api/v1/drift/snapshot",
                json={
                    "file_path": f"src/baseline_{i}.py",
                    "code": f"x = {i}",
                    "trust_score": 75.0,
                    "ai_probability": 0.6,
                    "findings": [],
                },
            )

        response = client.post(
            "/api/v1/drift/baseline",
            json={"days": 30},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["baseline_established"] is True
        assert "metrics" in data
        assert data["period_days"] == 30


class TestAlerts:
    """Tests for /api/v1/drift/alerts endpoint."""

    def test_get_all_alerts(self, client):
        """Should return all alerts."""
        response = client.get("/api/v1/drift/alerts")
        assert response.status_code == 200
        data = response.json()

        assert "alerts" in data
        assert "count" in data
        assert isinstance(data["alerts"], list)

    def test_filter_alerts_by_severity(self, client):
        """Should filter alerts by severity."""
        response = client.get("/api/v1/drift/alerts?severity=critical")
        assert response.status_code == 200
        data = response.json()
        assert data["filters"]["severity"] == "critical"

    def test_filter_alerts_by_category(self, client):
        """Should filter alerts by category."""
        response = client.get("/api/v1/drift/alerts?category=quality_degradation")
        assert response.status_code == 200
        data = response.json()
        assert data["filters"]["category"] == "quality_degradation"

    def test_invalid_severity_filter(self, client):
        """Should reject invalid severity filter."""
        response = client.get("/api/v1/drift/alerts?severity=invalid")
        assert response.status_code == 400

    def test_invalid_category_filter(self, client):
        """Should reject invalid category filter."""
        response = client.get("/api/v1/drift/alerts?category=invalid")
        assert response.status_code == 400


class TestStats:
    """Tests for /api/v1/drift/stats endpoint."""

    def test_get_stats(self, client):
        """Should return detector statistics."""
        response = client.get("/api/v1/drift/stats")
        assert response.status_code == 200
        data = response.json()

        assert "available" in data
        if data["available"]:
            assert "total_snapshots" in data
            assert "active_alerts" in data
            assert "baseline_established" in data
            assert "thresholds" in data


class TestThresholds:
    """Tests for /api/v1/drift/thresholds endpoint."""

    def test_update_thresholds(self, client):
        """Should update detection thresholds."""
        response = client.put(
            "/api/v1/drift/thresholds",
            json={
                "trust_score_min": 70.0,
                "review_rate_min": 90.0,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["updated"] is True
        assert data["thresholds"]["trust_score_min"] == 70.0
        assert data["thresholds"]["review_rate_min"] == 90.0

    def test_partial_threshold_update(self, client):
        """Should allow partial threshold updates."""
        response = client.put(
            "/api/v1/drift/thresholds",
            json={
                "security_score_min": 80.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["thresholds"]["security_score_min"] == 80.0


class TestHealthScore:
    """Tests for /api/v1/drift/health endpoint."""

    def test_get_health_score(self, client):
        """Should return current health score."""
        response = client.get("/api/v1/drift/health")
        assert response.status_code == 200
        data = response.json()

        assert "health_score" in data
        assert "health_trend" in data
        assert "alert_count" in data
        assert "critical_alerts" in data
        assert "period_days" in data

    def test_health_score_with_custom_period(self, client):
        """Should accept custom period."""
        response = client.get("/api/v1/drift/health?days=14")
        assert response.status_code == 200
        data = response.json()
        assert data["period_days"] == 14


class TestClearData:
    """Tests for /api/v1/drift/data endpoint."""

    def test_clear_data(self, client):
        """Should clear all drift data."""
        # Record some data
        client.post(
            "/api/v1/drift/snapshot",
            json={
                "file_path": "test.py",
                "code": "x = 1",
                "trust_score": 80.0,
                "ai_probability": 0.5,
                "findings": [],
            },
        )

        # Clear
        response = client.delete("/api/v1/drift/data")
        assert response.status_code == 200
        data = response.json()
        assert data["cleared"] is True

        # Verify data is cleared
        stats = client.get("/api/v1/drift/stats").json()
        if stats["available"]:
            assert stats["total_snapshots"] == 0


class TestDriftDetection:
    """Integration tests for drift detection."""

    def test_detects_trust_score_decline(self, client):
        """Should detect declining trust scores."""
        # Clear previous data
        client.delete("/api/v1/drift/data")

        # Record baseline high trust scores
        for i in range(5):
            client.post(
                "/api/v1/drift/snapshot",
                json={
                    "file_path": f"src/good_{i}.py",
                    "code": f"def safe_{i}(): pass",
                    "trust_score": 85.0,
                    "ai_probability": 0.7,
                    "findings": [],
                },
            )

        # Establish baseline
        client.post("/api/v1/drift/baseline", json={"days": 30})

        # Record lower trust scores
        for i in range(5):
            client.post(
                "/api/v1/drift/snapshot",
                json={
                    "file_path": f"src/declining_{i}.py",
                    "code": f"def risky_{i}(): pass",
                    "trust_score": 55.0,
                    "ai_probability": 0.9,
                    "findings": [{"severity": "medium", "title": "Issue"}],
                },
            )

        # Generate report - should detect drift
        response = client.post("/api/v1/drift/report", json={"days": 30})
        data = response.json()

        # Should have alerts about declining quality
        assert data["health_score"] < 100
        # Trust trend should be negative or alerts present
        has_quality_concern = (
            data["current_metrics"]["trust_trend"] < 0 or
            len(data["alerts"]) > 0 or
            data["health_trend"] == "declining"
        )
        assert has_quality_concern

    def test_detects_unreviewed_code(self, client):
        """Should detect unreviewed AI code."""
        client.delete("/api/v1/drift/data")

        # Record unreviewed high-probability AI code
        response = client.post(
            "/api/v1/drift/snapshot",
            json={
                "file_path": "src/unreviewed.py",
                "code": "# AI generated code",
                "trust_score": 65.0,
                "ai_probability": 0.95,
                "findings": [],
                "was_reviewed": False,
            },
        )

        data = response.json()
        # Should generate review alert
        alerts = data["immediate_alerts"]
        review_alerts = [
            a for a in alerts
            if "review" in a.get("category", "").lower() or
               "review" in a.get("message", "").lower()
        ]
        assert len(review_alerts) > 0 or len(alerts) > 0
