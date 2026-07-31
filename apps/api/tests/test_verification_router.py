"""Integration tests for verification and analyses API endpoints."""

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from codeverify_api.routers.verification_api import (
    _api_subscriptions,
    _api_usage,
    _rate_limit_windows,
)
from codeverify_api.routers.verification_api import (
    router as verification_router,
)


@pytest.fixture(autouse=True)
def _clear_state():
    """Clear in-memory state between tests."""
    _api_subscriptions.clear()
    _rate_limit_windows.clear()
    _api_usage.clear()
    yield
    _api_subscriptions.clear()
    _rate_limit_windows.clear()
    _api_usage.clear()


def _create_verification_app():
    app = FastAPI()
    app.include_router(verification_router)
    return app


@pytest.fixture
def client():
    return TestClient(_create_verification_app())


API_KEY = "test-api-key-12345"
HEADERS = {"X-API-Key": API_KEY}


class TestVerifyEndpoint:
    """Test POST /api/v1/verification/verify."""

    def test_verify_simple_code(self, client):
        response = client.post(
            "/api/v1/verification/verify",
            json={"code": "x = 1 + 2", "language": "python"},
            headers=HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "request_id" in data
        assert "trust_score" in data
        assert "findings" in data
        assert "remaining_quota" in data

    def test_verify_returns_findings_for_unsafe_code(self, client):
        response = client.post(
            "/api/v1/verification/verify",
            json={
                "code": "result = x.value + y.attribute",
                "language": "python",
            },
            headers=HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["findings"]) > 0
        assert any(f["category"] == "null_safety" for f in data["findings"])

    def test_verify_requires_api_key(self, client):
        response = client.post(
            "/api/v1/verification/verify",
            json={"code": "x = 1", "language": "python"},
        )
        assert response.status_code == 422

    def test_verify_requires_code_field(self, client):
        response = client.post(
            "/api/v1/verification/verify",
            json={"language": "python"},
            headers=HEADERS,
        )
        assert response.status_code == 422

    def test_verify_requires_language_field(self, client):
        response = client.post(
            "/api/v1/verification/verify",
            json={"code": "x = 1"},
            headers=HEADERS,
        )
        assert response.status_code == 422

    def test_verify_rejects_oversized_code(self, client):
        large_code = "x = 1\n" * 100000  # Way over 50KB free tier limit
        response = client.post(
            "/api/v1/verification/verify",
            json={"code": large_code, "language": "python"},
            headers=HEADERS,
        )
        assert response.status_code == 400
        assert "exceeds limit" in response.json()["detail"]

    def test_verify_clean_code_is_verified(self, client):
        response = client.post(
            "/api/v1/verification/verify",
            json={"code": "def add(a, b): return a + b", "language": "python"},
            headers=HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verified"] is True
        assert data["trust_score"] == 1.0

    def test_verify_tracks_remaining_quota(self, client):
        response = client.post(
            "/api/v1/verification/verify",
            json={"code": "x = 1", "language": "python"},
            headers=HEADERS,
        )
        data = response.json()
        quota = data["remaining_quota"]
        assert "requests_per_minute" in quota
        assert "requests_per_day" in quota
        assert "requests_per_month" in quota

    def test_verify_processes_division_pattern(self, client):
        response = client.post(
            "/api/v1/verification/verify",
            json={
                "code": "result = a / b",
                "language": "python",
            },
            headers=HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        categories = [f["category"] for f in data["findings"]]
        assert "division" in categories

    def test_verify_processes_array_access_pattern(self, client):
        response = client.post(
            "/api/v1/verification/verify",
            json={
                "code": "val = arr[i]",
                "language": "python",
            },
            headers=HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        categories = [f["category"] for f in data["findings"]]
        assert "bounds" in categories


class TestVerifyBatchEndpoint:
    """Test POST /api/v1/verification/verify/batch."""

    def test_batch_verify_single_file(self, client):
        response = client.post(
            "/api/v1/verification/verify/batch",
            json={
                "files": [{"path": "main.py", "content": "x = 1"}],
                "language": "python",
            },
            headers=HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1

    def test_batch_rejects_too_many_files(self, client):
        # Free tier limit is 1 file per request
        files = [{"path": f"file{i}.py", "content": "x = 1"} for i in range(10)]
        response = client.post(
            "/api/v1/verification/verify/batch",
            json={"files": files, "language": "python"},
            headers=HEADERS,
        )
        assert response.status_code == 400
        assert "Too many files" in response.json()["detail"]


class TestAnalysesEndpoint:
    """Test GET /analyses — list analyses."""

    def _create_analyses_app(self):
        from codeverify_api.routers.analyses import router as analyses_router

        app = FastAPI()
        app.include_router(analyses_router, prefix="/api/v1/analyses")
        return app

    def test_list_analyses_requires_db(self):
        """Analyses endpoint returns data with mocked DB."""
        from datetime import datetime

        app = self._create_analyses_app()

        from codeverify_api.db.database import get_db

        mock_analysis = MagicMock()
        mock_analysis.id = "analysis-1"
        mock_analysis.repo_id = "repo-1"
        mock_analysis.pr_number = 1
        mock_analysis.pr_title = "Test PR"
        mock_analysis.head_sha = "abc123"
        mock_analysis.base_sha = "def456"
        mock_analysis.status = "completed"
        mock_analysis.started_at = datetime(2024, 1, 1, tzinfo=UTC)
        mock_analysis.completed_at = datetime(2024, 1, 1, 0, 1, tzinfo=UTC)
        mock_analysis.created_at = datetime(2024, 1, 1, tzinfo=UTC)

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_analysis]

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_result, mock_count_result])

        async def mock_get_db():
            yield mock_session

        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        response = client.get("/api/v1/analyses")
        assert response.status_code == 200
        data = response.json()
        assert "analyses" in data
        assert "total" in data

    def test_list_analyses_empty(self):
        """Returns empty list when no analyses exist."""
        app = self._create_analyses_app()

        from codeverify_api.db.database import get_db

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_result, mock_count_result])

        async def mock_get_db():
            yield mock_session

        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        response = client.get("/api/v1/analyses")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["analyses"] == []

    def test_list_analyses_pagination(self):
        """Validates query param support for limit/offset."""
        app = self._create_analyses_app()

        from codeverify_api.db.database import get_db

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_result, mock_count_result])

        async def mock_get_db():
            yield mock_session

        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        response = client.get("/api/v1/analyses", params={"limit": 10, "offset": 5})
        assert response.status_code == 200
