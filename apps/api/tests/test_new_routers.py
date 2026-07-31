"""Tests for API routers added after the initial API surface."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from codeverify_api.auth.dependencies import get_current_user
from codeverify_api.auth.jwt import TokenData
from codeverify_api.middleware.rate_limit import limiter, setup_rate_limiting
from codeverify_api.routers import notifications, public_api, rules, scanning, trust_score


class TestTrustScoreRouter:
    """Tests for the current Trust Score API endpoints."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a test client with a deterministic trust-score agent."""
        from codeverify_agents import trust_score as trust_score_agent

        analyze = AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                data={
                    "score": 87.5,
                    "confidence": 0.92,
                    "risk_level": "low",
                    "factors": {
                        "complexity_score": 0.8,
                        "pattern_confidence": 0.9,
                        "historical_accuracy": 0.7,
                        "verification_coverage": 0.6,
                        "code_quality_signals": 0.85,
                        "ai_detection_confidence": 0.1,
                    },
                    "recommendations": ["Keep the tests current."],
                    "is_ai_generated": False,
                },
            )
        )
        agent = SimpleNamespace(analyze=analyze)
        monkeypatch.setattr(trust_score_agent, "TrustScoreAgent", lambda: agent)
        monkeypatch.setattr(trust_score_agent, "calculate_code_hash", lambda _code: "test-hash")

        app = FastAPI()
        app.include_router(trust_score.router, prefix="/api/v1/trust-score")
        with TestClient(app) as test_client:
            yield test_client, analyze

    def test_analyze_code_endpoint(self, client):
        """POST /trust-score returns the current response model."""
        test_client, analyze = client

        response = test_client.post(
            "/api/v1/trust-score",
            json={"code": "def test(): pass", "language": "python"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["score"] == 87.5
        assert data["risk_level"] == "low"
        assert data["code_hash"] == "test-hash"
        assert data["factors"]["code_quality_signals"] == 0.85
        analyze.assert_awaited_once()

    def test_analyze_code_with_context(self, client):
        """The request model maps file and author context to the agent."""
        test_client, analyze = client

        response = test_client.post(
            "/api/v1/trust-score",
            json={
                "code": "def test(): pass",
                "language": "python",
                "file_path": "tests/test_example.py",
                "author": "test-user",
            },
        )

        assert response.status_code == 200
        analyze.assert_awaited_once_with(
            "def test(): pass",
            {
                "file_path": "tests/test_example.py",
                "language": "python",
                "author": "test-user",
            },
        )

    def test_analyze_empty_code(self, client):
        """Empty code remains valid under the current request schema."""
        test_client, analyze = client

        response = test_client.post(
            "/api/v1/trust-score",
            json={"code": "", "language": "python"},
        )

        assert response.status_code == 200
        analyze.assert_awaited_once_with(
            "",
            {"file_path": "unknown", "language": "python", "author": None},
        )

    def test_batch_analysis(self, client):
        """POST /batch aggregates current trust-score responses."""
        test_client, analyze = client

        response = test_client.post(
            "/api/v1/trust-score/batch",
            json={
                "files": [
                    {"code": "x = 1", "file_path": "a.py"},
                    {"code": "y = 2", "file_path": "b.py"},
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert set(data["scores"]) == {"a.py", "b.py"}
        assert data["overall_score"] == 87.5
        assert data["overall_risk_level"] == "low"
        assert data["high_risk_files"] == []
        assert analyze.await_count == 2


class TestRulesRouter:
    """Tests for the current custom-rules API."""

    @pytest.fixture
    def client(self):
        """Create an isolated rules client."""
        rules._rules.clear()
        app = FastAPI()
        app.include_router(rules.router, prefix="/api/v1/rules")
        with TestClient(app) as test_client:
            yield test_client
        rules._rules.clear()

    @staticmethod
    def rule_payload() -> dict:
        """Return a request matching CreateRuleRequest."""
        return {
            "name": "No print statements",
            "description": "Use logging instead.",
            "rule_type": "pattern",
            "severity": "medium",
            "scope": "line",
            "conditions": [
                {
                    "field": "code",
                    "operator": "matches",
                    "value": r"print\(",
                }
            ],
            "actions": [{"message": "Replace print with logging."}],
            "languages": ["python"],
            "tags": ["logging"],
        }

    def test_list_rules(self, client):
        """GET /rules returns the stored rules."""
        response = client.get("/api/v1/rules")

        assert response.status_code == 200
        assert response.json() == []

    def test_create_rule(self, client):
        """POST /rules creates a rule using the current request model."""
        response = client.post("/api/v1/rules", json=self.rule_payload())

        assert response.status_code == 201
        data = response.json()
        UUID(data["id"])
        assert data["name"] == "No print statements"
        assert data["rule_type"] == "pattern"
        assert data["severity"] == "medium"
        assert data["conditions"][0]["operator"] == "matches"

    def test_get_rule(self, client):
        """GET /rules/{id} returns a created rule."""
        created = client.post("/api/v1/rules", json=self.rule_payload()).json()

        response = client.get(f"/api/v1/rules/{created['id']}")

        assert response.status_code == 200
        assert response.json()["id"] == created["id"]

    def test_delete_rule(self, client):
        """DELETE /rules/{id} removes a created rule."""
        created = client.post("/api/v1/rules", json=self.rule_payload()).json()

        response = client.delete(f"/api/v1/rules/{created['id']}")

        assert response.status_code == 200
        assert response.json() == {"deleted": True, "rule_id": created["id"]}
        assert client.get(f"/api/v1/rules/{created['id']}").status_code == 404

    def test_test_rule(self, client):
        """POST /rules/test evaluates a current rule definition."""
        response = client.post(
            "/api/v1/rules/test",
            json={
                "rule": self.rule_payload(),
                "code": "print('hello')",
                "file_path": "example.py",
                "language": "python",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["matches"] is True
        assert len(data["violations"]) == 1
        assert data["execution_time_ms"] >= 0

    def test_get_rule_templates(self, client):
        """GET /rules/templates returns the built-in templates."""
        response = client.get("/api/v1/rules/templates")

        assert response.status_code == 200
        assert "no-print" in response.json()["templates"]


class TestScanningRouter:
    """Tests for the current in-memory scanning API."""

    @pytest.fixture
    def client(self):
        """Create an isolated scanning client."""
        from codeverify_core import scanning as scanning_core

        scanning_core._scan_results.clear()
        scanning_core._scheduled_scans.clear()
        app = FastAPI()
        app.include_router(scanning.router, prefix="/api/v1/scans")
        with TestClient(app) as test_client:
            yield test_client
        scanning_core._scan_results.clear()
        scanning_core._scheduled_scans.clear()

    def test_trigger_scan(self, client):
        """POST /scans/trigger queues a scan."""
        response = client.post(
            "/api/v1/scans/trigger",
            json={"repo_full_name": "owner/repo", "branch": "main"},
        )

        assert response.status_code == 200
        data = response.json()
        UUID(data["scan_id"])
        assert data["repo_full_name"] == "owner/repo"
        assert data["status"] == "queued"
        assert data["files_scanned"] == 0

    def test_get_scan_status(self, client):
        """GET /scans/{id} returns a queued scan."""
        created = client.post(
            "/api/v1/scans/trigger",
            json={"repo_full_name": "owner/repo"},
        ).json()

        response = client.get(f"/api/v1/scans/{created['scan_id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["scan_id"] == created["scan_id"]
        assert data["status"] == "queued"
        assert data["findings_by_severity"] == {}

    def test_list_scheduled_scans(self, client):
        """GET /scans/schedules returns configured schedules."""
        created = client.post(
            "/api/v1/scans/schedule",
            json={"repo_full_name": "owner/repo", "schedule": "daily"},
        )
        assert created.status_code == 201

        response = client.get("/api/v1/scans/schedules")

        assert response.status_code == 200
        schedules = response.json()["schedules"]
        assert len(schedules) == 1
        assert schedules[0]["repo_full_name"] == "owner/repo"
        assert schedules[0]["schedule"] == "daily"

    def test_schedule_scan(self, client):
        """POST /scans/schedule creates a current scheduled scan."""
        response = client.post(
            "/api/v1/scans/schedule",
            json={
                "repo_full_name": "owner/repo",
                "schedule": "daily",
                "branch": "main",
            },
        )

        assert response.status_code == 201
        data = response.json()
        UUID(data["id"])
        assert data["repo_full_name"] == "owner/repo"
        assert data["schedule"] == "daily"
        assert data["enabled"] is True

    def test_get_scan_history(self, client):
        """GET /scans/repo/{repo}/history returns repository scans."""
        created = client.post(
            "/api/v1/scans/trigger",
            json={"repo_full_name": "owner/repo"},
        ).json()

        response = client.get("/api/v1/scans/repo/owner/repo/history")

        assert response.status_code == 200
        data = response.json()
        assert data["repo_full_name"] == "owner/repo"
        assert [scan["scan_id"] for scan in data["scans"]] == [created["scan_id"]]


class TestNotificationsRouter:
    """Tests for the current notifications API."""

    @pytest.fixture
    def client(self):
        """Create an isolated notifications client."""
        from codeverify_core import notifications as notifications_core

        notifications_core._notification_configs.clear()
        app = FastAPI()
        app.include_router(notifications.router, prefix="/api/v1/notifications")
        with TestClient(app) as test_client:
            yield test_client
        notifications_core._notification_configs.clear()

    def test_configure_slack(self, client):
        """POST /config stores a Slack configuration."""
        response = client.post(
            "/api/v1/notifications/config",
            params={"repo_full_name": "owner/repo"},
            json={
                "channel": "slack",
                "webhook_url": "https://hooks.slack.com/services/test",
                "notification_types": ["analysis_complete"],
                "channel_name": "#codeverify",
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "status": "created",
            "repo_full_name": "owner/repo",
            "channel": "slack",
        }

    def test_configure_teams(self, client):
        """POST /config stores a Teams configuration."""
        response = client.post(
            "/api/v1/notifications/config",
            params={"repo_full_name": "owner/repo"},
            json={
                "channel": "teams",
                "webhook_url": "https://example.com/teams-webhook",
                "notification_types": ["analysis_complete"],
            },
        )

        assert response.status_code == 200
        assert response.json()["channel"] == "teams"

    def test_test_notification(self, client, monkeypatch):
        """POST /test uses the sender without making an HTTP request."""
        from codeverify_core.notifications import NotificationSender

        send_notification = AsyncMock(return_value=[{"channel": "slack", "success": True}])
        monkeypatch.setattr(
            NotificationSender,
            "send_analysis_notification",
            send_notification,
        )

        response = client.post(
            "/api/v1/notifications/test",
            json={
                "channel": "slack",
                "webhook_url": "https://hooks.slack.com/services/test",
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "message": "Test notification sent successfully",
        }
        send_notification.assert_awaited_once()

    def test_list_configurations(self, client):
        """GET /config/{repo} returns stored configurations."""
        client.post(
            "/api/v1/notifications/config",
            params={"repo_full_name": "owner/repo"},
            json={
                "channel": "slack",
                "webhook_url": "https://hooks.slack.com/services/test",
                "notification_types": ["analysis_complete"],
            },
        )

        response = client.get("/api/v1/notifications/config/owner/repo")

        assert response.status_code == 200
        assert response.json() == {
            "repo_full_name": "owner/repo",
            "configs": [
                {
                    "channel": "slack",
                    "enabled": True,
                    "notification_types": ["analysis_complete"],
                    "mention_on_critical": True,
                }
            ],
        }


class TestPublicAPIRouter:
    """Tests for authenticated Public API management endpoints."""

    @pytest.fixture
    def client(self):
        """Create an authenticated and isolated public API client."""
        public_api._api_keys.clear()
        public_api._webhooks.clear()
        public_api._webhook_deliveries.clear()

        app = FastAPI()

        async def override_current_user() -> TokenData:
            return TokenData(user_id=uuid4(), github_id=42, username="test-user")

        app.dependency_overrides[get_current_user] = override_current_user
        app.include_router(public_api.router, prefix="/api")
        with TestClient(app) as test_client:
            yield test_client

        public_api._api_keys.clear()
        public_api._webhooks.clear()
        public_api._webhook_deliveries.clear()

    @pytest.fixture
    def unauthenticated_client(self):
        """Create a public API client without an auth override."""
        app = FastAPI()
        app.include_router(public_api.router, prefix="/api")
        with TestClient(app) as test_client:
            yield test_client

    def test_create_api_key(self, client):
        """POST /keys returns the secret once."""
        response = client.post(
            "/api/keys",
            json={"name": "Test Key", "scopes": ["read", "write"]},
        )

        assert response.status_code == 201
        data = response.json()
        UUID(data["id"])
        assert data["key"].startswith("cv_")
        assert data["scopes"] == ["read", "write"]

    def test_list_api_keys(self, client):
        """GET /keys omits API key secrets."""
        created = client.post("/api/keys", json={"name": "Test Key"}).json()

        response = client.get("/api/keys")

        assert response.status_code == 200
        assert len(response.json()) == 1
        listed = response.json()[0]
        assert listed["id"] == created["id"]
        assert listed["key_prefix"] == created["key_prefix"]
        assert "key" not in listed

    def test_revoke_api_key(self, client):
        """DELETE /keys/{id} revokes a key."""
        created = client.post("/api/keys", json={"name": "Revoke Test"}).json()

        response = client.delete(f"/api/keys/{created['id']}")

        assert response.status_code == 200
        assert response.json() == {"revoked": True, "key_id": created["id"]}
        assert client.get("/api/keys").json() == []

    def test_create_webhook(self, client):
        """POST /webhooks creates a subscription."""
        response = client.post(
            "/api/webhooks",
            json={
                "url": "https://example.com/webhook",
                "events": ["analysis.completed", "finding.created"],
            },
        )

        assert response.status_code == 201
        data = response.json()
        UUID(data["id"])
        assert data["url"] == "https://example.com/webhook"
        assert data["events"] == ["analysis.completed", "finding.created"]
        assert data["active"] is True

    def test_list_webhooks(self, client):
        """GET /webhooks returns current configurations."""
        created = client.post(
            "/api/webhooks",
            json={"url": "https://example.com/webhook", "events": ["analysis.completed"]},
        ).json()

        response = client.get("/api/webhooks")

        assert response.status_code == 200
        assert response.json() == [created]

    def test_test_webhook(self, client, monkeypatch):
        """POST /webhooks/{id}/test uses the delivery surface without network I/O."""
        deliver = AsyncMock(return_value={"success": True, "status": 204})
        monkeypatch.setattr(public_api, "_deliver_webhook", deliver)
        created = client.post(
            "/api/webhooks",
            json={"url": "https://example.com/webhook", "events": ["test"]},
        ).json()

        response = client.post(f"/api/webhooks/{created['id']}/test")

        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "response_status": 204,
            "message": "Test webhook delivered",
        }
        deliver.assert_awaited_once()

    def test_list_webhook_events(self, client):
        """GET /events returns the documented webhook event names."""
        response = client.get("/api/events")

        assert response.status_code == 200
        names = {event["name"] for event in response.json()["events"]}
        assert names == set(public_api.WEBHOOK_EVENTS)

    def test_public_api_analyses_requires_auth(self, unauthenticated_client):
        """The public analyses endpoint requires bearer authentication."""
        response = unauthenticated_client.get("/api/v1/analyses")

        assert response.status_code == 401
        assert response.json() == {"detail": "Not authenticated"}

    def test_public_api_stats_requires_auth(self, unauthenticated_client):
        """The public statistics endpoint requires bearer authentication."""
        response = unauthenticated_client.get("/api/v1/stats")

        assert response.status_code == 401
        assert response.json() == {"detail": "Not authenticated"}


class TestRateLimiting:
    """Tests for the SlowAPI setup used by the application."""

    def test_setup_rate_limiting_enforces_decorated_limit(self):
        """setup_rate_limiting attaches the limiter and returns 429 at the limit."""
        app = FastAPI()
        setup_rate_limiting(app)

        @app.get("/limited")
        @limiter.limit("2/minute")
        async def limited_endpoint(request: Request):  # noqa: ARG001
            return {"status": "ok"}

        with TestClient(app) as client:
            responses = [client.get("/limited") for _ in range(3)]

        assert app.state.limiter is limiter
        assert [response.status_code for response in responses] == [200, 200, 429]
        assert "Rate limit exceeded" in responses[-1].json()["error"]
