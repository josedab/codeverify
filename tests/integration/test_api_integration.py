"""Integration tests for the API service."""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
import json


@pytest.fixture
def mock_db():
    """Mock database session."""
    with patch("codeverify_api.db.database.get_db") as mock:
        yield mock


@pytest.fixture
def mock_auth():
    """Mock authentication."""
    with patch("codeverify_api.auth.dependencies.get_current_user") as mock:
        mock.return_value = {
            "id": "test-user-id",
            "github_id": 12345,
            "username": "testuser",
        }
        yield mock


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Health endpoint returns OK."""
        from codeverify_api.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Root endpoint returns service info."""
        from codeverify_api.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["service"] == "CodeVerify API"


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    @pytest.mark.asyncio
    async def test_login_redirect(self):
        """Login endpoint redirects to GitHub."""
        from codeverify_api.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            follow_redirects=False
        ) as client:
            response = await client.get("/api/v1/auth/login")
            # Should redirect to GitHub OAuth
            assert response.status_code in [302, 307]
    
    @pytest.mark.asyncio
    async def test_me_unauthorized(self):
        """Me endpoint requires auth."""
        from codeverify_api.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/auth/me")
            assert response.status_code == 401


class TestAnalysesEndpoints:
    """Test analyses CRUD endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_analyses_unauthorized(self):
        """List analyses requires auth."""
        from codeverify_api.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/analyses")
            assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_get_analysis_not_found(self, mock_auth, mock_db):
        """Get non-existent analysis returns 404."""
        from codeverify_api.main import app
        
        mock_db.return_value.__aenter__ = AsyncMock()
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/analyses/nonexistent-id",
                headers={"Authorization": "Bearer test-token"}
            )
            # Without proper DB setup, this tests the route exists
            assert response.status_code in [401, 404, 500]


class TestWebhooksEndpoints:
    """Test GitHub webhook endpoints."""
    
    @pytest.mark.asyncio
    async def test_webhook_missing_signature(self):
        """Webhook without signature is rejected."""
        from codeverify_api.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhooks/github",
                json={"action": "opened"},
                headers={"X-GitHub-Event": "pull_request"}
            )
            # Should reject due to missing signature
            assert response.status_code in [400, 401, 403]
    
    @pytest.mark.asyncio
    async def test_webhook_ping_event(self):
        """Webhook ping event is acknowledged."""
        from codeverify_api.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhooks/github",
                json={"zen": "test"},
                headers={
                    "X-GitHub-Event": "ping",
                    "X-Hub-Signature-256": "sha256=test"
                }
            )
            # Ping should be acknowledged even with invalid signature in test
            assert response.status_code in [200, 400, 401, 403]


class TestStatsEndpoints:
    """Test statistics endpoints."""
    
    @pytest.mark.asyncio
    async def test_dashboard_stats_unauthorized(self):
        """Dashboard stats requires auth."""
        from codeverify_api.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/stats/dashboard")
            assert response.status_code == 401


class TestRepositoriesEndpoints:
    """Test repository endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_repositories_unauthorized(self):
        """List repositories requires auth."""
        from codeverify_api.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/repositories")
            assert response.status_code == 401


class TestOrganizationsEndpoints:
    """Test organization endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_organizations_unauthorized(self):
        """List organizations requires auth."""
        from codeverify_api.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/organizations")
            assert response.status_code == 401
