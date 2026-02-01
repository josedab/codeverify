"""Locust load testing configuration for CodeVerify API."""
from locust import HttpUser, task, between, events
import json
import random
import string


class CodeVerifyUser(HttpUser):
    """Simulated CodeVerify user for load testing."""
    
    wait_time = between(1, 5)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token = None
        self.org_id = None
        self.repo_ids = []
        self.analysis_ids = []
    
    def on_start(self):
        """Set up user session."""
        # In real tests, would authenticate
        self.token = "test-token-" + "".join(random.choices(string.ascii_lowercase, k=8))
        self.org_id = "test-org-id"
    
    @property
    def headers(self):
        """Get auth headers."""
        return {"Authorization": f"Bearer {self.token}"}
    
    @task(10)
    def health_check(self):
        """Check API health."""
        self.client.get("/health")
    
    @task(5)
    def list_repositories(self):
        """List repositories."""
        self.client.get(
            "/api/v1/repositories",
            headers=self.headers,
            name="/api/v1/repositories"
        )
    
    @task(5)
    def list_analyses(self):
        """List analyses."""
        self.client.get(
            "/api/v1/analyses",
            headers=self.headers,
            name="/api/v1/analyses"
        )
    
    @task(3)
    def get_dashboard_stats(self):
        """Get dashboard statistics."""
        self.client.get(
            "/api/v1/stats/dashboard",
            headers=self.headers,
            name="/api/v1/stats/dashboard"
        )
    
    @task(2)
    def create_analysis(self):
        """Create a new analysis (simulated)."""
        payload = {
            "repository_id": "test-repo-id",
            "head_sha": "abc123",
            "pr_number": random.randint(1, 1000),
        }
        with self.client.post(
            "/api/v1/analyses",
            json=payload,
            headers=self.headers,
            name="/api/v1/analyses [POST]",
            catch_response=True
        ) as response:
            # Accept 401/403 as "expected" for unauthenticated tests
            if response.status_code in [401, 403]:
                response.success()
    
    @task(1)
    def get_usage(self):
        """Get usage information."""
        self.client.get(
            "/api/v1/usage/summary",
            headers=self.headers,
            name="/api/v1/usage/summary"
        )


class WebhookSimulator(HttpUser):
    """Simulated GitHub webhook events."""
    
    wait_time = between(0.5, 2)
    
    @task
    def send_pr_webhook(self):
        """Send a pull request webhook event."""
        payload = {
            "action": "opened",
            "pull_request": {
                "number": random.randint(1, 10000),
                "title": "Test PR",
                "head": {"sha": "abc123"},
                "base": {"sha": "def456"},
            },
            "repository": {
                "id": 12345,
                "full_name": "test/repo",
            },
            "installation": {"id": 1},
        }
        
        with self.client.post(
            "/webhooks/github",
            json=payload,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": "sha256=test",
            },
            name="/webhooks/github [PR]",
            catch_response=True
        ) as response:
            # Accept 400/401/403 as expected for invalid signatures
            if response.status_code in [400, 401, 403]:
                response.success()


class HighVolumeUser(HttpUser):
    """High volume user for stress testing."""
    
    wait_time = between(0.1, 0.5)
    
    @task(10)
    def rapid_health_check(self):
        """Rapid health checks."""
        self.client.get("/health")
    
    @task(5)
    def rapid_stats(self):
        """Rapid stats requests."""
        self.client.get("/")


# Event hooks for custom reporting
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """Log request metrics."""
    pass  # Could add custom metrics here


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("Load test starting...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("Load test completed.")
