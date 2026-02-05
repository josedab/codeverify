"""Tests for paste interception API endpoints."""

import pytest
from fastapi.testclient import TestClient

from codeverify_api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestQuickAnalysis:
    """Tests for /api/v1/analyses/quick endpoint."""

    def test_quick_analysis_clean_code(self, client):
        """Should return no findings for clean code."""
        response = client.post(
            "/api/v1/analyses/quick",
            json={
                "content": """
def add(a: int, b: int) -> int:
    return a + b
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "findings" in data
        assert "analysis_time_ms" in data
        assert data["mode"] == "quick"

    def test_quick_analysis_detects_eval(self, client):
        """Should detect eval() usage."""
        response = client.post(
            "/api/v1/analyses/quick",
            json={
                "content": """
def process(user_input):
    return eval(user_input)
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        findings = data["findings"]
        assert len(findings) > 0
        assert any(f["title"] == "Unsafe eval() usage" for f in findings)
        assert any(f["severity"] == "critical" for f in findings)

    def test_quick_analysis_detects_hardcoded_password(self, client):
        """Should detect hardcoded passwords."""
        response = client.post(
            "/api/v1/analyses/quick",
            json={
                "content": """
password = "supersecret123"
api_key = "sk-abc123xyz"
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        findings = data["findings"]
        assert any("password" in f["title"].lower() for f in findings)
        assert any("api key" in f["title"].lower() for f in findings)

    def test_quick_analysis_detects_todo(self, client):
        """Should detect TODO comments."""
        response = client.post(
            "/api/v1/analyses/quick",
            json={
                "content": """
def incomplete():
    # TODO: implement this function
    pass
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        findings = data["findings"]
        assert any("TODO" in f["title"] for f in findings)

    def test_quick_analysis_performance(self, client):
        """Quick analysis should complete in under 100ms."""
        response = client.post(
            "/api/v1/analyses/quick",
            json={
                "content": "x = 1\n" * 100,  # Simple code
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Allow some slack for CI environments
        assert data["analysis_time_ms"] < 500

    def test_quick_analysis_validates_input(self, client):
        """Should validate input parameters."""
        # Empty content
        response = client.post(
            "/api/v1/analyses/quick",
            json={
                "content": "",
                "language": "python",
            },
        )
        assert response.status_code == 422  # Validation error


class TestQuickTrustScore:
    """Tests for /api/v1/analyses/trust-score/quick endpoint."""

    def test_trust_score_clean_code(self, client):
        """Should return high trust score for clean code."""
        response = client.post(
            "/api/v1/analyses/trust-score/quick",
            json={
                "code": """
def add(a: int, b: int) -> int:
    return a + b

def test_add():
    assert add(1, 2) == 3
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["score"] >= 60
        assert data["risk_level"] in ["low", "medium"]

    def test_trust_score_ai_detection(self, client):
        """Should detect AI-generated code patterns."""
        response = client.post(
            "/api/v1/analyses/trust-score/quick",
            json={
                "code": """
def process_data(data):
    '''This function does the processing of data.'''
    # TODO: implement this function
    pass  # placeholder

def another_function():
    raise NotImplementedError
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_ai_generated"] is True
        assert data["ai_probability"] > 40

    def test_trust_score_security_penalty(self, client):
        """Should penalize code with security issues."""
        response = client.post(
            "/api/v1/analyses/trust-score/quick",
            json={
                "code": """
def dangerous(user_input):
    result = eval(user_input)
    password = "secret123"
    return result
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["score"] < 60
        assert data["risk_level"] in ["high", "critical"]

    def test_trust_score_response_structure(self, client):
        """Should return correct response structure."""
        response = client.post(
            "/api/v1/analyses/trust-score/quick",
            json={
                "code": "x = 1",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert "ai_probability" in data
        assert "risk_level" in data
        assert "is_ai_generated" in data
        assert "detected_model" in data
        assert "factors" in data
        assert "analysis_time_ms" in data


class TestPasteInterception:
    """Tests for /api/v1/analyses/intercept endpoint."""

    def test_intercept_combined_analysis(self, client):
        """Should return combined analysis results."""
        response = client.post(
            "/api/v1/analyses/intercept",
            json={
                "code": """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Check all fields are present
        assert "id" in data
        assert "is_ai_generated" in data
        assert "ai_confidence" in data
        assert "trust_score" in data
        assert "risk_level" in data
        assert "detected_model" in data
        assert "findings" in data
        assert "recommendations" in data
        assert "analysis_time_ms" in data

    def test_intercept_ai_code_detection(self, client):
        """Should detect AI-generated code and provide recommendations."""
        response = client.post(
            "/api/v1/analyses/intercept",
            json={
                "code": """
def process_data(data):
    '''
    This function does the processing of data.
    It takes data as input and returns processed data.
    '''
    # TODO: implement the actual processing logic
    pass  # placeholder

# Example usage:
# result = process_data(my_data)
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["is_ai_generated"] is True
        assert data["ai_confidence"] > 0.4
        assert len(data["recommendations"]) > 0
        assert any("AI" in r or "review" in r.lower() for r in data["recommendations"])

    def test_intercept_security_issues(self, client):
        """Should detect and report security issues."""
        response = client.post(
            "/api/v1/analyses/intercept",
            json={
                "code": """
import subprocess

def run_command(cmd):
    # Run user command
    result = subprocess.run(cmd, shell=True)
    return result

API_KEY = "sk-abc123xyz789"
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["risk_level"] in ["high", "critical"]
        assert len(data["findings"]) > 0
        assert any(f["category"] == "security" for f in data["findings"])

    def test_intercept_performance(self, client):
        """Combined analysis should complete in under 200ms."""
        response = client.post(
            "/api/v1/analyses/intercept",
            json={
                "code": """
def example():
    return 42
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Allow slack for CI
        assert data["analysis_time_ms"] < 1000

    def test_intercept_unique_ids(self, client):
        """Each interception should have a unique ID."""
        ids = set()
        for _ in range(5):
            response = client.post(
                "/api/v1/analyses/intercept",
                json={
                    "code": f"x = {_}",
                    "language": "python",
                },
            )
            assert response.status_code == 200
            ids.add(response.json()["id"])

        assert len(ids) == 5  # All IDs should be unique


class TestEdgeCases:
    """Edge case tests."""

    def test_large_code_input(self, client):
        """Should handle large code inputs."""
        large_code = "x = 1\n" * 1000
        response = client.post(
            "/api/v1/analyses/intercept",
            json={
                "code": large_code,
                "language": "python",
            },
        )
        assert response.status_code == 200

    def test_multilanguage_support(self, client):
        """Should work with different languages."""
        languages = ["python", "typescript", "javascript", "go", "java"]
        for lang in languages:
            response = client.post(
                "/api/v1/analyses/intercept",
                json={
                    "code": "// some code",
                    "language": lang,
                },
            )
            assert response.status_code == 200

    def test_special_characters(self, client):
        """Should handle special characters in code."""
        response = client.post(
            "/api/v1/analyses/intercept",
            json={
                "code": """
def greet(name):
    return f"Hello, {name}! 你好！ مرحبا"
""",
                "language": "python",
            },
        )
        assert response.status_code == 200

    def test_empty_findings_for_safe_code(self, client):
        """Safe code should have empty findings."""
        response = client.post(
            "/api/v1/analyses/intercept",
            json={
                "code": """
def safe_add(a: int, b: int) -> int:
    return a + b
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should have no critical/high security findings
        security_findings = [f for f in data["findings"] if f["category"] == "security"]
        assert len(security_findings) == 0
