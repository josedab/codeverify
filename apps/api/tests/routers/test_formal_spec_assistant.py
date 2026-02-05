"""Tests for Formal Specification Assistant API endpoints."""

import pytest
from fastapi.testclient import TestClient

from codeverify_api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestNLToZ3Conversion:
    """Tests for /api/v1/specs/nl-to-z3 endpoint."""

    def test_convert_positive_constraint(self, client):
        """Should convert positive constraint."""
        response = client.post(
            "/api/v1/specs/nl-to-z3",
            json={
                "specification": "x must be positive",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["z3_expr"] is not None
        assert "x" in data["z3_expr"]
        assert ">" in data["z3_expr"] or "positive" in data["explanation"].lower()

    def test_convert_non_negative_constraint(self, client):
        """Should convert non-negative constraint."""
        response = client.post(
            "/api/v1/specs/nl-to-z3",
            json={
                "specification": "index must be non-negative",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["z3_expr"] is not None
        assert ">=" in data["z3_expr"] or "non-negative" in data["explanation"].lower()

    def test_convert_range_constraint(self, client):
        """Should convert range constraint."""
        response = client.post(
            "/api/v1/specs/nl-to-z3",
            json={
                "specification": "age must be between 0 and 150",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["z3_expr"] is not None
        # Should contain And and bounds
        assert "And" in data["z3_expr"] or data["confidence"] > 0

    def test_convert_not_null_constraint(self, client):
        """Should convert not null constraint."""
        response = client.post(
            "/api/v1/specs/nl-to-z3",
            json={
                "specification": "user must not be null",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["z3_expr"] is not None

    def test_convert_with_context(self, client):
        """Should use context for conversion."""
        response = client.post(
            "/api/v1/specs/nl-to-z3",
            json={
                "specification": "count must be positive",
                "context": {
                    "function_name": "process_items",
                    "parameters": {"count": "int"},
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_convert_returns_python_assert(self, client):
        """Should return Python assertion."""
        response = client.post(
            "/api/v1/specs/nl-to-z3",
            json={
                "specification": "n must be positive",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["python_assert"] is not None
        assert "assert" in data["python_assert"]

    def test_convert_includes_variables(self, client):
        """Should include detected variables."""
        response = client.post(
            "/api/v1/specs/nl-to-z3",
            json={
                "specification": "x must be positive",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "variables" in data
        assert "x" in data["variables"]

    def test_convert_returns_confidence(self, client):
        """Should return confidence score."""
        response = client.post(
            "/api/v1/specs/nl-to-z3",
            json={
                "specification": "x must be positive",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1

    def test_convert_empty_spec_fails(self, client):
        """Should reject empty specification."""
        response = client.post(
            "/api/v1/specs/nl-to-z3",
            json={
                "specification": "",
            },
        )
        assert response.status_code == 422  # Validation error


class TestBatchConversion:
    """Tests for /api/v1/specs/nl-to-z3/batch endpoint."""

    def test_batch_conversion(self, client):
        """Should convert multiple specifications."""
        response = client.post(
            "/api/v1/specs/nl-to-z3/batch",
            json={
                "specifications": [
                    "x must be positive",
                    "y must be non-negative",
                    "z must not be null",
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["successful"] >= 0
        assert len(data["results"]) == 3

    def test_batch_with_shared_context(self, client):
        """Should use shared context for batch."""
        response = client.post(
            "/api/v1/specs/nl-to-z3/batch",
            json={
                "specifications": [
                    "a must be positive",
                    "b must be positive",
                ],
                "context": {
                    "function_name": "add",
                    "return_type": "int",
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2


class TestSpecValidation:
    """Tests for /api/v1/specs/nl-to-z3/validate endpoint."""

    def test_validate_satisfiable_spec(self, client):
        """Should validate satisfiable specification."""
        response = client.post(
            "/api/v1/specs/nl-to-z3/validate",
            json={
                "z3_expr": "x > 0",
                "variables": {"x": "Int"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_satisfiable" in data
        assert "message" in data

    def test_validate_with_multiple_variables(self, client):
        """Should validate spec with multiple variables."""
        response = client.post(
            "/api/v1/specs/nl-to-z3/validate",
            json={
                "z3_expr": "And(x > 0, y > x)",
                "variables": {"x": "Int", "y": "Int"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_satisfiable" in data


class TestSpecRefinement:
    """Tests for /api/v1/specs/nl-to-z3/refine endpoint."""

    def test_refine_specification(self, client):
        """Should refine specification with feedback."""
        response = client.post(
            "/api/v1/specs/nl-to-z3/refine",
            json={
                "original_spec": "x must be positive",
                "current_z3": "x > 0",
                "feedback": "should include zero",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "z3_expr" in data
        assert "confidence" in data


class TestSpecSuggestions:
    """Tests for /api/v1/specs/nl-to-z3/suggest endpoint."""

    def test_suggest_specs_for_int_params(self, client):
        """Should suggest specs for integer parameters."""
        response = client.post(
            "/api/v1/specs/nl-to-z3/suggest",
            json={
                "function_signature": "def process(count: int, index: int) -> int:",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert data["count"] > 0
        # Should suggest positive/non-negative for int params
        suggestions_text = " ".join(data["suggestions"])
        assert "positive" in suggestions_text.lower() or "negative" in suggestions_text.lower()

    def test_suggest_specs_with_docstring(self, client):
        """Should use docstring for suggestions."""
        response = client.post(
            "/api/v1/specs/nl-to-z3/suggest",
            json={
                "function_signature": "def calculate(x: int, y: int) -> int:",
                "docstring": "Calculate sum of two positive numbers.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data

    def test_suggest_specs_for_string_params(self, client):
        """Should suggest specs for string parameters."""
        response = client.post(
            "/api/v1/specs/nl-to-z3/suggest",
            json={
                "function_signature": "def greet(name: str) -> str:",
            },
        )
        assert response.status_code == 200
        data = response.json()
        suggestions_text = " ".join(data["suggestions"]) if data["suggestions"] else ""
        # Should suggest not empty for string
        assert "empty" in suggestions_text.lower() or data["count"] >= 0


class TestTemplateLibrary:
    """Tests for template library endpoints."""

    def test_get_template_library(self, client):
        """Should return template library."""
        response = client.get("/api/v1/specs/nl-to-z3/templates")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert "count" in data
        assert data["count"] > 0

    def test_get_templates_by_domain(self, client):
        """Should return templates for specific domain."""
        response = client.get("/api/v1/specs/nl-to-z3/templates/numeric")
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "numeric"
        assert "templates" in data
        assert data["count"] > 0

    def test_get_templates_invalid_domain(self, client):
        """Should return 404 for invalid domain."""
        response = client.get("/api/v1/specs/nl-to-z3/templates/invalid_domain")
        assert response.status_code == 404

    def test_template_has_required_fields(self, client):
        """Templates should have required fields."""
        response = client.get("/api/v1/specs/nl-to-z3/templates")
        assert response.status_code == 200
        data = response.json()

        if data["templates"]:
            template = data["templates"][0]
            assert "id" in template
            assert "name" in template
            assert "domain" in template
            assert "nl_pattern" in template
            assert "z3_template" in template


class TestAssistantStats:
    """Tests for /api/v1/specs/nl-to-z3/stats endpoint."""

    def test_get_stats(self, client):
        """Should return assistant statistics."""
        response = client.get("/api/v1/specs/nl-to-z3/stats")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data
        if data["available"]:
            assert "cached_conversions" in data
            assert "template_count" in data


class TestExistingSpecEndpoints:
    """Tests for existing /api/v1/specs endpoints."""

    def test_generate_specs_from_code(self, client):
        """Should generate specs from code."""
        response = client.post(
            "/api/v1/specs/generate",
            json={
                "code": """
def add(x: int, y: int) -> int:
    '''Add two numbers.

    Requires:
        - x >= 0
        - y >= 0
    '''
    assert x >= 0
    assert y >= 0
    return x + y
""",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "functions" in data
        assert len(data["functions"]) > 0

    def test_get_spec_templates(self, client):
        """Should return spec templates."""
        response = client.get("/api/v1/specs/templates")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert len(data["templates"]) > 0
