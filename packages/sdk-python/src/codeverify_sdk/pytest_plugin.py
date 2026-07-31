"""pytest plugin for CodeVerify — @pytest.mark.verified decorator.

Usage:
    @pytest.mark.verified(checks=["null_safety", "bounds_check"])
    def test_my_function():
        assert my_function(42) == 84

    @pytest.mark.verified(property="result >= 0")
    def test_positive_result():
        assert compute(10) >= 0
"""

from __future__ import annotations

import inspect

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the 'verified' marker."""
    config.addinivalue_line(
        "markers",
        "verified(checks=None, property=None, language='python'): "
        "Run CodeVerify verification on the test's target function",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Process items with @pytest.mark.verified."""
    for item in items:
        marker = item.get_closest_marker("verified")
        if marker is not None:
            item.add_marker(pytest.mark.usefixtures("_codeverify_check"))


@pytest.fixture
def _codeverify_check(request: pytest.FixtureRequest) -> None:
    """Fixture that runs CodeVerify checks before the test executes."""
    marker = request.node.get_closest_marker("verified")
    if marker is None:
        return

    checks: list[str] | None = marker.kwargs.get("checks")
    prop: str | None = marker.kwargs.get("property")
    language: str = marker.kwargs.get("language", "python")

    # Get the test function's source code
    test_func = request.node.obj
    try:
        source = inspect.getsource(test_func)
    except (OSError, TypeError):
        return

    from codeverify_sdk import check_safety, generate_proof, verify

    # Run verification
    result = verify(source, language=language, checks=checks)
    if not result.passed:
        critical = [f for f in result.findings if f.severity in ("critical", "high")]
        if critical:
            messages = [f"  - [{f.severity}] {f.title}: {f.description}" for f in critical]
            pytest.fail(
                f"CodeVerify verification failed with {len(critical)} critical finding(s):\n"
                + "\n".join(messages),
                pytrace=False,
            )

    # Run safety check
    safety = check_safety(source, language=language)
    if not safety.safe:
        issues = [f"  - [{i.severity}] {i.title}" for i in safety.issues]
        pytest.fail(
            f"CodeVerify safety check failed ({safety.risk_level} risk):\n" + "\n".join(issues),
            pytrace=False,
        )

    # Run property proof if specified
    if prop:
        proof = generate_proof(source, property=prop, language=language)
        if proof.status == "disproved":
            ce_info = ""
            if proof.counterexamples:
                ce_info = f"\n  Counterexample: {proof.counterexamples[0]}"
            pytest.fail(
                f"CodeVerify property '{prop}' disproved for test.{ce_info}",
                pytrace=False,
            )


@pytest.fixture
def codeverify():
    """Fixture providing direct access to the CodeVerify SDK client."""
    from codeverify_sdk import CodeVerifyClient

    client = CodeVerifyClient()
    yield client
    client.close()
