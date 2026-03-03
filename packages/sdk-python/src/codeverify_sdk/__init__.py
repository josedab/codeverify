"""CodeVerify SDK — embed formal verification checks programmatically.

Usage:
    from codeverify_sdk import verify, check_safety, generate_proof

    result = verify("def add(a, b): return a + b")
    assert result.passed

    safety = check_safety("user_input = input()")
    print(safety.risk_level)

    proof = generate_proof("def divide(a, b): return a / b", property="b != 0")
    print(proof.status)
"""

from codeverify_sdk.client import CodeVerifyClient
from codeverify_sdk.models import (
    CheckResult,
    ProofResult,
    SafetyResult,
    VerificationResult,
)

__version__ = "0.1.0"

# Module-level client (configured via env or explicit init)
_client: CodeVerifyClient | None = None


def _get_client() -> CodeVerifyClient:
    global _client
    if _client is None:
        _client = CodeVerifyClient()
    return _client


def configure(
    api_key: str | None = None,
    api_url: str | None = None,
    timeout: float = 30.0,
) -> None:
    """Configure the global SDK client."""
    global _client
    _client = CodeVerifyClient(api_key=api_key, api_url=api_url, timeout=timeout)


def verify(
    code: str,
    language: str = "python",
    checks: list[str] | None = None,
) -> VerificationResult:
    """Verify code and return findings.

    Args:
        code: Source code to verify.
        language: Programming language (python, typescript, go, java).
        checks: Specific checks to run. None = all checks.

    Returns:
        VerificationResult with findings and status.
    """
    return _get_client().verify(code, language, checks)


def check_safety(
    code: str,
    language: str = "python",
) -> SafetyResult:
    """Check code for safety issues (null safety, bounds, injection, etc.).

    Returns:
        SafetyResult with risk_level and issues found.
    """
    return _get_client().check_safety(code, language)


def generate_proof(
    code: str,
    property: str | None = None,
    language: str = "python",
) -> ProofResult:
    """Generate a formal proof for a code property.

    Args:
        code: Source code containing the function to prove.
        property: Z3 property expression. Auto-inferred if None.
        language: Programming language.

    Returns:
        ProofResult with proof status and counterexamples.
    """
    return _get_client().generate_proof(code, property, language)


__all__ = [
    "configure",
    "verify",
    "check_safety",
    "generate_proof",
    "CodeVerifyClient",
    "VerificationResult",
    "SafetyResult",
    "ProofResult",
    "CheckResult",
]
