"""Seed agents for the CodeVerify Agent Marketplace.

Creates and publishes 5 first-party agents to bootstrap the marketplace.
Run with: python -m codeverify_core.seed_marketplace_agents

Agents:
1. OWASP Top 10 Scanner
2. Python Best Practices
3. TypeScript Strict Mode
4. Terraform Security
5. SQL Injection Scanner
"""

from __future__ import annotations

from codeverify_core.agent_marketplace import (
    AgentCategory,
    AgentManifest,
    AgentMarketplaceService,
    PricingModel,
)

SEED_AGENTS: list[AgentManifest] = [
    AgentManifest(
        name="OWASP Top 10 Scanner",
        version="1.0.0",
        description=(
            "Scans code for OWASP Top 10 2021 vulnerabilities: injection, "
            "broken auth, XSS, SSRF, security misconfiguration, and more. "
            "Maps findings to CWE IDs and provides fix suggestions."
        ),
        author="CodeVerify Team",
        author_id="codeverify",
        category=AgentCategory.SECURITY,
        languages=["python", "typescript", "go", "java"],
        entry_point="owasp_scanner.OWASPScannerAgent",
        pricing=PricingModel.FREE,
        tags=["owasp", "security", "cwe", "injection", "xss"],
    ),
    AgentManifest(
        name="Python Best Practices",
        version="1.0.0",
        description=(
            "Enforces Python best practices: proper exception handling, "
            "type annotations, f-string usage, pathlib over os.path, "
            "dataclass patterns, and PEP 8 compliance beyond linting."
        ),
        author="CodeVerify Team",
        author_id="codeverify",
        category=AgentCategory.QUALITY,
        languages=["python"],
        entry_point="python_best_practices.PythonBPAgent",
        pricing=PricingModel.FREE,
        tags=["python", "best-practices", "pep8", "typing", "quality"],
    ),
    AgentManifest(
        name="TypeScript Strict Mode",
        version="1.0.0",
        description=(
            "Enforces TypeScript strict mode patterns: no any, no type "
            "assertions, proper null checks, exhaustive switch, readonly "
            "properties, and discriminated unions."
        ),
        author="CodeVerify Team",
        author_id="codeverify",
        category=AgentCategory.QUALITY,
        languages=["typescript"],
        entry_point="typescript_strict.TSStrictAgent",
        pricing=PricingModel.FREE,
        tags=["typescript", "strict", "type-safety", "quality"],
    ),
    AgentManifest(
        name="Terraform Security",
        version="1.0.0",
        description=(
            "Scans Terraform configurations for security issues: open "
            "security groups, unencrypted resources, public S3 buckets, "
            "hardcoded credentials, and missing logging."
        ),
        author="CodeVerify Team",
        author_id="codeverify",
        category=AgentCategory.SECURITY,
        languages=[],
        entry_point="terraform_security.TerraformSecAgent",
        pricing=PricingModel.FREE,
        tags=["terraform", "iac", "aws", "security", "cloud"],
    ),
    AgentManifest(
        name="SQL Injection Scanner",
        version="1.0.0",
        description=(
            "Detects SQL injection vulnerabilities: string concatenation "
            "in queries, f-string SQL, unsanitized user input in database "
            "operations. Supports Python, Go, Java, and TypeScript ORMs."
        ),
        author="CodeVerify Team",
        author_id="codeverify",
        category=AgentCategory.SECURITY,
        languages=["python", "typescript", "go", "java"],
        entry_point="sql_injection.SQLInjectionAgent",
        pricing=PricingModel.FREE,
        tags=["sql", "injection", "security", "database", "cwe-89"],
    ),
]


def seed_marketplace(svc: AgentMarketplaceService | None = None) -> list[str]:
    """Seed the marketplace with first-party agents. Returns list of agent IDs."""
    if svc is None:
        svc = AgentMarketplaceService()

    agent_ids: list[str] = []
    for manifest in SEED_AGENTS:
        agent = svc.submit_agent(manifest)
        svc.review_agent(agent.id, "codeverify-team", approved=True,
                        comments="First-party agent — auto-approved")
        svc.publish_agent(agent.id)
        agent_ids.append(agent.id)
        print(f"  ✅ Published: {manifest.name} ({agent.id})")

    print(f"\n🎉 Seeded {len(agent_ids)} agents to marketplace")
    return agent_ids


if __name__ == "__main__":
    from codeverify_core.agent_marketplace import get_agent_marketplace_service
    svc = get_agent_marketplace_service()
    seed_marketplace(svc)
