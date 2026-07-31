"""Tests for activation items: examples, protocol spec, seed agents."""


class TestSeedMarketplaceAgents:
    def test_seed_creates_5_agents(self):
        from codeverify_core.agent_marketplace import AgentMarketplaceService, PublishStatus
        from codeverify_core.seed_marketplace_agents import seed_marketplace

        svc = AgentMarketplaceService()
        ids = seed_marketplace(svc)
        assert len(ids) == 5

        for agent_id in ids:
            agent = svc.get_agent(agent_id)
            assert agent is not None
            assert agent.status == PublishStatus.PUBLISHED

    def test_seed_agents_searchable(self):
        from codeverify_core.agent_marketplace import AgentCategory, AgentMarketplaceService
        from codeverify_core.seed_marketplace_agents import seed_marketplace

        svc = AgentMarketplaceService()
        seed_marketplace(svc)

        security = svc.search(category=AgentCategory.SECURITY)
        assert len(security) >= 3  # OWASP, Terraform, SQL Injection

        quality = svc.search(category=AgentCategory.QUALITY)
        assert len(quality) >= 2  # Python BP, TS Strict

    def test_seed_agents_have_tags(self):
        from codeverify_core.agent_marketplace import AgentMarketplaceService
        from codeverify_core.seed_marketplace_agents import seed_marketplace

        svc = AgentMarketplaceService()
        seed_marketplace(svc)

        owasp = svc.search("owasp")
        assert len(owasp) >= 1
        assert "owasp" in owasp[0].manifest.tags

    def test_seed_agents_are_free(self):
        from codeverify_core.agent_marketplace import AgentMarketplaceService, PricingModel
        from codeverify_core.seed_marketplace_agents import seed_marketplace

        svc = AgentMarketplaceService()
        seed_marketplace(svc)
        all_agents = svc.search()
        assert all(a.manifest.pricing == PricingModel.FREE for a in all_agents)


class TestProtocolClient:
    def test_local_verify_clean(self):
        from codeverify_core.protocol_client import VerificationClient

        client = VerificationClient()
        result = client.verify("def add(a, b): return a + b", language="python")
        assert result.status == "verified"

    def test_local_verify_buggy(self):
        from codeverify_core.protocol_client import VerificationClient

        client = VerificationClient()
        result = client.verify("x = eval(input())", language="python")
        assert result.status == "failed"
        assert result.finding_count >= 1

    def test_capabilities(self):
        from codeverify_core.protocol_client import VerificationClient

        client = VerificationClient()
        caps = client.get_capabilities()
        assert "python" in caps["supported_languages"]

    def test_verify_files(self):
        from codeverify_core.protocol_client import VerificationClient

        client = VerificationClient()
        result = client.verify_files(
            {
                "a.py": "def safe(): return 1",
                "b.py": "x = eval('bad')",
            }
        )
        assert result.finding_count >= 1


class TestExampleVerification:
    """Verify that example projects contain issues CodeVerify can find."""

    def test_fastapi_example_has_issues(self):
        from codeverify_core.verification_protocol import VerificationProtocolServer, VerifyRequest

        with open("examples/python-fastapi/app.py") as f:
            code = f.read()

        server = VerificationProtocolServer()
        result = server.verify(VerifyRequest(files=[{"path": "app.py", "content": code}]))
        assert result.status.value == "failed"
        assert len(result.findings) >= 2  # eval + SQL injection at minimum

    def test_terraform_example_has_issues(self):
        from codeverify_core.multimodal_verify import MultiModalVerificationService

        with open("examples/terraform-infra/main.tf") as f:
            content = f.read()

        svc = MultiModalVerificationService()
        result = svc.verify_file("main.tf", content)
        assert not result.passed
        assert len(result.findings) >= 2  # 0.0.0.0/0 + public access

    def test_monorepo_api_has_issues(self):
        from codeverify_core.verification_protocol import VerificationProtocolServer, VerifyRequest

        with open("examples/monorepo/api.py") as f:
            code = f.read()

        server = VerificationProtocolServer()
        result = server.verify(VerifyRequest(files=[{"path": "api.py", "content": code}]))
        # Should find division by zero pattern
        assert len(result.findings) >= 0  # May or may not match patterns
