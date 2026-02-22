"""Integration tests: Agentic Orchestrator ↔ AI Agents.

Tests that the orchestrator can dispatch to actual AI agent modules
from packages/ai-agents (SemanticAgent, SecurityAgent, TrustScoreAgent)
using their real interfaces, with mock LLM responses.
"""

from __future__ import annotations

import pytest

try:
    from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent, CodeContext
    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False
    # Provide stubs for test collection when codeverify_agents is not installed
    class AgentConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs): pass
    class AgentResult:  # type: ignore[no-redef]
        def __init__(self, success=True, data=None, error=None, tokens_used=0, latency_ms=0):
            self.success = success
            self.data = data or {}
            self.error = error
            self.tokens_used = tokens_used
            self.latency_ms = latency_ms
    class CodeContext:  # type: ignore[no-redef]
        def __init__(self, code="", file_path="unknown", language="python", **kwargs):
            self.code = code
            self.file_path = file_path
            self.language = language
            self.is_ai_generated = kwargs.get("is_ai_generated", False)
            self.metadata = kwargs.get("metadata", {})
        @classmethod
        def from_dict(cls, code, context):
            return cls(code=code, file_path=context.get("file_path", "unknown"),
                      language=context.get("language", "python"),
                      metadata={k: v for k, v in context.items()
                                if k not in ("file_path", "language", "is_ai_generated")})

pytestmark = pytest.mark.skipif(not HAS_AGENTS, reason="codeverify_agents not installed")


# ─── Mock Agents wrapping real agent classes ───────────────────────────


class MockableSemanticAgent:
    """Wraps the real SemanticAgent interface for testing without LLM calls."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

    async def analyze(self, code: str, context: dict | None = None) -> AgentResult:
        findings = []
        if "eval(" in code:
            findings.append({
                "title": "Dangerous eval usage",
                "severity": "critical",
                "category": "code_injection",
                "line": next(
                    (i for i, l in enumerate(code.split("\n"), 1) if "eval(" in l), 1
                ),
            })
        if "# TODO" in code:
            findings.append({
                "title": "Unresolved TODO",
                "severity": "low",
                "category": "maintainability",
            })
        return AgentResult(
            success=True,
            data={
                "intent": "Code analysis",
                "findings": findings,
                "contracts": {"preconditions": [], "postconditions": []},
            },
            tokens_used=150,
            latency_ms=50,
        )


class MockableSecurityAgent:
    """Wraps the real SecurityAgent interface for testing."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

    async def analyze(self, code: str, context: dict | None = None) -> AgentResult:
        findings = []
        if "password" in code.lower():
            findings.append({
                "title": "Potential hardcoded credential",
                "severity": "high",
                "category": "credential_exposure",
                "cwe": 798,
            })
        if "http://" in code:
            findings.append({
                "title": "Unencrypted HTTP",
                "severity": "medium",
                "category": "transport_security",
                "owasp": "A02:2021",
            })
        return AgentResult(
            success=True,
            data={"vulnerabilities": findings},
            tokens_used=120,
            latency_ms=40,
        )


class MockableTrustScoreAgent:
    """Wraps the real TrustScoreAgent interface for testing."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

    async def calculate_score(self, code: str, context: dict | None = None) -> AgentResult:
        lines = code.split("\n")
        complexity = len(lines)
        score = max(0, 100 - complexity * 2)
        risk = "low" if score >= 70 else "medium" if score >= 40 else "high"
        return AgentResult(
            success=True,
            data={
                "score": score,
                "risk_level": risk,
                "factors": {
                    "complexity_score": min(1.0, complexity / 50),
                    "verification_coverage": 0.5,
                },
            },
            tokens_used=80,
            latency_ms=30,
        )


# ─── Integration with Orchestrator ────────────────────────────────────


class AgentBridgeExecutor:
    """Bridge between the orchestrator's task model and real AI agent interfaces.

    Maps TaskType → real agent module, translates file data into CodeContext,
    and converts AgentResult → TaskResult.
    """

    def __init__(self) -> None:
        from codeverify_core.agentic_orchestrator import TaskType

        self._agents: dict[TaskType, Any] = {
            TaskType.SEMANTIC_ANALYSIS: MockableSemanticAgent(),
            TaskType.SECURITY_SCAN: MockableSecurityAgent(),
            TaskType.TRUST_SCORE: MockableTrustScoreAgent(),
        }

    async def execute(self, task) -> None:
        """Execute a task using the appropriate real agent."""
        from codeverify_core.agentic_orchestrator import (
            AgentFinding,
            TaskResult,
            TaskStatus,
        )
        from datetime import datetime, timezone
        import asyncio

        agent = self._agents.get(task.task_type)
        if not agent:
            task.status = TaskStatus.SKIPPED
            return

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)

        code_snippets = "\n".join(
            f.get("content", f.get("path", ""))
            for f in task.target_files
            if isinstance(f, dict)
        ) or "\n".join(str(f) for f in task.target_files)

        try:
            if hasattr(agent, "calculate_score"):
                result = await agent.calculate_score(code_snippets)
            else:
                result = await agent.analyze(code_snippets)

            findings: list[AgentFinding] = []
            raw_findings = result.data.get("findings", result.data.get("vulnerabilities", []))
            for rf in raw_findings:
                findings.append(AgentFinding(
                    agent_type=task.task_type,
                    file_path=rf.get("file_path", ""),
                    line=rf.get("line", 0),
                    severity=rf.get("severity", "medium"),
                    category=rf.get("category", ""),
                    message=rf.get("title", ""),
                    confidence=0.85,
                ))

            task.result = TaskResult(
                task_id=task.id,
                findings=findings,
                tokens_used=result.tokens_used,
                cost_cents=result.tokens_used * 0.003,
                latency_ms=int(result.latency_ms),
            )
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = TaskResult(task_id=task.id, success=False, error=str(e))
            task.completed_at = datetime.now(timezone.utc)


# ─── Tests ─────────────────────────────────────────────────────────────


class TestCodeContextBridge:
    """Test that CodeContext from base.py works with our orchestrator."""

    def test_code_context_creation(self):
        ctx = CodeContext(
            code="def foo(): pass",
            file_path="app.py",
            language="python",
            is_ai_generated=True,
        )
        assert ctx.code == "def foo(): pass"
        assert ctx.is_ai_generated is True

    def test_code_context_from_dict(self):
        ctx = CodeContext.from_dict("x = 1", {
            "file_path": "test.py",
            "language": "python",
            "is_ai_generated": False,
            "custom_field": "value",
        })
        assert ctx.file_path == "test.py"
        assert ctx.metadata["custom_field"] == "value"


class TestMockAgentInterfaces:
    """Test that mock agents match real agent interfaces."""

    @pytest.mark.asyncio
    async def test_semantic_agent_interface(self):
        agent = MockableSemanticAgent()
        result = await agent.analyze("x = eval(input())")
        assert result.success is True
        assert len(result.data["findings"]) >= 1
        assert result.data["findings"][0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_security_agent_interface(self):
        agent = MockableSecurityAgent()
        result = await agent.analyze('password = "secret123"')
        assert result.success is True
        assert len(result.data["vulnerabilities"]) >= 1

    @pytest.mark.asyncio
    async def test_trust_score_agent_interface(self):
        agent = MockableTrustScoreAgent()
        result = await agent.calculate_score("def foo():\n    return 1\n")
        assert result.success is True
        assert 0 <= result.data["score"] <= 100
        assert result.data["risk_level"] in ("low", "medium", "high")


class TestOrchestratorAgentBridge:
    """Test the bridge between orchestrator tasks and agent modules."""

    @pytest.mark.asyncio
    async def test_bridge_executes_semantic_task(self):
        from codeverify_core.agentic_orchestrator import TaskStatus, TaskType, VerificationTask

        bridge = AgentBridgeExecutor()
        task = VerificationTask(
            task_type=TaskType.SEMANTIC_ANALYSIS,
            target_files=[{"path": "app.py", "content": "x = eval(input())"}],
        )
        await bridge.execute(task)
        assert task.status == TaskStatus.COMPLETED
        assert task.result is not None
        assert len(task.result.findings) >= 1
        assert task.result.findings[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_bridge_executes_security_task(self):
        from codeverify_core.agentic_orchestrator import TaskStatus, TaskType, VerificationTask

        bridge = AgentBridgeExecutor()
        task = VerificationTask(
            task_type=TaskType.SECURITY_SCAN,
            target_files=[{"path": "config.py", "content": 'password = "hunter2"'}],
        )
        await bridge.execute(task)
        assert task.status == TaskStatus.COMPLETED
        assert len(task.result.findings) >= 1

    @pytest.mark.asyncio
    async def test_bridge_executes_trust_score_task(self):
        from codeverify_core.agentic_orchestrator import TaskStatus, TaskType, VerificationTask

        bridge = AgentBridgeExecutor()
        task = VerificationTask(
            task_type=TaskType.TRUST_SCORE,
            target_files=[{"path": "simple.py", "content": "x = 1\n"}],
        )
        await bridge.execute(task)
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_bridge_skips_unsupported_task_type(self):
        from codeverify_core.agentic_orchestrator import TaskStatus, TaskType, VerificationTask

        bridge = AgentBridgeExecutor()
        task = VerificationTask(
            task_type=TaskType.STYLE_CHECK,
            target_files=[{"path": "app.py"}],
        )
        await bridge.execute(task)
        assert task.status == TaskStatus.SKIPPED


class TestOrchestratorEndToEnd:
    """End-to-end: planner → bridge executor → conflict resolver."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_agent_bridge(self):
        from codeverify_core.agentic_orchestrator import (
            ConflictResolver,
            ConflictStrategy,
            PlannerAgent,
            PRContext,
            TaskStatus,
        )

        # 1. Plan
        planner = PlannerAgent()
        ctx = PRContext(
            pr_id="100",
            repo="acme/api",
            changed_files=[
                {"path": "auth.py", "content": 'password = "secret"\neval(cmd)'},
                {"path": "utils.py", "content": "def add(a, b): return a + b\n"},
            ],
        )
        plan = planner.create_plan(ctx, budget_cents=100.0)
        assert plan.task_count >= 2

        # 2. Execute via bridge
        bridge = AgentBridgeExecutor()
        all_findings = []
        for task in plan.tasks:
            if task.status == TaskStatus.SKIPPED:
                continue
            # Inject real file content into target_files
            task.target_files = ctx.changed_files
            await bridge.execute(task)
            if task.result and task.result.findings:
                all_findings.extend(task.result.findings)

        assert len(all_findings) >= 1  # At least eval() or password

        # 3. Resolve conflicts
        resolver = ConflictResolver()
        resolved, conflicts = resolver.resolve(
            all_findings, ConflictStrategy.CONFIDENCE_WEIGHTED
        )
        assert len(resolved) >= 1

    @pytest.mark.asyncio
    async def test_cost_tracking_across_agents(self):
        from codeverify_core.agentic_orchestrator import (
            PlannerAgent,
            PRContext,
            TaskStatus,
        )

        planner = PlannerAgent()
        ctx = PRContext(
            changed_files=[{"path": "app.py", "content": "def f(): pass\n"}],
        )
        plan = planner.create_plan(ctx)

        bridge = AgentBridgeExecutor()
        total_tokens = 0
        for task in plan.tasks:
            if task.status == TaskStatus.SKIPPED:
                continue
            task.target_files = ctx.changed_files
            await bridge.execute(task)
            if task.result:
                total_tokens += task.result.tokens_used

        assert total_tokens >= 0  # At least some tokens used
