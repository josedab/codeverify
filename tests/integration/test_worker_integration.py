"""Integration tests for the current worker and analysis package APIs."""

from unittest.mock import AsyncMock, patch

import pytest


class TestAnalysisPipelineIntegration:
    """Integration tests for worker pipeline construction and orchestration."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline with the current metadata-only constructor."""
        from codeverify_worker.tasks.analysis import AnalysisPipeline

        return AnalysisPipeline(
            repo_full_name="testorg/testrepo",
            pr_number=42,
            head_sha="abcdef1234567890",
            base_sha="1234567890abcdef",
            installation_id=99,
        )

    def test_pipeline_initialization(self, pipeline):
        """Pipeline retains all pull-request identity inputs."""
        assert pipeline.repo_full_name == "testorg/testrepo"
        assert pipeline.pr_number == 42
        assert pipeline.head_sha == "abcdef1234567890"
        assert pipeline.base_sha == "1234567890abcdef"
        assert pipeline.installation_id == 99
        assert pipeline.stages == []
        assert pipeline.findings == []

    @pytest.mark.asyncio
    async def test_pipeline_stage_tracking(self, pipeline, monkeypatch: pytest.MonkeyPatch):
        """A complete run records every current stage in execution order."""
        stage_methods = (
            "_fetch_pr_data",
            "_parse_code",
            "_semantic_analysis",
            "_formal_verification",
            "_security_analysis",
            "_synthesize_results",
        )
        for method_name in stage_methods:
            monkeypatch.setattr(
                pipeline,
                method_name,
                AsyncMock(return_value={"stage": method_name}),
            )

        result = await pipeline.run()

        assert result.status == "completed"
        assert [stage["name"] for stage in result.stages] == [
            "fetch",
            "parse",
            "semantic",
            "verify",
            "security",
            "synthesize",
        ]
        assert all(stage["status"] == "completed" for stage in result.stages)


class TestCodeParserIntegration:
    """Integration tests for structured parser results."""

    def test_python_parser_extracts_functions(self):
        """Python parser separates module functions from class methods."""
        from codeverify_verifier.parsers.python_parser import PythonParser

        code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(x, y):
    return x * y

class Calculator:
    def divide(self, a, b):
        return a / b
'''
        result = PythonParser().parse(code, "calculator.py")

        assert result.path == "calculator.py"
        assert result.language == "python"
        assert {function.name for function in result.functions} == {"add", "multiply"}
        assert [class_.name for class_ in result.classes] == ["Calculator"]
        assert [method.name for method in result.classes[0].methods] == ["divide"]
        assert result.errors == []

    def test_typescript_parser_extracts_functions(self):
        """TypeScript parser returns typed function models."""
        from codeverify_verifier.parsers.typescript_parser import TypeScriptParser

        code = """
function add(a: number, b: number): number {
    return a + b;
}

const multiply = (x: number, y: number) => x * y;

export async function fetchData(url: string): Promise<any> {
    return fetch(url);
}
"""
        result = TypeScriptParser().parse(code, "math.ts")
        function_names = {function.name for function in result.functions}

        assert result.path == "math.ts"
        assert result.language == "typescript"
        assert {"add", "fetchData"}.issubset(function_names)
        assert result.errors == []


class TestZ3VerifierIntegration:
    """Integration tests for public Z3 verification helpers."""

    def test_verifier_detects_division_by_zero(self):
        """An unconstrained divisor produces a concrete zero counterexample."""
        from codeverify_verifier.z3_verifier import Z3Verifier

        result = Z3Verifier().check_division_by_zero(
            divisor_var="divisor",
            divisor_range=None,
        )

        assert result["satisfiable"] is True
        assert result["counterexample"] == {"divisor": 0}
        assert "Division by zero possible" in result["message"]

    def test_verifier_detects_integer_overflow(self):
        """Signed 32-bit addition reports an overflowing operand pair."""
        from codeverify_verifier.z3_verifier import Z3Verifier

        result = Z3Verifier().check_integer_overflow(
            var_name="total",
            operation="add",
            operand1_range=(2**31 - 1, 2**31 - 1),
            operand2_range=(1, 1),
            bit_width=32,
        )

        assert result["satisfiable"] is True
        assert result["counterexample"]["a"] == 2**31 - 1
        assert result["counterexample"]["b"] == 1
        assert result["counterexample"]["result"] == "overflow"

    def test_verifier_checks_array_bounds(self):
        """A constrained index range proves a fixed-size access safe."""
        from codeverify_verifier.z3_verifier import Z3Verifier

        result = Z3Verifier().check_array_bounds(
            index_var="index",
            index_range=(0, 9),
            array_length=10,
        )

        assert result["satisfiable"] is False
        assert result["counterexample"] is None
        assert result["message"] == "Array access is always within bounds"


class TestAIAgentIntegration:
    """Integration tests for the shared AgentResult contract."""

    @pytest.mark.asyncio
    async def test_semantic_agent_analyzes_code(self):
        """SemanticAgent parses a mocked provider response into AgentResult."""
        from codeverify_agents import AgentResult, SemanticAgent

        agent = SemanticAgent()
        response = {
            "content": (
                '{"summary":"Adds two values","functions":[{"name":"add",'
                '"purpose":"Calculate a sum","preconditions":[],"postconditions":[],'
                '"assumptions":[],"edge_cases":[],"concerns":[]}],'
                '"behavioral_changes":[],"verification_hints":[]}'
            ),
            "tokens": 17,
        }

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=response)):
            result = await agent.analyze(
                "def add(a, b): return a + b",
                {"file_path": "math.py", "language": "python"},
            )

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.tokens_used == 17
        assert result.data["functions"][0]["name"] == "add"

    @pytest.mark.asyncio
    async def test_security_agent_detects_vulnerabilities(self):
        """SecurityAgent exposes provider findings through AgentResult."""
        from codeverify_agents import AgentResult, SecurityAgent

        agent = SecurityAgent()
        response = {
            "content": (
                '{"vulnerabilities":[{"id":"vuln-1","severity":"critical",'
                '"category":"command_injection","title":"Unsafe shell command"}],'
                '"secrets_detected":[],"security_score":20,'
                '"summary":"Command injection found"}'
            ),
            "tokens": 23,
        }

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=response)):
            result = await agent.analyze(
                "import os\nos.system(user_input)",
                {"file_path": "commands.py", "language": "python"},
            )

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.data["vulnerabilities"][0]["category"] == "command_injection"
        assert result.data["security_score"] == 20

    @pytest.mark.asyncio
    async def test_synthesis_agent_consolidates_findings(self):
        """SynthesisAgent accepts cross-agent inputs through its BaseAgent API."""
        from codeverify_agents import AgentResult, SynthesisAgent

        agent = SynthesisAgent()
        response = {
            "content": (
                '{"summary":{"total_issues":1,"critical":0,"high":0,'
                '"medium":1,"low":0,"pass":true},'
                '"findings":[{"id":"finding-1","title":"Missing error handling",'
                '"severity":"medium"}],"github_comment":"Analysis complete"}'
            ),
            "tokens": 31,
        }
        context = {
            "semantic_results": {"issues": [{"title": "Missing error handling"}]},
            "verification_results": {"satisfiable": False},
            "security_results": {"vulnerabilities": []},
        }

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=response)):
            result = await agent.analyze("def load(): ...", context)

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.data["summary"]["total_issues"] == 1
        assert result.data["findings"][0]["title"] == "Missing error handling"


class TestConfigIntegration:
    """Integration tests for configuration parsing and path filtering."""

    def test_config_parser_handles_full_config(self):
        """Config parser maps YAML fields into current dataclasses."""
        from codeverify_core.config import parse_config

        config_yaml = """
version: "1"
languages:
  - python
  - typescript
include:
  - "src/**/*.py"
exclude:
  - "venv/**"
verification:
  enabled: true
  timeout: 30
ai:
  enabled: true
  semantic: true
  security: true
thresholds:
  critical: 0
  high: 0
  medium: 5
  low: 10
"""

        config = parse_config(config_yaml)

        assert config.version == "1"
        assert config.languages == ["python", "typescript"]
        assert config.verification.enabled is True
        assert config.ai.semantic_analysis is True
        assert config.thresholds.critical == 0

    def test_should_analyze_file_respects_patterns(self):
        """File analysis applies current fnmatch include/exclude semantics."""
        from codeverify_core.config import CodeVerifyConfig, should_analyze_file

        config = CodeVerifyConfig(
            include_patterns=["src/*.py"],
            exclude_patterns=["src/test_*.py"],
        )

        assert should_analyze_file(config, "src/main.py") is True
        assert should_analyze_file(config, "src/test_main.py") is False
        assert should_analyze_file(config, "venv/lib/foo.py") is False
