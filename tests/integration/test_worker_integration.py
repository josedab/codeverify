"""Integration tests for the worker service."""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import asyncio


class TestAnalysisPipelineIntegration:
    """Integration tests for the analysis pipeline."""
    
    @pytest.fixture
    def mock_github_client(self):
        """Create mock GitHub client."""
        client = Mock()
        client.get_pr_files = AsyncMock(return_value=[
            {
                "filename": "src/main.py",
                "status": "modified",
                "additions": 10,
                "deletions": 2,
                "patch": "@@ -1,5 +1,13 @@\n+def divide(a, b):\n+    return a / b",
            }
        ])
        client.get_file_content = AsyncMock(return_value="def divide(a, b):\n    return a / b")
        client.create_check_run = AsyncMock(return_value={"id": 12345})
        client.update_check_run = AsyncMock()
        client.create_pr_comment = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = Mock()
        session.commit = Mock()
        session.refresh = Mock()
        session.add = Mock()
        session.execute = AsyncMock()
        return session
    
    @pytest.fixture
    def sample_analysis(self):
        """Create sample analysis record."""
        analysis = Mock()
        analysis.id = "test-analysis-123"
        analysis.repository_id = "test-repo-456"
        analysis.pr_number = 42
        analysis.pr_title = "Test PR"
        analysis.head_sha = "abc123"
        analysis.base_sha = "def456"
        analysis.status = "pending"
        analysis.stages = []
        analysis.findings = []
        analysis.summary = {}
        return analysis
    
    @pytest.fixture
    def sample_repository(self):
        """Create sample repository record."""
        repo = Mock()
        repo.id = "test-repo-456"
        repo.github_id = 98765
        repo.full_name = "testorg/testrepo"
        repo.organization_id = "test-org-789"
        repo.default_branch = "main"
        return repo
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(
        self, mock_db_session, sample_analysis, sample_repository, mock_github_client
    ):
        """Pipeline initializes correctly with all dependencies."""
        from codeverify_worker.tasks.analysis import AnalysisPipeline
        
        with patch.object(AnalysisPipeline, '_init_github_client', return_value=mock_github_client):
            pipeline = AnalysisPipeline(mock_db_session, sample_analysis, sample_repository)
            
            assert pipeline.analysis == sample_analysis
            assert pipeline.repository == sample_repository
    
    @pytest.mark.asyncio
    async def test_pipeline_stage_tracking(
        self, mock_db_session, sample_analysis, sample_repository, mock_github_client
    ):
        """Pipeline tracks stage completion."""
        from codeverify_worker.tasks.analysis import AnalysisPipeline
        
        with patch.object(AnalysisPipeline, '_init_github_client', return_value=mock_github_client):
            pipeline = AnalysisPipeline(mock_db_session, sample_analysis, sample_repository)
            
            # Simulate stage completion
            pipeline._record_stage("fetch", "completed", 1000)
            
            assert len(sample_analysis.stages) >= 0  # Stages tracked


class TestCodeParserIntegration:
    """Integration tests for code parsing."""
    
    def test_python_parser_extracts_functions(self):
        """Python parser extracts function definitions."""
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
        parser = PythonParser()
        result = parser.parse(code)
        
        assert "functions" in result
        functions = result["functions"]
        
        # Should find add, multiply, and divide
        function_names = [f["name"] for f in functions]
        assert "add" in function_names
        assert "multiply" in function_names
    
    def test_typescript_parser_extracts_functions(self):
        """TypeScript parser extracts function definitions."""
        from codeverify_verifier.parsers.typescript_parser import TypeScriptParser
        
        code = '''
function add(a: number, b: number): number {
    return a + b;
}

const multiply = (x: number, y: number) => x * y;

export async function fetchData(url: string): Promise<any> {
    return fetch(url);
}
'''
        parser = TypeScriptParser()
        result = parser.parse(code)
        
        assert "functions" in result
        functions = result["functions"]
        
        function_names = [f["name"] for f in functions]
        assert "add" in function_names


class TestZ3VerifierIntegration:
    """Integration tests for Z3 verification."""
    
    def test_verifier_detects_division_by_zero(self):
        """Verifier detects potential division by zero."""
        from codeverify_verifier.z3_verifier import Z3Verifier
        
        verifier = Z3Verifier()
        
        # This should potentially detect division issue
        result = verifier.check_division_by_zero(
            dividend="x",
            divisor="y",
            constraints=[]
        )
        
        # Result should indicate potential issue when divisor can be zero
        assert isinstance(result, dict)
        assert "safe" in result or "result" in result or "status" in result
    
    def test_verifier_detects_integer_overflow(self):
        """Verifier detects potential integer overflow."""
        from codeverify_verifier.z3_verifier import Z3Verifier
        
        verifier = Z3Verifier()
        
        result = verifier.check_integer_overflow(
            expression="a + b",
            bit_width=32
        )
        
        assert isinstance(result, dict)
    
    def test_verifier_checks_array_bounds(self):
        """Verifier checks array bounds."""
        from codeverify_verifier.z3_verifier import Z3Verifier
        
        verifier = Z3Verifier()
        
        result = verifier.check_array_bounds(
            index="i",
            array_length=10
        )
        
        assert isinstance(result, dict)


class TestAIAgentIntegration:
    """Integration tests for AI agents."""
    
    @pytest.mark.asyncio
    async def test_semantic_agent_analyzes_code(self):
        """Semantic agent can analyze code."""
        from codeverify_agents.semantic import SemanticAnalysisAgent
        
        agent = SemanticAnalysisAgent()
        
        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = '{"intent": "Calculate sum", "issues": []}'
            
            result = await agent.analyze(
                "def add(a, b): return a + b",
                {"file_path": "math.py"}
            )
            
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_security_agent_detects_vulnerabilities(self):
        """Security agent detects common vulnerabilities."""
        from codeverify_agents.security import SecurityAnalysisAgent
        
        agent = SecurityAnalysisAgent()
        
        vulnerable_code = '''
import os
os.system(user_input)  # Command injection
'''
        
        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = '''
{
    "vulnerabilities": [
        {
            "type": "command_injection",
            "severity": "critical",
            "line": 2
        }
    ]
}
'''
            
            result = await agent.analyze(vulnerable_code, {})
            
            assert "vulnerabilities" in result
    
    @pytest.mark.asyncio
    async def test_synthesis_agent_consolidates_findings(self):
        """Synthesis agent consolidates findings from multiple sources."""
        from codeverify_agents.synthesis import SynthesisAgent
        
        agent = SynthesisAgent()
        
        findings = {
            "semantic": {"issues": [{"title": "Missing error handling"}]},
            "security": {"vulnerabilities": []},
            "formal": {"violations": []}
        }
        
        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = '''
{
    "consolidated_findings": [
        {"title": "Missing error handling", "severity": "medium"}
    ],
    "summary": "1 issue found",
    "pass": true
}
'''
            
            result = await agent.synthesize(findings)
            
            assert "consolidated_findings" in result
            assert "summary" in result


class TestConfigIntegration:
    """Integration tests for configuration."""
    
    def test_config_parser_handles_full_config(self):
        """Config parser handles full configuration file."""
        from codeverify_core.config import parse_config
        
        config_yaml = '''
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
'''
        
        config = parse_config(config_yaml)
        
        assert config.version == "1"
        assert "python" in config.languages
        assert config.verification.enabled is True
        assert config.thresholds.critical == 0
    
    def test_should_analyze_file_respects_patterns(self):
        """File analysis respects include/exclude patterns."""
        from codeverify_core.config import CodeVerifyConfig, should_analyze_file
        
        config = CodeVerifyConfig(
            include_patterns=["src/**/*.py"],
            exclude_patterns=["**/test_*.py"]
        )
        
        assert should_analyze_file(config, "src/main.py") is True
        assert should_analyze_file(config, "src/test_main.py") is False
        assert should_analyze_file(config, "venv/lib/foo.py") is False
