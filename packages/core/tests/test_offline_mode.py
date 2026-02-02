"""Tests for offline/air-gapped mode."""

import pytest
import tempfile
from pathlib import Path
from codeverify_core.offline_mode import (
    LocalModelConfig,
    LocalModelType,
    LocalZ3Verifier,
    OfflineAnalysisResult,
    OfflineCapability,
    OfflineModeConfig,
    OfflineModeManager,
    OllamaClient,
    get_offline_manager,
    reset_offline_manager,
)


class TestLocalModelConfig:
    """Tests for LocalModelConfig."""

    def test_creation(self):
        """Test config creation."""
        config = LocalModelConfig(
            name="codellama:7b",
            model_type=LocalModelType.OLLAMA,
            model_path="codellama:7b",
            context_length=4096,
            capabilities=[OfflineCapability.SEMANTIC_ANALYSIS],
        )
        
        assert config.name == "codellama:7b"
        assert config.model_type == LocalModelType.OLLAMA
        assert OfflineCapability.SEMANTIC_ANALYSIS in config.capabilities

    def test_default_values(self):
        """Test default configuration values."""
        config = LocalModelConfig(
            name="test",
            model_type=LocalModelType.OLLAMA,
            model_path="test",
        )
        
        assert config.context_length == 4096
        assert config.temperature == 0.1
        assert config.num_threads == 4


class TestOfflineModeConfig:
    """Tests for OfflineModeConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = OfflineModeConfig()
        
        assert config.enabled is False
        assert config.z3_timeout_ms == 30000
        assert config.sync_interval_days == 7

    def test_custom_config(self):
        """Test custom configuration."""
        config = OfflineModeConfig(
            enabled=True,
            z3_timeout_ms=60000,
            z3_memory_limit_mb=2048,
        )
        
        assert config.enabled is True
        assert config.z3_timeout_ms == 60000
        assert config.z3_memory_limit_mb == 2048


class TestLocalZ3Verifier:
    """Tests for LocalZ3Verifier."""

    @pytest.fixture
    def verifier(self):
        return LocalZ3Verifier(timeout_ms=5000)

    def test_is_available(self, verifier):
        """Test Z3 availability check."""
        # Z3 may or may not be installed
        assert isinstance(verifier.is_available, bool)

    def test_verify_null_safety_with_issue(self, verifier):
        """Test null safety verification detecting issue."""
        if not verifier.is_available:
            pytest.skip("Z3 not installed")
        
        code = "x = obj.value"
        result = verifier.verify_null_safety(code, "python")
        
        assert result["status"] == "success"
        # Should find potential null issue
        assert len(result.get("findings", [])) > 0

    def test_verify_null_safety_safe_code(self, verifier):
        """Test null safety verification with safe code."""
        if not verifier.is_available:
            pytest.skip("Z3 not installed")
        
        code = "if obj is not None:\n    x = obj.value"
        result = verifier.verify_null_safety(code, "python")
        
        assert result["status"] == "success"

    def test_verify_bounds_with_issue(self, verifier):
        """Test bounds verification detecting issue."""
        if not verifier.is_available:
            pytest.skip("Z3 not installed")
        
        code = "x = arr[i]"
        result = verifier.verify_bounds(code, "python")
        
        assert result["status"] == "success"
        assert len(result.get("findings", [])) > 0

    def test_verify_bounds_safe_code(self, verifier):
        """Test bounds verification with safe code."""
        if not verifier.is_available:
            pytest.skip("Z3 not installed")
        
        code = "if i < len(arr):\n    x = arr[i]"
        result = verifier.verify_bounds(code, "python")
        
        assert result["status"] == "success"

    def test_verify_division_with_issue(self, verifier):
        """Test division verification detecting issue."""
        if not verifier.is_available:
            pytest.skip("Z3 not installed")
        
        code = "result = x / y"
        result = verifier.verify_division(code, "python")
        
        assert result["status"] == "success"
        assert len(result.get("findings", [])) > 0

    def test_verify_division_safe_code(self, verifier):
        """Test division verification with safe code."""
        if not verifier.is_available:
            pytest.skip("Z3 not installed")
        
        code = "if y != 0:\n    result = x / y"
        result = verifier.verify_division(code, "python")
        
        assert result["status"] == "success"

    def test_verify_all(self, verifier):
        """Test running all verifications."""
        if not verifier.is_available:
            pytest.skip("Z3 not installed")
        
        code = "result = arr[i] / divisor"
        result = verifier.verify_all(code, "python")
        
        assert result["status"] == "success"
        assert "findings" in result
        assert "total_findings" in result


class TestOllamaClient:
    """Tests for OllamaClient."""

    @pytest.fixture
    def client(self):
        return OllamaClient(base_url="http://localhost:11434")

    @pytest.mark.asyncio
    async def test_is_available_when_not_running(self, client):
        """Test availability check when Ollama is not running."""
        # This test assumes Ollama is not running by default
        # In CI, Ollama won't be available
        available = await client.is_available()
        assert isinstance(available, bool)

    @pytest.mark.asyncio
    async def test_list_models_when_not_running(self, client):
        """Test listing models when not available."""
        models = await client.list_models()
        assert isinstance(models, list)


class TestOfflineModeManager:
    """Tests for OfflineModeManager."""

    @pytest.fixture
    def manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OfflineModeConfig(
                cache_dir=tmpdir + "/cache",
                models_dir=tmpdir + "/models",
            )
            yield OfflineModeManager(config)

    @pytest.mark.asyncio
    async def test_check_offline_readiness(self, manager):
        """Test checking offline readiness."""
        status = await manager.check_offline_readiness()
        
        assert "ready" in status
        assert "z3_available" in status
        assert "ollama_available" in status
        assert "models_available" in status

    @pytest.mark.asyncio
    async def test_analyze_code_offline_z3_only(self, manager):
        """Test offline analysis using Z3 only."""
        result = await manager.analyze_code_offline(
            code="result = x / y",
            language="python",
            include_llm_analysis=False,  # Skip LLM
        )
        
        assert isinstance(result, OfflineAnalysisResult)
        assert result.offline_mode is True
        
        # If Z3 is available, should have findings
        if manager.z3_verifier.is_available:
            assert "formal_verification" in result.capabilities_used

    def test_cache_result(self, manager):
        """Test caching results."""
        code = "test code"
        code_hash = manager.hash_code(code)
        
        result = OfflineAnalysisResult(
            success=True,
            findings=[{"test": "finding"}],
            trust_score=0.85,
        )
        
        manager.cache_result(code_hash, result)
        
        # Retrieve cached result
        cached = manager.get_cached_result(code_hash)
        assert cached is not None
        assert cached.success is True
        assert cached.trust_score == 0.85

    def test_get_cached_result_miss(self, manager):
        """Test cache miss."""
        result = manager.get_cached_result("nonexistent_hash")
        assert result is None

    def test_hash_code(self, manager):
        """Test code hashing."""
        code1 = "def foo(): pass"
        code2 = "def bar(): pass"
        
        hash1 = manager.hash_code(code1)
        hash2 = manager.hash_code(code2)
        
        assert hash1 != hash2
        assert len(hash1) == 16  # Should be truncated
        
        # Same code should have same hash
        assert manager.hash_code(code1) == hash1

    def test_clear_cache(self, manager):
        """Test clearing cache."""
        # Add some cached items
        manager.cache_result("hash1", OfflineAnalysisResult(success=True))
        manager.cache_result("hash2", OfflineAnalysisResult(success=True))
        
        count = manager.clear_cache()
        assert count == 2
        
        # Cache should be empty
        assert manager.get_cached_result("hash1") is None


class TestOfflineAnalysisResult:
    """Tests for OfflineAnalysisResult."""

    def test_default_values(self):
        """Test default result values."""
        result = OfflineAnalysisResult(success=False)
        
        assert result.success is False
        assert result.findings == []
        assert result.trust_score == 0.0
        assert result.offline_mode is True
        assert result.capabilities_used == []

    def test_with_findings(self):
        """Test result with findings."""
        result = OfflineAnalysisResult(
            success=True,
            findings=[
                {"category": "null_safety", "severity": "high"},
                {"category": "bounds", "severity": "medium"},
            ],
            trust_score=0.7,
            capabilities_used=["formal_verification", "semantic_analysis"],
        )
        
        assert result.success is True
        assert len(result.findings) == 2
        assert "formal_verification" in result.capabilities_used


class TestGlobalManager:
    """Tests for global manager functions."""

    def teardown_method(self):
        reset_offline_manager()

    def test_get_manager_singleton(self):
        """Test singleton pattern."""
        manager1 = get_offline_manager()
        manager2 = get_offline_manager()
        assert manager1 is manager2

    def test_reset_manager(self):
        """Test manager reset."""
        manager1 = get_offline_manager()
        reset_offline_manager()
        manager2 = get_offline_manager()
        assert manager1 is not manager2

    def test_get_manager_with_config(self):
        """Test creating manager with custom config."""
        config = OfflineModeConfig(enabled=True)
        reset_offline_manager()
        manager = get_offline_manager(config)
        assert manager.config.enabled is True


class TestModelTypes:
    """Tests for model type enums."""

    def test_local_model_types(self):
        """Test LocalModelType enum values."""
        assert LocalModelType.OLLAMA.value == "ollama"
        assert LocalModelType.LLAMA_CPP.value == "llama_cpp"
        assert LocalModelType.GGUF.value == "gguf"
        assert LocalModelType.ONNX.value == "onnx"

    def test_offline_capabilities(self):
        """Test OfflineCapability enum values."""
        assert OfflineCapability.SEMANTIC_ANALYSIS.value == "semantic_analysis"
        assert OfflineCapability.FORMAL_VERIFICATION.value == "formal_verification"
        assert OfflineCapability.CODE_FIXES.value == "code_fixes"
