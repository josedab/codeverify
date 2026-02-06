"""Zero-Config AI Model - Local fine-tuned model for offline verification.

This module provides a fine-tuned CodeVerify-specific LLM that works offline,
reducing latency and eliminating API costs for core verification tasks.

Key features:
1. Local Inference: Run verification without external API calls
2. Hybrid Routing: Route simple queries to local, complex to cloud
3. Model Management: Download, update, and manage local models
4. Quantization Support: Optimized models for different hardware
"""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import structlog

logger = structlog.get_logger()


class ModelSize(str, Enum):
    """Size of the local model."""

    TINY = "tiny"  # ~500MB, fastest, basic analysis
    SMALL = "small"  # ~2GB, good balance
    MEDIUM = "medium"  # ~4GB, higher quality
    LARGE = "large"  # ~8GB, best quality


class ModelStatus(str, Enum):
    """Status of a local model."""

    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    READY = "ready"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class InferenceBackend(str, Enum):
    """Backend for model inference."""

    ONNX = "onnx"  # Cross-platform, good performance
    LLAMA_CPP = "llama_cpp"  # Optimized for LLMs
    TRANSFORMERS = "transformers"  # HuggingFace transformers
    OLLAMA = "ollama"  # Ollama API


class QueryComplexity(str, Enum):
    """Complexity level of a verification query."""

    SIMPLE = "simple"  # Pattern matching, basic checks
    MODERATE = "moderate"  # Semantic understanding needed
    COMPLEX = "complex"  # Requires deep analysis
    EXPERT = "expert"  # Requires cloud LLM


@dataclass
class ModelInfo:
    """Information about a local model."""

    id: str
    name: str
    size: ModelSize
    version: str
    download_url: str
    file_size_mb: int
    capabilities: list[str]
    min_ram_gb: float
    backend: InferenceBackend
    quantization: str | None = None  # "q4_0", "q8_0", "f16", etc.
    checksum: str | None = None
    description: str = ""


@dataclass
class LocalModelConfig:
    """Configuration for local model inference."""

    model_dir: str = ""
    preferred_size: ModelSize = ModelSize.SMALL
    preferred_backend: InferenceBackend = InferenceBackend.LLAMA_CPP
    max_tokens: int = 2048
    temperature: float = 0.1
    num_threads: int = 4
    use_gpu: bool = False
    gpu_layers: int = 0


@dataclass
class InferenceResult:
    """Result from local model inference."""

    success: bool
    content: str
    tokens_used: int = 0
    latency_ms: float = 0
    model_id: str = ""
    from_cache: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "model_id": self.model_id,
            "from_cache": self.from_cache,
            "error": self.error,
        }


@dataclass
class RoutingDecision:
    """Decision about where to route a query."""

    use_local: bool
    complexity: QueryComplexity
    reason: str
    estimated_quality_local: float  # 0-1
    estimated_latency_local_ms: float
    estimated_latency_cloud_ms: float


# Available models
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    "codeverify-tiny": ModelInfo(
        id="codeverify-tiny",
        name="CodeVerify Tiny",
        size=ModelSize.TINY,
        version="1.0.0",
        download_url="https://models.codeverify.io/codeverify-tiny-q4.gguf",
        file_size_mb=500,
        capabilities=["null_check", "bounds_check", "syntax"],
        min_ram_gb=1.0,
        backend=InferenceBackend.LLAMA_CPP,
        quantization="q4_0",
        description="Fast model for basic pattern detection",
    ),
    "codeverify-small": ModelInfo(
        id="codeverify-small",
        name="CodeVerify Small",
        size=ModelSize.SMALL,
        version="1.0.0",
        download_url="https://models.codeverify.io/codeverify-small-q4.gguf",
        file_size_mb=2000,
        capabilities=["null_check", "bounds_check", "security", "semantic"],
        min_ram_gb=4.0,
        backend=InferenceBackend.LLAMA_CPP,
        quantization="q4_0",
        description="Balanced model for most verification tasks",
    ),
    "codeverify-medium": ModelInfo(
        id="codeverify-medium",
        name="CodeVerify Medium",
        size=ModelSize.MEDIUM,
        version="1.0.0",
        download_url="https://models.codeverify.io/codeverify-medium-q8.gguf",
        file_size_mb=4000,
        capabilities=["null_check", "bounds_check", "security", "semantic", "formal"],
        min_ram_gb=8.0,
        backend=InferenceBackend.LLAMA_CPP,
        quantization="q8_0",
        description="High quality model for complex analysis",
    ),
}


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""

    @abstractmethod
    async def load_model(self, model_path: str, config: LocalModelConfig) -> bool:
        """Load a model."""
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the current model."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> InferenceResult:
        """Generate a response."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        pass


class LlamaCppEngine(InferenceEngine):
    """Inference engine using llama.cpp."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_path: str = ""

    async def load_model(self, model_path: str, config: LocalModelConfig) -> bool:
        """Load a GGUF model using llama.cpp."""
        try:
            # Import llama-cpp-python
            from llama_cpp import Llama

            self._model = Llama(
                model_path=model_path,
                n_ctx=config.max_tokens * 2,
                n_threads=config.num_threads,
                n_gpu_layers=config.gpu_layers if config.use_gpu else 0,
                verbose=False,
            )
            self._model_path = model_path

            logger.info("Model loaded", model_path=model_path)
            return True

        except ImportError:
            logger.error("llama-cpp-python not installed")
            return False
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            return False

    async def unload_model(self) -> None:
        """Unload the model."""
        self._model = None
        self._model_path = ""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> InferenceResult:
        """Generate using llama.cpp."""
        if not self._model:
            return InferenceResult(
                success=False,
                content="",
                error="Model not loaded",
            )

        import time
        start_time = time.time()

        try:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "\n\n\n"],
            )

            elapsed_ms = (time.time() - start_time) * 1000
            content = output["choices"][0]["text"] if output["choices"] else ""
            tokens = output.get("usage", {}).get("total_tokens", 0)

            return InferenceResult(
                success=True,
                content=content.strip(),
                tokens_used=tokens,
                latency_ms=elapsed_ms,
                model_id=Path(self._model_path).stem,
            )

        except Exception as e:
            logger.error("Generation failed", error=str(e))
            return InferenceResult(
                success=False,
                content="",
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def is_loaded(self) -> bool:
        return self._model is not None


class OllamaEngine(InferenceEngine):
    """Inference engine using Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url
        self._model_name: str = ""

    async def load_model(self, model_path: str, config: LocalModelConfig) -> bool:
        """'Load' a model (Ollama manages this)."""
        import httpx

        try:
            # Just check if Ollama is running and model exists
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self._base_url}/api/tags")
                if response.status_code == 200:
                    self._model_name = model_path  # model_path is actually model name for Ollama
                    return True
        except Exception as e:
            logger.error("Failed to connect to Ollama", error=str(e))

        return False

    async def unload_model(self) -> None:
        """Unload model (no-op for Ollama)."""
        self._model_name = ""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> InferenceResult:
        """Generate using Ollama API."""
        import httpx
        import time

        if not self._model_name:
            return InferenceResult(
                success=False,
                content="",
                error="Model not loaded",
            )

        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model_name,
                        "prompt": prompt,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                        "stream": False,
                    },
                    timeout=60.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    elapsed_ms = (time.time() - start_time) * 1000

                    return InferenceResult(
                        success=True,
                        content=data.get("response", "").strip(),
                        tokens_used=data.get("eval_count", 0),
                        latency_ms=elapsed_ms,
                        model_id=self._model_name,
                    )
                else:
                    return InferenceResult(
                        success=False,
                        content="",
                        error=f"Ollama error: {response.status_code}",
                    )

        except Exception as e:
            logger.error("Ollama generation failed", error=str(e))
            return InferenceResult(
                success=False,
                content="",
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def is_loaded(self) -> bool:
        return bool(self._model_name)


class QueryRouter:
    """Routes queries between local and cloud models."""

    # Keywords that indicate different complexity levels
    SIMPLE_KEYWORDS = ["null", "none", "undefined", "bounds", "index", "syntax"]
    MODERATE_KEYWORDS = ["security", "injection", "xss", "leak", "thread"]
    COMPLEX_KEYWORDS = ["semantic", "intent", "refactor", "architecture", "design"]

    def __init__(
        self,
        local_capabilities: list[str],
        prefer_local: bool = True,
        quality_threshold: float = 0.7,
    ) -> None:
        self._local_capabilities = set(local_capabilities)
        self._prefer_local = prefer_local
        self._quality_threshold = quality_threshold
        self._history: list[dict[str, Any]] = []

    def route(self, query: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        """Decide whether to use local or cloud model."""
        complexity = self._assess_complexity(query)
        estimated_quality = self._estimate_local_quality(query, complexity)

        # Check if local model can handle this
        can_handle_locally = self._can_handle_locally(query, context)

        # Decision logic
        if complexity == QueryComplexity.EXPERT:
            # Always use cloud for expert queries
            use_local = False
            reason = "Query requires expert-level analysis"
        elif complexity == QueryComplexity.COMPLEX:
            # Use cloud unless quality is very high locally
            use_local = can_handle_locally and estimated_quality > 0.8
            reason = "Complex query" + (" handled locally" if use_local else " routed to cloud")
        elif self._prefer_local and can_handle_locally:
            use_local = True
            reason = "Local model capable, preferring local"
        else:
            use_local = estimated_quality >= self._quality_threshold
            reason = "Based on quality threshold"

        decision = RoutingDecision(
            use_local=use_local,
            complexity=complexity,
            reason=reason,
            estimated_quality_local=estimated_quality,
            estimated_latency_local_ms=50 if complexity == QueryComplexity.SIMPLE else 200,
            estimated_latency_cloud_ms=1000,
        )

        # Record for learning
        self._history.append({
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],
            "complexity": complexity.value,
            "use_local": use_local,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return decision

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess the complexity of a query."""
        query_lower = query.lower()

        # Count complexity indicators
        simple_count = sum(1 for kw in self.SIMPLE_KEYWORDS if kw in query_lower)
        moderate_count = sum(1 for kw in self.MODERATE_KEYWORDS if kw in query_lower)
        complex_count = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in query_lower)

        # Check query length and structure
        is_long = len(query) > 2000
        has_code = "```" in query or "def " in query or "function " in query

        if complex_count > 0 or (is_long and moderate_count > 0):
            return QueryComplexity.COMPLEX
        elif moderate_count > 0 or (is_long and simple_count > 0):
            return QueryComplexity.MODERATE
        elif simple_count > 0:
            return QueryComplexity.SIMPLE
        else:
            # Default based on length
            return QueryComplexity.MODERATE if is_long else QueryComplexity.SIMPLE

    def _estimate_local_quality(self, query: str, complexity: QueryComplexity) -> float:
        """Estimate quality of local model for this query."""
        base_quality = {
            QueryComplexity.SIMPLE: 0.95,
            QueryComplexity.MODERATE: 0.75,
            QueryComplexity.COMPLEX: 0.5,
            QueryComplexity.EXPERT: 0.2,
        }

        quality = base_quality[complexity]

        # Adjust based on capabilities
        query_lower = query.lower()
        for capability in self._local_capabilities:
            if capability in query_lower:
                quality = min(quality + 0.1, 1.0)

        return quality

    def _can_handle_locally(self, query: str, context: dict[str, Any] | None) -> bool:
        """Check if local model has required capabilities."""
        query_lower = query.lower()

        # Required capabilities based on query content
        required = []
        if "null" in query_lower or "none" in query_lower:
            required.append("null_check")
        if "bounds" in query_lower or "index" in query_lower:
            required.append("bounds_check")
        if "security" in query_lower or "inject" in query_lower:
            required.append("security")
        if "semantic" in query_lower or "meaning" in query_lower:
            required.append("semantic")

        # Check if all required capabilities are available
        return all(cap in self._local_capabilities for cap in required)


class ModelManager:
    """Manages downloading and updating local models."""

    def __init__(self, model_dir: str | None = None) -> None:
        self._model_dir = model_dir or os.path.expanduser("~/.codeverify/models")
        self._download_progress: dict[str, float] = {}
        Path(self._model_dir).mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_id: str) -> str | None:
        """Get path to a downloaded model."""
        model_info = AVAILABLE_MODELS.get(model_id)
        if not model_info:
            return None

        model_file = Path(self._model_dir) / f"{model_id}.gguf"
        if model_file.exists():
            return str(model_file)
        return None

    def is_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded."""
        return self.get_model_path(model_id) is not None

    def get_status(self, model_id: str) -> ModelStatus:
        """Get status of a model."""
        if model_id in self._download_progress:
            return ModelStatus.DOWNLOADING

        if self.is_downloaded(model_id):
            return ModelStatus.READY

        return ModelStatus.NOT_DOWNLOADED

    async def download_model(
        self,
        model_id: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> bool:
        """Download a model."""
        model_info = AVAILABLE_MODELS.get(model_id)
        if not model_info:
            logger.error("Unknown model", model_id=model_id)
            return False

        # In a real implementation, this would download from model_info.download_url
        # For now, we simulate the download
        model_path = Path(self._model_dir) / f"{model_id}.gguf"

        logger.info("Starting model download", model_id=model_id)
        self._download_progress[model_id] = 0.0

        try:
            # Simulate download progress
            for i in range(10):
                await asyncio.sleep(0.1)  # Simulated delay
                progress = (i + 1) / 10
                self._download_progress[model_id] = progress
                if progress_callback:
                    progress_callback(progress)

            # In reality, download and write the file
            # For simulation, create a placeholder
            model_path.write_text(f"PLACEHOLDER FOR {model_id}")

            del self._download_progress[model_id]
            logger.info("Model downloaded", model_id=model_id, path=str(model_path))
            return True

        except Exception as e:
            logger.error("Download failed", model_id=model_id, error=str(e))
            self._download_progress.pop(model_id, None)
            return False

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model."""
        model_path = self.get_model_path(model_id)
        if model_path:
            Path(model_path).unlink()
            return True
        return False

    def list_models(self) -> list[dict[str, Any]]:
        """List all available models with their status."""
        result = []
        for model_id, info in AVAILABLE_MODELS.items():
            result.append({
                "id": model_id,
                "name": info.name,
                "size": info.size.value,
                "file_size_mb": info.file_size_mb,
                "status": self.get_status(model_id).value,
                "capabilities": info.capabilities,
                "description": info.description,
            })
        return result


class ZeroConfigAIModel:
    """Main interface for zero-config local AI verification.

    Usage:
        model = ZeroConfigAIModel()

        # Ensure model is ready
        await model.ensure_ready()

        # Analyze code
        result = await model.analyze(code, "python")

        # Or with hybrid routing (auto-fallback to cloud)
        result = await model.analyze_hybrid(code, "python", cloud_fallback=True)
    """

    def __init__(self, config: LocalModelConfig | None = None) -> None:
        self._config = config or LocalModelConfig()

        # Set default model directory
        if not self._config.model_dir:
            self._config.model_dir = os.path.expanduser("~/.codeverify/models")

        self._model_manager = ModelManager(self._config.model_dir)
        self._engine: InferenceEngine | None = None
        self._router: QueryRouter | None = None
        self._loaded_model_id: str | None = None
        self._cache: dict[str, InferenceResult] = {}

    async def ensure_ready(
        self,
        model_id: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> bool:
        """Ensure a model is downloaded and loaded."""
        # Determine which model to use
        if model_id is None:
            # Find best available model for preferred size
            for mid, info in AVAILABLE_MODELS.items():
                if info.size == self._config.preferred_size:
                    model_id = mid
                    break

            if model_id is None:
                model_id = "codeverify-small"  # Default

        # Download if needed
        if not self._model_manager.is_downloaded(model_id):
            logger.info("Downloading model", model_id=model_id)
            if not await self._model_manager.download_model(model_id, progress_callback):
                return False

        # Load the model
        return await self._load_model(model_id)

    async def _load_model(self, model_id: str) -> bool:
        """Load a model into memory."""
        model_path = self._model_manager.get_model_path(model_id)
        if not model_path:
            return False

        model_info = AVAILABLE_MODELS.get(model_id)
        if not model_info:
            return False

        # Create appropriate engine
        if model_info.backend == InferenceBackend.LLAMA_CPP:
            self._engine = LlamaCppEngine()
        elif model_info.backend == InferenceBackend.OLLAMA:
            self._engine = OllamaEngine()
        else:
            logger.error("Unsupported backend", backend=model_info.backend)
            return False

        # Load model
        if await self._engine.load_model(model_path, self._config):
            self._loaded_model_id = model_id
            self._router = QueryRouter(
                local_capabilities=model_info.capabilities,
                prefer_local=True,
            )
            return True

        return False

    async def analyze(
        self,
        code: str,
        language: str,
        analysis_type: str = "all",
    ) -> InferenceResult:
        """Analyze code using local model."""
        if not self._engine or not self._engine.is_loaded():
            return InferenceResult(
                success=False,
                content="",
                error="Model not loaded. Call ensure_ready() first.",
            )

        # Build prompt
        prompt = self._build_analysis_prompt(code, language, analysis_type)

        # Check cache
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self._cache:
            result = self._cache[cache_key]
            result.from_cache = True
            return result

        # Generate
        result = await self._engine.generate(
            prompt,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        # Cache successful results
        if result.success:
            self._cache[cache_key] = result

        return result

    async def analyze_hybrid(
        self,
        code: str,
        language: str,
        analysis_type: str = "all",
        cloud_fallback: bool = True,
        cloud_provider: Any = None,  # Optional cloud LLM provider
    ) -> InferenceResult:
        """Analyze with hybrid local/cloud routing."""
        # Route the query
        prompt = self._build_analysis_prompt(code, language, analysis_type)

        if self._router:
            decision = self._router.route(prompt)

            if decision.use_local:
                result = await self.analyze(code, language, analysis_type)
                if result.success:
                    return result

                # Fall through to cloud if local failed
                if not cloud_fallback:
                    return result

            # Use cloud
            if cloud_provider:
                # In a real implementation, call the cloud provider
                logger.info("Routing to cloud", reason=decision.reason)
                # return await cloud_provider.analyze(code, language, analysis_type)

        # Default to local
        return await self.analyze(code, language, analysis_type)

    def _build_analysis_prompt(
        self,
        code: str,
        language: str,
        analysis_type: str,
    ) -> str:
        """Build prompt for code analysis."""
        prompt = f"""You are a code verification assistant. Analyze the following {language} code for potential issues.

Analysis type: {analysis_type}

Code:
```{language}
{code}
```

Provide a JSON response with the following structure:
{{
    "findings": [
        {{
            "type": "issue type (null_safety, bounds_check, security, etc.)",
            "severity": "critical/high/medium/low",
            "line": line_number,
            "message": "description of the issue",
            "fix": "suggested fix (if applicable)"
        }}
    ],
    "summary": "overall assessment",
    "confidence": 0.0-1.0
}}

Response:"""

        return prompt

    def is_ready(self) -> bool:
        """Check if the model is ready for inference."""
        return self._engine is not None and self._engine.is_loaded()

    def get_loaded_model(self) -> str | None:
        """Get the ID of the currently loaded model."""
        return self._loaded_model_id

    async def unload(self) -> None:
        """Unload the current model."""
        if self._engine:
            await self._engine.unload_model()
        self._loaded_model_id = None
        self._router = None

    def clear_cache(self) -> None:
        """Clear the inference cache."""
        self._cache.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "model_loaded": self._loaded_model_id,
            "is_ready": self.is_ready(),
            "cache_size": len(self._cache),
            "available_models": self._model_manager.list_models(),
        }


# Singleton instance
_local_model: ZeroConfigAIModel | None = None


def get_local_model(config: LocalModelConfig | None = None) -> ZeroConfigAIModel:
    """Get the global local model instance."""
    global _local_model
    if _local_model is None:
        _local_model = ZeroConfigAIModel(config)
    return _local_model


def reset_local_model() -> None:
    """Reset the global local model (for testing)."""
    global _local_model
    _local_model = None
