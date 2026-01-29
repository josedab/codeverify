"""CodeVerify Agent SDK - Agent development framework.

This module provides:
- Agent interface (analyze, report, fix methods)
- Agent packaging format (.cvagent)
- Agent manifest and metadata
- Runtime hooks and lifecycle management
"""

import hashlib
import json
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


class AgentCapability(str, Enum):
    """Capabilities that an agent can provide."""

    ANALYZE = "analyze"  # Code analysis/verification
    REPORT = "report"  # Generate reports/findings
    FIX = "fix"  # Suggest/apply fixes
    TRANSFORM = "transform"  # Code transformation
    VALIDATE = "validate"  # Validation checks
    EXPLAIN = "explain"  # Explanations/documentation


class AgentCategory(str, Enum):
    """Category for marketplace organization."""

    SECURITY = "security"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    COMPLIANCE = "compliance"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    AI_DETECTION = "ai-detection"
    LANGUAGE_SPECIFIC = "language-specific"
    FRAMEWORK_SPECIFIC = "framework-specific"
    CUSTOM = "custom"


class AgentLanguage(str, Enum):
    """Programming languages supported by agents."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"
    CSHARP = "csharp"
    CPP = "cpp"
    RUBY = "ruby"
    PHP = "php"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    ALL = "all"


class SeverityLevel(str, Enum):
    """Severity levels for findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Finding(BaseModel):
    """A finding reported by an agent."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str
    severity: SeverityLevel
    file_path: str
    line_start: int
    line_end: int | None = None
    column_start: int | None = None
    column_end: int | None = None
    code_snippet: str | None = None
    suggested_fix: str | None = None
    fix_diff: str | None = None
    rule_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AnalysisContext(BaseModel):
    """Context provided to agents for analysis."""

    # Repository info
    repo_full_name: str
    ref: str | None = None
    commit_sha: str | None = None

    # Files to analyze
    files: dict[str, str] = Field(default_factory=dict)  # path -> content
    file_paths: list[str] = Field(default_factory=list)

    # PR context (if analyzing a PR)
    pr_number: int | None = None
    pr_title: str | None = None
    pr_description: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    diff: str | None = None

    # Additional context
    config: dict[str, Any] = Field(default_factory=dict)
    previous_findings: list[Finding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Result of an agent's analysis."""

    agent_id: str
    agent_version: str
    status: str = "success"  # success, error, timeout
    findings: list[Finding] = Field(default_factory=list)
    summary: str | None = None
    execution_time_ms: int = 0
    files_analyzed: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AgentManifest(BaseModel):
    """Manifest file for a CodeVerify agent package."""

    # Required fields
    name: str = Field(..., min_length=1, max_length=128)
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+")
    description: str = Field(..., min_length=10, max_length=1000)
    author: str = Field(..., min_length=1, max_length=256)

    # Agent metadata
    id: UUID = Field(default_factory=uuid4)
    display_name: str | None = None
    icon: str | None = None  # Base64 or URL
    homepage: str | None = None
    repository: str | None = None
    license: str = "MIT"

    # Capabilities
    capabilities: list[AgentCapability] = Field(default_factory=lambda: [AgentCapability.ANALYZE])
    category: AgentCategory = AgentCategory.CUSTOM
    languages: list[AgentLanguage] = Field(default_factory=lambda: [AgentLanguage.ALL])
    tags: list[str] = Field(default_factory=list)

    # Entry points
    entry_point: str = "agent.py"
    main_class: str = "Agent"

    # Requirements
    python_version: str = ">=3.10"
    dependencies: list[str] = Field(default_factory=list)

    # Sandbox requirements
    requires_network: bool = False
    requires_filesystem: bool = True
    max_memory_mb: int = 512
    max_cpu_seconds: int = 60

    # Pricing
    is_free: bool = True
    price_per_analysis: float | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def qualified_name(self) -> str:
        """Fully qualified agent name."""
        return f"{self.author}/{self.name}"

    def to_manifest_json(self) -> str:
        """Serialize to manifest.json format."""
        return self.model_dump_json(indent=2)


class BaseAgent(ABC):
    """Base class for CodeVerify agents.
    
    Implement this class to create custom verification agents.
    
    Example:
        ```python
        class SecurityAgent(BaseAgent):
            def analyze(self, context: AnalysisContext) -> AnalysisResult:
                findings = []
                for path, content in context.files.items():
                    if "eval(" in content:
                        findings.append(Finding(
                            title="Dangerous eval() usage",
                            description="eval() can execute arbitrary code",
                            severity=SeverityLevel.HIGH,
                            file_path=path,
                            line_start=1,
                        ))
                return AnalysisResult(
                    agent_id=self.manifest.qualified_name,
                    agent_version=self.manifest.version,
                    findings=findings,
                )
        ```
    """

    def __init__(self, manifest: AgentManifest, config: dict[str, Any] | None = None):
        self.manifest = manifest
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """
        Analyze code and return findings.
        
        Args:
            context: Analysis context with files and metadata
            
        Returns:
            AnalysisResult with findings and summary
        """
        pass

    def report(self, result: AnalysisResult) -> str:
        """
        Generate a human-readable report from analysis results.
        
        Override to customize report format.
        """
        lines = [
            f"# Analysis Report - {self.manifest.display_name or self.manifest.name}",
            f"",
            f"**Version**: {self.manifest.version}",
            f"**Status**: {result.status}",
            f"**Files Analyzed**: {result.files_analyzed}",
            f"**Findings**: {len(result.findings)}",
            f"",
        ]

        if result.summary:
            lines.extend([f"## Summary", f"", result.summary, f""])

        if result.findings:
            lines.append("## Findings")
            lines.append("")
            for i, finding in enumerate(result.findings, 1):
                lines.extend([
                    f"### {i}. [{finding.severity.value.upper()}] {finding.title}",
                    f"",
                    f"**File**: `{finding.file_path}:{finding.line_start}`",
                    f"",
                    finding.description,
                    f"",
                ])
                if finding.suggested_fix:
                    lines.extend([
                        f"**Suggested Fix**:",
                        f"```",
                        finding.suggested_fix,
                        f"```",
                        f"",
                    ])

        return "\n".join(lines)

    def fix(self, finding: Finding, context: AnalysisContext) -> str | None:
        """
        Generate a fix for a finding.
        
        Override to provide automatic fixes.
        
        Args:
            finding: The finding to fix
            context: Analysis context
            
        Returns:
            Fixed code or None if fix not available
        """
        return finding.suggested_fix

    def validate(self, context: AnalysisContext) -> bool:
        """
        Validate that the agent can analyze the given context.
        
        Override to add custom validation.
        """
        return True

    def initialize(self) -> None:
        """
        Initialize agent resources.
        
        Override to set up resources needed for analysis.
        Called once before first analysis.
        """
        self._initialized = True

    def cleanup(self) -> None:
        """
        Clean up agent resources.
        
        Override to release resources.
        Called when agent is unloaded.
        """
        self._initialized = False

    def get_config_schema(self) -> dict[str, Any]:
        """
        Return JSON Schema for agent configuration.
        
        Override to define configuration options.
        """
        return {"type": "object", "properties": {}}


# Agent packaging utilities
class AgentPackage:
    """Utility for creating and reading .cvagent packages."""

    MANIFEST_FILENAME = "manifest.json"

    @staticmethod
    def create(
        manifest: AgentManifest,
        source_dir: Path,
        output_path: Path | None = None,
    ) -> Path:
        """
        Create a .cvagent package from source directory.
        
        Args:
            manifest: Agent manifest
            source_dir: Directory containing agent source code
            output_path: Output path for .cvagent file (optional)
            
        Returns:
            Path to created .cvagent file
        """
        if output_path is None:
            output_path = source_dir.parent / f"{manifest.name}-{manifest.version}.cvagent"

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write manifest
            zf.writestr(AgentPackage.MANIFEST_FILENAME, manifest.to_manifest_json())

            # Write source files
            for file_path in source_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    arcname = file_path.relative_to(source_dir)
                    zf.write(file_path, arcname)

        return output_path

    @staticmethod
    def create_from_bytes(
        manifest: AgentManifest,
        files: dict[str, bytes],
    ) -> bytes:
        """
        Create a .cvagent package from file bytes.
        
        Args:
            manifest: Agent manifest
            files: Dictionary of filename -> content bytes
            
        Returns:
            Package bytes
        """
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(AgentPackage.MANIFEST_FILENAME, manifest.to_manifest_json())
            for name, content in files.items():
                zf.writestr(name, content)
        return buffer.getvalue()

    @staticmethod
    def read(package_path: Path) -> tuple[AgentManifest, dict[str, bytes]]:
        """
        Read a .cvagent package.
        
        Args:
            package_path: Path to .cvagent file
            
        Returns:
            Tuple of (manifest, files dict)
        """
        files = {}
        manifest = None

        with zipfile.ZipFile(package_path, "r") as zf:
            for name in zf.namelist():
                content = zf.read(name)
                if name == AgentPackage.MANIFEST_FILENAME:
                    manifest = AgentManifest.model_validate_json(content)
                else:
                    files[name] = content

        if manifest is None:
            raise ValueError(f"Missing {AgentPackage.MANIFEST_FILENAME} in package")

        return manifest, files

    @staticmethod
    def read_from_bytes(package_bytes: bytes) -> tuple[AgentManifest, dict[str, bytes]]:
        """Read a .cvagent package from bytes."""
        buffer = BytesIO(package_bytes)
        files = {}
        manifest = None

        with zipfile.ZipFile(buffer, "r") as zf:
            for name in zf.namelist():
                content = zf.read(name)
                if name == AgentPackage.MANIFEST_FILENAME:
                    manifest = AgentManifest.model_validate_json(content)
                else:
                    files[name] = content

        if manifest is None:
            raise ValueError(f"Missing {AgentPackage.MANIFEST_FILENAME} in package")

        return manifest, files

    @staticmethod
    def get_checksum(package_path: Path) -> str:
        """Calculate SHA256 checksum of package."""
        sha256 = hashlib.sha256()
        with open(package_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# Lifecycle hooks for agent runtime
class AgentLifecycle:
    """Lifecycle management for agent instances."""

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}
        self._hooks: dict[str, list[Callable]] = {
            "before_analyze": [],
            "after_analyze": [],
            "on_finding": [],
            "on_error": [],
        }

    def register(self, agent: BaseAgent) -> None:
        """Register an agent instance."""
        key = f"{agent.manifest.qualified_name}@{agent.manifest.version}"
        self._agents[key] = agent
        agent.initialize()

    def unregister(self, agent: BaseAgent) -> None:
        """Unregister an agent instance."""
        key = f"{agent.manifest.qualified_name}@{agent.manifest.version}"
        if key in self._agents:
            agent.cleanup()
            del self._agents[key]

    def get_agent(self, qualified_name: str, version: str | None = None) -> BaseAgent | None:
        """Get a registered agent."""
        if version:
            key = f"{qualified_name}@{version}"
            return self._agents.get(key)
        # Find latest version
        matching = [
            (k, a) for k, a in self._agents.items()
            if k.startswith(f"{qualified_name}@")
        ]
        if matching:
            return sorted(matching, key=lambda x: x[0])[-1][1]
        return None

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a lifecycle hook."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def run_hooks(self, event: str, *args, **kwargs) -> None:
        """Run all hooks for an event."""
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass  # Don't let hooks break execution

    async def analyze(
        self,
        agent: BaseAgent,
        context: AnalysisContext,
    ) -> AnalysisResult:
        """Run analysis with lifecycle hooks."""
        import time

        self.run_hooks("before_analyze", agent, context)
        start_time = time.time()

        try:
            result = agent.analyze(context)
            result.execution_time_ms = int((time.time() - start_time) * 1000)

            for finding in result.findings:
                self.run_hooks("on_finding", agent, finding)

            self.run_hooks("after_analyze", agent, context, result)
            return result

        except Exception as e:
            self.run_hooks("on_error", agent, context, e)
            return AnalysisResult(
                agent_id=agent.manifest.qualified_name,
                agent_version=agent.manifest.version,
                status="error",
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )


# Global lifecycle instance
agent_lifecycle = AgentLifecycle()


# Decorators for agent development
T = TypeVar("T", bound=BaseAgent)


def agent(
    name: str,
    version: str,
    author: str,
    description: str,
    **kwargs: Any,
) -> Callable[[type[T]], type[T]]:
    """
    Decorator to define an agent class with manifest.
    
    Example:
        ```python
        @agent(
            name="security-scanner",
            version="1.0.0",
            author="codeverify",
            description="Scans for security vulnerabilities",
            category=AgentCategory.SECURITY,
            languages=[AgentLanguage.PYTHON, AgentLanguage.TYPESCRIPT],
        )
        class SecurityScanner(BaseAgent):
            def analyze(self, context):
                ...
        ```
    """
    def decorator(cls: type[T]) -> type[T]:
        manifest = AgentManifest(
            name=name,
            version=version,
            author=author,
            description=description,
            main_class=cls.__name__,
            **kwargs,
        )

        # Store manifest on class
        cls._manifest = manifest

        # Wrap __init__ to inject manifest
        original_init = cls.__init__

        def new_init(self, config: dict[str, Any] | None = None):
            original_init(self, cls._manifest, config)

        cls.__init__ = new_init
        return cls

    return decorator
