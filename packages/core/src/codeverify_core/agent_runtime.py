"""CodeVerify Agent Runtime - Sandboxed execution environment.

This module provides:
- Sandboxed agent execution (process isolation)
- Resource limits (memory, CPU, filesystem)
- Agent loading and instantiation
- Security boundaries
"""

import asyncio
import importlib.util
import multiprocessing
import os
import resource
import signal
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from codeverify_core.agent_sdk import (
    AgentManifest,
    AgentPackage,
    AnalysisContext,
    AnalysisResult,
    BaseAgent,
)

logger = structlog.get_logger()


@dataclass
class SandboxConfig:
    """Configuration for agent sandbox."""

    max_memory_mb: int = 512
    max_cpu_seconds: int = 60
    max_file_size_mb: int = 10
    max_open_files: int = 64
    allowed_imports: list[str] | None = None
    blocked_imports: list[str] | None = None
    network_access: bool = False
    filesystem_read: bool = True
    filesystem_write: bool = False
    temp_dir: Path | None = None


class SandboxError(Exception):
    """Error during sandboxed execution."""

    pass


class AgentLoadError(Exception):
    """Error loading an agent."""

    pass


class ResourceLimitExceeded(SandboxError):
    """Agent exceeded resource limits."""

    pass


class SecurityViolation(SandboxError):
    """Agent violated security policy."""

    pass


def _set_resource_limits(config: SandboxConfig) -> None:
    """Set resource limits for the current process (Unix only)."""
    if sys.platform == "win32":
        return

    # Memory limit
    memory_bytes = config.max_memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, resource.error):
        pass  # May not be supported

    # CPU time limit
    try:
        resource.setrlimit(
            resource.RLIMIT_CPU, (config.max_cpu_seconds, config.max_cpu_seconds)
        )
    except (ValueError, resource.error):
        pass

    # File size limit
    file_bytes = config.max_file_size_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    except (ValueError, resource.error):
        pass

    # Open file limit
    try:
        resource.setrlimit(
            resource.RLIMIT_NOFILE, (config.max_open_files, config.max_open_files)
        )
    except (ValueError, resource.error):
        pass


def _create_restricted_builtins(config: SandboxConfig) -> dict[str, Any]:
    """Create a restricted set of builtins for agent execution."""
    import builtins

    safe_builtins = {}

    # Allow safe builtins
    safe_names = [
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "globals",
        "hasattr",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "locals",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "vars",
        "zip",
        "__import__",
        "Exception",
        "BaseException",
        "TypeError",
        "ValueError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "StopIteration",
        "True",
        "False",
        "None",
    ]

    for name in safe_names:
        if hasattr(builtins, name):
            safe_builtins[name] = getattr(builtins, name)

    # Wrap __import__ to restrict imports
    original_import = builtins.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Check blocked imports
        if config.blocked_imports:
            for blocked in config.blocked_imports:
                if name == blocked or name.startswith(f"{blocked}."):
                    raise ImportError(f"Import of '{name}' is not allowed")

        # Check allowed imports
        if config.allowed_imports:
            allowed = False
            for pattern in config.allowed_imports:
                if name == pattern or name.startswith(f"{pattern}."):
                    allowed = True
                    break
            if not allowed:
                raise ImportError(f"Import of '{name}' is not in allowlist")

        # Block dangerous modules by default
        dangerous_modules = [
            "subprocess",
            "os.system",
            "os.popen",
            "os.spawn",
            "commands",
            "popen2",
            "socket",
            "asyncio.subprocess",
            "multiprocessing",
            "ctypes",
            "cffi",
        ]

        if not config.network_access:
            dangerous_modules.extend(["socket", "ssl", "urllib", "http", "requests", "aiohttp"])

        for dangerous in dangerous_modules:
            if name == dangerous or name.startswith(f"{dangerous}."):
                raise ImportError(f"Import of '{name}' is not allowed for security")

        return original_import(name, globals, locals, fromlist, level)

    safe_builtins["__import__"] = restricted_import

    # Block file operations if not allowed
    if not config.filesystem_read:
        safe_builtins.pop("open", None)

    return safe_builtins


class AgentSandbox:
    """Sandboxed execution environment for agents."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._temp_dirs: list[Path] = []

    def _create_temp_dir(self) -> Path:
        """Create a temporary directory for agent execution."""
        if self.config.temp_dir:
            temp_dir = self.config.temp_dir / f"agent_{os.getpid()}_{time.time_ns()}"
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="cvagent_"))
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def _cleanup_temp_dirs(self) -> None:
        """Clean up temporary directories."""
        import shutil

        for temp_dir in self._temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        self._temp_dirs.clear()

    def load_agent(
        self,
        manifest: AgentManifest,
        source_files: dict[str, bytes],
    ) -> BaseAgent:
        """
        Load an agent from source files.
        
        Args:
            manifest: Agent manifest
            source_files: Dictionary of filename -> content bytes
            
        Returns:
            Loaded agent instance
        """
        # Create temp directory for agent files
        temp_dir = self._create_temp_dir()

        try:
            # Write source files
            for name, content in source_files.items():
                file_path = temp_dir / name
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(content)

            # Find entry point
            entry_path = temp_dir / manifest.entry_point
            if not entry_path.exists():
                raise AgentLoadError(f"Entry point not found: {manifest.entry_point}")

            # Load module
            spec = importlib.util.spec_from_file_location(
                f"cvagent_{manifest.name}", entry_path
            )
            if spec is None or spec.loader is None:
                raise AgentLoadError(f"Failed to load module: {manifest.entry_point}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module

            # Execute module
            spec.loader.exec_module(module)

            # Get agent class
            if not hasattr(module, manifest.main_class):
                raise AgentLoadError(f"Agent class not found: {manifest.main_class}")

            agent_class = getattr(module, manifest.main_class)

            if not issubclass(agent_class, BaseAgent):
                raise AgentLoadError(f"{manifest.main_class} must inherit from BaseAgent")

            # Instantiate agent
            agent = agent_class(manifest)
            return agent

        except AgentLoadError:
            raise
        except Exception as e:
            raise AgentLoadError(f"Failed to load agent: {e}")

    def load_from_package(self, package_path: Path) -> BaseAgent:
        """Load an agent from a .cvagent package file."""
        manifest, files = AgentPackage.read(package_path)
        return self.load_agent(manifest, files)

    def load_from_package_bytes(self, package_bytes: bytes) -> BaseAgent:
        """Load an agent from package bytes."""
        manifest, files = AgentPackage.read_from_bytes(package_bytes)
        return self.load_agent(manifest, files)


def _run_agent_in_process(
    manifest_json: str,
    source_files: dict[str, bytes],
    context_json: str,
    config_dict: dict[str, Any],
    result_queue: multiprocessing.Queue,
) -> None:
    """Run agent analysis in a separate process (target for multiprocessing)."""
    try:
        # Set resource limits
        config = SandboxConfig(**config_dict)
        _set_resource_limits(config)

        # Load manifest and context
        manifest = AgentManifest.model_validate_json(manifest_json)
        context = AnalysisContext.model_validate_json(context_json)

        # Load and run agent
        sandbox = AgentSandbox(config)
        agent = sandbox.load_agent(manifest, source_files)
        agent.initialize()

        try:
            result = agent.analyze(context)
            result_queue.put(("success", result.model_dump_json()))
        finally:
            agent.cleanup()
            sandbox._cleanup_temp_dirs()

    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


class IsolatedAgentRunner:
    """Run agents in isolated processes with resource limits."""

    def __init__(self, sandbox_config: SandboxConfig | None = None):
        self.config = sandbox_config or SandboxConfig()

    async def run(
        self,
        manifest: AgentManifest,
        source_files: dict[str, bytes],
        context: AnalysisContext,
    ) -> AnalysisResult:
        """
        Run an agent in an isolated process.
        
        Args:
            manifest: Agent manifest
            source_files: Agent source files
            context: Analysis context
            
        Returns:
            Analysis result
        """
        # Override sandbox config with manifest requirements
        config = SandboxConfig(
            max_memory_mb=min(self.config.max_memory_mb, manifest.max_memory_mb),
            max_cpu_seconds=min(self.config.max_cpu_seconds, manifest.max_cpu_seconds),
            network_access=manifest.requires_network and self.config.network_access,
            filesystem_read=manifest.requires_filesystem and self.config.filesystem_read,
        )

        # Create result queue
        result_queue = multiprocessing.Queue()

        # Start process
        process = multiprocessing.Process(
            target=_run_agent_in_process,
            args=(
                manifest.model_dump_json(),
                source_files,
                context.model_dump_json(),
                {
                    "max_memory_mb": config.max_memory_mb,
                    "max_cpu_seconds": config.max_cpu_seconds,
                    "network_access": config.network_access,
                    "filesystem_read": config.filesystem_read,
                    "filesystem_write": config.filesystem_write,
                },
                result_queue,
            ),
        )

        start_time = time.time()
        process.start()

        # Wait for result with timeout
        timeout = config.max_cpu_seconds + 10  # Extra buffer
        try:
            # Poll for result
            while process.is_alive() and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
                try:
                    status, data = result_queue.get_nowait()
                    if status == "success":
                        return AnalysisResult.model_validate_json(data)
                    else:
                        return AnalysisResult(
                            agent_id=manifest.qualified_name,
                            agent_version=manifest.version,
                            status="error",
                            error=data,
                            execution_time_ms=int((time.time() - start_time) * 1000),
                        )
                except Exception:
                    pass

            # Check if still running (timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()

                return AnalysisResult(
                    agent_id=manifest.qualified_name,
                    agent_version=manifest.version,
                    status="timeout",
                    error=f"Agent exceeded time limit of {config.max_cpu_seconds}s",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )

            # Get final result
            try:
                status, data = result_queue.get_nowait()
                if status == "success":
                    return AnalysisResult.model_validate_json(data)
                else:
                    return AnalysisResult(
                        agent_id=manifest.qualified_name,
                        agent_version=manifest.version,
                        status="error",
                        error=data,
                        execution_time_ms=int((time.time() - start_time) * 1000),
                    )
            except Exception:
                return AnalysisResult(
                    agent_id=manifest.qualified_name,
                    agent_version=manifest.version,
                    status="error",
                    error="Agent process exited without result",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )

        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)


# Convenience function for running agents
async def run_agent(
    package_path: Path | None = None,
    package_bytes: bytes | None = None,
    manifest: AgentManifest | None = None,
    source_files: dict[str, bytes] | None = None,
    context: AnalysisContext | None = None,
    sandbox_config: SandboxConfig | None = None,
    isolated: bool = True,
) -> AnalysisResult:
    """
    Convenience function to run an agent.
    
    Args:
        package_path: Path to .cvagent package
        package_bytes: Package bytes (alternative to path)
        manifest: Agent manifest (if not using package)
        source_files: Agent source files (if not using package)
        context: Analysis context
        sandbox_config: Sandbox configuration
        isolated: Run in isolated process (default True)
        
    Returns:
        Analysis result
    """
    if context is None:
        raise ValueError("context is required")

    # Load from package if provided
    if package_path:
        manifest, source_files = AgentPackage.read(package_path)
    elif package_bytes:
        manifest, source_files = AgentPackage.read_from_bytes(package_bytes)
    elif manifest is None or source_files is None:
        raise ValueError("Either package or (manifest, source_files) required")

    if isolated:
        runner = IsolatedAgentRunner(sandbox_config)
        return await runner.run(manifest, source_files, context)
    else:
        # Run in current process (for development/testing)
        sandbox = AgentSandbox(sandbox_config)
        agent = sandbox.load_agent(manifest, source_files)
        agent.initialize()
        try:
            start_time = time.time()
            result = agent.analyze(context)
            result.execution_time_ms = int((time.time() - start_time) * 1000)
            return result
        finally:
            agent.cleanup()
            sandbox._cleanup_temp_dirs()
