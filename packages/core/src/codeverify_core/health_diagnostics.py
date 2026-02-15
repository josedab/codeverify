"""Health Check & Diagnostics.

System health checks for Z3 solver, LLM APIs, Redis, database,
with a self-test runner and diagnostic report generation.
"""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health check result for a single component."""

    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Full system diagnostic report."""

    overall_status: HealthStatus
    components: list[ComponentHealth] = field(default_factory=list)
    system_info: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def healthy_count(self) -> int:
        return sum(1 for c in self.components if c.status == HealthStatus.HEALTHY)

    @property
    def unhealthy_count(self) -> int:
        return sum(1 for c in self.components if c.status == HealthStatus.UNHEALTHY)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.overall_status.value,
            "timestamp": self.timestamp,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": c.latency_ms,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.components
            ],
            "system_info": self.system_info,
            "summary": {
                "total": len(self.components),
                "healthy": self.healthy_count,
                "unhealthy": self.unhealthy_count,
            },
        }


HealthCheckFn = Callable[[], ComponentHealth]


class HealthChecker:
    """System health checker with pluggable component checks.

    Usage:
        checker = HealthChecker()
        checker.register("z3", check_z3_solver)
        checker.register("llm_api", check_llm_api)
        report = checker.run_all()
    """

    def __init__(self):
        self._checks: dict[str, HealthCheckFn] = {}

    def register(self, name: str, check_fn: HealthCheckFn) -> None:
        """Register a health check function."""
        self._checks[name] = check_fn

    def check(self, name: str) -> ComponentHealth:
        """Run a single health check by name."""
        fn = self._checks.get(name)
        if fn is None:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"No check registered for '{name}'",
            )
        try:
            start = time.time()
            result = fn()
            result.latency_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
            )

    def run_all(self) -> DiagnosticReport:
        """Run all registered health checks and produce a diagnostic report."""
        components: list[ComponentHealth] = []
        for name in self._checks:
            components.append(self.check(name))

        # Determine overall status
        if not components:
            overall = HealthStatus.UNKNOWN
        elif all(c.status == HealthStatus.HEALTHY for c in components):
            overall = HealthStatus.HEALTHY
        elif any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        return DiagnosticReport(
            overall_status=overall,
            components=components,
            system_info=_collect_system_info(),
        )

    @property
    def registered_checks(self) -> list[str]:
        return list(self._checks.keys())


def _collect_system_info() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }


# =============================================================================
# Built-in health checks
# =============================================================================


def check_z3_available() -> ComponentHealth:
    """Check if Z3 solver is importable and functional."""
    try:
        import z3

        solver = z3.Solver()
        x = z3.Int("x")
        solver.add(x > 0, x < 10)
        result = solver.check()
        return ComponentHealth(
            name="z3_solver",
            status=HealthStatus.HEALTHY,
            message=f"Z3 v{z3.get_version_string()} operational",
            details={"version": z3.get_version_string(), "sat_result": str(result)},
        )
    except ImportError:
        return ComponentHealth(
            name="z3_solver",
            status=HealthStatus.UNHEALTHY,
            message="Z3 not installed",
        )
    except Exception as e:
        return ComponentHealth(
            name="z3_solver",
            status=HealthStatus.UNHEALTHY,
            message=f"Z3 check failed: {e}",
        )


def check_python_version() -> ComponentHealth:
    """Check Python version meets minimum requirements."""
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 11):
        return ComponentHealth(
            name="python_version",
            status=HealthStatus.HEALTHY,
            message=f"Python {major}.{minor} meets minimum 3.11",
            details={"version": f"{major}.{minor}", "minimum": "3.11"},
        )
    else:
        return ComponentHealth(
            name="python_version",
            status=HealthStatus.UNHEALTHY,
            message=f"Python {major}.{minor} below minimum 3.11",
            details={"version": f"{major}.{minor}", "minimum": "3.11"},
        )


def check_dependencies() -> ComponentHealth:
    """Check that core dependencies are importable."""
    required = ["structlog", "pydantic", "yaml"]
    missing: list[str] = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return ComponentHealth(
            name="dependencies",
            status=HealthStatus.HEALTHY,
            message="All core dependencies available",
        )
    else:
        return ComponentHealth(
            name="dependencies",
            status=HealthStatus.DEGRADED,
            message=f"Missing: {', '.join(missing)}",
            details={"missing": missing},
        )


def create_default_checker() -> HealthChecker:
    """Create a health checker with built-in checks."""
    checker = HealthChecker()
    checker.register("z3_solver", check_z3_available)
    checker.register("python_version", check_python_version)
    checker.register("dependencies", check_dependencies)
    return checker
