"""Tests for health_diagnostics module."""

from __future__ import annotations

from codeverify_core.health_diagnostics import (
    ComponentHealth,
    DiagnosticReport,
    HealthChecker,
    HealthStatus,
    check_dependencies,
    check_python_version,
    check_z3_available,
    create_default_checker,
)


class TestComponentHealth:
    def test_healthy_component(self):
        h = ComponentHealth(name="test", status=HealthStatus.HEALTHY)
        assert h.status == HealthStatus.HEALTHY

    def test_unhealthy_component(self):
        h = ComponentHealth(
            name="test",
            status=HealthStatus.UNHEALTHY,
            message="down",
        )
        assert h.message == "down"


class TestDiagnosticReport:
    def test_healthy_counts(self):
        report = DiagnosticReport(
            overall_status=HealthStatus.HEALTHY,
            components=[
                ComponentHealth("a", HealthStatus.HEALTHY),
                ComponentHealth("b", HealthStatus.HEALTHY),
                ComponentHealth("c", HealthStatus.UNHEALTHY),
            ],
        )
        assert report.healthy_count == 2
        assert report.unhealthy_count == 1

    def test_to_dict(self):
        report = DiagnosticReport(
            overall_status=HealthStatus.HEALTHY,
            components=[ComponentHealth("a", HealthStatus.HEALTHY)],
        )
        d = report.to_dict()
        assert d["status"] == "healthy"
        assert d["summary"]["total"] == 1
        assert d["summary"]["healthy"] == 1


class TestHealthChecker:
    def test_register_and_check(self):
        checker = HealthChecker()
        checker.register("test", lambda: ComponentHealth("test", HealthStatus.HEALTHY))
        result = checker.check("test")
        assert result.status == HealthStatus.HEALTHY

    def test_unknown_check(self):
        checker = HealthChecker()
        result = checker.check("nonexistent")
        assert result.status == HealthStatus.UNKNOWN

    def test_check_exception_handled(self):
        def bad_check():
            raise RuntimeError("boom")

        checker = HealthChecker()
        checker.register("bad", bad_check)
        result = checker.check("bad")
        assert result.status == HealthStatus.UNHEALTHY
        assert "boom" in result.message

    def test_run_all_healthy(self):
        checker = HealthChecker()
        checker.register("a", lambda: ComponentHealth("a", HealthStatus.HEALTHY))
        checker.register("b", lambda: ComponentHealth("b", HealthStatus.HEALTHY))
        report = checker.run_all()
        assert report.overall_status == HealthStatus.HEALTHY
        assert len(report.components) == 2

    def test_run_all_unhealthy(self):
        checker = HealthChecker()
        checker.register("good", lambda: ComponentHealth("good", HealthStatus.HEALTHY))
        checker.register("bad", lambda: ComponentHealth("bad", HealthStatus.UNHEALTHY))
        report = checker.run_all()
        assert report.overall_status == HealthStatus.UNHEALTHY

    def test_run_all_degraded(self):
        checker = HealthChecker()
        checker.register("ok", lambda: ComponentHealth("ok", HealthStatus.HEALTHY))
        checker.register("meh", lambda: ComponentHealth("meh", HealthStatus.DEGRADED))
        report = checker.run_all()
        assert report.overall_status == HealthStatus.DEGRADED

    def test_empty_checker(self):
        checker = HealthChecker()
        report = checker.run_all()
        assert report.overall_status == HealthStatus.UNKNOWN

    def test_registered_checks_list(self):
        checker = HealthChecker()
        checker.register("a", lambda: ComponentHealth("a", HealthStatus.HEALTHY))
        checker.register("b", lambda: ComponentHealth("b", HealthStatus.HEALTHY))
        assert set(checker.registered_checks) == {"a", "b"}

    def test_latency_measured(self):
        import time

        def slow_check():
            time.sleep(0.01)
            return ComponentHealth("slow", HealthStatus.HEALTHY)

        checker = HealthChecker()
        checker.register("slow", slow_check)
        result = checker.check("slow")
        assert result.latency_ms >= 5  # At least 5ms

    def test_system_info_populated(self):
        checker = HealthChecker()
        checker.register("a", lambda: ComponentHealth("a", HealthStatus.HEALTHY))
        report = checker.run_all()
        assert "python_version" in report.system_info
        assert "platform" in report.system_info


class TestBuiltInChecks:
    def test_python_version_check(self):
        result = check_python_version()
        assert result.status == HealthStatus.HEALTHY
        assert "3.1" in result.message

    def test_dependencies_check(self):
        result = check_dependencies()
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def test_z3_check(self):
        result = check_z3_available()
        # Z3 may or may not be installed
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.UNHEALTHY)

    def test_default_checker(self):
        checker = create_default_checker()
        assert len(checker.registered_checks) >= 3
        report = checker.run_all()
        assert report.overall_status in HealthStatus
