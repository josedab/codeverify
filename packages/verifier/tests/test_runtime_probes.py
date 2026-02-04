"""Tests for Runtime Verification Probes."""

import pytest
from datetime import datetime, timedelta

from codeverify_verifier.runtime_probes import (
    MonitorMode,
    ProbeConfig,
    ProbeGenerator,
    ProbeType,
    RuntimeMonitor,
    RuntimeSpec,
    RuntimeVerificationReport,
    SpecViolation,
    check,
    create_monitor,
    runtime_invariant,
    runtime_postcondition,
    runtime_precondition,
)


class TestRuntimeSpec:
    """Tests for RuntimeSpec compilation."""
    
    def test_compile_simple_condition(self):
        spec = RuntimeSpec(
            id="test-1",
            name="test_spec",
            description="Test",
            probe_type=ProbeType.PRECONDITION,
            function_name="test_func",
            condition="x > 0",
            parameters=["x"],
        )
        
        spec.compile()
        
        assert spec.compiled_condition is not None
        assert spec.compiled_condition(x=5) is True
        assert spec.compiled_condition(x=-1) is False
    
    def test_compile_multiple_params(self):
        spec = RuntimeSpec(
            id="test-2",
            name="test_multi",
            description="Test multiple params",
            probe_type=ProbeType.PRECONDITION,
            function_name="test_func",
            condition="x < y",
            parameters=["x", "y"],
        )
        
        spec.compile()
        
        assert spec.compiled_condition(x=1, y=5) is True
        assert spec.compiled_condition(x=10, y=5) is False
    
    def test_compile_with_len(self):
        spec = RuntimeSpec(
            id="test-3",
            name="test_len",
            description="Test len()",
            probe_type=ProbeType.PRECONDITION,
            function_name="test_func",
            condition="len(items) > 0",
            parameters=["items"],
        )
        
        spec.compile()
        
        assert spec.compiled_condition(items=[1, 2, 3]) is True
        assert spec.compiled_condition(items=[]) is False
    
    def test_compile_complex_condition(self):
        spec = RuntimeSpec(
            id="test-4",
            name="test_complex",
            description="Complex condition",
            probe_type=ProbeType.POSTCONDITION,
            function_name="test_func",
            condition="result is not None and result >= 0",
            parameters=["result"],
        )
        
        spec.compile()
        
        assert spec.compiled_condition(result=5) is True
        assert spec.compiled_condition(result=None) is False
        assert spec.compiled_condition(result=-1) is False


class TestRuntimeMonitor:
    """Tests for RuntimeMonitor."""
    
    @pytest.fixture(autouse=True)
    def reset_monitor(self):
        """Reset the singleton before each test."""
        RuntimeMonitor.reset()
        yield
        RuntimeMonitor.reset()
    
    def test_singleton_pattern(self):
        m1 = RuntimeMonitor.get_instance()
        m2 = RuntimeMonitor.get_instance()
        assert m1 is m2
    
    def test_register_spec(self):
        monitor = RuntimeMonitor.get_instance()
        spec = RuntimeSpec(
            id="reg-test",
            name="registration_test",
            description="Test",
            probe_type=ProbeType.ASSERTION,
            function_name="test",
            condition="True",
            parameters=[],
        )
        
        monitor.register_spec(spec)
        
        assert "reg-test" in monitor.specs
        assert monitor.specs["reg-test"].compiled_condition is not None
    
    def test_check_spec_passes(self):
        monitor = RuntimeMonitor.get_instance()
        spec = RuntimeSpec(
            id="check-pass",
            name="passing_check",
            description="Test",
            probe_type=ProbeType.ASSERTION,
            function_name="test",
            condition="x > 0",
            parameters=["x"],
        )
        monitor.register_spec(spec)
        
        result = monitor.check_spec("check-pass", x=5)
        
        assert result is True
        assert len(monitor.violations) == 0
    
    def test_check_spec_fails_log_mode(self):
        monitor = RuntimeMonitor(ProbeConfig(mode=MonitorMode.LOG))
        RuntimeMonitor._instance = monitor
        
        spec = RuntimeSpec(
            id="check-fail",
            name="failing_check",
            description="Test",
            probe_type=ProbeType.ASSERTION,
            function_name="test",
            condition="x > 0",
            parameters=["x"],
        )
        monitor.register_spec(spec)
        
        result = monitor.check_spec("check-fail", x=-1)
        
        assert result is False
        assert len(monitor.violations) == 1
        assert monitor.violations[0].spec_id == "check-fail"
    
    def test_check_spec_fails_enforce_mode(self):
        monitor = RuntimeMonitor(ProbeConfig(mode=MonitorMode.ENFORCE))
        RuntimeMonitor._instance = monitor
        
        spec = RuntimeSpec(
            id="enforce-fail",
            name="enforce_check",
            description="Test",
            probe_type=ProbeType.ASSERTION,
            function_name="test",
            condition="x > 0",
            parameters=["x"],
        )
        monitor.register_spec(spec)
        
        with pytest.raises(SpecViolation) as exc_info:
            monitor.check_spec("enforce-fail", x=-1)
        
        assert exc_info.value.spec_id == "enforce-fail"
        assert exc_info.value.inputs == {"x": -1}
    
    def test_check_spec_off_mode(self):
        monitor = RuntimeMonitor(ProbeConfig(mode=MonitorMode.OFF))
        RuntimeMonitor._instance = monitor
        
        spec = RuntimeSpec(
            id="off-test",
            name="off_check",
            description="Test",
            probe_type=ProbeType.ASSERTION,
            function_name="test",
            condition="x > 0",  # Would fail
            parameters=["x"],
        )
        monitor.register_spec(spec)
        
        result = monitor.check_spec("off-test", x=-1)
        
        # Should return True (pass) even though condition fails
        assert result is True
        assert len(monitor.violations) == 0
    
    def test_get_stats(self):
        monitor = RuntimeMonitor.get_instance()
        spec1 = RuntimeSpec(
            id="stats-1", name="s1", description="", 
            probe_type=ProbeType.ASSERTION, function_name="f1", condition="True"
        )
        spec2 = RuntimeSpec(
            id="stats-2", name="s2", description="", 
            probe_type=ProbeType.ASSERTION, function_name="f2", condition="True",
            enabled=False
        )
        monitor.register_spec(spec1)
        monitor.register_spec(spec2)
        
        stats = monitor.get_stats()
        
        assert stats["total_specs"] == 2
        assert stats["enabled_specs"] == 1


class TestDecorators:
    """Tests for runtime decorator functions."""
    
    @pytest.fixture(autouse=True)
    def reset_monitor(self):
        RuntimeMonitor.reset()
        yield
        RuntimeMonitor.reset()
    
    def test_runtime_precondition_passes(self):
        @runtime_precondition("x > 0", name="positive_x")
        def sqrt_approx(x: float) -> float:
            return x ** 0.5
        
        # Should not raise
        result = sqrt_approx(4)
        assert result == 2.0
    
    def test_runtime_precondition_fails_log_mode(self):
        monitor = RuntimeMonitor(ProbeConfig(mode=MonitorMode.LOG))
        RuntimeMonitor._instance = monitor
        
        @runtime_precondition("x > 0")
        def sqrt_approx(x: float) -> float:
            return abs(x) ** 0.5
        
        # Should not raise in LOG mode
        result = sqrt_approx(-4)
        assert result == 2.0
        assert len(monitor.violations) == 1
    
    def test_runtime_postcondition_passes(self):
        @runtime_postcondition("result >= 0")
        def abs_value(x: float) -> float:
            return abs(x)
        
        result = abs_value(-5)
        assert result == 5
    
    def test_runtime_postcondition_fails(self):
        monitor = RuntimeMonitor(ProbeConfig(mode=MonitorMode.LOG))
        RuntimeMonitor._instance = monitor
        
        @runtime_postcondition("result >= 0")
        def buggy_abs(x: float) -> float:
            return x  # Bug: doesn't actually take abs
        
        buggy_abs(-5)
        assert len(monitor.violations) == 1
    
    def test_runtime_invariant(self):
        monitor = RuntimeMonitor(ProbeConfig(mode=MonitorMode.LOG))
        RuntimeMonitor._instance = monitor
        
        class Counter:
            def __init__(self):
                self.value = 0
            
            @runtime_invariant("self.value >= 0")
            def decrement(self):
                self.value -= 1
        
        c = Counter()
        c.decrement()  # value becomes -1, violates invariant
        
        # Should have logged violations (checked before and after)
        # Before: value=0 >= 0 (passes)
        # After: value=-1 >= 0 (fails)
        assert any(v.probe_type == ProbeType.INVARIANT for v in monitor.violations)


class TestProbeGenerator:
    """Tests for ProbeGenerator."""
    
    def test_z3_to_python_simple(self):
        gen = ProbeGenerator()
        
        result = gen._z3_to_python("x > 0")
        assert result == "x > 0"
    
    def test_z3_to_python_and(self):
        gen = ProbeGenerator()
        
        result = gen._z3_to_python("And(x > 0, y > 0)")
        # Note: The simple replacement gives "(x > 0 and y > 0)"
        assert "and" in result
    
    def test_z3_to_python_implies(self):
        gen = ProbeGenerator()
        
        result = gen._z3_to_python("Implies(x > 0, result > 0)")
        assert "not" in result or "or" in result
    
    def test_from_z3_spec(self):
        gen = ProbeGenerator()
        
        spec = gen.from_z3_spec(
            z3_spec="x >= 0",
            function_name="sqrt",
            parameters=["x"],
        )
        
        assert spec is not None
        assert spec.function_name == "sqrt"
        assert "x" in spec.parameters
        assert len(gen.generated_probes) == 1
    
    def test_generate_python_decorator(self):
        gen = ProbeGenerator()
        spec = RuntimeSpec(
            id="test",
            name="test_spec",
            description="Test",
            probe_type=ProbeType.PRECONDITION,
            function_name="test",
            condition="x > 0",
            parameters=["x"],
        )
        
        decorator = gen.generate_python_decorator(spec)
        
        assert "@runtime_precondition" in decorator
        assert "x > 0" in decorator
    
    def test_generate_typescript_middleware(self):
        gen = ProbeGenerator()
        spec = RuntimeSpec(
            id="test",
            name="test_spec",
            description="Test",
            probe_type=ProbeType.PRECONDITION,
            function_name="test",
            condition="x > 0",
            parameters=["x"],
        )
        
        middleware = gen.generate_typescript_middleware(spec)
        
        assert "function testGuard" in middleware
        assert "Request" in middleware


class TestRuntimeVerificationReport:
    """Tests for RuntimeVerificationReport."""
    
    @pytest.fixture(autouse=True)
    def reset_monitor(self):
        RuntimeMonitor.reset()
        yield
        RuntimeMonitor.reset()
    
    def test_generate_report_empty(self):
        monitor = RuntimeMonitor.get_instance()
        report = RuntimeVerificationReport(monitor)
        
        result = report.generate_report()
        
        assert result["summary"]["total_specs"] == 0
        assert result["summary"]["total_violations"] == 0
        assert result["summary"]["health_percentage"] == 100
    
    def test_generate_report_with_violations(self):
        monitor = RuntimeMonitor(ProbeConfig(mode=MonitorMode.LOG))
        RuntimeMonitor._instance = monitor
        
        spec = RuntimeSpec(
            id="report-test",
            name="report_spec",
            description="Test",
            probe_type=ProbeType.ASSERTION,
            function_name="test",
            condition="x > 0",
            parameters=["x"],
        )
        monitor.register_spec(spec)
        
        # Generate some violations
        monitor.check_spec("report-test", x=-1)
        monitor.check_spec("report-test", x=-2)
        
        report = RuntimeVerificationReport(monitor)
        result = report.generate_report()
        
        assert result["summary"]["total_specs"] == 1
        assert result["summary"]["total_violations"] == 2
        assert result["summary"]["specs_with_violations"] == 1
        assert "report-test" in result["violations_by_spec"]


class TestCheckFunction:
    """Tests for the check() convenience function."""
    
    def test_check_passes(self):
        result = check("x > 0 and y > 0", x=5, y=10)
        assert result is True
    
    def test_check_fails(self):
        result = check("x > 0", x=-5)
        assert result is False
    
    def test_check_complex(self):
        result = check("len(items) > 0 and all(isinstance(i, int) for i in items)", items=[1, 2, 3])
        assert result is True


class TestCreateMonitor:
    """Tests for create_monitor function."""
    
    @pytest.fixture(autouse=True)
    def reset_monitor(self):
        RuntimeMonitor.reset()
        yield
        RuntimeMonitor.reset()
    
    def test_create_default_monitor(self):
        monitor = create_monitor()
        assert monitor is RuntimeMonitor.get_instance()
    
    def test_create_with_config(self):
        config = ProbeConfig(mode=MonitorMode.ENFORCE, sample_rate=0.5)
        monitor = create_monitor(config)
        
        assert monitor.config.mode == MonitorMode.ENFORCE
        assert monitor.config.sample_rate == 0.5


class TestSpecViolation:
    """Tests for SpecViolation exception."""
    
    def test_violation_attributes(self):
        exc = SpecViolation(
            message="Test violation",
            spec_id="spec-123",
            function_name="test_func",
            inputs={"x": 5},
            expected=True,
            actual=False,
        )
        
        assert exc.spec_id == "spec-123"
        assert exc.function_name == "test_func"
        assert exc.inputs == {"x": 5}
        assert exc.expected is True
        assert exc.actual is False
        assert str(exc) == "Test violation"
