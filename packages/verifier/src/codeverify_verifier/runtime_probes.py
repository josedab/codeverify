"""Runtime Verification Probes - Generate runtime monitors from Z3 specs.

This module bridges static verification → runtime assurance by generating
lightweight runtime monitors that validate code behavior in production.

Unique value: Formal methods + observability for safety-critical deployments.
"""

import ast
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger()

F = TypeVar("F", bound=Callable[..., Any])


class SpecViolation(Exception):
    """Raised when a runtime specification is violated."""
    
    def __init__(
        self,
        message: str,
        spec_id: str,
        function_name: str,
        inputs: dict[str, Any],
        expected: Any,
        actual: Any,
    ) -> None:
        super().__init__(message)
        self.spec_id = spec_id
        self.function_name = function_name
        self.inputs = inputs
        self.expected = expected
        self.actual = actual


class ProbeType(str, Enum):
    """Type of runtime probe."""
    
    PRECONDITION = "precondition"  # Check before function execution
    POSTCONDITION = "postcondition"  # Check after function execution
    INVARIANT = "invariant"  # Check during execution
    ASSERTION = "assertion"  # Point-in-time check
    TYPE_GUARD = "type_guard"  # Type constraint check


class MonitorMode(str, Enum):
    """How to handle spec violations."""
    
    ENFORCE = "enforce"  # Raise exception on violation
    LOG = "log"  # Log violation but continue
    SAMPLE = "sample"  # Probabilistic checking
    OFF = "off"  # Disabled (no-op)


@dataclass
class SpecViolationEvent:
    """Recorded specification violation event."""
    
    spec_id: str
    probe_type: ProbeType
    function_name: str
    timestamp: datetime
    inputs: dict[str, Any]
    expected: str
    actual: str
    stack_trace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "probe_type": self.probe_type.value,
            "function_name": self.function_name,
            "timestamp": self.timestamp.isoformat(),
            "inputs": {k: repr(v) for k, v in self.inputs.items()},
            "expected": self.expected,
            "actual": self.actual,
            "stack_trace": self.stack_trace,
            "metadata": self.metadata,
        }


@dataclass
class RuntimeSpec:
    """A specification to be checked at runtime."""
    
    id: str
    name: str
    description: str
    probe_type: ProbeType
    function_name: str
    condition: str  # Python expression or lambda string
    compiled_condition: Callable[..., bool] | None = None
    parameters: list[str] = field(default_factory=list)
    enabled: bool = True
    sample_rate: float = 1.0  # For SAMPLE mode
    
    def compile(self) -> None:
        """Compile the condition string to a callable."""
        if self.compiled_condition is not None:
            return
        
        try:
            # Security: Only allow safe operations
            safe_names = {"len", "abs", "min", "max", "sum", "all", "any", "isinstance", "type"}
            
            # Create a restricted namespace
            namespace: dict[str, Any] = {
                name: getattr(__builtins__ if hasattr(__builtins__, name) else __builtins__, name, None)
                for name in safe_names
            }
            namespace["True"] = True
            namespace["False"] = False
            namespace["None"] = None
            
            # Compile as lambda if it's an expression
            if not self.condition.startswith("lambda"):
                params = ", ".join(self.parameters) if self.parameters else "_"
                lambda_str = f"lambda {params}: {self.condition}"
            else:
                lambda_str = self.condition
            
            self.compiled_condition = eval(lambda_str, namespace)
            
        except Exception as e:
            logger.error(f"Failed to compile spec {self.id}: {e}")
            raise


@dataclass
class ProbeConfig:
    """Configuration for runtime probes."""
    
    mode: MonitorMode = MonitorMode.LOG
    sample_rate: float = 1.0  # 1.0 = check everything, 0.1 = check 10%
    max_violations_per_minute: int = 100
    apm_integration: str | None = None  # "datadog", "newrelic", "prometheus"
    webhook_url: str | None = None  # For real-time alerts


class RuntimeMonitor:
    """Central monitor for all runtime probes.
    
    Integrates with APM tools and provides a dashboard of violations.
    """
    
    _instance: "RuntimeMonitor | None" = None
    
    def __init__(self, config: ProbeConfig | None = None) -> None:
        self.config = config or ProbeConfig()
        self.specs: dict[str, RuntimeSpec] = {}
        self.violations: list[SpecViolationEvent] = []
        self._violation_count_minute: dict[str, int] = {}
        self._last_minute_reset: float = time.time()
    
    @classmethod
    def get_instance(cls) -> "RuntimeMonitor":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton for testing."""
        cls._instance = None
    
    def register_spec(self, spec: RuntimeSpec) -> None:
        """Register a specification for monitoring."""
        spec.compile()
        self.specs[spec.id] = spec
        logger.info(f"Registered runtime spec: {spec.id}")
    
    def unregister_spec(self, spec_id: str) -> None:
        """Unregister a specification."""
        if spec_id in self.specs:
            del self.specs[spec_id]
    
    def check_spec(
        self,
        spec_id: str,
        **kwargs: Any,
    ) -> bool:
        """Check a spec and record any violations."""
        spec = self.specs.get(spec_id)
        if not spec or not spec.enabled:
            return True
        
        if self.config.mode == MonitorMode.OFF:
            return True
        
        # Rate limiting
        if self._should_skip_sample(spec):
            return True
        
        try:
            if spec.compiled_condition is None:
                spec.compile()
            
            # Execute the condition
            result = spec.compiled_condition(**kwargs) if kwargs else spec.compiled_condition()
            
            if not result:
                self._record_violation(spec, kwargs, expected=True, actual=result)
                
                if self.config.mode == MonitorMode.ENFORCE:
                    raise SpecViolation(
                        message=f"Specification '{spec.name}' violated",
                        spec_id=spec.id,
                        function_name=spec.function_name,
                        inputs=kwargs,
                        expected=True,
                        actual=result,
                    )
            
            return result
            
        except SpecViolation:
            raise
        except Exception as e:
            logger.error(f"Error checking spec {spec_id}: {e}")
            return True  # Fail open
    
    def _should_skip_sample(self, spec: RuntimeSpec) -> bool:
        """Determine if we should skip this check based on sampling."""
        import random
        
        effective_rate = min(spec.sample_rate, self.config.sample_rate)
        if self.config.mode == MonitorMode.SAMPLE:
            return random.random() > effective_rate
        return False
    
    def _record_violation(
        self,
        spec: RuntimeSpec,
        inputs: dict[str, Any],
        expected: Any,
        actual: Any,
    ) -> None:
        """Record a specification violation."""
        import traceback
        
        # Rate limiting per spec
        self._reset_minute_counter_if_needed()
        count = self._violation_count_minute.get(spec.id, 0)
        if count >= self.config.max_violations_per_minute:
            return
        self._violation_count_minute[spec.id] = count + 1
        
        event = SpecViolationEvent(
            spec_id=spec.id,
            probe_type=spec.probe_type,
            function_name=spec.function_name,
            timestamp=datetime.utcnow(),
            inputs=inputs,
            expected=str(expected),
            actual=str(actual),
            stack_trace=traceback.format_stack()[-5:-1] if traceback else None,
        )
        
        self.violations.append(event)
        
        # Log the violation
        logger.warning(
            "Specification violation",
            spec_id=spec.id,
            function=spec.function_name,
            probe_type=spec.probe_type.value,
        )
        
        # Send to APM if configured
        self._send_to_apm(event)
    
    def _reset_minute_counter_if_needed(self) -> None:
        """Reset per-minute violation counters."""
        now = time.time()
        if now - self._last_minute_reset > 60:
            self._violation_count_minute.clear()
            self._last_minute_reset = now
    
    def _send_to_apm(self, event: SpecViolationEvent) -> None:
        """Send violation event to APM integration."""
        if not self.config.apm_integration:
            return
        
        # Placeholder for APM integrations
        if self.config.apm_integration == "datadog":
            self._send_to_datadog(event)
        elif self.config.apm_integration == "newrelic":
            self._send_to_newrelic(event)
        elif self.config.apm_integration == "prometheus":
            self._send_to_prometheus(event)
    
    def _send_to_datadog(self, event: SpecViolationEvent) -> None:
        """Send event to Datadog."""
        # In production, use datadog library
        logger.debug("Would send to Datadog", event=event.to_dict())
    
    def _send_to_newrelic(self, event: SpecViolationEvent) -> None:
        """Send event to New Relic."""
        logger.debug("Would send to New Relic", event=event.to_dict())
    
    def _send_to_prometheus(self, event: SpecViolationEvent) -> None:
        """Send event to Prometheus."""
        logger.debug("Would send to Prometheus", event=event.to_dict())
    
    def get_violations(
        self,
        spec_id: str | None = None,
        since: datetime | None = None,
    ) -> list[SpecViolationEvent]:
        """Get recorded violations."""
        violations = self.violations
        
        if spec_id:
            violations = [v for v in violations if v.spec_id == spec_id]
        
        if since:
            violations = [v for v in violations if v.timestamp >= since]
        
        return violations
    
    def get_stats(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "total_specs": len(self.specs),
            "enabled_specs": sum(1 for s in self.specs.values() if s.enabled),
            "total_violations": len(self.violations),
            "violations_by_spec": {
                spec_id: sum(1 for v in self.violations if v.spec_id == spec_id)
                for spec_id in self.specs
            },
            "mode": self.config.mode.value,
        }


def runtime_precondition(
    condition: str,
    name: str | None = None,
    mode: MonitorMode = MonitorMode.LOG,
) -> Callable[[F], F]:
    """Decorator to add a runtime precondition check.
    
    Args:
        condition: Python expression that must evaluate to True
        name: Human-readable name for the spec
        mode: How to handle violations
        
    Example:
        @runtime_precondition("x > 0", name="positive_input")
        def sqrt(x: float) -> float:
            return x ** 0.5
    """
    def decorator(func: F) -> F:
        # Generate unique spec ID
        spec_id = hashlib.md5(
            f"{func.__module__}.{func.__name__}.pre.{condition}".encode()
        ).hexdigest()[:12]
        
        # Get parameter names from function signature
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # Register the spec
        spec = RuntimeSpec(
            id=spec_id,
            name=name or f"precondition_{func.__name__}",
            description=f"Precondition for {func.__name__}: {condition}",
            probe_type=ProbeType.PRECONDITION,
            function_name=func.__name__,
            condition=condition,
            parameters=params,
        )
        RuntimeMonitor.get_instance().register_spec(spec)
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build kwargs from args
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            RuntimeMonitor.get_instance().check_spec(spec_id, **bound.arguments)
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


def runtime_postcondition(
    condition: str,
    name: str | None = None,
    mode: MonitorMode = MonitorMode.LOG,
) -> Callable[[F], F]:
    """Decorator to add a runtime postcondition check.
    
    Args:
        condition: Python expression with access to 'result' and input params
        name: Human-readable name for the spec
        mode: How to handle violations
        
    Example:
        @runtime_postcondition("result >= 0", name="non_negative_result")
        def abs_value(x: float) -> float:
            return abs(x)
    """
    def decorator(func: F) -> F:
        spec_id = hashlib.md5(
            f"{func.__module__}.{func.__name__}.post.{condition}".encode()
        ).hexdigest()[:12]
        
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys()) + ["result"]
        
        spec = RuntimeSpec(
            id=spec_id,
            name=name or f"postcondition_{func.__name__}",
            description=f"Postcondition for {func.__name__}: {condition}",
            probe_type=ProbeType.POSTCONDITION,
            function_name=func.__name__,
            condition=condition,
            parameters=params,
        )
        RuntimeMonitor.get_instance().register_spec(spec)
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            check_kwargs = {**bound.arguments, "result": result}
            RuntimeMonitor.get_instance().check_spec(spec_id, **check_kwargs)
            
            return result
        
        return wrapper  # type: ignore
    
    return decorator


def runtime_invariant(
    condition: str,
    name: str | None = None,
) -> Callable[[F], F]:
    """Decorator for class methods to check invariants.
    
    Checks condition on 'self' before and after method execution.
    
    Example:
        class BankAccount:
            @runtime_invariant("self.balance >= 0")
            def withdraw(self, amount):
                self.balance -= amount
    """
    def decorator(func: F) -> F:
        spec_id = hashlib.md5(
            f"{func.__module__}.{func.__name__}.inv.{condition}".encode()
        ).hexdigest()[:12]
        
        spec = RuntimeSpec(
            id=spec_id,
            name=name or f"invariant_{func.__name__}",
            description=f"Invariant for {func.__name__}: {condition}",
            probe_type=ProbeType.INVARIANT,
            function_name=func.__name__,
            condition=condition,
            parameters=["self"],
        )
        RuntimeMonitor.get_instance().register_spec(spec)
        
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Check invariant before
            RuntimeMonitor.get_instance().check_spec(spec_id, self=self)
            
            result = func(self, *args, **kwargs)
            
            # Check invariant after
            RuntimeMonitor.get_instance().check_spec(spec_id, self=self)
            
            return result
        
        return wrapper  # type: ignore
    
    return decorator


class ProbeGenerator:
    """Generates runtime probes from Z3 specifications."""
    
    def __init__(self) -> None:
        self.generated_probes: list[RuntimeSpec] = []
    
    def from_z3_spec(
        self,
        z3_spec: str,
        function_name: str,
        parameters: list[str],
    ) -> RuntimeSpec:
        """Convert a Z3 specification to a runtime probe.
        
        Args:
            z3_spec: Z3 assertion string
            function_name: Name of the function being specified
            parameters: Parameter names
            
        Returns:
            RuntimeSpec that can be registered with the monitor
        """
        # Convert Z3 syntax to Python
        python_condition = self._z3_to_python(z3_spec)
        
        spec_id = hashlib.md5(
            f"{function_name}.{z3_spec}".encode()
        ).hexdigest()[:12]
        
        spec = RuntimeSpec(
            id=spec_id,
            name=f"z3_spec_{function_name}",
            description=f"Generated from Z3: {z3_spec}",
            probe_type=ProbeType.ASSERTION,
            function_name=function_name,
            condition=python_condition,
            parameters=parameters,
        )
        
        self.generated_probes.append(spec)
        return spec
    
    def _z3_to_python(self, z3_spec: str) -> str:
        """Convert Z3 syntax to Python expression."""
        # Handle common Z3 constructs
        result = z3_spec
        
        # Z3 And/Or/Not -> Python and/or/not
        result = result.replace("And(", "(")
        result = result.replace("Or(", "(")
        result = result.replace("Not(", "not (")
        result = result.replace(", ", " and ")
        
        # Z3 comparisons are similar to Python
        result = result.replace("==", "==")
        result = result.replace("!=", "!=")
        result = result.replace(">=", ">=")
        result = result.replace("<=", "<=")
        result = result.replace(">", ">")
        result = result.replace("<", "<")
        
        # Handle Implies(a, b) -> (not a) or b
        import re
        implies_pattern = r"Implies\(([^,]+),\s*([^)]+)\)"
        while re.search(implies_pattern, result):
            result = re.sub(implies_pattern, r"(not (\1) or (\2))", result)
        
        # Handle ForAll/Exists (simplified: assume single variable)
        forall_pattern = r"ForAll\((\w+),\s*([^)]+)\)"
        result = re.sub(forall_pattern, r"all(\2 for \1 in range(len(\1)))", result)
        
        return result
    
    def generate_python_decorator(self, spec: RuntimeSpec) -> str:
        """Generate Python code with decorator for a spec."""
        if spec.probe_type == ProbeType.PRECONDITION:
            decorator = "runtime_precondition"
        elif spec.probe_type == ProbeType.POSTCONDITION:
            decorator = "runtime_postcondition"
        elif spec.probe_type == ProbeType.INVARIANT:
            decorator = "runtime_invariant"
        else:
            decorator = "runtime_precondition"  # Default
        
        return f'''@{decorator}("{spec.condition}", name="{spec.name}")'''
    
    def generate_typescript_middleware(self, spec: RuntimeSpec) -> str:
        """Generate TypeScript middleware for a spec."""
        # Convert condition to TypeScript
        ts_condition = self._python_to_typescript(spec.condition)
        
        return f'''
// Generated from spec: {spec.name}
export function {spec.function_name}Guard(req: Request, res: Response, next: NextFunction) {{
    const valid = {ts_condition};
    if (!valid) {{
        console.warn('Specification violation: {spec.name}');
        // Choose: return error or continue
        // return res.status(400).json({{ error: 'Specification violation' }});
    }}
    next();
}}
'''
    
    def _python_to_typescript(self, python_expr: str) -> str:
        """Convert Python expression to TypeScript."""
        result = python_expr
        result = result.replace(" and ", " && ")
        result = result.replace(" or ", " || ")
        result = result.replace("not ", "!")
        result = result.replace("None", "null")
        result = result.replace("True", "true")
        result = result.replace("False", "false")
        result = result.replace("len(", ".length of ")  # Simplified
        return result


class RuntimeVerificationReport:
    """Generates reports on runtime verification status."""
    
    def __init__(self, monitor: RuntimeMonitor) -> None:
        self.monitor = monitor
    
    def generate_report(
        self,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive runtime verification report."""
        violations = self.monitor.get_violations(since=since)
        stats = self.monitor.get_stats()
        
        # Group violations by spec
        violations_by_spec: dict[str, list[SpecViolationEvent]] = {}
        for v in violations:
            if v.spec_id not in violations_by_spec:
                violations_by_spec[v.spec_id] = []
            violations_by_spec[v.spec_id].append(v)
        
        # Calculate health metrics
        total_specs = stats["total_specs"]
        specs_with_violations = len(violations_by_spec)
        healthy_specs = total_specs - specs_with_violations
        
        return {
            "summary": {
                "total_specs": total_specs,
                "specs_with_violations": specs_with_violations,
                "healthy_specs": healthy_specs,
                "health_percentage": (
                    healthy_specs / total_specs * 100 if total_specs > 0 else 100
                ),
                "total_violations": len(violations),
            },
            "violations_by_spec": {
                spec_id: {
                    "count": len(events),
                    "latest": events[-1].to_dict() if events else None,
                    "spec_name": self.monitor.specs.get(spec_id, RuntimeSpec(
                        id=spec_id, name="unknown", description="", 
                        probe_type=ProbeType.ASSERTION, function_name=""
                    )).name,
                }
                for spec_id, events in violations_by_spec.items()
            },
            "recommendations": self._generate_recommendations(violations_by_spec),
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def _generate_recommendations(
        self,
        violations_by_spec: dict[str, list[SpecViolationEvent]],
    ) -> list[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        for spec_id, events in violations_by_spec.items():
            spec = self.monitor.specs.get(spec_id)
            if not spec:
                continue
            
            if len(events) > 10:
                recommendations.append(
                    f"High violation rate for '{spec.name}'. "
                    f"Consider reviewing the implementation of {spec.function_name}."
                )
            
            if spec.probe_type == ProbeType.PRECONDITION:
                recommendations.append(
                    f"Precondition violations in '{spec.function_name}'. "
                    f"Add input validation at the call sites."
                )
        
        return recommendations


# Convenience function for creating monitors
def create_monitor(config: ProbeConfig | None = None) -> RuntimeMonitor:
    """Create or get the global runtime monitor."""
    if config:
        RuntimeMonitor._instance = RuntimeMonitor(config)
    return RuntimeMonitor.get_instance()


def check(condition: str, **kwargs: Any) -> bool:
    """Check an ad-hoc condition at runtime.
    
    Example:
        check("x > 0 and y > 0", x=x, y=y)
    """
    spec = RuntimeSpec(
        id=hashlib.md5(condition.encode()).hexdigest()[:12],
        name="adhoc_check",
        description=condition,
        probe_type=ProbeType.ASSERTION,
        function_name="<inline>",
        condition=condition,
        parameters=list(kwargs.keys()),
    )
    spec.compile()
    
    try:
        return spec.compiled_condition(**kwargs) if kwargs else spec.compiled_condition()
    except Exception as e:
        logger.warning(f"Runtime check failed: {e}")
        return False
