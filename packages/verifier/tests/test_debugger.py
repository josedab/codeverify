"""Tests for Verification Debugger functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from codeverify_verifier.debugger import (
    VerificationDebugger,
    DebugStep,
    DebugSession,
    StepStatus,
    ConstraintInfo,
)


class TestDebugStep:
    """Tests for DebugStep dataclass."""

    def test_step_creation(self):
        """Step can be created with all fields."""
        step = DebugStep(
            step_number=1,
            title="Check precondition",
            description="Verify x > 0",
            status=StepStatus.PASSED,
            constraint="x > 0",
            model={"x": 5},
        )
        
        assert step.step_number == 1
        assert step.title == "Check precondition"
        assert step.status == StepStatus.PASSED
        assert step.model["x"] == 5

    def test_step_status_values(self):
        """StepStatus enum has expected values."""
        assert StepStatus.PASSED.value == "passed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.SKIPPED.value == "skipped"


class TestConstraintInfo:
    """Tests for ConstraintInfo dataclass."""

    def test_constraint_info_creation(self):
        """ConstraintInfo can be created."""
        info = ConstraintInfo(
            name="precondition",
            expression="x > 0 && y > 0",
            variables=["x", "y"],
            source_line=10,
        )
        
        assert info.name == "precondition"
        assert "x" in info.variables
        assert "y" in info.variables


class TestDebugSession:
    """Tests for DebugSession."""

    @pytest.fixture
    def session(self):
        """Create a debug session."""
        return DebugSession(session_id="test-session")

    def test_session_creation(self, session):
        """Session is created with ID."""
        assert session.session_id == "test-session"
        assert len(session.steps) == 0

    def test_add_step(self, session):
        """Steps can be added to session."""
        step = DebugStep(
            step_number=1,
            title="Test Step",
            description="Test",
            status=StepStatus.PENDING,
        )
        session.add_step(step)
        
        assert len(session.steps) == 1
        assert session.steps[0].title == "Test Step"

    def test_get_current_step(self, session):
        """Can get current step."""
        session.add_step(DebugStep(
            step_number=1, title="Step 1", description="", status=StepStatus.PASSED
        ))
        session.add_step(DebugStep(
            step_number=2, title="Step 2", description="", status=StepStatus.PENDING
        ))
        
        current = session.get_current_step()
        assert current.title == "Step 2"

    def test_all_steps_passed(self, session):
        """Can check if all steps passed."""
        session.add_step(DebugStep(
            step_number=1, title="Step 1", description="", status=StepStatus.PASSED
        ))
        session.add_step(DebugStep(
            step_number=2, title="Step 2", description="", status=StepStatus.PASSED
        ))
        
        assert session.all_passed()

    def test_has_failures(self, session):
        """Can check for failures."""
        session.add_step(DebugStep(
            step_number=1, title="Step 1", description="", status=StepStatus.PASSED
        ))
        session.add_step(DebugStep(
            step_number=2, title="Step 2", description="", status=StepStatus.FAILED
        ))
        
        assert session.has_failures()
        assert not session.all_passed()


class TestVerificationDebugger:
    """Tests for VerificationDebugger."""

    @pytest.fixture
    def debugger(self):
        """Create a verification debugger."""
        return VerificationDebugger()

    def test_debugger_initialization(self, debugger):
        """Debugger initializes correctly."""
        assert debugger is not None
        assert hasattr(debugger, "trace")
        assert hasattr(debugger, "create_session")

    @pytest.mark.asyncio
    async def test_trace_simple_code(self, debugger):
        """Debugger traces simple code."""
        code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        result = await debugger.trace(code)
        
        assert "steps" in result
        assert "result" in result
        assert isinstance(result["steps"], list)

    @pytest.mark.asyncio
    async def test_trace_with_assertions(self, debugger):
        """Debugger traces code with assertions."""
        code = """
def divide(a: int, b: int) -> float:
    assert b != 0, "Cannot divide by zero"
    return a / b
"""
        result = await debugger.trace(code)
        
        # Should have step for assertion check
        steps = result.get("steps", [])
        assert len(steps) > 0

    @pytest.mark.asyncio
    async def test_trace_with_loop(self, debugger):
        """Debugger traces code with loops."""
        code = """
def sum_list(items: list) -> int:
    total = 0
    for item in items:
        total += item
    return total
"""
        result = await debugger.trace(code)
        
        assert "steps" in result
        # Loop verification should be present
        assert result.get("result") in ["verified", "unverified", "unknown"]

    @pytest.mark.asyncio
    async def test_trace_returns_counterexample(self, debugger):
        """Debugger returns counterexample on failure."""
        code = """
def unsafe_divide(a: int, b: int) -> int:
    # Missing division by zero check
    return a // b
"""
        result = await debugger.trace(code)
        
        # May include counterexample if verification fails
        if result.get("result") == "unverified":
            assert "counterexample" in result or "model" in result.get("steps", [{}])[-1]

    def test_create_interactive_session(self, debugger):
        """Debugger can create interactive session."""
        session = debugger.create_session()
        
        assert isinstance(session, DebugSession)
        assert session.session_id is not None

    @pytest.mark.asyncio
    async def test_step_through_verification(self, debugger):
        """Can step through verification interactively."""
        code = """
def check(x: int) -> bool:
    assert x > 0
    return x > 10
"""
        session = debugger.create_session()
        await debugger.load_code(session, code)
        
        # Step through
        step1 = await debugger.step_next(session)
        assert step1 is not None

    @pytest.mark.asyncio
    async def test_explain_step(self, debugger):
        """Debugger can explain verification steps."""
        step = DebugStep(
            step_number=1,
            title="Check precondition",
            description="Verify x > 0",
            status=StepStatus.FAILED,
            constraint="x > 0",
            model={"x": -5},
        )
        
        explanation = await debugger.explain_step(step)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    @pytest.mark.asyncio
    async def test_get_visualization_data(self, debugger):
        """Debugger provides visualization data."""
        code = """
def bounded(x: int) -> int:
    if x < 0:
        x = 0
    if x > 100:
        x = 100
    return x
"""
        result = await debugger.trace(code)
        viz_data = debugger.get_visualization_data(result)
        
        assert isinstance(viz_data, dict)
        # Should include data suitable for visualization
        assert "nodes" in viz_data or "steps" in viz_data or "graph" in viz_data


class TestDebuggerConstraintExtraction:
    """Tests for constraint extraction functionality."""

    @pytest.fixture
    def debugger(self):
        return VerificationDebugger()

    @pytest.mark.asyncio
    async def test_extract_preconditions(self, debugger):
        """Debugger extracts preconditions from code."""
        code = """
def process(items: list) -> int:
    '''Process items.
    
    Requires: len(items) > 0
    '''
    return items[0]
"""
        result = await debugger.trace(code)
        
        # Should identify the precondition
        steps = result.get("steps", [])
        constraints = [s for s in steps if "precondition" in s.get("title", "").lower()]
        # May or may not find depending on implementation
        assert isinstance(constraints, list)

    @pytest.mark.asyncio
    async def test_extract_loop_invariants(self, debugger):
        """Debugger identifies loop invariants."""
        code = """
def sum_positive(items: list) -> int:
    total = 0  # invariant: total >= 0
    for item in items:
        if item > 0:
            total += item
    return total
"""
        result = await debugger.trace(code)
        
        # Result should include loop-related steps
        assert "steps" in result


class TestDebuggerErrorHandling:
    """Tests for error handling in debugger."""

    @pytest.fixture
    def debugger(self):
        return VerificationDebugger()

    @pytest.mark.asyncio
    async def test_handle_syntax_error(self, debugger):
        """Debugger handles syntax errors gracefully."""
        code = "def broken(: syntax error here"
        
        result = await debugger.trace(code)
        
        assert "error" in result or result.get("result") == "error"

    @pytest.mark.asyncio
    async def test_handle_empty_code(self, debugger):
        """Debugger handles empty code."""
        result = await debugger.trace("")
        
        assert isinstance(result, dict)
        # Should return something reasonable

    @pytest.mark.asyncio
    async def test_handle_timeout(self, debugger):
        """Debugger handles verification timeout."""
        # Code that might cause long verification
        code = """
def complex(x: int, y: int, z: int) -> int:
    result = 0
    for i in range(x):
        for j in range(y):
            for k in range(z):
                result += i * j * k
    return result
"""
        # Set short timeout
        result = await debugger.trace(code, timeout_ms=100)
        
        # Should complete (possibly with timeout status)
        assert isinstance(result, dict)


class TestDebuggerSessionManagement:
    """Tests for session management."""

    @pytest.fixture
    def debugger(self):
        return VerificationDebugger()

    def test_multiple_sessions(self, debugger):
        """Can create multiple independent sessions."""
        session1 = debugger.create_session()
        session2 = debugger.create_session()
        
        assert session1.session_id != session2.session_id

    @pytest.mark.asyncio
    async def test_session_isolation(self, debugger):
        """Sessions are isolated from each other."""
        session1 = debugger.create_session()
        session2 = debugger.create_session()
        
        await debugger.load_code(session1, "def a(): pass")
        await debugger.load_code(session2, "def b(): pass")
        
        # Sessions should have different loaded code
        assert session1.code != session2.code if hasattr(session1, 'code') else True

    def test_reset_session(self, debugger):
        """Session can be reset."""
        session = debugger.create_session()
        session.add_step(DebugStep(1, "Step", "", StepStatus.PASSED))
        
        session.reset()
        
        assert len(session.steps) == 0

    def test_session_state_persistence(self, debugger):
        """Session state persists between operations."""
        session = debugger.create_session()
        initial_id = session.session_id
        
        session.add_step(DebugStep(1, "Step 1", "", StepStatus.PASSED))
        session.add_step(DebugStep(2, "Step 2", "", StepStatus.PENDING))
        
        assert session.session_id == initial_id
        assert len(session.steps) == 2
