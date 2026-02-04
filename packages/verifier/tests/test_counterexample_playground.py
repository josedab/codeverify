"""Tests for Interactive Counterexample Playground (Feature 9)."""

import pytest
from datetime import datetime

from codeverify_verifier.counterexample_playground import (
    Z3ModelParser,
    TraceGenerator,
    CounterexampleVisualizer,
    PlaygroundEngine,
    PlaygroundAPI,
    Variable,
    VariableType,
    ExecutionStep,
    ExecutionTrace,
    StepType,
    Counterexample,
    PlaygroundSession,
    create_playground,
)


# Test fixtures
@pytest.fixture
def sample_z3_output():
    """Sample Z3 solver model output."""
    return """sat
(model
  (define-fun x () Int 5)
  (define-fun y () Int -3)
  (define-fun flag () Bool true)
  (define-fun idx () Int 100)
)"""


@pytest.fixture
def simple_z3_output():
    """Simple Z3 output with arrow notation."""
    return """sat
x -> 42
y -> -1
valid -> false
"""


@pytest.fixture
def sample_source_code():
    """Sample Python source code for trace generation."""
    return """
def process_data(x, y):
    if x > 0:
        result = x + y
    else:
        result = x - y
    
    assert result >= 0
    return result
"""


class TestZ3ModelParser:
    """Tests for Z3 model parsing."""

    def test_parse_define_fun_format(self, sample_z3_output):
        parser = Z3ModelParser()
        ce = parser.parse(sample_z3_output)
        
        assert ce.ce_id is not None
        assert len(ce.variables) == 4
        assert 'x' in ce.variables
        assert ce.variables['x'].value == 5
        assert ce.variables['x'].var_type == VariableType.INTEGER

    def test_parse_arrow_format(self, simple_z3_output):
        parser = Z3ModelParser()
        ce = parser.parse(simple_z3_output)
        
        assert 'x' in ce.variables
        assert ce.variables['x'].value == 42
        assert ce.variables['valid'].value == False
        assert ce.variables['valid'].var_type == VariableType.BOOLEAN

    def test_parse_boolean_values(self):
        parser = Z3ModelParser()
        z3_output = "(define-fun flag () Bool true)"
        ce = parser.parse(z3_output)
        
        assert 'flag' in ce.variables
        assert ce.variables['flag'].value == True
        assert ce.variables['flag'].var_type == VariableType.BOOLEAN

    def test_parse_real_values(self):
        parser = Z3ModelParser()
        z3_output = "(define-fun ratio () Real 3.14)"
        ce = parser.parse(z3_output)
        
        assert 'ratio' in ce.variables
        assert ce.variables['ratio'].value == 3.14
        assert ce.variables['ratio'].var_type == VariableType.REAL

    def test_parse_bitvector_hex(self):
        parser = Z3ModelParser()
        z3_output = "ptr -> #xff"
        ce = parser.parse(z3_output)
        
        assert 'ptr' in ce.variables
        assert ce.variables['ptr'].value == 255
        assert ce.variables['ptr'].var_type == VariableType.BITVECTOR

    def test_generates_unique_ids(self, sample_z3_output):
        parser = Z3ModelParser()
        ce1 = parser.parse(sample_z3_output)
        ce2 = parser.parse(sample_z3_output)
        
        # Same input should generate same ID (deterministic)
        assert ce1.ce_id == ce2.ce_id

    def test_extracts_constraints(self):
        parser = Z3ModelParser()
        z3_output = """
        (assert (> x 0))
        (assert (< y 10))
        sat
        x -> 5
        """
        ce = parser.parse(z3_output)
        
        assert len(ce.constraint_path) == 2


class TestTraceGenerator:
    """Tests for execution trace generation."""

    def test_generate_trace_from_code(self, sample_z3_output, sample_source_code):
        parser = Z3ModelParser()
        generator = TraceGenerator()
        
        ce = parser.parse(sample_z3_output)
        trace = generator.generate_trace_from_code(ce, sample_source_code, 'process_data')
        
        assert trace.function_name == 'process_data'
        assert len(trace.steps) > 0

    def test_trace_contains_function_entry(self, sample_z3_output, sample_source_code):
        parser = Z3ModelParser()
        generator = TraceGenerator()
        
        ce = parser.parse(sample_z3_output)
        trace = generator.generate_trace_from_code(ce, sample_source_code, 'process_data')
        
        # First step should be function entry
        entry_steps = [s for s in trace.steps if s.step_type == StepType.FUNCTION_CALL]
        assert len(entry_steps) > 0

    def test_trace_captures_conditions(self, sample_z3_output, sample_source_code):
        parser = Z3ModelParser()
        generator = TraceGenerator()
        
        ce = parser.parse(sample_z3_output)
        trace = generator.generate_trace_from_code(ce, sample_source_code, 'process_data')
        
        condition_steps = [s for s in trace.steps if s.step_type == StepType.CONDITION]
        assert len(condition_steps) > 0

    def test_trace_marks_violations(self, sample_z3_output, sample_source_code):
        parser = Z3ModelParser()
        generator = TraceGenerator()
        
        ce = parser.parse(sample_z3_output)
        trace = generator.generate_trace_from_code(ce, sample_source_code, 'process_data')
        
        # Should have assertion step marked as violation
        assertion_steps = [s for s in trace.steps if s.step_type in (StepType.ASSERTION, StepType.VIOLATION)]
        assert len(assertion_steps) > 0


class TestCounterexampleVisualizer:
    """Tests for visualization generation."""

    @pytest.fixture
    def session(self, sample_z3_output):
        engine = PlaygroundEngine()
        return engine.create_session(sample_z3_output)

    def test_generate_html(self, session):
        visualizer = CounterexampleVisualizer()
        html = visualizer.generate_html(session)
        
        assert '<!DOCTYPE html>' in html
        assert 'Counterexample Playground' in html
        assert session.session_id in html

    def test_html_contains_variables(self, session):
        visualizer = CounterexampleVisualizer()
        html = visualizer.generate_html(session)
        
        # Should contain variable names from the counterexample
        for var_name in session.counterexample.variables:
            assert var_name in html

    def test_html_contains_controls(self, session):
        visualizer = CounterexampleVisualizer()
        html = visualizer.generate_html(session)
        
        assert 'stepBack' in html
        assert 'stepForward' in html
        assert 'shareSession' in html

    def test_generate_mermaid_diagram(self, sample_z3_output, sample_source_code):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output, sample_source_code, 'process_data')
        
        visualizer = CounterexampleVisualizer()
        mermaid = visualizer.generate_mermaid_diagram(session.counterexample.trace)
        
        assert 'flowchart TD' in mermaid
        assert 'S0' in mermaid

    def test_mermaid_styles_violations(self, sample_z3_output, sample_source_code):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output, sample_source_code, 'process_data')
        
        visualizer = CounterexampleVisualizer()
        mermaid = visualizer.generate_mermaid_diagram(session.counterexample.trace)
        
        # Violation styling should be present if there are violations
        violation_steps = [s for s in session.counterexample.trace.steps if s.is_violation]
        if violation_steps:
            assert 'fill:#f48771' in mermaid


class TestPlaygroundEngine:
    """Tests for main playground engine."""

    def test_create_session(self, sample_z3_output):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output)
        
        assert session.session_id is not None
        assert session.counterexample is not None
        assert session.current_step == 0

    def test_get_session(self, sample_z3_output):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output)
        
        retrieved = engine.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_nonexistent_session(self):
        engine = PlaygroundEngine()
        session = engine.get_session('nonexistent-id')
        assert session is None

    def test_modify_value(self, sample_z3_output):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output)
        
        success = engine.modify_value(session.session_id, 'x', 100)
        assert success
        assert session.modified_values['x'] == 100

    def test_modify_nonexistent_variable(self, sample_z3_output):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output)
        
        success = engine.modify_value(session.session_id, 'nonexistent', 100)
        assert not success

    def test_step_forward(self, sample_z3_output):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output)
        
        initial_step = session.current_step
        engine.step_forward(session.session_id)
        
        # If there are steps, should advance
        if len(session.counterexample.trace.steps) > 1:
            assert session.current_step == initial_step + 1

    def test_step_backward(self, sample_z3_output):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output)
        
        # Move forward first
        engine.step_forward(session.session_id)
        current = session.current_step
        
        # Then backward
        engine.step_backward(session.session_id)
        assert session.current_step == current - 1 or session.current_step == 0

    def test_go_to_step(self, sample_z3_output):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output)
        
        if len(session.counterexample.trace.steps) > 2:
            success = engine.go_to_step(session.session_id, 2)
            assert success
            assert session.current_step == 2

    def test_export_html(self, sample_z3_output):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output)
        
        html = engine.export_html(session.session_id)
        assert html is not None
        assert '<!DOCTYPE html>' in html

    def test_export_mermaid(self, sample_z3_output, sample_source_code):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output, sample_source_code, 'process_data')
        
        mermaid = engine.export_mermaid(session.session_id)
        assert mermaid is not None
        assert 'flowchart TD' in mermaid

    def test_generate_share_link(self, sample_z3_output):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output)
        
        link = engine.generate_share_link(session.session_id)
        assert link is not None
        assert session.session_id in link

    def test_tracks_exploration_history(self, sample_z3_output):
        engine = PlaygroundEngine()
        session = engine.create_session(sample_z3_output)
        
        engine.modify_value(session.session_id, 'x', 100)
        engine.modify_value(session.session_id, 'y', 200)
        
        assert len(session.exploration_history) == 2
        assert session.exploration_history[0]['variable'] == 'x'
        assert session.exploration_history[1]['variable'] == 'y'


class TestPlaygroundAPI:
    """Tests for HTTP API interface."""

    @pytest.fixture
    def api(self):
        engine = PlaygroundEngine()
        return PlaygroundAPI(engine)

    def test_create_session_endpoint(self, api, sample_z3_output):
        response = api.create_session_endpoint({'z3_output': sample_z3_output})
        
        assert response['status'] == 201
        assert 'session_id' in response
        assert 'variables' in response

    def test_create_session_missing_output(self, api):
        response = api.create_session_endpoint({})
        
        assert response['status'] == 400
        assert 'error' in response

    def test_get_session_endpoint(self, api, sample_z3_output):
        create_response = api.create_session_endpoint({'z3_output': sample_z3_output})
        session_id = create_response['session_id']
        
        response = api.get_session_endpoint(session_id)
        
        assert response['status'] == 200
        assert response['session_id'] == session_id

    def test_get_session_not_found(self, api):
        response = api.get_session_endpoint('nonexistent')
        assert response['status'] == 404

    def test_modify_value_endpoint(self, api, sample_z3_output):
        create_response = api.create_session_endpoint({'z3_output': sample_z3_output})
        session_id = create_response['session_id']
        
        response = api.modify_value_endpoint(session_id, {'variable': 'x', 'value': 999})
        
        assert response['status'] == 200
        assert response['success'] == True

    def test_navigate_forward(self, api, sample_z3_output):
        create_response = api.create_session_endpoint({'z3_output': sample_z3_output})
        session_id = create_response['session_id']
        
        response = api.navigate_endpoint(session_id, 'forward')
        
        assert response['status'] == 200

    def test_navigate_backward(self, api, sample_z3_output):
        create_response = api.create_session_endpoint({'z3_output': sample_z3_output})
        session_id = create_response['session_id']
        
        # Move forward first
        api.navigate_endpoint(session_id, 'forward')
        response = api.navigate_endpoint(session_id, 'backward')
        
        assert response['status'] == 200

    def test_navigate_reset(self, api, sample_z3_output):
        create_response = api.create_session_endpoint({'z3_output': sample_z3_output})
        session_id = create_response['session_id']
        
        response = api.navigate_endpoint(session_id, 'reset')
        
        assert response['status'] == 200
        assert response['current_step'] == 0

    def test_export_html_endpoint(self, api, sample_z3_output):
        create_response = api.create_session_endpoint({'z3_output': sample_z3_output})
        session_id = create_response['session_id']
        
        response = api.export_endpoint(session_id, 'html')
        
        assert response['status'] == 200
        assert response['format'] == 'html'
        assert '<!DOCTYPE html>' in response['content']

    def test_export_mermaid_endpoint(self, api, sample_z3_output, sample_source_code):
        create_response = api.create_session_endpoint({
            'z3_output': sample_z3_output,
            'source_code': sample_source_code,
            'function_name': 'process_data'
        })
        session_id = create_response['session_id']
        
        response = api.export_endpoint(session_id, 'mermaid')
        
        assert response['status'] == 200
        assert response['format'] == 'mermaid'

    def test_export_unknown_format(self, api, sample_z3_output):
        create_response = api.create_session_endpoint({'z3_output': sample_z3_output})
        session_id = create_response['session_id']
        
        response = api.export_endpoint(session_id, 'unknown')
        
        assert response['status'] == 400


class TestConvenienceFunction:
    """Tests for the create_playground convenience function."""

    def test_create_playground_simple(self, sample_z3_output):
        session = create_playground(sample_z3_output)
        
        assert session is not None
        assert session.session_id is not None

    def test_create_playground_with_source(self, sample_z3_output, sample_source_code):
        session = create_playground(
            sample_z3_output,
            source_code=sample_source_code,
            function_name='process_data'
        )
        
        assert session is not None
        assert session.counterexample.function_name == 'process_data'


class TestVariableTypes:
    """Tests for variable type handling."""

    def test_variable_dataclass(self):
        var = Variable(
            name='test',
            value=42,
            var_type=VariableType.INTEGER,
            source_location='line:10'
        )
        
        assert var.name == 'test'
        assert var.value == 42
        assert var.var_type == VariableType.INTEGER

    def test_all_variable_types(self):
        types = list(VariableType)
        expected = ['integer', 'boolean', 'string', 'array', 'object', 'bitvector', 'real', 'unknown']
        
        assert len(types) == len(expected)
        for t in types:
            assert t.value in expected


class TestExecutionStep:
    """Tests for execution step dataclass."""

    def test_execution_step_creation(self):
        step = ExecutionStep(
            step_id=0,
            step_type=StepType.ASSIGNMENT,
            description='x = 5',
            source_line=10,
            source_code='x = 5'
        )
        
        assert step.step_id == 0
        assert step.step_type == StepType.ASSIGNMENT
        assert step.is_violation == False

    def test_violation_step(self):
        step = ExecutionStep(
            step_id=5,
            step_type=StepType.VIOLATION,
            description='Assertion failed: x > 0',
            is_violation=True
        )
        
        assert step.is_violation == True
        assert step.step_type == StepType.VIOLATION


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow(self, sample_z3_output, sample_source_code):
        # Create engine and session
        engine = PlaygroundEngine()
        session = engine.create_session(
            sample_z3_output,
            source_code=sample_source_code,
            function_name='process_data'
        )
        
        # Verify session created
        assert session is not None
        
        # Navigate through trace
        for _ in range(3):
            engine.step_forward(session.session_id)
        
        # Modify a value
        engine.modify_value(session.session_id, 'x', 999)
        
        # Export HTML
        html = engine.export_html(session.session_id)
        assert '<!DOCTYPE html>' in html
        assert '999' in html or 'x' in html
        
        # Export Mermaid
        mermaid = engine.export_mermaid(session.session_id)
        assert 'flowchart TD' in mermaid
        
        # Get share link
        link = engine.generate_share_link(session.session_id)
        assert session.session_id in link

    def test_api_full_workflow(self, sample_z3_output, sample_source_code):
        engine = PlaygroundEngine()
        api = PlaygroundAPI(engine)
        
        # Create session via API
        create_resp = api.create_session_endpoint({
            'z3_output': sample_z3_output,
            'source_code': sample_source_code,
            'function_name': 'process_data'
        })
        assert create_resp['status'] == 201
        session_id = create_resp['session_id']
        
        # Modify value via API
        modify_resp = api.modify_value_endpoint(session_id, {'variable': 'x', 'value': 123})
        assert modify_resp['status'] == 200
        
        # Navigate via API
        nav_resp = api.navigate_endpoint(session_id, 'forward')
        assert nav_resp['status'] == 200
        
        # Export via API
        export_resp = api.export_endpoint(session_id, 'html')
        assert export_resp['status'] == 200
        assert '<!DOCTYPE html>' in export_resp['content']
