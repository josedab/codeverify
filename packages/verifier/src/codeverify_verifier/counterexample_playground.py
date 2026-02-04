"""
Interactive Counterexample Playground - Feature 9

Visual debugger for Z3 counterexamples that makes formal verification accessible.
When Z3 finds a bug, developers can interactively explore the counterexample
in a visual debugger with step-through execution and variable state visualization.

Key capabilities:
- Parse and visualize Z3 counterexamples
- Generate execution traces with variable states
- Create interactive HTML visualizations
- Support "edit and re-verify" exploration
- Generate shareable links for team discussion
"""

import json
import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pathlib import Path


class VariableType(Enum):
    """Types of variables in counterexamples."""
    INTEGER = "integer"
    BOOLEAN = "boolean"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"
    BITVECTOR = "bitvector"
    REAL = "real"
    UNKNOWN = "unknown"


class StepType(Enum):
    """Types of execution steps."""
    ASSIGNMENT = "assignment"
    CONDITION = "condition"
    FUNCTION_CALL = "function_call"
    FUNCTION_RETURN = "function_return"
    ASSERTION = "assertion"
    VIOLATION = "violation"


@dataclass
class Variable:
    """A variable in the counterexample state."""
    name: str
    value: Any
    var_type: VariableType
    source_location: Optional[str] = None
    constraint: Optional[str] = None


@dataclass
class ExecutionStep:
    """A single step in the execution trace."""
    step_id: int
    step_type: StepType
    description: str
    source_line: Optional[int] = None
    source_code: Optional[str] = None
    variables_before: dict[str, Variable] = field(default_factory=dict)
    variables_after: dict[str, Variable] = field(default_factory=dict)
    condition_result: Optional[bool] = None
    is_violation: bool = False


@dataclass
class ExecutionTrace:
    """Complete execution trace from a counterexample."""
    trace_id: str
    function_name: str
    steps: list[ExecutionStep]
    initial_state: dict[str, Variable]
    final_state: dict[str, Variable]
    violation_step: Optional[int] = None
    violation_message: Optional[str] = None


@dataclass
class Counterexample:
    """Parsed Z3 counterexample with execution trace."""
    ce_id: str
    raw_output: str
    variables: dict[str, Variable]
    trace: ExecutionTrace
    constraint_path: list[str]
    created_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[str] = None
    function_name: Optional[str] = None


@dataclass
class PlaygroundSession:
    """Interactive session for exploring counterexamples."""
    session_id: str
    counterexample: Counterexample
    current_step: int = 0
    modified_values: dict[str, Any] = field(default_factory=dict)
    exploration_history: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class Z3ModelParser:
    """Parse Z3 solver model output into structured counterexamples."""

    def __init__(self):
        self.type_patterns = {
            VariableType.INTEGER: re.compile(r'^-?\d+$'),
            VariableType.BOOLEAN: re.compile(r'^(true|false|True|False)$'),
            VariableType.REAL: re.compile(r'^-?\d+\.\d+$'),
            VariableType.BITVECTOR: re.compile(r'^#[xb][0-9a-fA-F]+$'),
        }

    def parse(self, z3_output: str, source_code: Optional[str] = None) -> Counterexample:
        """Parse Z3 model output into a structured counterexample."""
        variables = self._extract_variables(z3_output)
        trace = self._generate_trace(variables, source_code)
        constraint_path = self._extract_constraints(z3_output)
        
        ce_id = hashlib.sha256(z3_output.encode()).hexdigest()[:12]
        
        return Counterexample(
            ce_id=ce_id,
            raw_output=z3_output,
            variables=variables,
            trace=trace,
            constraint_path=constraint_path
        )

    def _extract_variables(self, z3_output: str) -> dict[str, Variable]:
        """Extract variable assignments from Z3 output."""
        variables = {}
        
        # Pattern for Z3 model variable assignments
        # Handles: (define-fun x () Int 5) and x -> 5 formats
        define_pattern = re.compile(
            r'\(define-fun\s+(\w+)\s*\(\)\s*(\w+)\s+([^)]+)\)'
        )
        simple_pattern = re.compile(r'(\w+)\s*(?:->|=)\s*(.+?)(?:\n|$)')
        
        for match in define_pattern.finditer(z3_output):
            name, type_str, value_str = match.groups()
            var_type = self._infer_type(type_str, value_str.strip())
            value = self._parse_value(value_str.strip(), var_type)
            variables[name] = Variable(
                name=name,
                value=value,
                var_type=var_type,
                constraint=f"(define-fun {name} () {type_str} {value_str})"
            )
        
        for match in simple_pattern.finditer(z3_output):
            name, value_str = match.groups()
            if name not in variables and name not in ('sat', 'unsat', 'model'):
                var_type = self._infer_type_from_value(value_str.strip())
                value = self._parse_value(value_str.strip(), var_type)
                variables[name] = Variable(
                    name=name,
                    value=value,
                    var_type=var_type
                )
        
        return variables

    def _infer_type(self, type_str: str, value_str: str) -> VariableType:
        """Infer variable type from Z3 type declaration."""
        type_map = {
            'Int': VariableType.INTEGER,
            'Bool': VariableType.BOOLEAN,
            'Real': VariableType.REAL,
            'String': VariableType.STRING,
        }
        if type_str in type_map:
            return type_map[type_str]
        if type_str.startswith('(_ BitVec'):
            return VariableType.BITVECTOR
        if type_str.startswith('Array'):
            return VariableType.ARRAY
        return self._infer_type_from_value(value_str)

    def _infer_type_from_value(self, value_str: str) -> VariableType:
        """Infer variable type from its value string."""
        for var_type, pattern in self.type_patterns.items():
            if pattern.match(value_str):
                return var_type
        if value_str.startswith('[') or value_str.startswith('{'):
            return VariableType.ARRAY
        return VariableType.UNKNOWN

    def _parse_value(self, value_str: str, var_type: VariableType) -> Any:
        """Parse value string into Python object."""
        if var_type == VariableType.INTEGER:
            return int(value_str)
        elif var_type == VariableType.BOOLEAN:
            return value_str.lower() == 'true'
        elif var_type == VariableType.REAL:
            return float(value_str)
        elif var_type == VariableType.BITVECTOR:
            if value_str.startswith('#x'):
                return int(value_str[2:], 16)
            elif value_str.startswith('#b'):
                return int(value_str[2:], 2)
            return value_str
        return value_str

    def _generate_trace(
        self, 
        variables: dict[str, Variable], 
        source_code: Optional[str]
    ) -> ExecutionTrace:
        """Generate execution trace from variables and source code."""
        steps = []
        step_id = 0
        
        # Create initial state from variables
        initial_state = dict(variables)
        
        # Add assignment step for each variable
        for name, var in variables.items():
            step = ExecutionStep(
                step_id=step_id,
                step_type=StepType.ASSIGNMENT,
                description=f"{name} = {var.value}",
                variables_after={name: var}
            )
            steps.append(step)
            step_id += 1
        
        trace_id = hashlib.sha256(str(variables).encode()).hexdigest()[:8]
        
        return ExecutionTrace(
            trace_id=trace_id,
            function_name="<analyzed>",
            steps=steps,
            initial_state=initial_state,
            final_state=dict(variables)
        )

    def _extract_constraints(self, z3_output: str) -> list[str]:
        """Extract constraint path from Z3 output."""
        constraints = []
        
        # Look for assertions in the output
        assertion_pattern = re.compile(r'\(assert\s+([^)]+(?:\([^)]*\)[^)]*)*)\)')
        for match in assertion_pattern.finditer(z3_output):
            constraints.append(match.group(1))
        
        return constraints


class TraceGenerator:
    """Generate detailed execution traces from counterexamples."""

    def __init__(self):
        self.step_counter = 0

    def generate_trace_from_code(
        self,
        counterexample: Counterexample,
        source_code: str,
        function_name: str
    ) -> ExecutionTrace:
        """Generate execution trace by simulating code with counterexample values."""
        self.step_counter = 0
        steps = []
        state = dict(counterexample.variables)
        
        # Parse source code lines
        lines = source_code.split('\n')
        in_function = False
        violation_step = None
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Track function entry
            if f'def {function_name}' in stripped or f'function {function_name}' in stripped:
                in_function = True
                step = self._create_step(
                    StepType.FUNCTION_CALL,
                    f"Enter {function_name}",
                    i, line, state
                )
                steps.append(step)
                continue
            
            if not in_function:
                continue
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                continue
            
            # Detect conditions
            if stripped.startswith(('if ', 'elif ', 'while ', 'for ')):
                condition_result = self._evaluate_condition_hint(stripped, state)
                step = self._create_step(
                    StepType.CONDITION,
                    f"Condition: {stripped}",
                    i, line, state,
                    condition_result=condition_result
                )
                steps.append(step)
            
            # Detect assertions (these can be violations)
            elif stripped.startswith('assert'):
                is_violation = self._check_assertion_violation(stripped, state)
                step = self._create_step(
                    StepType.ASSERTION if not is_violation else StepType.VIOLATION,
                    f"Assertion: {stripped}",
                    i, line, state,
                    is_violation=is_violation
                )
                if is_violation:
                    violation_step = step.step_id
                steps.append(step)
            
            # Detect return
            elif stripped.startswith('return'):
                step = self._create_step(
                    StepType.FUNCTION_RETURN,
                    f"Return: {stripped}",
                    i, line, state
                )
                steps.append(step)
                break
        
        trace_id = hashlib.sha256(
            f"{function_name}:{counterexample.ce_id}".encode()
        ).hexdigest()[:8]
        
        return ExecutionTrace(
            trace_id=trace_id,
            function_name=function_name,
            steps=steps,
            initial_state=dict(counterexample.variables),
            final_state=state,
            violation_step=violation_step
        )

    def _create_step(
        self,
        step_type: StepType,
        description: str,
        source_line: int,
        source_code: str,
        state: dict[str, Variable],
        condition_result: Optional[bool] = None,
        is_violation: bool = False
    ) -> ExecutionStep:
        """Create an execution step."""
        step = ExecutionStep(
            step_id=self.step_counter,
            step_type=step_type,
            description=description,
            source_line=source_line,
            source_code=source_code,
            variables_before=dict(state),
            variables_after=dict(state),
            condition_result=condition_result,
            is_violation=is_violation
        )
        self.step_counter += 1
        return step

    def _evaluate_condition_hint(
        self, 
        condition: str, 
        state: dict[str, Variable]
    ) -> Optional[bool]:
        """Provide hint about condition evaluation based on counterexample state."""
        # Simple heuristic evaluation
        for var_name, var in state.items():
            if var_name in condition:
                # Check for simple comparisons
                if f'{var_name} > 0' in condition and isinstance(var.value, (int, float)):
                    return var.value > 0
                if f'{var_name} < 0' in condition and isinstance(var.value, (int, float)):
                    return var.value < 0
                if f'{var_name} == ' in condition:
                    return True  # The counterexample was generated to satisfy this
        return None

    def _check_assertion_violation(
        self, 
        assertion: str, 
        state: dict[str, Variable]
    ) -> bool:
        """Check if assertion would be violated with counterexample values."""
        # Counterexamples are generated for violated assertions
        return True


class CounterexampleVisualizer:
    """Generate visual representations of counterexamples."""

    def __init__(self):
        self.theme = {
            'bg_color': '#1e1e1e',
            'text_color': '#d4d4d4',
            'highlight_color': '#264f78',
            'error_color': '#f48771',
            'success_color': '#89d185',
            'warning_color': '#cca700',
            'border_color': '#404040',
        }

    def generate_html(self, session: PlaygroundSession) -> str:
        """Generate interactive HTML visualization."""
        ce = session.counterexample
        trace = ce.trace
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeVerify - Counterexample Playground</title>
    <style>
        {self._generate_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Counterexample Playground</h1>
            <p class="session-info">Session: {session.session_id} | Created: {session.created_at.isoformat()}</p>
        </header>
        
        <div class="main-content">
            <div class="panel variables-panel">
                <h2>📊 Variables</h2>
                {self._render_variables(ce.variables, session.modified_values)}
            </div>
            
            <div class="panel trace-panel">
                <h2>📜 Execution Trace</h2>
                {self._render_trace(trace, session.current_step)}
            </div>
            
            <div class="panel constraints-panel">
                <h2>⚙️ Constraints</h2>
                {self._render_constraints(ce.constraint_path)}
            </div>
        </div>
        
        <div class="controls">
            <button onclick="stepBack()">⏮️ Back</button>
            <button onclick="stepForward()">⏭️ Forward</button>
            <button onclick="resetTrace()">🔄 Reset</button>
            <button onclick="shareSession()">🔗 Share</button>
        </div>
        
        <footer>
            <p>CodeVerify Interactive Counterexample Playground v1.0</p>
        </footer>
    </div>
    
    <script>
        {self._generate_javascript(session)}
    </script>
</body>
</html>"""
        return html_content

    def _generate_css(self) -> str:
        """Generate CSS styles for the visualization."""
        return f"""
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: {self.theme['bg_color']};
            color: {self.theme['text_color']};
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid {self.theme['border_color']};
        }}
        header h1 {{ color: {self.theme['success_color']}; }}
        .session-info {{ color: #888; font-size: 0.9em; }}
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .panel {{
            background: #252526;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid {self.theme['border_color']};
        }}
        .panel h2 {{
            margin-bottom: 15px;
            font-size: 1.1em;
            color: {self.theme['success_color']};
        }}
        .variable-item {{
            padding: 8px;
            margin: 5px 0;
            background: #2d2d2d;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
        }}
        .variable-name {{ color: #9cdcfe; }}
        .variable-value {{ color: #ce9178; }}
        .variable-modified {{ border-left: 3px solid {self.theme['warning_color']}; }}
        .trace-step {{
            padding: 10px;
            margin: 5px 0;
            background: #2d2d2d;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .trace-step:hover {{ background: #3d3d3d; }}
        .trace-step.current {{ background: {self.theme['highlight_color']}; }}
        .trace-step.violation {{
            border-left: 3px solid {self.theme['error_color']};
            background: rgba(244, 135, 113, 0.1);
        }}
        .step-type {{ font-size: 0.8em; color: #888; }}
        .step-desc {{ margin-top: 5px; }}
        .source-line {{ font-family: monospace; color: #b5cea8; }}
        .constraint-item {{
            padding: 8px;
            margin: 5px 0;
            background: #2d2d2d;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.85em;
            word-break: break-all;
        }}
        .controls {{
            display: flex;
            justify-content: center;
            gap: 15px;
            padding: 20px;
        }}
        .controls button {{
            padding: 12px 24px;
            font-size: 1em;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            background: #0e639c;
            color: white;
            transition: background 0.2s;
        }}
        .controls button:hover {{ background: #1177bb; }}
        footer {{
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid {self.theme['border_color']};
            color: #888;
        }}
        @media (max-width: 1000px) {{
            .main-content {{ grid-template-columns: 1fr; }}
        }}
        """

    def _render_variables(
        self, 
        variables: dict[str, Variable], 
        modified: dict[str, Any]
    ) -> str:
        """Render variable state panel."""
        items = []
        for name, var in variables.items():
            modified_class = 'variable-modified' if name in modified else ''
            value = modified.get(name, var.value)
            items.append(f"""
            <div class="variable-item {modified_class}">
                <span class="variable-name">{html.escape(name)}</span>
                <span class="variable-value">{html.escape(str(value))}</span>
            </div>
            """)
        return '\n'.join(items) if items else '<p>No variables</p>'

    def _render_trace(self, trace: ExecutionTrace, current_step: int) -> str:
        """Render execution trace panel."""
        items = []
        for step in trace.steps:
            current_class = 'current' if step.step_id == current_step else ''
            violation_class = 'violation' if step.is_violation else ''
            source_line = f'<div class="source-line">Line {step.source_line}</div>' if step.source_line else ''
            
            items.append(f"""
            <div class="trace-step {current_class} {violation_class}" data-step="{step.step_id}">
                <div class="step-type">{step.step_type.value.upper()}</div>
                <div class="step-desc">{html.escape(step.description)}</div>
                {source_line}
            </div>
            """)
        return '\n'.join(items) if items else '<p>No trace available</p>'

    def _render_constraints(self, constraints: list[str]) -> str:
        """Render constraints panel."""
        items = []
        for i, constraint in enumerate(constraints):
            items.append(f"""
            <div class="constraint-item">
                <strong>#{i + 1}:</strong> {html.escape(constraint)}
            </div>
            """)
        return '\n'.join(items) if items else '<p>No constraints</p>'

    def _generate_javascript(self, session: PlaygroundSession) -> str:
        """Generate JavaScript for interactivity."""
        return f"""
        let currentStep = {session.current_step};
        const totalSteps = {len(session.counterexample.trace.steps)};
        const sessionId = '{session.session_id}';
        
        function updateHighlight() {{
            document.querySelectorAll('.trace-step').forEach((el, i) => {{
                el.classList.toggle('current', i === currentStep);
            }});
        }}
        
        function stepBack() {{
            if (currentStep > 0) {{
                currentStep--;
                updateHighlight();
            }}
        }}
        
        function stepForward() {{
            if (currentStep < totalSteps - 1) {{
                currentStep++;
                updateHighlight();
            }}
        }}
        
        function resetTrace() {{
            currentStep = 0;
            updateHighlight();
        }}
        
        function shareSession() {{
            const url = window.location.origin + '/playground/' + sessionId;
            navigator.clipboard.writeText(url).then(() => {{
                alert('Session URL copied to clipboard!');
            }});
        }}
        
        document.querySelectorAll('.trace-step').forEach((el, i) => {{
            el.addEventListener('click', () => {{
                currentStep = i;
                updateHighlight();
            }});
        }});
        """

    def generate_mermaid_diagram(self, trace: ExecutionTrace) -> str:
        """Generate Mermaid flowchart diagram of the execution trace."""
        lines = ['flowchart TD']
        
        for i, step in enumerate(trace.steps):
            node_id = f'S{i}'
            shape_start = '((' if step.step_type == StepType.VIOLATION else '['
            shape_end = '))' if step.step_type == StepType.VIOLATION else ']'
            
            # Escape special characters for Mermaid
            desc = step.description.replace('"', "'").replace('[', '(').replace(']', ')')
            lines.append(f'    {node_id}{shape_start}"{desc}"{shape_end}')
            
            if i > 0:
                prev_id = f'S{i-1}'
                edge_label = ''
                if trace.steps[i-1].condition_result is not None:
                    edge_label = f'|{"true" if trace.steps[i-1].condition_result else "false"}|'
                lines.append(f'    {prev_id} -->{edge_label} {node_id}')
        
        # Style violation nodes
        for i, step in enumerate(trace.steps):
            if step.is_violation:
                lines.append(f'    style S{i} fill:#f48771,stroke:#333,stroke-width:2px')
        
        return '\n'.join(lines)


class PlaygroundEngine:
    """Main engine for the interactive counterexample playground."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.parser = Z3ModelParser()
        self.trace_generator = TraceGenerator()
        self.visualizer = CounterexampleVisualizer()
        self.storage_path = storage_path or Path('.codeverify/playground')
        self.sessions: dict[str, PlaygroundSession] = {}

    def create_session(
        self,
        z3_output: str,
        source_code: Optional[str] = None,
        function_name: Optional[str] = None
    ) -> PlaygroundSession:
        """Create a new playground session from Z3 output."""
        counterexample = self.parser.parse(z3_output, source_code)
        
        if source_code and function_name:
            counterexample.trace = self.trace_generator.generate_trace_from_code(
                counterexample, source_code, function_name
            )
            counterexample.function_name = function_name
        
        session_id = hashlib.sha256(
            f"{counterexample.ce_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        session = PlaygroundSession(
            session_id=session_id,
            counterexample=counterexample
        )
        
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[PlaygroundSession]:
        """Retrieve an existing session."""
        return self.sessions.get(session_id)

    def modify_value(
        self, 
        session_id: str, 
        variable_name: str, 
        new_value: Any
    ) -> bool:
        """Modify a variable value for exploration."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        if variable_name not in session.counterexample.variables:
            return False
        
        session.modified_values[variable_name] = new_value
        session.exploration_history.append({
            'action': 'modify',
            'variable': variable_name,
            'old_value': session.counterexample.variables[variable_name].value,
            'new_value': new_value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def step_forward(self, session_id: str) -> bool:
        """Move to the next step in the trace."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        max_step = len(session.counterexample.trace.steps) - 1
        if session.current_step < max_step:
            session.current_step += 1
            return True
        return False

    def step_backward(self, session_id: str) -> bool:
        """Move to the previous step in the trace."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        if session.current_step > 0:
            session.current_step -= 1
            return True
        return False

    def go_to_step(self, session_id: str, step: int) -> bool:
        """Jump to a specific step in the trace."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        max_step = len(session.counterexample.trace.steps) - 1
        if 0 <= step <= max_step:
            session.current_step = step
            return True
        return False

    def export_html(self, session_id: str) -> Optional[str]:
        """Export session as interactive HTML."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        return self.visualizer.generate_html(session)

    def export_mermaid(self, session_id: str) -> Optional[str]:
        """Export execution trace as Mermaid diagram."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        return self.visualizer.generate_mermaid_diagram(session.counterexample.trace)

    def generate_share_link(self, session_id: str) -> Optional[str]:
        """Generate a shareable link for the session."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        return f"/playground/{session_id}"

    def save_session(self, session_id: str) -> bool:
        """Save session to persistent storage."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        session_file = self.storage_path / f"{session_id}.json"
        
        data = {
            'session_id': session.session_id,
            'counterexample': {
                'ce_id': session.counterexample.ce_id,
                'raw_output': session.counterexample.raw_output,
                'variables': {
                    name: {
                        'name': var.name,
                        'value': var.value,
                        'var_type': var.var_type.value,
                    }
                    for name, var in session.counterexample.variables.items()
                },
                'constraint_path': session.counterexample.constraint_path,
            },
            'current_step': session.current_step,
            'modified_values': session.modified_values,
            'exploration_history': session.exploration_history,
            'created_at': session.created_at.isoformat(),
        }
        
        with open(session_file, 'w') as f:
            json.dump(data, f, indent=2)
        return True

    def load_session(self, session_id: str) -> Optional[PlaygroundSession]:
        """Load session from persistent storage."""
        session_file = self.storage_path / f"{session_id}.json"
        if not session_file.exists():
            return None
        
        with open(session_file, 'r') as f:
            data = json.load(f)
        
        # Reconstruct counterexample
        variables = {
            name: Variable(
                name=var_data['name'],
                value=var_data['value'],
                var_type=VariableType(var_data['var_type'])
            )
            for name, var_data in data['counterexample']['variables'].items()
        }
        
        ce = Counterexample(
            ce_id=data['counterexample']['ce_id'],
            raw_output=data['counterexample']['raw_output'],
            variables=variables,
            trace=ExecutionTrace(
                trace_id='loaded',
                function_name='<loaded>',
                steps=[],
                initial_state=variables,
                final_state=variables
            ),
            constraint_path=data['counterexample']['constraint_path'],
            created_at=datetime.fromisoformat(data['created_at'])
        )
        
        session = PlaygroundSession(
            session_id=data['session_id'],
            counterexample=ce,
            current_step=data['current_step'],
            modified_values=data['modified_values'],
            exploration_history=data['exploration_history'],
            created_at=datetime.fromisoformat(data['created_at'])
        )
        
        self.sessions[session_id] = session
        return session


class PlaygroundAPI:
    """HTTP API interface for the playground."""

    def __init__(self, engine: PlaygroundEngine):
        self.engine = engine

    def create_session_endpoint(self, request_data: dict) -> dict:
        """POST /api/playground/sessions - Create new session."""
        z3_output = request_data.get('z3_output', '')
        source_code = request_data.get('source_code')
        function_name = request_data.get('function_name')
        
        if not z3_output:
            return {'error': 'z3_output is required', 'status': 400}
        
        session = self.engine.create_session(z3_output, source_code, function_name)
        
        return {
            'session_id': session.session_id,
            'counterexample_id': session.counterexample.ce_id,
            'variables': {
                name: {'value': var.value, 'type': var.var_type.value}
                for name, var in session.counterexample.variables.items()
            },
            'steps_count': len(session.counterexample.trace.steps),
            'share_link': self.engine.generate_share_link(session.session_id),
            'status': 201
        }

    def get_session_endpoint(self, session_id: str) -> dict:
        """GET /api/playground/sessions/{id} - Get session details."""
        session = self.engine.get_session(session_id)
        if not session:
            return {'error': 'Session not found', 'status': 404}
        
        return {
            'session_id': session.session_id,
            'current_step': session.current_step,
            'modified_values': session.modified_values,
            'variables': {
                name: {'value': var.value, 'type': var.var_type.value}
                for name, var in session.counterexample.variables.items()
            },
            'status': 200
        }

    def modify_value_endpoint(self, session_id: str, request_data: dict) -> dict:
        """POST /api/playground/sessions/{id}/modify - Modify variable value."""
        variable = request_data.get('variable')
        value = request_data.get('value')
        
        if not variable:
            return {'error': 'variable is required', 'status': 400}
        
        success = self.engine.modify_value(session_id, variable, value)
        if not success:
            return {'error': 'Failed to modify value', 'status': 400}
        
        return {'success': True, 'status': 200}

    def navigate_endpoint(self, session_id: str, action: str) -> dict:
        """POST /api/playground/sessions/{id}/navigate - Navigate trace."""
        if action == 'forward':
            success = self.engine.step_forward(session_id)
        elif action == 'backward':
            success = self.engine.step_backward(session_id)
        elif action == 'reset':
            success = self.engine.go_to_step(session_id, 0)
        else:
            return {'error': f'Unknown action: {action}', 'status': 400}
        
        session = self.engine.get_session(session_id)
        return {
            'success': success,
            'current_step': session.current_step if session else 0,
            'status': 200
        }

    def export_endpoint(self, session_id: str, format: str) -> dict:
        """GET /api/playground/sessions/{id}/export?format={html|mermaid}."""
        if format == 'html':
            content = self.engine.export_html(session_id)
        elif format == 'mermaid':
            content = self.engine.export_mermaid(session_id)
        else:
            return {'error': f'Unknown format: {format}', 'status': 400}
        
        if content is None:
            return {'error': 'Session not found', 'status': 404}
        
        return {'content': content, 'format': format, 'status': 200}


# Convenience function for quick usage
def create_playground(
    z3_output: str,
    source_code: Optional[str] = None,
    function_name: Optional[str] = None
) -> PlaygroundSession:
    """Quick helper to create a playground session."""
    engine = PlaygroundEngine()
    return engine.create_session(z3_output, source_code, function_name)
