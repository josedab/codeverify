"""Semantic Diff Visualization - Graph-based behavioral change visualization.

Provides visualization of behavioral changes between commits, showing
what actually changed semantically rather than just line-level diffs.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class ChangeType(str, Enum):
    """Type of semantic change."""
    BEHAVIOR_CHANGE = "behavior_change"
    SIGNATURE_CHANGE = "signature_change"
    EXCEPTION_CHANGE = "exception_change"
    RETURN_CHANGE = "return_change"
    SIDE_EFFECT_CHANGE = "side_effect_change"
    DEPENDENCY_CHANGE = "dependency_change"
    INVARIANT_CHANGE = "invariant_change"
    NO_CHANGE = "no_change"


class RiskLevel(str, Enum):
    """Risk level of a change."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SemanticNode:
    """A node in the semantic diff graph."""
    id: str
    name: str
    node_type: str  # function, class, method, variable
    file_path: str
    line_start: int
    line_end: int
    signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticEdge:
    """An edge representing a relationship between nodes."""
    source_id: str
    target_id: str
    edge_type: str  # calls, imports, extends, modifies
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorChange:
    """A detected behavioral change."""
    id: str
    change_type: ChangeType
    description: str
    before_behavior: str
    after_behavior: str
    affected_node: SemanticNode
    risk_level: RiskLevel
    evidence: list[str] = field(default_factory=list)
    suggested_tests: list[str] = field(default_factory=list)


@dataclass
class SemanticDiff:
    """Complete semantic diff between two versions."""
    base_commit: str
    head_commit: str
    nodes_added: list[SemanticNode]
    nodes_removed: list[SemanticNode]
    nodes_modified: list[SemanticNode]
    behavior_changes: list[BehaviorChange]
    call_graph_changes: list[SemanticEdge]
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationData:
    """Data for rendering the semantic diff visualization."""
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    clusters: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    layout_hints: dict[str, Any] = field(default_factory=dict)


class CodeParser:
    """Parses code to extract semantic structure."""

    def parse_python(self, code: str, file_path: str) -> list[SemanticNode]:
        """Parse Python code to extract semantic nodes."""
        nodes = []
        
        # Extract functions
        func_pattern = r'^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?:'
        for match in re.finditer(func_pattern, code, re.MULTILINE):
            name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)
            
            line_start = code[:match.start()].count('\n') + 1
            
            # Find function end (simplified)
            line_end = self._find_block_end(code, match.end())
            
            signature = f"def {name}({params})"
            if return_type:
                signature += f" -> {return_type}"
            
            nodes.append(SemanticNode(
                id=f"{file_path}:{name}",
                name=name,
                node_type="function",
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                signature=signature,
            ))
        
        # Extract classes
        class_pattern = r'^class\s+(\w+)\s*(?:\(([^)]*)\))?:'
        for match in re.finditer(class_pattern, code, re.MULTILINE):
            name = match.group(1)
            bases = match.group(2) or ""
            
            line_start = code[:match.start()].count('\n') + 1
            line_end = self._find_block_end(code, match.end())
            
            nodes.append(SemanticNode(
                id=f"{file_path}:{name}",
                name=name,
                node_type="class",
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                signature=f"class {name}({bases})" if bases else f"class {name}",
            ))
        
        return nodes

    def parse_typescript(self, code: str, file_path: str) -> list[SemanticNode]:
        """Parse TypeScript code to extract semantic nodes."""
        nodes = []
        
        # Extract functions
        func_patterns = [
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*([^{]+))?',
            r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*([^=]+))?\s*=>',
        ]
        
        for pattern in func_patterns:
            for match in re.finditer(pattern, code, re.MULTILINE):
                name = match.group(1)
                line_start = code[:match.start()].count('\n') + 1
                
                nodes.append(SemanticNode(
                    id=f"{file_path}:{name}",
                    name=name,
                    node_type="function",
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_start + 10,  # Simplified
                    signature=match.group(0)[:100],
                ))
        
        # Extract classes/interfaces
        class_pattern = r'(?:export\s+)?(?:class|interface)\s+(\w+)\s*(?:<[^>]+>)?\s*(?:extends\s+\w+)?(?:implements\s+[^{]+)?'
        for match in re.finditer(class_pattern, code, re.MULTILINE):
            name = match.group(1)
            line_start = code[:match.start()].count('\n') + 1
            
            nodes.append(SemanticNode(
                id=f"{file_path}:{name}",
                name=name,
                node_type="class",
                file_path=file_path,
                line_start=line_start,
                line_end=line_start + 20,
                signature=match.group(0)[:80],
            ))
        
        return nodes

    def _find_block_end(self, code: str, start_pos: int) -> int:
        """Find the end line of an indented block."""
        lines = code[start_pos:].split('\n')
        if not lines:
            return code[:start_pos].count('\n') + 1
        
        # Find base indentation of first content line
        base_indent = None
        line_count = 0
        
        for line in lines[1:]:  # Skip the definition line
            if line.strip():
                if base_indent is None:
                    base_indent = len(line) - len(line.lstrip())
                else:
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent < base_indent and line.strip():
                        break
            line_count += 1
        
        start_line = code[:start_pos].count('\n') + 1
        return start_line + line_count

    def extract_call_graph(self, code: str, nodes: list[SemanticNode]) -> list[SemanticEdge]:
        """Extract call relationships between nodes."""
        edges = []
        node_names = {n.name for n in nodes}
        
        for node in nodes:
            if node.node_type != "function":
                continue
            
            # Find the function body
            func_start = code.find(f"def {node.name}(")
            if func_start == -1:
                func_start = code.find(f"function {node.name}(")
            if func_start == -1:
                continue
            
            # Extract function body (simplified)
            lines = code[func_start:].split('\n')
            body = '\n'.join(lines[1:50])  # Limit body scan
            
            # Find calls to other functions
            for other_name in node_names:
                if other_name == node.name:
                    continue
                if re.search(rf'\b{other_name}\s*\(', body):
                    edges.append(SemanticEdge(
                        source_id=node.id,
                        target_id=f"{node.file_path}:{other_name}",
                        edge_type="calls",
                    ))
        
        return edges


class BehaviorAnalyzer:
    """Analyzes behavioral differences between code versions."""

    def analyze(
        self,
        before_code: str,
        after_code: str,
        before_nodes: list[SemanticNode],
        after_nodes: list[SemanticNode],
    ) -> list[BehaviorChange]:
        """Analyze behavioral changes between two versions."""
        changes = []
        
        before_map = {n.name: n for n in before_nodes}
        after_map = {n.name: n for n in after_nodes}
        
        # Check modified nodes
        for name in set(before_map.keys()) & set(after_map.keys()):
            before_node = before_map[name]
            after_node = after_map[name]
            
            # Check signature changes
            if before_node.signature != after_node.signature:
                changes.append(BehaviorChange(
                    id=f"sig-{name}",
                    change_type=ChangeType.SIGNATURE_CHANGE,
                    description=f"Signature of {name} changed",
                    before_behavior=before_node.signature or "",
                    after_behavior=after_node.signature or "",
                    affected_node=after_node,
                    risk_level=RiskLevel.HIGH,
                    evidence=[
                        f"Before: {before_node.signature}",
                        f"After: {after_node.signature}",
                    ],
                    suggested_tests=[
                        f"Test {name} with old parameter patterns",
                        f"Verify callers updated for new signature",
                    ],
                ))
            
            # Check for exception handling changes
            before_body = self._extract_body(before_code, before_node)
            after_body = self._extract_body(after_code, after_node)
            
            exception_change = self._detect_exception_change(before_body, after_body)
            if exception_change:
                changes.append(BehaviorChange(
                    id=f"exc-{name}",
                    change_type=ChangeType.EXCEPTION_CHANGE,
                    description=f"Exception handling changed in {name}",
                    before_behavior=exception_change["before"],
                    after_behavior=exception_change["after"],
                    affected_node=after_node,
                    risk_level=RiskLevel.MEDIUM,
                    evidence=exception_change["evidence"],
                ))
            
            # Check for return behavior changes
            return_change = self._detect_return_change(before_body, after_body)
            if return_change:
                changes.append(BehaviorChange(
                    id=f"ret-{name}",
                    change_type=ChangeType.RETURN_CHANGE,
                    description=f"Return behavior changed in {name}",
                    before_behavior=return_change["before"],
                    after_behavior=return_change["after"],
                    affected_node=after_node,
                    risk_level=RiskLevel.HIGH,
                    evidence=return_change["evidence"],
                ))
        
        return changes

    def _extract_body(self, code: str, node: SemanticNode) -> str:
        """Extract the body of a function/class."""
        lines = code.split('\n')
        start = max(0, node.line_start - 1)
        end = min(len(lines), node.line_end)
        return '\n'.join(lines[start:end])

    def _detect_exception_change(
        self,
        before: str,
        after: str,
    ) -> dict[str, Any] | None:
        """Detect changes in exception handling."""
        # Extract raise statements
        before_raises = set(re.findall(r'raise\s+(\w+)', before))
        after_raises = set(re.findall(r'raise\s+(\w+)', after))
        
        # Extract try/except patterns
        before_catches = set(re.findall(r'except\s+(\w+)', before))
        after_catches = set(re.findall(r'except\s+(\w+)', after))
        
        if before_raises != after_raises or before_catches != after_catches:
            return {
                "before": f"Raises: {before_raises}, Catches: {before_catches}",
                "after": f"Raises: {after_raises}, Catches: {after_catches}",
                "evidence": [
                    f"Added raises: {after_raises - before_raises}",
                    f"Removed raises: {before_raises - after_raises}",
                    f"Added catches: {after_catches - before_catches}",
                    f"Removed catches: {before_catches - after_catches}",
                ],
            }
        
        return None

    def _detect_return_change(
        self,
        before: str,
        after: str,
    ) -> dict[str, Any] | None:
        """Detect changes in return behavior."""
        # Check for None returns
        before_has_none = 'return None' in before or re.search(r'return\s*$', before, re.MULTILINE)
        after_has_none = 'return None' in after or re.search(r'return\s*$', after, re.MULTILINE)
        
        # Check for exception returns
        before_has_raise = 'raise' in before
        after_has_raise = 'raise' in after
        
        if before_has_none != after_has_none:
            return {
                "before": "Can return None" if before_has_none else "Always returns value",
                "after": "Can return None" if after_has_none else "Always returns value",
                "evidence": [
                    f"None return {'added' if after_has_none and not before_has_none else 'removed'}",
                ],
            }
        
        if not before_has_raise and after_has_raise:
            return {
                "before": "Returns normally",
                "after": "May raise exception",
                "evidence": ["Exception handling added to previously non-throwing function"],
            }
        
        return None


class SemanticDiffVisualizer:
    """Generates visualization data from semantic diffs."""

    def generate_visualization(self, diff: SemanticDiff) -> VisualizationData:
        """Generate visualization data from a semantic diff."""
        nodes = []
        edges = []
        clusters = []
        annotations = []
        
        # Add nodes
        node_ids = set()
        
        # Added nodes (green)
        for node in diff.nodes_added:
            nodes.append({
                "id": node.id,
                "label": node.name,
                "type": node.node_type,
                "status": "added",
                "color": "#4CAF50",
                "file": node.file_path,
                "line": node.line_start,
            })
            node_ids.add(node.id)
        
        # Removed nodes (red)
        for node in diff.nodes_removed:
            nodes.append({
                "id": node.id,
                "label": node.name,
                "type": node.node_type,
                "status": "removed",
                "color": "#F44336",
                "file": node.file_path,
                "line": node.line_start,
            })
            node_ids.add(node.id)
        
        # Modified nodes (yellow)
        for node in diff.nodes_modified:
            nodes.append({
                "id": node.id,
                "label": node.name,
                "type": node.node_type,
                "status": "modified",
                "color": "#FF9800",
                "file": node.file_path,
                "line": node.line_start,
            })
            node_ids.add(node.id)
        
        # Add edges
        for edge in diff.call_graph_changes:
            if edge.source_id in node_ids and edge.target_id in node_ids:
                edges.append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type,
                    "color": "#2196F3",
                })
        
        # Group by file (clusters)
        files = set(n["file"] for n in nodes)
        for file_path in files:
            file_nodes = [n["id"] for n in nodes if n["file"] == file_path]
            clusters.append({
                "id": f"cluster-{file_path}",
                "label": file_path.split("/")[-1],
                "nodes": file_nodes,
            })
        
        # Add behavior change annotations
        for change in diff.behavior_changes:
            annotations.append({
                "node_id": change.affected_node.id,
                "type": change.change_type.value,
                "risk": change.risk_level.value,
                "message": change.description,
                "details": {
                    "before": change.before_behavior,
                    "after": change.after_behavior,
                    "evidence": change.evidence,
                },
            })
        
        return VisualizationData(
            nodes=nodes,
            edges=edges,
            clusters=clusters,
            annotations=annotations,
            layout_hints={
                "direction": "LR",
                "rankSep": 100,
                "nodeSep": 50,
            },
        )

    def to_mermaid(self, viz: VisualizationData) -> str:
        """Convert visualization to Mermaid diagram."""
        lines = ["graph LR"]
        
        # Define node styles
        lines.append("    classDef added fill:#4CAF50,color:white")
        lines.append("    classDef removed fill:#F44336,color:white")
        lines.append("    classDef modified fill:#FF9800,color:white")
        
        # Add nodes
        for node in viz.nodes:
            node_id = node["id"].replace(":", "_").replace("/", "_").replace(".", "_")
            label = node["label"]
            lines.append(f"    {node_id}[{label}]:::{node['status']}")
        
        # Add edges
        for edge in viz.edges:
            source = edge["source"].replace(":", "_").replace("/", "_").replace(".", "_")
            target = edge["target"].replace(":", "_").replace("/", "_").replace(".", "_")
            lines.append(f"    {source} --> {target}")
        
        return "\n".join(lines)

    def to_dot(self, viz: VisualizationData) -> str:
        """Convert visualization to GraphViz DOT format."""
        lines = ["digraph SemanticDiff {"]
        lines.append("    rankdir=LR;")
        lines.append("    node [shape=box, style=filled];")
        
        # Color definitions
        colors = {
            "added": "#4CAF50",
            "removed": "#F44336",
            "modified": "#FF9800",
        }
        
        # Add nodes
        for node in viz.nodes:
            node_id = node["id"].replace(":", "_").replace("/", "_").replace(".", "_")
            color = colors.get(node["status"], "#CCCCCC")
            lines.append(f'    {node_id} [label="{node["label"]}", fillcolor="{color}"];')
        
        # Add edges
        for edge in viz.edges:
            source = edge["source"].replace(":", "_").replace("/", "_").replace(".", "_")
            target = edge["target"].replace(":", "_").replace("/", "_").replace(".", "_")
            lines.append(f'    {source} -> {target};')
        
        lines.append("}")
        return "\n".join(lines)

    def to_html(self, viz: VisualizationData) -> str:
        """Generate interactive HTML visualization."""
        import json
        
        return f'''
<!DOCTYPE html>
<html>
<head>
    <title>Semantic Diff Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
        .node {{ cursor: pointer; }}
        .node.added rect {{ fill: #4CAF50; }}
        .node.removed rect {{ fill: #F44336; }}
        .node.modified rect {{ fill: #FF9800; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; stroke-width: 2px; }}
        .tooltip {{ position: absolute; background: white; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }}
        .legend {{ margin-bottom: 20px; }}
        .legend-item {{ display: inline-block; margin-right: 20px; }}
        .legend-color {{ width: 20px; height: 20px; display: inline-block; margin-right: 5px; }}
    </style>
</head>
<body>
    <h1>Semantic Diff Visualization</h1>
    <div class="legend">
        <span class="legend-item"><span class="legend-color" style="background:#4CAF50"></span>Added</span>
        <span class="legend-item"><span class="legend-color" style="background:#F44336"></span>Removed</span>
        <span class="legend-item"><span class="legend-color" style="background:#FF9800"></span>Modified</span>
    </div>
    <div id="graph"></div>
    <script>
        const data = {json.dumps({"nodes": viz.nodes, "edges": viz.edges, "annotations": viz.annotations})};
        // D3 visualization code would go here
        console.log("Visualization data:", data);
    </script>
</body>
</html>
'''


class SemanticDiffAgent(BaseAgent):
    """
    Agent for generating semantic diffs between code versions.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the agent."""
        super().__init__(config)
        self._parser = CodeParser()
        self._analyzer = BehaviorAnalyzer()
        self._visualizer = SemanticDiffVisualizer()

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Generate semantic diff between two code versions.
        
        Args:
            code: The new code version
            context: Contains:
                - before_code: Previous code version
                - file_path: Path to the file
                - language: Programming language
                - base_commit: Base commit SHA
                - head_commit: Head commit SHA
                
        Returns:
            AgentResult with semantic diff and visualization
        """
        try:
            before_code = context.get("before_code", "")
            file_path = context.get("file_path", "unknown")
            language = context.get("language", "python")
            base_commit = context.get("base_commit", "base")
            head_commit = context.get("head_commit", "head")
            
            diff = await self.generate_semantic_diff(
                before_code=before_code,
                after_code=code,
                file_path=file_path,
                language=language,
                base_commit=base_commit,
                head_commit=head_commit,
            )
            
            viz = self._visualizer.generate_visualization(diff)
            
            return AgentResult(
                success=True,
                data={
                    "summary": diff.summary,
                    "behavior_changes": [
                        {
                            "type": c.change_type.value,
                            "description": c.description,
                            "risk": c.risk_level.value,
                            "before": c.before_behavior,
                            "after": c.after_behavior,
                            "evidence": c.evidence,
                        }
                        for c in diff.behavior_changes
                    ],
                    "nodes_added": len(diff.nodes_added),
                    "nodes_removed": len(diff.nodes_removed),
                    "nodes_modified": len(diff.nodes_modified),
                    "visualization": {
                        "mermaid": self._visualizer.to_mermaid(viz),
                        "dot": self._visualizer.to_dot(viz),
                    },
                },
            )
            
        except Exception as e:
            logger.error("Semantic diff failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    async def generate_semantic_diff(
        self,
        before_code: str,
        after_code: str,
        file_path: str,
        language: str,
        base_commit: str,
        head_commit: str,
    ) -> SemanticDiff:
        """Generate a semantic diff between two code versions."""
        # Parse both versions
        if language in ("python", "py"):
            before_nodes = self._parser.parse_python(before_code, file_path)
            after_nodes = self._parser.parse_python(after_code, file_path)
        else:
            before_nodes = self._parser.parse_typescript(before_code, file_path)
            after_nodes = self._parser.parse_typescript(after_code, file_path)
        
        # Categorize nodes
        before_names = {n.name for n in before_nodes}
        after_names = {n.name for n in after_nodes}
        
        added = [n for n in after_nodes if n.name not in before_names]
        removed = [n for n in before_nodes if n.name not in after_names]
        modified = [n for n in after_nodes if n.name in before_names]
        
        # Analyze behavioral changes
        behavior_changes = self._analyzer.analyze(
            before_code, after_code, before_nodes, after_nodes
        )
        
        # Extract call graph changes
        before_edges = self._parser.extract_call_graph(before_code, before_nodes)
        after_edges = self._parser.extract_call_graph(after_code, after_nodes)
        
        # Find edge differences
        before_edge_set = {(e.source_id, e.target_id) for e in before_edges}
        after_edge_set = {(e.source_id, e.target_id) for e in after_edges}
        
        new_edges = [e for e in after_edges if (e.source_id, e.target_id) not in before_edge_set]
        
        # Generate summary
        summary = {
            "total_changes": len(added) + len(removed) + len(modified),
            "functions_added": len([n for n in added if n.node_type == "function"]),
            "functions_removed": len([n for n in removed if n.node_type == "function"]),
            "functions_modified": len([n for n in modified if n.node_type == "function"]),
            "behavior_changes": len(behavior_changes),
            "high_risk_changes": len([c for c in behavior_changes if c.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)]),
        }
        
        logger.info(
            "Generated semantic diff",
            added=len(added),
            removed=len(removed),
            modified=len(modified),
            behavior_changes=len(behavior_changes),
        )
        
        return SemanticDiff(
            base_commit=base_commit,
            head_commit=head_commit,
            nodes_added=added,
            nodes_removed=removed,
            nodes_modified=modified,
            behavior_changes=behavior_changes,
            call_graph_changes=new_edges,
            summary=summary,
        )
