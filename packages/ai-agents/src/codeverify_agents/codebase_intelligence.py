"""Codebase Intelligence Engine - Persistent context-aware analysis.

This module provides deep codebase understanding that persists across reviews,
learns from historical patterns, and provides intelligent context for AI agents.

Key differentiator: Knows "This function is critical", "This pattern failed before"
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class PatternType(str, Enum):
    """Type of code pattern."""
    
    ARCHITECTURAL = "architectural"  # Design patterns, module structure
    SECURITY = "security"  # Security-related patterns
    ERROR_HANDLING = "error_handling"  # How errors are handled
    API_USAGE = "api_usage"  # How APIs are called
    NAMING = "naming"  # Naming conventions
    TESTING = "testing"  # Testing patterns
    ANTI_PATTERN = "anti_pattern"  # Known problematic patterns


class ComponentCriticality(str, Enum):
    """Criticality level of a component."""
    
    CRITICAL = "critical"  # Core business logic, security
    HIGH = "high"  # Important functionality
    MEDIUM = "medium"  # Standard functionality
    LOW = "low"  # Utilities, helpers
    UNKNOWN = "unknown"


@dataclass
class CodePattern:
    """A learned code pattern in the codebase."""
    
    id: str
    pattern_type: PatternType
    name: str
    description: str
    regex: str | None = None  # Pattern to match
    files: list[str] = field(default_factory=list)  # Files where this appears
    occurrences: int = 0
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    is_violation: bool = False  # True if this is a pattern to avoid
    bug_correlation: float = 0.0  # How often this pattern leads to bugs
    examples: list[str] = field(default_factory=list)  # Code examples
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "name": self.name,
            "description": self.description,
            "regex": self.regex,
            "files": self.files[:10],  # Limit for size
            "occurrences": self.occurrences,
            "is_violation": self.is_violation,
            "bug_correlation": round(self.bug_correlation, 3),
        }


@dataclass
class ComponentInfo:
    """Information about a code component (file, module, class)."""
    
    id: str
    path: str
    name: str
    component_type: str  # "file", "module", "class", "function"
    criticality: ComponentCriticality
    description: str | None = None
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)  # Pattern IDs
    owners: list[str] = field(default_factory=list)  # CODEOWNERS
    last_modified: datetime | None = None
    modification_frequency: float = 0.0  # Commits per month
    bug_count: int = 0
    security_findings: int = 0
    test_coverage: float | None = None
    complexity_score: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "name": self.name,
            "component_type": self.component_type,
            "criticality": self.criticality.value,
            "description": self.description,
            "dependencies": self.dependencies[:20],
            "dependents": self.dependents[:20],
            "bug_count": self.bug_count,
            "complexity_score": self.complexity_score,
        }


@dataclass
class BugCorrelation:
    """Correlation between code changes and bugs."""
    
    pattern_id: str | None
    file_path: str
    bug_id: str
    bug_title: str
    introduced_commit: str
    fix_commit: str | None = None
    days_to_fix: int | None = None
    severity: str = "medium"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "file_path": self.file_path,
            "bug_id": self.bug_id,
            "severity": self.severity,
            "days_to_fix": self.days_to_fix,
        }


@dataclass
class CodebaseContext:
    """Rich context about the codebase for AI agents."""
    
    file_path: str
    component: ComponentInfo | None
    related_patterns: list[CodePattern]
    similar_bugs: list[BugCorrelation]
    architectural_notes: list[str]
    security_considerations: list[str]
    testing_requirements: list[str]
    confidence: float  # How confident we are in this context
    
    def to_prompt_context(self) -> str:
        """Convert to a string suitable for LLM prompts."""
        parts = []
        
        if self.component:
            parts.append(f"Component: {self.component.name} ({self.component.criticality.value} criticality)")
            if self.component.description:
                parts.append(f"Description: {self.component.description}")
        
        if self.related_patterns:
            patterns_str = ", ".join(p.name for p in self.related_patterns[:5])
            parts.append(f"Patterns in this code: {patterns_str}")
        
        if self.similar_bugs:
            bugs_str = "; ".join(
                f"{b.bug_title} ({b.severity})" for b in self.similar_bugs[:3]
            )
            parts.append(f"Similar past bugs: {bugs_str}")
        
        if self.architectural_notes:
            parts.append(f"Architecture notes: {'; '.join(self.architectural_notes[:3])}")
        
        if self.security_considerations:
            parts.append(f"Security: {'; '.join(self.security_considerations[:3])}")
        
        return "\n".join(parts)


class PatternDetector:
    """Detects code patterns in source files."""
    
    # Common patterns to detect
    BUILTIN_PATTERNS = [
        {
            "id": "singleton",
            "name": "Singleton Pattern",
            "type": PatternType.ARCHITECTURAL,
            "regex": r"_instance\s*=\s*None|__new__\s*\(",
        },
        {
            "id": "factory",
            "name": "Factory Pattern",
            "type": PatternType.ARCHITECTURAL,
            "regex": r"def\s+create_\w+|class\s+\w+Factory",
        },
        {
            "id": "try_except_pass",
            "name": "Silent Exception",
            "type": PatternType.ANTI_PATTERN,
            "regex": r"except.*:\s*pass",
            "is_violation": True,
        },
        {
            "id": "sql_concat",
            "name": "SQL String Concatenation",
            "type": PatternType.SECURITY,
            "regex": r'f["\'].*SELECT.*\{|".*SELECT.*"\s*\+',
            "is_violation": True,
        },
        {
            "id": "hardcoded_secret",
            "name": "Hardcoded Secret",
            "type": PatternType.SECURITY,
            "regex": r'(?:password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']',
            "is_violation": True,
        },
        {
            "id": "generic_exception",
            "name": "Generic Exception Handler",
            "type": PatternType.ERROR_HANDLING,
            "regex": r"except\s+Exception\s*:",
        },
        {
            "id": "async_await",
            "name": "Async/Await Pattern",
            "type": PatternType.ARCHITECTURAL,
            "regex": r"async\s+def|await\s+",
        },
        {
            "id": "dependency_injection",
            "name": "Dependency Injection",
            "type": PatternType.ARCHITECTURAL,
            "regex": r"def\s+__init__\s*\([^)]*:\s*\w+[^)]*\)",
        },
    ]
    
    def __init__(self) -> None:
        self.patterns: dict[str, CodePattern] = {}
        self._load_builtin_patterns()
    
    def _load_builtin_patterns(self) -> None:
        """Load built-in pattern definitions."""
        for p in self.BUILTIN_PATTERNS:
            pattern = CodePattern(
                id=p["id"],
                pattern_type=p["type"],
                name=p["name"],
                description=f"Built-in pattern: {p['name']}",
                regex=p.get("regex"),
                is_violation=p.get("is_violation", False),
            )
            self.patterns[p["id"]] = pattern
    
    def detect_patterns(self, code: str, file_path: str) -> list[CodePattern]:
        """Detect patterns in code."""
        found_patterns = []
        
        for pattern_id, pattern in self.patterns.items():
            if pattern.regex:
                matches = re.findall(pattern.regex, code, re.MULTILINE | re.IGNORECASE)
                if matches:
                    # Update pattern stats
                    pattern.occurrences += len(matches)
                    if file_path not in pattern.files:
                        pattern.files.append(file_path)
                    pattern.last_seen = datetime.utcnow()
                    
                    # Add example
                    if len(pattern.examples) < 5:
                        pattern.examples.append(matches[0][:100])
                    
                    found_patterns.append(pattern)
        
        return found_patterns
    
    def add_custom_pattern(
        self,
        name: str,
        pattern_type: PatternType,
        regex: str,
        description: str = "",
        is_violation: bool = False,
    ) -> CodePattern:
        """Add a custom pattern to detect."""
        pattern_id = hashlib.md5(f"{name}:{regex}".encode()).hexdigest()[:12]
        
        pattern = CodePattern(
            id=pattern_id,
            pattern_type=pattern_type,
            name=name,
            description=description or f"Custom pattern: {name}",
            regex=regex,
            is_violation=is_violation,
        )
        
        self.patterns[pattern_id] = pattern
        return pattern


class DependencyAnalyzer:
    """Analyzes dependencies between components."""
    
    def __init__(self) -> None:
        self.import_cache: dict[str, list[str]] = {}
    
    def analyze_imports(self, code: str, file_path: str) -> list[str]:
        """Extract imports from code."""
        imports = []
        
        # Python imports
        import_patterns = [
            r"^import\s+(\S+)",
            r"^from\s+(\S+)\s+import",
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            imports.extend(matches)
        
        # TypeScript/JavaScript imports
        ts_pattern = r"import\s+.*from\s+['\"]([^'\"]+)['\"]"
        ts_matches = re.findall(ts_pattern, code)
        imports.extend(ts_matches)
        
        self.import_cache[file_path] = imports
        return imports
    
    def build_dependency_graph(
        self,
        files: dict[str, str],  # path -> content
    ) -> dict[str, list[str]]:
        """Build a dependency graph from files."""
        graph: dict[str, list[str]] = {}
        
        for file_path, content in files.items():
            imports = self.analyze_imports(content, file_path)
            graph[file_path] = imports
        
        return graph
    
    def find_dependents(
        self,
        component_path: str,
        graph: dict[str, list[str]],
    ) -> list[str]:
        """Find all files that depend on a component."""
        component_name = Path(component_path).stem
        dependents = []
        
        for file_path, imports in graph.items():
            if file_path == component_path:
                continue
            
            for imp in imports:
                if component_name in imp or component_path in imp:
                    dependents.append(file_path)
                    break
        
        return dependents


class BugTracker:
    """Tracks bugs and correlates them with code changes."""
    
    def __init__(self) -> None:
        self.bugs: list[BugCorrelation] = []
        self.file_bug_count: dict[str, int] = {}
        self.pattern_bug_count: dict[str, int] = {}
    
    def record_bug(
        self,
        file_path: str,
        bug_id: str,
        bug_title: str,
        introduced_commit: str,
        pattern_id: str | None = None,
        severity: str = "medium",
    ) -> BugCorrelation:
        """Record a bug correlation."""
        bug = BugCorrelation(
            pattern_id=pattern_id,
            file_path=file_path,
            bug_id=bug_id,
            bug_title=bug_title,
            introduced_commit=introduced_commit,
            severity=severity,
        )
        
        self.bugs.append(bug)
        
        # Update counts
        self.file_bug_count[file_path] = self.file_bug_count.get(file_path, 0) + 1
        if pattern_id:
            self.pattern_bug_count[pattern_id] = self.pattern_bug_count.get(pattern_id, 0) + 1
        
        return bug
    
    def get_bugs_for_file(self, file_path: str) -> list[BugCorrelation]:
        """Get bugs associated with a file."""
        return [b for b in self.bugs if b.file_path == file_path]
    
    def get_bugs_for_pattern(self, pattern_id: str) -> list[BugCorrelation]:
        """Get bugs associated with a pattern."""
        return [b for b in self.bugs if b.pattern_id == pattern_id]
    
    def get_bug_prone_files(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Get the most bug-prone files."""
        sorted_files = sorted(
            self.file_bug_count.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_files[:top_n]


class CodebaseIntelligenceEngine:
    """Main intelligence engine that provides context-aware analysis."""
    
    def __init__(
        self,
        storage_path: str | None = None,
    ) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        
        self.pattern_detector = PatternDetector()
        self.dependency_analyzer = DependencyAnalyzer()
        self.bug_tracker = BugTracker()
        
        self.components: dict[str, ComponentInfo] = {}
        self.codeowners: dict[str, list[str]] = {}
        
        # Load persisted state
        if self.storage_path and self.storage_path.exists():
            self._load_state()
    
    def index_file(
        self,
        file_path: str,
        content: str,
        commit_hash: str | None = None,
    ) -> ComponentInfo:
        """Index a file and extract intelligence."""
        # Generate component ID
        component_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(content, file_path)
        pattern_ids = [p.id for p in patterns]
        
        # Analyze dependencies
        dependencies = self.dependency_analyzer.analyze_imports(content, file_path)
        
        # Calculate complexity (simplified)
        complexity = self._calculate_complexity(content)
        
        # Determine criticality
        criticality = self._determine_criticality(file_path, patterns, dependencies)
        
        # Get or create component
        component = self.components.get(component_id)
        if component:
            # Update existing
            component.patterns = list(set(component.patterns + pattern_ids))
            component.dependencies = dependencies
            component.complexity_score = complexity
            component.last_modified = datetime.utcnow()
        else:
            # Create new
            component = ComponentInfo(
                id=component_id,
                path=file_path,
                name=Path(file_path).stem,
                component_type="file",
                criticality=criticality,
                dependencies=dependencies,
                patterns=pattern_ids,
                last_modified=datetime.utcnow(),
                complexity_score=complexity,
            )
            self.components[component_id] = component
        
        # Update dependents
        self._update_dependents(component)
        
        return component
    
    def index_repository(
        self,
        files: dict[str, str],
        repo_root: str | None = None,
    ) -> dict[str, Any]:
        """Index an entire repository."""
        indexed_count = 0
        patterns_found = set()
        
        for file_path, content in files.items():
            component = self.index_file(file_path, content)
            indexed_count += 1
            patterns_found.update(component.patterns)
        
        # Build full dependency graph
        dep_graph = self.dependency_analyzer.build_dependency_graph(files)
        
        # Update all dependents
        for file_path in files:
            component_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
            if component_id in self.components:
                self.components[component_id].dependents = self.dependency_analyzer.find_dependents(
                    file_path, dep_graph
                )
        
        # Persist state
        self._save_state()
        
        return {
            "indexed_files": indexed_count,
            "patterns_found": len(patterns_found),
            "components": len(self.components),
        }
    
    def get_context(
        self,
        file_path: str,
        code: str | None = None,
    ) -> CodebaseContext:
        """Get rich context for a file."""
        component_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
        component = self.components.get(component_id)
        
        # If we have code but no component, index it
        if not component and code:
            component = self.index_file(file_path, code)
        
        # Get related patterns
        related_patterns = []
        if component:
            for pattern_id in component.patterns:
                if pattern_id in self.pattern_detector.patterns:
                    related_patterns.append(self.pattern_detector.patterns[pattern_id])
        
        # Get similar bugs
        similar_bugs = self.bug_tracker.get_bugs_for_file(file_path)
        
        # Generate architectural notes
        architectural_notes = self._generate_architectural_notes(component)
        
        # Generate security considerations
        security_considerations = self._generate_security_notes(related_patterns)
        
        # Generate testing requirements
        testing_requirements = self._generate_testing_notes(component, related_patterns)
        
        return CodebaseContext(
            file_path=file_path,
            component=component,
            related_patterns=related_patterns,
            similar_bugs=similar_bugs,
            architectural_notes=architectural_notes,
            security_considerations=security_considerations,
            testing_requirements=testing_requirements,
            confidence=0.8 if component else 0.3,
        )
    
    def learn_from_bug(
        self,
        file_path: str,
        bug_id: str,
        bug_title: str,
        introduced_commit: str,
        severity: str = "medium",
    ) -> None:
        """Learn from a bug to improve future analysis."""
        # Find patterns in the buggy code
        component_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
        component = self.components.get(component_id)
        
        pattern_id = None
        if component and component.patterns:
            # Associate with the first anti-pattern if any
            for pid in component.patterns:
                pattern = self.pattern_detector.patterns.get(pid)
                if pattern and pattern.is_violation:
                    pattern_id = pid
                    break
        
        # Record the bug
        self.bug_tracker.record_bug(
            file_path=file_path,
            bug_id=bug_id,
            bug_title=bug_title,
            introduced_commit=introduced_commit,
            pattern_id=pattern_id,
            severity=severity,
        )
        
        # Update component bug count
        if component:
            component.bug_count += 1
            if severity == "critical":
                component.criticality = ComponentCriticality.CRITICAL
        
        # Update pattern bug correlation
        if pattern_id:
            pattern = self.pattern_detector.patterns.get(pattern_id)
            if pattern:
                total_files = len(pattern.files) or 1
                bugs_with_pattern = self.bug_tracker.pattern_bug_count.get(pattern_id, 0)
                pattern.bug_correlation = bugs_with_pattern / total_files
        
        self._save_state()
    
    def get_similar_code(
        self,
        code_snippet: str,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Find similar code in the codebase (for pattern matching)."""
        # Simplified similarity based on shared patterns
        snippet_patterns = self.pattern_detector.detect_patterns(code_snippet, "<snippet>")
        snippet_pattern_ids = {p.id for p in snippet_patterns}
        
        if not snippet_pattern_ids:
            return []
        
        similarities: list[tuple[str, float]] = []
        for component_id, component in self.components.items():
            component_patterns = set(component.patterns)
            if component_patterns:
                # Jaccard similarity
                intersection = len(snippet_pattern_ids & component_patterns)
                union = len(snippet_pattern_ids | component_patterns)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0:
                    similarities.append((component.path, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def get_statistics(self) -> dict[str, Any]:
        """Get intelligence engine statistics."""
        criticality_dist = {}
        for component in self.components.values():
            crit = component.criticality.value
            criticality_dist[crit] = criticality_dist.get(crit, 0) + 1
        
        pattern_dist = {}
        for pattern in self.pattern_detector.patterns.values():
            ptype = pattern.pattern_type.value
            pattern_dist[ptype] = pattern_dist.get(ptype, 0) + 1
        
        return {
            "total_components": len(self.components),
            "total_patterns": len(self.pattern_detector.patterns),
            "total_bugs_tracked": len(self.bug_tracker.bugs),
            "criticality_distribution": criticality_dist,
            "pattern_type_distribution": pattern_dist,
            "most_bug_prone_files": self.bug_tracker.get_bug_prone_files(5),
            "anti_patterns_found": sum(
                1 for p in self.pattern_detector.patterns.values() if p.is_violation
            ),
        }
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate code complexity score (simplified cyclomatic complexity)."""
        complexity = 1  # Base
        
        # Count control flow statements
        control_flow = [
            r"\bif\b", r"\belif\b", r"\belse\b",
            r"\bfor\b", r"\bwhile\b",
            r"\btry\b", r"\bexcept\b",
            r"\band\b", r"\bor\b",
            r"\?.*:",  # Ternary
        ]
        
        for pattern in control_flow:
            complexity += len(re.findall(pattern, code))
        
        # Normalize to 0-1 scale (assuming max reasonable complexity of 50)
        return min(complexity / 50, 1.0)
    
    def _determine_criticality(
        self,
        file_path: str,
        patterns: list[CodePattern],
        dependencies: list[str],
    ) -> ComponentCriticality:
        """Determine criticality of a component."""
        path_lower = file_path.lower()
        
        # Check path indicators
        if any(
            keyword in path_lower
            for keyword in ["auth", "security", "crypto", "payment", "secret"]
        ):
            return ComponentCriticality.CRITICAL
        
        # Check for security anti-patterns
        if any(p.is_violation and p.pattern_type == PatternType.SECURITY for p in patterns):
            return ComponentCriticality.CRITICAL
        
        # Check for high dependency count (many things depend on this)
        if len(dependencies) > 10:
            return ComponentCriticality.HIGH
        
        # Check for test files (lower criticality for review purposes)
        if "test" in path_lower or "spec" in path_lower:
            return ComponentCriticality.LOW
        
        return ComponentCriticality.MEDIUM
    
    def _update_dependents(self, component: ComponentInfo) -> None:
        """Update dependents for a component."""
        component_name = component.name
        
        for other_id, other in self.components.items():
            if other_id == component.id:
                continue
            
            # Check if other component imports this one
            for dep in other.dependencies:
                if component_name in dep:
                    if component.id not in other.dependents:
                        other.dependents.append(component.id)
                    break
    
    def _generate_architectural_notes(
        self,
        component: ComponentInfo | None,
    ) -> list[str]:
        """Generate architectural notes for context."""
        notes = []
        
        if not component:
            return notes
        
        if component.criticality == ComponentCriticality.CRITICAL:
            notes.append("⚠️ This is a CRITICAL component - changes require extra review")
        
        if len(component.dependents) > 5:
            notes.append(
                f"This component has {len(component.dependents)} dependents - "
                "changes may have wide impact"
            )
        
        if component.bug_count > 0:
            notes.append(
                f"This file has {component.bug_count} historical bugs - "
                "review carefully"
            )
        
        if component.complexity_score and component.complexity_score > 0.5:
            notes.append("High complexity - consider refactoring if making changes")
        
        return notes
    
    def _generate_security_notes(
        self,
        patterns: list[CodePattern],
    ) -> list[str]:
        """Generate security considerations."""
        notes = []
        
        security_patterns = [p for p in patterns if p.pattern_type == PatternType.SECURITY]
        
        for pattern in security_patterns:
            if pattern.is_violation:
                notes.append(f"🔴 Security issue: {pattern.name} - {pattern.description}")
            else:
                notes.append(f"Security pattern: {pattern.name}")
        
        return notes
    
    def _generate_testing_notes(
        self,
        component: ComponentInfo | None,
        patterns: list[CodePattern],
    ) -> list[str]:
        """Generate testing requirements."""
        notes = []
        
        if component and component.criticality in [
            ComponentCriticality.CRITICAL,
            ComponentCriticality.HIGH,
        ]:
            notes.append("Requires comprehensive test coverage")
        
        # Check for anti-patterns that need specific tests
        anti_patterns = [p for p in patterns if p.is_violation]
        if anti_patterns:
            notes.append("Add tests to verify anti-patterns are handled")
        
        return notes
    
    def _save_state(self) -> None:
        """Persist state to storage."""
        if not self.storage_path:
            return
        
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Save components
            components_file = self.storage_path / "components.json"
            with open(components_file, "w") as f:
                json.dump(
                    {cid: c.to_dict() for cid, c in self.components.items()},
                    f,
                    indent=2,
                )
            
            # Save patterns
            patterns_file = self.storage_path / "patterns.json"
            with open(patterns_file, "w") as f:
                json.dump(
                    {pid: p.to_dict() for pid, p in self.pattern_detector.patterns.items()},
                    f,
                    indent=2,
                )
            
            # Save bugs
            bugs_file = self.storage_path / "bugs.json"
            with open(bugs_file, "w") as f:
                json.dump(
                    [b.to_dict() for b in self.bug_tracker.bugs],
                    f,
                    indent=2,
                )
            
            logger.debug("Intelligence state saved", path=str(self.storage_path))
            
        except Exception as e:
            logger.error("Failed to save intelligence state", error=str(e))
    
    def _load_state(self) -> None:
        """Load state from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            # Load components
            components_file = self.storage_path / "components.json"
            if components_file.exists():
                with open(components_file) as f:
                    data = json.load(f)
                    for cid, cdata in data.items():
                        self.components[cid] = ComponentInfo(
                            id=cdata["id"],
                            path=cdata["path"],
                            name=cdata["name"],
                            component_type=cdata["component_type"],
                            criticality=ComponentCriticality(cdata["criticality"]),
                            description=cdata.get("description"),
                            dependencies=cdata.get("dependencies", []),
                            dependents=cdata.get("dependents", []),
                            bug_count=cdata.get("bug_count", 0),
                            complexity_score=cdata.get("complexity_score"),
                        )
            
            logger.debug("Intelligence state loaded", components=len(self.components))
            
        except Exception as e:
            logger.error("Failed to load intelligence state", error=str(e))
