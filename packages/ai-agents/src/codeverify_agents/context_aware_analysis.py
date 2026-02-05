"""
Context-Aware Analysis Engine

Understands project context (architecture, coding patterns, conventions)
to provide more relevant findings with context-aware severity adjustment.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4


class ArchitectureType(str, Enum):
    """Types of software architecture."""
    MONOLITH = "monolith"
    MICROSERVICES = "microservices"
    MODULAR_MONOLITH = "modular_monolith"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    UNKNOWN = "unknown"


class ProjectType(str, Enum):
    """Types of projects."""
    WEB_FRONTEND = "web_frontend"
    WEB_BACKEND = "web_backend"
    CLI_TOOL = "cli_tool"
    LIBRARY = "library"
    API_SERVICE = "api_service"
    MOBILE_APP = "mobile_app"
    DESKTOP_APP = "desktop_app"
    DATA_PIPELINE = "data_pipeline"
    ML_PROJECT = "ml_project"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


class ConventionType(str, Enum):
    """Types of coding conventions."""
    NAMING = "naming"
    FORMATTING = "formatting"
    STRUCTURE = "structure"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    ERROR_HANDLING = "error_handling"
    LOGGING = "logging"
    SECURITY = "security"


class Severity(str, Enum):
    """Finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DetectedArchitecture:
    """Detected architecture information."""
    type: ArchitectureType
    confidence: float
    indicators: List[str]
    layers: List[str]
    components: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "layers": self.layers,
            "components": self.components,
        }


@dataclass
class DetectedPattern:
    """A detected coding pattern."""
    id: str
    name: str
    pattern_type: str
    description: str
    occurrences: int
    examples: List[str]
    file_patterns: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "occurrences": self.occurrences,
            "examples": self.examples[:3],
            "file_patterns": self.file_patterns,
            "confidence": self.confidence,
        }


@dataclass
class Convention:
    """A detected coding convention."""
    id: str
    type: ConventionType
    name: str
    description: str
    rule: str
    examples: List[str]
    adherence_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "rule": self.rule,
            "examples": self.examples[:3],
            "adherence_rate": self.adherence_rate,
        }


@dataclass
class ProjectContext:
    """Complete project context."""
    id: str
    name: str
    project_type: ProjectType
    architecture: DetectedArchitecture
    patterns: List[DetectedPattern]
    conventions: List[Convention]
    languages: Dict[str, int]  # language -> file count
    frameworks: List[str]
    dependencies: List[str]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "project_type": self.project_type.value,
            "architecture": self.architecture.to_dict(),
            "patterns": [p.to_dict() for p in self.patterns],
            "conventions": [c.to_dict() for c in self.conventions],
            "languages": self.languages,
            "frameworks": self.frameworks,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class SeverityAdjustment:
    """A severity adjustment based on context."""
    original_severity: Severity
    adjusted_severity: Severity
    reason: str
    context_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_severity": self.original_severity.value,
            "adjusted_severity": self.adjusted_severity.value,
            "reason": self.reason,
            "context_factors": self.context_factors,
        }


@dataclass
class ContextualFinding:
    """A finding with contextual information."""
    finding_id: str
    finding_type: str
    original_severity: Severity
    adjusted_severity: Severity
    adjustment: Optional[SeverityAdjustment]
    relevance_score: float
    context_notes: List[str]
    similar_patterns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "finding_id": self.finding_id,
            "finding_type": self.finding_type,
            "original_severity": self.original_severity.value,
            "adjusted_severity": self.adjusted_severity.value,
            "adjustment": self.adjustment.to_dict() if self.adjustment else None,
            "relevance_score": self.relevance_score,
            "context_notes": self.context_notes,
            "similar_patterns": self.similar_patterns,
        }


class ArchitectureDetector:
    """Detects project architecture from code structure."""

    # Architecture indicators
    MICROSERVICES_INDICATORS = [
        "docker-compose.yml",
        "kubernetes/",
        "k8s/",
        "services/",
        "microservices/",
        "api-gateway",
        "service-mesh",
    ]

    SERVERLESS_INDICATORS = [
        "serverless.yml",
        "serverless.yaml",
        "sam.yaml",
        "template.yaml",
        "functions/",
        "lambda/",
        "cloud-functions/",
    ]

    HEXAGONAL_INDICATORS = [
        "adapters/",
        "ports/",
        "domain/",
        "application/",
        "infrastructure/",
    ]

    LAYERED_INDICATORS = [
        "controllers/",
        "services/",
        "repositories/",
        "models/",
        "views/",
        "handlers/",
    ]

    def detect(
        self,
        file_paths: List[str],
        file_contents: Optional[Dict[str, str]] = None,
    ) -> DetectedArchitecture:
        """Detect architecture from file structure."""
        scores: Dict[ArchitectureType, float] = defaultdict(float)
        indicators: Dict[ArchitectureType, List[str]] = defaultdict(list)
        layers: Set[str] = set()
        components: Set[str] = set()

        # Analyze file paths
        for path in file_paths:
            path_lower = path.lower()

            # Check microservices
            for ind in self.MICROSERVICES_INDICATORS:
                if ind in path_lower:
                    scores[ArchitectureType.MICROSERVICES] += 1
                    indicators[ArchitectureType.MICROSERVICES].append(ind)

            # Check serverless
            for ind in self.SERVERLESS_INDICATORS:
                if ind in path_lower:
                    scores[ArchitectureType.SERVERLESS] += 1
                    indicators[ArchitectureType.SERVERLESS].append(ind)

            # Check hexagonal
            for ind in self.HEXAGONAL_INDICATORS:
                if ind in path_lower:
                    scores[ArchitectureType.HEXAGONAL] += 1
                    indicators[ArchitectureType.HEXAGONAL].append(ind)
                    layers.add(ind.rstrip("/"))

            # Check layered
            for ind in self.LAYERED_INDICATORS:
                if ind in path_lower:
                    scores[ArchitectureType.LAYERED] += 1
                    indicators[ArchitectureType.LAYERED].append(ind)
                    layers.add(ind.rstrip("/"))

            # Extract components from path
            parts = path.split("/")
            if len(parts) > 1:
                components.add(parts[0])

        # Determine architecture type
        if not scores:
            arch_type = ArchitectureType.MONOLITH
            confidence = 0.5
            detected_indicators = ["No specific architecture patterns detected"]
        else:
            arch_type = max(scores.keys(), key=lambda k: scores[k])
            max_score = scores[arch_type]
            total_score = sum(scores.values())
            confidence = max_score / max(1, total_score)
            detected_indicators = list(set(indicators[arch_type]))

        return DetectedArchitecture(
            type=arch_type,
            confidence=min(1.0, confidence),
            indicators=detected_indicators[:10],
            layers=list(layers)[:10],
            components=list(components)[:20],
        )


class PatternExtractor:
    """Extracts coding patterns from code."""

    # Common patterns to detect
    PATTERNS = {
        "singleton": r"class\s+\w+.*\n.*_instance\s*=\s*None",
        "factory": r"def\s+create_\w+|class\s+\w+Factory",
        "repository": r"class\s+\w+Repository|def\s+get_all|def\s+find_by",
        "service": r"class\s+\w+Service|@service",
        "controller": r"class\s+\w+Controller|@controller|@app\.(get|post|put|delete)",
        "decorator": r"def\s+\w+\(.*\):\s*\n\s+def\s+wrapper",
        "builder": r"class\s+\w+Builder|def\s+build\(self\)|\.with_\w+\(",
        "observer": r"def\s+subscribe|def\s+notify|observers\s*=|listeners\s*=",
        "strategy": r"class\s+\w+Strategy|def\s+execute\(self",
        "dependency_injection": r"def\s+__init__\(self,\s*\w+:\s*\w+",
    }

    def extract(
        self,
        file_paths: List[str],
        file_contents: Dict[str, str],
    ) -> List[DetectedPattern]:
        """Extract patterns from code."""
        patterns: List[DetectedPattern] = []
        pattern_occurrences: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for path, content in file_contents.items():
            for pattern_name, pattern_regex in self.PATTERNS.items():
                matches = re.findall(pattern_regex, content, re.MULTILINE | re.IGNORECASE)
                if matches:
                    # Get context around match
                    for match in matches[:3]:  # Limit examples
                        if isinstance(match, tuple):
                            match = match[0]
                        pattern_occurrences[pattern_name].append((path, str(match)[:100]))

        for pattern_name, occurrences in pattern_occurrences.items():
            if len(occurrences) >= 2:  # Only include if used multiple times
                patterns.append(DetectedPattern(
                    id=str(uuid4()),
                    name=pattern_name.replace("_", " ").title(),
                    pattern_type="design_pattern",
                    description=f"Detected {pattern_name} pattern usage",
                    occurrences=len(occurrences),
                    examples=[f"{path}: {ex}" for path, ex in occurrences[:3]],
                    file_patterns=[path for path, _ in occurrences[:5]],
                    confidence=min(1.0, len(occurrences) / 10),
                ))

        return patterns


class ConventionDetector:
    """Detects coding conventions."""

    def detect(
        self,
        file_paths: List[str],
        file_contents: Dict[str, str],
    ) -> List[Convention]:
        """Detect coding conventions."""
        conventions: List[Convention] = []

        # Detect naming conventions
        naming_conv = self._detect_naming_conventions(file_contents)
        if naming_conv:
            conventions.append(naming_conv)

        # Detect documentation conventions
        doc_conv = self._detect_doc_conventions(file_contents)
        if doc_conv:
            conventions.append(doc_conv)

        # Detect error handling conventions
        error_conv = self._detect_error_conventions(file_contents)
        if error_conv:
            conventions.append(error_conv)

        # Detect testing conventions
        test_conv = self._detect_test_conventions(file_paths, file_contents)
        if test_conv:
            conventions.append(test_conv)

        return conventions

    def _detect_naming_conventions(self, file_contents: Dict[str, str]) -> Optional[Convention]:
        """Detect naming conventions."""
        snake_case = 0
        camel_case = 0
        examples: List[str] = []

        for content in file_contents.values():
            # Check function definitions
            snake_funcs = re.findall(r"def\s+([a-z_][a-z0-9_]*)\s*\(", content)
            camel_funcs = re.findall(r"def\s+([a-z][a-zA-Z0-9]*)\s*\(", content)

            snake_case += len([f for f in snake_funcs if "_" in f])
            camel_case += len([f for f in camel_funcs if "_" not in f and any(c.isupper() for c in f)])

            for func in snake_funcs[:2]:
                if "_" in func:
                    examples.append(f"def {func}()")

        if snake_case > camel_case and snake_case > 5:
            return Convention(
                id=str(uuid4()),
                type=ConventionType.NAMING,
                name="Snake Case Functions",
                description="Functions use snake_case naming convention",
                rule="Function names should use snake_case",
                examples=examples[:3],
                adherence_rate=snake_case / max(1, snake_case + camel_case),
            )
        elif camel_case > snake_case and camel_case > 5:
            return Convention(
                id=str(uuid4()),
                type=ConventionType.NAMING,
                name="Camel Case Functions",
                description="Functions use camelCase naming convention",
                rule="Function names should use camelCase",
                examples=examples[:3],
                adherence_rate=camel_case / max(1, snake_case + camel_case),
            )

        return None

    def _detect_doc_conventions(self, file_contents: Dict[str, str]) -> Optional[Convention]:
        """Detect documentation conventions."""
        docstrings = 0
        functions = 0
        examples: List[str] = []

        for content in file_contents.values():
            func_matches = re.findall(
                r'def\s+(\w+)\s*\([^)]*\):\s*\n\s*"""([^"]+)"""',
                content,
                re.MULTILINE,
            )
            functions += len(re.findall(r"def\s+\w+\s*\(", content))
            docstrings += len(func_matches)

            for name, doc in func_matches[:2]:
                examples.append(f'{name}(): """{doc[:50]}..."""')

        if functions > 5:
            rate = docstrings / functions
            if rate > 0.3:
                return Convention(
                    id=str(uuid4()),
                    type=ConventionType.DOCUMENTATION,
                    name="Docstring Usage",
                    description="Functions are documented with docstrings",
                    rule="Functions should have docstrings",
                    examples=examples[:3],
                    adherence_rate=rate,
                )

        return None

    def _detect_error_conventions(self, file_contents: Dict[str, str]) -> Optional[Convention]:
        """Detect error handling conventions."""
        try_blocks = 0
        custom_exceptions = 0
        examples: List[str] = []

        for content in file_contents.values():
            try_blocks += len(re.findall(r"\btry\s*:", content))
            custom_ex = re.findall(r"class\s+(\w+Exception|\w+Error)\s*\(", content)
            custom_exceptions += len(custom_ex)

            for ex in custom_ex[:2]:
                examples.append(f"class {ex}")

        if try_blocks > 5 or custom_exceptions > 2:
            return Convention(
                id=str(uuid4()),
                type=ConventionType.ERROR_HANDLING,
                name="Structured Error Handling",
                description="Uses try/except blocks and custom exceptions",
                rule="Errors should be handled with proper exception types",
                examples=examples[:3],
                adherence_rate=min(1.0, (try_blocks + custom_exceptions * 2) / 20),
            )

        return None

    def _detect_test_conventions(
        self,
        file_paths: List[str],
        file_contents: Dict[str, str],
    ) -> Optional[Convention]:
        """Detect testing conventions."""
        test_files = [p for p in file_paths if "test" in p.lower()]
        pytest_tests = 0
        unittest_tests = 0
        examples: List[str] = []

        for path, content in file_contents.items():
            if "test" in path.lower():
                pytest_tests += len(re.findall(r"def\s+test_\w+\s*\(", content))
                unittest_tests += len(re.findall(r"class\s+Test\w+\s*\(.*TestCase", content))

                funcs = re.findall(r"def\s+(test_\w+)\s*\(", content)
                for func in funcs[:2]:
                    examples.append(f"def {func}()")

        if pytest_tests > unittest_tests and pytest_tests > 3:
            return Convention(
                id=str(uuid4()),
                type=ConventionType.TESTING,
                name="Pytest Testing",
                description="Uses pytest-style test functions",
                rule="Test functions should start with test_",
                examples=examples[:3],
                adherence_rate=min(1.0, len(test_files) / max(1, len(file_paths) / 10)),
            )
        elif unittest_tests > pytest_tests and unittest_tests > 2:
            return Convention(
                id=str(uuid4()),
                type=ConventionType.TESTING,
                name="Unittest Testing",
                description="Uses unittest.TestCase classes",
                rule="Test classes should inherit from TestCase",
                examples=examples[:3],
                adherence_rate=min(1.0, len(test_files) / max(1, len(file_paths) / 10)),
            )

        return None


class SeverityAdjuster:
    """Adjusts finding severity based on context."""

    # Severity adjustments based on context
    ADJUSTMENTS = {
        # Architecture-based adjustments
        "microservices": {
            "cross_service_call": ("increase", "Cross-service calls are critical in microservices"),
            "api_versioning": ("increase", "API versioning is important in microservices"),
            "local_variable": ("decrease", "Service-scoped variables have limited impact"),
        },
        "monolith": {
            "global_state": ("increase", "Global state is problematic in monoliths"),
            "tight_coupling": ("increase", "Tight coupling affects entire application"),
        },
        # Project type-based adjustments
        "library": {
            "breaking_change": ("increase", "Breaking changes affect library consumers"),
            "public_api": ("increase", "Public API stability is crucial"),
            "internal_only": ("decrease", "Internal changes have limited impact"),
        },
        "api_service": {
            "authentication": ("increase", "Auth issues are critical for APIs"),
            "rate_limiting": ("increase", "Rate limiting protects API services"),
        },
    }

    def adjust(
        self,
        finding_type: str,
        original_severity: Severity,
        context: ProjectContext,
    ) -> SeverityAdjustment:
        """Adjust severity based on context."""
        factors: List[str] = []
        adjustment_direction = None
        reason = "No contextual adjustment needed"

        # Check architecture-based adjustments
        arch_key = context.architecture.type.value
        if arch_key in self.ADJUSTMENTS:
            for pattern, (direction, desc) in self.ADJUSTMENTS[arch_key].items():
                if pattern in finding_type.lower():
                    adjustment_direction = direction
                    reason = desc
                    factors.append(f"Architecture: {context.architecture.type.value}")
                    break

        # Check project type-based adjustments
        proj_key = context.project_type.value
        if proj_key in self.ADJUSTMENTS:
            for pattern, (direction, desc) in self.ADJUSTMENTS[proj_key].items():
                if pattern in finding_type.lower():
                    adjustment_direction = direction
                    reason = desc
                    factors.append(f"Project type: {context.project_type.value}")
                    break

        # Calculate adjusted severity
        severity_order = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        current_idx = severity_order.index(original_severity)

        if adjustment_direction == "increase" and current_idx < len(severity_order) - 1:
            adjusted = severity_order[current_idx + 1]
        elif adjustment_direction == "decrease" and current_idx > 0:
            adjusted = severity_order[current_idx - 1]
        else:
            adjusted = original_severity

        return SeverityAdjustment(
            original_severity=original_severity,
            adjusted_severity=adjusted,
            reason=reason,
            context_factors=factors,
        )


class ContextAwareAnalyzer:
    """Main analyzer for context-aware analysis."""

    def __init__(self):
        self.architecture_detector = ArchitectureDetector()
        self.pattern_extractor = PatternExtractor()
        self.convention_detector = ConventionDetector()
        self.severity_adjuster = SeverityAdjuster()

        self.contexts: Dict[str, ProjectContext] = {}

    def analyze_project(
        self,
        project_name: str,
        file_paths: List[str],
        file_contents: Dict[str, str],
        dependencies: Optional[List[str]] = None,
    ) -> ProjectContext:
        """Analyze a project and build context."""
        # Detect architecture
        architecture = self.architecture_detector.detect(file_paths, file_contents)

        # Extract patterns
        patterns = self.pattern_extractor.extract(file_paths, file_contents)

        # Detect conventions
        conventions = self.convention_detector.detect(file_paths, file_contents)

        # Detect languages from file extensions
        languages: Dict[str, int] = defaultdict(int)
        for path in file_paths:
            ext = path.split(".")[-1] if "." in path else "unknown"
            languages[ext] += 1

        # Detect project type
        project_type = self._detect_project_type(file_paths, file_contents, languages)

        # Detect frameworks
        frameworks = self._detect_frameworks(file_contents, dependencies or [])

        context = ProjectContext(
            id=str(uuid4()),
            name=project_name,
            project_type=project_type,
            architecture=architecture,
            patterns=patterns,
            conventions=conventions,
            languages=dict(languages),
            frameworks=frameworks,
            dependencies=dependencies or [],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self.contexts[context.id] = context
        return context

    def _detect_project_type(
        self,
        file_paths: List[str],
        file_contents: Dict[str, str],
        languages: Dict[str, int],
    ) -> ProjectType:
        """Detect project type."""
        paths_lower = [p.lower() for p in file_paths]
        all_content = " ".join(file_contents.values())

        # Check for web frontend
        if "package.json" in paths_lower and ("react" in all_content.lower() or "vue" in all_content.lower()):
            return ProjectType.WEB_FRONTEND

        # Check for API service
        if "fastapi" in all_content.lower() or "flask" in all_content.lower() or "express" in all_content.lower():
            return ProjectType.API_SERVICE

        # Check for CLI tool
        if "argparse" in all_content or "click" in all_content or "commander" in all_content:
            return ProjectType.CLI_TOOL

        # Check for library
        if "setup.py" in paths_lower or "pyproject.toml" in paths_lower:
            return ProjectType.LIBRARY

        # Check for ML project
        if "tensorflow" in all_content.lower() or "pytorch" in all_content.lower() or "sklearn" in all_content.lower():
            return ProjectType.ML_PROJECT

        # Check for data pipeline
        if "airflow" in all_content.lower() or "luigi" in all_content.lower() or "prefect" in all_content.lower():
            return ProjectType.DATA_PIPELINE

        # Default to web backend
        if languages.get("py", 0) > 5:
            return ProjectType.WEB_BACKEND

        return ProjectType.UNKNOWN

    def _detect_frameworks(
        self,
        file_contents: Dict[str, str],
        dependencies: List[str],
    ) -> List[str]:
        """Detect frameworks used."""
        frameworks: Set[str] = set()
        all_content = " ".join(file_contents.values()).lower()

        framework_patterns = {
            "FastAPI": ["fastapi", "from fastapi"],
            "Flask": ["from flask", "import flask"],
            "Django": ["from django", "import django"],
            "React": ["from react", "import react"],
            "Vue": ["from vue", "import vue"],
            "Express": ["require('express')", "from 'express'"],
            "Pytest": ["import pytest", "from pytest"],
            "SQLAlchemy": ["from sqlalchemy", "import sqlalchemy"],
            "Pydantic": ["from pydantic", "import pydantic"],
            "Celery": ["from celery", "import celery"],
            "Redis": ["import redis", "from redis"],
        }

        for framework, patterns in framework_patterns.items():
            for pattern in patterns:
                if pattern.lower() in all_content:
                    frameworks.add(framework)
                    break

        # Check dependencies
        for dep in dependencies:
            dep_lower = dep.lower()
            for framework in framework_patterns.keys():
                if framework.lower() in dep_lower:
                    frameworks.add(framework)

        return list(frameworks)

    def adjust_finding_severity(
        self,
        finding_id: str,
        finding_type: str,
        original_severity: str,
        context_id: str,
        code_snippet: Optional[str] = None,
    ) -> ContextualFinding:
        """Adjust finding severity based on context."""
        context = self.contexts.get(context_id)
        if not context:
            # Return unadjusted finding
            sev = Severity(original_severity) if original_severity in [s.value for s in Severity] else Severity.MEDIUM
            return ContextualFinding(
                finding_id=finding_id,
                finding_type=finding_type,
                original_severity=sev,
                adjusted_severity=sev,
                adjustment=None,
                relevance_score=1.0,
                context_notes=["No project context available"],
                similar_patterns=[],
            )

        sev = Severity(original_severity) if original_severity in [s.value for s in Severity] else Severity.MEDIUM

        # Get severity adjustment
        adjustment = self.severity_adjuster.adjust(finding_type, sev, context)

        # Calculate relevance score
        relevance = self._calculate_relevance(finding_type, context)

        # Find similar patterns
        similar = self._find_similar_patterns(finding_type, context)

        # Generate context notes
        notes = self._generate_context_notes(finding_type, context)

        return ContextualFinding(
            finding_id=finding_id,
            finding_type=finding_type,
            original_severity=sev,
            adjusted_severity=adjustment.adjusted_severity,
            adjustment=adjustment if adjustment.adjusted_severity != sev else None,
            relevance_score=relevance,
            context_notes=notes,
            similar_patterns=similar,
        )

    def _calculate_relevance(self, finding_type: str, context: ProjectContext) -> float:
        """Calculate relevance score for a finding."""
        relevance = 1.0

        # Adjust based on project type
        type_relevance = {
            ProjectType.API_SERVICE: ["security", "auth", "validation", "rate"],
            ProjectType.LIBRARY: ["api", "breaking", "deprecat", "public"],
            ProjectType.WEB_FRONTEND: ["xss", "inject", "render", "state"],
            ProjectType.ML_PROJECT: ["data", "model", "train", "accuracy"],
        }

        finding_lower = finding_type.lower()
        proj_keywords = type_relevance.get(context.project_type, [])

        for keyword in proj_keywords:
            if keyword in finding_lower:
                relevance += 0.2

        return min(1.5, relevance)

    def _find_similar_patterns(self, finding_type: str, context: ProjectContext) -> List[str]:
        """Find similar patterns in the project."""
        similar: List[str] = []
        finding_lower = finding_type.lower()

        for pattern in context.patterns:
            if any(word in finding_lower for word in pattern.name.lower().split()):
                similar.append(pattern.name)

        return similar[:5]

    def _generate_context_notes(self, finding_type: str, context: ProjectContext) -> List[str]:
        """Generate context notes for a finding."""
        notes: List[str] = []

        notes.append(f"Project type: {context.project_type.value}")
        notes.append(f"Architecture: {context.architecture.type.value}")

        if context.frameworks:
            notes.append(f"Frameworks: {', '.join(context.frameworks[:3])}")

        return notes

    def get_context(self, context_id: str) -> Optional[ProjectContext]:
        """Get a project context by ID."""
        return self.contexts.get(context_id)

    def list_contexts(self) -> List[ProjectContext]:
        """List all project contexts."""
        return list(self.contexts.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        if not self.contexts:
            return {
                "total_contexts": 0,
                "architecture_types": {},
                "project_types": {},
                "total_patterns": 0,
                "total_conventions": 0,
            }

        arch_types: Dict[str, int] = defaultdict(int)
        proj_types: Dict[str, int] = defaultdict(int)
        total_patterns = 0
        total_conventions = 0

        for context in self.contexts.values():
            arch_types[context.architecture.type.value] += 1
            proj_types[context.project_type.value] += 1
            total_patterns += len(context.patterns)
            total_conventions += len(context.conventions)

        return {
            "total_contexts": len(self.contexts),
            "architecture_types": dict(arch_types),
            "project_types": dict(proj_types),
            "total_patterns": total_patterns,
            "total_conventions": total_conventions,
        }
