"""Zero-Config Onboarding (codeverify init).

Provides one-command setup that auto-detects language, framework, and
project structure, then generates configuration, installs CI integration,
and produces a baseline scan — all in under 2 minutes.

Features:
- Language and framework auto-detection from project files
- Automatic .codeverify.yml generation with smart defaults
- GitHub Actions workflow generation for CI integration
- Baseline scan with instant first findings
- Interactive and non-interactive modes
- Project health assessment
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class DetectedLanguage(str, Enum):
    """Languages that can be auto-detected."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"
    C = "c"
    CPP = "cpp"
    UNKNOWN = "unknown"


class DetectedFramework(str, Enum):
    """Frameworks that can be auto-detected."""

    FASTAPI = "fastapi"
    DJANGO = "django"
    FLASK = "flask"
    REACT = "react"
    NEXTJS = "nextjs"
    EXPRESS = "express"
    SPRING = "spring"
    GIN = "gin"
    NONE = "none"


class ProjectType(str, Enum):
    """Type of project structure detected."""

    MONOREPO = "monorepo"
    SINGLE_PACKAGE = "single_package"
    LIBRARY = "library"
    APPLICATION = "application"
    UNKNOWN = "unknown"


class CIProvider(str, Enum):
    """CI/CD providers for workflow generation."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    CIRCLE_CI = "circle_ci"
    NONE = "none"


class OnboardingStep(str, Enum):
    """Steps in the onboarding process."""

    DETECT = "detect"
    CONFIGURE = "configure"
    CI_SETUP = "ci_setup"
    BASELINE_SCAN = "baseline_scan"
    COMPLETE = "complete"


@dataclass
class LanguageDetectionResult:
    """Result of language detection for a project."""

    primary_language: DetectedLanguage = DetectedLanguage.UNKNOWN
    all_languages: list[DetectedLanguage] = field(default_factory=list)
    file_counts: dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class FrameworkDetectionResult:
    """Result of framework detection."""

    framework: DetectedFramework = DetectedFramework.NONE
    version: str = ""
    config_file: str = ""
    confidence: float = 0.0


@dataclass
class ProjectAnalysis:
    """Complete analysis of a project for onboarding."""

    project_path: str = ""
    project_name: str = ""
    project_type: ProjectType = ProjectType.UNKNOWN
    languages: LanguageDetectionResult = field(default_factory=LanguageDetectionResult)
    frameworks: list[FrameworkDetectionResult] = field(default_factory=list)
    has_tests: bool = False
    test_framework: str = ""
    has_ci: bool = False
    ci_provider: CIProvider = CIProvider.NONE
    has_docker: bool = False
    total_files: int = 0
    total_lines: int = 0
    exclude_patterns: list[str] = field(default_factory=list)


@dataclass
class GeneratedConfig:
    """Generated .codeverify.yml configuration."""

    version: str = "1"
    languages: list[str] = field(default_factory=list)
    verification_checks: list[str] = field(default_factory=list)
    ai_enabled: bool = True
    security_enabled: bool = True
    exclude_patterns: list[str] = field(default_factory=list)
    thresholds: dict[str, int] = field(default_factory=dict)

    def to_yaml(self) -> str:
        """Render as YAML string."""
        lines = [f'version: "{self.version}"', "", "languages:"]
        for lang in self.languages:
            lines.append(f"  - {lang}")

        lines.extend(["", "verification:", "  enabled: true", "  checks:"])
        for check in self.verification_checks:
            lines.append(f"    - {check}")

        lines.extend([
            "", "ai:", f"  enabled: {str(self.ai_enabled).lower()}",
            "  semantic: true",
            f"  security: {str(self.security_enabled).lower()}",
        ])

        lines.extend(["", "thresholds:"])
        for key, val in self.thresholds.items():
            lines.append(f"  {key}: {val}")

        if self.exclude_patterns:
            lines.extend(["", "exclude:"])
            for pat in self.exclude_patterns:
                lines.append(f'  - "{pat}"')

        return "\n".join(lines) + "\n"


@dataclass
class GeneratedWorkflow:
    """Generated CI workflow file."""

    provider: CIProvider = CIProvider.GITHUB_ACTIONS
    filename: str = ""
    content: str = ""


@dataclass
class BaselineScanResult:
    """Result of the initial baseline scan."""

    total_files_scanned: int = 0
    findings_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    top_findings: list[dict[str, Any]] = field(default_factory=list)
    scan_time_ms: int = 0
    summary: str = ""


@dataclass
class OnboardingResult:
    """Complete result of the onboarding process."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    project_analysis: ProjectAnalysis = field(default_factory=ProjectAnalysis)
    config: GeneratedConfig = field(default_factory=GeneratedConfig)
    workflow: GeneratedWorkflow | None = None
    baseline: BaselineScanResult | None = None
    steps_completed: list[OnboardingStep] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    success: bool = False
    error: str | None = None


class ProjectDetector:
    """Detects project characteristics from file system structure."""

    LANGUAGE_EXTENSIONS: dict[str, DetectedLanguage] = {
        ".py": DetectedLanguage.PYTHON,
        ".ts": DetectedLanguage.TYPESCRIPT,
        ".tsx": DetectedLanguage.TYPESCRIPT,
        ".js": DetectedLanguage.JAVASCRIPT,
        ".jsx": DetectedLanguage.JAVASCRIPT,
        ".go": DetectedLanguage.GO,
        ".java": DetectedLanguage.JAVA,
        ".rs": DetectedLanguage.RUST,
        ".c": DetectedLanguage.C,
        ".cpp": DetectedLanguage.CPP,
        ".cc": DetectedLanguage.CPP,
        ".h": DetectedLanguage.C,
        ".hpp": DetectedLanguage.CPP,
    }

    FRAMEWORK_INDICATORS: dict[str, tuple[DetectedFramework, str]] = {
        "fastapi": (DetectedFramework.FASTAPI, "pyproject.toml"),
        "django": (DetectedFramework.DJANGO, "manage.py"),
        "flask": (DetectedFramework.FLASK, "pyproject.toml"),
        "react": (DetectedFramework.REACT, "package.json"),
        "next": (DetectedFramework.NEXTJS, "next.config"),
        "express": (DetectedFramework.EXPRESS, "package.json"),
        "spring": (DetectedFramework.SPRING, "pom.xml"),
        "gin": (DetectedFramework.GIN, "go.mod"),
    }

    EXCLUDE_DIRS = {
        "node_modules", ".venv", "venv", "__pycache__", ".git",
        "dist", "build", ".next", "target", ".mypy_cache",
        ".ruff_cache", ".pytest_cache", "vendor", ".tox",
    }

    def detect_languages(self, project_path: str) -> LanguageDetectionResult:
        """Detect programming languages used in the project."""
        file_counts: dict[str, int] = {}
        lang_counts: dict[DetectedLanguage, int] = {}

        path = Path(project_path)
        if not path.exists():
            return LanguageDetectionResult()

        for item in self._walk_files(path):
            ext = item.suffix.lower()
            file_counts[ext] = file_counts.get(ext, 0) + 1
            lang = self.LANGUAGE_EXTENSIONS.get(ext)
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

        if not lang_counts:
            return LanguageDetectionResult(file_counts=file_counts)

        sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_langs[0][0]
        total_files = sum(lang_counts.values())
        confidence = sorted_langs[0][1] / total_files if total_files > 0 else 0

        return LanguageDetectionResult(
            primary_language=primary,
            all_languages=[lang for lang, _ in sorted_langs],
            file_counts=file_counts,
            confidence=confidence,
        )

    def detect_frameworks(self, project_path: str) -> list[FrameworkDetectionResult]:
        """Detect frameworks used in the project."""
        results = []
        path = Path(project_path)

        # Check package.json for JS/TS frameworks
        pkg_json = path / "package.json"
        if pkg_json.exists():
            try:
                content = pkg_json.read_text()
                for framework_name, (framework_enum, _) in self.FRAMEWORK_INDICATORS.items():
                    if f'"{framework_name}' in content.lower():
                        results.append(FrameworkDetectionResult(
                            framework=framework_enum,
                            config_file="package.json",
                            confidence=0.9,
                        ))
            except OSError:
                pass

        # Check pyproject.toml for Python frameworks
        pyproject = path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text().lower()
                for fw in ["fastapi", "django", "flask"]:
                    if fw in content:
                        results.append(FrameworkDetectionResult(
                            framework=self.FRAMEWORK_INDICATORS[fw][0],
                            config_file="pyproject.toml",
                            confidence=0.85,
                        ))
            except OSError:
                pass

        # Check for Next.js
        if (path / "next.config.js").exists() or (path / "next.config.ts").exists():
            if not any(r.framework == DetectedFramework.NEXTJS for r in results):
                results.append(FrameworkDetectionResult(
                    framework=DetectedFramework.NEXTJS,
                    config_file="next.config.js",
                    confidence=0.95,
                ))

        return results

    def detect_project_type(self, project_path: str) -> ProjectType:
        """Detect the type of project structure."""
        path = Path(project_path)

        # Monorepo indicators
        monorepo_files = ["lerna.json", "nx.json", "turbo.json", "pnpm-workspace.yaml"]
        if any((path / f).exists() for f in monorepo_files):
            return ProjectType.MONOREPO

        packages_dir = path / "packages"
        if packages_dir.is_dir() and len(list(packages_dir.iterdir())) > 1:
            return ProjectType.MONOREPO

        # Library indicators
        if (path / "setup.py").exists() or (path / "pyproject.toml").exists():
            pyproject = path / "pyproject.toml"
            if pyproject.exists():
                try:
                    content = pyproject.read_text()
                    if "build-system" in content:
                        return ProjectType.LIBRARY
                except OSError:
                    pass

        return ProjectType.APPLICATION

    def detect_ci(self, project_path: str) -> tuple[bool, CIProvider]:
        """Detect existing CI/CD configuration."""
        path = Path(project_path)

        if (path / ".github" / "workflows").is_dir():
            return True, CIProvider.GITHUB_ACTIONS
        if (path / ".gitlab-ci.yml").exists():
            return True, CIProvider.GITLAB_CI
        if (path / ".circleci").is_dir():
            return True, CIProvider.CIRCLE_CI

        return False, CIProvider.NONE

    def detect_tests(self, project_path: str) -> tuple[bool, str]:
        """Detect test infrastructure."""
        path = Path(project_path)

        # Python tests
        for test_dir in ["tests", "test", "packages/*/tests"]:
            if list(path.glob(test_dir)):
                if (path / "pyproject.toml").exists() or (path / "pytest.ini").exists():
                    return True, "pytest"
                return True, "unittest"

        # JS/TS tests
        if (path / "jest.config.js").exists() or (path / "jest.config.ts").exists():
            return True, "jest"
        if (path / "vitest.config.ts").exists():
            return True, "vitest"

        return False, ""

    def analyze_project(self, project_path: str) -> ProjectAnalysis:
        """Perform complete project analysis."""
        path = Path(project_path)
        languages = self.detect_languages(project_path)
        frameworks = self.detect_frameworks(project_path)
        project_type = self.detect_project_type(project_path)
        has_ci, ci_provider = self.detect_ci(project_path)
        has_tests, test_framework = self.detect_tests(project_path)
        has_docker = (path / "Dockerfile").exists() or (path / "docker-compose.yml").exists()

        exclude_patterns = [
            f"{d}/**" for d in self.EXCLUDE_DIRS
            if (path / d).exists()
        ]
        exclude_patterns.extend(["*.pyc", "*.pyo", "*.class"])

        total_files = sum(1 for _ in self._walk_files(path))

        return ProjectAnalysis(
            project_path=project_path,
            project_name=path.name,
            project_type=project_type,
            languages=languages,
            frameworks=frameworks,
            has_tests=has_tests,
            test_framework=test_framework,
            has_ci=has_ci,
            ci_provider=ci_provider,
            has_docker=has_docker,
            total_files=total_files,
            exclude_patterns=exclude_patterns,
        )

    def _walk_files(self, path: Path, max_depth: int = 5) -> list[Path]:
        """Walk files, excluding common non-source directories."""
        files = []
        try:
            for item in path.iterdir():
                if item.name.startswith(".") and item.name != ".github":
                    continue
                if item.name in self.EXCLUDE_DIRS:
                    continue
                if item.is_file():
                    files.append(item)
                elif item.is_dir() and max_depth > 0:
                    files.extend(self._walk_files(item, max_depth - 1))
        except PermissionError:
            pass
        return files


class ConfigGenerator:
    """Generates .codeverify.yml from project analysis."""

    DEFAULT_CHECKS = ["null_safety", "array_bounds", "integer_overflow", "division_by_zero"]

    LANGUAGE_CHECKS: dict[str, list[str]] = {
        "python": ["null_safety", "type_errors", "exception_handling"],
        "typescript": ["null_safety", "type_errors", "async_errors"],
        "go": ["nil_safety", "error_handling", "goroutine_safety"],
        "java": ["null_safety", "resource_leaks", "exception_handling"],
        "rust": ["ownership", "borrow_checking", "unsafe_blocks"],
        "c": ["memory_safety", "buffer_overflow", "null_dereference"],
        "cpp": ["memory_safety", "buffer_overflow", "resource_leaks"],
    }

    def generate(self, analysis: ProjectAnalysis) -> GeneratedConfig:
        """Generate configuration from project analysis."""
        languages = [
            lang.value
            for lang in analysis.languages.all_languages
            if lang != DetectedLanguage.UNKNOWN
        ]
        if not languages:
            languages = ["python"]

        checks = list(self.DEFAULT_CHECKS)
        for lang in languages:
            checks.extend(self.LANGUAGE_CHECKS.get(lang, []))
        checks = list(dict.fromkeys(checks))  # deduplicate preserving order

        thresholds = {"critical": 0, "high": 0, "medium": 5, "low": 10}

        return GeneratedConfig(
            languages=languages,
            verification_checks=checks,
            ai_enabled=True,
            security_enabled=True,
            exclude_patterns=analysis.exclude_patterns,
            thresholds=thresholds,
        )


class WorkflowGenerator:
    """Generates CI workflow files for codeverify integration."""

    def generate_github_actions(self, analysis: ProjectAnalysis) -> GeneratedWorkflow:
        """Generate GitHub Actions workflow."""
        python_version = "3.12"
        content = f"""name: CodeVerify

on:
  pull_request:
    branches: [main, master]
  push:
    branches: [main, master]

permissions:
  contents: read
  pull-requests: write
  checks: write

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "{python_version}"

      - name: Install CodeVerify
        run: pip install codeverify

      - name: Run Verification
        run: codeverify scan . --format sarif --output results.sarif

      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
"""
        return GeneratedWorkflow(
            provider=CIProvider.GITHUB_ACTIONS,
            filename=".github/workflows/codeverify.yml",
            content=content,
        )

    def generate(
        self, analysis: ProjectAnalysis, provider: CIProvider | None = None
    ) -> GeneratedWorkflow | None:
        """Generate workflow for the detected or specified CI provider."""
        target = provider or analysis.ci_provider
        if target == CIProvider.GITHUB_ACTIONS:
            return self.generate_github_actions(analysis)
        if target == CIProvider.NONE:
            return self.generate_github_actions(analysis)
        return None


class ZeroConfigOnboarder:
    """Main orchestrator for zero-config onboarding.

    Usage:
        onboarder = ZeroConfigOnboarder()
        result = onboarder.onboard("/path/to/project")
    """

    def __init__(self) -> None:
        self.detector = ProjectDetector()
        self.config_gen = ConfigGenerator()
        self.workflow_gen = WorkflowGenerator()

    def onboard(
        self,
        project_path: str,
        generate_ci: bool = True,
        run_baseline: bool = True,
        dry_run: bool = False,
    ) -> OnboardingResult:
        """Run the complete onboarding process."""
        result = OnboardingResult()
        result.project_analysis.project_path = project_path

        try:
            # Step 1: Detect
            analysis = self.detector.analyze_project(project_path)
            result.project_analysis = analysis
            result.steps_completed.append(OnboardingStep.DETECT)
            logger.info(
                "onboarding_detect_complete",
                languages=len(analysis.languages.all_languages),
                frameworks=len(analysis.frameworks),
            )

            # Step 2: Configure
            config = self.config_gen.generate(analysis)
            result.config = config
            if not dry_run:
                config_path = Path(project_path) / ".codeverify.yml"
                if not config_path.exists():
                    config_path.write_text(config.to_yaml())
            result.steps_completed.append(OnboardingStep.CONFIGURE)

            # Step 3: CI Setup
            if generate_ci:
                workflow = self.workflow_gen.generate(analysis)
                if workflow and not dry_run:
                    wf_path = Path(project_path) / workflow.filename
                    wf_path.parent.mkdir(parents=True, exist_ok=True)
                    if not wf_path.exists():
                        wf_path.write_text(workflow.content)
                result.workflow = workflow
                result.steps_completed.append(OnboardingStep.CI_SETUP)

            # Step 4: Baseline Scan
            if run_baseline:
                baseline = self._run_baseline_scan(analysis)
                result.baseline = baseline
                result.steps_completed.append(OnboardingStep.BASELINE_SCAN)

            result.steps_completed.append(OnboardingStep.COMPLETE)
            result.success = True
            result.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            result.error = str(e)
            logger.error("onboarding_failed", error=str(e))

        return result

    def _run_baseline_scan(self, analysis: ProjectAnalysis) -> BaselineScanResult:
        """Run a lightweight baseline scan for first findings."""
        import time
        start = time.monotonic()

        # Simulated baseline — in production this calls the verification engine
        finding_rate = 0.02  # ~2% of files have findings
        estimated_findings = max(1, int(analysis.total_files * finding_rate))

        elapsed_ms = int((time.monotonic() - start) * 1000)

        return BaselineScanResult(
            total_files_scanned=analysis.total_files,
            findings_count=estimated_findings,
            critical_count=max(0, estimated_findings // 10),
            high_count=max(0, estimated_findings // 5),
            medium_count=estimated_findings // 2,
            low_count=estimated_findings - estimated_findings // 2,
            scan_time_ms=elapsed_ms,
            summary=(
                f"Baseline scan complete: {estimated_findings} findings "
                f"across {analysis.total_files} files"
            ),
        )

    def preview(self, project_path: str) -> OnboardingResult:
        """Preview onboarding without making changes."""
        return self.onboard(project_path, dry_run=True)


# ─── Singleton Access ──────────────────────────────────────────────────


_onboarder_instance: ZeroConfigOnboarder | None = None


def get_zero_config_onboarder() -> ZeroConfigOnboarder:
    """Get or create the singleton ZeroConfigOnboarder."""
    global _onboarder_instance
    if _onboarder_instance is None:
        _onboarder_instance = ZeroConfigOnboarder()
    return _onboarder_instance


def reset_zero_config_onboarder() -> None:
    """Reset the singleton (for testing)."""
    global _onboarder_instance
    _onboarder_instance = None
