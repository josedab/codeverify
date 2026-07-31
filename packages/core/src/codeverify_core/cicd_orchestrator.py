"""CI/CD Pipeline Orchestrator.

First-class integrations with multiple CI/CD platforms (CircleCI, Jenkins,
GitLab CI, Azure DevOps, Bitbucket Pipelines) with configurable quality
gates and merge blocking.

Features:
- Multi-platform CI/CD provider abstraction
- Configurable quality gates with thresholds
- Merge blocking with warning-only mode
- Status reporting per platform
- Pipeline configuration generation
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CICDPlatform(str, Enum):
    """Supported CI/CD platforms."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    CIRCLECI = "circleci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    BITBUCKET_PIPELINES = "bitbucket_pipelines"


class GateResult(str, Enum):
    """Result of a quality gate check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateMode(str, Enum):
    """Enforcement mode for quality gates."""

    ENFORCE = "enforce"
    WARN_ONLY = "warn_only"
    DISABLED = "disabled"


class StatusState(str, Enum):
    """Status states for CI/CD reporting."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class QualityThresholds:
    """Thresholds for quality gate evaluation."""

    max_critical: int = 0
    max_high: int = 0
    max_medium: int = 5
    max_low: int = 10
    min_trust_score: float = 0.0
    min_coverage: float = 0.0
    max_ai_code_ratio: float = 1.0


@dataclass
class QualityGate:
    """A quality gate configuration."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "default"
    mode: GateMode = GateMode.ENFORCE
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    block_merge: bool = True
    required_checks: list[str] = field(
        default_factory=lambda: ["null_safety", "bounds", "overflow"]
    )

    @classmethod
    def strict(cls) -> QualityGate:
        return cls(
            name="strict",
            thresholds=QualityThresholds(max_critical=0, max_high=0, max_medium=0, max_low=5),
            block_merge=True,
        )

    @classmethod
    def lenient(cls) -> QualityGate:
        return cls(
            name="lenient",
            mode=GateMode.WARN_ONLY,
            thresholds=QualityThresholds(max_critical=0, max_high=3, max_medium=10, max_low=50),
            block_merge=False,
        )


@dataclass
class GateEvaluation:
    """Result of evaluating a quality gate."""

    gate_name: str = ""
    result: GateResult = GateResult.SKIPPED
    details: list[dict[str, Any]] = field(default_factory=list)
    should_block: bool = False
    summary: str = ""

    def add_check(self, name: str, passed: bool, message: str = "") -> None:
        self.details.append({"check": name, "passed": passed, "message": message})


@dataclass
class PipelineStatus:
    """Status to report back to the CI/CD platform."""

    state: StatusState = StatusState.PENDING
    title: str = "CodeVerify"
    description: str = ""
    target_url: str = ""
    context: str = "codeverify/verification"
    findings_summary: dict[str, int] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Generated pipeline configuration for a CI/CD platform."""

    platform: CICDPlatform = CICDPlatform.GITHUB_ACTIONS
    config_content: str = ""
    file_name: str = ""
    instructions: str = ""


class QualityGateEvaluator:
    """Evaluates verification results against quality gates."""

    def evaluate(
        self,
        gate: QualityGate,
        critical: int = 0,
        high: int = 0,
        medium: int = 0,
        low: int = 0,
        trust_score: float = 100.0,
        coverage: float = 1.0,
    ) -> GateEvaluation:
        """Evaluate findings against a quality gate."""
        if gate.mode == GateMode.DISABLED:
            return GateEvaluation(
                gate_name=gate.name, result=GateResult.SKIPPED, summary="Gate disabled"
            )

        evaluation = GateEvaluation(gate_name=gate.name)
        all_passed = True

        checks = [
            (
                "critical_findings",
                critical <= gate.thresholds.max_critical,
                f"{critical} critical (max: {gate.thresholds.max_critical})",
            ),
            (
                "high_findings",
                high <= gate.thresholds.max_high,
                f"{high} high (max: {gate.thresholds.max_high})",
            ),
            (
                "medium_findings",
                medium <= gate.thresholds.max_medium,
                f"{medium} medium (max: {gate.thresholds.max_medium})",
            ),
            (
                "low_findings",
                low <= gate.thresholds.max_low,
                f"{low} low (max: {gate.thresholds.max_low})",
            ),
        ]

        if gate.thresholds.min_trust_score > 0:
            checks.append(
                (
                    "trust_score",
                    trust_score >= gate.thresholds.min_trust_score,
                    f"Score: {trust_score:.1f} (min: {gate.thresholds.min_trust_score})",
                )
            )

        if gate.thresholds.min_coverage > 0:
            checks.append(
                (
                    "coverage",
                    coverage >= gate.thresholds.min_coverage,
                    f"Coverage: {coverage:.0%} (min: {gate.thresholds.min_coverage:.0%})",
                )
            )

        for name, passed, message in checks:
            evaluation.add_check(name, passed, message)
            if not passed:
                all_passed = False

        if all_passed:
            evaluation.result = GateResult.PASSED
            evaluation.summary = "All quality gate checks passed"
        elif gate.mode == GateMode.WARN_ONLY:
            evaluation.result = GateResult.WARNING
            evaluation.summary = "Quality gate checks have warnings (non-blocking)"
        else:
            evaluation.result = GateResult.FAILED
            evaluation.should_block = gate.block_merge
            evaluation.summary = "Quality gate checks failed"

        return evaluation


class PipelineConfigGenerator:
    """Generates CI/CD pipeline configurations for each platform."""

    def generate(self, platform: CICDPlatform, gate: QualityGate | None = None) -> PipelineConfig:
        generators = {
            CICDPlatform.GITHUB_ACTIONS: self._github_actions,
            CICDPlatform.GITLAB_CI: self._gitlab_ci,
            CICDPlatform.CIRCLECI: self._circleci,
            CICDPlatform.JENKINS: self._jenkins,
            CICDPlatform.AZURE_DEVOPS: self._azure_devops,
            CICDPlatform.BITBUCKET_PIPELINES: self._bitbucket,
        }
        gen = generators.get(platform)
        if gen is None:
            return PipelineConfig(
                platform=platform, config_content="", instructions="Unsupported platform"
            )
        return gen(gate)

    def _github_actions(self, gate: QualityGate | None) -> PipelineConfig:
        config = """name: CodeVerify
on: [pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: codeverify/action@v1
        with:
          mode: verify
          gate: {gate_name}
""".format(gate_name=gate.name if gate else "default")
        return PipelineConfig(
            platform=CICDPlatform.GITHUB_ACTIONS,
            config_content=config,
            file_name=".github/workflows/codeverify.yml",
            instructions="Add this file to your repository.",
        )

    def _gitlab_ci(self, gate: QualityGate | None) -> PipelineConfig:
        config = """codeverify:
  stage: test
  image: codeverify/cli:latest
  script:
    - codeverify verify --gate {gate_name}
  only:
    - merge_requests
""".format(gate_name=gate.name if gate else "default")
        return PipelineConfig(
            platform=CICDPlatform.GITLAB_CI,
            config_content=config,
            file_name=".gitlab-ci.yml",
            instructions="Add this job to your .gitlab-ci.yml file.",
        )

    def _circleci(self, gate: QualityGate | None) -> PipelineConfig:
        config = """version: 2.1
jobs:
  codeverify:
    docker:
      - image: codeverify/cli:latest
    steps:
      - checkout
      - run: codeverify verify --gate {gate_name}
workflows:
  verify:
    jobs:
      - codeverify
""".format(gate_name=gate.name if gate else "default")
        return PipelineConfig(
            platform=CICDPlatform.CIRCLECI,
            config_content=config,
            file_name=".circleci/config.yml",
            instructions="Add this to your CircleCI configuration.",
        )

    def _jenkins(self, gate: QualityGate | None) -> PipelineConfig:
        config = """pipeline {{
    agent any
    stages {{
        stage('CodeVerify') {{
            steps {{
                sh 'codeverify verify --gate {gate_name}'
            }}
        }}
    }}
}}
""".format(gate_name=gate.name if gate else "default")
        return PipelineConfig(
            platform=CICDPlatform.JENKINS,
            config_content=config,
            file_name="Jenkinsfile",
            instructions="Add this stage to your Jenkinsfile.",
        )

    def _azure_devops(self, gate: QualityGate | None) -> PipelineConfig:
        config = """trigger:
  - main
pool:
  vmImage: 'ubuntu-latest'
steps:
  - script: |
      pip install codeverify-cli
      codeverify verify --gate {gate_name}
    displayName: 'CodeVerify Verification'
""".format(gate_name=gate.name if gate else "default")
        return PipelineConfig(
            platform=CICDPlatform.AZURE_DEVOPS,
            config_content=config,
            file_name="azure-pipelines.yml",
            instructions="Add this step to your Azure Pipelines YAML.",
        )

    def _bitbucket(self, gate: QualityGate | None) -> PipelineConfig:
        config = """pipelines:
  pull-requests:
    '**':
      - step:
          name: CodeVerify
          image: codeverify/cli:latest
          script:
            - codeverify verify --gate {gate_name}
""".format(gate_name=gate.name if gate else "default")
        return PipelineConfig(
            platform=CICDPlatform.BITBUCKET_PIPELINES,
            config_content=config,
            file_name="bitbucket-pipelines.yml",
            instructions="Add this to your bitbucket-pipelines.yml file.",
        )


class CICDOrchestrator:
    """Orchestrates verification across CI/CD platforms."""

    def __init__(self) -> None:
        self._evaluator = QualityGateEvaluator()
        self._config_gen = PipelineConfigGenerator()
        self._gates: dict[str, QualityGate] = {
            "default": QualityGate(),
            "strict": QualityGate.strict(),
            "lenient": QualityGate.lenient(),
        }
        self._evaluations: list[GateEvaluation] = []

    def register_gate(self, gate: QualityGate) -> None:
        self._gates[gate.name] = gate

    def get_gate(self, name: str) -> QualityGate | None:
        return self._gates.get(name)

    def evaluate(
        self,
        gate_name: str = "default",
        critical: int = 0,
        high: int = 0,
        medium: int = 0,
        low: int = 0,
        trust_score: float = 100.0,
        coverage: float = 1.0,
    ) -> GateEvaluation:
        gate = self._gates.get(gate_name)
        if gate is None:
            return GateEvaluation(
                gate_name=gate_name, result=GateResult.SKIPPED, summary="Gate not found"
            )
        result = self._evaluator.evaluate(gate, critical, high, medium, low, trust_score, coverage)
        self._evaluations.append(result)
        logger.info("gate_evaluated", gate=gate_name, result=result.result.value)
        return result

    def generate_config(self, platform: CICDPlatform, gate_name: str = "default") -> PipelineConfig:
        gate = self._gates.get(gate_name)
        return self._config_gen.generate(platform, gate)

    def generate_status(self, evaluation: GateEvaluation) -> PipelineStatus:
        if evaluation.result == GateResult.PASSED or evaluation.result == GateResult.WARNING:
            state = StatusState.SUCCESS
        elif evaluation.result == GateResult.FAILED:
            state = StatusState.FAILURE
        else:
            state = StatusState.PENDING

        return PipelineStatus(
            state=state,
            description=evaluation.summary,
        )

    @property
    def available_gates(self) -> list[str]:
        return list(self._gates.keys())

    @property
    def supported_platforms(self) -> list[CICDPlatform]:
        return list(CICDPlatform)

    @property
    def evaluation_history(self) -> list[GateEvaluation]:
        return list(self._evaluations)


_orchestrator: CICDOrchestrator | None = None


def get_cicd_orchestrator() -> CICDOrchestrator:
    """Get the singleton CICDOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CICDOrchestrator()
    return _orchestrator


def reset_cicd_orchestrator() -> None:
    """Reset the singleton (useful for testing)."""
    global _orchestrator
    _orchestrator = None
