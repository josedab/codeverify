"""CI/CD Native Actions & Plugins.

Provides first-class GitHub Action, GitLab CI component, and Jenkins pipeline
configuration generators with SARIF output, quality gates, and one-line YAML setup.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CIPlatform(str, Enum):
    """Supported CI/CD platforms."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    CIRCLECI = "circleci"


class GateResult(str, Enum):
    """Quality gate result."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""

    max_critical: int = 0
    max_high: int = 0
    max_medium: int = 5
    max_low: int = 10
    min_trust_score: float = 0.0
    min_verification_coverage: float = 0.0
    fail_on_new_findings: bool = False


@dataclass
class SARIFResult:
    """A single SARIF result (finding)."""

    rule_id: str = ""
    message: str = ""
    level: str = "warning"
    file_path: str = ""
    start_line: int = 1
    end_line: int = 1
    start_column: int = 1
    end_column: int = 1


@dataclass
class SARIFReport:
    """SARIF 2.1.0 report."""

    tool_name: str = "codeverify"
    tool_version: str = "0.8.0"
    results: list[SARIFResult] = field(default_factory=list)
    invocation_success: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Export as SARIF 2.1.0 JSON-compatible dict."""
        rules = {}
        for r in self.results:
            if r.rule_id not in rules:
                rules[r.rule_id] = {
                    "id": r.rule_id,
                    "shortDescription": {"text": r.rule_id},
                }

        return {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": self.tool_name,
                            "version": self.tool_version,
                            "rules": list(rules.values()),
                        }
                    },
                    "results": [
                        {
                            "ruleId": r.rule_id,
                            "level": r.level,
                            "message": {"text": r.message},
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {"uri": r.file_path},
                                        "region": {
                                            "startLine": r.start_line,
                                            "endLine": r.end_line,
                                            "startColumn": r.start_column,
                                            "endColumn": r.end_column,
                                        },
                                    }
                                }
                            ],
                        }
                        for r in self.results
                    ],
                    "invocations": [{"executionSuccessful": self.invocation_success}],
                }
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class QualityGateEvaluator:
    """Evaluates findings against quality gate thresholds."""

    def evaluate(
        self,
        findings: list[dict[str, Any]],
        config: QualityGateConfig,
    ) -> dict[str, Any]:
        """Evaluate findings against quality gate configuration."""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in findings:
            sev = f.get("severity", "low").lower()
            if sev in counts:
                counts[sev] += 1

        violations: list[str] = []
        if counts["critical"] > config.max_critical:
            violations.append(f"Critical findings: {counts['critical']} > {config.max_critical}")
        if counts["high"] > config.max_high:
            violations.append(f"High findings: {counts['high']} > {config.max_high}")
        if counts["medium"] > config.max_medium:
            violations.append(f"Medium findings: {counts['medium']} > {config.max_medium}")
        if counts["low"] > config.max_low:
            violations.append(f"Low findings: {counts['low']} > {config.max_low}")

        if violations:
            result = GateResult.FAIL
        elif counts["medium"] > 0 or counts["low"] > 0:
            result = GateResult.WARN
        else:
            result = GateResult.PASS

        return {
            "result": result.value,
            "counts": counts,
            "violations": violations,
            "total_findings": sum(counts.values()),
        }


class CIConfigGenerator:
    """Generates CI/CD configuration for various platforms."""

    def generate(
        self,
        platform: CIPlatform,
        languages: list[str] | None = None,
        checks: list[str] | None = None,
        gate_config: QualityGateConfig | None = None,
    ) -> str:
        """Generate CI configuration YAML for the given platform."""
        languages = languages or ["python", "typescript"]
        checks = checks or ["null_safety", "array_bounds", "integer_overflow"]
        gate = gate_config or QualityGateConfig()

        if platform == CIPlatform.GITHUB_ACTIONS:
            return self._github_actions(languages, checks, gate)
        elif platform == CIPlatform.GITLAB_CI:
            return self._gitlab_ci(languages, checks, gate)
        elif platform == CIPlatform.JENKINS:
            return self._jenkinsfile(languages, checks, gate)
        elif platform == CIPlatform.CIRCLECI:
            return self._circleci(languages, checks, gate)
        raise ValueError(f"Unsupported platform: {platform}")

    def _github_actions(
        self, languages: list[str], checks: list[str], gate: QualityGateConfig
    ) -> str:
        langs = ", ".join(languages)
        chks = ", ".join(checks)
        return f"""name: CodeVerify Analysis
on:
  pull_request:
    types: [opened, synchronize]
  push:
    branches: [main]

permissions:
  contents: read
  security-events: write
  pull-requests: write

jobs:
  codeverify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: codeverify/action@v1
        with:
          languages: "{langs}"
          checks: "{chks}"
          max-critical: {gate.max_critical}
          max-high: {gate.max_high}
          sarif-upload: true
"""

    def _gitlab_ci(self, languages: list[str], checks: list[str], gate: QualityGateConfig) -> str:
        langs = ",".join(languages)
        chks = ",".join(checks)
        return f"""codeverify:
  stage: test
  image: codeverify/scanner:latest
  script:
    - codeverify scan --languages {langs} --checks {chks} --sarif codeverify.sarif
    - codeverify gate --max-critical {gate.max_critical} --max-high {gate.max_high}
  artifacts:
    reports:
      sast: codeverify.sarif
  rules:
    - if: $CI_MERGE_REQUEST_IID
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
"""

    def _jenkinsfile(self, languages: list[str], checks: list[str], gate: QualityGateConfig) -> str:
        langs = ",".join(languages)
        chks = ",".join(checks)
        return f"""pipeline {{
    agent any
    stages {{
        stage('CodeVerify') {{
            steps {{
                sh 'pip install codeverify-cli'
                sh 'codeverify scan --languages {langs} --checks {chks} --sarif codeverify.sarif'
                sh 'codeverify gate --max-critical {gate.max_critical} --max-high {gate.max_high}'
            }}
            post {{
                always {{
                    archiveArtifacts artifacts: 'codeverify.sarif'
                }}
            }}
        }}
    }}
}}
"""

    def _circleci(self, languages: list[str], checks: list[str], gate: QualityGateConfig) -> str:
        langs = ",".join(languages)
        chks = ",".join(checks)
        return f"""version: 2.1
jobs:
  codeverify:
    docker:
      - image: codeverify/scanner:latest
    steps:
      - checkout
      - run:
          name: CodeVerify Analysis
          command: codeverify scan --languages {langs} --checks {chks} --sarif codeverify.sarif
      - run:
          name: Quality Gate
          command: codeverify gate --max-critical {gate.max_critical} --max-high {gate.max_high}
      - store_artifacts:
          path: codeverify.sarif
workflows:
  verify:
    jobs:
      - codeverify
"""


class CICDActionRunner:
    """Orchestrates a CI/CD verification run."""

    def __init__(self) -> None:
        self._gate_evaluator = QualityGateEvaluator()
        self._config_generator = CIConfigGenerator()

    def run_analysis(
        self,
        findings: list[dict[str, Any]],
        gate_config: QualityGateConfig | None = None,
    ) -> dict[str, Any]:
        """Run the analysis pipeline and return results with SARIF."""
        gate_config = gate_config or QualityGateConfig()
        start = time.time()

        sarif = SARIFReport()
        for f in findings:
            level_map = {"critical": "error", "high": "error", "medium": "warning", "low": "note"}
            sarif.results.append(
                SARIFResult(
                    rule_id=f.get("rule_id", "unknown"),
                    message=f.get("message", ""),
                    level=level_map.get(f.get("severity", "low"), "note"),
                    file_path=f.get("file_path", ""),
                    start_line=f.get("line", 1),
                )
            )

        gate_result = self._gate_evaluator.evaluate(findings, gate_config)
        elapsed = (time.time() - start) * 1000

        return {
            "sarif": sarif.to_dict(),
            "gate": gate_result,
            "findings_count": len(findings),
            "elapsed_ms": elapsed,
        }

    def generate_config(
        self,
        platform: CIPlatform,
        languages: list[str] | None = None,
        gate_config: QualityGateConfig | None = None,
    ) -> str:
        return self._config_generator.generate(platform, languages, gate_config=gate_config)
