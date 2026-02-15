"""Fine-Tuning Evaluation and Model Versioning.

Provides:
- Benchmark evaluation framework for fine-tuned models
- Model version management with rollback support
- Quality metrics comparison against baseline
- Training data export for reproducibility
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# =============================================================================
# Evaluation Framework
# =============================================================================


@dataclass
class EvalSample:
    """A single evaluation sample for benchmarking."""

    code: str
    language: str
    expected_findings: list[dict[str, Any]]
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class EvalResult:
    """Result of evaluating a model on a single sample."""

    sample_id: int
    predicted_findings: list[dict[str, Any]]
    expected_findings: list[dict[str, Any]]
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    latency_ms: float = 0.0

    @property
    def precision(self) -> float:
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0.0

    @property
    def recall(self) -> float:
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class BenchmarkResult:
    """Aggregated results from a full benchmark run."""

    model_id: str
    benchmark_name: str
    eval_results: list[EvalResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def precision(self) -> float:
        tp = sum(r.true_positives for r in self.eval_results)
        fp = sum(r.false_positives for r in self.eval_results)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = sum(r.true_positives for r in self.eval_results)
        fn = sum(r.false_negatives for r in self.eval_results)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        if not self.eval_results:
            return 0.0
        return sum(r.latency_ms for r in self.eval_results) / len(self.eval_results)

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "benchmark": self.benchmark_name,
            "samples": len(self.eval_results),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_time_ms": round(self.total_time_ms, 1),
            "timestamp": self.timestamp,
        }


class ModelBenchmark:
    """Benchmark framework for comparing model quality.

    Usage:
        benchmark = ModelBenchmark("security-checks")
        benchmark.add_sample(EvalSample(code, "python", expected))
        result = benchmark.evaluate(model_predict_fn)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._samples: list[EvalSample] = []

    def add_sample(self, sample: EvalSample) -> None:
        self._samples.append(sample)

    def add_samples(self, samples: list[EvalSample]) -> None:
        self._samples.extend(samples)

    def evaluate(
        self,
        predict_fn: Any,
        model_id: str = "unknown",
    ) -> BenchmarkResult:
        """Evaluate a prediction function against all samples.

        Args:
            predict_fn: Callable(code, language) -> list[dict] of findings
            model_id: Identifier for the model being evaluated
        """
        result = BenchmarkResult(model_id=model_id, benchmark_name=self._name)
        start = time.time()

        for i, sample in enumerate(self._samples):
            sample_start = time.time()

            try:
                predicted = predict_fn(sample.code, sample.language)
            except Exception:
                predicted = []

            latency = (time.time() - sample_start) * 1000

            eval_result = self._score_predictions(i, predicted, sample.expected_findings, latency)
            result.eval_results.append(eval_result)

        result.total_time_ms = (time.time() - start) * 1000
        return result

    def _score_predictions(
        self,
        sample_id: int,
        predicted: list[dict[str, Any]],
        expected: list[dict[str, Any]],
        latency_ms: float,
    ) -> EvalResult:
        """Score predictions against expected findings."""
        pred_set = {self._finding_key(f) for f in predicted}
        exp_set = {self._finding_key(f) for f in expected}

        tp = len(pred_set & exp_set)
        fp = len(pred_set - exp_set)
        fn = len(exp_set - pred_set)

        return EvalResult(
            sample_id=sample_id,
            predicted_findings=predicted,
            expected_findings=expected,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _finding_key(finding: dict[str, Any]) -> str:
        """Create a comparable key for a finding."""
        return f"{finding.get('type', '')}:{finding.get('line', 0)}:{finding.get('severity', '')}"


# =============================================================================
# Model Version Management
# =============================================================================


class ModelVersion(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


@dataclass
class VersionedModel:
    """A versioned fine-tuned model with lifecycle management."""

    model_id: str
    version: str
    base_model: str
    status: ModelVersion = ModelVersion.DRAFT
    benchmark_result: BenchmarkResult | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    promoted_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_production(self) -> bool:
        return self.status == ModelVersion.PRODUCTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "base_model": self.base_model,
            "status": self.status.value,
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "benchmark": self.benchmark_result.summary() if self.benchmark_result else None,
        }


class ModelVersionManager:
    """Manages model versions with promotion and rollback.

    Usage:
        mgr = ModelVersionManager()
        mgr.register("codeverify-v1", "1.0.0", "codellama-7b")
        mgr.promote("codeverify-v1", "1.0.0", ModelVersion.CANDIDATE, benchmark)
        mgr.promote("codeverify-v1", "1.0.0", ModelVersion.PRODUCTION)
        mgr.rollback("codeverify-v1")
    """

    def __init__(self, min_f1_for_promotion: float = 0.7) -> None:
        self._versions: dict[str, list[VersionedModel]] = {}
        self._min_f1 = min_f1_for_promotion

    def register(
        self,
        model_id: str,
        version: str,
        base_model: str,
        metadata: dict[str, Any] | None = None,
    ) -> VersionedModel:
        """Register a new model version."""
        vm = VersionedModel(
            model_id=model_id,
            version=version,
            base_model=base_model,
            metadata=metadata or {},
        )
        self._versions.setdefault(model_id, []).append(vm)
        return vm

    def promote(
        self,
        model_id: str,
        version: str,
        target_status: ModelVersion,
        benchmark: BenchmarkResult | None = None,
    ) -> bool:
        """Promote a model version to a higher status."""
        vm = self._find_version(model_id, version)
        if not vm:
            return False

        # Quality gate: require benchmark for production
        if target_status == ModelVersion.PRODUCTION:
            br = benchmark or vm.benchmark_result
            if not br or br.f1 < self._min_f1:
                return False
            vm.benchmark_result = br

            # Demote current production version
            for v in self._versions.get(model_id, []):
                if v.status == ModelVersion.PRODUCTION and v.version != version:
                    v.status = ModelVersion.DEPRECATED

        if target_status == ModelVersion.CANDIDATE and benchmark:
            vm.benchmark_result = benchmark

        vm.status = target_status
        vm.promoted_at = datetime.now(UTC).isoformat()
        return True

    def rollback(self, model_id: str) -> VersionedModel | None:
        """Roll back to the previous production version."""
        versions = self._versions.get(model_id, [])

        # Find current production and previous
        current_prod = None
        previous = None
        for v in reversed(versions):
            if v.status == ModelVersion.PRODUCTION:
                current_prod = v
            elif v.status in (ModelVersion.DEPRECATED, ModelVersion.CANDIDATE):
                if current_prod and not previous:
                    previous = v

        if current_prod and previous:
            current_prod.status = ModelVersion.ROLLED_BACK
            previous.status = ModelVersion.PRODUCTION
            previous.promoted_at = datetime.now(UTC).isoformat()
            return previous

        return None

    def get_production(self, model_id: str) -> VersionedModel | None:
        """Get the current production version."""
        for v in reversed(self._versions.get(model_id, [])):
            if v.status == ModelVersion.PRODUCTION:
                return v
        return None

    def list_versions(self, model_id: str) -> list[dict[str, Any]]:
        """List all versions of a model."""
        return [v.to_dict() for v in self._versions.get(model_id, [])]

    def _find_version(self, model_id: str, version: str) -> VersionedModel | None:
        for v in self._versions.get(model_id, []):
            if v.version == version:
                return v
        return None

    def compare_versions(
        self,
        model_id: str,
        version_a: str,
        version_b: str,
    ) -> dict[str, Any]:
        """Compare benchmark results between two versions."""
        va = self._find_version(model_id, version_a)
        vb = self._find_version(model_id, version_b)

        if not va or not vb:
            return {"error": "Version not found"}

        result = {
            "version_a": version_a,
            "version_b": version_b,
        }

        if va.benchmark_result and vb.benchmark_result:
            ba = va.benchmark_result
            bb = vb.benchmark_result
            result["f1_diff"] = round(bb.f1 - ba.f1, 4)
            result["precision_diff"] = round(bb.precision - ba.precision, 4)
            result["recall_diff"] = round(bb.recall - ba.recall, 4)
            result["latency_diff_ms"] = round(bb.avg_latency_ms - ba.avg_latency_ms, 1)
            result["improved"] = bb.f1 > ba.f1

        return result


# =============================================================================
# Training Data Export
# =============================================================================


def export_training_data(
    examples: list[dict[str, Any]],
    format_type: str = "jsonl",
) -> str:
    """Export training examples to a standard format.

    Args:
        examples: List of training example dicts with 'input' and 'output' keys
        format_type: 'jsonl' for JSON Lines, 'alpaca' for Alpaca format

    Returns:
        Formatted string ready for file export
    """
    lines: list[str] = []

    for ex in examples:
        if format_type == "jsonl":
            lines.append(json.dumps(ex, ensure_ascii=False))
        elif format_type == "alpaca":
            lines.append(
                json.dumps(
                    {
                        "instruction": ex.get("system", "Analyze this code."),
                        "input": ex.get("input", ""),
                        "output": ex.get("output", ""),
                    },
                    ensure_ascii=False,
                )
            )

    return "\n".join(lines) + "\n" if lines else ""
