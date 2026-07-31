"""Tests for model evaluation and version management."""

from codeverify_core.model_evaluation import (
    BenchmarkResult,
    EvalResult,
    EvalSample,
    ModelBenchmark,
    ModelVersion,
    ModelVersionManager,
    export_training_data,
)


def _fake_predict(code: str, _language: str) -> list[dict]:
    """Fake prediction function for testing."""
    findings = []
    if "eval(" in code:
        findings.append({"type": "security", "line": 1, "severity": "critical"})
    if "None" in code and "is None" not in code:
        findings.append({"type": "null_safety", "line": 1, "severity": "high"})
    return findings


class TestModelBenchmark:
    def test_evaluate_perfect_model(self):
        benchmark = ModelBenchmark("test")
        benchmark.add_sample(
            EvalSample(
                code="x = eval(input())",
                language="python",
                expected_findings=[{"type": "security", "line": 1, "severity": "critical"}],
            )
        )

        result = benchmark.evaluate(_fake_predict, model_id="perfect")
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_evaluate_with_false_positives(self):
        benchmark = ModelBenchmark("test")
        benchmark.add_sample(
            EvalSample(
                code="x = eval(None)",  # Predict both security and null_safety
                language="python",
                expected_findings=[{"type": "security", "line": 1, "severity": "critical"}],
            )
        )

        result = benchmark.evaluate(_fake_predict, model_id="noisy")
        assert result.precision < 1.0  # Has false positive
        assert result.recall == 1.0  # Catches the expected

    def test_evaluate_with_false_negatives(self):
        benchmark = ModelBenchmark("test")
        benchmark.add_sample(
            EvalSample(
                code="safe_code()",
                language="python",
                expected_findings=[{"type": "security", "line": 1, "severity": "high"}],
            )
        )

        result = benchmark.evaluate(_fake_predict, model_id="weak")
        assert result.recall == 0.0  # Missed the expected

    def test_benchmark_summary(self):
        benchmark = ModelBenchmark("security")
        benchmark.add_sample(
            EvalSample(
                code="eval(x)",
                language="python",
                expected_findings=[{"type": "security", "line": 1, "severity": "critical"}],
            )
        )
        result = benchmark.evaluate(_fake_predict, model_id="v1")
        summary = result.summary()
        assert summary["model_id"] == "v1"
        assert "f1" in summary
        assert "avg_latency_ms" in summary

    def test_multiple_samples(self):
        benchmark = ModelBenchmark("mixed")
        benchmark.add_samples(
            [
                EvalSample(
                    "eval(x)", "python", [{"type": "security", "line": 1, "severity": "critical"}]
                ),
                EvalSample("safe()", "python", []),
            ]
        )
        result = benchmark.evaluate(_fake_predict, model_id="v1")
        assert len(result.eval_results) == 2


class TestEvalResult:
    def test_precision_recall_f1(self):
        r = EvalResult(
            sample_id=0,
            predicted_findings=[],
            expected_findings=[],
            true_positives=8,
            false_positives=2,
            false_negatives=3,
        )
        assert abs(r.precision - 0.8) < 0.01
        assert abs(r.recall - 0.727) < 0.01
        assert r.f1 > 0

    def test_zero_predictions(self):
        r = EvalResult(
            sample_id=0,
            predicted_findings=[],
            expected_findings=[],
            true_positives=0,
            false_positives=0,
            false_negatives=0,
        )
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0


class TestModelVersionManager:
    def test_register_and_list(self):
        mgr = ModelVersionManager()
        mgr.register("cv-model", "1.0.0", "codellama-7b")
        mgr.register("cv-model", "1.1.0", "codellama-7b")

        versions = mgr.list_versions("cv-model")
        assert len(versions) == 2

    def test_promote_to_production(self):
        mgr = ModelVersionManager(min_f1_for_promotion=0.5)
        mgr.register("cv-model", "1.0.0", "codellama-7b")

        benchmark = BenchmarkResult(model_id="cv-model", benchmark_name="test")
        benchmark.eval_results.append(
            EvalResult(
                sample_id=0,
                predicted_findings=[],
                expected_findings=[],
                true_positives=9,
                false_positives=1,
                false_negatives=1,
            )
        )

        success = mgr.promote("cv-model", "1.0.0", ModelVersion.PRODUCTION, benchmark)
        assert success is True
        prod = mgr.get_production("cv-model")
        assert prod is not None
        assert prod.version == "1.0.0"

    def test_promotion_fails_below_quality(self):
        mgr = ModelVersionManager(min_f1_for_promotion=0.9)
        mgr.register("cv-model", "1.0.0", "codellama-7b")

        benchmark = BenchmarkResult(model_id="cv-model", benchmark_name="test")
        benchmark.eval_results.append(
            EvalResult(
                sample_id=0,
                predicted_findings=[],
                expected_findings=[],
                true_positives=5,
                false_positives=5,
                false_negatives=5,
            )
        )

        success = mgr.promote("cv-model", "1.0.0", ModelVersion.PRODUCTION, benchmark)
        assert success is False

    def test_rollback(self):
        mgr = ModelVersionManager(min_f1_for_promotion=0.5)
        mgr.register("cv-model", "1.0.0", "codellama-7b")
        mgr.register("cv-model", "2.0.0", "codellama-7b")

        good_bench = BenchmarkResult(model_id="cv-model", benchmark_name="test")
        good_bench.eval_results.append(
            EvalResult(
                0,
                [],
                [],
                true_positives=9,
                false_positives=1,
                false_negatives=1,
            )
        )

        mgr.promote("cv-model", "1.0.0", ModelVersion.PRODUCTION, good_bench)
        mgr.promote("cv-model", "2.0.0", ModelVersion.PRODUCTION, good_bench)

        # v2 is now production, v1 is deprecated
        rolled = mgr.rollback("cv-model")
        assert rolled is not None
        assert rolled.version == "1.0.0"
        assert rolled.status == ModelVersion.PRODUCTION

    def test_compare_versions(self):
        mgr = ModelVersionManager(min_f1_for_promotion=0.5)
        mgr.register("cv-model", "1.0.0", "codellama-7b")
        mgr.register("cv-model", "2.0.0", "codellama-7b")

        b1 = BenchmarkResult(model_id="cv-model", benchmark_name="test")
        b1.eval_results.append(EvalResult(0, [], [], 7, 3, 2))

        b2 = BenchmarkResult(model_id="cv-model", benchmark_name="test")
        b2.eval_results.append(EvalResult(0, [], [], 9, 1, 1))

        mgr.promote("cv-model", "1.0.0", ModelVersion.CANDIDATE, b1)
        mgr.promote("cv-model", "2.0.0", ModelVersion.CANDIDATE, b2)

        comparison = mgr.compare_versions("cv-model", "1.0.0", "2.0.0")
        assert "f1_diff" in comparison
        assert comparison["improved"] is True


class TestTrainingDataExport:
    def test_jsonl_export(self):
        examples = [
            {"input": "eval(x)", "output": "security issue"},
            {"input": "safe(x)", "output": "no issues"},
        ]
        output = export_training_data(examples, "jsonl")
        lines = output.strip().split("\n")
        assert len(lines) == 2

    def test_alpaca_export(self):
        examples = [
            {"system": "Analyze code", "input": "eval(x)", "output": "issue found"},
        ]
        output = export_training_data(examples, "alpaca")
        import json

        data = json.loads(output.strip())
        assert data["instruction"] == "Analyze code"
