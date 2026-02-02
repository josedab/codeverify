"""Tests for LLM Fine-Tuning Pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from codeverify_agents.fine_tuning import (
    DataCollector,
    DataSourceType,
    FineTunedModel,
    FineTuningManager,
    ModelServer,
    ModelType,
    TrainingConfig,
    TrainingDataset,
    TrainingExample,
    TrainingJob,
    TrainingPipeline,
    TrainingStatus,
    get_fine_tuning_manager,
    reset_fine_tuning_manager,
)


class TestTrainingExample:
    """Tests for TrainingExample."""

    def test_default_creation(self) -> None:
        """Test default example creation."""
        example = TrainingExample(
            input_code="def foo(): pass",
            input_context={"language": "python"},
            expected_output={"findings": []},
        )
        assert example.input_code == "def foo(): pass"
        assert example.quality_score == 1.0
        assert example.source_type == DataSourceType.VERIFICATION_RESULTS

    def test_to_openai_format(self) -> None:
        """Test conversion to OpenAI training format."""
        example = TrainingExample(
            input_code="def foo(): pass",
            input_context={"language": "python", "file_path": "test.py"},
            expected_output={"findings": [{"title": "Bug"}]},
        )
        
        formatted = example.to_training_format("openai")
        
        assert "messages" in formatted
        assert len(formatted["messages"]) == 3
        assert formatted["messages"][0]["role"] == "system"
        assert formatted["messages"][1]["role"] == "user"
        assert formatted["messages"][2]["role"] == "assistant"
        assert "python" in formatted["messages"][1]["content"]

    def test_to_alpaca_format(self) -> None:
        """Test conversion to Alpaca training format."""
        example = TrainingExample(
            input_code="def foo(): pass",
            input_context={"language": "python"},
            expected_output={"findings": []},
        )
        
        formatted = example.to_training_format("alpaca")
        
        assert "instruction" in formatted
        assert "input" in formatted
        assert "output" in formatted


class TestTrainingDataset:
    """Tests for TrainingDataset."""

    def test_add_example(self) -> None:
        """Test adding examples to dataset."""
        dataset = TrainingDataset(name="test")
        
        example = TrainingExample(
            input_code="test code",
            expected_output={"findings": []},
        )
        dataset.add_example(example)
        
        assert len(dataset.examples) == 1
        assert dataset.stats["total_examples"] == 1

    def test_stats_calculation(self) -> None:
        """Test dataset statistics calculation."""
        dataset = TrainingDataset(name="test")
        
        dataset.add_example(TrainingExample(
            input_code="code1",
            source_type=DataSourceType.VERIFICATION_RESULTS,
            quality_score=0.8,
        ))
        dataset.add_example(TrainingExample(
            input_code="code2",
            source_type=DataSourceType.USER_CORRECTIONS,
            quality_score=1.0,
        ))
        
        assert dataset.stats["total_examples"] == 2
        assert dataset.stats["avg_quality_score"] == 0.9
        assert DataSourceType.VERIFICATION_RESULTS.value in dataset.stats["by_source"]

    def test_export(self) -> None:
        """Test dataset export."""
        dataset = TrainingDataset(name="test")
        dataset.add_example(TrainingExample(
            input_code="code",
            input_context={"language": "python"},
            expected_output={"findings": []},
            quality_score=0.8,
        ))
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            count = dataset.export(output_path, "openai")
            assert count == 1
            
            with open(output_path) as f:
                data = json.loads(f.readline())
            assert "messages" in data
        finally:
            output_path.unlink()


class TestDataCollector:
    """Tests for DataCollector."""

    def test_collect_from_verification(self) -> None:
        """Test collecting from verification results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(Path(tmpdir))
            
            example = collector.collect_from_verification(
                code="def foo(): pass",
                context={"language": "python"},
                findings=[{"title": "Bug", "confidence": 0.9}],
                org_id="test-org",
            )
            
            assert example.source_type == DataSourceType.VERIFICATION_RESULTS
            assert example.org_id == "test-org"
            assert example.quality_score > 0.5

    def test_collect_from_user_correction(self) -> None:
        """Test collecting from user corrections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(Path(tmpdir))
            
            example = collector.collect_from_user_correction(
                code="def foo(): pass",
                context={"language": "python"},
                original_findings=[{"title": "False positive"}],
                corrected_findings=[],
                org_id="test-org",
            )
            
            assert example.source_type == DataSourceType.USER_CORRECTIONS
            assert example.quality_score == 1.0
            assert example.metadata["correction_type"] == "removed_false_positives"

    def test_collect_from_dismissed_finding(self) -> None:
        """Test collecting from dismissed findings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(Path(tmpdir))
            
            example = collector.collect_from_dismissed_finding(
                code="def foo(): pass",
                context={"language": "python"},
                dismissed_finding={"title": "Not a bug"},
                dismiss_reason="Working as intended",
            )
            
            assert example.source_type == DataSourceType.DISMISSED_FINDINGS
            assert "dismissed_finding" in example.metadata

    def test_get_dataset(self) -> None:
        """Test getting datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(Path(tmpdir))
            
            collector.collect_from_verification(
                code="test",
                context={},
                findings=[],
                dataset_name="my_dataset",
            )
            
            dataset = collector.get_dataset("my_dataset")
            assert dataset is not None
            assert len(dataset.examples) == 1


class TestTrainingPipeline:
    """Tests for TrainingPipeline."""

    def test_create_job(self) -> None:
        """Test creating a training job."""
        pipeline = TrainingPipeline()
        dataset = TrainingDataset(name="test")
        dataset.add_example(TrainingExample(input_code="test"))
        
        job = pipeline.create_job(dataset)
        
        assert job.status == TrainingStatus.PENDING
        assert job.dataset_id == dataset.id

    @pytest.mark.asyncio
    async def test_run_training(self) -> None:
        """Test running training pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(output_dir=tmpdir)
            pipeline = TrainingPipeline(config)
            
            dataset = TrainingDataset(name="test")
            for i in range(20):
                dataset.add_example(TrainingExample(
                    input_code=f"def func_{i}(): pass",
                    input_context={"language": "python"},
                    expected_output={"findings": []},
                    quality_score=0.8,
                ))
            
            job = pipeline.create_job(dataset, config)
            job = await pipeline.run_training(job, dataset)
            
            assert job.status == TrainingStatus.COMPLETED
            assert job.output_model_path is not None


class TestModelServer:
    """Tests for ModelServer."""

    def test_register_model(self) -> None:
        """Test registering a model."""
        server = ModelServer()
        model = FineTunedModel(
            name="test-model",
            org_id="test-org",
            model_path="/path/to/model",
        )
        
        server.register_model(model)
        
        retrieved = server.get_active_model("test-org")
        assert retrieved is not None
        assert retrieved.id == model.id

    def test_fallback_to_global(self) -> None:
        """Test fallback to global model."""
        server = ModelServer()
        global_model = FineTunedModel(
            name="global-model",
            org_id=None,  # Global
            model_path="/path/to/model",
        )
        
        server.register_model(global_model)
        
        # Should get global model when no org-specific model
        retrieved = server.get_active_model("unknown-org")
        assert retrieved is not None
        assert retrieved.org_id is None


class TestFineTuningManager:
    """Tests for FineTuningManager."""

    def test_get_training_summary(self) -> None:
        """Test getting training summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FineTuningManager(
                storage_path=Path(tmpdir),
                models_path=Path(tmpdir) / "models",
            )
            
            # Collect some data
            manager.collector.collect_from_verification(
                code="test code",
                context={"language": "python"},
                findings=[],
                dataset_name="test_dataset",
            )
            
            summary = manager.get_training_summary()
            
            assert "datasets" in summary
            assert "models" in summary
            assert len(summary["datasets"]) == 1


class TestGlobalManager:
    """Tests for global manager functions."""

    def test_get_manager(self) -> None:
        """Test getting global manager."""
        reset_fine_tuning_manager()
        
        manager1 = get_fine_tuning_manager()
        manager2 = get_fine_tuning_manager()
        
        assert manager1 is manager2

    def test_reset_manager(self) -> None:
        """Test resetting global manager."""
        manager1 = get_fine_tuning_manager()
        reset_fine_tuning_manager()
        manager2 = get_fine_tuning_manager()
        
        assert manager1 is not manager2
