"""LLM Fine-Tuning Pipeline for organization-specific model training.

This module provides:
1. Data collection from verified code patterns
2. Training pipeline for LoRA/QLoRA fine-tuning
3. Model serving with automatic fallback to base models
4. Continuous feedback loop for improvement
"""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger()


class TrainingStatus(str, Enum):
    """Status of a fine-tuning job."""
    PENDING = "pending"
    COLLECTING = "collecting"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataSourceType(str, Enum):
    """Type of data source for training."""
    VERIFICATION_RESULTS = "verification_results"
    USER_CORRECTIONS = "user_corrections"
    APPROVED_FINDINGS = "approved_findings"
    DISMISSED_FINDINGS = "dismissed_findings"
    CODE_PATTERNS = "code_patterns"


class ModelType(str, Enum):
    """Type of model for fine-tuning."""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    LLAMA_7B = "llama_7b"
    LLAMA_13B = "llama_13b"
    MISTRAL_7B = "mistral_7b"
    CODELLAMA_7B = "codellama_7b"


@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""
    id: str = field(default_factory=lambda: str(uuid4()))
    input_code: str = ""
    input_context: dict[str, Any] = field(default_factory=dict)
    expected_output: dict[str, Any] = field(default_factory=dict)
    source_type: DataSourceType = DataSourceType.VERIFICATION_RESULTS
    quality_score: float = 1.0
    org_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_training_format(self, format_type: str = "openai") -> dict[str, Any]:
        """Convert to training format for specific provider."""
        if format_type == "openai":
            return {
                "messages": [
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": self._build_user_prompt()},
                    {"role": "assistant", "content": json.dumps(self.expected_output)},
                ]
            }
        elif format_type == "alpaca":
            return {
                "instruction": self._build_system_prompt(),
                "input": self._build_user_prompt(),
                "output": json.dumps(self.expected_output),
            }
        else:
            raise ValueError(f"Unknown format type: {format_type}")

    def _build_system_prompt(self) -> str:
        """Build system prompt from context."""
        return self.input_context.get(
            "system_prompt",
            "You are an expert code reviewer. Analyze code for bugs, security issues, and quality problems."
        )

    def _build_user_prompt(self) -> str:
        """Build user prompt from code and context."""
        language = self.input_context.get("language", "unknown")
        file_path = self.input_context.get("file_path", "unknown")
        return f"Analyze this {language} code from {file_path}:\n\n```{language}\n{self.input_code}\n```"


@dataclass
class TrainingDataset:
    """A dataset for fine-tuning."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    org_id: str | None = None
    examples: list[TrainingExample] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_example(self, example: TrainingExample) -> None:
        """Add an example to the dataset."""
        self.examples.append(example)
        self.updated_at = datetime.utcnow()
        self._update_stats()

    def _update_stats(self) -> None:
        """Update dataset statistics."""
        self.stats = {
            "total_examples": len(self.examples),
            "by_source": {},
            "avg_quality_score": 0.0,
        }
        if self.examples:
            source_counts: dict[str, int] = {}
            total_quality = 0.0
            for ex in self.examples:
                source_counts[ex.source_type.value] = source_counts.get(ex.source_type.value, 0) + 1
                total_quality += ex.quality_score
            self.stats["by_source"] = source_counts
            self.stats["avg_quality_score"] = total_quality / len(self.examples)

    def export(self, path: Path, format_type: str = "openai") -> int:
        """Export dataset to file in specified format."""
        examples_exported = 0
        with open(path, "w") as f:
            for example in self.examples:
                if example.quality_score >= 0.5:  # Quality threshold
                    f.write(json.dumps(example.to_training_format(format_type)) + "\n")
                    examples_exported += 1
        return examples_exported


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning job."""
    model_type: ModelType = ModelType.LLAMA_7B
    base_model_path: str | None = None
    output_dir: str = "./fine_tuned_models"
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4
    
    # Validation
    eval_steps: int = 100
    save_steps: int = 500
    validation_split: float = 0.1
    
    # Resource limits
    max_gpu_memory_gb: float | None = None
    use_8bit: bool = True
    use_4bit: bool = False


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    train_loss: float = 0.0
    eval_loss: float = 0.0
    train_accuracy: float = 0.0
    eval_accuracy: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    samples_per_second: float = 0.0
    gpu_memory_used_gb: float = 0.0


@dataclass
class TrainingJob:
    """A fine-tuning training job."""
    id: str = field(default_factory=lambda: str(uuid4()))
    org_id: str | None = None
    dataset_id: str = ""
    config: TrainingConfig = field(default_factory=TrainingConfig)
    status: TrainingStatus = TrainingStatus.PENDING
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    output_model_path: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate job duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class FineTunedModel:
    """A fine-tuned model artifact."""
    id: str = field(default_factory=lambda: str(uuid4()))
    org_id: str | None = None
    name: str = ""
    base_model: ModelType = ModelType.LLAMA_7B
    model_path: str = ""
    training_job_id: str = ""
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    is_active: bool = False
    performance_stats: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def compute_hash(self) -> str:
        """Compute hash of model for integrity verification."""
        if not self.model_path or not Path(self.model_path).exists():
            return ""
        hasher = hashlib.sha256()
        model_dir = Path(self.model_path)
        for file_path in sorted(model_dir.glob("**/*")):
            if file_path.is_file():
                hasher.update(file_path.read_bytes())
        return hasher.hexdigest()[:16]


class DataCollector:
    """Collects training data from verification outcomes."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or Path("./training_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._datasets: dict[str, TrainingDataset] = {}

    def collect_from_verification(
        self,
        code: str,
        context: dict[str, Any],
        findings: list[dict[str, Any]],
        org_id: str | None = None,
        dataset_name: str = "default",
    ) -> TrainingExample:
        """Collect training example from verification result."""
        example = TrainingExample(
            input_code=code,
            input_context=context,
            expected_output={"findings": findings},
            source_type=DataSourceType.VERIFICATION_RESULTS,
            quality_score=self._calculate_quality_score(findings),
            org_id=org_id,
        )
        
        self._add_to_dataset(example, dataset_name, org_id)
        logger.info(
            "Collected training example from verification",
            example_id=example.id,
            findings_count=len(findings),
        )
        return example

    def collect_from_user_correction(
        self,
        code: str,
        context: dict[str, Any],
        original_findings: list[dict[str, Any]],
        corrected_findings: list[dict[str, Any]],
        org_id: str | None = None,
        dataset_name: str = "corrections",
    ) -> TrainingExample:
        """Collect training example from user correction (high-quality signal)."""
        example = TrainingExample(
            input_code=code,
            input_context=context,
            expected_output={"findings": corrected_findings},
            source_type=DataSourceType.USER_CORRECTIONS,
            quality_score=1.0,  # User corrections are highest quality
            org_id=org_id,
            metadata={
                "original_findings": original_findings,
                "correction_type": self._classify_correction(original_findings, corrected_findings),
            },
        )
        
        self._add_to_dataset(example, dataset_name, org_id)
        logger.info(
            "Collected training example from user correction",
            example_id=example.id,
            original_count=len(original_findings),
            corrected_count=len(corrected_findings),
        )
        return example

    def collect_from_dismissed_finding(
        self,
        code: str,
        context: dict[str, Any],
        dismissed_finding: dict[str, Any],
        dismiss_reason: str,
        org_id: str | None = None,
        dataset_name: str = "dismissed",
    ) -> TrainingExample:
        """Collect example from dismissed finding (false positive signal)."""
        # Create example showing this finding should NOT be reported
        remaining_findings = context.get("other_findings", [])
        example = TrainingExample(
            input_code=code,
            input_context=context,
            expected_output={"findings": remaining_findings},
            source_type=DataSourceType.DISMISSED_FINDINGS,
            quality_score=0.8,  # Slightly lower since dismissals may be wrong
            org_id=org_id,
            metadata={
                "dismissed_finding": dismissed_finding,
                "dismiss_reason": dismiss_reason,
            },
        )
        
        self._add_to_dataset(example, dataset_name, org_id)
        logger.info(
            "Collected training example from dismissed finding",
            example_id=example.id,
            finding_title=dismissed_finding.get("title", "unknown"),
        )
        return example

    def get_dataset(self, name: str, org_id: str | None = None) -> TrainingDataset | None:
        """Get dataset by name."""
        key = f"{org_id or 'global'}:{name}"
        return self._datasets.get(key)

    def list_datasets(self, org_id: str | None = None) -> list[TrainingDataset]:
        """List all datasets for an org."""
        prefix = f"{org_id or 'global'}:"
        return [ds for key, ds in self._datasets.items() if key.startswith(prefix)]

    def export_dataset(
        self,
        name: str,
        output_path: Path,
        org_id: str | None = None,
        format_type: str = "openai",
    ) -> int:
        """Export dataset to file."""
        dataset = self.get_dataset(name, org_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {name}")
        return dataset.export(output_path, format_type)

    def _add_to_dataset(
        self,
        example: TrainingExample,
        dataset_name: str,
        org_id: str | None,
    ) -> None:
        """Add example to appropriate dataset."""
        key = f"{org_id or 'global'}:{dataset_name}"
        if key not in self._datasets:
            self._datasets[key] = TrainingDataset(name=dataset_name, org_id=org_id)
        self._datasets[key].add_example(example)

    def _calculate_quality_score(self, findings: list[dict[str, Any]]) -> float:
        """Calculate quality score for training example."""
        if not findings:
            return 0.7  # No findings is valid but less informative
        
        # Higher score for findings with high confidence
        avg_confidence = sum(f.get("confidence", 0.5) for f in findings) / len(findings)
        
        # Higher score for verified findings (formal verification)
        verified_count = sum(1 for f in findings if f.get("verification_type") == "formal")
        verified_ratio = verified_count / len(findings) if findings else 0
        
        return min(1.0, 0.5 + (avg_confidence * 0.25) + (verified_ratio * 0.25))

    def _classify_correction(
        self,
        original: list[dict[str, Any]],
        corrected: list[dict[str, Any]],
    ) -> str:
        """Classify the type of user correction."""
        orig_count = len(original)
        corr_count = len(corrected)
        
        if corr_count < orig_count:
            return "removed_false_positives"
        elif corr_count > orig_count:
            return "added_missed_findings"
        else:
            return "modified_findings"


class TrainingPipeline:
    """Pipeline for fine-tuning models on collected data."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self._jobs: dict[str, TrainingJob] = {}

    def create_job(
        self,
        dataset: TrainingDataset,
        config: TrainingConfig | None = None,
    ) -> TrainingJob:
        """Create a new training job."""
        job_config = config or self.config
        job = TrainingJob(
            org_id=dataset.org_id,
            dataset_id=dataset.id,
            config=job_config,
        )
        self._jobs[job.id] = job
        logger.info(
            "Created training job",
            job_id=job.id,
            dataset_size=len(dataset.examples),
            model_type=job_config.model_type.value,
        )
        return job

    async def run_training(self, job: TrainingJob, dataset: TrainingDataset) -> TrainingJob:
        """Run the training pipeline."""
        try:
            job.status = TrainingStatus.PREPROCESSING
            job.started_at = datetime.utcnow()
            
            # Step 1: Preprocess data
            train_data, val_data = self._preprocess_data(dataset, job.config)
            
            # Step 2: Run training
            job.status = TrainingStatus.TRAINING
            output_path = await self._run_training_loop(train_data, val_data, job)
            
            # Step 3: Validate model
            job.status = TrainingStatus.VALIDATING
            metrics = await self._validate_model(output_path, val_data)
            job.metrics = metrics
            
            # Step 4: Complete
            job.status = TrainingStatus.COMPLETED
            job.output_model_path = output_path
            job.completed_at = datetime.utcnow()
            
            logger.info(
                "Training completed",
                job_id=job.id,
                duration_seconds=job.duration_seconds,
                eval_loss=metrics.eval_loss,
            )
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            logger.error("Training failed", job_id=job.id, error=str(e))
            
        return job

    def _preprocess_data(
        self,
        dataset: TrainingDataset,
        config: TrainingConfig,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Preprocess dataset for training."""
        # Filter by quality score
        quality_examples = [ex for ex in dataset.examples if ex.quality_score >= 0.5]
        
        # Convert to training format
        format_type = "openai" if "openai" in config.model_type.value else "alpaca"
        formatted = [ex.to_training_format(format_type) for ex in quality_examples]
        
        # Split into train/validation
        split_idx = int(len(formatted) * (1 - config.validation_split))
        return formatted[:split_idx], formatted[split_idx:]

    async def _run_training_loop(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        job: TrainingJob,
    ) -> str:
        """Run the actual training loop.
        
        In production, this would integrate with:
        - HuggingFace Transformers + PEFT for LoRA
        - OpenAI fine-tuning API
        - Local training infrastructure
        """
        config = job.config
        output_dir = Path(config.output_dir) / job.id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate training for now (in production, integrate with actual training)
        logger.info(
            "Starting training loop",
            job_id=job.id,
            train_samples=len(train_data),
            val_samples=len(val_data),
            epochs=config.num_epochs,
        )
        
        # Save training data for later use
        train_file = output_dir / "train.jsonl"
        val_file = output_dir / "val.jsonl"
        
        with open(train_file, "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")
        
        with open(val_file, "w") as f:
            for item in val_data:
                f.write(json.dumps(item) + "\n")
        
        # Save config
        config_file = output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump({
                "model_type": config.model_type.value,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            }, f, indent=2)
        
        # In production: call HuggingFace trainer or OpenAI API
        # For now, return the output directory
        return str(output_dir)

    async def _validate_model(
        self,
        model_path: str,
        val_data: list[dict[str, Any]],
    ) -> TrainingMetrics:
        """Validate the fine-tuned model."""
        # In production: load model and run validation
        # For now, return placeholder metrics
        return TrainingMetrics(
            train_loss=0.5,
            eval_loss=0.6,
            train_accuracy=0.85,
            eval_accuracy=0.82,
            epoch=3,
        )


class ModelServer:
    """Serves fine-tuned models with automatic fallback."""

    def __init__(
        self,
        models_dir: Path | None = None,
        fallback_provider: str = "openai",
    ) -> None:
        self.models_dir = models_dir or Path("./fine_tuned_models")
        self.fallback_provider = fallback_provider
        self._active_models: dict[str, FineTunedModel] = {}
        self._model_stats: dict[str, dict[str, Any]] = {}

    def register_model(self, model: FineTunedModel) -> None:
        """Register a fine-tuned model for serving."""
        key = model.org_id or "global"
        self._active_models[key] = model
        self._model_stats[model.id] = {
            "requests": 0,
            "successes": 0,
            "fallbacks": 0,
            "avg_latency_ms": 0.0,
        }
        logger.info(
            "Registered fine-tuned model",
            model_id=model.id,
            org_id=model.org_id,
            base_model=model.base_model.value,
        )

    def get_active_model(self, org_id: str | None = None) -> FineTunedModel | None:
        """Get the active model for an org."""
        key = org_id or "global"
        return self._active_models.get(key) or self._active_models.get("global")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        org_id: str | None = None,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate response using fine-tuned model with fallback."""
        model = self.get_active_model(org_id)
        
        if model and model.is_active:
            try:
                result = await self._generate_with_finetuned(
                    model, prompt, system_prompt, max_tokens
                )
                self._record_success(model.id, result.get("latency_ms", 0))
                return result
            except Exception as e:
                logger.warning(
                    "Fine-tuned model failed, falling back",
                    model_id=model.id,
                    error=str(e),
                )
                self._record_fallback(model.id)
        
        # Fallback to base model
        return await self._generate_with_fallback(prompt, system_prompt, max_tokens)

    async def _generate_with_finetuned(
        self,
        model: FineTunedModel,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Generate using fine-tuned model.
        
        In production: load model and run inference
        """
        import time
        start_time = time.time()
        
        # Placeholder for actual model inference
        # In production: use HuggingFace pipeline or vLLM
        
        latency_ms = (time.time() - start_time) * 1000
        return {
            "content": "{}",  # JSON response
            "model": f"finetuned:{model.id}",
            "latency_ms": latency_ms,
            "tokens": 100,  # Placeholder
        }

    async def _generate_with_fallback(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Generate using fallback base model."""
        if self.fallback_provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            
            return {
                "content": response.choices[0].message.content or "",
                "model": "gpt-4-turbo-preview",
                "tokens": response.usage.total_tokens if response.usage else 0,
            }
        
        raise ValueError(f"Unknown fallback provider: {self.fallback_provider}")

    def _record_success(self, model_id: str, latency_ms: float) -> None:
        """Record successful model call."""
        stats = self._model_stats.get(model_id, {})
        stats["requests"] = stats.get("requests", 0) + 1
        stats["successes"] = stats.get("successes", 0) + 1
        
        # Update rolling average latency
        n = stats["successes"]
        old_avg = stats.get("avg_latency_ms", 0)
        stats["avg_latency_ms"] = old_avg + (latency_ms - old_avg) / n
        
        self._model_stats[model_id] = stats

    def _record_fallback(self, model_id: str) -> None:
        """Record fallback to base model."""
        stats = self._model_stats.get(model_id, {})
        stats["requests"] = stats.get("requests", 0) + 1
        stats["fallbacks"] = stats.get("fallbacks", 0) + 1
        self._model_stats[model_id] = stats

    def get_model_stats(self, model_id: str) -> dict[str, Any]:
        """Get statistics for a model."""
        return self._model_stats.get(model_id, {})


class FineTuningManager:
    """Main manager for the fine-tuning pipeline."""

    def __init__(
        self,
        storage_path: Path | None = None,
        models_path: Path | None = None,
    ) -> None:
        self.collector = DataCollector(storage_path)
        self.pipeline = TrainingPipeline()
        self.server = ModelServer(models_path)
        self._models: dict[str, FineTunedModel] = {}

    async def start_training(
        self,
        dataset_name: str,
        org_id: str | None = None,
        config: TrainingConfig | None = None,
    ) -> TrainingJob:
        """Start a training job for a dataset."""
        dataset = self.collector.get_dataset(dataset_name, org_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        if len(dataset.examples) < 10:
            raise ValueError(f"Dataset too small: {len(dataset.examples)} examples (minimum 10)")
        
        job = self.pipeline.create_job(dataset, config)
        job = await self.pipeline.run_training(job, dataset)
        
        if job.status == TrainingStatus.COMPLETED and job.output_model_path:
            # Create model artifact
            model = FineTunedModel(
                org_id=org_id,
                name=f"{dataset_name}_{job.id[:8]}",
                base_model=job.config.model_type,
                model_path=job.output_model_path,
                training_job_id=job.id,
                metrics=job.metrics,
            )
            self._models[model.id] = model
            logger.info("Created fine-tuned model", model_id=model.id)
        
        return job

    def activate_model(self, model_id: str) -> bool:
        """Activate a model for serving."""
        model = self._models.get(model_id)
        if not model:
            return False
        
        # Deactivate other models for same org
        for m in self._models.values():
            if m.org_id == model.org_id:
                m.is_active = False
        
        model.is_active = True
        self.server.register_model(model)
        return True

    def get_training_summary(self, org_id: str | None = None) -> dict[str, Any]:
        """Get summary of training activity."""
        datasets = self.collector.list_datasets(org_id)
        models = [m for m in self._models.values() if m.org_id == org_id or org_id is None]
        
        return {
            "datasets": [
                {
                    "name": ds.name,
                    "examples": len(ds.examples),
                    "stats": ds.stats,
                }
                for ds in datasets
            ],
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "is_active": m.is_active,
                    "base_model": m.base_model.value,
                    "metrics": {
                        "eval_loss": m.metrics.eval_loss,
                        "eval_accuracy": m.metrics.eval_accuracy,
                    },
                }
                for m in models
            ],
            "active_model": next(
                (m.id for m in models if m.is_active), None
            ),
        }


# Global manager instance
_fine_tuning_manager: FineTuningManager | None = None


def get_fine_tuning_manager() -> FineTuningManager:
    """Get the global fine-tuning manager."""
    global _fine_tuning_manager
    if _fine_tuning_manager is None:
        _fine_tuning_manager = FineTuningManager()
    return _fine_tuning_manager


def reset_fine_tuning_manager() -> None:
    """Reset the global fine-tuning manager (for testing)."""
    global _fine_tuning_manager
    _fine_tuning_manager = None
