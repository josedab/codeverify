"""
AI Model Integration for Code Fingerprinting

Provides integration with real ML models for production-ready AI code detection:
- ONNX Runtime for efficient inference
- HuggingFace transformers integration
- Local model loading and caching
- Batch processing support
- Model ensemble for improved accuracy
"""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import numpy as np


class ModelBackend(Enum):
    """Supported model backends."""
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"
    SKLEARN = "sklearn"
    CUSTOM = "custom"
    MOCK = "mock"  # For testing


@dataclass
class ModelConfig:
    """Configuration for an AI detection model."""
    name: str
    version: str
    backend: ModelBackend
    model_path: Optional[Path] = None
    model_id: Optional[str] = None  # HuggingFace model ID
    input_features: list[str] = field(default_factory=list)
    output_classes: list[str] = field(default_factory=lambda: ["human", "ai_generated"])
    threshold: float = 0.5
    max_sequence_length: int = 2048
    batch_size: int = 32


@dataclass
class PredictionResult:
    """Result from model prediction."""
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    features_used: dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    model_version: str = ""


class FeatureExtractor:
    """Extract features from code for model input."""

    # Feature names matching research papers
    STRUCTURAL_FEATURES = [
        "line_count", "avg_line_length", "line_length_variance",
        "indent_depth_avg", "indent_depth_max", "blank_line_ratio",
        "comment_density", "function_count", "class_count"
    ]

    STYLE_FEATURES = [
        "naming_snake_case_ratio", "naming_camel_case_ratio",
        "avg_identifier_length", "identifier_length_variance",
        "docstring_coverage", "type_hint_ratio"
    ]

    AI_PATTERN_FEATURES = [
        "has_placeholder_pattern", "has_todo_pattern",
        "has_example_usage", "verbose_comment_ratio",
        "perfect_formatting_score", "generic_variable_ratio"
    ]

    ENTROPY_FEATURES = [
        "token_entropy", "structure_regularity",
        "repetition_score", "uniqueness_score"
    ]

    def __init__(self):
        self.feature_names = (
            self.STRUCTURAL_FEATURES +
            self.STYLE_FEATURES +
            self.AI_PATTERN_FEATURES +
            self.ENTROPY_FEATURES
        )

    def extract(self, code: str, language: str = "python") -> dict[str, float]:
        """Extract all features from code."""
        features = {}
        
        # Structural features
        features.update(self._extract_structural(code))
        
        # Style features
        features.update(self._extract_style(code, language))
        
        # AI pattern features
        features.update(self._extract_ai_patterns(code))
        
        # Entropy features
        features.update(self._extract_entropy(code))
        
        return features

    def _extract_structural(self, code: str) -> dict[str, float]:
        """Extract structural features."""
        lines = code.split('\n')
        non_empty = [l for l in lines if l.strip()]
        
        line_lengths = [len(l) for l in non_empty]
        avg_length = sum(line_lengths) / max(len(line_lengths), 1)
        
        # Calculate variance
        if len(line_lengths) > 1:
            variance = sum((l - avg_length) ** 2 for l in line_lengths) / len(line_lengths)
        else:
            variance = 0.0
        
        # Indent depths
        indent_depths = []
        for line in non_empty:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_depths.append(indent)
        
        avg_indent = sum(indent_depths) / max(len(indent_depths), 1)
        max_indent = max(indent_depths) if indent_depths else 0
        
        # Blank line ratio
        blank_ratio = (len(lines) - len(non_empty)) / max(len(lines), 1)
        
        # Comment density
        comment_lines = len([l for l in lines if l.strip().startswith(('#', '//', '/*', '*'))])
        comment_density = comment_lines / max(len(non_empty), 1)
        
        # Count functions and classes
        function_count = len([l for l in lines if 'def ' in l or 'function ' in l])
        class_count = len([l for l in lines if 'class ' in l])
        
        return {
            "line_count": float(len(lines)),
            "avg_line_length": avg_length,
            "line_length_variance": variance,
            "indent_depth_avg": avg_indent,
            "indent_depth_max": float(max_indent),
            "blank_line_ratio": blank_ratio,
            "comment_density": comment_density,
            "function_count": float(function_count),
            "class_count": float(class_count)
        }

    def _extract_style(self, code: str, language: str) -> dict[str, float]:
        """Extract style features."""
        import re
        
        # Find identifiers
        identifier_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
        identifiers = identifier_pattern.findall(code)
        
        # Filter out keywords
        python_keywords = {'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 
                          'import', 'from', 'try', 'except', 'finally', 'with', 'as',
                          'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is'}
        identifiers = [i for i in identifiers if i not in python_keywords]
        
        if not identifiers:
            return {
                "naming_snake_case_ratio": 0.0,
                "naming_camel_case_ratio": 0.0,
                "avg_identifier_length": 0.0,
                "identifier_length_variance": 0.0,
                "docstring_coverage": 0.0,
                "type_hint_ratio": 0.0
            }
        
        # Naming conventions
        snake_case = sum(1 for i in identifiers if '_' in i and i.islower())
        camel_case = sum(1 for i in identifiers if not '_' in i and any(c.isupper() for c in i[1:]))
        
        snake_ratio = snake_case / len(identifiers)
        camel_ratio = camel_case / len(identifiers)
        
        # Identifier lengths
        lengths = [len(i) for i in identifiers]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths) if len(lengths) > 1 else 0
        
        # Docstring coverage
        docstring_pattern = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
        docstrings = docstring_pattern.findall(code)
        function_count = len(re.findall(r'\bdef\s+\w+', code))
        docstring_coverage = len(docstrings) / max(function_count, 1)
        
        # Type hint ratio
        type_hints = len(re.findall(r':\s*\w+(?:\[[\w\[\],\s]+\])?\s*[=)]', code))
        type_hint_ratio = type_hints / max(function_count, 1)
        
        return {
            "naming_snake_case_ratio": snake_ratio,
            "naming_camel_case_ratio": camel_ratio,
            "avg_identifier_length": avg_length,
            "identifier_length_variance": variance,
            "docstring_coverage": min(docstring_coverage, 1.0),
            "type_hint_ratio": min(type_hint_ratio, 1.0)
        }

    def _extract_ai_patterns(self, code: str) -> dict[str, float]:
        """Extract AI-specific pattern features."""
        import re
        
        code_lower = code.lower()
        lines = code.split('\n')
        
        # Placeholder patterns
        placeholder_patterns = [
            r'# TODO:', r'# FIXME:', r'# implement', r'pass\s*$',
            r'raise NotImplementedError', r'\.\.\.', r'# your code here'
        ]
        has_placeholder = any(re.search(p, code, re.IGNORECASE) for p in placeholder_patterns)
        
        # TODO patterns
        has_todo = 'todo' in code_lower or 'fixme' in code_lower
        
        # Example usage patterns
        example_patterns = [r'# example', r'# usage', r'>>> ', r'# output:']
        has_example = any(re.search(p, code, re.IGNORECASE) for p in example_patterns)
        
        # Verbose comments (AI tends to over-explain)
        comment_lines = [l for l in lines if l.strip().startswith('#')]
        total_comment_length = sum(len(l) for l in comment_lines)
        verbose_ratio = total_comment_length / max(len(code), 1)
        
        # Perfect formatting (AI produces very consistent formatting)
        indent_sizes = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indent_sizes.append(indent)
        
        if indent_sizes:
            # Check if all indents are multiples of 4
            perfect_indent = all(i % 4 == 0 for i in indent_sizes)
            # Check consistency
            unique_indents = set(indent_sizes)
            formatting_score = 1.0 if perfect_indent and len(unique_indents) <= 5 else 0.5
        else:
            formatting_score = 0.5
        
        # Generic variable names (AI uses generic names like 'data', 'result', 'item')
        generic_names = ['data', 'result', 'item', 'value', 'temp', 'obj', 'arr', 'lst']
        name_pattern = re.compile(r'\b(' + '|'.join(generic_names) + r')\b', re.IGNORECASE)
        generic_count = len(name_pattern.findall(code))
        total_identifiers = len(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code))
        generic_ratio = generic_count / max(total_identifiers, 1)
        
        return {
            "has_placeholder_pattern": 1.0 if has_placeholder else 0.0,
            "has_todo_pattern": 1.0 if has_todo else 0.0,
            "has_example_usage": 1.0 if has_example else 0.0,
            "verbose_comment_ratio": min(verbose_ratio, 1.0),
            "perfect_formatting_score": formatting_score,
            "generic_variable_ratio": generic_ratio
        }

    def _extract_entropy(self, code: str) -> dict[str, float]:
        """Extract entropy and uniqueness features."""
        import math
        from collections import Counter
        
        # Tokenize
        tokens = code.split()
        if not tokens:
            return {
                "token_entropy": 0.0,
                "structure_regularity": 0.5,
                "repetition_score": 0.0,
                "uniqueness_score": 1.0
            }
        
        # Token entropy
        token_counts = Counter(tokens)
        total = len(tokens)
        entropy = 0.0
        for count in token_counts.values():
            prob = count / total
            entropy -= prob * math.log2(prob)
        
        # Normalize entropy
        max_entropy = math.log2(len(token_counts)) if len(token_counts) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Structure regularity (AI code tends to be very regular)
        lines = [l for l in code.split('\n') if l.strip()]
        if len(lines) > 1:
            line_lengths = [len(l) for l in lines]
            avg = sum(line_lengths) / len(line_lengths)
            regularity = 1.0 - (sum(abs(l - avg) for l in line_lengths) / (len(lines) * max(avg, 1)))
        else:
            regularity = 0.5
        
        # Repetition score
        unique_tokens = len(set(tokens))
        repetition = 1.0 - (unique_tokens / len(tokens))
        
        # Uniqueness score
        uniqueness = unique_tokens / len(tokens)
        
        return {
            "token_entropy": normalized_entropy,
            "structure_regularity": max(0, min(regularity, 1.0)),
            "repetition_score": repetition,
            "uniqueness_score": uniqueness
        }

    def to_array(self, features: dict[str, float]) -> np.ndarray:
        """Convert features dict to numpy array in consistent order."""
        return np.array([features.get(name, 0.0) for name in self.feature_names])


class BaseModelBackend(ABC):
    """Abstract base class for model backends."""

    @abstractmethod
    def load(self, config: ModelConfig) -> bool:
        """Load the model."""
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> PredictionResult:
        """Run prediction on features."""
        pass

    @abstractmethod
    def predict_batch(self, features_batch: np.ndarray) -> list[PredictionResult]:
        """Run batch prediction."""
        pass


class MockModelBackend(BaseModelBackend):
    """Mock backend for testing without real model files."""

    def __init__(self):
        self.loaded = False
        self.config = None
        # Learned weights (simulated from research)
        self.weights = {
            "perfect_formatting_score": 0.25,
            "verbose_comment_ratio": 0.20,
            "has_placeholder_pattern": 0.15,
            "generic_variable_ratio": 0.15,
            "structure_regularity": 0.10,
            "docstring_coverage": 0.08,
            "has_example_usage": 0.07,
        }
        self.bias = -0.35  # Threshold adjustment

    def load(self, config: ModelConfig) -> bool:
        self.config = config
        self.loaded = True
        return True

    def predict(self, features: np.ndarray) -> PredictionResult:
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # Simple weighted sum based on research features
        extractor = FeatureExtractor()
        feature_dict = {name: float(features[i]) for i, name in enumerate(extractor.feature_names)}
        
        score = self.bias
        features_used = {}
        for feature, weight in self.weights.items():
            value = feature_dict.get(feature, 0.0)
            score += weight * value
            features_used[feature] = value
        
        # Sigmoid to get probability
        import math
        prob_ai = 1 / (1 + math.exp(-score * 5))  # Scale for sharper decision
        prob_human = 1 - prob_ai
        
        predicted_class = "ai_generated" if prob_ai > self.config.threshold else "human"
        
        return PredictionResult(
            predicted_class=predicted_class,
            confidence=max(prob_ai, prob_human),
            class_probabilities={"human": prob_human, "ai_generated": prob_ai},
            features_used=features_used,
            model_version=self.config.version
        )

    def predict_batch(self, features_batch: np.ndarray) -> list[PredictionResult]:
        return [self.predict(features) for features in features_batch]


class ONNXModelBackend(BaseModelBackend):
    """ONNX Runtime backend for production inference."""

    def __init__(self):
        self.session = None
        self.config = None

    def load(self, config: ModelConfig) -> bool:
        try:
            import onnxruntime as ort
            
            if not config.model_path or not config.model_path.exists():
                return False
            
            self.session = ort.InferenceSession(str(config.model_path))
            self.config = config
            return True
        except ImportError:
            print("ONNX Runtime not installed. Run: pip install onnxruntime")
            return False
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            return False

    def predict(self, features: np.ndarray) -> PredictionResult:
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        import time
        start = time.time()
        
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: features.reshape(1, -1).astype(np.float32)})
        
        inference_time = (time.time() - start) * 1000
        
        probabilities = outputs[0][0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = self.config.output_classes[predicted_idx]
        
        return PredictionResult(
            predicted_class=predicted_class,
            confidence=float(probabilities[predicted_idx]),
            class_probabilities={
                cls: float(probabilities[i]) 
                for i, cls in enumerate(self.config.output_classes)
            },
            inference_time_ms=inference_time,
            model_version=self.config.version
        )

    def predict_batch(self, features_batch: np.ndarray) -> list[PredictionResult]:
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        import time
        start = time.time()
        
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: features_batch.astype(np.float32)})
        
        inference_time = (time.time() - start) * 1000 / len(features_batch)
        
        results = []
        for i, probs in enumerate(outputs[0]):
            predicted_idx = int(np.argmax(probs))
            predicted_class = self.config.output_classes[predicted_idx]
            
            results.append(PredictionResult(
                predicted_class=predicted_class,
                confidence=float(probs[predicted_idx]),
                class_probabilities={
                    cls: float(probs[j])
                    for j, cls in enumerate(self.config.output_classes)
                },
                inference_time_ms=inference_time,
                model_version=self.config.version
            ))
        
        return results


class HuggingFaceModelBackend(BaseModelBackend):
    """HuggingFace Transformers backend for code language models."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None

    def load(self, config: ModelConfig) -> bool:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            model_id = config.model_id or str(config.model_path)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
            self.config = config
            return True
        except ImportError:
            print("Transformers not installed. Run: pip install transformers")
            return False
        except Exception as e:
            print(f"Failed to load HuggingFace model: {e}")
            return False

    def predict(self, features: np.ndarray) -> PredictionResult:
        raise NotImplementedError("Use predict_from_code for HuggingFace models")

    def predict_from_code(self, code: str) -> PredictionResult:
        """Predict directly from code string (HuggingFace uses text, not features)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        import time
        import torch
        
        start = time.time()
        
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_sequence_length
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
        
        inference_time = (time.time() - start) * 1000
        
        predicted_idx = int(torch.argmax(probs))
        predicted_class = self.config.output_classes[predicted_idx]
        
        return PredictionResult(
            predicted_class=predicted_class,
            confidence=float(probs[predicted_idx]),
            class_probabilities={
                cls: float(probs[i])
                for i, cls in enumerate(self.config.output_classes)
            },
            inference_time_ms=inference_time,
            model_version=self.config.version
        )

    def predict_batch(self, features_batch: np.ndarray) -> list[PredictionResult]:
        raise NotImplementedError("Use predict_batch_from_code for HuggingFace models")


class ModelEnsemble:
    """Ensemble of multiple models for improved accuracy."""

    def __init__(self, models: list[tuple[BaseModelBackend, float]]):
        """Initialize with list of (model, weight) tuples."""
        self.models = models
        total_weight = sum(w for _, w in models)
        self.normalized_weights = [w / total_weight for _, w in models]

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Weighted ensemble prediction."""
        combined_probs = {}
        total_inference_time = 0
        all_features_used = {}
        
        for (model, _), weight in zip(self.models, self.normalized_weights):
            result = model.predict(features)
            total_inference_time += result.inference_time_ms
            
            for cls, prob in result.class_probabilities.items():
                combined_probs[cls] = combined_probs.get(cls, 0) + prob * weight
            
            all_features_used.update(result.features_used)
        
        predicted_class = max(combined_probs, key=combined_probs.get)
        
        return PredictionResult(
            predicted_class=predicted_class,
            confidence=combined_probs[predicted_class],
            class_probabilities=combined_probs,
            features_used=all_features_used,
            inference_time_ms=total_inference_time,
            model_version="ensemble"
        )


class AICodeDetector:
    """Main class for AI code detection with model integration."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(
            name="codeverify-ai-detector",
            version="1.0.0",
            backend=ModelBackend.MOCK,
            threshold=0.5
        )
        self.feature_extractor = FeatureExtractor()
        self.backend: Optional[BaseModelBackend] = None
        self._load_backend()

    def _load_backend(self):
        """Load appropriate backend based on config."""
        if self.config.backend == ModelBackend.MOCK:
            self.backend = MockModelBackend()
        elif self.config.backend == ModelBackend.ONNX:
            self.backend = ONNXModelBackend()
        elif self.config.backend == ModelBackend.HUGGINGFACE:
            self.backend = HuggingFaceModelBackend()
        else:
            self.backend = MockModelBackend()
        
        self.backend.load(self.config)

    def detect(self, code: str, language: str = "python") -> PredictionResult:
        """Detect if code is AI-generated."""
        # Extract features
        features = self.feature_extractor.extract(code, language)
        features_array = self.feature_extractor.to_array(features)
        
        # Run prediction
        result = self.backend.predict(features_array)
        result.features_used = features
        
        return result

    def detect_batch(self, codes: list[str], language: str = "python") -> list[PredictionResult]:
        """Batch detection for multiple code samples."""
        all_features = []
        for code in codes:
            features = self.feature_extractor.extract(code, language)
            all_features.append(self.feature_extractor.to_array(features))
        
        features_batch = np.array(all_features)
        return self.backend.predict_batch(features_batch)


@dataclass
class ModelRegistry:
    """Registry of available models."""
    models: dict[str, ModelConfig] = field(default_factory=dict)

    def register(self, config: ModelConfig):
        """Register a model configuration."""
        self.models[config.name] = config

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get model config by name."""
        return self.models.get(name)

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self.models.keys())


# Global model registry with default models
_model_registry = ModelRegistry()
_model_registry.register(ModelConfig(
    name="default",
    version="1.0.0",
    backend=ModelBackend.MOCK,
    threshold=0.5
))


def get_detector(model_name: str = "default") -> AICodeDetector:
    """Get a detector with specified model."""
    config = _model_registry.get(model_name)
    if config is None:
        config = _model_registry.get("default")
    return AICodeDetector(config)


def detect_ai_code(code: str, language: str = "python") -> PredictionResult:
    """Convenience function for quick detection."""
    detector = get_detector()
    return detector.detect(code, language)


# Testing
if __name__ == "__main__":
    # Test with sample code
    human_code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.qty
    return total
"""

    ai_code = """
def calculate_total(items: list) -> float:
    \"\"\"
    Calculate the total price of all items.
    
    Args:
        items: A list of item objects with price and quantity attributes.
        
    Returns:
        The total price as a float.
        
    Example:
        >>> items = [Item(price=10, qty=2), Item(price=5, qty=3)]
        >>> calculate_total(items)
        35.0
    \"\"\"
    # Initialize the total to zero
    total = 0.0
    
    # Iterate through each item and add to total
    for item in items:
        # Calculate subtotal for this item
        subtotal = item.price * item.qty
        total += subtotal
    
    # Return the final total
    return total
"""

    detector = AICodeDetector()
    
    print("Human code detection:")
    result = detector.detect(human_code)
    print(f"  Predicted: {result.predicted_class}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Probabilities: {result.class_probabilities}")
    
    print("\nAI code detection:")
    result = detector.detect(ai_code)
    print(f"  Predicted: {result.predicted_class}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Probabilities: {result.class_probabilities}")
