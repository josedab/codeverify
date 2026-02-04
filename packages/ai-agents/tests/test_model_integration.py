"""Tests for AI Model Integration."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from codeverify_agents.model_integration import (
    ModelBackend,
    ModelConfig,
    PredictionResult,
    FeatureExtractor,
    MockModelBackend,
    ONNXModelBackend,
    ModelEnsemble,
    AICodeDetector,
    ModelRegistry,
    get_detector,
    detect_ai_code,
)


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_extract_basic(self):
        """Test basic feature extraction."""
        code = '''
def hello():
    print("Hello, World!")

def add(a, b):
    return a + b
'''
        extractor = FeatureExtractor()
        features = extractor.extract(code)
        
        assert "line_count" in features
        assert "avg_line_length" in features
        assert "function_count" in features
        assert features["function_count"] == 2.0

    def test_extract_structural_features(self):
        """Test structural feature extraction."""
        code = '''
class MyClass:
    def method1(self):
        pass
    
    def method2(self):
        return 42
'''
        extractor = FeatureExtractor()
        features = extractor.extract(code)
        
        assert features["class_count"] == 1.0
        assert features["function_count"] == 2.0
        assert features["blank_line_ratio"] > 0

    def test_extract_style_features(self):
        """Test style feature extraction."""
        code = '''
def snake_case_function():
    my_variable = 10
    another_var = 20
    return my_variable + another_var
'''
        extractor = FeatureExtractor()
        features = extractor.extract(code)
        
        assert features["naming_snake_case_ratio"] > 0.5

    def test_extract_ai_patterns(self):
        """Test AI pattern detection."""
        ai_style_code = '''
def calculate_sum(numbers: list) -> int:
    """
    Calculate the sum of all numbers in a list.
    
    Args:
        numbers: A list of integers.
        
    Returns:
        The sum as an integer.
        
    Example:
        >>> calculate_sum([1, 2, 3])
        6
    """
    # Initialize result to zero
    result = 0
    
    # TODO: Add validation
    
    # Iterate through each number
    for num in numbers:
        result += num
    
    # Return the final result
    return result
'''
        extractor = FeatureExtractor()
        features = extractor.extract(ai_style_code)
        
        assert features["has_example_usage"] == 1.0
        assert features["has_todo_pattern"] == 1.0
        assert features["verbose_comment_ratio"] > 0.1

    def test_extract_entropy_features(self):
        """Test entropy feature extraction."""
        repetitive_code = '''
x = 1
x = 2
x = 3
x = 4
x = 5
'''
        extractor = FeatureExtractor()
        features = extractor.extract(repetitive_code)
        
        assert "token_entropy" in features
        assert "repetition_score" in features
        assert features["repetition_score"] > 0.3  # High repetition

    def test_to_array(self):
        """Test conversion to numpy array."""
        code = "def test(): pass"
        extractor = FeatureExtractor()
        features = extractor.extract(code)
        
        array = extractor.to_array(features)
        
        assert isinstance(array, np.ndarray)
        assert len(array) == len(extractor.feature_names)

    def test_empty_code(self):
        """Test with empty code."""
        extractor = FeatureExtractor()
        features = extractor.extract("")
        
        assert features["line_count"] == 1.0  # One empty line
        assert features["function_count"] == 0.0


class TestMockModelBackend:
    """Tests for MockModelBackend."""

    def test_load(self):
        """Test model loading."""
        config = ModelConfig(
            name="test",
            version="1.0",
            backend=ModelBackend.MOCK
        )
        backend = MockModelBackend()
        assert backend.load(config)
        assert backend.loaded

    def test_predict(self):
        """Test prediction."""
        config = ModelConfig(
            name="test",
            version="1.0",
            backend=ModelBackend.MOCK
        )
        backend = MockModelBackend()
        backend.load(config)
        
        extractor = FeatureExtractor()
        features = extractor.extract("def test(): pass")
        features_array = extractor.to_array(features)
        
        result = backend.predict(features_array)
        
        assert isinstance(result, PredictionResult)
        assert result.predicted_class in ["human", "ai_generated"]
        assert 0 <= result.confidence <= 1
        assert "human" in result.class_probabilities
        assert "ai_generated" in result.class_probabilities

    def test_predict_batch(self):
        """Test batch prediction."""
        config = ModelConfig(
            name="test",
            version="1.0",
            backend=ModelBackend.MOCK
        )
        backend = MockModelBackend()
        backend.load(config)
        
        extractor = FeatureExtractor()
        codes = ["def a(): pass", "class B: pass", "x = 1"]
        features_batch = np.array([
            extractor.to_array(extractor.extract(code))
            for code in codes
        ])
        
        results = backend.predict_batch(features_batch)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, PredictionResult)

    def test_predict_not_loaded(self):
        """Test prediction fails if not loaded."""
        backend = MockModelBackend()
        
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.predict(np.array([0.0] * 21))


class TestONNXModelBackend:
    """Tests for ONNXModelBackend."""

    def test_load_missing_model(self):
        """Test loading with missing model file."""
        config = ModelConfig(
            name="test",
            version="1.0",
            backend=ModelBackend.ONNX
        )
        backend = ONNXModelBackend()
        assert not backend.load(config)

    def test_load_nonexistent_path(self):
        """Test loading with nonexistent path."""
        from pathlib import Path
        config = ModelConfig(
            name="test",
            version="1.0",
            backend=ModelBackend.ONNX,
            model_path=Path("/nonexistent/model.onnx")
        )
        backend = ONNXModelBackend()
        assert not backend.load(config)


class TestModelEnsemble:
    """Tests for ModelEnsemble."""

    def test_ensemble_prediction(self):
        """Test ensemble combines predictions correctly."""
        config = ModelConfig(
            name="test",
            version="1.0",
            backend=ModelBackend.MOCK
        )
        
        # Create two mock backends with same config
        backend1 = MockModelBackend()
        backend1.load(config)
        backend1.bias = 0.5  # Biased towards AI
        
        backend2 = MockModelBackend()
        backend2.load(config)
        backend2.bias = -0.5  # Biased towards human
        
        ensemble = ModelEnsemble([
            (backend1, 0.6),
            (backend2, 0.4)
        ])
        
        extractor = FeatureExtractor()
        features = extractor.extract("def test(): pass")
        features_array = extractor.to_array(features)
        
        result = ensemble.predict(features_array)
        
        assert isinstance(result, PredictionResult)
        assert result.model_version == "ensemble"
        # Probabilities should sum to ~1
        assert 0.99 <= sum(result.class_probabilities.values()) <= 1.01


class TestAICodeDetector:
    """Tests for AICodeDetector."""

    def test_detect_human_code(self):
        """Test detection of human-like code."""
        detector = AICodeDetector()
        
        code = '''
def calc(items):
    t = 0
    for i in items:
        t += i.p * i.q
    return t
'''
        result = detector.detect(code)
        
        assert isinstance(result, PredictionResult)
        assert result.features_used is not None

    def test_detect_ai_code(self):
        """Test detection of AI-like code."""
        detector = AICodeDetector()
        
        code = '''
def calculate_total_price(items: list) -> float:
    """
    Calculate the total price of all items in the shopping cart.
    
    This function iterates through each item and computes the sum
    of price multiplied by quantity for all items.
    
    Args:
        items: A list of Item objects with price and quantity attributes.
        
    Returns:
        The total price as a floating point number.
        
    Example:
        >>> items = [Item(price=10.0, qty=2), Item(price=5.0, qty=3)]
        >>> calculate_total_price(items)
        35.0
    """
    # Initialize the total price to zero
    total_price = 0.0
    
    # Iterate through each item in the list
    for item in items:
        # Calculate the subtotal for this item
        item_subtotal = item.price * item.quantity
        # Add to the running total
        total_price += item_subtotal
    
    # Return the final calculated total
    return total_price
'''
        result = detector.detect(code)
        
        assert isinstance(result, PredictionResult)
        # AI code should have higher verbose comment ratio
        assert result.features_used.get("verbose_comment_ratio", 0) > 0.1

    def test_detect_batch(self):
        """Test batch detection."""
        detector = AICodeDetector()
        
        codes = [
            "def a(): pass",
            "class B:\n    def c(self):\n        return 1"
        ]
        
        results = detector.detect_batch(codes)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, PredictionResult)

    def test_custom_config(self):
        """Test detector with custom config."""
        config = ModelConfig(
            name="custom",
            version="2.0",
            backend=ModelBackend.MOCK,
            threshold=0.7
        )
        
        detector = AICodeDetector(config)
        result = detector.detect("def test(): pass")
        
        assert result.model_version == "2.0"


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving models."""
        registry = ModelRegistry()
        
        config = ModelConfig(
            name="my_model",
            version="1.0",
            backend=ModelBackend.MOCK
        )
        
        registry.register(config)
        
        retrieved = registry.get("my_model")
        assert retrieved == config

    def test_get_nonexistent(self):
        """Test getting nonexistent model."""
        registry = ModelRegistry()
        assert registry.get("nonexistent") is None

    def test_list_models(self):
        """Test listing models."""
        registry = ModelRegistry()
        
        registry.register(ModelConfig(name="model1", version="1.0", backend=ModelBackend.MOCK))
        registry.register(ModelConfig(name="model2", version="1.0", backend=ModelBackend.MOCK))
        
        models = registry.list_models()
        
        assert "model1" in models
        assert "model2" in models


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_detector_default(self):
        """Test getting default detector."""
        detector = get_detector()
        assert isinstance(detector, AICodeDetector)

    def test_get_detector_nonexistent(self):
        """Test getting nonexistent model falls back to default."""
        detector = get_detector("nonexistent_model")
        assert isinstance(detector, AICodeDetector)

    def test_detect_ai_code_function(self):
        """Test quick detection function."""
        result = detect_ai_code("def test(): pass")
        
        assert isinstance(result, PredictionResult)
        assert result.predicted_class in ["human", "ai_generated"]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_short_code(self):
        """Test with very short code."""
        detector = AICodeDetector()
        result = detector.detect("x=1")
        
        assert isinstance(result, PredictionResult)

    def test_very_long_code(self):
        """Test with very long code."""
        detector = AICodeDetector()
        
        # Generate long repetitive code
        code = "\n".join([f"def func_{i}(): pass" for i in range(1000)])
        result = detector.detect(code)
        
        assert isinstance(result, PredictionResult)

    def test_code_with_unicode(self):
        """Test with unicode characters."""
        detector = AICodeDetector()
        
        code = '''
def 日本語関数():
    """日本語のドキュメント"""
    return "こんにちは"
'''
        result = detector.detect(code)
        
        assert isinstance(result, PredictionResult)

    def test_code_with_special_chars(self):
        """Test with special characters."""
        detector = AICodeDetector()
        
        code = '''
def special():
    regex = r"^\d+\.\d+$"
    template = f"Value: {{{value}}}"
    return regex, template
'''
        result = detector.detect(code)
        
        assert isinstance(result, PredictionResult)

    def test_mixed_language_code(self):
        """Test with mixed language styles."""
        detector = AICodeDetector()
        
        javascript_style = '''
function calculateTotal(items) {
    let total = 0;
    for (const item of items) {
        total += item.price * item.qty;
    }
    return total;
}
'''
        result = detector.detect(javascript_style, language="javascript")
        
        assert isinstance(result, PredictionResult)


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_creation(self):
        """Test creating prediction result."""
        result = PredictionResult(
            predicted_class="human",
            confidence=0.85,
            class_probabilities={"human": 0.85, "ai_generated": 0.15}
        )
        
        assert result.predicted_class == "human"
        assert result.confidence == 0.85
        assert len(result.class_probabilities) == 2

    def test_default_values(self):
        """Test default values."""
        result = PredictionResult(
            predicted_class="ai_generated",
            confidence=0.9,
            class_probabilities={"ai_generated": 0.9, "human": 0.1}
        )
        
        assert result.features_used == {}
        assert result.inference_time_ms == 0.0
        assert result.model_version == ""
