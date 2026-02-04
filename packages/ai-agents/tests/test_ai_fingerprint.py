"""Tests for AI Code Fingerprinting Agent."""

import pytest

from codeverify_agents.ai_fingerprint import (
    AIClassifier,
    AIFingerprintAgent,
    AIModel,
    CodeMetrics,
    FeatureExtractor,
    FingerprintResult,
    compute_code_hash,
)


# Sample code snippets for testing
AI_GENERATED_CODE = '''
def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.
    
    This function calculates the Fibonacci sequence recursively.
    The Fibonacci sequence is a series of numbers where each number
    is the sum of the two preceding ones.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
        
    Example:
        >>> calculate_fibonacci(5)
        5
    """
    # Base case: first two Fibonacci numbers
    if n <= 1:
        return n
    
    # Recursive case: sum of previous two numbers
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)


def calculate_factorial(n: int) -> int:
    """
    Calculate the factorial of a number.
    
    Args:
        n: A non-negative integer
        
    Returns:
        The factorial of n
        
    Raises:
        ValueError: If n is negative
    """
    # TODO: implement error handling
    if n < 0:
        raise ValueError("n must be non-negative")
    
    # Base case
    if n == 0:
        return 1
    
    # Recursive case
    return n * calculate_factorial(n - 1)


if __name__ == "__main__":
    # Example usage
    pass
'''

HUMAN_WRITTEN_CODE = '''
def fib(n):
    """Get fib number"""
    if n<2: return n
    a,b = 0,1
    for _ in range(n-1):
        a,b=b,a+b
    return b

def fact(n):
    res = 1
    for i in range(2,n+1):
        res *= i
    return res

# quick test
assert fib(10) == 55
print("ok")
'''

MIXED_CODE_WITH_PLACEHOLDERS = '''
class UserService:
    """Service for user operations."""
    
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id: int) -> dict:
        """Get user by ID."""
        raise NotImplementedError
    
    def create_user(self, data: dict) -> int:
        """Create a new user."""
        pass  # TODO: implement this
    
    def update_user(self, user_id: int, data: dict) -> bool:
        """Update user."""
        ...  # placeholder
'''


class TestFeatureExtractor:
    """Test the feature extraction component."""
    
    def test_extract_basic_metrics(self):
        """Test extraction of basic code metrics."""
        extractor = FeatureExtractor()
        metrics, features = extractor.extract(AI_GENERATED_CODE, "python")
        
        assert metrics.line_count > 0
        assert metrics.function_count == 2
        assert metrics.docstring_count >= 2
        assert metrics.has_type_hints is True
    
    def test_extract_comment_metrics(self):
        """Test comment-related metrics."""
        extractor = FeatureExtractor()
        metrics, features = extractor.extract(AI_GENERATED_CODE, "python")
        
        assert metrics.comment_lines > 0
        assert metrics.comment_density > 0
    
    def test_extract_ai_patterns(self):
        """Test detection of AI-specific patterns."""
        extractor = FeatureExtractor()
        metrics, _ = extractor.extract(MIXED_CODE_WITH_PLACEHOLDERS, "python")
        
        assert metrics.has_placeholder_code is True
    
    def test_extract_human_code_patterns(self):
        """Test that human code has different metrics."""
        extractor = FeatureExtractor()
        metrics_ai, _ = extractor.extract(AI_GENERATED_CODE, "python")
        metrics_human, _ = extractor.extract(HUMAN_WRITTEN_CODE, "python")
        
        # Human code typically has less consistent formatting
        # and shorter average line lengths
        assert metrics_human.docstring_count < metrics_ai.docstring_count
    
    def test_feature_vector_complete(self):
        """Test that feature vector contains expected keys."""
        extractor = FeatureExtractor()
        _, features = extractor.extract(AI_GENERATED_CODE, "python")
        
        expected_features = [
            "f_line_count_norm",
            "f_comment_density",
            "f_docstring_coverage",
            "f_placeholder_code",
            "f_indent_consistency",
        ]
        
        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"


class TestAIClassifier:
    """Test the AI classification component."""
    
    def test_classify_ai_generated(self):
        """Test classification of AI-generated code."""
        extractor = FeatureExtractor()
        classifier = AIClassifier()
        
        _, features = extractor.extract(AI_GENERATED_CODE, "python")
        is_ai, confidence = classifier.classify(features)
        
        # Should detect as AI-generated with reasonable confidence
        assert is_ai is True
        assert confidence > 0.5
    
    def test_classify_human_written(self):
        """Test classification of human-written code."""
        extractor = FeatureExtractor()
        classifier = AIClassifier()
        
        _, features = extractor.extract(HUMAN_WRITTEN_CODE, "python")
        is_ai, confidence = classifier.classify(features)
        
        # Should detect as human-written (or low AI confidence)
        assert is_ai is False or confidence < 0.7
    
    def test_predict_model(self):
        """Test prediction of specific AI model."""
        extractor = FeatureExtractor()
        classifier = AIClassifier()
        
        _, features = extractor.extract(AI_GENERATED_CODE, "python")
        model, model_confidence = classifier.predict_model(AI_GENERATED_CODE, features)
        
        assert isinstance(model, AIModel)
        assert sum(model_confidence.values()) > 0


class TestAIFingerprintAgent:
    """Test the full fingerprinting agent."""
    
    @pytest.mark.asyncio
    async def test_analyze_ai_generated(self):
        """Test analyzing AI-generated code."""
        agent = AIFingerprintAgent()
        result = await agent.analyze(AI_GENERATED_CODE, {"language": "python"})
        
        assert result.success is True
        assert "is_ai_generated" in result.data
        assert "confidence" in result.data
        assert "detected_model" in result.data
    
    @pytest.mark.asyncio
    async def test_analyze_human_written(self):
        """Test analyzing human-written code."""
        agent = AIFingerprintAgent()
        result = await agent.analyze(HUMAN_WRITTEN_CODE, {"language": "python"})
        
        assert result.success is True
        data = result.data
        
        # Human code should be detected as such
        assert data["is_ai_generated"] is False or data["confidence"] < 0.6
    
    @pytest.mark.asyncio
    async def test_fingerprint_with_placeholders(self):
        """Test that placeholder code is flagged."""
        agent = AIFingerprintAgent()
        result = await agent.fingerprint(MIXED_CODE_WITH_PLACEHOLDERS, {"language": "python"})
        
        assert isinstance(result, FingerprintResult)
        assert "placeholder" in str(result.risk_factors).lower() or \
               "NotImplementedError" in str(result.risk_factors)
    
    @pytest.mark.asyncio
    async def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        agent = AIFingerprintAgent()
        result = await agent.fingerprint(AI_GENERATED_CODE, {"language": "python"})
        
        # Should have some recommendations for AI-generated code
        if result.is_ai_generated and result.confidence > 0.7:
            assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_explanation_generated(self):
        """Test that explanations are generated."""
        agent = AIFingerprintAgent()
        result = await agent.fingerprint(AI_GENERATED_CODE, {"language": "python"})
        
        assert result.explanation
        assert "confidence" in result.explanation.lower() or "%" in result.explanation


class TestFingerprintResult:
    """Test the result data class."""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = FingerprintResult(
            is_ai_generated=True,
            confidence=0.85,
            detected_model=AIModel.GITHUB_COPILOT,
            model_confidence={"github_copilot": 0.6, "chatgpt": 0.3},
            features={"f_test": 0.5},
            explanation="Test explanation",
            risk_factors=["risk1"],
            recommendations=["rec1"],
        )
        
        data = result.to_dict()
        
        assert data["is_ai_generated"] is True
        assert data["confidence"] == 0.85
        assert data["detected_model"] == "github_copilot"
        assert "github_copilot" in data["model_confidence"]


class TestCodeHash:
    """Test code hashing utility."""
    
    def test_hash_stability(self):
        """Test that hash is stable for same code."""
        hash1 = compute_code_hash(AI_GENERATED_CODE)
        hash2 = compute_code_hash(AI_GENERATED_CODE)
        
        assert hash1 == hash2
    
    def test_hash_whitespace_normalization(self):
        """Test that whitespace differences don't affect hash."""
        code1 = "def foo():\n    pass"
        code2 = "def foo():\n    pass  "  # trailing space
        
        hash1 = compute_code_hash(code1)
        hash2 = compute_code_hash(code2)
        
        assert hash1 == hash2
    
    def test_hash_different_code(self):
        """Test that different code produces different hash."""
        hash1 = compute_code_hash(AI_GENERATED_CODE)
        hash2 = compute_code_hash(HUMAN_WRITTEN_CODE)
        
        assert hash1 != hash2
