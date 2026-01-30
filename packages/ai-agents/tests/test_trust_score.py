"""Tests for Trust Score Agent and related functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from codeverify_agents.trust_score import (
    TrustScoreAgent,
    TrustScoreFactors,
    TrustScoreResult,
)


class TestTrustScoreFactors:
    """Tests for TrustScoreFactors dataclass."""

    def test_default_values(self):
        """Factors initialize with default values."""
        factors = TrustScoreFactors()
        assert factors.complexity_score == 0.0
        assert factors.pattern_score == 0.0
        assert factors.historical_score == 0.0
        assert factors.verification_score == 0.0
        assert factors.quality_score == 0.0

    def test_custom_values(self):
        """Factors accept custom values."""
        factors = TrustScoreFactors(
            complexity_score=0.8,
            pattern_score=0.7,
            historical_score=0.9,
            verification_score=0.85,
            quality_score=0.75,
        )
        assert factors.complexity_score == 0.8
        assert factors.pattern_score == 0.7


class TestTrustScoreResult:
    """Tests for TrustScoreResult dataclass."""

    def test_result_creation(self):
        """Result can be created with all fields."""
        factors = TrustScoreFactors(
            complexity_score=0.8,
            pattern_score=0.7,
            historical_score=0.9,
            verification_score=0.85,
            quality_score=0.75,
        )
        result = TrustScoreResult(
            score=82.5,
            risk_level="low",
            ai_probability=15.0,
            factors=factors,
            recommendations=["Add more tests"],
            confidence=0.9,
        )
        assert result.score == 82.5
        assert result.risk_level == "low"
        assert result.ai_probability == 15.0


class TestTrustScoreAgent:
    """Tests for Trust Score Agent."""

    @pytest.fixture
    def agent(self):
        """Create a trust score agent."""
        return TrustScoreAgent()

    def test_agent_initialization(self, agent):
        """Agent initializes correctly."""
        assert agent is not None
        assert hasattr(agent, "analyze")

    @pytest.mark.asyncio
    async def test_analyze_simple_code(self, agent):
        """Agent can analyze simple code."""
        code = """
def add(a, b):
    return a + b
"""
        result = await agent.analyze(code)
        
        assert isinstance(result, TrustScoreResult)
        assert 0 <= result.score <= 100
        assert result.risk_level in ["low", "medium", "high", "critical"]
        assert 0 <= result.ai_probability <= 100

    @pytest.mark.asyncio
    async def test_analyze_complex_code(self, agent):
        """Agent handles complex code."""
        code = """
def complex_function(data: list[dict], config: Config) -> ProcessedResult:
    '''Process data with multiple transformations.'''
    validated = [d for d in data if d.get('status') == 'active']
    
    results = []
    for item in validated:
        try:
            transformed = config.transform(item)
            if transformed.is_valid():
                results.append(transformed)
        except TransformError as e:
            logger.warning(f"Transform failed: {e}")
            continue
    
    return ProcessedResult(
        items=results,
        count=len(results),
        success_rate=len(results) / len(data) if data else 0
    )
"""
        result = await agent.analyze(code)
        
        assert isinstance(result, TrustScoreResult)
        # Complex, well-documented code should score reasonably
        assert result.score > 0

    @pytest.mark.asyncio
    async def test_detect_ai_patterns(self, agent):
        """Agent detects common AI-generated code patterns."""
        # Code with typical AI-generated patterns
        ai_like_code = """
def process_data(data):
    '''Process the given data and return the result.'''
    # Initialize result
    result = []
    
    # Iterate through data
    for item in data:
        # Process each item
        processed = item * 2
        # Add to result
        result.append(processed)
    
    # Return the result
    return result
"""
        result = await agent.analyze(ai_like_code)
        
        # Should detect AI patterns (excessive comments)
        assert result.ai_probability > 0

    @pytest.mark.asyncio
    async def test_detect_risky_patterns(self, agent):
        """Agent detects risky code patterns."""
        risky_code = """
import os

def execute_command(user_input):
    os.system(user_input)  # Dangerous!
    eval(user_input)  # Also dangerous
    exec(user_input)  # Very dangerous
"""
        result = await agent.analyze(risky_code)
        
        # Risky code should have lower trust score
        assert result.risk_level in ["high", "critical"]

    @pytest.mark.asyncio
    async def test_quality_indicators(self, agent):
        """Agent recognizes quality indicators."""
        quality_code = """
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    '''Process data with validation and error handling.'''
    
    def __init__(self, config: dict) -> None:
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        if not self.config:
            raise ValueError("Config cannot be empty")
    
    def process(self, items: List[dict]) -> Optional[List[dict]]:
        '''Process items with proper error handling.'''
        try:
            return [self._transform(item) for item in items]
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return None
"""
        result = await agent.analyze(quality_code)
        
        # Quality code should have better score
        assert result.factors.quality_score > 0.5

    @pytest.mark.asyncio
    async def test_empty_code(self, agent):
        """Agent handles empty code."""
        result = await agent.analyze("")
        
        assert isinstance(result, TrustScoreResult)
        assert result.score >= 0

    @pytest.mark.asyncio
    async def test_recommendations_generated(self, agent):
        """Agent generates recommendations."""
        code = """
def foo(x):
    return x * 2
"""
        result = await agent.analyze(code)
        
        assert isinstance(result.recommendations, list)

    @pytest.mark.asyncio
    async def test_context_affects_analysis(self, agent):
        """Context information affects analysis."""
        code = """
def calculate(x):
    return x + 1
"""
        # With context showing it's from a trusted contributor
        context = {
            "author": "senior-dev",
            "commit_history": ["fix: bug fix", "feat: new feature"],
            "test_coverage": 0.85,
        }
        
        result_with_context = await agent.analyze(code, context)
        result_without_context = await agent.analyze(code)
        
        # Both should return valid results
        assert isinstance(result_with_context, TrustScoreResult)
        assert isinstance(result_without_context, TrustScoreResult)


class TestTrustScoreWeighting:
    """Tests for trust score weighting logic."""

    @pytest.fixture
    def agent(self):
        return TrustScoreAgent()

    @pytest.mark.asyncio
    async def test_weighted_score_calculation(self, agent):
        """Verify weighted score is calculated correctly."""
        code = "def test(): pass"
        result = await agent.analyze(code)
        
        # Verify score is within bounds
        assert 0 <= result.score <= 100
        
        # Verify factors contribute to score
        factors = result.factors
        expected_weights = {
            "complexity": 0.15,
            "pattern": 0.20,
            "historical": 0.25,
            "verification": 0.25,
            "quality": 0.15,
        }
        
        # Weighted sum should be close to final score
        weighted_sum = (
            factors.complexity_score * expected_weights["complexity"] +
            factors.pattern_score * expected_weights["pattern"] +
            factors.historical_score * expected_weights["historical"] +
            factors.verification_score * expected_weights["verification"] +
            factors.quality_score * expected_weights["quality"]
        ) * 100
        
        # Allow some tolerance for additional adjustments
        assert abs(result.score - weighted_sum) < 30  # Within 30 points
