"""Tests for Trust Score Agent and related functionality."""

import pytest

from codeverify_agents.base import AgentResult, BaseAgent
from codeverify_agents.trust_score import (
    RecommendationGenerator,
    TrustScoreAgent,
    TrustScoreCalculator,
    TrustScoreFactors,
    TrustScoreResult,
)


class TestTrustScoreFactors:
    """Tests for TrustScoreFactors dataclass."""

    def test_default_values(self):
        """Factors initialize with the current six scoring inputs."""
        factors = TrustScoreFactors()

        assert factors.to_dict() == {
            "complexity_score": 0.0,
            "pattern_confidence": 0.0,
            "historical_accuracy": 0.0,
            "verification_coverage": 0.0,
            "code_quality_signals": 0.0,
            "ai_detection_confidence": 0.0,
        }

    def test_custom_values(self):
        """Factors accept and serialize current field names."""
        factors = TrustScoreFactors(
            complexity_score=0.8,
            pattern_confidence=0.7,
            historical_accuracy=0.9,
            verification_coverage=0.85,
            code_quality_signals=0.75,
            ai_detection_confidence=0.2,
        )

        assert factors.complexity_score == 0.8
        assert factors.pattern_confidence == 0.7
        assert factors.to_dict()["verification_coverage"] == 0.85


class TestTrustScoreResult:
    """Tests for TrustScoreResult dataclass."""

    def test_result_creation(self):
        """Result exposes current score, confidence, and AI fields."""
        factors = TrustScoreFactors(
            complexity_score=0.2,
            pattern_confidence=0.7,
            historical_accuracy=0.9,
            verification_coverage=0.85,
            code_quality_signals=0.75,
            ai_detection_confidence=0.15,
        )
        result = TrustScoreResult(
            score=82.5,
            confidence=0.9,
            risk_level="low",
            factors=factors,
            recommendations=["Add more tests"],
            is_ai_generated=False,
        )

        assert result.to_dict() == {
            "score": 82.5,
            "confidence": 0.9,
            "risk_level": "low",
            "factors": factors.to_dict(),
            "recommendations": ["Add more tests"],
            "is_ai_generated": False,
        }


class TestTrustScoreAgent:
    """Tests for TrustScoreAgent through its AgentResult interface."""

    @pytest.fixture
    def agent(self):
        return TrustScoreAgent()

    def test_agent_initialization(self, agent):
        """Agent follows the standard BaseAgent configuration contract."""
        assert isinstance(agent, BaseAgent)
        assert agent.config.provider == "openai"
        assert agent.historical_data == {}

    @pytest.mark.asyncio
    async def test_analyze_simple_code(self, agent):
        """Simple code produces a successful, structured AgentResult."""
        result = await agent.analyze(
            "def add(a, b):\n    return a + b\n",
            {},
        )

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.data["risk_level"] == "high"
        assert result.data["factors"]["pattern_confidence"] == 1.0
        assert result.data["factors"]["historical_accuracy"] == 0.5
        assert result.data["is_ai_generated"] is False

    @pytest.mark.asyncio
    async def test_analyze_complex_code(self, agent):
        """Control flow raises measured complexity relative to a simple function."""
        simple = await agent.analyze("def identity(value):\n    return value\n", {})
        complex_result = await agent.analyze(
            """
def process(data):
    results = []
    for item in data:
        try:
            if item.is_valid():
                results.append(item.transform())
        except ValueError:
            continue
    return results
""",
            {},
        )

        assert complex_result.success is True
        assert (
            complex_result.data["factors"]["complexity_score"]
            > simple.data["factors"]["complexity_score"]
        )

    @pytest.mark.asyncio
    async def test_detect_ai_patterns(self, agent):
        """Multiple documented AI patterns cross the detector threshold."""
        ai_like_code = """
# TODO: implement parser
# This function does the parsing
def parse(data):
    pass  # placeholder
raise NotImplementedError
# Example usage
"""

        result = await agent.analyze(ai_like_code, {})

        assert result.success is True
        assert result.data["is_ai_generated"] is True
        assert result.data["factors"]["ai_detection_confidence"] > 0.8
        assert (
            "Code appears AI-generated. Manual review recommended for business logic."
            in result.data["recommendations"]
        )

    @pytest.mark.asyncio
    async def test_detect_risky_patterns(self, agent):
        """Risk patterns lower both pattern confidence and the resulting score."""
        safe_result = await agent.analyze(
            "def echo(value):\n    return value\n",
            {},
        )
        risky_result = await agent.analyze(
            """
def execute(user_input):
    eval(user_input)
    exec(user_input)
    password = "secret"
""",
            {},
        )

        assert (
            risky_result.data["factors"]["pattern_confidence"]
            < safe_result.data["factors"]["pattern_confidence"]
        )
        assert risky_result.data["score"] < safe_result.data["score"]

    @pytest.mark.asyncio
    async def test_quality_indicators(self, agent):
        """Type hints, assertions, and logging increase quality signals."""
        plain_result = await agent.analyze("def process(items):\n    return items\n", {})
        quality_result = await agent.analyze(
            """
from typing import Optional
import logging

def process(items: list) -> Optional[list]:
    assert items
    logging.info("processing")
    return items
""",
            {},
        )

        assert (
            quality_result.data["factors"]["code_quality_signals"]
            > plain_result.data["factors"]["code_quality_signals"]
        )
        assert (
            "Missing quality signals. Add type hints, docstrings, and error handling."
            not in quality_result.data["recommendations"]
        )

    @pytest.mark.asyncio
    async def test_empty_code(self, agent):
        """Empty code has zero complexity and the neutral historical baseline."""
        result = await agent.analyze("", {})

        assert result.success is True
        assert result.data["score"] == 47.5
        assert result.data["factors"]["complexity_score"] == 0.0
        assert result.data["factors"]["historical_accuracy"] == 0.5

    def test_recommendations_generated(self):
        """Recommendation generation maps each weak factor to actionable guidance."""
        factors = TrustScoreFactors(
            complexity_score=0.8,
            pattern_confidence=0.4,
            historical_accuracy=0.3,
            verification_coverage=0.2,
            code_quality_signals=0.1,
        )

        recommendations = RecommendationGenerator().generate(
            factors,
            is_ai_generated=True,
        )

        assert recommendations == [
            "Code appears AI-generated. Manual review recommended for business logic.",
            "High complexity detected. Consider breaking into smaller functions.",
            "Potentially risky patterns detected. Review security implications.",
            "Low verification coverage. Add assertions or run formal verification.",
            "Missing quality signals. Add type hints, docstrings, and error handling.",
            "Historical accuracy is low for this pattern. Extra scrutiny recommended.",
        ]

    @pytest.mark.asyncio
    async def test_context_affects_analysis(self):
        """Documented history and verification context increase trust inputs."""
        agent = TrustScoreAgent(
            historical_data={"senior-dev": {"accuracy": 0.9}},
        )
        code = "def calculate(x):\n    return x + 1\n"
        context = {
            "author": "senior-dev",
            "verification_results": {
                "conditions_checked": 10,
                "conditions_passed": 10,
            },
        }

        with_context = await agent.analyze(code, context)
        without_context = await agent.analyze(code, {})

        assert with_context.data["factors"]["historical_accuracy"] == 0.9
        assert with_context.data["factors"]["verification_coverage"] == 1.0
        assert without_context.data["factors"]["historical_accuracy"] == 0.5
        assert without_context.data["factors"]["verification_coverage"] == 0.0
        assert with_context.data["score"] > without_context.data["score"]


class TestTrustScoreWeighting:
    """Tests for current trust-score weighting logic."""

    def test_weighted_score_calculation(self):
        """Calculator uses inverted complexity and the documented weights."""
        factors = TrustScoreFactors(
            complexity_score=0.2,
            pattern_confidence=0.8,
            historical_accuracy=0.6,
            verification_coverage=0.9,
            code_quality_signals=0.5,
        )
        calculator = TrustScoreCalculator()
        expected = ((1 - 0.2) * 0.15 + 0.8 * 0.20 + 0.6 * 0.25 + 0.9 * 0.25 + 0.5 * 0.15) * 100

        assert calculator.calculate(factors, False, 0.0) == pytest.approx(expected)
        assert calculator.calculate(factors, True, 0.8) == pytest.approx(expected * 0.85)
