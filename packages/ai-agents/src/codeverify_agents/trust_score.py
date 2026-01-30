"""Trust Score Agent - Scores AI-generated code reliability."""

import hashlib
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent, CodeContext

logger = structlog.get_logger()


@dataclass
class TrustScoreFactors:
    """Factors contributing to the trust score."""

    complexity_score: float = 0.0  # 0-1, lower is better
    pattern_confidence: float = 0.0  # 0-1, higher is better
    historical_accuracy: float = 0.0  # 0-1, higher is better
    verification_coverage: float = 0.0  # 0-1, higher is better
    code_quality_signals: float = 0.0  # 0-1, higher is better
    ai_detection_confidence: float = 0.0  # 0-1, confidence code is AI-generated

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "complexity_score": round(self.complexity_score, 3),
            "pattern_confidence": round(self.pattern_confidence, 3),
            "historical_accuracy": round(self.historical_accuracy, 3),
            "verification_coverage": round(self.verification_coverage, 3),
            "code_quality_signals": round(self.code_quality_signals, 3),
            "ai_detection_confidence": round(self.ai_detection_confidence, 3),
        }


@dataclass
class TrustScoreResult:
    """Result of trust score calculation."""

    score: float  # 0-100, higher means more trustworthy
    confidence: float  # 0-1, confidence in the score itself
    risk_level: str  # "low", "medium", "high", "critical"
    factors: TrustScoreFactors = field(default_factory=TrustScoreFactors)
    recommendations: list[str] = field(default_factory=list)
    is_ai_generated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": round(self.score, 1),
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level,
            "factors": self.factors.to_dict(),
            "recommendations": self.recommendations,
            "is_ai_generated": self.is_ai_generated,
        }


# ============================================================================
# Decomposed Components for SRP compliance
# ============================================================================


class AIDetector:
    """Detects whether code is AI-generated based on patterns."""

    # Patterns commonly found in AI-generated code
    AI_CODE_PATTERNS = [
        r"# TODO:?\s*(implement|add|fix|complete)",  # Generic TODOs
        r"# ?(This|The) (function|method|class) (does|will|should)",  # Overly explanatory comments
        r"pass\s*# ?(placeholder|implement)",  # Placeholder passes
        r"raise NotImplementedError",  # Unimplemented stubs
        r"\"\"\".*\.\.\.""",  # Docstrings with ...
        r"# Example usage",  # Example sections
        r"if __name__ == ['\"]__main__['\"]:\s*#",  # Main blocks with comments
    ]

    def detect(self, code: str) -> tuple[bool, float]:
        """Detect if code is likely AI-generated.
        
        Returns:
            Tuple of (is_ai_generated, confidence)
        """
        ai_pattern_matches = 0
        total_patterns = len(self.AI_CODE_PATTERNS)

        for pattern in self.AI_CODE_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                ai_pattern_matches += 1

        # Check for uniform comment style (AI tends to be consistent)
        comment_lines = re.findall(r"^\s*#\s*.+$", code, re.MULTILINE)
        if len(comment_lines) > 3:
            comment_lengths = [len(c.strip()) for c in comment_lines]
            if comment_lengths:
                avg_len = sum(comment_lengths) / len(comment_lengths)
                variance = sum((l - avg_len) ** 2 for l in comment_lengths) / len(comment_lengths)
                if variance < 100:  # Low variance = likely AI
                    ai_pattern_matches += 1
                    total_patterns += 1

        confidence = ai_pattern_matches / max(total_patterns, 1)

        # Boost confidence if multiple strong signals
        if ai_pattern_matches >= 3:
            confidence = min(confidence * 1.2, 1.0)

        is_ai_generated = confidence > 0.4
        return is_ai_generated, confidence


class PatternMatcher:
    """Matches code against risky and quality patterns."""

    # Patterns indicating potential issues
    RISKY_PATTERNS = [
        (r"eval\s*\(", "Unsafe eval usage", 0.9),
        (r"exec\s*\(", "Unsafe exec usage", 0.9),
        (r"pickle\.loads?\(", "Insecure deserialization", 0.85),
        (r"shell\s*=\s*True", "Shell injection risk", 0.8),
        (r"verify\s*=\s*False", "SSL verification disabled", 0.85),
        (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password", 0.95),
        (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key", 0.95),
        (r"except\s*:\s*pass", "Silent exception swallowing", 0.7),
        (r"# type:\s*ignore", "Type checking disabled", 0.5),
    ]

    # Quality indicators (positive patterns)
    QUALITY_PATTERNS = [
        (r"def test_", "Has tests", 0.2),
        (r"assert\s+", "Has assertions", 0.15),
        (r"logging\.(debug|info|warning|error)", "Uses logging", 0.1),
        (r"try:\s*\n.*\n\s*except\s+\w+", "Specific exception handling", 0.15),
        (r"typing\s+import|:\s*(int|str|float|bool|list|dict|Optional|Union)", "Type hints", 0.2),
        (r'"""[\s\S]*?Args:', "Documented parameters", 0.1),
        (r'"""[\s\S]*?Returns:', "Documented returns", 0.1),
    ]

    def calculate_risk_confidence(self, code: str) -> float:
        """Calculate confidence based on risky patterns (0-1, higher is better)."""
        total_risk = 0.0

        for pattern, _desc, risk_weight in self.RISKY_PATTERNS:
            matches = len(re.findall(pattern, code, re.IGNORECASE))
            if matches > 0:
                total_risk += risk_weight * min(matches, 3) / 3

        max_possible_risk = len(self.RISKY_PATTERNS)
        normalized_risk = min(total_risk / max_possible_risk, 1.0)
        return 1.0 - normalized_risk

    def calculate_quality_signals(self, code: str) -> float:
        """Calculate quality based on positive patterns."""
        total_quality = 0.0

        for pattern, _desc, quality_weight in self.QUALITY_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                total_quality += quality_weight

        return min(total_quality, 1.0)


class ComplexityAnalyzer:
    """Analyzes code complexity metrics."""

    def calculate(self, code: str) -> float:
        """Calculate code complexity score (0-1, lower is better)."""
        lines = code.split("\n")
        non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]

        if not non_empty_lines:
            return 0.0

        factors = []

        # Line length complexity
        long_lines = sum(1 for l in non_empty_lines if len(l) > 100)
        factors.append(min(long_lines / len(non_empty_lines), 1.0))

        # Nesting depth
        max_indent = 0
        for line in non_empty_lines:
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent)
        factors.append(min(max_indent / 32, 1.0))

        # Control flow complexity
        control_keywords = len(re.findall(
            r"\b(if|elif|else|for|while|try|except|with)\b",
            code,
        ))
        factors.append(min(control_keywords / (len(non_empty_lines) / 5), 1.0))

        # Function/method count relative to size
        functions = len(re.findall(r"^\s*(?:async\s+)?def\s+\w+", code, re.MULTILINE))
        classes = len(re.findall(r"^\s*class\s+\w+", code, re.MULTILINE))
        if functions + classes > 0:
            lines_per_unit = len(non_empty_lines) / (functions + classes)
            factors.append(min(lines_per_unit / 50, 1.0))
        else:
            factors.append(0.5)

        return sum(factors) / len(factors)


class VerificationCoverageCalculator:
    """Calculates verification coverage from verification results."""

    def calculate(self, verification_results: dict[str, Any]) -> float:
        """Calculate how much of the code is covered by verification."""
        if not verification_results:
            return 0.0

        conditions_checked = verification_results.get("conditions_checked", 0)
        conditions_passed = verification_results.get("conditions_passed", 0)

        if conditions_checked == 0:
            return 0.0

        pass_rate = conditions_passed / conditions_checked
        coverage = min(conditions_checked / 10, 1.0)
        return pass_rate * 0.7 + coverage * 0.3


class HistoricalAccuracyTracker:
    """Tracks historical accuracy for authors and file patterns."""

    def __init__(self, historical_data: dict[str, Any] | None = None) -> None:
        self.data = historical_data or {}

    def get_accuracy(self, context: dict[str, Any]) -> float:
        """Get historical accuracy for similar code patterns."""
        author = context.get("author")
        if author and author in self.data:
            author_data = self.data[author]
            if "accuracy" in author_data:
                return author_data["accuracy"]

        file_path = context.get("file_path", "")
        for pattern, data in self.data.get("file_patterns", {}).items():
            if re.search(pattern, file_path):
                return data.get("accuracy", 0.5)

        return 0.5  # Default neutral

    def update(
        self,
        code_hash: str,
        actual_outcome: dict[str, Any],
    ) -> None:
        """Update historical data based on actual outcomes."""
        author = actual_outcome.get("author")
        was_accurate = actual_outcome.get("was_accurate", True)

        if author:
            if author not in self.data:
                self.data[author] = {"total": 0, "accurate": 0, "accuracy": 0.5}

            self.data[author]["total"] += 1
            if was_accurate:
                self.data[author]["accurate"] += 1

            total = self.data[author]["total"]
            accurate = self.data[author]["accurate"]
            self.data[author]["accuracy"] = accurate / total

        logger.info(
            "Historical data updated",
            code_hash=code_hash[:16],
            author=author,
            was_accurate=was_accurate,
        )


class TrustScoreCalculator:
    """Calculates final trust score from component factors."""

    WEIGHTS = {
        "complexity": 0.15,
        "pattern_confidence": 0.20,
        "historical_accuracy": 0.25,
        "verification_coverage": 0.25,
        "code_quality": 0.15,
    }

    def calculate(
        self,
        factors: TrustScoreFactors,
        is_ai_generated: bool,
        ai_confidence: float,
    ) -> float:
        """Calculate final trust score (0-100)."""
        raw_score = (
            (1 - factors.complexity_score) * self.WEIGHTS["complexity"]
            + factors.pattern_confidence * self.WEIGHTS["pattern_confidence"]
            + factors.historical_accuracy * self.WEIGHTS["historical_accuracy"]
            + factors.verification_coverage * self.WEIGHTS["verification_coverage"]
            + factors.code_quality_signals * self.WEIGHTS["code_quality"]
        )

        score = raw_score * 100

        if is_ai_generated and ai_confidence > 0.7:
            score *= 0.85  # 15% penalty for high-confidence AI detection

        return score

    def determine_risk_level(self, score: float, factors: TrustScoreFactors) -> str:
        """Determine risk level based on score and factors."""
        if factors.pattern_confidence < 0.3:
            return "critical"

        if score >= 80:
            return "low"
        elif score >= 60:
            return "medium"
        elif score >= 40:
            return "high"
        else:
            return "critical"

    def calculate_confidence(
        self,
        factors: TrustScoreFactors,
        context: dict[str, Any],
        historical_data: dict[str, Any],
    ) -> float:
        """Calculate confidence in the trust score itself."""
        confidence_factors = []

        if context.get("verification_results"):
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)

        if context.get("author") in historical_data:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)

        if factors.pattern_confidence < 0.2 or factors.pattern_confidence > 0.8:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)

        return sum(confidence_factors) / len(confidence_factors)


class RecommendationGenerator:
    """Generates recommendations based on trust score factors."""

    def generate(
        self,
        factors: TrustScoreFactors,
        is_ai_generated: bool,
    ) -> list[str]:
        """Generate recommendations based on factors."""
        recommendations = []

        if is_ai_generated:
            recommendations.append(
                "Code appears AI-generated. Manual review recommended for business logic."
            )

        if factors.complexity_score > 0.7:
            recommendations.append(
                "High complexity detected. Consider breaking into smaller functions."
            )

        if factors.pattern_confidence < 0.5:
            recommendations.append(
                "Potentially risky patterns detected. Review security implications."
            )

        if factors.verification_coverage < 0.3:
            recommendations.append(
                "Low verification coverage. Add assertions or run formal verification."
            )

        if factors.code_quality_signals < 0.3:
            recommendations.append(
                "Missing quality signals. Add type hints, docstrings, and error handling."
            )

        if factors.historical_accuracy < 0.4:
            recommendations.append(
                "Historical accuracy is low for this pattern. Extra scrutiny recommended."
            )

        return recommendations


# ============================================================================
# Legacy patterns for backward compatibility
# ============================================================================

# Patterns commonly found in AI-generated code
AI_CODE_PATTERNS = AIDetector.AI_CODE_PATTERNS

# Patterns indicating potential issues in AI code
RISKY_AI_PATTERNS = PatternMatcher.RISKY_PATTERNS

# Quality indicators (positive patterns)
QUALITY_PATTERNS = PatternMatcher.QUALITY_PATTERNS


class TrustScoreAgent(BaseAgent):
    """
    Agent for calculating trust scores for code, especially AI-generated code.

    This agent uses decomposed components for SRP compliance:
    - AIDetector: Detects AI-generated code
    - PatternMatcher: Matches risky and quality patterns
    - ComplexityAnalyzer: Calculates complexity metrics
    - VerificationCoverageCalculator: Assesses verification coverage
    - HistoricalAccuracyTracker: Tracks historical accuracy
    - TrustScoreCalculator: Computes final score
    - RecommendationGenerator: Generates recommendations
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        historical_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize trust score agent with decomposed components."""
        super().__init__(config)
        
        # Inject dependencies - allows for testing and customization
        self._ai_detector = AIDetector()
        self._pattern_matcher = PatternMatcher()
        self._complexity_analyzer = ComplexityAnalyzer()
        self._verification_calculator = VerificationCoverageCalculator()
        self._historical_tracker = HistoricalAccuracyTracker(historical_data)
        self._score_calculator = TrustScoreCalculator()
        self._recommendation_generator = RecommendationGenerator()
        
        # Backward compatibility
        self.historical_data = self._historical_tracker.data

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Analyze code and return trust score.

        Args:
            code: The code to analyze
            context: Additional context including:
                - file_path: Path to the file
                - language: Programming language
                - verification_results: Results from formal verification
                - author: Code author (for historical lookup)

        Returns:
            AgentResult with trust score data
        """
        start_time = time.time()

        try:
            trust_result = await self.calculate_trust_score(code, context)
            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                "Trust score calculated",
                score=trust_result.score,
                risk_level=trust_result.risk_level,
                is_ai_generated=trust_result.is_ai_generated,
                latency_ms=elapsed_ms,
            )

            return AgentResult(
                success=True,
                data=trust_result.to_dict(),
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("Trust score calculation failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def calculate_trust_score(
        self,
        code: str,
        context: dict[str, Any],
    ) -> TrustScoreResult:
        """Calculate comprehensive trust score for code using decomposed components."""
        factors = TrustScoreFactors()

        # 1. Detect if code is AI-generated
        is_ai_generated, ai_confidence = self._ai_detector.detect(code)
        factors.ai_detection_confidence = ai_confidence

        # 2. Calculate complexity score
        factors.complexity_score = self._complexity_analyzer.calculate(code)

        # 3. Pattern-based confidence (checks for risky patterns)
        factors.pattern_confidence = self._pattern_matcher.calculate_risk_confidence(code)

        # 4. Historical accuracy
        factors.historical_accuracy = self._historical_tracker.get_accuracy(context)

        # 5. Verification coverage
        factors.verification_coverage = self._verification_calculator.calculate(
            context.get("verification_results", {})
        )

        # 6. Code quality signals
        factors.code_quality_signals = self._pattern_matcher.calculate_quality_signals(code)

        # Calculate final score
        score = self._score_calculator.calculate(factors, is_ai_generated, ai_confidence)

        # Determine risk level
        risk_level = self._score_calculator.determine_risk_level(score, factors)

        # Generate recommendations
        recommendations = self._recommendation_generator.generate(factors, is_ai_generated)

        # Calculate confidence in our score
        confidence = self._score_calculator.calculate_confidence(
            factors, context, self._historical_tracker.data
        )

        return TrustScoreResult(
            score=score,
            confidence=confidence,
            risk_level=risk_level,
            factors=factors,
            recommendations=recommendations,
            is_ai_generated=is_ai_generated,
        )

    async def update_historical_data(
        self,
        code_hash: str,
        actual_outcome: dict[str, Any],
    ) -> None:
        """Update historical data based on actual outcomes."""
        self._historical_tracker.update(code_hash, actual_outcome)

    # Backward compatibility methods - delegate to components
    def _detect_ai_generated(self, code: str) -> tuple[bool, float]:
        """Detect if code is likely AI-generated. Deprecated: use AIDetector."""
        return self._ai_detector.detect(code)

    def _calculate_complexity(self, code: str) -> float:
        """Calculate complexity. Deprecated: use ComplexityAnalyzer."""
        return self._complexity_analyzer.calculate(code)

    def _calculate_pattern_confidence(self, code: str) -> float:
        """Calculate pattern confidence. Deprecated: use PatternMatcher."""
        return self._pattern_matcher.calculate_risk_confidence(code)

    def _get_historical_accuracy(self, context: dict[str, Any]) -> float:
        """Get historical accuracy. Deprecated: use HistoricalAccuracyTracker."""
        return self._historical_tracker.get_accuracy(context)

    def _calculate_verification_coverage(self, context: dict[str, Any]) -> float:
        """Calculate verification coverage. Deprecated: use VerificationCoverageCalculator."""
        return self._verification_calculator.calculate(context.get("verification_results", {}))

    def _calculate_quality_signals(self, code: str) -> float:
        """Calculate quality signals. Deprecated: use PatternMatcher."""
        return self._pattern_matcher.calculate_quality_signals(code)

    def _determine_risk_level(self, score: float, factors: TrustScoreFactors) -> str:
        """Determine risk level. Deprecated: use TrustScoreCalculator."""
        return self._score_calculator.determine_risk_level(score, factors)

    def _generate_recommendations(
        self,
        factors: TrustScoreFactors,
        is_ai_generated: bool,
    ) -> list[str]:
        """Generate recommendations. Deprecated: use RecommendationGenerator."""
        return self._recommendation_generator.generate(factors, is_ai_generated)

    def _calculate_score_confidence(
        self,
        factors: TrustScoreFactors,
        context: dict[str, Any],
    ) -> float:
        """Calculate score confidence. Deprecated: use TrustScoreCalculator."""
        return self._score_calculator.calculate_confidence(
            factors, context, self._historical_tracker.data
        )


def calculate_code_hash(code: str) -> str:
    """Calculate hash of code for deduplication."""
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", code.strip())
    return hashlib.sha256(normalized.encode()).hexdigest()
