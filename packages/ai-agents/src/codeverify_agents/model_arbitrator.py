"""Competing Model Arbitration - Advanced multi-LLM consensus with voting strategies.

Extends multi_model_consensus.py with:
- Multiple voting strategies (Borda count, approval, ranked choice)
- Confidence-weighted arbitration
- Model debate for disagreements
- Specialization-aware routing
"""

import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent
from codeverify_agents.multi_model_consensus import (
    ConsensusStrategy,
    ModelConfig,
    ModelFinding,
    ModelProvider,
)

logger = structlog.get_logger()


class VotingMethod(str, Enum):
    """Advanced voting methods for arbitration."""
    BORDA_COUNT = "borda_count"  # Points based on ranking
    APPROVAL = "approval"  # Binary approve/reject
    RANKED_CHOICE = "ranked_choice"  # Instant runoff
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by confidence
    SPECIALIZATION = "specialization"  # Weight by model specialty


class ModelSpecialization(str, Enum):
    """Areas where specific models excel."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    CONCURRENCY = "concurrency"


@dataclass
class ModelProfile:
    """Profile of a model's capabilities."""
    provider: ModelProvider
    specializations: list[ModelSpecialization]
    base_weight: float = 1.0
    historical_accuracy: dict[str, float] = field(default_factory=dict)
    avg_confidence: float = 0.5
    false_positive_rate: float = 0.1


@dataclass
class ArbitrationVote:
    """A vote from a model on a finding."""
    model: ModelProvider
    finding_id: str
    vote: str  # "confirm", "reject", "uncertain"
    confidence: float
    reasoning: str
    rank: int | None = None  # For ranked voting


@dataclass
class DebateRound:
    """A round of model debate."""
    round_number: int
    model: ModelProvider
    argument: str
    stance: str  # "for", "against"
    confidence: float


@dataclass
class ArbitrationResult:
    """Result of arbitration process."""
    finding_id: str
    final_verdict: str  # "confirmed", "rejected", "uncertain"
    confidence: float
    votes: list[ArbitrationVote]
    debate_rounds: list[DebateRound]
    voting_method: VotingMethod
    vote_breakdown: dict[str, int]
    reasoning: str


@dataclass
class ArbitratedFinding:
    """A finding that has been arbitrated."""
    original_finding: ModelFinding
    arbitration: ArbitrationResult
    should_report: bool
    adjusted_severity: str
    adjusted_confidence: float


MODEL_PROFILES: dict[ModelProvider, ModelProfile] = {
    ModelProvider.OPENAI_GPT5: ModelProfile(
        provider=ModelProvider.OPENAI_GPT5,
        specializations=[
            ModelSpecialization.CORRECTNESS,
            ModelSpecialization.SECURITY,
        ],
        base_weight=1.0,
        historical_accuracy={"security": 0.92, "correctness": 0.90, "performance": 0.85},
    ),
    ModelProvider.ANTHROPIC_CLAUDE: ModelProfile(
        provider=ModelProvider.ANTHROPIC_CLAUDE,
        specializations=[
            ModelSpecialization.SECURITY,
            ModelSpecialization.DOCUMENTATION,
        ],
        base_weight=1.0,
        historical_accuracy={"security": 0.94, "documentation": 0.95, "correctness": 0.88},
    ),
    ModelProvider.OPENAI_GPT4: ModelProfile(
        provider=ModelProvider.OPENAI_GPT4,
        specializations=[
            ModelSpecialization.STYLE,
            ModelSpecialization.PERFORMANCE,
        ],
        base_weight=0.9,
        historical_accuracy={"style": 0.90, "performance": 0.88, "security": 0.85},
    ),
    ModelProvider.GOOGLE_GEMINI: ModelProfile(
        provider=ModelProvider.GOOGLE_GEMINI,
        specializations=[
            ModelSpecialization.CONCURRENCY,
            ModelSpecialization.PERFORMANCE,
        ],
        base_weight=0.85,
        historical_accuracy={"concurrency": 0.87, "performance": 0.86, "correctness": 0.82},
    ),
}


VOTE_PROMPT = """You are reviewing a code finding from another AI model.

Original Finding:
- ID: {finding_id}
- Severity: {severity}
- Category: {category}
- Title: {title}
- Description: {description}
- Location: Line {line}

Code Context:
```{language}
{code_snippet}
```

Your task: Vote on whether this finding is valid.

Respond in JSON:
{{
  "vote": "confirm" | "reject" | "uncertain",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}"""


DEBATE_PROMPT = """You are participating in a debate about a code finding.

Finding:
- Title: {title}
- Description: {description}

Your stance: {stance}
Previous arguments:
{previous_arguments}

Provide your argument (2-3 sentences) supporting your stance.

Respond in JSON:
{{
  "argument": "Your argument",
  "confidence": 0.0-1.0
}}"""


class VotingEngine:
    """Implements various voting methods."""

    def borda_count(self, votes: list[ArbitrationVote]) -> dict[str, float]:
        """Calculate Borda count scores."""
        scores: dict[str, float] = {"confirm": 0, "reject": 0, "uncertain": 0}
        
        for vote in votes:
            # Points: confirm=3, uncertain=1, reject=0
            if vote.vote == "confirm":
                scores["confirm"] += 3 * vote.confidence
            elif vote.vote == "uncertain":
                scores["uncertain"] += 1 * vote.confidence
            # reject gets 0 points
        
        return scores

    def approval_voting(self, votes: list[ArbitrationVote]) -> dict[str, int]:
        """Simple approval voting."""
        counts: Counter[str] = Counter()
        for vote in votes:
            counts[vote.vote] += 1
        return dict(counts)

    def ranked_choice(
        self,
        votes: list[ArbitrationVote],
        threshold: float = 0.5,
    ) -> str:
        """Instant runoff ranked choice voting."""
        # Simplified: count first-choice votes
        first_choice: Counter[str] = Counter()
        
        for vote in votes:
            first_choice[vote.vote] += 1
        
        total = sum(first_choice.values())
        if total == 0:
            return "uncertain"
        
        # Check for majority
        for choice, count in first_choice.most_common():
            if count / total >= threshold:
                return choice
        
        # No majority - return plurality winner
        return first_choice.most_common(1)[0][0] if first_choice else "uncertain"

    def confidence_weighted(
        self,
        votes: list[ArbitrationVote],
    ) -> tuple[str, float]:
        """Weight votes by confidence."""
        weighted_scores: dict[str, float] = {"confirm": 0, "reject": 0, "uncertain": 0}
        total_weight = 0
        
        for vote in votes:
            weighted_scores[vote.vote] += vote.confidence
            total_weight += vote.confidence
        
        if total_weight == 0:
            return "uncertain", 0.0
        
        # Normalize
        for key in weighted_scores:
            weighted_scores[key] /= total_weight
        
        winner = max(weighted_scores, key=lambda k: weighted_scores[k])
        return winner, weighted_scores[winner]

    def specialization_weighted(
        self,
        votes: list[ArbitrationVote],
        category: str,
        profiles: dict[ModelProvider, ModelProfile],
    ) -> tuple[str, float]:
        """Weight votes by model specialization in the relevant category."""
        weighted_scores: dict[str, float] = {"confirm": 0, "reject": 0, "uncertain": 0}
        total_weight = 0
        
        category_lower = category.lower()
        
        for vote in votes:
            profile = profiles.get(vote.model)
            if not profile:
                weight = 1.0
            else:
                # Check if model has relevant specialization
                spec_bonus = 0.3 if any(
                    s.value in category_lower for s in profile.specializations
                ) else 0
                
                # Use historical accuracy if available
                historical = profile.historical_accuracy.get(category_lower, 0.5)
                
                weight = profile.base_weight * (1 + spec_bonus) * historical
            
            weighted_scores[vote.vote] += vote.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return "uncertain", 0.0
        
        # Normalize
        for key in weighted_scores:
            weighted_scores[key] /= total_weight
        
        winner = max(weighted_scores, key=lambda k: weighted_scores[k])
        return winner, weighted_scores[winner]


class DebateEngine:
    """Facilitates model debates on disputed findings."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialize debate engine."""
        self.config = config
        self._max_rounds = 3

    async def run_debate(
        self,
        finding: ModelFinding,
        code_snippet: str,
        models_for: list[ModelConfig],
        models_against: list[ModelConfig],
    ) -> list[DebateRound]:
        """Run a debate between models."""
        rounds = []
        previous_args: list[str] = []
        
        for round_num in range(self._max_rounds):
            # Pro side
            if round_num < len(models_for):
                pro_model = models_for[round_num % len(models_for)]
                pro_round = await self._get_argument(
                    model=pro_model,
                    finding=finding,
                    stance="for",
                    previous=previous_args,
                    round_num=round_num,
                )
                rounds.append(pro_round)
                previous_args.append(f"FOR: {pro_round.argument}")
            
            # Con side
            if round_num < len(models_against):
                con_model = models_against[round_num % len(models_against)]
                con_round = await self._get_argument(
                    model=con_model,
                    finding=finding,
                    stance="against",
                    previous=previous_args,
                    round_num=round_num,
                )
                rounds.append(con_round)
                previous_args.append(f"AGAINST: {con_round.argument}")
        
        return rounds

    async def _get_argument(
        self,
        model: ModelConfig,
        finding: ModelFinding,
        stance: str,
        previous: list[str],
        round_num: int,
    ) -> DebateRound:
        """Get an argument from a model."""
        # Placeholder - would call actual LLM
        return DebateRound(
            round_number=round_num,
            model=model.provider,
            argument=f"[{stance.upper()}] Based on the evidence, this finding {'appears valid' if stance == 'for' else 'may be a false positive'}.",
            stance=stance,
            confidence=0.7,
        )


class CompetingModelArbitrator(BaseAgent):
    """
    Advanced multi-LLM arbitration with voting strategies and debate.
    
    Extends multi_model_consensus with:
    - Multiple voting methods
    - Specialization-aware routing
    - Model debate for disagreements
    - Confidence calibration
    """

    def __init__(
        self,
        models: list[ModelConfig] | None = None,
        voting_method: VotingMethod = VotingMethod.CONFIDENCE_WEIGHTED,
        config: AgentConfig | None = None,
        enable_debate: bool = True,
    ) -> None:
        """Initialize arbitrator."""
        super().__init__(config)
        
        self.models = models or [
            ModelConfig(
                provider=ModelProvider.OPENAI_GPT5,
                model_name="gpt-5-turbo",
                weight=1.0,
            ),
            ModelConfig(
                provider=ModelProvider.ANTHROPIC_CLAUDE,
                model_name="claude-3-sonnet-20240229",
                weight=1.0,
            ),
        ]
        
        self.voting_method = voting_method
        self.enable_debate = enable_debate
        self.voting_engine = VotingEngine()
        self.debate_engine = DebateEngine(self.config) if enable_debate else None
        self.model_profiles = MODEL_PROFILES.copy()
        
        # Thresholds
        self.confirm_threshold = 0.6  # Min confidence to confirm
        self.debate_threshold = 0.4  # Max confidence diff to trigger debate
        self.report_threshold = 0.5  # Min final confidence to report

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Analyze code with multi-model arbitration.
        
        Args:
            code: Code to analyze
            context: Additional context
            
        Returns:
            AgentResult with arbitrated findings
        """
        import time
        start_time = time.time()
        
        file_path = context.get("file_path", "unknown")
        language = context.get("language", "python")
        
        try:
            # Step 1: Get initial findings from all models
            all_findings = await self._collect_findings(code, context)
            
            # Step 2: Arbitrate each finding
            arbitrated = []
            for finding in all_findings:
                result = await self._arbitrate_finding(
                    finding=finding,
                    code=code,
                    language=language,
                )
                arbitrated.append(result)
            
            # Step 3: Filter and adjust based on arbitration
            final_findings = [
                af for af in arbitrated
                if af.should_report
            ]
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "Arbitration completed",
                total_findings=len(all_findings),
                confirmed=len(final_findings),
                voting_method=self.voting_method.value,
                latency_ms=elapsed_ms,
            )
            
            return AgentResult(
                success=True,
                data=self._format_result(final_findings, arbitrated),
                latency_ms=elapsed_ms,
            )
            
        except Exception as e:
            logger.error("Arbitration failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _collect_findings(
        self,
        code: str,
        context: dict[str, Any],
    ) -> list[ModelFinding]:
        """Collect findings from all models."""
        # Placeholder - would call actual models
        # In practice, integrates with MultiModelConsensus
        return []

    async def _arbitrate_finding(
        self,
        finding: ModelFinding,
        code: str,
        language: str,
    ) -> ArbitratedFinding:
        """Arbitrate a single finding."""
        # Step 1: Collect votes from all models
        votes = await self._collect_votes(finding, code, language)
        
        # Step 2: Apply voting method
        verdict, confidence, breakdown = self._apply_voting(
            votes=votes,
            category=finding.category,
        )
        
        # Step 3: Check if debate is needed
        debate_rounds = []
        if self.enable_debate and self._should_debate(votes, confidence):
            debate_rounds = await self._run_debate(finding, code, votes)
            # Re-evaluate after debate
            verdict, confidence = self._evaluate_after_debate(
                votes, debate_rounds, verdict, confidence
            )
        
        # Step 4: Create arbitration result
        arbitration = ArbitrationResult(
            finding_id=finding.finding_id,
            final_verdict=verdict,
            confidence=confidence,
            votes=votes,
            debate_rounds=debate_rounds,
            voting_method=self.voting_method,
            vote_breakdown=breakdown,
            reasoning=self._generate_reasoning(votes, debate_rounds, verdict),
        )
        
        # Step 5: Determine if finding should be reported
        should_report = (
            verdict == "confirm" and
            confidence >= self.report_threshold
        )
        
        # Adjust severity based on arbitration
        adjusted_severity = self._adjust_severity(
            finding.severity, confidence, votes
        )
        
        return ArbitratedFinding(
            original_finding=finding,
            arbitration=arbitration,
            should_report=should_report,
            adjusted_severity=adjusted_severity,
            adjusted_confidence=confidence,
        )

    async def _collect_votes(
        self,
        finding: ModelFinding,
        code: str,
        language: str,
    ) -> list[ArbitrationVote]:
        """Collect votes from all models."""
        votes = []
        
        # Exclude the original model from voting
        voting_models = [m for m in self.models if m.provider != finding.model]
        
        for model in voting_models:
            vote = await self._get_model_vote(model, finding, code, language)
            votes.append(vote)
        
        # Original model automatically confirms
        votes.append(ArbitrationVote(
            model=finding.model,
            finding_id=finding.finding_id,
            vote="confirm",
            confidence=finding.confidence,
            reasoning="Original finding from this model",
        ))
        
        return votes

    async def _get_model_vote(
        self,
        model: ModelConfig,
        finding: ModelFinding,
        code: str,
        language: str,
    ) -> ArbitrationVote:
        """Get a vote from a specific model."""
        # Placeholder - would call actual LLM
        # For now, simulate based on model profile
        profile = self.model_profiles.get(model.provider)
        
        # Simplified logic: models agree more often on their specializations
        base_confidence = 0.6
        if profile and any(
            s.value in finding.category.lower()
            for s in profile.specializations
        ):
            base_confidence = 0.8
        
        # Simulate vote
        import random
        roll = random.random()
        
        if roll < 0.7:  # 70% confirm
            vote = "confirm"
            confidence = base_confidence + random.uniform(0, 0.2)
        elif roll < 0.85:  # 15% uncertain
            vote = "uncertain"
            confidence = 0.5
        else:  # 15% reject
            vote = "reject"
            confidence = base_confidence + random.uniform(0, 0.2)
        
        return ArbitrationVote(
            model=model.provider,
            finding_id=finding.finding_id,
            vote=vote,
            confidence=min(confidence, 1.0),
            reasoning="Automated vote based on model analysis",
        )

    def _apply_voting(
        self,
        votes: list[ArbitrationVote],
        category: str,
    ) -> tuple[str, float, dict[str, int]]:
        """Apply the configured voting method."""
        breakdown = self.voting_engine.approval_voting(votes)
        
        if self.voting_method == VotingMethod.BORDA_COUNT:
            scores = self.voting_engine.borda_count(votes)
            winner = max(scores, key=lambda k: scores[k])
            total = sum(scores.values())
            confidence = scores[winner] / total if total > 0 else 0.5
            
        elif self.voting_method == VotingMethod.RANKED_CHOICE:
            winner = self.voting_engine.ranked_choice(votes)
            confidence = sum(
                v.confidence for v in votes if v.vote == winner
            ) / max(len(votes), 1)
            
        elif self.voting_method == VotingMethod.SPECIALIZATION:
            winner, confidence = self.voting_engine.specialization_weighted(
                votes, category, self.model_profiles
            )
            
        else:  # CONFIDENCE_WEIGHTED (default)
            winner, confidence = self.voting_engine.confidence_weighted(votes)
        
        return winner, confidence, breakdown

    def _should_debate(
        self,
        votes: list[ArbitrationVote],
        confidence: float,
    ) -> bool:
        """Determine if finding needs debate."""
        if not self.debate_engine:
            return False
        
        # Debate if confidence is low or votes are split
        if confidence < self.debate_threshold:
            return True
        
        confirm_count = sum(1 for v in votes if v.vote == "confirm")
        reject_count = sum(1 for v in votes if v.vote == "reject")
        
        # Significant disagreement
        if confirm_count > 0 and reject_count > 0:
            ratio = min(confirm_count, reject_count) / max(confirm_count, reject_count)
            if ratio > 0.5:  # Close split
                return True
        
        return False

    async def _run_debate(
        self,
        finding: ModelFinding,
        code: str,
        votes: list[ArbitrationVote],
    ) -> list[DebateRound]:
        """Run a debate on a disputed finding."""
        if not self.debate_engine:
            return []
        
        # Divide models into camps
        models_for = [
            m for m in self.models
            if any(v.model == m.provider and v.vote == "confirm" for v in votes)
        ]
        models_against = [
            m for m in self.models
            if any(v.model == m.provider and v.vote == "reject" for v in votes)
        ]
        
        if not models_for or not models_against:
            return []
        
        return await self.debate_engine.run_debate(
            finding=finding,
            code_snippet=code[:500],  # Limit snippet
            models_for=models_for,
            models_against=models_against,
        )

    def _evaluate_after_debate(
        self,
        votes: list[ArbitrationVote],
        debate_rounds: list[DebateRound],
        current_verdict: str,
        current_confidence: float,
    ) -> tuple[str, float]:
        """Re-evaluate verdict after debate."""
        if not debate_rounds:
            return current_verdict, current_confidence
        
        # Analyze debate outcomes
        for_confidence = [
            r.confidence for r in debate_rounds if r.stance == "for"
        ]
        against_confidence = [
            r.confidence for r in debate_rounds if r.stance == "against"
        ]
        
        avg_for = sum(for_confidence) / len(for_confidence) if for_confidence else 0
        avg_against = sum(against_confidence) / len(against_confidence) if against_confidence else 0
        
        # Adjust based on debate
        if avg_for > avg_against + 0.2:
            return "confirm", min(current_confidence + 0.1, 0.95)
        elif avg_against > avg_for + 0.2:
            return "reject", min(current_confidence + 0.1, 0.95)
        
        return current_verdict, current_confidence

    def _generate_reasoning(
        self,
        votes: list[ArbitrationVote],
        debate_rounds: list[DebateRound],
        verdict: str,
    ) -> str:
        """Generate reasoning for the arbitration result."""
        confirm_count = sum(1 for v in votes if v.vote == "confirm")
        reject_count = sum(1 for v in votes if v.vote == "reject")
        uncertain_count = sum(1 for v in votes if v.vote == "uncertain")
        
        parts = [
            f"Verdict: {verdict}.",
            f"Votes: {confirm_count} confirm, {reject_count} reject, {uncertain_count} uncertain.",
        ]
        
        if debate_rounds:
            parts.append(f"After {len(debate_rounds)} debate rounds.")
        
        return " ".join(parts)

    def _adjust_severity(
        self,
        original_severity: str,
        confidence: float,
        votes: list[ArbitrationVote],
    ) -> str:
        """Adjust severity based on arbitration confidence."""
        severity_order = ["low", "medium", "high", "critical"]
        
        try:
            idx = severity_order.index(original_severity.lower())
        except ValueError:
            return original_severity
        
        # Lower severity if confidence is low
        if confidence < 0.5 and idx > 0:
            return severity_order[idx - 1]
        
        # Raise severity if unanimous high-confidence confirmation
        if confidence > 0.9 and all(v.vote == "confirm" for v in votes):
            if idx < len(severity_order) - 1:
                return severity_order[idx + 1]
        
        return original_severity

    def _format_result(
        self,
        confirmed: list[ArbitratedFinding],
        all_arbitrated: list[ArbitratedFinding],
    ) -> dict[str, Any]:
        """Format the arbitration result."""
        return {
            "voting_method": self.voting_method.value,
            "debate_enabled": self.enable_debate,
            "summary": {
                "total_findings": len(all_arbitrated),
                "confirmed": len(confirmed),
                "rejected": len([a for a in all_arbitrated if a.arbitration.final_verdict == "reject"]),
                "uncertain": len([a for a in all_arbitrated if a.arbitration.final_verdict == "uncertain"]),
            },
            "findings": [
                {
                    "id": af.original_finding.finding_id,
                    "original_severity": af.original_finding.severity,
                    "adjusted_severity": af.adjusted_severity,
                    "category": af.original_finding.category,
                    "title": af.original_finding.title,
                    "verdict": af.arbitration.final_verdict,
                    "confidence": round(af.adjusted_confidence, 3),
                    "vote_breakdown": af.arbitration.vote_breakdown,
                    "reasoning": af.arbitration.reasoning,
                    "debated": len(af.arbitration.debate_rounds) > 0,
                }
                for af in confirmed
            ],
            "rejected_findings": [
                {
                    "id": af.original_finding.finding_id,
                    "title": af.original_finding.title,
                    "reason": af.arbitration.reasoning,
                }
                for af in all_arbitrated
                if not af.should_report
            ],
        }

    def set_voting_method(self, method: VotingMethod) -> None:
        """Change the voting method."""
        self.voting_method = method
        logger.info("Voting method changed", method=method.value)

    def update_model_profile(
        self,
        provider: ModelProvider,
        profile: ModelProfile,
    ) -> None:
        """Update a model's profile based on historical performance."""
        self.model_profiles[provider] = profile
        logger.info(
            "Model profile updated",
            provider=provider.value,
            specializations=[s.value for s in profile.specializations],
        )

    def calibrate_thresholds(
        self,
        confirm_threshold: float | None = None,
        debate_threshold: float | None = None,
        report_threshold: float | None = None,
    ) -> None:
        """Calibrate arbitration thresholds."""
        if confirm_threshold is not None:
            self.confirm_threshold = confirm_threshold
        if debate_threshold is not None:
            self.debate_threshold = debate_threshold
        if report_threshold is not None:
            self.report_threshold = report_threshold
        
        logger.info(
            "Thresholds calibrated",
            confirm=self.confirm_threshold,
            debate=self.debate_threshold,
            report=self.report_threshold,
        )
