"""Multi-Model Consensus Verification - Require model agreement for high-confidence findings."""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class ConsensusStrategy(str, Enum):
    """Strategy for reaching consensus."""
    UNANIMOUS = "unanimous"  # All models must agree
    MAJORITY = "majority"  # >50% must agree
    WEIGHTED = "weighted"  # Weighted by model confidence
    ANY = "any"  # Any model finding is reported


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI_GPT5 = "openai_gpt5"
    OPENAI_GPT4 = "openai_gpt4"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    GOOGLE_GEMINI = "google_gemini"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ModelProvider
    model_name: str
    weight: float = 1.0  # Weight for weighted consensus
    timeout_seconds: int = 60
    temperature: float = 0.1


@dataclass
class ModelFinding:
    """A finding from a single model."""
    model: ModelProvider
    finding_id: str
    severity: str
    category: str
    title: str
    description: str
    location: dict[str, Any]
    confidence: float
    suggested_fix: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusFinding:
    """A finding that has achieved consensus across models."""
    finding_id: str
    severity: str
    category: str
    title: str
    description: str
    location: dict[str, Any]
    consensus_confidence: float  # Combined confidence from all agreeing models
    agreeing_models: list[ModelProvider] = field(default_factory=list)
    dissenting_models: list[ModelProvider] = field(default_factory=list)
    model_findings: list[ModelFinding] = field(default_factory=list)
    suggested_fix: str | None = None
    consensus_type: ConsensusStrategy = ConsensusStrategy.MAJORITY


@dataclass
class ConsensusResult:
    """Result of consensus verification."""
    code_hash: str
    consensus_findings: list[ConsensusFinding] = field(default_factory=list)
    model_only_findings: dict[str, list[ModelFinding]] = field(default_factory=dict)
    overall_confidence: float = 0.0
    models_queried: list[ModelProvider] = field(default_factory=list)
    consensus_strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY
    total_latency_ms: float = 0.0
    tokens_used: dict[str, int] = field(default_factory=dict)


CONSENSUS_SYSTEM_PROMPT = """You are an expert code reviewer. Analyze the provided code for issues.

For each issue found, provide:
1. Severity: critical, high, medium, low
2. Category: security, logic, performance, correctness, maintainability
3. Title: Brief description
4. Description: Detailed explanation
5. Location: File, line, and column
6. Confidence: 0.0 to 1.0
7. Suggested fix (if applicable)

Be precise and avoid false positives. Only report issues you are confident about.

Respond in JSON:
{
  "findings": [
    {
      "id": "unique_id",
      "severity": "high",
      "category": "security",
      "title": "SQL Injection",
      "description": "User input directly in SQL query",
      "location": {"line": 42, "column": 10},
      "confidence": 0.95,
      "suggested_fix": "Use parameterized query"
    }
  ]
}"""


class MultiModelConsensus(BaseAgent):
    """
    Agent that runs verification across multiple LLM models and
    requires consensus for findings to reduce false positives.
    """

    def __init__(
        self,
        models: list[ModelConfig] | None = None,
        consensus_strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize multi-model consensus verifier."""
        super().__init__(config)
        
        # Default model configuration
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
            ModelConfig(
                provider=ModelProvider.OPENAI_GPT4,
                model_name="gpt-4-turbo-preview",
                weight=0.8,
            ),
        ]
        
        self.consensus_strategy = consensus_strategy
        self._similarity_threshold = 0.7  # For matching findings across models

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Analyze code using multiple models and return consensus findings.

        Args:
            code: The code to analyze
            context: Additional context including:
                - file_path: Path to the file
                - language: Programming language
                - consensus_strategy: Override default strategy
                - required_models: List of models that must participate

        Returns:
            AgentResult with consensus findings
        """
        start_time = time.time()
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        
        strategy = context.get("consensus_strategy", self.consensus_strategy)
        if isinstance(strategy, str):
            strategy = ConsensusStrategy(strategy)
        
        try:
            # Run all models in parallel
            model_results = await self._query_all_models(code, context)
            
            # Build consensus from model findings
            consensus = self._build_consensus(
                model_results=model_results,
                strategy=strategy,
                code_hash=code_hash,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            consensus.total_latency_ms = elapsed_ms
            
            logger.info(
                "Consensus verification completed",
                code_hash=code_hash,
                models_queried=len(consensus.models_queried),
                consensus_findings=len(consensus.consensus_findings),
                strategy=strategy.value,
                latency_ms=elapsed_ms,
            )
            
            return AgentResult(
                success=True,
                data=self._consensus_to_dict(consensus),
                tokens_used=sum(consensus.tokens_used.values()),
                latency_ms=elapsed_ms,
            )
            
        except Exception as e:
            logger.error("Consensus verification failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _query_all_models(
        self, code: str, context: dict[str, Any]
    ) -> dict[ModelProvider, list[ModelFinding]]:
        """Query all configured models in parallel."""
        tasks = []
        
        for model_config in self.models:
            task = self._query_model(model_config, code, context)
            tasks.append((model_config.provider, task))
        
        results = {}
        for provider, task in tasks:
            try:
                findings = await task
                results[provider] = findings
            except Exception as e:
                logger.warning(
                    "Model query failed",
                    provider=provider.value,
                    error=str(e),
                )
                results[provider] = []
        
        return results

    async def _query_model(
        self,
        model_config: ModelConfig,
        code: str,
        context: dict[str, Any],
    ) -> list[ModelFinding]:
        """Query a single model for findings."""
        file_path = context.get("file_path", "unknown")
        language = context.get("language", "python")
        
        user_prompt = f"""Analyze this {language} code from `{file_path}`:

```{language}
{code}
```

Identify all issues including bugs, security vulnerabilities, and code quality problems."""

        # Select appropriate API based on provider
        if model_config.provider == ModelProvider.ANTHROPIC_CLAUDE:
            response = await self._call_anthropic(
                system_prompt=CONSENSUS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        else:
            # OpenAI-compatible models
            original_model = self.config.openai_model
            self.config.openai_model = model_config.model_name
            
            try:
                response = await self._call_openai(
                    system_prompt=CONSENSUS_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    json_mode=True,
                )
            finally:
                self.config.openai_model = original_model
        
        # Parse findings
        findings = []
        try:
            data = json.loads(response["content"])
            for finding_data in data.get("findings", []):
                findings.append(ModelFinding(
                    model=model_config.provider,
                    finding_id=finding_data.get("id", "unknown"),
                    severity=finding_data.get("severity", "medium"),
                    category=finding_data.get("category", "unknown"),
                    title=finding_data.get("title", "Unknown Issue"),
                    description=finding_data.get("description", ""),
                    location=finding_data.get("location", {}),
                    confidence=float(finding_data.get("confidence", 0.5)),
                    suggested_fix=finding_data.get("suggested_fix"),
                    raw_response=finding_data,
                ))
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse model response",
                provider=model_config.provider.value,
            )
        
        return findings

    def _build_consensus(
        self,
        model_results: dict[ModelProvider, list[ModelFinding]],
        strategy: ConsensusStrategy,
        code_hash: str,
    ) -> ConsensusResult:
        """Build consensus from model findings."""
        # Group similar findings
        finding_groups = self._group_similar_findings(model_results)
        
        consensus_findings = []
        model_only_findings: dict[str, list[ModelFinding]] = {}
        
        total_models = len([m for m, findings in model_results.items() if findings])
        
        for group_key, findings in finding_groups.items():
            agreeing_models = list(set(f.model for f in findings))
            dissenting_models = [
                m for m in model_results.keys()
                if m not in agreeing_models
            ]
            
            # Check if consensus is reached based on strategy
            has_consensus = self._check_consensus(
                agreeing_count=len(agreeing_models),
                total_count=total_models,
                strategy=strategy,
                findings=findings,
            )
            
            if has_consensus:
                # Merge findings into consensus
                consensus_finding = self._merge_findings(
                    findings=findings,
                    agreeing_models=agreeing_models,
                    dissenting_models=dissenting_models,
                    strategy=strategy,
                )
                consensus_findings.append(consensus_finding)
            else:
                # Track as model-only finding
                for finding in findings:
                    model_key = finding.model.value
                    if model_key not in model_only_findings:
                        model_only_findings[model_key] = []
                    model_only_findings[model_key].append(finding)
        
        # Calculate overall confidence
        overall_confidence = 0.0
        if consensus_findings:
            overall_confidence = sum(
                f.consensus_confidence for f in consensus_findings
            ) / len(consensus_findings)
        
        # Track tokens used
        tokens_used = {}
        for model_config in self.models:
            tokens_used[model_config.provider.value] = 0  # Placeholder
        
        return ConsensusResult(
            code_hash=code_hash,
            consensus_findings=consensus_findings,
            model_only_findings=model_only_findings,
            overall_confidence=overall_confidence,
            models_queried=list(model_results.keys()),
            consensus_strategy=strategy,
            tokens_used=tokens_used,
        )

    def _group_similar_findings(
        self, model_results: dict[ModelProvider, list[ModelFinding]]
    ) -> dict[str, list[ModelFinding]]:
        """Group findings that describe the same issue."""
        groups: dict[str, list[ModelFinding]] = {}
        
        all_findings = [
            finding
            for findings in model_results.values()
            for finding in findings
        ]
        
        for finding in all_findings:
            # Generate a key based on location and category
            location_key = self._get_location_key(finding)
            
            # Check if this finding is similar to an existing group
            matched_group = None
            for group_key, group_findings in groups.items():
                if self._findings_similar(finding, group_findings[0]):
                    matched_group = group_key
                    break
            
            if matched_group:
                groups[matched_group].append(finding)
            else:
                groups[location_key] = [finding]
        
        return groups

    def _get_location_key(self, finding: ModelFinding) -> str:
        """Generate a key from finding location."""
        line = finding.location.get("line", 0)
        col = finding.location.get("column", 0)
        return f"{finding.category}:{line}:{col}"

    def _findings_similar(
        self, finding1: ModelFinding, finding2: ModelFinding
    ) -> bool:
        """Check if two findings describe the same issue."""
        # Same category
        if finding1.category != finding2.category:
            return False
        
        # Similar location (within 5 lines)
        line1 = finding1.location.get("line", 0)
        line2 = finding2.location.get("line", 0)
        if abs(line1 - line2) > 5:
            return False
        
        # Similar titles (simple word overlap)
        words1 = set(finding1.title.lower().split())
        words2 = set(finding2.title.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap >= self._similarity_threshold
    
    def _check_consensus(
        self,
        agreeing_count: int,
        total_count: int,
        strategy: ConsensusStrategy,
        findings: list[ModelFinding],
    ) -> bool:
        """Check if findings meet consensus requirements."""
        if total_count == 0:
            return False
        
        if strategy == ConsensusStrategy.UNANIMOUS:
            return agreeing_count == total_count
        
        elif strategy == ConsensusStrategy.MAJORITY:
            return agreeing_count > total_count / 2
        
        elif strategy == ConsensusStrategy.WEIGHTED:
            # Get weights for agreeing models
            total_weight = sum(m.weight for m in self.models)
            agreeing_weight = sum(
                m.weight for m in self.models
                if m.provider in [f.model for f in findings]
            )
            return agreeing_weight > total_weight / 2
        
        elif strategy == ConsensusStrategy.ANY:
            return agreeing_count >= 1
        
        return False

    def _merge_findings(
        self,
        findings: list[ModelFinding],
        agreeing_models: list[ModelProvider],
        dissenting_models: list[ModelProvider],
        strategy: ConsensusStrategy,
    ) -> ConsensusFinding:
        """Merge multiple model findings into a consensus finding."""
        # Use highest confidence finding as base
        primary = max(findings, key=lambda f: f.confidence)
        
        # Combine confidences
        if strategy == ConsensusStrategy.WEIGHTED:
            model_weights = {m.provider: m.weight for m in self.models}
            total_weight = sum(model_weights.get(f.model, 1.0) for f in findings)
            consensus_confidence = sum(
                f.confidence * model_weights.get(f.model, 1.0)
                for f in findings
            ) / total_weight if total_weight > 0 else 0.5
        else:
            # Average confidence
            consensus_confidence = sum(f.confidence for f in findings) / len(findings)
        
        # Boost confidence based on agreement
        agreement_boost = len(agreeing_models) / (len(agreeing_models) + len(dissenting_models))
        consensus_confidence = min(
            consensus_confidence * (1 + agreement_boost * 0.2),
            0.99
        )
        
        # Merge descriptions
        descriptions = list(set(f.description for f in findings))
        merged_description = primary.description
        if len(descriptions) > 1:
            merged_description += "\n\n[Additional perspectives from other models:]"
            for desc in descriptions[1:]:
                if desc != primary.description:
                    merged_description += f"\n- {desc[:200]}"
        
        # Get best suggested fix
        suggested_fix = None
        for finding in sorted(findings, key=lambda f: f.confidence, reverse=True):
            if finding.suggested_fix:
                suggested_fix = finding.suggested_fix
                break
        
        return ConsensusFinding(
            finding_id=f"consensus_{primary.finding_id}",
            severity=primary.severity,
            category=primary.category,
            title=primary.title,
            description=merged_description,
            location=primary.location,
            consensus_confidence=consensus_confidence,
            agreeing_models=agreeing_models,
            dissenting_models=dissenting_models,
            model_findings=findings,
            suggested_fix=suggested_fix,
            consensus_type=strategy,
        )

    def _consensus_to_dict(self, consensus: ConsensusResult) -> dict[str, Any]:
        """Convert ConsensusResult to dictionary."""
        return {
            "code_hash": consensus.code_hash,
            "consensus_strategy": consensus.consensus_strategy.value,
            "models_queried": [m.value for m in consensus.models_queried],
            "overall_confidence": round(consensus.overall_confidence, 3),
            "total_latency_ms": round(consensus.total_latency_ms, 1),
            "consensus_findings": [
                {
                    "id": f.finding_id,
                    "severity": f.severity,
                    "category": f.category,
                    "title": f.title,
                    "description": f.description,
                    "location": f.location,
                    "confidence": round(f.consensus_confidence, 3),
                    "agreeing_models": [m.value for m in f.agreeing_models],
                    "dissenting_models": [m.value for m in f.dissenting_models],
                    "suggested_fix": f.suggested_fix,
                }
                for f in consensus.consensus_findings
            ],
            "model_only_findings": {
                model: [
                    {
                        "id": f.finding_id,
                        "severity": f.severity,
                        "category": f.category,
                        "title": f.title,
                        "confidence": round(f.confidence, 3),
                    }
                    for f in findings
                ]
                for model, findings in consensus.model_only_findings.items()
            },
            "summary": {
                "total_consensus_findings": len(consensus.consensus_findings),
                "total_model_only_findings": sum(
                    len(f) for f in consensus.model_only_findings.values()
                ),
                "by_severity": self._count_by_severity(consensus.consensus_findings),
            },
        }

    def _count_by_severity(
        self, findings: list[ConsensusFinding]
    ) -> dict[str, int]:
        """Count findings by severity."""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for finding in findings:
            sev = finding.severity.lower()
            if sev in counts:
                counts[sev] += 1
        return counts

    async def verify_with_escalation(
        self,
        code: str,
        context: dict[str, Any],
    ) -> AgentResult:
        """
        Progressive verification that escalates to more models for uncertain findings.
        
        Starts with fast model, escalates to consensus if findings are uncertain.
        """
        # First pass with fast model
        fast_config = self.models[0] if self.models else None
        if not fast_config:
            return await self.analyze(code, context)
        
        initial_findings = await self._query_model(fast_config, code, context)
        
        # Check if any findings need consensus verification
        uncertain_findings = [
            f for f in initial_findings
            if f.confidence < 0.8 or f.severity in ["critical", "high"]
        ]
        
        if not uncertain_findings:
            # High confidence, no need for consensus
            return AgentResult(
                success=True,
                data={
                    "mode": "fast",
                    "findings": [
                        {
                            "id": f.finding_id,
                            "severity": f.severity,
                            "category": f.category,
                            "title": f.title,
                            "description": f.description,
                            "confidence": f.confidence,
                        }
                        for f in initial_findings
                    ],
                },
                latency_ms=0,
            )
        
        # Escalate to full consensus
        logger.info(
            "Escalating to consensus verification",
            uncertain_count=len(uncertain_findings),
        )
        
        return await self.analyze(code, context)

    def set_models(self, models: list[ModelConfig]) -> None:
        """Update the model configuration."""
        self.models = models
        logger.info(
            "Model configuration updated",
            models=[m.provider.value for m in models],
        )

    def set_similarity_threshold(self, threshold: float) -> None:
        """Set the threshold for matching similar findings (0.0 to 1.0)."""
        self._similarity_threshold = max(0.0, min(1.0, threshold))
