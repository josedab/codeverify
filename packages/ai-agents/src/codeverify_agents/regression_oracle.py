"""Regression Oracle - ML-powered prediction of bug-prone changes."""

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class RiskLevel(str, Enum):
    """Risk level classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ChangeMetrics:
    """Metrics about a code change."""
    lines_added: int = 0
    lines_deleted: int = 0
    files_changed: int = 0
    functions_modified: int = 0
    complexity_delta: float = 0.0
    test_coverage_delta: float = 0.0
    touches_critical_path: bool = False
    modifies_interfaces: bool = False
    cross_module_changes: bool = False


@dataclass
class HistoricalSignal:
    """Historical signals about code/author/area."""
    file_bug_frequency: float = 0.0  # Bugs per 100 changes in this file
    author_bug_rate: float = 0.0  # Author's bug introduction rate
    area_churn_rate: float = 0.0  # How often this area changes
    recent_bug_fixes: int = 0  # Bug fixes in last 30 days
    time_since_last_incident: int = 0  # Days since last bug
    code_age_days: int = 0  # How old is this code


@dataclass
class RiskPrediction:
    """Prediction result for a code change."""
    change_id: str
    risk_level: RiskLevel
    risk_score: float  # 0-100
    confidence: float  # 0-1
    risk_factors: list[dict[str, Any]] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    similar_past_bugs: list[dict[str, Any]] = field(default_factory=list)
    verification_priority: int = 1  # 1 = highest priority


@dataclass
class BugRecord:
    """Historical bug record for training."""
    bug_id: str
    timestamp: datetime
    file_path: str
    function_name: str | None
    change_metrics: ChangeMetrics
    author: str
    severity: str
    root_cause: str
    fix_complexity: str


class RegressionOracle(BaseAgent):
    """
    ML-powered regression prediction that identifies high-risk changes.
    
    This oracle analyzes code changes and predicts which are most likely
    to introduce bugs, allowing focused verification effort.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize regression oracle."""
        super().__init__(config)
        
        # Feature weights learned from historical data
        self._feature_weights: dict[str, float] = {
            "lines_changed": 0.15,
            "complexity_delta": 0.20,
            "file_bug_history": 0.25,
            "author_bug_rate": 0.10,
            "touches_critical_path": 0.15,
            "cross_module": 0.10,
            "code_churn": 0.05,
        }
        
        # Bug history storage (in production, use database)
        self._bug_history: list[BugRecord] = []
        self._file_bug_counts: dict[str, int] = {}
        self._file_change_counts: dict[str, int] = {}
        self._author_stats: dict[str, dict[str, int]] = {}

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Analyze a code change and predict bug risk.

        Args:
            code: The diff or changed code
            context: Additional context including:
                - change_id: Unique identifier for the change
                - file_paths: List of files changed
                - author: Author of the change
                - commit_message: Commit message
                - base_branch: Branch being merged into
                - change_metrics: Pre-computed ChangeMetrics (optional)

        Returns:
            AgentResult with risk prediction
        """
        start_time = time.time()
        change_id = context.get("change_id", self._generate_change_id(code))
        
        try:
            # Extract or use provided metrics
            metrics = context.get("change_metrics")
            if metrics is None:
                metrics = self._extract_change_metrics(code, context)
            
            # Get historical signals
            file_paths = context.get("file_paths", [])
            author = context.get("author", "unknown")
            historical = self._get_historical_signals(file_paths, author)
            
            # Compute base risk score
            base_score = self._compute_base_risk_score(metrics, historical)
            
            # Use LLM for semantic risk analysis
            llm_analysis = await self._semantic_risk_analysis(code, context)
            
            # Combine scores
            final_score = self._combine_scores(base_score, llm_analysis)
            
            # Find similar past bugs
            similar_bugs = self._find_similar_bugs(code, metrics)
            
            # Generate prediction
            prediction = self._generate_prediction(
                change_id=change_id,
                score=final_score,
                metrics=metrics,
                historical=historical,
                llm_analysis=llm_analysis,
                similar_bugs=similar_bugs,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "Risk prediction generated",
                change_id=change_id,
                risk_level=prediction.risk_level.value,
                risk_score=prediction.risk_score,
                confidence=prediction.confidence,
                latency_ms=elapsed_ms,
            )
            
            return AgentResult(
                success=True,
                data=self._prediction_to_dict(prediction),
                tokens_used=llm_analysis.get("tokens", 0),
                latency_ms=elapsed_ms,
            )
            
        except Exception as e:
            logger.error("Risk prediction failed", error=str(e), change_id=change_id)
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _generate_change_id(self, code: str) -> str:
        """Generate a unique ID for a code change."""
        return hashlib.sha256(code.encode()).hexdigest()[:12]

    def _extract_change_metrics(
        self, diff: str, context: dict[str, Any]
    ) -> ChangeMetrics:
        """Extract metrics from a diff."""
        lines = diff.split("\n")
        
        added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
        deleted = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
        
        # Count modified functions (rough heuristic)
        func_patterns = ["def ", "function ", "async def ", "async function "]
        functions_modified = sum(
            1 for line in lines 
            if any(p in line for p in func_patterns) and (line.startswith("+") or line.startswith("-"))
        )
        
        file_paths = context.get("file_paths", [])
        
        # Check for cross-module changes
        modules = set()
        for path in file_paths:
            parts = path.split("/")
            if len(parts) > 1:
                modules.add(parts[0])
        cross_module = len(modules) > 1
        
        # Check for interface modifications
        interface_keywords = ["interface ", "class ", "type ", "export ", "public "]
        modifies_interfaces = any(
            any(kw in line for kw in interface_keywords)
            for line in lines
            if line.startswith("+") or line.startswith("-")
        )
        
        return ChangeMetrics(
            lines_added=added,
            lines_deleted=deleted,
            files_changed=len(file_paths),
            functions_modified=functions_modified,
            complexity_delta=self._estimate_complexity_delta(diff),
            cross_module_changes=cross_module,
            modifies_interfaces=modifies_interfaces,
        )

    def _estimate_complexity_delta(self, diff: str) -> float:
        """Estimate change in cyclomatic complexity from diff."""
        complexity_indicators = [
            "if ", "else ", "elif ", "else:", "for ", "while ", 
            "try:", "except ", "catch ", "switch ", "case ",
            " and ", " or ", "&&", "||", "?",
        ]
        
        lines = diff.split("\n")
        added_complexity = 0
        removed_complexity = 0
        
        for line in lines:
            for indicator in complexity_indicators:
                if indicator in line.lower():
                    if line.startswith("+"):
                        added_complexity += 1
                    elif line.startswith("-"):
                        removed_complexity += 1
        
        return added_complexity - removed_complexity

    def _get_historical_signals(
        self, file_paths: list[str], author: str
    ) -> HistoricalSignal:
        """Get historical signals for risk assessment."""
        # Calculate file bug frequency
        total_bugs = 0
        total_changes = 0
        
        for path in file_paths:
            total_bugs += self._file_bug_counts.get(path, 0)
            total_changes += self._file_change_counts.get(path, 1)  # Avoid div by 0
        
        file_bug_freq = (total_bugs / total_changes * 100) if total_changes > 0 else 0
        
        # Get author stats
        author_data = self._author_stats.get(author, {"bugs": 0, "commits": 1})
        author_bug_rate = author_data["bugs"] / author_data["commits"]
        
        # Count recent bug fixes in these files
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_fixes = sum(
            1 for bug in self._bug_history
            if bug.file_path in file_paths and bug.timestamp > thirty_days_ago
        )
        
        return HistoricalSignal(
            file_bug_frequency=file_bug_freq,
            author_bug_rate=author_bug_rate,
            recent_bug_fixes=recent_fixes,
        )

    def _compute_base_risk_score(
        self, metrics: ChangeMetrics, historical: HistoricalSignal
    ) -> float:
        """Compute base risk score from metrics and history."""
        score = 0.0
        
        # Lines changed factor (logarithmic)
        lines_changed = metrics.lines_added + metrics.lines_deleted
        lines_factor = min(math.log(lines_changed + 1) / math.log(1000), 1.0)
        score += lines_factor * self._feature_weights["lines_changed"] * 100
        
        # Complexity delta
        complexity_factor = min(abs(metrics.complexity_delta) / 20, 1.0)
        score += complexity_factor * self._feature_weights["complexity_delta"] * 100
        
        # File bug history
        history_factor = min(historical.file_bug_frequency / 10, 1.0)
        score += history_factor * self._feature_weights["file_bug_history"] * 100
        
        # Author bug rate
        author_factor = min(historical.author_bug_rate * 10, 1.0)
        score += author_factor * self._feature_weights["author_bug_rate"] * 100
        
        # Critical path
        if metrics.touches_critical_path:
            score += self._feature_weights["touches_critical_path"] * 100
        
        # Cross-module changes
        if metrics.cross_module_changes:
            score += self._feature_weights["cross_module"] * 100
        
        return min(score, 100)

    async def _semantic_risk_analysis(
        self, code: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Use LLM for semantic risk analysis."""
        system_prompt = """You are an expert code reviewer analyzing changes for regression risk.

Evaluate the code change for:
1. Logic errors or edge cases that could cause bugs
2. Patterns known to cause regressions (null handling, state management, concurrency)
3. Missing error handling or input validation
4. Breaking changes to interfaces or contracts
5. Performance regressions

Respond in JSON:
{
  "risk_score": 0-100,
  "risk_factors": [
    {"factor": "description", "severity": "high/medium/low", "confidence": 0.0-1.0}
  ],
  "potential_issues": ["issue1", "issue2"],
  "recommended_tests": ["test1", "test2"]
}"""

        user_prompt = f"""Analyze this code change for regression risk:

Commit message: {context.get('commit_message', 'No message')}
Files: {', '.join(context.get('file_paths', []))}

```diff
{code[:10000]}
```"""

        try:
            response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_mode=True,
            )
            
            result = json.loads(response["content"])
            result["tokens"] = response.get("tokens", 0)
            return result
            
        except Exception as e:
            logger.warning("LLM risk analysis failed", error=str(e))
            return {"risk_score": 50, "risk_factors": [], "tokens": 0}

    def _combine_scores(
        self, base_score: float, llm_analysis: dict[str, Any]
    ) -> float:
        """Combine base heuristic score with LLM analysis."""
        llm_score = llm_analysis.get("risk_score", 50)
        
        # Weighted average: 60% heuristic, 40% LLM
        combined = base_score * 0.6 + llm_score * 0.4
        
        # Boost if LLM found high-severity factors
        high_severity_count = sum(
            1 for f in llm_analysis.get("risk_factors", [])
            if f.get("severity") == "high"
        )
        if high_severity_count > 0:
            combined = min(combined * 1.2, 100)
        
        return combined

    def _find_similar_bugs(
        self, code: str, metrics: ChangeMetrics
    ) -> list[dict[str, Any]]:
        """Find similar historical bugs."""
        similar = []
        
        for bug in self._bug_history[-100:]:  # Check recent bugs
            similarity = self._compute_bug_similarity(code, metrics, bug)
            if similarity > 0.5:
                similar.append({
                    "bug_id": bug.bug_id,
                    "similarity": similarity,
                    "root_cause": bug.root_cause,
                    "severity": bug.severity,
                })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)[:5]

    def _compute_bug_similarity(
        self, code: str, metrics: ChangeMetrics, bug: BugRecord
    ) -> float:
        """Compute similarity between current change and historical bug."""
        similarity = 0.0
        
        # Size similarity
        bug_size = bug.change_metrics.lines_added + bug.change_metrics.lines_deleted
        current_size = metrics.lines_added + metrics.lines_deleted
        
        if bug_size > 0 and current_size > 0:
            size_ratio = min(bug_size, current_size) / max(bug_size, current_size)
            similarity += size_ratio * 0.3
        
        # Complexity similarity
        if abs(bug.change_metrics.complexity_delta) > 0:
            complexity_ratio = min(
                abs(metrics.complexity_delta),
                abs(bug.change_metrics.complexity_delta)
            ) / max(
                abs(metrics.complexity_delta),
                abs(bug.change_metrics.complexity_delta),
                1
            )
            similarity += complexity_ratio * 0.3
        
        # Same file bonus
        if bug.file_path in code:
            similarity += 0.4
        
        return similarity

    def _generate_prediction(
        self,
        change_id: str,
        score: float,
        metrics: ChangeMetrics,
        historical: HistoricalSignal,
        llm_analysis: dict[str, Any],
        similar_bugs: list[dict[str, Any]],
    ) -> RiskPrediction:
        """Generate the final risk prediction."""
        # Determine risk level
        if score >= 80:
            risk_level = RiskLevel.CRITICAL
            priority = 1
        elif score >= 60:
            risk_level = RiskLevel.HIGH
            priority = 2
        elif score >= 40:
            risk_level = RiskLevel.MEDIUM
            priority = 3
        else:
            risk_level = RiskLevel.LOW
            priority = 4
        
        # Compile risk factors
        risk_factors = []
        
        if metrics.lines_added + metrics.lines_deleted > 500:
            risk_factors.append({
                "factor": "Large change size",
                "details": f"{metrics.lines_added + metrics.lines_deleted} lines changed",
                "contribution": 15,
            })
        
        if metrics.complexity_delta > 5:
            risk_factors.append({
                "factor": "Increased complexity",
                "details": f"Complexity increased by {metrics.complexity_delta}",
                "contribution": 20,
            })
        
        if historical.file_bug_frequency > 5:
            risk_factors.append({
                "factor": "Bug-prone file",
                "details": f"Historical bug rate: {historical.file_bug_frequency:.1f}%",
                "contribution": 25,
            })
        
        if metrics.cross_module_changes:
            risk_factors.append({
                "factor": "Cross-module changes",
                "details": "Changes span multiple modules",
                "contribution": 10,
            })
        
        # Add LLM-identified factors
        for factor in llm_analysis.get("risk_factors", []):
            risk_factors.append({
                "factor": factor.get("factor", "Unknown"),
                "details": f"Confidence: {factor.get('confidence', 0):.0%}",
                "contribution": 10 if factor.get("severity") == "high" else 5,
            })
        
        # Generate recommendations
        recommendations = []
        
        if score >= 60:
            recommendations.append("Full formal verification recommended")
        
        if metrics.modifies_interfaces:
            recommendations.append("Review interface contracts for breaking changes")
        
        if historical.recent_bug_fixes > 2:
            recommendations.append("Area has recent bug history - extra review needed")
        
        recommendations.extend(llm_analysis.get("recommended_tests", []))
        
        # Confidence based on data availability
        confidence = 0.7
        if len(self._bug_history) > 100:
            confidence += 0.1
        if historical.file_bug_frequency > 0:
            confidence += 0.1
        confidence = min(confidence, 0.95)
        
        return RiskPrediction(
            change_id=change_id,
            risk_level=risk_level,
            risk_score=score,
            confidence=confidence,
            risk_factors=risk_factors,
            recommended_actions=recommendations[:5],
            similar_past_bugs=similar_bugs,
            verification_priority=priority,
        )

    def _prediction_to_dict(self, prediction: RiskPrediction) -> dict[str, Any]:
        """Convert prediction to dictionary."""
        return {
            "change_id": prediction.change_id,
            "risk_level": prediction.risk_level.value,
            "risk_score": round(prediction.risk_score, 1),
            "confidence": round(prediction.confidence, 2),
            "verification_priority": prediction.verification_priority,
            "risk_factors": prediction.risk_factors,
            "recommended_actions": prediction.recommended_actions,
            "similar_past_bugs": prediction.similar_past_bugs,
        }

    def record_bug(self, bug: BugRecord) -> None:
        """Record a bug for training the oracle."""
        self._bug_history.append(bug)
        
        # Update file stats
        self._file_bug_counts[bug.file_path] = (
            self._file_bug_counts.get(bug.file_path, 0) + 1
        )
        
        # Update author stats
        if bug.author not in self._author_stats:
            self._author_stats[bug.author] = {"bugs": 0, "commits": 1}
        self._author_stats[bug.author]["bugs"] += 1
        
        logger.info(
            "Bug recorded for training",
            bug_id=bug.bug_id,
            file=bug.file_path,
            severity=bug.severity,
        )

    def record_change(self, file_path: str, author: str) -> None:
        """Record a change (for normalization)."""
        self._file_change_counts[file_path] = (
            self._file_change_counts.get(file_path, 0) + 1
        )
        
        if author not in self._author_stats:
            self._author_stats[author] = {"bugs": 0, "commits": 0}
        self._author_stats[author]["commits"] += 1

    def update_weights(self, feedback: list[dict[str, Any]]) -> None:
        """Update feature weights based on prediction feedback."""
        if not feedback:
            return
        
        # Simple weight adjustment based on correct/incorrect predictions
        for item in feedback:
            predicted_high = item.get("predicted_risk", 0) >= 60
            was_bug = item.get("was_bug", False)
            
            if predicted_high and was_bug:
                # True positive - weights are good
                pass
            elif predicted_high and not was_bug:
                # False positive - reduce weights slightly
                for key in self._feature_weights:
                    self._feature_weights[key] *= 0.99
            elif not predicted_high and was_bug:
                # False negative - increase weights
                for key in self._feature_weights:
                    self._feature_weights[key] *= 1.01
        
        # Normalize weights to sum to 1
        total = sum(self._feature_weights.values())
        for key in self._feature_weights:
            self._feature_weights[key] /= total
        
        logger.info("Feature weights updated", weights=self._feature_weights)

    async def batch_predict(
        self, changes: list[dict[str, Any]]
    ) -> list[RiskPrediction]:
        """
        Predict risk for multiple changes and prioritize verification.
        
        Returns changes sorted by verification priority (highest risk first).
        """
        predictions = []
        
        for change in changes:
            result = await self.analyze(
                code=change.get("diff", ""),
                context=change,
            )
            if result.success:
                pred = RiskPrediction(
                    change_id=result.data["change_id"],
                    risk_level=RiskLevel(result.data["risk_level"]),
                    risk_score=result.data["risk_score"],
                    confidence=result.data["confidence"],
                    risk_factors=result.data["risk_factors"],
                    recommended_actions=result.data["recommended_actions"],
                    similar_past_bugs=result.data["similar_past_bugs"],
                    verification_priority=result.data["verification_priority"],
                )
                predictions.append(pred)
        
        # Sort by risk score (highest first)
        predictions.sort(key=lambda p: p.risk_score, reverse=True)
        
        return predictions

    def get_verification_budget_allocation(
        self,
        predictions: list[RiskPrediction],
        total_budget_minutes: int,
    ) -> dict[str, int]:
        """
        Allocate verification time budget across changes.
        
        Returns dict mapping change_id to allocated minutes.
        """
        if not predictions:
            return {}
        
        # Allocate proportionally to risk score
        total_risk = sum(p.risk_score for p in predictions)
        
        allocations = {}
        for pred in predictions:
            if total_risk > 0:
                share = pred.risk_score / total_risk
            else:
                share = 1 / len(predictions)
            
            minutes = max(1, int(total_budget_minutes * share))
            allocations[pred.change_id] = minutes
        
        return allocations
