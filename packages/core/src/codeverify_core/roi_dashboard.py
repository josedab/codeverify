"""ROI Dashboard & Cost Transparency - Show verification value vs cost.

This module tracks verification costs, bugs found, and calculates ROI
to help teams justify their investment in formal verification.

Key insight: "3 critical bugs caught = $150K in breach costs avoided"
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class BugSeverity(str, Enum):
    """Severity levels for bugs."""
    
    CRITICAL = "critical"  # Security breach, data loss
    HIGH = "high"  # Major functionality broken
    MEDIUM = "medium"  # Feature defect
    LOW = "low"  # Minor issues


class CostCategory(str, Enum):
    """Categories of verification costs."""
    
    LLM_TOKENS = "llm_tokens"  # Token usage for AI agents
    Z3_COMPUTE = "z3_compute"  # SMT solver compute time
    STORAGE = "storage"  # Data storage costs
    API_CALLS = "api_calls"  # External API calls
    INFRASTRUCTURE = "infrastructure"  # General infrastructure


# Industry benchmarks for bug costs (source: IBM Systems Sciences Institute, Ponemon)
BUG_COST_ESTIMATES = {
    BugSeverity.CRITICAL: {
        "min": 50000,
        "avg": 150000,
        "max": 500000,
        "description": "Security breach, data loss, regulatory fines",
    },
    BugSeverity.HIGH: {
        "min": 10000,
        "avg": 25000,
        "max": 75000,
        "description": "Major outage, customer impact",
    },
    BugSeverity.MEDIUM: {
        "min": 1000,
        "avg": 5000,
        "max": 15000,
        "description": "Feature regression, developer time",
    },
    BugSeverity.LOW: {
        "min": 100,
        "avg": 500,
        "max": 2000,
        "description": "Minor fix, documentation",
    },
}


@dataclass
class VerificationCost:
    """A single verification cost entry."""
    
    id: str
    timestamp: datetime
    category: CostCategory
    amount_usd: float
    description: str
    repository: str | None = None
    pr_number: int | None = None
    tokens_used: int | None = None
    compute_seconds: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "amount_usd": round(self.amount_usd, 4),
            "description": self.description,
            "repository": self.repository,
            "pr_number": self.pr_number,
        }


@dataclass
class BugCaught:
    """A bug caught by verification."""
    
    id: str
    timestamp: datetime
    severity: BugSeverity
    title: str
    description: str
    repository: str | None = None
    pr_number: int | None = None
    file_path: str | None = None
    finding_type: str | None = None  # e.g., "null_dereference", "sql_injection"
    estimated_cost_avoided: float | None = None
    was_production: bool = False  # Would this have reached production?
    verification_proof: str | None = None  # Z3 proof if available
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "repository": self.repository,
            "pr_number": self.pr_number,
            "estimated_cost_avoided": self.estimated_cost_avoided,
            "was_production": self.was_production,
        }


@dataclass
class ROIMetrics:
    """Calculated ROI metrics."""
    
    period_start: datetime
    period_end: datetime
    
    # Costs
    total_cost_usd: float
    cost_by_category: dict[str, float]
    cost_per_pr: float
    cost_per_1k_loc: float
    
    # Value
    bugs_caught: int
    bugs_by_severity: dict[str, int]
    estimated_cost_avoided: float
    estimated_cost_avoided_min: float
    estimated_cost_avoided_max: float
    
    # ROI
    roi_percentage: float  # (value - cost) / cost * 100
    net_savings: float
    payback_period_days: float | None
    
    # Trends
    bugs_per_week: float
    cost_trend: str  # "increasing", "stable", "decreasing"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "costs": {
                "total_usd": round(self.total_cost_usd, 2),
                "by_category": {k: round(v, 2) for k, v in self.cost_by_category.items()},
                "per_pr": round(self.cost_per_pr, 2),
                "per_1k_loc": round(self.cost_per_1k_loc, 2),
            },
            "value": {
                "bugs_caught": self.bugs_caught,
                "by_severity": self.bugs_by_severity,
                "estimated_cost_avoided": {
                    "avg": round(self.estimated_cost_avoided, 2),
                    "min": round(self.estimated_cost_avoided_min, 2),
                    "max": round(self.estimated_cost_avoided_max, 2),
                },
            },
            "roi": {
                "percentage": round(self.roi_percentage, 1),
                "net_savings": round(self.net_savings, 2),
                "payback_period_days": self.payback_period_days,
            },
            "trends": {
                "bugs_per_week": round(self.bugs_per_week, 2),
                "cost_trend": self.cost_trend,
            },
        }
    
    def to_summary_string(self) -> str:
        """Generate human-readable summary."""
        return f"""
📊 ROI Summary ({self.period_start.strftime('%b %d')} - {self.period_end.strftime('%b %d, %Y')})

💰 Costs: ${self.total_cost_usd:,.2f}
   • Per PR: ${self.cost_per_pr:.2f}
   • Per 1K LOC: ${self.cost_per_1k_loc:.2f}

🐛 Bugs Caught: {self.bugs_caught}
   • Critical: {self.bugs_by_severity.get('critical', 0)}
   • High: {self.bugs_by_severity.get('high', 0)}
   • Medium: {self.bugs_by_severity.get('medium', 0)}
   • Low: {self.bugs_by_severity.get('low', 0)}

💵 Estimated Cost Avoided: ${self.estimated_cost_avoided:,.2f}
   (Range: ${self.estimated_cost_avoided_min:,.2f} - ${self.estimated_cost_avoided_max:,.2f})

📈 ROI: {self.roi_percentage:.0f}%
   Net Savings: ${self.net_savings:,.2f}
"""


@dataclass
class CostConfig:
    """Configuration for cost calculations."""
    
    # Token costs (per 1K tokens)
    input_token_cost: float = 0.003  # $3 per 1M input
    output_token_cost: float = 0.015  # $15 per 1M output
    
    # Compute costs
    z3_cost_per_minute: float = 0.01  # $0.01 per minute
    
    # Bug cost multipliers (for custom estimates)
    bug_cost_multiplier: float = 1.0  # Adjust based on industry
    
    # Industry-specific adjustments
    industry: str = "technology"  # "technology", "finance", "healthcare"
    
    def get_industry_multiplier(self) -> float:
        """Get cost multiplier based on industry."""
        multipliers = {
            "technology": 1.0,
            "finance": 2.5,  # Higher regulatory costs
            "healthcare": 3.0,  # HIPAA, patient safety
            "government": 2.0,
            "retail": 0.8,
        }
        return multipliers.get(self.industry, 1.0)


class CostTracker:
    """Tracks verification costs."""
    
    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or CostConfig()
        self.costs: list[VerificationCost] = []
        self._cost_counter = 0
    
    def record_cost(
        self,
        category: CostCategory,
        amount_usd: float,
        description: str,
        repository: str | None = None,
        pr_number: int | None = None,
        tokens_used: int | None = None,
        compute_seconds: float | None = None,
    ) -> VerificationCost:
        """Record a verification cost."""
        self._cost_counter += 1
        
        cost = VerificationCost(
            id=f"cost-{self._cost_counter}",
            timestamp=datetime.utcnow(),
            category=category,
            amount_usd=amount_usd,
            description=description,
            repository=repository,
            pr_number=pr_number,
            tokens_used=tokens_used,
            compute_seconds=compute_seconds,
        )
        
        self.costs.append(cost)
        return cost
    
    def record_llm_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "default",
        repository: str | None = None,
        pr_number: int | None = None,
    ) -> VerificationCost:
        """Record LLM token usage and calculate cost."""
        input_cost = (input_tokens / 1000) * self.config.input_token_cost
        output_cost = (output_tokens / 1000) * self.config.output_token_cost
        total_cost = input_cost + output_cost
        
        return self.record_cost(
            category=CostCategory.LLM_TOKENS,
            amount_usd=total_cost,
            description=f"LLM {model}: {input_tokens} in, {output_tokens} out",
            repository=repository,
            pr_number=pr_number,
            tokens_used=input_tokens + output_tokens,
        )
    
    def record_z3_compute(
        self,
        compute_seconds: float,
        repository: str | None = None,
        pr_number: int | None = None,
    ) -> VerificationCost:
        """Record Z3 compute time and calculate cost."""
        minutes = compute_seconds / 60
        cost = minutes * self.config.z3_cost_per_minute
        
        return self.record_cost(
            category=CostCategory.Z3_COMPUTE,
            amount_usd=cost,
            description=f"Z3 verification: {compute_seconds:.1f}s",
            repository=repository,
            pr_number=pr_number,
            compute_seconds=compute_seconds,
        )
    
    def get_total_cost(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> float:
        """Get total cost for a period."""
        costs = self._filter_by_period(self.costs, since, until)
        return sum(c.amount_usd for c in costs)
    
    def get_cost_by_category(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, float]:
        """Get costs broken down by category."""
        costs = self._filter_by_period(self.costs, since, until)
        
        by_category: dict[str, float] = {}
        for cost in costs:
            cat = cost.category.value
            by_category[cat] = by_category.get(cat, 0) + cost.amount_usd
        
        return by_category
    
    def _filter_by_period(
        self,
        items: list,
        since: datetime | None,
        until: datetime | None,
    ) -> list:
        """Filter items by time period."""
        filtered = items
        if since:
            filtered = [i for i in filtered if i.timestamp >= since]
        if until:
            filtered = [i for i in filtered if i.timestamp <= until]
        return filtered


class BugValueCalculator:
    """Calculates the value of bugs caught."""
    
    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or CostConfig()
        self.bugs_caught: list[BugCaught] = []
        self._bug_counter = 0
    
    def record_bug(
        self,
        severity: BugSeverity,
        title: str,
        description: str,
        repository: str | None = None,
        pr_number: int | None = None,
        file_path: str | None = None,
        finding_type: str | None = None,
        was_production: bool = False,
        verification_proof: str | None = None,
    ) -> BugCaught:
        """Record a bug caught by verification."""
        self._bug_counter += 1
        
        # Calculate estimated cost avoided
        multiplier = self.config.bug_cost_multiplier * self.config.get_industry_multiplier()
        base_cost = BUG_COST_ESTIMATES[severity]["avg"]
        
        # Production bugs cost more (already deployed, more users affected)
        if was_production:
            multiplier *= 1.5
        
        estimated_cost = base_cost * multiplier
        
        bug = BugCaught(
            id=f"bug-{self._bug_counter}",
            timestamp=datetime.utcnow(),
            severity=severity,
            title=title,
            description=description,
            repository=repository,
            pr_number=pr_number,
            file_path=file_path,
            finding_type=finding_type,
            estimated_cost_avoided=estimated_cost,
            was_production=was_production,
            verification_proof=verification_proof,
        )
        
        self.bugs_caught.append(bug)
        return bug
    
    def get_bugs_by_severity(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, int]:
        """Get bug counts by severity."""
        bugs = self._filter_by_period(self.bugs_caught, since, until)
        
        by_severity: dict[str, int] = {}
        for bug in bugs:
            sev = bug.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        return by_severity
    
    def get_total_value(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> tuple[float, float, float]:
        """Get total estimated cost avoided (avg, min, max)."""
        bugs = self._filter_by_period(self.bugs_caught, since, until)
        
        total_avg = 0.0
        total_min = 0.0
        total_max = 0.0
        
        multiplier = self.config.bug_cost_multiplier * self.config.get_industry_multiplier()
        
        for bug in bugs:
            estimates = BUG_COST_ESTIMATES[bug.severity]
            prod_mult = 1.5 if bug.was_production else 1.0
            
            total_avg += estimates["avg"] * multiplier * prod_mult
            total_min += estimates["min"] * multiplier * prod_mult
            total_max += estimates["max"] * multiplier * prod_mult
        
        return total_avg, total_min, total_max
    
    def _filter_by_period(
        self,
        items: list,
        since: datetime | None,
        until: datetime | None,
    ) -> list:
        """Filter items by time period."""
        filtered = items
        if since:
            filtered = [i for i in filtered if i.timestamp >= since]
        if until:
            filtered = [i for i in filtered if i.timestamp <= until]
        return filtered


class ROIDashboard:
    """Main ROI dashboard that combines costs and value."""
    
    def __init__(
        self,
        config: CostConfig | None = None,
        storage_path: str | None = None,
    ) -> None:
        self.config = config or CostConfig()
        self.storage_path = Path(storage_path) if storage_path else None
        
        self.cost_tracker = CostTracker(self.config)
        self.bug_calculator = BugValueCalculator(self.config)
        
        # Additional tracking
        self.prs_analyzed: int = 0
        self.lines_of_code_analyzed: int = 0
        
        # Load persisted state
        if self.storage_path and self.storage_path.exists():
            self._load_state()
    
    def record_pr_analysis(
        self,
        repository: str,
        pr_number: int,
        lines_of_code: int,
        input_tokens: int,
        output_tokens: int,
        z3_seconds: float,
        bugs_found: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Record a complete PR analysis with costs and findings."""
        # Record costs
        self.cost_tracker.record_llm_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            repository=repository,
            pr_number=pr_number,
        )
        
        if z3_seconds > 0:
            self.cost_tracker.record_z3_compute(
                compute_seconds=z3_seconds,
                repository=repository,
                pr_number=pr_number,
            )
        
        # Record bugs
        recorded_bugs = []
        if bugs_found:
            for bug_data in bugs_found:
                bug = self.bug_calculator.record_bug(
                    severity=BugSeverity(bug_data.get("severity", "medium")),
                    title=bug_data.get("title", "Unknown bug"),
                    description=bug_data.get("description", ""),
                    repository=repository,
                    pr_number=pr_number,
                    file_path=bug_data.get("file_path"),
                    finding_type=bug_data.get("finding_type"),
                    was_production=bug_data.get("was_production", False),
                    verification_proof=bug_data.get("verification_proof"),
                )
                recorded_bugs.append(bug)
        
        # Update counters
        self.prs_analyzed += 1
        self.lines_of_code_analyzed += lines_of_code
        
        # Save state
        self._save_state()
        
        return {
            "pr": f"{repository}#{pr_number}",
            "cost_usd": self.cost_tracker.costs[-1].amount_usd if self.cost_tracker.costs else 0,
            "bugs_found": len(recorded_bugs),
            "value_generated": sum(b.estimated_cost_avoided or 0 for b in recorded_bugs),
        }
    
    def calculate_roi(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> ROIMetrics:
        """Calculate comprehensive ROI metrics."""
        # Default to last 30 days
        if not until:
            until = datetime.utcnow()
        if not since:
            since = until - timedelta(days=30)
        
        # Get costs
        total_cost = self.cost_tracker.get_total_cost(since, until)
        cost_by_category = self.cost_tracker.get_cost_by_category(since, until)
        
        # Get value
        bugs_by_severity = self.bug_calculator.get_bugs_by_severity(since, until)
        total_bugs = sum(bugs_by_severity.values())
        value_avg, value_min, value_max = self.bug_calculator.get_total_value(since, until)
        
        # Calculate per-unit costs
        period_prs = max(1, self.prs_analyzed)  # Avoid division by zero
        period_loc = max(1, self.lines_of_code_analyzed)
        
        cost_per_pr = total_cost / period_prs
        cost_per_1k_loc = (total_cost / period_loc) * 1000
        
        # Calculate ROI
        net_savings = value_avg - total_cost
        roi_percentage = (net_savings / max(total_cost, 0.01)) * 100
        
        # Calculate payback period (days to pay off cost with bug savings)
        days_in_period = (until - since).days or 1
        daily_value = value_avg / days_in_period
        payback_days = total_cost / max(daily_value, 0.01) if daily_value > 0 else None
        
        # Calculate trends
        bugs_per_week = (total_bugs / days_in_period) * 7
        
        # Cost trend (simplified)
        cost_trend = "stable"
        if len(self.cost_tracker.costs) > 10:
            recent_costs = self.cost_tracker.costs[-5:]
            older_costs = self.cost_tracker.costs[-10:-5]
            recent_avg = sum(c.amount_usd for c in recent_costs) / len(recent_costs)
            older_avg = sum(c.amount_usd for c in older_costs) / len(older_costs)
            if recent_avg > older_avg * 1.1:
                cost_trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                cost_trend = "decreasing"
        
        return ROIMetrics(
            period_start=since,
            period_end=until,
            total_cost_usd=total_cost,
            cost_by_category=cost_by_category,
            cost_per_pr=cost_per_pr,
            cost_per_1k_loc=cost_per_1k_loc,
            bugs_caught=total_bugs,
            bugs_by_severity=bugs_by_severity,
            estimated_cost_avoided=value_avg,
            estimated_cost_avoided_min=value_min,
            estimated_cost_avoided_max=value_max,
            roi_percentage=roi_percentage,
            net_savings=net_savings,
            payback_period_days=payback_days,
            bugs_per_week=bugs_per_week,
            cost_trend=cost_trend,
        )
    
    def generate_report(
        self,
        format: str = "dict",
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, Any] | str:
        """Generate ROI report in specified format."""
        metrics = self.calculate_roi(since, until)
        
        if format == "summary":
            return metrics.to_summary_string()
        elif format == "dict":
            return metrics.to_dict()
        elif format == "json":
            return json.dumps(metrics.to_dict(), indent=2)
        else:
            return metrics.to_dict()
    
    def get_recent_activity(self, limit: int = 10) -> dict[str, Any]:
        """Get recent costs and bugs."""
        recent_costs = self.cost_tracker.costs[-limit:]
        recent_bugs = self.bug_calculator.bugs_caught[-limit:]
        
        return {
            "recent_costs": [c.to_dict() for c in recent_costs],
            "recent_bugs": [b.to_dict() for b in recent_bugs],
        }
    
    def export_for_audit(
        self,
        since: datetime,
        until: datetime,
    ) -> dict[str, Any]:
        """Export data for compliance/audit purposes."""
        costs = [
            c for c in self.cost_tracker.costs
            if since <= c.timestamp <= until
        ]
        bugs = [
            b for b in self.bug_calculator.bugs_caught
            if since <= b.timestamp <= until
        ]
        
        return {
            "export_timestamp": datetime.utcnow().isoformat(),
            "period": {
                "start": since.isoformat(),
                "end": until.isoformat(),
            },
            "summary": self.calculate_roi(since, until).to_dict(),
            "detailed_costs": [c.to_dict() for c in costs],
            "detailed_bugs": [b.to_dict() for b in bugs],
            "metadata": {
                "total_prs_analyzed": self.prs_analyzed,
                "total_loc_analyzed": self.lines_of_code_analyzed,
                "cost_config": {
                    "industry": self.config.industry,
                    "input_token_cost": self.config.input_token_cost,
                    "output_token_cost": self.config.output_token_cost,
                },
            },
        }
    
    def _save_state(self) -> None:
        """Persist state to storage."""
        if not self.storage_path:
            return
        
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            state = {
                "prs_analyzed": self.prs_analyzed,
                "lines_of_code_analyzed": self.lines_of_code_analyzed,
                "costs": [c.to_dict() for c in self.cost_tracker.costs],
                "bugs": [b.to_dict() for b in self.bug_calculator.bugs_caught],
            }
            
            with open(self.storage_path / "roi_state.json", "w") as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error("Failed to save ROI state", error=str(e))
    
    def _load_state(self) -> None:
        """Load state from storage."""
        if not self.storage_path:
            return
        
        state_file = self.storage_path / "roi_state.json"
        if not state_file.exists():
            return
        
        try:
            with open(state_file) as f:
                state = json.load(f)
            
            self.prs_analyzed = state.get("prs_analyzed", 0)
            self.lines_of_code_analyzed = state.get("lines_of_code_analyzed", 0)
            
            logger.debug("ROI state loaded", prs=self.prs_analyzed)
            
        except Exception as e:
            logger.error("Failed to load ROI state", error=str(e))


# Convenience function
def create_dashboard(
    config: CostConfig | None = None,
    storage_path: str | None = None,
) -> ROIDashboard:
    """Create an ROI dashboard."""
    return ROIDashboard(config, storage_path)
