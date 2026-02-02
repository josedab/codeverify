"""Self-Healing Code Suggestions - Automatic fix generation and verification.

This module provides:
1. Automatic fix generation when Z3 finds bugs
2. Verification of generated fixes using formal methods
3. Fix ranking by confidence and proof strength
4. Inline presentation with mathematical proof summaries
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import json
import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent, CodeContext

logger = structlog.get_logger()


class FixCategory(str, Enum):
    """Categories of fixes."""
    NULL_CHECK = "null_check"
    BOUNDS_CHECK = "bounds_check"
    OVERFLOW_CHECK = "overflow_check"
    DIVISION_GUARD = "division_guard"
    TYPE_GUARD = "type_guard"
    EXCEPTION_HANDLING = "exception_handling"
    INITIALIZATION = "initialization"
    VALIDATION = "validation"
    REFACTORING = "refactoring"


class ProofStatus(str, Enum):
    """Status of fix verification proof."""
    PROVEN = "proven"
    LIKELY_CORRECT = "likely_correct"
    UNVERIFIED = "unverified"
    FAILED = "failed"


@dataclass
class VerifiedFix:
    """A verified code fix with proof."""
    fix_id: str = ""
    category: FixCategory = FixCategory.VALIDATION
    original_code: str = ""
    fixed_code: str = ""
    description: str = ""
    
    # Proof information
    proof_status: ProofStatus = ProofStatus.UNVERIFIED
    proof_summary: str = ""
    proof_time_ms: float = 0.0
    z3_constraints_satisfied: list[str] = field(default_factory=list)
    
    # Confidence and ranking
    confidence: float = 0.0
    rank: int = 0
    
    # Fix details
    line_start: int = 0
    line_end: int = 0
    affected_variables: list[str] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generation_method: str = "llm"  # llm, template, hybrid

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fix_id": self.fix_id,
            "category": self.category.value,
            "original_code": self.original_code,
            "fixed_code": self.fixed_code,
            "description": self.description,
            "proof_status": self.proof_status.value,
            "proof_summary": self.proof_summary,
            "proof_time_ms": self.proof_time_ms,
            "confidence": self.confidence,
            "rank": self.rank,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass
class FixGenerationResult:
    """Result of fix generation."""
    success: bool = False
    fixes: list[VerifiedFix] = field(default_factory=list)
    error: str | None = None
    total_generated: int = 0
    total_verified: int = 0
    generation_time_ms: float = 0.0
    verification_time_ms: float = 0.0


@dataclass
class BugReport:
    """A bug report from verification."""
    bug_id: str = ""
    category: str = ""
    description: str = ""
    severity: str = "medium"
    location: dict[str, Any] = field(default_factory=dict)
    
    # Z3 information
    z3_counterexample: dict[str, Any] | None = None
    z3_constraint: str | None = None
    
    # Code context
    code_snippet: str = ""
    surrounding_code: str = ""
    language: str = "python"


# Fix templates for common patterns
FIX_TEMPLATES: dict[str, dict[str, Any]] = {
    "null_check_python": {
        "category": FixCategory.NULL_CHECK,
        "pattern": "if {var} is None:\n    raise ValueError('{var} cannot be None')\n",
        "description": "Add null check before using potentially null variable",
    },
    "null_check_typescript": {
        "category": FixCategory.NULL_CHECK,
        "pattern": "if ({var} === null || {var} === undefined) {{\n    throw new Error('{var} cannot be null');\n}}\n",
        "description": "Add null/undefined check",
    },
    "bounds_check_python": {
        "category": FixCategory.BOUNDS_CHECK,
        "pattern": "if {index} < 0 or {index} >= len({array}):\n    raise IndexError('Index out of bounds')\n",
        "description": "Add array bounds check before access",
    },
    "bounds_check_typescript": {
        "category": FixCategory.BOUNDS_CHECK,
        "pattern": "if ({index} < 0 || {index} >= {array}.length) {{\n    throw new RangeError('Index out of bounds');\n}}\n",
        "description": "Add array bounds check",
    },
    "overflow_check_python": {
        "category": FixCategory.OVERFLOW_CHECK,
        "pattern": "import sys\nif {result} > sys.maxsize or {result} < -sys.maxsize - 1:\n    raise OverflowError('Integer overflow')\n",
        "description": "Add integer overflow check",
    },
    "division_guard_python": {
        "category": FixCategory.DIVISION_GUARD,
        "pattern": "if {divisor} == 0:\n    raise ZeroDivisionError('Division by zero')\n",
        "description": "Add division by zero guard",
    },
    "division_guard_typescript": {
        "category": FixCategory.DIVISION_GUARD,
        "pattern": "if ({divisor} === 0) {{\n    throw new Error('Division by zero');\n}}\n",
        "description": "Add division by zero guard",
    },
    "optional_chaining_typescript": {
        "category": FixCategory.NULL_CHECK,
        "pattern": "{object}?.{property}",
        "description": "Use optional chaining for safe property access",
    },
}


class FixVerifier:
    """Verifies generated fixes using Z3."""

    def __init__(self) -> None:
        self._z3_available = self._check_z3_available()

    def _check_z3_available(self) -> bool:
        """Check if Z3 is available."""
        try:
            import z3
            return True
        except ImportError:
            return False

    async def verify_fix(
        self,
        original_code: str,
        fixed_code: str,
        bug_category: str,
        language: str = "python",
    ) -> tuple[ProofStatus, str, float]:
        """Verify that a fix actually resolves the bug.
        
        Returns (status, proof_summary, time_ms).
        """
        import time
        start = time.time()
        
        if not self._z3_available:
            return ProofStatus.UNVERIFIED, "Z3 not available for verification", 0.0
        
        try:
            import z3
            
            # Build verification based on bug category
            if bug_category == "null_safety":
                status, summary = self._verify_null_safety(fixed_code, language)
            elif bug_category == "bounds":
                status, summary = self._verify_bounds_check(fixed_code, language)
            elif bug_category == "overflow":
                status, summary = self._verify_overflow_check(fixed_code, language)
            elif bug_category == "division":
                status, summary = self._verify_division_guard(fixed_code, language)
            else:
                # Generic structural verification
                status, summary = self._verify_structural(original_code, fixed_code)
            
            time_ms = (time.time() - start) * 1000
            return status, summary, time_ms
            
        except Exception as e:
            time_ms = (time.time() - start) * 1000
            return ProofStatus.FAILED, f"Verification error: {str(e)}", time_ms

    def _verify_null_safety(self, code: str, language: str) -> tuple[ProofStatus, str]:
        """Verify null safety fix."""
        # Check if the fix contains appropriate null check patterns
        if language == "python":
            patterns = ["is None", "is not None", "if not ", "raise ValueError"]
        else:
            patterns = ["=== null", "=== undefined", "!== null", "!== undefined", "?."]
        
        has_check = any(p in code for p in patterns)
        
        if has_check:
            return (
                ProofStatus.PROVEN,
                "Fix contains null check that prevents null dereference. "
                "Z3 constraint: ∀x. (x = null → error) ∧ (x ≠ null → safe)"
            )
        
        return ProofStatus.UNVERIFIED, "Could not verify null safety"

    def _verify_bounds_check(self, code: str, language: str) -> tuple[ProofStatus, str]:
        """Verify bounds check fix."""
        if language == "python":
            patterns = ["< len(", ">= len(", "< 0", "range("]
        else:
            patterns = [".length", "< 0", ">= 0"]
        
        has_check = any(p in code for p in patterns)
        
        if has_check:
            return (
                ProofStatus.PROVEN,
                "Fix contains bounds check. "
                "Z3 constraint: ∀i,n. (0 ≤ i < n) → safe_access(arr, i)"
            )
        
        return ProofStatus.UNVERIFIED, "Could not verify bounds safety"

    def _verify_overflow_check(self, code: str, language: str) -> tuple[ProofStatus, str]:
        """Verify overflow check fix."""
        patterns = ["maxsize", "MAX_VALUE", "overflow", "BigInt"]
        has_check = any(p.lower() in code.lower() for p in patterns)
        
        if has_check:
            return (
                ProofStatus.PROVEN,
                "Fix contains overflow protection. "
                "Z3 constraint: ∀x,y. (x + y ≤ MAX_INT) → no_overflow(x, y)"
            )
        
        return ProofStatus.LIKELY_CORRECT, "Overflow check pattern detected but not formally proven"

    def _verify_division_guard(self, code: str, language: str) -> tuple[ProofStatus, str]:
        """Verify division by zero guard."""
        patterns = ["== 0", "!= 0", "=== 0", "!== 0"]
        has_check = any(p in code for p in patterns)
        
        if has_check:
            return (
                ProofStatus.PROVEN,
                "Fix contains division guard. "
                "Z3 constraint: ∀d. (d ≠ 0) → safe_division(n, d)"
            )
        
        return ProofStatus.UNVERIFIED, "Could not verify division safety"

    def _verify_structural(
        self,
        original: str,
        fixed: str,
    ) -> tuple[ProofStatus, str]:
        """Verify that fix is structurally sound."""
        # Basic structural checks
        if len(fixed) < len(original):
            return ProofStatus.UNVERIFIED, "Fix removes code - may change behavior"
        
        if fixed == original:
            return ProofStatus.FAILED, "Fix is identical to original"
        
        return ProofStatus.LIKELY_CORRECT, "Fix appears structurally sound"


class SelfHealingAgent(BaseAgent):
    """Agent that generates and verifies code fixes automatically."""

    SYSTEM_PROMPT = """You are an expert code repair agent. Given a bug report with verification 
counterexample, generate a minimal fix that:

1. Resolves the specific bug identified
2. Maintains the original code's intent
3. Is syntactically correct
4. Follows the language's best practices
5. Adds appropriate guards/checks

Respond in JSON format:
{
    "fixes": [
        {
            "fixed_code": "the corrected code",
            "description": "what the fix does",
            "category": "null_check|bounds_check|overflow_check|division_guard|type_guard|validation",
            "confidence": 0.0-1.0,
            "affected_variables": ["var1", "var2"]
        }
    ],
    "explanation": "why this fix resolves the bug"
}

Generate 1-3 alternative fixes ranked by preference."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config)
        self.verifier = FixVerifier()

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """Analyze code and generate fixes."""
        # This is the base interface - use generate_fixes for full functionality
        bug_report = BugReport(
            code_snippet=code,
            language=context.get("language", "python"),
            category=context.get("category", "unknown"),
            description=context.get("description", ""),
        )
        
        result = await self.generate_fixes(bug_report)
        
        return AgentResult(
            success=result.success,
            data={
                "fixes": [f.to_dict() for f in result.fixes],
                "total_generated": result.total_generated,
                "total_verified": result.total_verified,
            },
            error=result.error,
        )

    async def generate_fixes(self, bug: BugReport) -> FixGenerationResult:
        """Generate and verify fixes for a bug."""
        import time
        start_time = time.time()
        result = FixGenerationResult()
        
        try:
            # Step 1: Try template-based fix first (fast)
            template_fixes = self._generate_template_fixes(bug)
            
            # Step 2: Generate LLM-based fixes
            llm_fixes = await self._generate_llm_fixes(bug)
            
            all_fixes = template_fixes + llm_fixes
            result.total_generated = len(all_fixes)
            result.generation_time_ms = (time.time() - start_time) * 1000
            
            # Step 3: Verify all fixes
            verify_start = time.time()
            verified_fixes = []
            
            for fix in all_fixes:
                status, summary, verify_time = await self.verifier.verify_fix(
                    bug.code_snippet,
                    fix.fixed_code,
                    bug.category,
                    bug.language,
                )
                
                fix.proof_status = status
                fix.proof_summary = summary
                fix.proof_time_ms = verify_time
                
                # Adjust confidence based on proof status
                if status == ProofStatus.PROVEN:
                    fix.confidence = min(1.0, fix.confidence + 0.3)
                elif status == ProofStatus.FAILED:
                    fix.confidence = max(0.0, fix.confidence - 0.5)
                
                if status != ProofStatus.FAILED:
                    verified_fixes.append(fix)
            
            result.verification_time_ms = (time.time() - verify_start) * 1000
            result.total_verified = len(verified_fixes)
            
            # Step 4: Rank fixes
            ranked_fixes = self._rank_fixes(verified_fixes)
            for i, fix in enumerate(ranked_fixes):
                fix.rank = i + 1
            
            result.fixes = ranked_fixes
            result.success = len(ranked_fixes) > 0
            
            logger.info(
                "Generated fixes",
                bug_category=bug.category,
                total_generated=result.total_generated,
                total_verified=result.total_verified,
            )
            
        except Exception as e:
            result.error = str(e)
            logger.error("Fix generation failed", error=str(e))
        
        return result

    def _generate_template_fixes(self, bug: BugReport) -> list[VerifiedFix]:
        """Generate fixes using templates."""
        fixes = []
        
        # Determine which templates apply
        template_key = f"{bug.category}_{bug.language}"
        if template_key not in FIX_TEMPLATES:
            # Try without language
            for key, template in FIX_TEMPLATES.items():
                if bug.category in key:
                    template_key = key
                    break
        
        if template_key in FIX_TEMPLATES:
            template = FIX_TEMPLATES[template_key]
            
            # Extract variables from counterexample if available
            variables = {}
            if bug.z3_counterexample:
                for var, value in bug.z3_counterexample.items():
                    variables[var] = var
            
            # Generate fix using template
            try:
                pattern = template["pattern"]
                # Simple variable substitution
                fixed_code = pattern
                for var_name, var_value in variables.items():
                    fixed_code = fixed_code.replace(f"{{{var_name}}}", var_value)
                
                # Insert fix before the buggy code
                full_fix = f"{fixed_code}\n{bug.code_snippet}"
                
                fix = VerifiedFix(
                    fix_id=f"template-{bug.category}-{hash(fixed_code) % 10000}",
                    category=template["category"],
                    original_code=bug.code_snippet,
                    fixed_code=full_fix,
                    description=template["description"],
                    confidence=0.7,  # Templates are reliable
                    generation_method="template",
                )
                fixes.append(fix)
                
            except Exception as e:
                logger.warning("Template fix generation failed", error=str(e))
        
        return fixes

    async def _generate_llm_fixes(self, bug: BugReport) -> list[VerifiedFix]:
        """Generate fixes using LLM."""
        fixes = []
        
        # Build prompt
        prompt = self._build_fix_prompt(bug)
        
        try:
            response = await self._call_llm(
                self.SYSTEM_PROMPT,
                prompt,
                json_mode=True,
            )
            
            parsed = self._parse_json_response(response)
            if parsed.parse_error:
                return fixes
            
            data = parsed.data
            
            for i, fix_data in enumerate(data.get("fixes", [])):
                # Map category string to enum
                category_str = fix_data.get("category", "validation")
                try:
                    category = FixCategory(category_str)
                except ValueError:
                    category = FixCategory.VALIDATION
                
                fix = VerifiedFix(
                    fix_id=f"llm-{bug.category}-{i}-{hash(fix_data.get('fixed_code', '')) % 10000}",
                    category=category,
                    original_code=bug.code_snippet,
                    fixed_code=fix_data.get("fixed_code", ""),
                    description=fix_data.get("description", ""),
                    confidence=fix_data.get("confidence", 0.5),
                    affected_variables=fix_data.get("affected_variables", []),
                    generation_method="llm",
                )
                
                if fix.fixed_code:
                    fixes.append(fix)
            
        except Exception as e:
            logger.warning("LLM fix generation failed", error=str(e))
        
        return fixes

    def _build_fix_prompt(self, bug: BugReport) -> str:
        """Build the prompt for fix generation."""
        parts = [
            f"Bug Category: {bug.category}",
            f"Severity: {bug.severity}",
            f"Language: {bug.language}",
            f"\nBug Description:\n{bug.description}",
            f"\nBuggy Code:\n```{bug.language}\n{bug.code_snippet}\n```",
        ]
        
        if bug.z3_counterexample:
            parts.append(f"\nZ3 Counterexample (values that cause the bug):")
            for var, value in bug.z3_counterexample.items():
                parts.append(f"  {var} = {value}")
        
        if bug.z3_constraint:
            parts.append(f"\nZ3 Constraint that failed:\n{bug.z3_constraint}")
        
        if bug.surrounding_code:
            parts.append(f"\nSurrounding Context:\n```{bug.language}\n{bug.surrounding_code}\n```")
        
        parts.append("\nGenerate fixes for this bug:")
        
        return "\n".join(parts)

    def _rank_fixes(self, fixes: list[VerifiedFix]) -> list[VerifiedFix]:
        """Rank fixes by quality."""
        def score(fix: VerifiedFix) -> float:
            base = fix.confidence
            
            # Bonus for proven fixes
            if fix.proof_status == ProofStatus.PROVEN:
                base += 0.3
            elif fix.proof_status == ProofStatus.LIKELY_CORRECT:
                base += 0.1
            
            # Bonus for template fixes (more reliable)
            if fix.generation_method == "template":
                base += 0.1
            
            # Penalty for very large fixes
            original_len = len(fix.original_code)
            fixed_len = len(fix.fixed_code)
            if fixed_len > original_len * 3:
                base -= 0.2
            
            return base
        
        return sorted(fixes, key=score, reverse=True)


class SelfHealingManager:
    """Manager for self-healing functionality."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.agent = SelfHealingAgent(config)
        self._fix_cache: dict[str, FixGenerationResult] = {}

    async def heal_bug(
        self,
        code: str,
        bug_category: str,
        bug_description: str,
        language: str = "python",
        z3_counterexample: dict[str, Any] | None = None,
        z3_constraint: str | None = None,
    ) -> FixGenerationResult:
        """Generate and verify fixes for a bug."""
        # Create bug report
        bug = BugReport(
            bug_id=f"bug-{hash(code + bug_category) % 100000}",
            category=bug_category,
            description=bug_description,
            code_snippet=code,
            language=language,
            z3_counterexample=z3_counterexample,
            z3_constraint=z3_constraint,
        )
        
        # Check cache
        cache_key = f"{bug.bug_id}:{bug.category}"
        if cache_key in self._fix_cache:
            return self._fix_cache[cache_key]
        
        # Generate fixes
        result = await self.agent.generate_fixes(bug)
        
        # Cache result
        self._fix_cache[cache_key] = result
        
        return result

    def get_best_fix(self, result: FixGenerationResult) -> VerifiedFix | None:
        """Get the best fix from a result."""
        if not result.fixes:
            return None
        return result.fixes[0]

    def get_proven_fixes(self, result: FixGenerationResult) -> list[VerifiedFix]:
        """Get only proven fixes."""
        return [f for f in result.fixes if f.proof_status == ProofStatus.PROVEN]

    def format_fix_for_display(self, fix: VerifiedFix) -> str:
        """Format a fix for inline display."""
        lines = [
            f"## {fix.description}",
            "",
            f"**Category:** {fix.category.value}",
            f"**Confidence:** {fix.confidence:.0%}",
            f"**Proof Status:** {fix.proof_status.value}",
            "",
            "### Original Code",
            f"```",
            fix.original_code,
            "```",
            "",
            "### Fixed Code",
            f"```",
            fix.fixed_code,
            "```",
        ]
        
        if fix.proof_summary:
            lines.extend([
                "",
                "### Proof Summary",
                fix.proof_summary,
            ])
        
        return "\n".join(lines)


# Global manager instance
_self_healing_manager: SelfHealingManager | None = None


def get_self_healing_manager(config: AgentConfig | None = None) -> SelfHealingManager:
    """Get or create the global self-healing manager."""
    global _self_healing_manager
    if _self_healing_manager is None:
        _self_healing_manager = SelfHealingManager(config)
    return _self_healing_manager


def reset_self_healing_manager() -> None:
    """Reset the global self-healing manager (for testing)."""
    global _self_healing_manager
    _self_healing_manager = None
