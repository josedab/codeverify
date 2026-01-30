"""Natural Language Invariant Specs - Convert English requirements to Z3 assertions.

Allows developers to write requirements in natural language that compile
to Z3 assertions and persist across PRs.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class InvariantType(str, Enum):
    """Type of invariant."""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    ASSERTION = "assertion"


class ValueConstraint(str, Enum):
    """Common value constraints."""
    POSITIVE = "positive"
    NON_NEGATIVE = "non_negative"
    NEGATIVE = "negative"
    NON_ZERO = "non_zero"
    NOT_NULL = "not_null"
    NOT_EMPTY = "not_empty"
    BOUNDED = "bounded"


@dataclass
class ParsedConstraint:
    """A parsed constraint from natural language."""
    variable: str
    constraint_type: ValueConstraint | str
    parameters: dict[str, Any] = field(default_factory=dict)
    original_text: str = ""


@dataclass
class NaturalLanguageInvariant:
    """An invariant expressed in natural language."""
    id: str
    text: str
    invariant_type: InvariantType
    scope: str  # function name, class name, or module
    parsed_constraints: list[ParsedConstraint] = field(default_factory=list)
    z3_formula: str | None = None
    smtlib_formula: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvariantCompilationResult:
    """Result of compiling natural language to Z3."""
    success: bool
    invariant: NaturalLanguageInvariant
    z3_code: str
    smtlib_formula: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# Pattern matching for common natural language constraints
NL_PATTERNS = [
    # Numeric constraints
    (r"(\w+)\s+(?:should|must|shall)\s+(?:be|remain)\s+positive", ValueConstraint.POSITIVE),
    (r"(\w+)\s+(?:should|must|shall)\s+(?:be|remain)\s+(?:non-negative|>= ?0|at least 0)", ValueConstraint.NON_NEGATIVE),
    (r"(\w+)\s+(?:should|must|shall)\s+(?:be|remain)\s+negative", ValueConstraint.NEGATIVE),
    (r"(\w+)\s+(?:should|must|shall)\s+(?:never|not)\s+(?:be|equal|become)\s+(?:zero|0)", ValueConstraint.NON_ZERO),
    
    # Null/None constraints
    (r"(\w+)\s+(?:should|must|shall)\s+(?:never|not)\s+(?:be|become)\s+(?:null|None|nil)", ValueConstraint.NOT_NULL),
    (r"(\w+)\s+(?:is|must be|should be)\s+(?:never|not)\s+(?:null|None|nil)", ValueConstraint.NOT_NULL),
    
    # Collection constraints
    (r"(\w+)\s+(?:should|must|shall)\s+(?:never|not)\s+(?:be|become)\s+empty", ValueConstraint.NOT_EMPTY),
    (r"(?:length|size|len)\s+of\s+(\w+)\s+(?:should|must|shall)\s+(?:be|remain)\s+(?:positive|> ?0|at least 1)", ValueConstraint.NOT_EMPTY),
    
    # Range constraints - captured separately with parameters
    (r"(\w+)\s+(?:should|must|shall)\s+(?:be|remain)\s+(?:between|in range)\s+(-?\d+(?:\.\d+)?)\s+(?:and|to)\s+(-?\d+(?:\.\d+)?)", "range"),
    (r"(\w+)\s+(?:should|must|shall)\s+(?:be|remain)\s+(?:less than|<)\s+(-?\d+(?:\.\d+)?)", "less_than"),
    (r"(\w+)\s+(?:should|must|shall)\s+(?:be|remain)\s+(?:greater than|>)\s+(-?\d+(?:\.\d+)?)", "greater_than"),
    (r"(\w+)\s+(?:should|must|shall)\s+(?:be|remain)\s+(?:at most|<=)\s+(-?\d+(?:\.\d+)?)", "at_most"),
    (r"(\w+)\s+(?:should|must|shall)\s+(?:be|remain)\s+(?:at least|>=)\s+(-?\d+(?:\.\d+)?)", "at_least"),
]

# Z3 code templates
Z3_TEMPLATES = {
    ValueConstraint.POSITIVE: "{var} > 0",
    ValueConstraint.NON_NEGATIVE: "{var} >= 0",
    ValueConstraint.NEGATIVE: "{var} < 0",
    ValueConstraint.NON_ZERO: "{var} != 0",
    ValueConstraint.NOT_NULL: "{var} != None",  # Will be adapted per language
    ValueConstraint.NOT_EMPTY: "len({var}) > 0",
    "range": "And({var} >= {min}, {var} <= {max})",
    "less_than": "{var} < {bound}",
    "greater_than": "{var} > {bound}",
    "at_most": "{var} <= {bound}",
    "at_least": "{var} >= {bound}",
}

# SMT-LIB templates
SMTLIB_TEMPLATES = {
    ValueConstraint.POSITIVE: "(> {var} 0)",
    ValueConstraint.NON_NEGATIVE: "(>= {var} 0)",
    ValueConstraint.NEGATIVE: "(< {var} 0)",
    ValueConstraint.NON_ZERO: "(not (= {var} 0))",
    ValueConstraint.NOT_NULL: "(not (= {var} null))",
    ValueConstraint.NOT_EMPTY: "(> (len {var}) 0)",
    "range": "(and (>= {var} {min}) (<= {var} {max}))",
    "less_than": "(< {var} {bound})",
    "greater_than": "(> {var} {bound})",
    "at_most": "(<= {var} {bound})",
    "at_least": "(>= {var} {bound})",
}


class NaturalLanguageParser:
    """
    Parses natural language constraints into structured form.
    """

    def parse(self, text: str) -> list[ParsedConstraint]:
        """Parse natural language text into constraints."""
        constraints = []
        text_lower = text.lower()
        
        for pattern, constraint_type in NL_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                constraint = self._extract_constraint(match, constraint_type, text)
                if constraint:
                    constraints.append(constraint)
        
        # Remove duplicates
        seen = set()
        unique_constraints = []
        for c in constraints:
            key = (c.variable, str(c.constraint_type), str(c.parameters))
            if key not in seen:
                seen.add(key)
                unique_constraints.append(c)
        
        return unique_constraints

    def _extract_constraint(
        self,
        match: re.Match,
        constraint_type: ValueConstraint | str,
        original_text: str,
    ) -> ParsedConstraint | None:
        """Extract constraint from regex match."""
        groups = match.groups()
        
        if not groups:
            return None
        
        variable = groups[0]
        parameters: dict[str, Any] = {}
        
        # Handle parameterized constraints
        if constraint_type == "range" and len(groups) >= 3:
            parameters["min"] = float(groups[1])
            parameters["max"] = float(groups[2])
        elif constraint_type in ("less_than", "greater_than", "at_most", "at_least") and len(groups) >= 2:
            parameters["bound"] = float(groups[1])
        
        return ParsedConstraint(
            variable=variable,
            constraint_type=constraint_type,
            parameters=parameters,
            original_text=match.group(0),
        )


class Z3Compiler:
    """
    Compiles parsed constraints to Z3 Python code and SMT-LIB.
    """

    def compile(
        self,
        constraints: list[ParsedConstraint],
        variable_types: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Compile constraints to Z3 code and SMT-LIB."""
        z3_parts = []
        smtlib_parts = []
        variable_types = variable_types or {}
        
        # Track declared variables
        declared_vars = set()
        z3_declarations = []
        smtlib_declarations = []
        
        for constraint in constraints:
            var = constraint.variable
            
            # Determine variable type
            var_type = variable_types.get(var, "Int")
            
            # Add declaration if needed
            if var not in declared_vars:
                declared_vars.add(var)
                z3_declarations.append(f"{var} = {var_type}('{var}')")
                smtlib_type = "Int" if var_type in ("Int", "int") else "Real"
                smtlib_declarations.append(f"(declare-const {var} {smtlib_type})")
            
            # Generate constraint
            z3_constraint = self._to_z3(constraint)
            smtlib_constraint = self._to_smtlib(constraint)
            
            if z3_constraint:
                z3_parts.append(z3_constraint)
            if smtlib_constraint:
                smtlib_parts.append(smtlib_constraint)
        
        # Build complete Z3 code
        z3_code = self._build_z3_code(z3_declarations, z3_parts)
        
        # Build complete SMT-LIB
        smtlib_code = self._build_smtlib_code(smtlib_declarations, smtlib_parts)
        
        return z3_code, smtlib_code

    def _to_z3(self, constraint: ParsedConstraint) -> str | None:
        """Convert constraint to Z3 Python code."""
        template = Z3_TEMPLATES.get(constraint.constraint_type)
        if not template:
            return None
        
        return template.format(
            var=constraint.variable,
            **constraint.parameters,
        )

    def _to_smtlib(self, constraint: ParsedConstraint) -> str | None:
        """Convert constraint to SMT-LIB format."""
        template = SMTLIB_TEMPLATES.get(constraint.constraint_type)
        if not template:
            return None
        
        return template.format(
            var=constraint.variable,
            **constraint.parameters,
        )

    def _build_z3_code(
        self,
        declarations: list[str],
        constraints: list[str],
    ) -> str:
        """Build complete Z3 Python code."""
        lines = [
            "from z3 import *",
            "",
            "# Variable declarations",
        ]
        lines.extend(declarations)
        lines.append("")
        lines.append("# Constraints")
        lines.append("solver = Solver()")
        
        for constraint in constraints:
            lines.append(f"solver.add({constraint})")
        
        lines.append("")
        lines.append("# Check satisfiability")
        lines.append("result = solver.check()")
        
        return "\n".join(lines)

    def _build_smtlib_code(
        self,
        declarations: list[str],
        constraints: list[str],
    ) -> str:
        """Build complete SMT-LIB code."""
        lines = ["; Auto-generated SMT-LIB from natural language"]
        lines.extend(declarations)
        lines.append("")
        
        for constraint in constraints:
            lines.append(f"(assert {constraint})")
        
        lines.append("")
        lines.append("(check-sat)")
        lines.append("(get-model)")
        
        return "\n".join(lines)


class NaturalLanguageInvariantAgent(BaseAgent):
    """
    Agent for converting natural language requirements to Z3 assertions.
    
    Uses pattern matching and optionally LLM for complex expressions.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the agent."""
        super().__init__(config)
        self._parser = NaturalLanguageParser()
        self._compiler = Z3Compiler()

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Process natural language invariants.
        
        Args:
            code: Not used - invariants come from context
            context: Contains:
                - invariant_text: Natural language requirement
                - scope: Function/class/module name
                - invariant_type: Type of invariant
                - variable_types: Optional variable type hints
                
        Returns:
            AgentResult with compiled Z3 code
        """
        try:
            invariant_text = context.get("invariant_text", "")
            scope = context.get("scope", "global")
            invariant_type = InvariantType(
                context.get("invariant_type", "invariant")
            )
            variable_types = context.get("variable_types", {})
            
            result = await self.compile_invariant(
                text=invariant_text,
                scope=scope,
                invariant_type=invariant_type,
                variable_types=variable_types,
            )
            
            return AgentResult(
                success=result.success,
                data={
                    "invariant_id": result.invariant.id,
                    "z3_code": result.z3_code,
                    "smtlib": result.smtlib_formula,
                    "constraints": [
                        {
                            "variable": c.variable,
                            "type": str(c.constraint_type),
                            "parameters": c.parameters,
                            "original": c.original_text,
                        }
                        for c in result.invariant.parsed_constraints
                    ],
                    "confidence": result.invariant.confidence,
                    "warnings": result.warnings,
                },
                error="; ".join(result.errors) if result.errors else None,
            )
            
        except Exception as e:
            logger.error("Invariant compilation failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    async def compile_invariant(
        self,
        text: str,
        scope: str,
        invariant_type: InvariantType,
        variable_types: dict[str, str] | None = None,
        use_llm_fallback: bool = True,
    ) -> InvariantCompilationResult:
        """Compile a natural language invariant to Z3."""
        import uuid
        
        invariant_id = str(uuid.uuid4())[:8]
        warnings: list[str] = []
        errors: list[str] = []
        
        # Parse natural language
        constraints = self._parser.parse(text)
        
        # If no constraints found, try LLM
        if not constraints and use_llm_fallback:
            constraints, llm_warnings = await self._parse_with_llm(text)
            warnings.extend(llm_warnings)
        
        if not constraints:
            errors.append("Could not parse any constraints from the text")
            return InvariantCompilationResult(
                success=False,
                invariant=NaturalLanguageInvariant(
                    id=invariant_id,
                    text=text,
                    invariant_type=invariant_type,
                    scope=scope,
                ),
                z3_code="",
                smtlib_formula="",
                errors=errors,
            )
        
        # Compile to Z3
        try:
            z3_code, smtlib = self._compiler.compile(constraints, variable_types)
        except Exception as e:
            errors.append(f"Compilation error: {str(e)}")
            return InvariantCompilationResult(
                success=False,
                invariant=NaturalLanguageInvariant(
                    id=invariant_id,
                    text=text,
                    invariant_type=invariant_type,
                    scope=scope,
                    parsed_constraints=constraints,
                ),
                z3_code="",
                smtlib_formula="",
                errors=errors,
            )
        
        # Calculate confidence
        confidence = self._calculate_confidence(constraints, text)
        
        invariant = NaturalLanguageInvariant(
            id=invariant_id,
            text=text,
            invariant_type=invariant_type,
            scope=scope,
            parsed_constraints=constraints,
            z3_formula=z3_code,
            smtlib_formula=smtlib,
            confidence=confidence,
        )
        
        logger.info(
            "Compiled natural language invariant",
            invariant_id=invariant_id,
            constraint_count=len(constraints),
            confidence=confidence,
        )
        
        return InvariantCompilationResult(
            success=True,
            invariant=invariant,
            z3_code=z3_code,
            smtlib_formula=smtlib,
            warnings=warnings,
        )

    async def _parse_with_llm(
        self,
        text: str,
    ) -> tuple[list[ParsedConstraint], list[str]]:
        """Use LLM to parse complex natural language."""
        warnings = []
        constraints = []
        
        system_prompt = """You are a formal verification expert. Extract constraints from natural language.

Output JSON with this structure:
{
    "constraints": [
        {
            "variable": "variable_name",
            "constraint_type": "positive|non_negative|negative|non_zero|not_null|not_empty|range|less_than|greater_than|at_most|at_least",
            "parameters": {"min": 0, "max": 100} // only for range, bounds
        }
    ]
}

Examples:
- "balance should never be negative" -> {"variable": "balance", "constraint_type": "non_negative", "parameters": {}}
- "age must be between 0 and 150" -> {"variable": "age", "constraint_type": "range", "parameters": {"min": 0, "max": 150}}
"""

        try:
            response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=f"Extract constraints from: {text}",
                json_mode=True,
            )
            
            import json
            result = json.loads(response["content"])
            
            for c in result.get("constraints", []):
                constraint_type = c.get("constraint_type", "")
                
                # Map to ValueConstraint if possible
                try:
                    ctype = ValueConstraint(constraint_type)
                except ValueError:
                    ctype = constraint_type
                
                constraints.append(ParsedConstraint(
                    variable=c.get("variable", ""),
                    constraint_type=ctype,
                    parameters=c.get("parameters", {}),
                    original_text=text,
                ))
                
        except Exception as e:
            warnings.append(f"LLM parsing failed: {str(e)}")
        
        return constraints, warnings

    def _calculate_confidence(
        self,
        constraints: list[ParsedConstraint],
        original_text: str,
    ) -> float:
        """Calculate confidence score for the compilation."""
        if not constraints:
            return 0.0
        
        # Start with base confidence
        confidence = 0.7
        
        # Boost for pattern matches vs LLM
        pattern_matches = sum(
            1 for c in constraints if c.original_text in original_text.lower()
        )
        if pattern_matches == len(constraints):
            confidence += 0.2
        
        # Reduce for complex expressions
        if len(constraints) > 5:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)


class InvariantStore:
    """
    Storage for natural language invariants.
    
    Persists invariants and their compiled forms for reuse.
    """

    def __init__(self) -> None:
        """Initialize the store."""
        self._invariants: dict[str, NaturalLanguageInvariant] = {}
        self._by_scope: dict[str, list[str]] = {}

    def add(self, invariant: NaturalLanguageInvariant) -> None:
        """Add an invariant to the store."""
        self._invariants[invariant.id] = invariant
        
        if invariant.scope not in self._by_scope:
            self._by_scope[invariant.scope] = []
        self._by_scope[invariant.scope].append(invariant.id)

    def get(self, invariant_id: str) -> NaturalLanguageInvariant | None:
        """Get an invariant by ID."""
        return self._invariants.get(invariant_id)

    def get_for_scope(self, scope: str) -> list[NaturalLanguageInvariant]:
        """Get all invariants for a scope."""
        ids = self._by_scope.get(scope, [])
        return [self._invariants[id] for id in ids if id in self._invariants]

    def get_all(self) -> list[NaturalLanguageInvariant]:
        """Get all invariants."""
        return list(self._invariants.values())

    def remove(self, invariant_id: str) -> bool:
        """Remove an invariant."""
        if invariant_id not in self._invariants:
            return False
        
        invariant = self._invariants.pop(invariant_id)
        
        if invariant.scope in self._by_scope:
            self._by_scope[invariant.scope] = [
                id for id in self._by_scope[invariant.scope] if id != invariant_id
            ]
        
        return True

    def export(self) -> list[dict[str, Any]]:
        """Export all invariants."""
        return [
            {
                "id": inv.id,
                "text": inv.text,
                "invariant_type": inv.invariant_type.value,
                "scope": inv.scope,
                "z3_formula": inv.z3_formula,
                "smtlib_formula": inv.smtlib_formula,
                "confidence": inv.confidence,
                "constraints": [
                    {
                        "variable": c.variable,
                        "type": str(c.constraint_type),
                        "parameters": c.parameters,
                    }
                    for c in inv.parsed_constraints
                ],
            }
            for inv in self._invariants.values()
        ]

    def import_invariants(self, data: list[dict[str, Any]]) -> int:
        """Import invariants from exported data."""
        count = 0
        for item in data:
            try:
                constraints = [
                    ParsedConstraint(
                        variable=c["variable"],
                        constraint_type=c["type"],
                        parameters=c.get("parameters", {}),
                    )
                    for c in item.get("constraints", [])
                ]
                
                invariant = NaturalLanguageInvariant(
                    id=item["id"],
                    text=item["text"],
                    invariant_type=InvariantType(item["invariant_type"]),
                    scope=item["scope"],
                    parsed_constraints=constraints,
                    z3_formula=item.get("z3_formula"),
                    smtlib_formula=item.get("smtlib_formula"),
                    confidence=item.get("confidence", 0.0),
                )
                self.add(invariant)
                count += 1
            except Exception as e:
                logger.warning("Failed to import invariant", error=str(e))
        
        return count


def parse_codeverify_comments(code: str) -> list[tuple[str, str, str]]:
    """
    Parse CodeVerify invariant comments from source code.
    
    Supports:
        # @codeverify: balance should never be negative
        // @invariant: count must be positive
        /* @precondition: input should not be null */
    
    Returns list of (invariant_type, scope, text) tuples.
    """
    results = []
    
    # Pattern for various comment styles
    patterns = [
        # Python/Shell style
        r'#\s*@(codeverify|invariant|precondition|postcondition)(?:\((\w+)\))?:\s*(.+?)$',
        # C/JS/Java style single line
        r'//\s*@(codeverify|invariant|precondition|postcondition)(?:\((\w+)\))?:\s*(.+?)$',
        # C/JS/Java style multi-line
        r'/\*\s*@(codeverify|invariant|precondition|postcondition)(?:\((\w+)\))?:\s*(.+?)\s*\*/',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, code, re.MULTILINE):
            inv_type = match.group(1)
            scope = match.group(2) or "global"
            text = match.group(3).strip()
            
            # Map annotation type to InvariantType
            type_map = {
                "codeverify": "invariant",
                "invariant": "invariant",
                "precondition": "precondition",
                "postcondition": "postcondition",
            }
            
            results.append((type_map.get(inv_type, "invariant"), scope, text))
    
    return results
