"""Specification Generator Agent - LLM-powered formal specification inference.

This agent provides:
- Auto-generation of function contracts from code + documentation
- Pre/post condition inference using LLM
- Invariant detection for loops and classes
- Specification validation via Z3
- Interactive refinement with counterexample feedback
- Specification coverage metrics
"""

import ast
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from .base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class SpecificationType(str, Enum):
    """Types of specifications."""
    
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    LOOP_INVARIANT = "loop_invariant"
    CLASS_INVARIANT = "class_invariant"
    TYPE_CONSTRAINT = "type_constraint"
    ASSERTION = "assertion"


class SpecificationSource(str, Enum):
    """Source of specification."""
    
    INFERRED = "inferred"  # AI-inferred from code
    DOCUMENTED = "documented"  # Extracted from docstring
    USER_DEFINED = "user_defined"  # User-provided
    TEMPLATE = "template"  # From pattern template


@dataclass
class TypeInfo:
    """Type information for a parameter or return value."""
    
    name: str
    python_type: str | None = None
    z3_sort: str | None = None
    constraints: list[str] = field(default_factory=list)
    is_nullable: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "python_type": self.python_type,
            "z3_sort": self.z3_sort,
            "constraints": self.constraints,
            "is_nullable": self.is_nullable,
        }


@dataclass
class GeneratedSpec:
    """A generated formal specification."""
    
    id: str
    spec_type: SpecificationType
    source: SpecificationSource
    
    # Natural language
    description: str
    
    # Formal representations
    z3_expr: str | None = None
    smt_lib: str | None = None
    python_assert: str | None = None
    
    # Context
    function_name: str | None = None
    class_name: str | None = None
    line_number: int | None = None
    
    # Variables
    variables: list[TypeInfo] = field(default_factory=list)
    
    # Quality metrics
    confidence: float = 0.8
    validated: bool = False
    validation_result: str | None = None  # "valid", "invalid", "timeout"
    counterexample: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "spec_type": self.spec_type.value,
            "source": self.source.value,
            "description": self.description,
            "z3_expr": self.z3_expr,
            "smt_lib": self.smt_lib,
            "python_assert": self.python_assert,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "line_number": self.line_number,
            "variables": [v.to_dict() for v in self.variables],
            "confidence": self.confidence,
            "validated": self.validated,
            "validation_result": self.validation_result,
            "counterexample": self.counterexample,
        }


@dataclass
class FunctionContract:
    """Complete contract for a function."""
    
    function_name: str
    class_name: str | None = None
    
    # Signature
    parameters: list[TypeInfo] = field(default_factory=list)
    return_type: TypeInfo | None = None
    
    # Specifications
    preconditions: list[GeneratedSpec] = field(default_factory=list)
    postconditions: list[GeneratedSpec] = field(default_factory=list)
    invariants: list[GeneratedSpec] = field(default_factory=list)
    
    # Overall metrics
    coverage_score: float = 0.0
    confidence_score: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "class_name": self.class_name,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type.to_dict() if self.return_type else None,
            "preconditions": [s.to_dict() for s in self.preconditions],
            "postconditions": [s.to_dict() for s in self.postconditions],
            "invariants": [s.to_dict() for s in self.invariants],
            "coverage_score": self.coverage_score,
            "confidence_score": self.confidence_score,
        }


@dataclass
class ClassInvariant:
    """Invariants for a class."""
    
    class_name: str
    invariants: list[GeneratedSpec] = field(default_factory=list)
    
    # Methods that must maintain invariants
    methods: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "invariants": [s.to_dict() for s in self.invariants],
            "methods": self.methods,
        }


class SpecificationGeneratorAgent(BaseAgent):
    """AI agent for generating formal specifications from code.
    
    Features:
    - LLM-powered inference from code + docstrings
    - Type-guided specification templates
    - Z3 validation of generated specs
    - Counterexample-driven refinement
    - Coverage metrics
    """
    
    SYSTEM_PROMPT = """You are an expert in formal methods and program verification.
Your task is to generate formal specifications (pre/post conditions, invariants) from code.

RESPOND ONLY WITH VALID JSON in this exact format:
{
    "preconditions": [
        {
            "description": "Natural language description",
            "z3_expr": "Z3 Python expression (e.g., 'x > 0')",
            "python_assert": "Python assertion (e.g., 'assert x > 0')",
            "variables": ["x"],
            "confidence": 0.9
        }
    ],
    "postconditions": [
        {
            "description": "Description of what should be true after execution",
            "z3_expr": "Z3 expression for return value/state",
            "python_assert": "Python assertion",
            "variables": ["result", "x"],
            "confidence": 0.85
        }
    ],
    "invariants": [
        {
            "description": "Loop or state invariant",
            "z3_expr": "Z3 expression",
            "python_assert": "Python assertion",
            "variables": ["i", "acc"],
            "confidence": 0.8
        }
    ],
    "type_constraints": [
        {
            "variable": "x",
            "python_type": "int",
            "z3_sort": "Int",
            "constraints": ["x >= 0"]
        }
    ]
}

GUIDELINES:
1. Infer preconditions from:
   - Parameter types and type annotations
   - Docstring @param descriptions
   - Guard conditions in code
   - Implicit assumptions (null checks, bounds)

2. Infer postconditions from:
   - Return type annotation
   - Docstring @returns/@raises
   - Observable effects on state
   - Relationship between input and output

3. Infer invariants from:
   - Loop patterns (accumulator, counter, etc.)
   - Class state consistency requirements
   - Data structure properties (sorted, non-empty, etc.)

4. For Z3 expressions, use:
   - Basic comparisons: >, <, >=, <=, ==, !=
   - Logical operators: And(), Or(), Not(), Implies()
   - Arithmetic: +, -, *, /, %
   - Special: ForAll(), Exists(), If()

5. Only generate specifications you're confident about (>0.7)
6. Include the 'result' variable for postconditions referring to return value
"""

    def __init__(self, config: AgentConfig | None = None):
        super().__init__(config)
        self._spec_cache: dict[str, list[GeneratedSpec]] = {}
        self._validation_cache: dict[str, str] = {}  # spec_hash -> validation_result
    
    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """Analyze code and generate specifications."""
        start_time = time.time()
        
        try:
            # Parse code to extract functions and classes
            functions, classes = self._parse_code_structure(code, context.get("language", "python"))
            
            # Generate specifications for each function
            contracts = []
            for func_info in functions:
                contract = await self._generate_function_contract(func_info, code, context)
                contracts.append(contract)
            
            # Generate class invariants
            class_invariants = []
            for class_info in classes:
                invariant = await self._generate_class_invariants(class_info, code, context)
                class_invariants.append(invariant)
            
            # Calculate coverage
            coverage = self._calculate_coverage(contracts, class_invariants)
            
            return AgentResult(
                success=True,
                data={
                    "contracts": [c.to_dict() for c in contracts],
                    "class_invariants": [i.to_dict() for i in class_invariants],
                    "coverage": coverage,
                    "total_specs": sum(
                        len(c.preconditions) + len(c.postconditions) + len(c.invariants)
                        for c in contracts
                    ) + sum(len(i.invariants) for i in class_invariants),
                },
                latency_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            logger.error("Specification generation failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    async def generate_for_function(
        self,
        code: str,
        function_name: str,
        docstring: str | None = None,
        language: str = "python",
    ) -> FunctionContract:
        """Generate specifications for a single function."""
        context = {
            "language": language,
            "function_name": function_name,
            "docstring": docstring,
        }
        
        # Find function in code
        functions, _ = self._parse_code_structure(code, language)
        func_info = next((f for f in functions if f["name"] == function_name), None)
        
        if func_info:
            return await self._generate_function_contract(func_info, code, context)
        
        # If function not found, try to generate from the whole code
        return await self._generate_function_contract(
            {"name": function_name, "code": code, "line_start": 1},
            code,
            context,
        )
    
    async def validate_specification(
        self,
        spec: GeneratedSpec,
        code: str,
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Validate a specification against the code using Z3.
        
        Returns:
            Tuple of (is_valid, result_message, counterexample)
        """
        if not spec.z3_expr:
            return False, "No Z3 expression", None
        
        # Check cache
        spec_hash = hashlib.md5(f"{spec.z3_expr}:{code}".encode()).hexdigest()
        if spec_hash in self._validation_cache:
            cached = self._validation_cache[spec_hash]
            return cached == "valid", cached, None
        
        try:
            # Try to import Z3 and verify
            from z3 import And, Int, Not, Or, Solver, sat, unsat
            
            solver = Solver()
            solver.set("timeout", 5000)  # 5 second timeout
            
            # Parse the Z3 expression
            # This is simplified - production would properly parse
            local_vars: dict[str, Any] = {"And": And, "Or": Or, "Not": Not, "Int": Int}
            
            # Create variables for each variable in spec
            for var in spec.variables:
                local_vars[var.name] = Int(var.name)
            
            # Try to evaluate the expression
            try:
                z3_constraint = eval(spec.z3_expr, {"__builtins__": {}}, local_vars)
                
                # Check if negation is satisfiable (finding counterexample)
                solver.add(Not(z3_constraint))
                result = solver.check()
                
                if result == unsat:
                    # No counterexample found - spec is valid
                    self._validation_cache[spec_hash] = "valid"
                    spec.validated = True
                    spec.validation_result = "valid"
                    return True, "Specification is valid", None
                    
                elif result == sat:
                    # Found counterexample
                    model = solver.model()
                    counterexample = {str(d): str(model[d]) for d in model.decls()}
                    
                    self._validation_cache[spec_hash] = "invalid"
                    spec.validated = True
                    spec.validation_result = "invalid"
                    spec.counterexample = counterexample
                    return False, "Counterexample found", counterexample
                    
                else:
                    self._validation_cache[spec_hash] = "timeout"
                    spec.validation_result = "timeout"
                    return False, "Verification timeout", None
                    
            except Exception as eval_error:
                logger.warning("Failed to evaluate Z3 expression", error=str(eval_error))
                return False, f"Expression error: {eval_error}", None
                
        except ImportError:
            logger.warning("Z3 not available for validation")
            return False, "Z3 not available", None
    
    async def refine_specification(
        self,
        spec: GeneratedSpec,
        counterexample: dict[str, Any],
        user_feedback: str | None = None,
    ) -> GeneratedSpec:
        """Refine a specification based on counterexample or user feedback.
        
        Args:
            spec: The original specification
            counterexample: Counterexample values from validation
            user_feedback: Optional user feedback/correction
            
        Returns:
            Refined specification
        """
        prompt = f"""The following specification was found to be invalid:

Specification: {spec.description}
Z3 Expression: {spec.z3_expr}

Counterexample that violates the specification:
{json.dumps(counterexample, indent=2)}

{f"User feedback: {user_feedback}" if user_feedback else ""}

Please provide a REFINED specification that:
1. Accounts for the counterexample
2. Is more precise about the conditions
3. Maintains the original intent

Respond with JSON containing:
{{
    "description": "Refined description",
    "z3_expr": "Refined Z3 expression",
    "python_assert": "Refined Python assertion",
    "confidence": 0.85
}}
"""
        
        try:
            response = await self._call_llm(
                self.SYSTEM_PROMPT,
                prompt,
                json_mode=True,
            )
            
            data = json.loads(response["content"])
            
            # Create refined spec
            refined = GeneratedSpec(
                id=f"{spec.id}_refined",
                spec_type=spec.spec_type,
                source=SpecificationSource.USER_DEFINED if user_feedback else SpecificationSource.INFERRED,
                description=data.get("description", spec.description),
                z3_expr=data.get("z3_expr", spec.z3_expr),
                python_assert=data.get("python_assert", spec.python_assert),
                function_name=spec.function_name,
                class_name=spec.class_name,
                line_number=spec.line_number,
                variables=spec.variables,
                confidence=data.get("confidence", 0.7),
            )
            
            return refined
            
        except Exception as e:
            logger.error("Failed to refine specification", error=str(e))
            return spec
    
    def _parse_code_structure(
        self,
        code: str,
        language: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse code to extract function and class information."""
        functions = []
        classes = []
        
        if language == "python":
            try:
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        func_info = {
                            "name": node.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno or node.lineno,
                            "params": [],
                            "return_annotation": None,
                            "docstring": ast.get_docstring(node),
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                        }
                        
                        # Extract parameters
                        for arg in node.args.args:
                            param = {"name": arg.arg, "type": None}
                            if arg.annotation:
                                try:
                                    param["type"] = ast.unparse(arg.annotation)
                                except Exception:
                                    pass
                            func_info["params"].append(param)
                        
                        # Extract return type
                        if node.returns:
                            try:
                                func_info["return_annotation"] = ast.unparse(node.returns)
                            except Exception:
                                pass
                        
                        # Get the function code
                        lines = code.split("\n")
                        func_info["code"] = "\n".join(lines[node.lineno - 1:node.end_lineno or node.lineno])
                        
                        functions.append(func_info)
                    
                    elif isinstance(node, ast.ClassDef):
                        class_info = {
                            "name": node.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno or node.lineno,
                            "docstring": ast.get_docstring(node),
                            "methods": [],
                            "attributes": [],
                        }
                        
                        # Extract methods
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                class_info["methods"].append(item.name)
                            elif isinstance(item, ast.Assign):
                                for target in item.targets:
                                    if isinstance(target, ast.Name):
                                        class_info["attributes"].append(target.id)
                        
                        classes.append(class_info)
                        
            except SyntaxError:
                # Fall back to regex-based parsing
                functions = self._parse_functions_regex(code)
                classes = self._parse_classes_regex(code)
        
        elif language in ("typescript", "javascript"):
            functions = self._parse_ts_functions(code)
            classes = self._parse_ts_classes(code)
        
        return functions, classes
    
    def _parse_functions_regex(self, code: str) -> list[dict[str, Any]]:
        """Regex-based function parsing fallback."""
        functions = []
        pattern = r"(async\s+)?def\s+(\w+)\s*\(([^)]*)\)"
        
        for match in re.finditer(pattern, code):
            func_info = {
                "name": match.group(2),
                "is_async": match.group(1) is not None,
                "params": [],
                "line_start": code[:match.start()].count("\n") + 1,
            }
            
            # Parse parameters
            params_str = match.group(3)
            for param in params_str.split(","):
                param = param.strip()
                if param and param != "self":
                    name = param.split(":")[0].split("=")[0].strip()
                    func_info["params"].append({"name": name, "type": None})
            
            functions.append(func_info)
        
        return functions
    
    def _parse_classes_regex(self, code: str) -> list[dict[str, Any]]:
        """Regex-based class parsing fallback."""
        classes = []
        pattern = r"class\s+(\w+)"
        
        for match in re.finditer(pattern, code):
            classes.append({
                "name": match.group(1),
                "line_start": code[:match.start()].count("\n") + 1,
                "methods": [],
                "attributes": [],
            })
        
        return classes
    
    def _parse_ts_functions(self, code: str) -> list[dict[str, Any]]:
        """Parse TypeScript/JavaScript functions."""
        functions = []
        pattern = r"(async\s+)?function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(async\s*)?\("
        
        for match in re.finditer(pattern, code):
            name = match.group(2) or match.group(3)
            if name:
                functions.append({
                    "name": name,
                    "is_async": bool(match.group(1) or match.group(4)),
                    "params": [],
                    "line_start": code[:match.start()].count("\n") + 1,
                })
        
        return functions
    
    def _parse_ts_classes(self, code: str) -> list[dict[str, Any]]:
        """Parse TypeScript/JavaScript classes."""
        classes = []
        pattern = r"class\s+(\w+)"
        
        for match in re.finditer(pattern, code):
            classes.append({
                "name": match.group(1),
                "line_start": code[:match.start()].count("\n") + 1,
                "methods": [],
                "attributes": [],
            })
        
        return classes
    
    async def _generate_function_contract(
        self,
        func_info: dict[str, Any],
        full_code: str,
        context: dict[str, Any],
    ) -> FunctionContract:
        """Generate a complete contract for a function."""
        function_name = func_info.get("name", "unknown")
        
        # Check cache
        cache_key = hashlib.md5(f"{function_name}:{func_info.get('code', full_code)[:500]}".encode()).hexdigest()
        if cache_key in self._spec_cache:
            # Rebuild contract from cached specs
            cached_specs = self._spec_cache[cache_key]
            contract = FunctionContract(function_name=function_name)
            for spec in cached_specs:
                if spec.spec_type == SpecificationType.PRECONDITION:
                    contract.preconditions.append(spec)
                elif spec.spec_type == SpecificationType.POSTCONDITION:
                    contract.postconditions.append(spec)
                else:
                    contract.invariants.append(spec)
            return contract
        
        # Build prompt
        prompt = self._build_generation_prompt(func_info, context)
        
        try:
            response = await self._call_llm(
                self.SYSTEM_PROMPT,
                prompt,
                json_mode=True,
            )
            
            data = json.loads(response["content"])
            
            # Parse response into specs
            contract = FunctionContract(
                function_name=function_name,
                class_name=context.get("class_name"),
            )
            
            # Parse parameters
            for param in func_info.get("params", []):
                type_info = TypeInfo(
                    name=param["name"],
                    python_type=param.get("type"),
                )
                contract.parameters.append(type_info)
            
            # Parse type constraints
            type_constraints = data.get("type_constraints", [])
            for tc in type_constraints:
                for param in contract.parameters:
                    if param.name == tc.get("variable"):
                        param.z3_sort = tc.get("z3_sort")
                        param.constraints = tc.get("constraints", [])
            
            # Parse preconditions
            for pre in data.get("preconditions", []):
                spec = GeneratedSpec(
                    id=f"{function_name}_pre_{len(contract.preconditions)}",
                    spec_type=SpecificationType.PRECONDITION,
                    source=SpecificationSource.INFERRED,
                    description=pre.get("description", ""),
                    z3_expr=pre.get("z3_expr"),
                    python_assert=pre.get("python_assert"),
                    function_name=function_name,
                    variables=[TypeInfo(name=v) for v in pre.get("variables", [])],
                    confidence=pre.get("confidence", 0.8),
                )
                contract.preconditions.append(spec)
            
            # Parse postconditions
            for post in data.get("postconditions", []):
                spec = GeneratedSpec(
                    id=f"{function_name}_post_{len(contract.postconditions)}",
                    spec_type=SpecificationType.POSTCONDITION,
                    source=SpecificationSource.INFERRED,
                    description=post.get("description", ""),
                    z3_expr=post.get("z3_expr"),
                    python_assert=post.get("python_assert"),
                    function_name=function_name,
                    variables=[TypeInfo(name=v) for v in post.get("variables", [])],
                    confidence=post.get("confidence", 0.8),
                )
                contract.postconditions.append(spec)
            
            # Parse invariants
            for inv in data.get("invariants", []):
                spec = GeneratedSpec(
                    id=f"{function_name}_inv_{len(contract.invariants)}",
                    spec_type=SpecificationType.INVARIANT,
                    source=SpecificationSource.INFERRED,
                    description=inv.get("description", ""),
                    z3_expr=inv.get("z3_expr"),
                    python_assert=inv.get("python_assert"),
                    function_name=function_name,
                    variables=[TypeInfo(name=v) for v in inv.get("variables", [])],
                    confidence=inv.get("confidence", 0.8),
                )
                contract.invariants.append(spec)
            
            # Calculate scores
            contract.coverage_score = self._calculate_function_coverage(contract, func_info)
            contract.confidence_score = self._calculate_confidence(contract)
            
            # Cache the specs
            all_specs = contract.preconditions + contract.postconditions + contract.invariants
            self._spec_cache[cache_key] = all_specs
            
            logger.info(
                "Generated function contract",
                function=function_name,
                preconditions=len(contract.preconditions),
                postconditions=len(contract.postconditions),
                invariants=len(contract.invariants),
            )
            
            return contract
            
        except Exception as e:
            logger.error("Failed to generate contract", function=function_name, error=str(e))
            return FunctionContract(function_name=function_name)
    
    async def _generate_class_invariants(
        self,
        class_info: dict[str, Any],
        full_code: str,
        context: dict[str, Any],
    ) -> ClassInvariant:
        """Generate invariants for a class."""
        class_name = class_info.get("name", "unknown")
        
        prompt = f"""Generate class invariants for the following class:

Class Name: {class_name}
Attributes: {class_info.get('attributes', [])}
Methods: {class_info.get('methods', [])}
Docstring: {class_info.get('docstring', 'None')}

Return JSON with class invariants that must hold across all method calls.
"""
        
        try:
            response = await self._call_llm(
                self.SYSTEM_PROMPT,
                prompt,
                json_mode=True,
            )
            
            data = json.loads(response["content"])
            
            class_inv = ClassInvariant(
                class_name=class_name,
                methods=class_info.get("methods", []),
            )
            
            for inv in data.get("invariants", []):
                spec = GeneratedSpec(
                    id=f"{class_name}_class_inv_{len(class_inv.invariants)}",
                    spec_type=SpecificationType.CLASS_INVARIANT,
                    source=SpecificationSource.INFERRED,
                    description=inv.get("description", ""),
                    z3_expr=inv.get("z3_expr"),
                    python_assert=inv.get("python_assert"),
                    class_name=class_name,
                    variables=[TypeInfo(name=v) for v in inv.get("variables", [])],
                    confidence=inv.get("confidence", 0.8),
                )
                class_inv.invariants.append(spec)
            
            return class_inv
            
        except Exception as e:
            logger.error("Failed to generate class invariants", class_name=class_name, error=str(e))
            return ClassInvariant(class_name=class_name)
    
    def _build_generation_prompt(
        self,
        func_info: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        """Build the prompt for specification generation."""
        parts = [
            f"Generate formal specifications for the following function:",
            f"",
            f"Function Name: {func_info.get('name', 'unknown')}",
        ]
        
        # Parameters
        params = func_info.get("params", [])
        if params:
            parts.append(f"Parameters:")
            for p in params:
                type_str = f": {p['type']}" if p.get('type') else ""
                parts.append(f"  - {p['name']}{type_str}")
        
        # Return type
        if func_info.get("return_annotation"):
            parts.append(f"Return Type: {func_info['return_annotation']}")
        
        # Docstring
        if func_info.get("docstring"):
            parts.append(f"")
            parts.append(f"Docstring:")
            parts.append(func_info["docstring"])
        
        # Code
        if func_info.get("code"):
            parts.append(f"")
            parts.append(f"Code:")
            parts.append("```")
            parts.append(func_info["code"][:1500])  # Limit code length
            parts.append("```")
        
        return "\n".join(parts)
    
    def _calculate_function_coverage(
        self,
        contract: FunctionContract,
        func_info: dict[str, Any],
    ) -> float:
        """Calculate specification coverage for a function."""
        total_aspects = 0
        covered_aspects = 0
        
        # Parameters should have preconditions
        total_aspects += len(contract.parameters)
        covered_params = set()
        for pre in contract.preconditions:
            for var in pre.variables:
                covered_params.add(var.name)
        covered_aspects += len([p for p in contract.parameters if p.name in covered_params])
        
        # Should have at least one postcondition
        total_aspects += 1
        if contract.postconditions:
            covered_aspects += 1
        
        # Loops should have invariants (estimated)
        code = func_info.get("code", "")
        loop_count = code.count("for ") + code.count("while ")
        total_aspects += loop_count
        covered_aspects += min(len(contract.invariants), loop_count)
        
        return covered_aspects / total_aspects if total_aspects > 0 else 0.0
    
    def _calculate_confidence(self, contract: FunctionContract) -> float:
        """Calculate overall confidence score for a contract."""
        all_specs = contract.preconditions + contract.postconditions + contract.invariants
        if not all_specs:
            return 0.0
        
        return sum(s.confidence for s in all_specs) / len(all_specs)
    
    def _calculate_coverage(
        self,
        contracts: list[FunctionContract],
        class_invariants: list[ClassInvariant],
    ) -> dict[str, Any]:
        """Calculate overall specification coverage."""
        if not contracts and not class_invariants:
            return {
                "overall": 0.0,
                "functions_covered": 0,
                "functions_total": 0,
                "classes_covered": 0,
                "classes_total": 0,
            }
        
        functions_with_specs = sum(
            1 for c in contracts
            if c.preconditions or c.postconditions or c.invariants
        )
        classes_with_specs = sum(1 for ci in class_invariants if ci.invariants)
        
        function_coverage = functions_with_specs / len(contracts) if contracts else 0.0
        class_coverage = classes_with_specs / len(class_invariants) if class_invariants else 0.0
        
        overall = (function_coverage + class_coverage) / 2 if class_invariants else function_coverage
        
        return {
            "overall": overall,
            "functions_covered": functions_with_specs,
            "functions_total": len(contracts),
            "classes_covered": classes_with_specs,
            "classes_total": len(class_invariants),
            "avg_function_coverage": sum(c.coverage_score for c in contracts) / len(contracts) if contracts else 0.0,
        }
    
    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "cached_specs": len(self._spec_cache),
            "validated_specs": len(self._validation_cache),
        }
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._spec_cache.clear()
        self._validation_cache.clear()
