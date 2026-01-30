"""
Formal Specification Generation

AI-powered automatic generation of formal specifications from code:
- Pre/post condition inference
- Invariant detection
- Contract generation
- SMT-LIB specification output

Bridges the gap between informal code and formal methods by automatically
inferring what the code is supposed to do, then verifying it does that.
"""

from __future__ import annotations

import ast
import hashlib
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# =============================================================================
# Specification Types
# =============================================================================

class SpecificationType(str, Enum):
    """Types of formal specifications."""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    ASSERTION = "assertion"
    ASSUME = "assume"
    LOOP_INVARIANT = "loop_invariant"
    TYPE_CONSTRAINT = "type_constraint"


@dataclass
class FormalSpec:
    """A single formal specification."""
    
    spec_id: str
    spec_type: SpecificationType
    
    # Natural language description
    description: str
    
    # Formal representations
    z3_formula: Optional[str] = None
    smt_lib: Optional[str] = None
    python_assertion: Optional[str] = None
    
    # Location in code
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    line_number: Optional[int] = None
    
    # Variables involved
    variables: List[str] = field(default_factory=list)
    
    # Confidence in inference
    confidence: float = 0.8
    source: str = "inferred"  # "inferred", "documented", "user_defined"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spec_id": self.spec_id,
            "spec_type": self.spec_type.value,
            "description": self.description,
            "z3_formula": self.z3_formula,
            "smt_lib": self.smt_lib,
            "python_assertion": self.python_assertion,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "line_number": self.line_number,
            "variables": self.variables,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class FunctionContract:
    """Complete contract for a function."""
    
    function_name: str
    class_name: Optional[str] = None
    
    # Signature info
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    
    # Specifications
    preconditions: List[FormalSpec] = field(default_factory=list)
    postconditions: List[FormalSpec] = field(default_factory=list)
    invariants: List[FormalSpec] = field(default_factory=list)
    
    # Effects
    modifies: List[str] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)
    
    # Overall confidence
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "class_name": self.class_name,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "preconditions": [p.to_dict() for p in self.preconditions],
            "postconditions": [p.to_dict() for p in self.postconditions],
            "invariants": [i.to_dict() for i in self.invariants],
            "modifies": self.modifies,
            "raises": self.raises,
            "confidence": self.confidence,
        }
    
    def to_docstring(self) -> str:
        """Generate docstring with contract annotations."""
        lines = ['"""']
        
        if self.preconditions:
            lines.append("")
            lines.append("Requires:")
            for pre in self.preconditions:
                lines.append(f"    - {pre.description}")
        
        if self.postconditions:
            lines.append("")
            lines.append("Ensures:")
            for post in self.postconditions:
                lines.append(f"    - {post.description}")
        
        if self.invariants:
            lines.append("")
            lines.append("Invariants:")
            for inv in self.invariants:
                lines.append(f"    - {inv.description}")
        
        if self.modifies:
            lines.append("")
            lines.append(f"Modifies: {', '.join(self.modifies)}")
        
        if self.raises:
            lines.append("")
            lines.append("Raises:")
            for exc in self.raises:
                lines.append(f"    - {exc}")
        
        lines.append('"""')
        return "\n".join(lines)
    
    def to_z3_assertions(self) -> str:
        """Generate Z3 Python assertions for the contract."""
        lines = ["from z3 import *", ""]
        
        # Declare variables
        vars_declared: Set[str] = set()
        for spec in self.preconditions + self.postconditions:
            for var in spec.variables:
                if var not in vars_declared:
                    lines.append(f"{var} = Int('{var}')")
                    vars_declared.add(var)
        
        lines.append("")
        lines.append("s = Solver()")
        
        # Add preconditions
        if self.preconditions:
            lines.append("")
            lines.append("# Preconditions")
            for pre in self.preconditions:
                if pre.z3_formula:
                    lines.append(f"s.add({pre.z3_formula})  # {pre.description}")
        
        # Add postconditions to verify
        if self.postconditions:
            lines.append("")
            lines.append("# Postconditions (negated for verification)")
            for post in self.postconditions:
                if post.z3_formula:
                    lines.append(f"# s.add(Not({post.z3_formula}))  # {post.description}")
        
        return "\n".join(lines)


@dataclass
class ClassInvariant:
    """Invariant that must hold for a class."""
    
    class_name: str
    invariants: List[FormalSpec] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "class_name": self.class_name,
            "invariants": [i.to_dict() for i in self.invariants],
        }


# =============================================================================
# Specification Generator
# =============================================================================

class SpecificationGenerator:
    """
    Generates formal specifications from code using static analysis
    and pattern matching.
    """
    
    def __init__(self):
        self._spec_counter = 0
    
    def _generate_spec_id(self) -> str:
        """Generate unique specification ID."""
        self._spec_counter += 1
        return f"spec_{self._spec_counter:04d}"
    
    def generate_from_code(
        self,
        code: str,
        language: str = "python",
    ) -> Dict[str, Any]:
        """
        Generate specifications from code.
        
        Returns a comprehensive specification document.
        """
        if language == "python":
            return self._generate_python_specs(code)
        elif language in ("typescript", "javascript"):
            return self._generate_typescript_specs(code)
        else:
            return {"error": f"Unsupported language: {language}"}
    
    def _generate_python_specs(self, code: str) -> Dict[str, Any]:
        """Generate specifications for Python code."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
        
        result = {
            "functions": [],
            "classes": [],
            "module_invariants": [],
            "generated_at": time.time(),
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                contract = self._analyze_function(node, code)
                result["functions"].append(contract.to_dict())
            
            elif isinstance(node, ast.ClassDef):
                class_inv = self._analyze_class(node, code)
                result["classes"].append(class_inv.to_dict())
        
        return result
    
    def _analyze_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        code: str,
    ) -> FunctionContract:
        """Analyze a function and generate its contract."""
        contract = FunctionContract(
            function_name=node.name,
            parameters=self._extract_parameters(node),
            return_type=self._extract_return_type(node),
        )
        
        # Extract from docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self._parse_docstring_specs(docstring, contract)
        
        # Infer preconditions from parameter checks
        contract.preconditions.extend(
            self._infer_preconditions(node)
        )
        
        # Infer postconditions from return statements
        contract.postconditions.extend(
            self._infer_postconditions(node)
        )
        
        # Infer what the function modifies
        contract.modifies = self._infer_modifications(node)
        
        # Infer exceptions
        contract.raises = self._infer_exceptions(node)
        
        return contract
    
    def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract parameter information."""
        params = []
        for arg in node.args.args:
            param = {"name": arg.arg, "type": None}
            if arg.annotation:
                param["type"] = ast.unparse(arg.annotation)
            params.append(param)
        return params
    
    def _extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation."""
        if node.returns:
            return ast.unparse(node.returns)
        return None
    
    def _parse_docstring_specs(
        self,
        docstring: str,
        contract: FunctionContract,
    ) -> None:
        """Parse specifications from docstring."""
        # Look for common patterns
        
        # Args/Parameters section for preconditions
        args_match = re.search(
            r'(?:Args|Parameters|Params):\s*(.*?)(?:\n\n|\Z)',
            docstring,
            re.DOTALL | re.IGNORECASE,
        )
        if args_match:
            for line in args_match.group(1).split('\n'):
                line = line.strip()
                if line and ':' in line:
                    self._parse_param_constraint(line, contract)
        
        # Returns section for postconditions
        returns_match = re.search(
            r'(?:Returns|Return):\s*(.*?)(?:\n\n|\Z)',
            docstring,
            re.DOTALL | re.IGNORECASE,
        )
        if returns_match:
            desc = returns_match.group(1).strip()
            if desc:
                contract.postconditions.append(FormalSpec(
                    spec_id=self._generate_spec_id(),
                    spec_type=SpecificationType.POSTCONDITION,
                    description=f"Returns: {desc}",
                    source="documented",
                    confidence=0.9,
                ))
        
        # Raises section
        raises_match = re.search(
            r'Raises:\s*(.*?)(?:\n\n|\Z)',
            docstring,
            re.DOTALL | re.IGNORECASE,
        )
        if raises_match:
            for line in raises_match.group(1).split('\n'):
                line = line.strip()
                if line and ':' in line:
                    exc = line.split(':')[0].strip()
                    if exc:
                        contract.raises.append(exc)
    
    def _parse_param_constraint(
        self,
        line: str,
        contract: FunctionContract,
    ) -> None:
        """Parse a parameter constraint from docstring line."""
        parts = line.split(':', 1)
        if len(parts) < 2:
            return
        
        param_name = parts[0].strip().split()[0] if parts[0].strip() else ""
        description = parts[1].strip()
        
        # Look for constraint keywords
        constraint_keywords = [
            "must be", "should be", "cannot be", "must not be",
            "positive", "non-negative", "non-empty", "not null",
            "greater than", "less than", "between",
        ]
        
        for keyword in constraint_keywords:
            if keyword.lower() in description.lower():
                spec = self._create_param_spec(param_name, description, keyword)
                if spec:
                    contract.preconditions.append(spec)
                break
    
    def _create_param_spec(
        self,
        param_name: str,
        description: str,
        keyword: str,
    ) -> Optional[FormalSpec]:
        """Create a specification from parameter constraint."""
        spec_id = self._generate_spec_id()
        z3_formula = None
        python_assert = None
        
        if "positive" in keyword.lower() or "greater than 0" in description.lower():
            z3_formula = f"{param_name} > 0"
            python_assert = f"assert {param_name} > 0"
        elif "non-negative" in description.lower() or ">= 0" in description:
            z3_formula = f"{param_name} >= 0"
            python_assert = f"assert {param_name} >= 0"
        elif "non-empty" in description.lower() or "not empty" in description.lower():
            z3_formula = f"Length({param_name}) > 0"
            python_assert = f"assert len({param_name}) > 0"
        elif "not null" in description.lower() or "cannot be none" in description.lower():
            z3_formula = f"{param_name} != None"
            python_assert = f"assert {param_name} is not None"
        
        return FormalSpec(
            spec_id=spec_id,
            spec_type=SpecificationType.PRECONDITION,
            description=f"{param_name}: {description}",
            z3_formula=z3_formula,
            python_assertion=python_assert,
            variables=[param_name],
            source="documented",
            confidence=0.85,
        )
    
    def _infer_preconditions(self, node: ast.FunctionDef) -> List[FormalSpec]:
        """Infer preconditions from parameter validation code."""
        specs = []
        
        for stmt in node.body[:5]:  # Check first few statements
            if isinstance(stmt, ast.If):
                # Look for validation patterns like: if x < 0: raise ValueError
                specs.extend(self._analyze_validation_if(stmt))
            
            elif isinstance(stmt, ast.Assert):
                # Direct assertions are preconditions
                spec = self._analyze_assert(stmt)
                if spec:
                    specs.append(spec)
        
        # Infer from type annotations
        for arg in node.args.args:
            if arg.annotation:
                spec = self._infer_type_precondition(arg)
                if spec:
                    specs.append(spec)
        
        return specs
    
    def _analyze_validation_if(self, node: ast.If) -> List[FormalSpec]:
        """Analyze if statement for validation pattern."""
        specs = []
        
        # Check if body raises exception
        raises_exception = any(
            isinstance(s, ast.Raise) for s in node.body
        )
        
        if raises_exception and isinstance(node.test, ast.Compare):
            # The negation of the condition is the precondition
            try:
                condition = ast.unparse(node.test)
                negated = self._negate_condition(condition)
                
                specs.append(FormalSpec(
                    spec_id=self._generate_spec_id(),
                    spec_type=SpecificationType.PRECONDITION,
                    description=f"Requires: {negated}",
                    python_assertion=f"assert {negated}",
                    z3_formula=self._to_z3_formula(negated),
                    variables=self._extract_variables(node.test),
                    source="inferred",
                    confidence=0.9,
                ))
            except Exception:
                pass
        
        return specs
    
    def _negate_condition(self, condition: str) -> str:
        """Negate a condition string."""
        negations = {
            " < ": " >= ",
            " > ": " <= ",
            " <= ": " > ",
            " >= ": " < ",
            " == ": " != ",
            " != ": " == ",
            " is None": " is not None",
            " is not None": " is None",
        }
        
        for old, new in negations.items():
            if old in condition:
                return condition.replace(old, new)
        
        return f"not ({condition})"
    
    def _to_z3_formula(self, condition: str) -> str:
        """Convert Python condition to Z3 formula."""
        # Simple conversion - in production would use proper parsing
        z3_condition = condition
        z3_condition = z3_condition.replace(" and ", " & ")
        z3_condition = z3_condition.replace(" or ", " | ")
        z3_condition = z3_condition.replace(" is not None", " != None")
        z3_condition = z3_condition.replace(" is None", " == None")
        return z3_condition
    
    def _extract_variables(self, node: ast.AST) -> List[str]:
        """Extract variable names from AST node."""
        variables = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                variables.append(child.id)
        return list(set(variables))
    
    def _analyze_assert(self, node: ast.Assert) -> Optional[FormalSpec]:
        """Analyze assert statement."""
        try:
            condition = ast.unparse(node.test)
            message = ast.unparse(node.msg) if node.msg else condition
            
            return FormalSpec(
                spec_id=self._generate_spec_id(),
                spec_type=SpecificationType.PRECONDITION,
                description=f"Asserts: {message}",
                python_assertion=f"assert {condition}",
                z3_formula=self._to_z3_formula(condition),
                variables=self._extract_variables(node.test),
                source="explicit",
                confidence=1.0,
            )
        except Exception:
            return None
    
    def _infer_type_precondition(self, arg: ast.arg) -> Optional[FormalSpec]:
        """Infer precondition from type annotation."""
        if not arg.annotation:
            return None
        
        type_str = ast.unparse(arg.annotation)
        
        # Common type constraints
        constraints = {
            "int": f"{arg.arg} is an integer",
            "str": f"{arg.arg} is a string",
            "float": f"{arg.arg} is a number",
            "bool": f"{arg.arg} is a boolean",
            "List": f"{arg.arg} is a list",
            "Dict": f"{arg.arg} is a dictionary",
            "Optional": f"{arg.arg} may be None",
        }
        
        for type_name, description in constraints.items():
            if type_name in type_str:
                return FormalSpec(
                    spec_id=self._generate_spec_id(),
                    spec_type=SpecificationType.TYPE_CONSTRAINT,
                    description=description,
                    variables=[arg.arg],
                    source="type_annotation",
                    confidence=1.0,
                )
        
        return None
    
    def _infer_postconditions(self, node: ast.FunctionDef) -> List[FormalSpec]:
        """Infer postconditions from return statements."""
        specs = []
        
        # Analyze return type
        if node.returns:
            return_type = ast.unparse(node.returns)
            
            if "Optional" in return_type or "None" in return_type:
                specs.append(FormalSpec(
                    spec_id=self._generate_spec_id(),
                    spec_type=SpecificationType.POSTCONDITION,
                    description="May return None",
                    variables=["__return__"],
                    source="type_annotation",
                    confidence=1.0,
                ))
            else:
                specs.append(FormalSpec(
                    spec_id=self._generate_spec_id(),
                    spec_type=SpecificationType.POSTCONDITION,
                    description=f"Returns value of type {return_type}",
                    variables=["__return__"],
                    source="type_annotation",
                    confidence=1.0,
                ))
        
        # Analyze return statements
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value:
                spec = self._analyze_return_statement(stmt)
                if spec:
                    specs.append(spec)
        
        return specs
    
    def _analyze_return_statement(self, node: ast.Return) -> Optional[FormalSpec]:
        """Analyze a return statement for postconditions."""
        if not node.value:
            return None
        
        # Check for common patterns
        if isinstance(node.value, ast.Compare):
            # Returns a comparison result
            try:
                condition = ast.unparse(node.value)
                return FormalSpec(
                    spec_id=self._generate_spec_id(),
                    spec_type=SpecificationType.POSTCONDITION,
                    description=f"Returns result of: {condition}",
                    variables=self._extract_variables(node.value),
                    source="inferred",
                    confidence=0.7,
                )
            except Exception:
                pass
        
        return None
    
    def _infer_modifications(self, node: ast.FunctionDef) -> List[str]:
        """Infer what variables/attributes the function modifies."""
        modifies = []
        
        for stmt in ast.walk(node):
            # Attribute assignment: self.x = ...
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name):
                            modifies.append(f"{target.value.id}.{target.attr}")
            
            # Augmented assignment: x += ...
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Attribute):
                    if isinstance(stmt.target.value, ast.Name):
                        modifies.append(
                            f"{stmt.target.value.id}.{stmt.target.attr}"
                        )
        
        return list(set(modifies))
    
    def _infer_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """Infer what exceptions the function may raise."""
        exceptions = []
        
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Raise):
                if stmt.exc:
                    if isinstance(stmt.exc, ast.Call):
                        if isinstance(stmt.exc.func, ast.Name):
                            exceptions.append(stmt.exc.func.id)
                    elif isinstance(stmt.exc, ast.Name):
                        exceptions.append(stmt.exc.id)
        
        return list(set(exceptions))
    
    def _analyze_class(self, node: ast.ClassDef, code: str) -> ClassInvariant:
        """Analyze a class for invariants."""
        class_inv = ClassInvariant(class_name=node.name)
        
        # Look for __init__ to find class invariants
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                class_inv.invariants.extend(
                    self._infer_class_invariants(item, node.name)
                )
        
        return class_inv
    
    def _infer_class_invariants(
        self,
        init_node: ast.FunctionDef,
        class_name: str,
    ) -> List[FormalSpec]:
        """Infer class invariants from __init__."""
        invariants = []
        
        # Track attributes and their constraints
        for stmt in init_node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute):
                        if target.attr.startswith("_"):
                            continue  # Skip private
                        
                        # Check if there's a validation after
                        invariants.append(FormalSpec(
                            spec_id=self._generate_spec_id(),
                            spec_type=SpecificationType.INVARIANT,
                            description=f"{class_name}.{target.attr} is initialized",
                            class_name=class_name,
                            variables=[f"self.{target.attr}"],
                            source="inferred",
                            confidence=0.6,
                        ))
        
        return invariants
    
    def _generate_typescript_specs(self, code: str) -> Dict[str, Any]:
        """Generate specifications for TypeScript code."""
        # Simplified TypeScript analysis using regex
        result = {
            "functions": [],
            "classes": [],
            "interfaces": [],
            "generated_at": time.time(),
        }
        
        # Find functions
        func_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^\{]+))?\s*\{'
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            params_str = match.group(2)
            return_type = match.group(3).strip() if match.group(3) else None
            
            contract = {
                "function_name": func_name,
                "parameters": self._parse_ts_params(params_str),
                "return_type": return_type,
                "preconditions": [],
                "postconditions": [],
            }
            
            # Infer specs from types
            if return_type:
                if "null" in return_type or "undefined" in return_type:
                    contract["postconditions"].append({
                        "description": "May return null/undefined",
                        "confidence": 1.0,
                    })
            
            result["functions"].append(contract)
        
        # Find interfaces for type constraints
        interface_pattern = r'interface\s+(\w+)\s*\{([^}]+)\}'
        for match in re.finditer(interface_pattern, code):
            interface_name = match.group(1)
            body = match.group(2)
            
            result["interfaces"].append({
                "name": interface_name,
                "properties": self._parse_interface_props(body),
            })
        
        return result
    
    def _parse_ts_params(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse TypeScript parameter string."""
        params = []
        if not params_str.strip():
            return params
        
        for param in params_str.split(','):
            param = param.strip()
            if ':' in param:
                parts = param.split(':')
                name = parts[0].strip().replace('?', '')
                type_str = parts[1].strip()
                optional = '?' in parts[0]
                params.append({
                    "name": name,
                    "type": type_str,
                    "optional": optional,
                })
            elif param:
                params.append({"name": param, "type": None})
        
        return params
    
    def _parse_interface_props(self, body: str) -> List[Dict[str, Any]]:
        """Parse interface property definitions."""
        props = []
        for line in body.split(';'):
            line = line.strip()
            if ':' in line:
                parts = line.split(':')
                name = parts[0].strip().replace('?', '')
                type_str = parts[1].strip()
                optional = '?' in parts[0]
                props.append({
                    "name": name,
                    "type": type_str,
                    "optional": optional,
                })
        return props


# =============================================================================
# SMT-LIB Generator
# =============================================================================

class SMTLibGenerator:
    """Generates SMT-LIB format specifications."""
    
    def generate(self, contract: FunctionContract) -> str:
        """Generate SMT-LIB specification for a function contract."""
        lines = [
            "; SMT-LIB Specification",
            f"; Function: {contract.function_name}",
            "; Generated by CodeVerify",
            "",
        ]
        
        # Declare variables
        declared: Set[str] = set()
        all_specs = (
            contract.preconditions +
            contract.postconditions +
            contract.invariants
        )
        
        for spec in all_specs:
            for var in spec.variables:
                if var not in declared:
                    lines.append(f"(declare-const {var} Int)")
                    declared.add(var)
        
        lines.append("")
        
        # Preconditions as assertions
        if contract.preconditions:
            lines.append("; Preconditions")
            for pre in contract.preconditions:
                if pre.smt_lib:
                    lines.append(f"(assert {pre.smt_lib}) ; {pre.description}")
                elif pre.z3_formula:
                    smt = self._z3_to_smtlib(pre.z3_formula)
                    lines.append(f"(assert {smt}) ; {pre.description}")
        
        lines.append("")
        
        # Check satisfiability
        lines.extend([
            "; Check satisfiability",
            "(check-sat)",
            "(get-model)",
        ])
        
        return "\n".join(lines)
    
    def _z3_to_smtlib(self, z3_formula: str) -> str:
        """Convert Z3 Python formula to SMT-LIB."""
        # Simple conversion
        smt = z3_formula
        smt = smt.replace(" > ", " (> ")
        smt = smt.replace(" >= ", " (>= ")
        smt = smt.replace(" < ", " (< ")
        smt = smt.replace(" <= ", " (<= ")
        smt = smt.replace(" == ", " (= ")
        smt = smt.replace(" != ", " (distinct ")
        
        # Add closing parens (simplified)
        open_count = smt.count("(")
        close_count = smt.count(")")
        smt += ")" * (open_count - close_count)
        
        return smt


# =============================================================================
# Verification Engine Integration
# =============================================================================

class SpecificationVerifier:
    """Verifies code against generated specifications."""
    
    def __init__(self, generator: SpecificationGenerator):
        self.generator = generator
    
    def verify_code(
        self,
        code: str,
        specs: Dict[str, Any],
        language: str = "python",
    ) -> Dict[str, Any]:
        """
        Verify code against specifications.
        
        Returns verification results.
        """
        results = {
            "verified": True,
            "violations": [],
            "checked": 0,
            "passed": 0,
            "failed": 0,
        }
        
        for func_spec in specs.get("functions", []):
            func_result = self._verify_function(
                code,
                func_spec,
                language,
            )
            results["checked"] += func_result["checked"]
            results["passed"] += func_result["passed"]
            results["failed"] += func_result["failed"]
            results["violations"].extend(func_result["violations"])
        
        results["verified"] = results["failed"] == 0
        return results
    
    def _verify_function(
        self,
        code: str,
        func_spec: Dict[str, Any],
        language: str,
    ) -> Dict[str, Any]:
        """Verify a function against its specification."""
        result = {
            "function": func_spec["function_name"],
            "checked": 0,
            "passed": 0,
            "failed": 0,
            "violations": [],
        }
        
        # In production, this would use Z3 for formal verification
        # For now, do basic static checks
        
        for pre in func_spec.get("preconditions", []):
            result["checked"] += 1
            # Check if precondition is enforced in code
            if self._is_precondition_checked(code, pre, func_spec["function_name"]):
                result["passed"] += 1
            else:
                result["failed"] += 1
                result["violations"].append({
                    "type": "missing_precondition_check",
                    "spec": pre,
                    "message": f"Precondition not enforced: {pre.get('description', '')}",
                })
        
        return result
    
    def _is_precondition_checked(
        self,
        code: str,
        precondition: Dict[str, Any],
        function_name: str,
    ) -> bool:
        """Check if a precondition is enforced in code."""
        python_assert = precondition.get("python_assertion", "")
        
        # Simple check: is there an assertion or if-raise?
        if python_assert and python_assert in code:
            return True
        
        # Check for validation pattern
        for var in precondition.get("variables", []):
            if f"if {var}" in code or f"if not {var}" in code:
                return True
        
        return False
