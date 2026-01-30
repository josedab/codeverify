"""Cross-Language Verification Bridge - Unified verification for polyglot codebases."""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"


@dataclass
class TypeContract:
    """A type contract that can be verified across languages."""
    contract_id: str
    name: str
    description: str
    
    # Type definition
    base_type: str  # int, float, string, bool, array, object, void, any
    nullable: bool = False
    constraints: list[str] = field(default_factory=list)  # Z3 constraints
    
    # Object type details
    properties: dict[str, "TypeContract"] = field(default_factory=dict)
    
    # Array type details
    element_type: "TypeContract | None" = None
    min_length: int | None = None
    max_length: int | None = None
    
    # Numeric constraints
    min_value: float | None = None
    max_value: float | None = None
    
    # String constraints
    pattern: str | None = None  # Regex pattern
    min_length_str: int | None = None
    max_length_str: int | None = None


@dataclass
class FunctionContract:
    """A function contract that spans language boundaries."""
    contract_id: str
    name: str
    description: str
    
    # Signature
    parameters: list[tuple[str, TypeContract]] = field(default_factory=list)
    return_type: TypeContract | None = None
    
    # Contracts
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    
    # Behavior
    throws: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    is_pure: bool = False


@dataclass
class InterfaceContract:
    """An interface contract for cross-language APIs."""
    contract_id: str
    name: str
    description: str
    
    # Methods
    methods: dict[str, FunctionContract] = field(default_factory=dict)
    
    # Properties
    properties: dict[str, TypeContract] = field(default_factory=dict)
    
    # Inheritance
    extends: list[str] = field(default_factory=list)
    
    # Implementation tracking
    implementations: dict[Language, str] = field(default_factory=dict)


@dataclass
class LanguageBinding:
    """Binding between universal contract and language-specific code."""
    language: Language
    file_path: str
    symbol_name: str  # Function/class name in that language
    line_number: int | None = None
    verified: bool = False
    verification_errors: list[str] = field(default_factory=list)


@dataclass
class CrossLanguageVerificationResult:
    """Result of cross-language verification."""
    contract_id: str
    contract_type: str  # function, interface, type
    bindings: dict[Language, LanguageBinding] = field(default_factory=dict)
    all_verified: bool = False
    compatibility_issues: list[dict[str, Any]] = field(default_factory=list)
    type_mismatches: list[dict[str, Any]] = field(default_factory=list)
    behavior_inconsistencies: list[dict[str, Any]] = field(default_factory=list)


# Type mapping between languages
TYPE_MAPPINGS: dict[str, dict[Language, str]] = {
    "int": {
        Language.PYTHON: "int",
        Language.TYPESCRIPT: "number",
        Language.JAVASCRIPT: "number",
        Language.GO: "int",
        Language.RUST: "i32",
        Language.JAVA: "int",
        Language.CSHARP: "int",
        Language.CPP: "int",
    },
    "float": {
        Language.PYTHON: "float",
        Language.TYPESCRIPT: "number",
        Language.JAVASCRIPT: "number",
        Language.GO: "float64",
        Language.RUST: "f64",
        Language.JAVA: "double",
        Language.CSHARP: "double",
        Language.CPP: "double",
    },
    "string": {
        Language.PYTHON: "str",
        Language.TYPESCRIPT: "string",
        Language.JAVASCRIPT: "string",
        Language.GO: "string",
        Language.RUST: "String",
        Language.JAVA: "String",
        Language.CSHARP: "string",
        Language.CPP: "std::string",
    },
    "bool": {
        Language.PYTHON: "bool",
        Language.TYPESCRIPT: "boolean",
        Language.JAVASCRIPT: "boolean",
        Language.GO: "bool",
        Language.RUST: "bool",
        Language.JAVA: "boolean",
        Language.CSHARP: "bool",
        Language.CPP: "bool",
    },
    "array": {
        Language.PYTHON: "list",
        Language.TYPESCRIPT: "Array",
        Language.JAVASCRIPT: "Array",
        Language.GO: "[]",
        Language.RUST: "Vec",
        Language.JAVA: "List",
        Language.CSHARP: "List",
        Language.CPP: "std::vector",
    },
    "object": {
        Language.PYTHON: "dict",
        Language.TYPESCRIPT: "object",
        Language.JAVASCRIPT: "object",
        Language.GO: "map",
        Language.RUST: "HashMap",
        Language.JAVA: "Map",
        Language.CSHARP: "Dictionary",
        Language.CPP: "std::map",
    },
    "void": {
        Language.PYTHON: "None",
        Language.TYPESCRIPT: "void",
        Language.JAVASCRIPT: "void",
        Language.GO: "",
        Language.RUST: "()",
        Language.JAVA: "void",
        Language.CSHARP: "void",
        Language.CPP: "void",
    },
}


class LanguageAdapter(ABC):
    """Abstract adapter for language-specific operations."""
    
    @property
    @abstractmethod
    def language(self) -> Language:
        """Get the language this adapter handles."""
        pass
    
    @abstractmethod
    def parse_function_signature(
        self, code: str, function_name: str
    ) -> FunctionContract | None:
        """Parse a function signature from code."""
        pass
    
    @abstractmethod
    def parse_interface(
        self, code: str, interface_name: str
    ) -> InterfaceContract | None:
        """Parse an interface/class from code."""
        pass
    
    @abstractmethod
    def generate_type_annotation(
        self, contract: TypeContract
    ) -> str:
        """Generate language-specific type annotation."""
        pass
    
    @abstractmethod
    def generate_contract_stub(
        self, contract: FunctionContract
    ) -> str:
        """Generate a contract stub/skeleton."""
        pass


class PythonAdapter(LanguageAdapter):
    """Adapter for Python language."""
    
    @property
    def language(self) -> Language:
        return Language.PYTHON
    
    def parse_function_signature(
        self, code: str, function_name: str
    ) -> FunctionContract | None:
        """Parse Python function signature."""
        import re
        
        # Match function definition
        pattern = rf"def\s+{re.escape(function_name)}\s*\((.*?)\)\s*(?:->\s*(\S+))?\s*:"
        match = re.search(pattern, code, re.DOTALL)
        
        if not match:
            return None
        
        params_str = match.group(1)
        return_type_str = match.group(2)
        
        # Parse parameters
        parameters = []
        if params_str.strip():
            for param in params_str.split(","):
                param = param.strip()
                if ":" in param:
                    name, type_hint = param.split(":", 1)
                    name = name.strip()
                    type_hint = type_hint.strip()
                    if "=" in type_hint:
                        type_hint = type_hint.split("=")[0].strip()
                    param_contract = self._parse_type_hint(type_hint)
                    parameters.append((name, param_contract))
                elif param and param != "self":
                    parameters.append((param, TypeContract(
                        contract_id=f"param_{param}",
                        name=param,
                        description="",
                        base_type="any",
                    )))
        
        return_contract = None
        if return_type_str:
            return_contract = self._parse_type_hint(return_type_str)
        
        return FunctionContract(
            contract_id=f"py_{function_name}",
            name=function_name,
            description="",
            parameters=parameters,
            return_type=return_contract,
        )
    
    def _parse_type_hint(self, hint: str) -> TypeContract:
        """Parse a Python type hint into TypeContract."""
        hint = hint.strip()
        
        # Handle Optional
        if hint.startswith("Optional["):
            inner = hint[9:-1]
            contract = self._parse_type_hint(inner)
            contract.nullable = True
            return contract
        
        # Handle List
        if hint.startswith("list[") or hint.startswith("List["):
            inner = hint[5:-1]
            return TypeContract(
                contract_id=f"list_{inner}",
                name=f"list[{inner}]",
                description="",
                base_type="array",
                element_type=self._parse_type_hint(inner),
            )
        
        # Handle Dict
        if hint.startswith("dict[") or hint.startswith("Dict["):
            return TypeContract(
                contract_id="dict",
                name="dict",
                description="",
                base_type="object",
            )
        
        # Map basic types
        type_map = {
            "int": "int",
            "float": "float",
            "str": "string",
            "bool": "bool",
            "None": "void",
            "Any": "any",
        }
        
        base_type = type_map.get(hint, "any")
        
        return TypeContract(
            contract_id=f"type_{hint}",
            name=hint,
            description="",
            base_type=base_type,
        )
    
    def parse_interface(
        self, code: str, interface_name: str
    ) -> InterfaceContract | None:
        """Parse a Python class as an interface."""
        import re
        
        # Find class definition
        pattern = rf"class\s+{re.escape(interface_name)}\s*(?:\((.*?)\))?\s*:"
        match = re.search(pattern, code)
        
        if not match:
            return None
        
        # Find class body
        class_start = match.end()
        indent_match = re.search(r"\n(\s+)", code[class_start:])
        if not indent_match:
            return None
        
        base_indent = len(indent_match.group(1))
        
        # Extract methods
        methods = {}
        method_pattern = rf"\n\s{{{base_indent}}}def\s+(\w+)"
        
        for method_match in re.finditer(method_pattern, code[class_start:]):
            method_name = method_match.group(1)
            if not method_name.startswith("_"):
                func_contract = self.parse_function_signature(
                    code[class_start:], method_name
                )
                if func_contract:
                    methods[method_name] = func_contract
        
        return InterfaceContract(
            contract_id=f"py_interface_{interface_name}",
            name=interface_name,
            description="",
            methods=methods,
        )
    
    def generate_type_annotation(self, contract: TypeContract) -> str:
        """Generate Python type annotation."""
        if contract.base_type == "array" and contract.element_type:
            inner = self.generate_type_annotation(contract.element_type)
            result = f"list[{inner}]"
        elif contract.base_type == "object":
            result = "dict[str, Any]"
        else:
            type_map = {
                "int": "int",
                "float": "float",
                "string": "str",
                "bool": "bool",
                "void": "None",
                "any": "Any",
            }
            result = type_map.get(contract.base_type, "Any")
        
        if contract.nullable:
            result = f"{result} | None"
        
        return result
    
    def generate_contract_stub(self, contract: FunctionContract) -> str:
        """Generate Python function stub."""
        params = []
        for name, type_contract in contract.parameters:
            type_hint = self.generate_type_annotation(type_contract)
            params.append(f"{name}: {type_hint}")
        
        return_hint = ""
        if contract.return_type:
            return_hint = f" -> {self.generate_type_annotation(contract.return_type)}"
        
        lines = [
            f"def {contract.name}({', '.join(params)}){return_hint}:",
            f'    """',
            f"    {contract.description or 'Generated from cross-language contract.'}",
        ]
        
        if contract.preconditions:
            lines.append("")
            lines.append("    Preconditions:")
            for pre in contract.preconditions:
                lines.append(f"        - {pre}")
        
        if contract.postconditions:
            lines.append("")
            lines.append("    Postconditions:")
            for post in contract.postconditions:
                lines.append(f"        - {post}")
        
        lines.append('    """')
        lines.append("    raise NotImplementedError()")
        
        return "\n".join(lines)


class TypeScriptAdapter(LanguageAdapter):
    """Adapter for TypeScript language."""
    
    @property
    def language(self) -> Language:
        return Language.TYPESCRIPT
    
    def parse_function_signature(
        self, code: str, function_name: str
    ) -> FunctionContract | None:
        """Parse TypeScript function signature."""
        import re
        
        # Match function definition
        patterns = [
            rf"function\s+{re.escape(function_name)}\s*(?:<.*?>)?\s*\((.*?)\)\s*:\s*(\S+)",
            rf"(?:const|let|var)\s+{re.escape(function_name)}\s*=\s*(?:async\s+)?\((.*?)\)\s*(?::\s*(\S+))?\s*=>",
            rf"{re.escape(function_name)}\s*(?:<.*?>)?\s*\((.*?)\)\s*:\s*(\S+)\s*\{{",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, code, re.DOTALL)
            if match:
                params_str = match.group(1)
                return_type_str = match.group(2) if len(match.groups()) > 1 else None
                
                parameters = self._parse_parameters(params_str)
                
                return_contract = None
                if return_type_str:
                    return_contract = self._parse_type(return_type_str)
                
                return FunctionContract(
                    contract_id=f"ts_{function_name}",
                    name=function_name,
                    description="",
                    parameters=parameters,
                    return_type=return_contract,
                )
        
        return None
    
    def _parse_parameters(
        self, params_str: str
    ) -> list[tuple[str, TypeContract]]:
        """Parse TypeScript parameter list."""
        parameters = []
        
        if not params_str.strip():
            return parameters
        
        # Simple split (doesn't handle nested generics well)
        depth = 0
        current = ""
        
        for char in params_str:
            if char in "<([{":
                depth += 1
            elif char in ">)]}":
                depth -= 1
            
            if char == "," and depth == 0:
                if current.strip():
                    parameters.append(self._parse_parameter(current.strip()))
                current = ""
            else:
                current += char
        
        if current.strip():
            parameters.append(self._parse_parameter(current.strip()))
        
        return parameters
    
    def _parse_parameter(self, param: str) -> tuple[str, TypeContract]:
        """Parse a single parameter."""
        # Handle optional parameters
        optional = "?" in param
        param = param.replace("?", "")
        
        if ":" in param:
            name, type_str = param.split(":", 1)
            name = name.strip()
            type_str = type_str.strip()
            
            # Remove default value
            if "=" in type_str:
                type_str = type_str.split("=")[0].strip()
            
            contract = self._parse_type(type_str)
            contract.nullable = optional or contract.nullable
            return (name, contract)
        else:
            return (param.strip(), TypeContract(
                contract_id=f"param_{param}",
                name=param,
                description="",
                base_type="any",
            ))
    
    def _parse_type(self, type_str: str) -> TypeContract:
        """Parse TypeScript type into contract."""
        type_str = type_str.strip()
        
        # Handle union with null/undefined
        nullable = False
        if "|" in type_str:
            parts = [p.strip() for p in type_str.split("|")]
            non_null_parts = [p for p in parts if p not in ("null", "undefined")]
            if len(non_null_parts) < len(parts):
                nullable = True
            if len(non_null_parts) == 1:
                type_str = non_null_parts[0]
        
        # Handle arrays
        if type_str.endswith("[]"):
            inner = type_str[:-2]
            return TypeContract(
                contract_id=f"array_{inner}",
                name=type_str,
                description="",
                base_type="array",
                element_type=self._parse_type(inner),
                nullable=nullable,
            )
        
        if type_str.startswith("Array<"):
            inner = type_str[6:-1]
            return TypeContract(
                contract_id=f"array_{inner}",
                name=type_str,
                description="",
                base_type="array",
                element_type=self._parse_type(inner),
                nullable=nullable,
            )
        
        # Map basic types
        type_map = {
            "number": "float",
            "string": "string",
            "boolean": "bool",
            "void": "void",
            "any": "any",
            "unknown": "any",
            "never": "void",
        }
        
        base_type = type_map.get(type_str, "any")
        
        return TypeContract(
            contract_id=f"type_{type_str}",
            name=type_str,
            description="",
            base_type=base_type,
            nullable=nullable,
        )
    
    def parse_interface(
        self, code: str, interface_name: str
    ) -> InterfaceContract | None:
        """Parse TypeScript interface."""
        import re
        
        pattern = rf"interface\s+{re.escape(interface_name)}\s*(?:extends\s+([\w\s,]+))?\s*\{{"
        match = re.search(pattern, code)
        
        if not match:
            return None
        
        extends = []
        if match.group(1):
            extends = [e.strip() for e in match.group(1).split(",")]
        
        # Find interface body
        start = match.end()
        depth = 1
        end = start
        
        for i, char in enumerate(code[start:]):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = start + i
                    break
        
        body = code[start:end]
        
        # Parse methods
        methods = {}
        method_pattern = r"(\w+)\s*\((.*?)\)\s*:\s*(\S+)"
        
        for method_match in re.finditer(method_pattern, body):
            method_name = method_match.group(1)
            params_str = method_match.group(2)
            return_type = method_match.group(3)
            
            parameters = self._parse_parameters(params_str)
            return_contract = self._parse_type(return_type)
            
            methods[method_name] = FunctionContract(
                contract_id=f"ts_{interface_name}_{method_name}",
                name=method_name,
                description="",
                parameters=parameters,
                return_type=return_contract,
            )
        
        return InterfaceContract(
            contract_id=f"ts_interface_{interface_name}",
            name=interface_name,
            description="",
            methods=methods,
            extends=extends,
        )
    
    def generate_type_annotation(self, contract: TypeContract) -> str:
        """Generate TypeScript type annotation."""
        if contract.base_type == "array" and contract.element_type:
            inner = self.generate_type_annotation(contract.element_type)
            result = f"{inner}[]"
        elif contract.base_type == "object":
            result = "Record<string, unknown>"
        else:
            type_map = {
                "int": "number",
                "float": "number",
                "string": "string",
                "bool": "boolean",
                "void": "void",
                "any": "any",
            }
            result = type_map.get(contract.base_type, "any")
        
        if contract.nullable:
            result = f"{result} | null"
        
        return result
    
    def generate_contract_stub(self, contract: FunctionContract) -> str:
        """Generate TypeScript function stub."""
        params = []
        for name, type_contract in contract.parameters:
            type_hint = self.generate_type_annotation(type_contract)
            params.append(f"{name}: {type_hint}")
        
        return_hint = "void"
        if contract.return_type:
            return_hint = self.generate_type_annotation(contract.return_type)
        
        lines = [
            "/**",
            f" * {contract.description or 'Generated from cross-language contract.'}",
        ]
        
        if contract.preconditions:
            lines.append(" *")
            lines.append(" * @preconditions")
            for pre in contract.preconditions:
                lines.append(f" *   - {pre}")
        
        if contract.postconditions:
            lines.append(" *")
            lines.append(" * @postconditions")
            for post in contract.postconditions:
                lines.append(f" *   - {post}")
        
        lines.append(" */")
        lines.append(f"function {contract.name}({', '.join(params)}): {return_hint} {{")
        lines.append("  throw new Error('Not implemented');")
        lines.append("}")
        
        return "\n".join(lines)


class GoAdapter(LanguageAdapter):
    """Adapter for Go language."""
    
    @property
    def language(self) -> Language:
        return Language.GO
    
    def parse_function_signature(
        self, code: str, function_name: str
    ) -> FunctionContract | None:
        """Parse Go function signature."""
        import re
        
        # Match function definition: func name(params) return_type
        pattern = rf"func\s+{re.escape(function_name)}\s*\((.*?)\)\s*(?:\((.*?)\)|(\w+))?"
        match = re.search(pattern, code, re.DOTALL)
        
        if not match:
            return None
        
        params_str = match.group(1)
        return_type_str = match.group(2) or match.group(3)
        
        parameters = self._parse_parameters(params_str)
        
        return_contract = None
        if return_type_str:
            return_contract = self._parse_type(return_type_str.strip())
        
        return FunctionContract(
            contract_id=f"go_{function_name}",
            name=function_name,
            description="",
            parameters=parameters,
            return_type=return_contract,
        )
    
    def _parse_parameters(
        self, params_str: str
    ) -> list[tuple[str, TypeContract]]:
        """Parse Go parameter list."""
        parameters = []
        
        if not params_str.strip():
            return parameters
        
        # Go groups parameters by type: x, y int, z string
        parts = params_str.split(",")
        current_names = []
        
        for part in parts:
            part = part.strip()
            tokens = part.split()
            
            if len(tokens) >= 2:
                # Has type
                type_str = tokens[-1]
                names = tokens[:-1]
                
                # Add any accumulated names with this type
                all_names = current_names + [n for n in names if n]
                current_names = []
                
                for name in all_names:
                    name = name.strip()
                    if name:
                        parameters.append((name, self._parse_type(type_str)))
            elif len(tokens) == 1:
                # Just a name, type comes later
                current_names.append(tokens[0])
        
        return parameters
    
    def _parse_type(self, type_str: str) -> TypeContract:
        """Parse Go type into contract."""
        type_str = type_str.strip()
        
        # Handle pointer
        nullable = type_str.startswith("*")
        if nullable:
            type_str = type_str[1:]
        
        # Handle slice
        if type_str.startswith("[]"):
            inner = type_str[2:]
            return TypeContract(
                contract_id=f"slice_{inner}",
                name=type_str,
                description="",
                base_type="array",
                element_type=self._parse_type(inner),
                nullable=nullable,
            )
        
        # Handle map
        if type_str.startswith("map["):
            return TypeContract(
                contract_id="map",
                name="map",
                description="",
                base_type="object",
                nullable=nullable,
            )
        
        # Map basic types
        type_map = {
            "int": "int",
            "int32": "int",
            "int64": "int",
            "float32": "float",
            "float64": "float",
            "string": "string",
            "bool": "bool",
            "error": "any",
        }
        
        base_type = type_map.get(type_str, "any")
        
        return TypeContract(
            contract_id=f"type_{type_str}",
            name=type_str,
            description="",
            base_type=base_type,
            nullable=nullable,
        )
    
    def parse_interface(
        self, code: str, interface_name: str
    ) -> InterfaceContract | None:
        """Parse Go interface."""
        import re
        
        pattern = rf"type\s+{re.escape(interface_name)}\s+interface\s*\{{"
        match = re.search(pattern, code)
        
        if not match:
            return None
        
        # Find interface body
        start = match.end()
        depth = 1
        end = start
        
        for i, char in enumerate(code[start:]):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = start + i
                    break
        
        body = code[start:end]
        
        # Parse methods
        methods = {}
        method_pattern = r"(\w+)\s*\((.*?)\)\s*(?:\((.*?)\)|(\w+))?"
        
        for method_match in re.finditer(method_pattern, body):
            method_name = method_match.group(1)
            params_str = method_match.group(2)
            return_type = method_match.group(3) or method_match.group(4)
            
            parameters = self._parse_parameters(params_str)
            return_contract = self._parse_type(return_type) if return_type else None
            
            methods[method_name] = FunctionContract(
                contract_id=f"go_{interface_name}_{method_name}",
                name=method_name,
                description="",
                parameters=parameters,
                return_type=return_contract,
            )
        
        return InterfaceContract(
            contract_id=f"go_interface_{interface_name}",
            name=interface_name,
            description="",
            methods=methods,
        )
    
    def generate_type_annotation(self, contract: TypeContract) -> str:
        """Generate Go type annotation."""
        if contract.base_type == "array" and contract.element_type:
            inner = self.generate_type_annotation(contract.element_type)
            result = f"[]{inner}"
        elif contract.base_type == "object":
            result = "map[string]interface{}"
        else:
            type_map = {
                "int": "int",
                "float": "float64",
                "string": "string",
                "bool": "bool",
                "void": "",
                "any": "interface{}",
            }
            result = type_map.get(contract.base_type, "interface{}")
        
        if contract.nullable and result:
            result = f"*{result}"
        
        return result
    
    def generate_contract_stub(self, contract: FunctionContract) -> str:
        """Generate Go function stub."""
        params = []
        for name, type_contract in contract.parameters:
            type_hint = self.generate_type_annotation(type_contract)
            params.append(f"{name} {type_hint}")
        
        return_hint = ""
        if contract.return_type:
            return_hint = f" {self.generate_type_annotation(contract.return_type)}"
        
        lines = [
            f"// {contract.description or 'Generated from cross-language contract.'}",
        ]
        
        if contract.preconditions:
            lines.append("// Preconditions:")
            for pre in contract.preconditions:
                lines.append(f"//   - {pre}")
        
        lines.append(f"func {contract.name}({', '.join(params)}){return_hint} {{")
        lines.append('    panic("Not implemented")')
        lines.append("}")
        
        return "\n".join(lines)


class RustAdapter(LanguageAdapter):
    """Adapter for Rust language."""
    
    @property
    def language(self) -> Language:
        return Language.RUST
    
    def parse_function_signature(
        self, code: str, function_name: str
    ) -> FunctionContract | None:
        """Parse Rust function signature."""
        import re
        
        # Match function definition
        pattern = rf"(?:pub\s+)?fn\s+{re.escape(function_name)}\s*(?:<.*?>)?\s*\((.*?)\)\s*(?:->\s*(\S+))?"
        match = re.search(pattern, code, re.DOTALL)
        
        if not match:
            return None
        
        params_str = match.group(1)
        return_type_str = match.group(2)
        
        parameters = self._parse_parameters(params_str)
        
        return_contract = None
        if return_type_str:
            return_contract = self._parse_type(return_type_str.strip())
        
        return FunctionContract(
            contract_id=f"rust_{function_name}",
            name=function_name,
            description="",
            parameters=parameters,
            return_type=return_contract,
        )
    
    def _parse_parameters(
        self, params_str: str
    ) -> list[tuple[str, TypeContract]]:
        """Parse Rust parameter list."""
        parameters = []
        
        if not params_str.strip():
            return parameters
        
        # Skip self parameter
        parts = params_str.split(",")
        
        for part in parts:
            part = part.strip()
            if part in ("self", "&self", "&mut self", "mut self"):
                continue
            
            if ":" in part:
                name, type_str = part.split(":", 1)
                name = name.strip().lstrip("mut ")
                type_str = type_str.strip()
                
                parameters.append((name, self._parse_type(type_str)))
        
        return parameters
    
    def _parse_type(self, type_str: str) -> TypeContract:
        """Parse Rust type into contract."""
        type_str = type_str.strip()
        
        # Handle Option
        nullable = False
        if type_str.startswith("Option<") and type_str.endswith(">"):
            nullable = True
            type_str = type_str[7:-1]
        
        # Handle reference
        if type_str.startswith("&"):
            type_str = type_str.lstrip("&mut ")
        
        # Handle Vec
        if type_str.startswith("Vec<") and type_str.endswith(">"):
            inner = type_str[4:-1]
            return TypeContract(
                contract_id=f"vec_{inner}",
                name=type_str,
                description="",
                base_type="array",
                element_type=self._parse_type(inner),
                nullable=nullable,
            )
        
        # Handle HashMap
        if type_str.startswith("HashMap<"):
            return TypeContract(
                contract_id="hashmap",
                name="HashMap",
                description="",
                base_type="object",
                nullable=nullable,
            )
        
        # Map basic types
        type_map = {
            "i32": "int",
            "i64": "int",
            "u32": "int",
            "u64": "int",
            "f32": "float",
            "f64": "float",
            "String": "string",
            "str": "string",
            "bool": "bool",
            "()": "void",
        }
        
        base_type = type_map.get(type_str, "any")
        
        return TypeContract(
            contract_id=f"type_{type_str}",
            name=type_str,
            description="",
            base_type=base_type,
            nullable=nullable,
        )
    
    def parse_interface(
        self, code: str, interface_name: str
    ) -> InterfaceContract | None:
        """Parse Rust trait as interface."""
        import re
        
        pattern = rf"(?:pub\s+)?trait\s+{re.escape(interface_name)}\s*(?:<.*?>)?\s*\{{"
        match = re.search(pattern, code)
        
        if not match:
            return None
        
        # Find trait body
        start = match.end()
        depth = 1
        end = start
        
        for i, char in enumerate(code[start:]):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = start + i
                    break
        
        body = code[start:end]
        
        # Parse methods
        methods = {}
        method_pattern = r"fn\s+(\w+)\s*\((.*?)\)\s*(?:->\s*(\S+))?"
        
        for method_match in re.finditer(method_pattern, body):
            method_name = method_match.group(1)
            params_str = method_match.group(2)
            return_type = method_match.group(3)
            
            parameters = self._parse_parameters(params_str)
            return_contract = self._parse_type(return_type) if return_type else None
            
            methods[method_name] = FunctionContract(
                contract_id=f"rust_{interface_name}_{method_name}",
                name=method_name,
                description="",
                parameters=parameters,
                return_type=return_contract,
            )
        
        return InterfaceContract(
            contract_id=f"rust_trait_{interface_name}",
            name=interface_name,
            description="",
            methods=methods,
        )
    
    def generate_type_annotation(self, contract: TypeContract) -> str:
        """Generate Rust type annotation."""
        if contract.base_type == "array" and contract.element_type:
            inner = self.generate_type_annotation(contract.element_type)
            result = f"Vec<{inner}>"
        elif contract.base_type == "object":
            result = "HashMap<String, Box<dyn std::any::Any>>"
        else:
            type_map = {
                "int": "i32",
                "float": "f64",
                "string": "String",
                "bool": "bool",
                "void": "()",
                "any": "Box<dyn std::any::Any>",
            }
            result = type_map.get(contract.base_type, "Box<dyn std::any::Any>")
        
        if contract.nullable:
            result = f"Option<{result}>"
        
        return result
    
    def generate_contract_stub(self, contract: FunctionContract) -> str:
        """Generate Rust function stub."""
        params = []
        for name, type_contract in contract.parameters:
            type_hint = self.generate_type_annotation(type_contract)
            params.append(f"{name}: {type_hint}")
        
        return_hint = ""
        if contract.return_type:
            return_hint = f" -> {self.generate_type_annotation(contract.return_type)}"
        
        lines = [
            f"/// {contract.description or 'Generated from cross-language contract.'}",
        ]
        
        if contract.preconditions:
            lines.append("///")
            lines.append("/// # Preconditions")
            for pre in contract.preconditions:
                lines.append(f"/// - {pre}")
        
        lines.append(f"pub fn {contract.name}({', '.join(params)}){return_hint} {{")
        lines.append('    todo!("Not implemented")')
        lines.append("}")
        
        return "\n".join(lines)


class JavaAdapter(LanguageAdapter):
    """Adapter for Java language."""
    
    @property
    def language(self) -> Language:
        return Language.JAVA
    
    def parse_function_signature(
        self, code: str, function_name: str
    ) -> FunctionContract | None:
        """Parse Java method signature."""
        import re
        
        # Match method definition
        pattern = rf"(?:public|private|protected)?\s*(?:static\s+)?(\w+(?:<.*?>)?)\s+{re.escape(function_name)}\s*\((.*?)\)"
        match = re.search(pattern, code, re.DOTALL)
        
        if not match:
            return None
        
        return_type_str = match.group(1)
        params_str = match.group(2)
        
        parameters = self._parse_parameters(params_str)
        return_contract = self._parse_type(return_type_str) if return_type_str else None
        
        return FunctionContract(
            contract_id=f"java_{function_name}",
            name=function_name,
            description="",
            parameters=parameters,
            return_type=return_contract,
        )
    
    def _parse_parameters(
        self, params_str: str
    ) -> list[tuple[str, TypeContract]]:
        """Parse Java parameter list."""
        parameters = []
        
        if not params_str.strip():
            return parameters
        
        parts = params_str.split(",")
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Handle final modifier and annotations
            part = re.sub(r"@\w+\s*", "", part)
            part = re.sub(r"\bfinal\s+", "", part)
            
            tokens = part.split()
            if len(tokens) >= 2:
                type_str = " ".join(tokens[:-1])
                name = tokens[-1]
                parameters.append((name, self._parse_type(type_str)))
        
        return parameters
    
    def _parse_type(self, type_str: str) -> TypeContract:
        """Parse Java type into contract."""
        type_str = type_str.strip()
        
        # Handle generics
        nullable = False
        
        # Handle List/ArrayList
        if type_str.startswith("List<") or type_str.startswith("ArrayList<"):
            inner_start = type_str.index("<") + 1
            inner = type_str[inner_start:-1]
            return TypeContract(
                contract_id=f"list_{inner}",
                name=type_str,
                description="",
                base_type="array",
                element_type=self._parse_type(inner),
                nullable=nullable,
            )
        
        # Handle Map/HashMap
        if type_str.startswith("Map<") or type_str.startswith("HashMap<"):
            return TypeContract(
                contract_id="map",
                name="Map",
                description="",
                base_type="object",
                nullable=nullable,
            )
        
        # Handle arrays
        if type_str.endswith("[]"):
            inner = type_str[:-2]
            return TypeContract(
                contract_id=f"array_{inner}",
                name=type_str,
                description="",
                base_type="array",
                element_type=self._parse_type(inner),
            )
        
        # Map basic types
        type_map = {
            "int": "int",
            "Integer": "int",
            "long": "int",
            "Long": "int",
            "double": "float",
            "Double": "float",
            "float": "float",
            "Float": "float",
            "String": "string",
            "boolean": "bool",
            "Boolean": "bool",
            "void": "void",
            "Object": "any",
        }
        
        base_type = type_map.get(type_str, "any")
        
        return TypeContract(
            contract_id=f"type_{type_str}",
            name=type_str,
            description="",
            base_type=base_type,
            nullable=nullable,
        )
    
    def parse_interface(
        self, code: str, interface_name: str
    ) -> InterfaceContract | None:
        """Parse Java interface."""
        import re
        
        pattern = rf"(?:public\s+)?interface\s+{re.escape(interface_name)}\s*(?:extends\s+([\w\s,<>]+))?\s*\{{"
        match = re.search(pattern, code)
        
        if not match:
            return None
        
        extends = []
        if match.group(1):
            extends = [e.strip() for e in match.group(1).split(",")]
        
        # Find interface body
        start = match.end()
        depth = 1
        end = start
        
        for i, char in enumerate(code[start:]):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = start + i
                    break
        
        body = code[start:end]
        
        # Parse methods
        methods = {}
        method_pattern = r"(\w+(?:<.*?>)?)\s+(\w+)\s*\((.*?)\)"
        
        for method_match in re.finditer(method_pattern, body):
            return_type = method_match.group(1)
            method_name = method_match.group(2)
            params_str = method_match.group(3)
            
            parameters = self._parse_parameters(params_str)
            return_contract = self._parse_type(return_type)
            
            methods[method_name] = FunctionContract(
                contract_id=f"java_{interface_name}_{method_name}",
                name=method_name,
                description="",
                parameters=parameters,
                return_type=return_contract,
            )
        
        return InterfaceContract(
            contract_id=f"java_interface_{interface_name}",
            name=interface_name,
            description="",
            methods=methods,
            extends=extends,
        )
    
    def generate_type_annotation(self, contract: TypeContract) -> str:
        """Generate Java type annotation."""
        if contract.base_type == "array" and contract.element_type:
            inner = self.generate_type_annotation(contract.element_type)
            result = f"List<{inner}>"
        elif contract.base_type == "object":
            result = "Map<String, Object>"
        else:
            type_map = {
                "int": "int",
                "float": "double",
                "string": "String",
                "bool": "boolean",
                "void": "void",
                "any": "Object",
            }
            result = type_map.get(contract.base_type, "Object")
        
        return result
    
    def generate_contract_stub(self, contract: FunctionContract) -> str:
        """Generate Java method stub."""
        params = []
        for name, type_contract in contract.parameters:
            type_hint = self.generate_type_annotation(type_contract)
            params.append(f"{type_hint} {name}")
        
        return_hint = "void"
        if contract.return_type:
            return_hint = self.generate_type_annotation(contract.return_type)
        
        lines = [
            "/**",
            f" * {contract.description or 'Generated from cross-language contract.'}",
        ]
        
        if contract.preconditions:
            lines.append(" *")
            lines.append(" * <p>Preconditions:</p>")
            lines.append(" * <ul>")
            for pre in contract.preconditions:
                lines.append(f" *   <li>{pre}</li>")
            lines.append(" * </ul>")
        
        lines.append(" */")
        lines.append(f"public {return_hint} {contract.name}({', '.join(params)}) {{")
        lines.append('    throw new UnsupportedOperationException("Not implemented");')
        lines.append("}")
        
        return "\n".join(lines)


class CrossLanguageVerificationBridge(BaseAgent):
    """
    Bridge for verifying polyglot codebases with unified contracts.
    
    This bridge:
    - Defines language-agnostic type and function contracts
    - Maps contracts to language-specific implementations
    - Verifies consistency across implementations
    - Detects type mismatches and behavior differences
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the cross-language bridge."""
        super().__init__(config)
        
        # Language adapters
        self._adapters: dict[Language, LanguageAdapter] = {
            Language.PYTHON: PythonAdapter(),
            Language.TYPESCRIPT: TypeScriptAdapter(),
            Language.GO: GoAdapter(),
            Language.RUST: RustAdapter(),
            Language.JAVA: JavaAdapter(),
        }
        
        # Contract storage
        self._function_contracts: dict[str, FunctionContract] = {}
        self._interface_contracts: dict[str, InterfaceContract] = {}
        self._type_contracts: dict[str, TypeContract] = {}
        
        # Bindings
        self._bindings: dict[str, dict[Language, LanguageBinding]] = {}

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Verify cross-language contract consistency.

        Args:
            code: Code to analyze (can be from multiple files)
            context: Additional context including:
                - language: Language of the provided code
                - contract_id: Contract to verify against
                - contract_type: "function", "interface", or "type"

        Returns:
            AgentResult with verification results
        """
        start_time = time.time()
        
        language_str = context.get("language", "python")
        language = Language(language_str)
        contract_id = context.get("contract_id")
        contract_type = context.get("contract_type", "function")
        
        try:
            if not contract_id:
                # Infer contract from code
                result = await self._infer_contract(code, language, context)
            else:
                # Verify against existing contract
                result = await self._verify_against_contract(
                    code, language, contract_id, contract_type
                )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "Cross-language verification completed",
                language=language.value,
                contract_id=contract_id,
                latency_ms=elapsed_ms,
            )
            
            return AgentResult(
                success=True,
                data=result,
                latency_ms=elapsed_ms,
            )
            
        except Exception as e:
            logger.error("Cross-language verification failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _infer_contract(
        self,
        code: str,
        language: Language,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Infer a contract from code."""
        adapter = self._adapters.get(language)
        if not adapter:
            raise ValueError(f"No adapter for language: {language}")
        
        symbol_name = context.get("symbol_name")
        symbol_type = context.get("symbol_type", "function")
        
        if symbol_type == "function" and symbol_name:
            contract = adapter.parse_function_signature(code, symbol_name)
            if contract:
                self._function_contracts[contract.contract_id] = contract
                return {
                    "contract_id": contract.contract_id,
                    "contract_type": "function",
                    "contract": self._function_contract_to_dict(contract),
                    "inferred_from": language.value,
                }
        
        elif symbol_type == "interface" and symbol_name:
            contract = adapter.parse_interface(code, symbol_name)
            if contract:
                self._interface_contracts[contract.contract_id] = contract
                return {
                    "contract_id": contract.contract_id,
                    "contract_type": "interface",
                    "contract": self._interface_contract_to_dict(contract),
                    "inferred_from": language.value,
                }
        
        return {"error": "Could not infer contract"}

    async def _verify_against_contract(
        self,
        code: str,
        language: Language,
        contract_id: str,
        contract_type: str,
    ) -> dict[str, Any]:
        """Verify code against an existing contract."""
        adapter = self._adapters.get(language)
        if not adapter:
            raise ValueError(f"No adapter for language: {language}")
        
        if contract_type == "function":
            contract = self._function_contracts.get(contract_id)
            if not contract:
                return {"error": f"Contract not found: {contract_id}"}
            
            # Parse the implementation
            impl = adapter.parse_function_signature(code, contract.name)
            if not impl:
                return {
                    "verified": False,
                    "error": f"Could not find function {contract.name} in code",
                }
            
            # Compare contracts
            issues = self._compare_function_contracts(contract, impl, language)
            
            return {
                "contract_id": contract_id,
                "language": language.value,
                "verified": len(issues) == 0,
                "issues": issues,
            }
        
        elif contract_type == "interface":
            contract = self._interface_contracts.get(contract_id)
            if not contract:
                return {"error": f"Contract not found: {contract_id}"}
            
            # Parse the implementation
            impl = adapter.parse_interface(code, contract.name)
            if not impl:
                return {
                    "verified": False,
                    "error": f"Could not find interface {contract.name} in code",
                }
            
            # Compare contracts
            issues = self._compare_interface_contracts(contract, impl, language)
            
            return {
                "contract_id": contract_id,
                "language": language.value,
                "verified": len(issues) == 0,
                "issues": issues,
            }
        
        return {"error": f"Unknown contract type: {contract_type}"}

    def _compare_function_contracts(
        self,
        expected: FunctionContract,
        actual: FunctionContract,
        language: Language,
    ) -> list[dict[str, Any]]:
        """Compare two function contracts."""
        issues = []
        
        # Compare parameter count
        if len(expected.parameters) != len(actual.parameters):
            issues.append({
                "type": "parameter_count_mismatch",
                "expected": len(expected.parameters),
                "actual": len(actual.parameters),
                "severity": "error",
            })
        
        # Compare parameter types
        for i, (exp_name, exp_type) in enumerate(expected.parameters):
            if i >= len(actual.parameters):
                break
            
            act_name, act_type = actual.parameters[i]
            
            if not self._types_compatible(exp_type, act_type, language):
                issues.append({
                    "type": "parameter_type_mismatch",
                    "parameter": exp_name,
                    "expected_type": exp_type.base_type,
                    "actual_type": act_type.base_type,
                    "severity": "error",
                })
        
        # Compare return type
        if expected.return_type and actual.return_type:
            if not self._types_compatible(
                expected.return_type, actual.return_type, language
            ):
                issues.append({
                    "type": "return_type_mismatch",
                    "expected": expected.return_type.base_type,
                    "actual": actual.return_type.base_type,
                    "severity": "error",
                })
        
        return issues

    def _compare_interface_contracts(
        self,
        expected: InterfaceContract,
        actual: InterfaceContract,
        language: Language,
    ) -> list[dict[str, Any]]:
        """Compare two interface contracts."""
        issues = []
        
        # Check for missing methods
        for method_name in expected.methods:
            if method_name not in actual.methods:
                issues.append({
                    "type": "missing_method",
                    "method": method_name,
                    "severity": "error",
                })
            else:
                # Compare method signatures
                method_issues = self._compare_function_contracts(
                    expected.methods[method_name],
                    actual.methods[method_name],
                    language,
                )
                for issue in method_issues:
                    issue["method"] = method_name
                    issues.append(issue)
        
        return issues

    def _types_compatible(
        self,
        type1: TypeContract,
        type2: TypeContract,
        language: Language,
    ) -> bool:
        """Check if two types are compatible."""
        # Same base type
        if type1.base_type == type2.base_type:
            return True
        
        # int/float compatibility
        if {type1.base_type, type2.base_type} == {"int", "float"}:
            return True
        
        # Any is compatible with everything
        if type1.base_type == "any" or type2.base_type == "any":
            return True
        
        return False

    def _function_contract_to_dict(
        self, contract: FunctionContract
    ) -> dict[str, Any]:
        """Convert FunctionContract to dictionary."""
        return {
            "contract_id": contract.contract_id,
            "name": contract.name,
            "description": contract.description,
            "parameters": [
                {
                    "name": name,
                    "type": self._type_contract_to_dict(type_c),
                }
                for name, type_c in contract.parameters
            ],
            "return_type": (
                self._type_contract_to_dict(contract.return_type)
                if contract.return_type else None
            ),
            "preconditions": contract.preconditions,
            "postconditions": contract.postconditions,
            "is_pure": contract.is_pure,
        }

    def _interface_contract_to_dict(
        self, contract: InterfaceContract
    ) -> dict[str, Any]:
        """Convert InterfaceContract to dictionary."""
        return {
            "contract_id": contract.contract_id,
            "name": contract.name,
            "description": contract.description,
            "methods": {
                name: self._function_contract_to_dict(method)
                for name, method in contract.methods.items()
            },
            "extends": contract.extends,
        }

    def _type_contract_to_dict(self, contract: TypeContract) -> dict[str, Any]:
        """Convert TypeContract to dictionary."""
        result = {
            "base_type": contract.base_type,
            "nullable": contract.nullable,
        }
        
        if contract.constraints:
            result["constraints"] = contract.constraints
        
        if contract.element_type:
            result["element_type"] = self._type_contract_to_dict(contract.element_type)
        
        return result

    def register_contract(
        self,
        contract: FunctionContract | InterfaceContract | TypeContract,
    ) -> str:
        """Register a contract for cross-language verification."""
        if isinstance(contract, FunctionContract):
            self._function_contracts[contract.contract_id] = contract
        elif isinstance(contract, InterfaceContract):
            self._interface_contracts[contract.contract_id] = contract
        elif isinstance(contract, TypeContract):
            self._type_contracts[contract.contract_id] = contract
        
        logger.info(
            "Contract registered",
            contract_id=contract.contract_id,
            type=type(contract).__name__,
        )
        
        return contract.contract_id

    def get_contract(self, contract_id: str) -> Any:
        """Get a contract by ID."""
        return (
            self._function_contracts.get(contract_id) or
            self._interface_contracts.get(contract_id) or
            self._type_contracts.get(contract_id)
        )

    def generate_stub(
        self,
        contract_id: str,
        language: Language,
    ) -> str | None:
        """Generate a stub implementation in the target language."""
        contract = self._function_contracts.get(contract_id)
        if not contract:
            return None
        
        adapter = self._adapters.get(language)
        if not adapter:
            return None
        
        return adapter.generate_contract_stub(contract)

    def get_type_mapping(
        self,
        base_type: str,
        source_language: Language,
        target_language: Language,
    ) -> str | None:
        """Get type mapping between languages."""
        if base_type in TYPE_MAPPINGS:
            return TYPE_MAPPINGS[base_type].get(target_language)
        return None

    def list_contracts(self) -> dict[str, list[str]]:
        """List all registered contracts."""
        return {
            "functions": list(self._function_contracts.keys()),
            "interfaces": list(self._interface_contracts.keys()),
            "types": list(self._type_contracts.keys()),
        }
