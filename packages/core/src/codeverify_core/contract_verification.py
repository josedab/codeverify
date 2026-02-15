"""Cross-Repository Contract Extraction and Verification.

Extends the cross_repo module with code-based contract extraction
and formal compatibility verification for API contracts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# Contract Types
# =============================================================================


class ContractType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    API_ENDPOINT = "api_endpoint"
    EVENT_SCHEMA = "event_schema"
    TYPE_ALIAS = "type_alias"


class BreakingChangeType(str, Enum):
    PARAM_REMOVED = "parameter_removed"
    PARAM_TYPE_CHANGED = "parameter_type_changed"
    RETURN_TYPE_CHANGED = "return_type_changed"
    PARAM_ADDED_REQUIRED = "required_parameter_added"
    METHOD_REMOVED = "method_removed"
    FIELD_REMOVED = "field_removed"
    FIELD_TYPE_CHANGED = "field_type_changed"
    ENDPOINT_REMOVED = "endpoint_removed"
    ENDPOINT_METHOD_CHANGED = "endpoint_method_changed"


@dataclass
class Parameter:
    """A function/method parameter."""

    name: str
    type_hint: str = ""
    default: str | None = None
    required: bool = True

    @property
    def is_optional(self) -> bool:
        return self.default is not None or not self.required


@dataclass
class FunctionContract:
    """Contract for a single function/method."""

    name: str
    parameters: list[Parameter] = field(default_factory=list)
    return_type: str = ""
    is_async: bool = False
    is_public: bool = True
    decorators: list[str] = field(default_factory=list)
    docstring: str = ""

    def signature_key(self) -> str:
        """Stable signature for comparison."""
        params = ", ".join(
            f"{p.name}: {p.type_hint}" + (f" = {p.default}" if p.default else "")
            for p in self.parameters
        )
        return f"{'async ' if self.is_async else ''}{self.name}({params}) -> {self.return_type}"


@dataclass
class ClassContract:
    """Contract for a class (public methods and attributes)."""

    name: str
    methods: list[FunctionContract] = field(default_factory=list)
    attributes: list[Parameter] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)

    def public_methods(self) -> list[FunctionContract]:
        return [m for m in self.methods if m.is_public]


@dataclass
class APIEndpointContract:
    """Contract for an API endpoint."""

    path: str
    method: str  # GET, POST, etc.
    request_params: list[Parameter] = field(default_factory=list)
    response_type: str = ""
    status_codes: list[int] = field(default_factory=list)


@dataclass
class ModuleContract:
    """Aggregated contract for an entire module/file."""

    module_path: str
    language: str = "python"
    functions: list[FunctionContract] = field(default_factory=list)
    classes: list[ClassContract] = field(default_factory=list)
    endpoints: list[APIEndpointContract] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)


@dataclass
class BreakingChange:
    """A detected breaking change between two contract versions."""

    change_type: BreakingChangeType
    entity_name: str
    description: str
    old_value: str = ""
    new_value: str = ""
    severity: str = "high"  # high, critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.change_type.value,
            "entity": self.entity_name,
            "description": self.description,
            "old": self.old_value,
            "new": self.new_value,
            "severity": self.severity,
        }


# =============================================================================
# Contract Extractor
# =============================================================================


class ContractExtractor:
    """Extracts API contracts from source code.

    Supports Python, TypeScript, Go, and Java function signatures,
    class definitions, and API endpoints.

    Usage:
        extractor = ContractExtractor()
        contract = extractor.extract("python", code, "src/api/users.py")
    """

    def extract(self, language: str, code: str, module_path: str = "") -> ModuleContract:
        """Extract contracts from source code."""
        contract = ModuleContract(module_path=module_path, language=language)

        if language == "python":
            contract.functions = self._extract_python_functions(code)
            contract.classes = self._extract_python_classes(code)
            contract.endpoints = self._extract_fastapi_endpoints(code)
        elif language == "typescript":
            contract.functions = self._extract_typescript_functions(code)
            contract.endpoints = self._extract_express_endpoints(code)
        elif language == "go":
            contract.functions = self._extract_go_functions(code)
        elif language == "java":
            contract.functions = self._extract_java_functions(code)

        return contract

    def _extract_python_functions(self, code: str) -> list[FunctionContract]:
        funcs: list[FunctionContract] = []
        pattern = re.compile(
            r"^([ \t]*)(?P<async>async\s+)?def\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)"
            r"(?:\s*->\s*(?P<ret>[^:]+))?\s*:",
            re.MULTILINE,
        )
        for m in pattern.finditer(code):
            indent = len(m.group(1))
            name = m.group("name")
            is_public = not name.startswith("_")
            is_async = bool(m.group("async"))
            ret = (m.group("ret") or "").strip()

            params = self._parse_python_params(m.group("params"))
            funcs.append(
                FunctionContract(
                    name=name,
                    parameters=params,
                    return_type=ret,
                    is_async=is_async,
                    is_public=is_public and indent == 0,
                )
            )
        return funcs

    def _parse_python_params(self, params_str: str) -> list[Parameter]:
        params: list[Parameter] = []
        if not params_str.strip():
            return params

        for part in self._split_params(params_str):
            part = part.strip()
            if not part or part == "self" or part == "cls":
                continue

            if "=" in part:
                name_type, default = part.rsplit("=", 1)
                default = default.strip()
            else:
                name_type = part
                default = None

            if ":" in name_type:
                name, type_hint = name_type.split(":", 1)
            else:
                name, type_hint = name_type, ""

            params.append(
                Parameter(
                    name=name.strip(),
                    type_hint=type_hint.strip(),
                    default=default,
                    required=default is None,
                )
            )
        return params

    def _split_params(self, params_str: str) -> list[str]:
        """Split parameter string respecting nested brackets."""
        parts: list[str] = []
        depth = 0
        current = ""
        for ch in params_str:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(current)
                current = ""
                continue
            current += ch
        if current.strip():
            parts.append(current)
        return parts

    def _extract_python_classes(self, code: str) -> list[ClassContract]:
        classes: list[ClassContract] = []
        class_pattern = re.compile(r"^class\s+(\w+)(?:\(([^)]*)\))?\s*:", re.MULTILINE)
        for m in class_pattern.finditer(code):
            name = m.group(1)
            bases = [b.strip() for b in (m.group(2) or "").split(",") if b.strip()]

            # Extract methods within the class (simplified: next indented block)
            class_body_start = m.end()
            methods = self._extract_python_functions(
                code[class_body_start : class_body_start + 2000]
            )

            classes.append(
                ClassContract(
                    name=name,
                    methods=methods,
                    bases=bases,
                )
            )
        return classes

    def _extract_fastapi_endpoints(self, code: str) -> list[APIEndpointContract]:
        endpoints: list[APIEndpointContract] = []
        pattern = re.compile(
            r'@\w+\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
            re.IGNORECASE,
        )
        for m in pattern.finditer(code):
            endpoints.append(
                APIEndpointContract(
                    path=m.group(2),
                    method=m.group(1).upper(),
                )
            )
        return endpoints

    def _extract_typescript_functions(self, code: str) -> list[FunctionContract]:
        funcs: list[FunctionContract] = []
        pattern = re.compile(
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*"
            r"(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?::\s*([^{]+))?\s*\{",
            re.MULTILINE,
        )
        for m in pattern.finditer(code):
            name = m.group(1)
            ret = (m.group(3) or "").strip()
            params = self._parse_ts_params(m.group(2))
            funcs.append(
                FunctionContract(
                    name=name,
                    parameters=params,
                    return_type=ret,
                    is_async="async" in code[max(0, m.start() - 10) : m.start()],
                )
            )
        return funcs

    def _parse_ts_params(self, params_str: str) -> list[Parameter]:
        params: list[Parameter] = []
        for part in self._split_params(params_str):
            part = part.strip()
            if not part:
                continue
            optional = "?" in part
            part = part.replace("?", "")
            if ":" in part:
                name, type_hint = part.split(":", 1)
            else:
                name, type_hint = part, ""
            if "=" in type_hint:
                type_hint, default = type_hint.rsplit("=", 1)
                default = default.strip()
            else:
                default = None
            params.append(
                Parameter(
                    name=name.strip(),
                    type_hint=type_hint.strip(),
                    default=default,
                    required=not optional and default is None,
                )
            )
        return params

    def _extract_express_endpoints(self, code: str) -> list[APIEndpointContract]:
        endpoints: list[APIEndpointContract] = []
        pattern = re.compile(
            r"\.\s*(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
            re.IGNORECASE,
        )
        for m in pattern.finditer(code):
            endpoints.append(
                APIEndpointContract(
                    path=m.group(2),
                    method=m.group(1).upper(),
                )
            )
        return endpoints

    def _extract_go_functions(self, code: str) -> list[FunctionContract]:
        funcs: list[FunctionContract] = []
        pattern = re.compile(
            r"^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(([^)]*)\)"
            r"(?:\s*(?:\(([^)]*)\)|(\w[^\s{]*)))?\s*\{",
            re.MULTILINE,
        )
        for m in pattern.finditer(code):
            name = m.group(1)
            ret = (m.group(3) or m.group(4) or "").strip()
            params = self._parse_go_params(m.group(2))
            funcs.append(
                FunctionContract(
                    name=name,
                    parameters=params,
                    return_type=ret,
                    is_public=name[0].isupper(),
                )
            )
        return funcs

    def _parse_go_params(self, params_str: str) -> list[Parameter]:
        params: list[Parameter] = []
        for part in params_str.split(","):
            part = part.strip()
            if not part:
                continue
            tokens = part.split()
            if len(tokens) >= 2:
                params.append(Parameter(name=tokens[0], type_hint=" ".join(tokens[1:])))
            elif tokens:
                params.append(Parameter(name=tokens[0]))
        return params

    def _extract_java_functions(self, code: str) -> list[FunctionContract]:
        funcs: list[FunctionContract] = []
        pattern = re.compile(
            r"(?:public|protected|private)\s+(?:static\s+)?(?:(?:final|abstract)\s+)?"
            r"(\w[\w<>\[\]?,\s]*?)\s+(\w+)\s*\(([^)]*)\)",
            re.MULTILINE,
        )
        for m in pattern.finditer(code):
            ret = m.group(1).strip()
            name = m.group(2)
            params_str = m.group(3)
            params = self._parse_java_params(params_str)
            funcs.append(
                FunctionContract(
                    name=name,
                    parameters=params,
                    return_type=ret,
                    is_public="public" in code[max(0, m.start() - 20) : m.start() + 10],
                )
            )
        return funcs

    def _parse_java_params(self, params_str: str) -> list[Parameter]:
        params: list[Parameter] = []
        for part in self._split_params(params_str):
            part = part.strip()
            if not part:
                continue
            # Handle annotations
            part = re.sub(r"@\w+\s*", "", part).strip()
            tokens = part.rsplit(None, 1)
            if len(tokens) == 2:
                params.append(Parameter(name=tokens[1], type_hint=tokens[0]))
        return params


# =============================================================================
# Contract Comparator
# =============================================================================


class ContractComparator:
    """Compares two versions of a module contract and finds breaking changes.

    Usage:
        comparator = ContractComparator()
        changes = comparator.compare(old_contract, new_contract)
    """

    def compare(self, old: ModuleContract, new: ModuleContract) -> list[BreakingChange]:
        """Compare two contract versions and return breaking changes."""
        changes: list[BreakingChange] = []
        changes.extend(self._compare_functions(old.functions, new.functions))
        changes.extend(self._compare_classes(old.classes, new.classes))
        changes.extend(self._compare_endpoints(old.endpoints, new.endpoints))
        return changes

    def _compare_functions(
        self,
        old_funcs: list[FunctionContract],
        new_funcs: list[FunctionContract],
    ) -> list[BreakingChange]:
        changes: list[BreakingChange] = []
        old_map = {f.name: f for f in old_funcs if f.is_public}
        new_map = {f.name: f for f in new_funcs if f.is_public}

        # Check for removed functions
        for name in old_map:
            if name not in new_map:
                changes.append(
                    BreakingChange(
                        change_type=BreakingChangeType.METHOD_REMOVED,
                        entity_name=name,
                        description=f"Public function '{name}' was removed.",
                        old_value=old_map[name].signature_key(),
                        severity="critical",
                    )
                )

        # Check for parameter changes
        for name, old_func in old_map.items():
            new_func = new_map.get(name)
            if not new_func:
                continue

            changes.extend(self._compare_params(name, old_func, new_func))

            if old_func.return_type and new_func.return_type:
                if old_func.return_type != new_func.return_type:
                    changes.append(
                        BreakingChange(
                            change_type=BreakingChangeType.RETURN_TYPE_CHANGED,
                            entity_name=name,
                            description=f"Return type of '{name}' changed.",
                            old_value=old_func.return_type,
                            new_value=new_func.return_type,
                        )
                    )

        return changes

    def _compare_params(
        self,
        func_name: str,
        old_func: FunctionContract,
        new_func: FunctionContract,
    ) -> list[BreakingChange]:
        changes: list[BreakingChange] = []
        old_params = {p.name: p for p in old_func.parameters}
        new_params = {p.name: p for p in new_func.parameters}

        # Removed parameters
        for pname, old_p in old_params.items():
            if pname not in new_params:
                changes.append(
                    BreakingChange(
                        change_type=BreakingChangeType.PARAM_REMOVED,
                        entity_name=f"{func_name}.{pname}",
                        description=f"Parameter '{pname}' removed from '{func_name}'.",
                        old_value=f"{pname}: {old_p.type_hint}",
                    )
                )

        # New required parameters
        for pname, new_p in new_params.items():
            if pname not in old_params and new_p.required:
                changes.append(
                    BreakingChange(
                        change_type=BreakingChangeType.PARAM_ADDED_REQUIRED,
                        entity_name=f"{func_name}.{pname}",
                        description=f"Required parameter '{pname}' added to '{func_name}'.",
                        new_value=f"{pname}: {new_p.type_hint}",
                    )
                )

        # Type changes
        for pname in old_params:
            if pname in new_params:
                old_t = old_params[pname].type_hint
                new_t = new_params[pname].type_hint
                if old_t and new_t and old_t != new_t:
                    changes.append(
                        BreakingChange(
                            change_type=BreakingChangeType.PARAM_TYPE_CHANGED,
                            entity_name=f"{func_name}.{pname}",
                            description=f"Type of '{pname}' in '{func_name}' changed.",
                            old_value=old_t,
                            new_value=new_t,
                        )
                    )

        return changes

    def _compare_classes(
        self,
        old_classes: list[ClassContract],
        new_classes: list[ClassContract],
    ) -> list[BreakingChange]:
        changes: list[BreakingChange] = []
        old_map = {c.name: c for c in old_classes}
        new_map = {c.name: c for c in new_classes}

        for name, old_cls in old_map.items():
            new_cls = new_map.get(name)
            if not new_cls:
                changes.append(
                    BreakingChange(
                        change_type=BreakingChangeType.METHOD_REMOVED,
                        entity_name=name,
                        description=f"Class '{name}' was removed.",
                        severity="critical",
                    )
                )
                continue
            changes.extend(
                self._compare_functions(old_cls.public_methods(), new_cls.public_methods())
            )

        return changes

    def _compare_endpoints(
        self,
        old_eps: list[APIEndpointContract],
        new_eps: list[APIEndpointContract],
    ) -> list[BreakingChange]:
        changes: list[BreakingChange] = []
        old_map = {(e.path, e.method): e for e in old_eps}
        new_map = {(e.path, e.method): e for e in new_eps}

        for key in old_map:
            if key not in new_map:
                # Check if path exists with different method
                same_path = [k for k in new_map if k[0] == key[0]]
                if same_path:
                    changes.append(
                        BreakingChange(
                            change_type=BreakingChangeType.ENDPOINT_METHOD_CHANGED,
                            entity_name=f"{key[1]} {key[0]}",
                            description=f"HTTP method changed for '{key[0]}'.",
                            old_value=key[1],
                            new_value=same_path[0][1],
                        )
                    )
                else:
                    changes.append(
                        BreakingChange(
                            change_type=BreakingChangeType.ENDPOINT_REMOVED,
                            entity_name=f"{key[1]} {key[0]}",
                            description=f"Endpoint '{key[1]} {key[0]}' was removed.",
                            severity="critical",
                        )
                    )

        return changes
