"""C# language parser implementation."""

from __future__ import annotations

import re
from typing import Any

from .base import (
    CodeParser,
    ParsedClass,
    ParsedFile,
    ParsedFunction,
    ParsedImport,
    ParsedParameter,
)


class CSharpParser(CodeParser):
    """Parser for C# source code.
    
    Uses regex-based parsing for simplicity.
    For production, consider using tree-sitter-c-sharp or Roslyn.
    """

    @property
    def language(self) -> str:
        return "csharp"

    @property
    def file_extensions(self) -> list[str]:
        return [".cs"]

    def parse(self, code: str, file_path: str = "") -> ParsedFile:
        """Parse C# source code into structured representation."""
        errors: list[str] = []
        
        try:
            classes = self._parse_classes(code)
            imports = self._parse_usings(code)
            
            # Extract all methods from classes
            functions: list[ParsedFunction] = []
            for cls in classes:
                functions.extend(cls.methods)
            
            return ParsedFile(
                path=file_path,
                language=self.language,
                functions=functions,
                classes=classes,
                imports=imports,
                global_variables=[],
                errors=errors,
            )
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
            return ParsedFile(
                path=file_path,
                language=self.language,
                errors=errors,
            )

    def parse_function(self, code: str) -> ParsedFunction | None:
        """Parse a single method from code."""
        methods = self._parse_methods(code)
        return methods[0] if methods else None

    def _parse_classes(self, code: str) -> list[ParsedClass]:
        """Parse all class/struct/interface/record definitions."""
        classes: list[ParsedClass] = []
        lines = code.split("\n")
        
        # Match class definitions
        class_pattern = re.compile(
            r"^\s*(?P<attrs>(?:\[[^\]]+\]\s*)*)"
            r"(?P<modifiers>(?:public|private|protected|internal|static|abstract|sealed|partial|\s)+)?"
            r"(?P<type>class|struct|interface|record)\s+"
            r"(?P<name>\w+)"
            r"(?:<[^>]+>)?"  # Generic parameters
            r"(?:\s*:\s*(?P<bases>[^{]+))?"
            r"\s*(?:where[^{]+)?"
            r"\s*{"
        )
        
        i = 0
        while i < len(lines):
            match = class_pattern.match(lines[i])
            
            if match:
                class_type = match.group("type")
                name = match.group("name")
                bases = match.group("bases")
                modifiers = match.group("modifiers") or ""
                attrs = match.group("attrs") or ""
                
                line_start = i + 1
                
                # Find class end
                brace_count = lines[i].count("{") - lines[i].count("}")
                body_lines = []
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    body_lines.append(lines[j])
                    brace_count += lines[j].count("{")
                    brace_count -= lines[j].count("}")
                    j += 1
                
                line_end = j
                body = "\n".join(body_lines)
                
                # Parse methods
                methods = self._parse_methods(body, line_offset=line_start)
                
                # Parse base classes/interfaces
                base_classes: list[str] = []
                if bases:
                    for base in bases.split(","):
                        base = base.strip()
                        if base:
                            base_classes.append(base)
                
                # Extract XML doc comment
                docstring = self._extract_xml_doc(lines, i)
                
                classes.append(ParsedClass(
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    methods=methods,
                    base_classes=base_classes,
                    docstring=docstring,
                ))
                
                i = j
            else:
                i += 1
        
        return classes

    def _parse_methods(self, code: str, line_offset: int = 0) -> list[ParsedFunction]:
        """Parse all methods from code."""
        methods: list[ParsedFunction] = []
        lines = code.split("\n")
        
        # Match method declarations
        method_pattern = re.compile(
            r"^(?P<indent>\s*)"
            r"(?P<attrs>(?:\[[^\]]+\]\s*)*)"
            r"(?P<modifiers>(?:public|private|protected|internal|static|virtual|override|abstract|sealed|async|extern|partial|new|\s)+)?"
            r"(?P<return>[\w<>[\],?\s.]+)\s+"
            r"(?P<name>\w+)\s*"
            r"(?:<[^>]+>)?"  # Generic parameters
            r"\((?P<params>[^)]*)\)"
            r"(?:\s*where[^{;]+)?"
            r"\s*(?:{|=>)"
        )
        
        i = 0
        while i < len(lines):
            match = method_pattern.match(lines[i])
            
            if match:
                name = match.group("name")
                params_str = match.group("params")
                return_type = match.group("return")
                modifiers = match.group("modifiers") or ""
                attrs = match.group("attrs") or ""
                
                # Skip properties and constructors
                if return_type.strip() in ("get", "set", "init"):
                    i += 1
                    continue
                
                line_start = i + 1 + line_offset
                is_async = "async" in modifiers
                
                # Parse parameters
                parameters = self._parse_parameters(params_str)
                
                # Check if expression-bodied (=>)
                is_expression = "=>" in lines[i]
                
                if is_expression:
                    # Find end of expression
                    j = i
                    while j < len(lines) and ";" not in lines[j]:
                        j += 1
                    line_end = j + 1 + line_offset
                    body = "\n".join(lines[i:j+1])
                else:
                    # Find method body end
                    brace_count = lines[i].count("{") - lines[i].count("}")
                    body_lines = []
                    j = i + 1
                    
                    while j < len(lines) and brace_count > 0:
                        body_lines.append(lines[j])
                        brace_count += lines[j].count("{")
                        brace_count -= lines[j].count("}")
                        j += 1
                    
                    line_end = j + line_offset
                    body = "\n".join(body_lines)
                
                # Build decorators from attributes and modifiers
                decorators: list[str] = []
                for attr in re.findall(r"\[([^\]]+)\]", attrs):
                    decorators.append(f"[{attr}]")
                for mod in ["public", "private", "protected", "internal", "static", "virtual", "override", "abstract", "sealed", "async", "partial"]:
                    if mod in modifiers:
                        decorators.append(mod)
                
                # Extract XML doc comment
                docstring = self._extract_xml_doc(lines, i)
                
                # Calculate complexity
                complexity = self._calculate_complexity(body)
                
                # Extract method calls
                calls = self._extract_calls(body)
                
                # Extract conditions
                conditions = self._extract_conditions(body)
                
                methods.append(ParsedFunction(
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    parameters=parameters,
                    return_type=return_type.strip(),
                    docstring=docstring,
                    body=body,
                    is_async=is_async,
                    decorators=decorators,
                    complexity=complexity,
                    calls=calls,
                    conditions=conditions,
                ))
                
                i = j + 1 if not is_expression else j + 1
            else:
                i += 1
        
        return methods

    def _parse_parameters(self, params_str: str) -> list[ParsedParameter]:
        """Parse C# method parameters."""
        parameters: list[ParsedParameter] = []
        
        if not params_str.strip():
            return parameters
        
        # Split parameters handling generics
        parts = self._split_parameters(params_str)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Handle attributes on parameters
            part = re.sub(r"\[[^\]]+\]\s*", "", part)
            
            # Handle modifiers: ref, out, in, params
            modifiers = []
            for mod in ["ref", "out", "in", "params", "this"]:
                if part.startswith(mod + " "):
                    modifiers.append(mod)
                    part = part[len(mod):].strip()
            
            # Handle default values
            default_value = None
            if "=" in part:
                part, default_value = part.rsplit("=", 1)
                default_value = default_value.strip()
                part = part.strip()
            
            # Parse type and name
            tokens = part.rsplit(None, 1)
            if len(tokens) >= 2:
                type_hint = tokens[0]
                name = tokens[1]
                
                if modifiers:
                    type_hint = " ".join(modifiers) + " " + type_hint
                
                parameters.append(ParsedParameter(
                    name=name,
                    type_hint=type_hint,
                    default_value=default_value,
                    is_optional=default_value is not None or type_hint.endswith("?"),
                ))
        
        return parameters

    def _split_parameters(self, params_str: str) -> list[str]:
        """Split parameters handling nested generics."""
        parts: list[str] = []
        current = ""
        depth = 0
        
        for char in params_str:
            if char in "<([":
                depth += 1
                current += char
            elif char in ">)]":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += char
        
        if current:
            parts.append(current)
        
        return parts

    def _parse_usings(self, code: str) -> list[ParsedImport]:
        """Parse C# using directives."""
        imports: list[ParsedImport] = []
        
        # using Namespace;
        # using Alias = Namespace;
        # using static Namespace.Class;
        using_pattern = re.compile(
            r"^\s*using\s+"
            r"(?P<static>static\s+)?"
            r"(?:(?P<alias>\w+)\s*=\s*)?"
            r"(?P<namespace>[\w.]+)\s*;",
            re.MULTILINE
        )
        
        for match in using_pattern.finditer(code):
            namespace = match.group("namespace")
            alias = match.group("alias")
            is_static = bool(match.group("static"))
            
            imports.append(ParsedImport(
                module=namespace,
                alias=alias,
                is_from_import=is_static,
            ))
        
        return imports

    def _extract_xml_doc(self, lines: list[str], method_line: int) -> str | None:
        """Extract XML documentation comment."""
        comments: list[str] = []
        i = method_line - 1
        
        # Skip attributes
        while i >= 0 and lines[i].strip().startswith("["):
            i -= 1
        
        # Collect XML doc comments
        while i >= 0:
            line = lines[i].strip()
            if line.startswith("///"):
                comment = line[3:].strip()
                # Strip XML tags for cleaner output
                comment = re.sub(r"<[^>]+>", "", comment)
                if comment:
                    comments.insert(0, comment)
                i -= 1
            elif not line:
                i -= 1
            else:
                break
        
        return "\n".join(comments) if comments else None

    def _calculate_complexity(self, body: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        patterns = [
            r"\bif\b",
            r"\belse\s+if\b",
            r"\bfor\b",
            r"\bforeach\b",
            r"\bwhile\b",
            r"\bswitch\b",
            r"\bcase\b",
            r"\bcatch\b",
            r"\b\?\?",  # Null coalescing
            r"\b\?:",   # Null conditional
            r"\b\?\.",  # Null propagation
            r"\b&&\b",
            r"\b\|\|\b",
        ]
        
        for pattern in patterns:
            complexity += len(re.findall(pattern, body))
        
        return complexity

    def _extract_calls(self, body: str) -> list[str]:
        """Extract method calls from body."""
        calls: list[str] = []
        
        # Match method calls
        call_pattern = re.compile(r"(\w+(?:\.\w+)*)\s*[<(]")
        
        keywords = {"if", "for", "foreach", "while", "switch", "catch", "return", "throw", "new", "typeof", "sizeof", "nameof", "await", "lock", "using", "fixed", "checked", "unchecked"}
        
        for match in call_pattern.finditer(body):
            call = match.group(1)
            func_name = call.split(".")[-1]
            if func_name not in keywords:
                calls.append(call)
        
        return list(set(calls))

    def _extract_conditions(self, body: str) -> list[str]:
        """Extract conditional expressions."""
        conditions: list[str] = []
        
        # Match conditions in if/while statements
        cond_pattern = re.compile(r"(?:if|while)\s*\((.+?)\)\s*(?:{|$)", re.DOTALL | re.MULTILINE)
        
        for match in cond_pattern.finditer(body):
            conditions.append(match.group(1).strip())
        
        return conditions
