"""Rust language parser implementation."""

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


class RustParser(CodeParser):
    """Parser for Rust source code.
    
    Uses regex-based parsing for simplicity.
    For production, consider using tree-sitter-rust.
    """

    @property
    def language(self) -> str:
        return "rust"

    @property
    def file_extensions(self) -> list[str]:
        return [".rs"]

    def parse(self, code: str, file_path: str = "") -> ParsedFile:
        """Parse Rust source code into structured representation."""
        errors: list[str] = []
        
        try:
            functions = self._parse_functions(code)
            structs = self._parse_structs(code)
            imports = self._parse_imports(code)
            
            return ParsedFile(
                path=file_path,
                language=self.language,
                functions=functions,
                classes=structs,
                imports=imports,
                global_variables=self._parse_constants(code),
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
        """Parse a single function from code."""
        functions = self._parse_functions(code)
        return functions[0] if functions else None

    def _parse_functions(self, code: str) -> list[ParsedFunction]:
        """Parse all functions from Rust code."""
        functions: list[ParsedFunction] = []
        lines = code.split("\n")
        
        # Match function declarations
        # pub fn name(params) -> ReturnType { ... }
        # fn name<T>(params) -> Result<T, Error> { ... }
        # async fn name(params) -> impl Future { ... }
        func_pattern = re.compile(
            r"^(?P<indent>\s*)"
            r"(?P<attrs>(?:#\[[^\]]+\]\s*)*)"
            r"(?P<vis>pub(?:\s*\([^)]+\))?\s+)?"
            r"(?P<unsafe>unsafe\s+)?"
            r"(?P<async>async\s+)?"
            r"(?P<const>const\s+)?"
            r"fn\s+"
            r"(?P<name>\w+)"
            r"(?:<[^>]+>)?"  # Generic parameters
            r"\s*\((?P<params>[^)]*)\)"
            r"(?:\s*->\s*(?P<return>[^{]+))?"
            r"\s*(?:where[^{]+)?"
            r"\s*{"
        )
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = func_pattern.match(line)
            
            if match:
                name = match.group("name")
                params_str = match.group("params")
                return_type = match.group("return")
                is_async = bool(match.group("async"))
                is_unsafe = bool(match.group("unsafe"))
                visibility = match.group("vis")
                attrs = match.group("attrs")
                
                line_start = i + 1
                
                # Parse parameters
                parameters = self._parse_parameters(params_str)
                
                # Find function body end
                brace_count = line.count("{") - line.count("}")
                body_lines = []
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    body_lines.append(lines[j])
                    brace_count += lines[j].count("{")
                    brace_count -= lines[j].count("}")
                    j += 1
                
                line_end = j
                body = "\n".join(body_lines)
                
                # Extract doc comment
                docstring = self._extract_doc_comment(lines, i)
                
                # Build decorators list
                decorators: list[str] = []
                if visibility:
                    decorators.append(visibility.strip())
                if is_unsafe:
                    decorators.append("unsafe")
                if attrs:
                    for attr in re.findall(r"#\[([^\]]+)\]", attrs):
                        decorators.append(f"#[{attr}]")
                
                # Calculate complexity
                complexity = self._calculate_complexity(body)
                
                # Extract function calls
                calls = self._extract_calls(body)
                
                # Extract conditions
                conditions = self._extract_conditions(body)
                
                func = ParsedFunction(
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    parameters=parameters,
                    return_type=return_type.strip() if return_type else None,
                    docstring=docstring,
                    body=body,
                    is_async=is_async,
                    decorators=decorators,
                    complexity=complexity,
                    calls=calls,
                    conditions=conditions,
                )
                functions.append(func)
                
                i = j
            else:
                i += 1
        
        return functions

    def _parse_parameters(self, params_str: str) -> list[ParsedParameter]:
        """Parse Rust function parameters."""
        parameters: list[ParsedParameter] = []
        
        if not params_str.strip():
            return parameters
        
        # Handle self parameter
        params_str = params_str.strip()
        if params_str.startswith("&self") or params_str.startswith("&mut self") or params_str.startswith("self"):
            # Add self as parameter
            if params_str.startswith("&mut self"):
                parameters.append(ParsedParameter(name="self", type_hint="&mut Self"))
                params_str = params_str[9:].lstrip(",").strip()
            elif params_str.startswith("&self"):
                parameters.append(ParsedParameter(name="self", type_hint="&Self"))
                params_str = params_str[5:].lstrip(",").strip()
            elif params_str.startswith("self"):
                parameters.append(ParsedParameter(name="self", type_hint="Self"))
                params_str = params_str[4:].lstrip(",").strip()
        
        if not params_str:
            return parameters
        
        # Split parameters handling nested generics
        parts = self._split_parameters(params_str)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Rust format: name: Type or mut name: Type
            if ":" in part:
                name_part, type_part = part.split(":", 1)
                name = name_part.strip().lstrip("mut").strip()
                type_hint = type_part.strip()
                
                parameters.append(ParsedParameter(
                    name=name,
                    type_hint=type_hint,
                    is_optional=type_hint.startswith("Option<"),
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

    def _parse_structs(self, code: str) -> list[ParsedClass]:
        """Parse Rust struct and enum definitions."""
        structs: list[ParsedClass] = []
        lines = code.split("\n")
        
        # Match struct/enum definitions
        struct_pattern = re.compile(
            r"^\s*(?:pub(?:\s*\([^)]+\))?\s+)?"
            r"(?:struct|enum)\s+"
            r"(\w+)"
            r"(?:<[^>]+>)?"
            r"\s*[{;]"
        )
        
        i = 0
        while i < len(lines):
            match = struct_pattern.match(lines[i])
            
            if match:
                name = match.group(1)
                line_start = i + 1
                
                # Check if it's a unit struct (ends with ;)
                if lines[i].strip().endswith(";"):
                    line_end = line_start
                    i += 1
                else:
                    # Find struct end
                    brace_count = lines[i].count("{") - lines[i].count("}")
                    j = i + 1
                    
                    while j < len(lines) and brace_count > 0:
                        brace_count += lines[j].count("{")
                        brace_count -= lines[j].count("}")
                        j += 1
                    
                    line_end = j
                    i = j
                
                # Find impl blocks for this struct
                methods = self._find_impl_methods(code, name)
                
                # Extract doc comment
                docstring = self._extract_doc_comment(lines, line_start - 1)
                
                structs.append(ParsedClass(
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    methods=methods,
                    base_classes=[],  # Rust uses traits, not inheritance
                    docstring=docstring,
                ))
            else:
                i += 1
        
        return structs

    def _find_impl_methods(self, code: str, struct_name: str) -> list[ParsedFunction]:
        """Find methods in impl blocks for a struct."""
        methods: list[ParsedFunction] = []
        
        # Find impl blocks
        impl_pattern = re.compile(
            rf"impl(?:<[^>]+>)?\s+(?:\w+\s+for\s+)?{struct_name}(?:<[^>]+>)?\s*{{"
        )
        
        for match in impl_pattern.finditer(code):
            # Find the end of this impl block
            start = match.end()
            brace_count = 1
            end = start
            
            while end < len(code) and brace_count > 0:
                if code[end] == "{":
                    brace_count += 1
                elif code[end] == "}":
                    brace_count -= 1
                end += 1
            
            impl_body = code[start:end-1]
            impl_methods = self._parse_functions(impl_body)
            methods.extend(impl_methods)
        
        return methods

    def _parse_imports(self, code: str) -> list[ParsedImport]:
        """Parse Rust use statements."""
        imports: list[ParsedImport] = []
        
        # use crate::module::Item;
        # use std::collections::{HashMap, HashSet};
        # use super::*;
        use_pattern = re.compile(r"^\s*(?:pub\s+)?use\s+([^;]+);", re.MULTILINE)
        
        for match in use_pattern.finditer(code):
            path = match.group(1).strip()
            
            # Handle grouped imports
            if "{" in path:
                base_match = re.match(r"(.+)::\{(.+)\}", path)
                if base_match:
                    base = base_match.group(1)
                    items = [i.strip() for i in base_match.group(2).split(",")]
                    imports.append(ParsedImport(
                        module=base,
                        names=items,
                        is_from_import=True,
                    ))
            else:
                # Simple import
                parts = path.rsplit("::", 1)
                if len(parts) == 2:
                    imports.append(ParsedImport(
                        module=parts[0],
                        names=[parts[1]],
                        is_from_import=True,
                    ))
                else:
                    imports.append(ParsedImport(
                        module=path,
                    ))
        
        return imports

    def _parse_constants(self, code: str) -> list[str]:
        """Parse constant and static declarations."""
        constants: list[str] = []
        
        # const NAME: Type = value;
        # static NAME: Type = value;
        const_pattern = re.compile(r"^\s*(?:pub\s+)?(?:const|static)\s+(\w+)\s*:", re.MULTILINE)
        
        for match in const_pattern.finditer(code):
            constants.append(match.group(1))
        
        return constants

    def _extract_doc_comment(self, lines: list[str], func_line: int) -> str | None:
        """Extract doc comment before a function."""
        comments: list[str] = []
        i = func_line - 1
        
        # Skip attribute lines
        while i >= 0 and lines[i].strip().startswith("#["):
            i -= 1
        
        # Collect doc comments
        while i >= 0:
            line = lines[i].strip()
            if line.startswith("///"):
                comments.insert(0, line[3:].strip())
                i -= 1
            elif line.startswith("//!"):
                comments.insert(0, line[3:].strip())
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
            r"\bwhile\b",
            r"\bloop\b",
            r"\bmatch\b",
            r"\b=>\s*\{",  # Match arms with blocks
            r"\b&&\b",
            r"\b\|\|\b",
            r"\?",  # ? operator
        ]
        
        for pattern in patterns:
            complexity += len(re.findall(pattern, body))
        
        return complexity

    def _extract_calls(self, body: str) -> list[str]:
        """Extract function calls from body."""
        calls: list[str] = []
        
        # Match function/method calls
        call_pattern = re.compile(r"(\w+(?:::\w+)*)\s*[!]?\s*\(")
        
        keywords = {"if", "for", "while", "match", "loop", "return", "let", "mut", "fn", "impl", "struct", "enum", "trait", "type", "where", "use", "mod", "pub", "crate", "self", "super", "as", "in", "ref", "move", "async", "await", "dyn", "unsafe"}
        
        for match in call_pattern.finditer(body):
            call = match.group(1)
            parts = call.split("::")
            func_name = parts[-1]
            if func_name not in keywords:
                calls.append(call)
        
        return list(set(calls))

    def _extract_conditions(self, body: str) -> list[str]:
        """Extract conditional expressions."""
        conditions: list[str] = []
        
        # Match conditions in if/while/match
        if_pattern = re.compile(r"if\s+(?:let\s+[^=]+=\s*)?(.+?)\s*\{", re.DOTALL)
        
        for match in if_pattern.finditer(body):
            conditions.append(match.group(1).strip())
        
        return conditions
