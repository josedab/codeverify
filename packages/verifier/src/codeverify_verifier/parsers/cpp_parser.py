"""C++ language parser implementation."""

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


class CppParser(CodeParser):
    """Parser for C++ source code.
    
    Uses regex-based parsing for simplicity.
    For production, consider using tree-sitter-cpp or libclang.
    """

    @property
    def language(self) -> str:
        return "cpp"

    @property
    def file_extensions(self) -> list[str]:
        return [".cpp", ".cc", ".cxx", ".hpp", ".h", ".hxx"]

    def parse(self, code: str, file_path: str = "") -> ParsedFile:
        """Parse C++ source code into structured representation."""
        errors: list[str] = []
        
        try:
            functions = self._parse_functions(code)
            classes = self._parse_classes(code)
            imports = self._parse_includes(code)
            
            return ParsedFile(
                path=file_path,
                language=self.language,
                functions=functions,
                classes=classes,
                imports=imports,
                global_variables=self._parse_globals(code),
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
        """Parse all functions from C++ code."""
        functions: list[ParsedFunction] = []
        lines = code.split("\n")
        
        # Match function declarations
        # ReturnType functionName(params) { ... }
        # virtual ReturnType functionName(params) override { ... }
        # template<typename T> ReturnType functionName(params) { ... }
        func_pattern = re.compile(
            r"^(?P<indent>\s*)"
            r"(?P<template>template\s*<[^>]+>\s*)?"
            r"(?P<modifiers>(?:static|virtual|inline|explicit|constexpr|friend|extern|const|\s)+)?"
            r"(?P<return>[\w:*&<>,\s]+?)\s+"
            r"(?P<name>~?\w+(?:::\w+)?)\s*"
            r"\((?P<params>[^)]*)\)\s*"
            r"(?P<qualifiers>(?:const|noexcept|override|final|\s)*)"
            r"\s*(?:->[\w:*&<>,\s]+)?"  # Trailing return type
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
                modifiers = match.group("modifiers") or ""
                qualifiers = match.group("qualifiers") or ""
                template = match.group("template")
                
                # Skip if it looks like a control structure
                if name in ("if", "for", "while", "switch", "catch"):
                    i += 1
                    continue
                
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
                
                # Build decorators list from modifiers
                decorators: list[str] = []
                for mod in ["static", "virtual", "inline", "explicit", "constexpr", "friend", "extern"]:
                    if mod in modifiers:
                        decorators.append(mod)
                for qual in ["const", "noexcept", "override", "final"]:
                    if qual in qualifiers:
                        decorators.append(qual)
                if template:
                    decorators.append(template.strip())
                
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
                    is_async=False,
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
        """Parse C++ function parameters."""
        parameters: list[ParsedParameter] = []
        
        if not params_str.strip():
            return parameters
        
        # Split parameters handling nested templates
        parts = self._split_parameters(params_str)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Handle default values
            default_value = None
            if "=" in part:
                part, default_value = part.rsplit("=", 1)
                default_value = default_value.strip()
                part = part.strip()
            
            # Parse type and name
            # Patterns: Type name, Type* name, Type& name, const Type& name
            tokens = part.split()
            if len(tokens) >= 2:
                name = tokens[-1].lstrip("*&")
                type_hint = " ".join(tokens[:-1])
                
                # Handle pointer/reference on name
                if tokens[-1].startswith("*"):
                    type_hint += "*"
                elif tokens[-1].startswith("&"):
                    type_hint += "&"
                
                parameters.append(ParsedParameter(
                    name=name,
                    type_hint=type_hint,
                    default_value=default_value,
                    is_optional=default_value is not None,
                ))
            elif len(tokens) == 1:
                # Just a type (like in declarations)
                parameters.append(ParsedParameter(
                    name="",
                    type_hint=tokens[0],
                ))
        
        return parameters

    def _split_parameters(self, params_str: str) -> list[str]:
        """Split parameters handling nested templates."""
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

    def _parse_classes(self, code: str) -> list[ParsedClass]:
        """Parse C++ class/struct definitions."""
        classes: list[ParsedClass] = []
        lines = code.split("\n")
        
        # Match class/struct definitions
        class_pattern = re.compile(
            r"^\s*(?:template\s*<[^>]+>\s*)?"
            r"(class|struct)\s+"
            r"(?:[\w_]+\s+)?"  # Optional declspec
            r"(\w+)"
            r"(?:\s*:\s*(?:public|protected|private)?\s*([\w:,\s]+))?"
            r"\s*{"
        )
        
        i = 0
        while i < len(lines):
            match = class_pattern.match(lines[i])
            
            if match:
                class_type = match.group(1)
                name = match.group(2)
                bases = match.group(3)
                
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
                
                # Parse methods in class body
                methods = self._parse_functions(body)
                
                # Parse base classes
                base_classes: list[str] = []
                if bases:
                    for base in bases.split(","):
                        base = base.strip()
                        # Remove access specifiers
                        for spec in ["public", "protected", "private", "virtual"]:
                            base = base.replace(spec, "").strip()
                        if base:
                            base_classes.append(base)
                
                # Extract doc comment
                docstring = self._extract_doc_comment(lines, i)
                
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

    def _parse_includes(self, code: str) -> list[ParsedImport]:
        """Parse C++ #include directives."""
        imports: list[ParsedImport] = []
        
        # #include <header>
        # #include "header"
        include_pattern = re.compile(r'^\s*#include\s*([<"])([^>"]+)[>"]', re.MULTILINE)
        
        for match in include_pattern.finditer(code):
            bracket = match.group(1)
            header = match.group(2)
            
            imports.append(ParsedImport(
                module=header,
                is_from_import=bracket == '"',  # Local include
            ))
        
        return imports

    def _parse_globals(self, code: str) -> list[str]:
        """Parse global variable declarations."""
        globals_list: list[str] = []
        
        # Global variable patterns (simplified)
        # extern Type name;
        # const Type name = value;
        global_pattern = re.compile(
            r"^\s*(?:extern|const|static|constexpr)?\s*"
            r"[\w:*&<>,\s]+\s+"
            r"(\w+)\s*[;=]",
            re.MULTILINE
        )
        
        for match in global_pattern.finditer(code):
            name = match.group(1)
            # Filter out function names and keywords
            if name not in ("if", "for", "while", "switch", "return", "class", "struct", "namespace"):
                globals_list.append(name)
        
        return globals_list[:20]  # Limit results

    def _extract_doc_comment(self, lines: list[str], func_line: int) -> str | None:
        """Extract doc comment before a function."""
        comments: list[str] = []
        i = func_line - 1
        
        # Look for Doxygen-style comments
        in_block = False
        
        while i >= 0:
            line = lines[i].strip()
            
            if line.endswith("*/"):
                in_block = True
                comment_text = line.rstrip("*/").strip()
                if comment_text:
                    comments.insert(0, comment_text)
            elif in_block:
                if line.startswith("/*") or line.startswith("/**"):
                    comments.insert(0, line.lstrip("/*").lstrip("*").strip())
                    break
                elif line.startswith("*"):
                    comment_text = line.lstrip("* ").strip()
                    if not comment_text.startswith("@"):  # Skip Doxygen tags
                        comments.insert(0, comment_text)
                else:
                    comments.insert(0, line)
            elif line.startswith("///") or line.startswith("//!"):
                comments.insert(0, line[3:].strip())
            elif line.startswith("//"):
                comments.insert(0, line[2:].strip())
            elif not line:
                if comments:
                    break
            else:
                break
            
            i -= 1
        
        return "\n".join(comments) if comments else None

    def _calculate_complexity(self, body: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        patterns = [
            r"\bif\b",
            r"\belse\s+if\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bswitch\b",
            r"\bcase\b",
            r"\bcatch\b",
            r"\b\?\s*:",  # Ternary
            r"\b&&\b",
            r"\b\|\|\b",
        ]
        
        for pattern in patterns:
            complexity += len(re.findall(pattern, body))
        
        return complexity

    def _extract_calls(self, body: str) -> list[str]:
        """Extract function calls from body."""
        calls: list[str] = []
        
        # Match function calls
        call_pattern = re.compile(r"(\w+(?:::\w+)*)\s*\(")
        
        keywords = {"if", "for", "while", "switch", "catch", "return", "throw", "new", "delete", "sizeof", "typeid", "static_cast", "dynamic_cast", "const_cast", "reinterpret_cast"}
        
        for match in call_pattern.finditer(body):
            call = match.group(1)
            func_name = call.split("::")[-1]
            if func_name not in keywords:
                calls.append(call)
        
        return list(set(calls))

    def _extract_conditions(self, body: str) -> list[str]:
        """Extract conditional expressions."""
        conditions: list[str] = []
        
        # Match conditions in if/while statements
        cond_pattern = re.compile(r"(?:if|while)\s*\((.+?)\)\s*{", re.DOTALL)
        
        for match in cond_pattern.finditer(body):
            conditions.append(match.group(1).strip())
        
        return conditions
