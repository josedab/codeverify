"""Java language parser implementation."""

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


class JavaParser(CodeParser):
    """Parser for Java source code.
    
    Uses regex-based parsing for simplicity.
    For production, consider using tree-sitter-java or javalang.
    """

    @property
    def language(self) -> str:
        return "java"

    @property
    def file_extensions(self) -> list[str]:
        return [".java"]

    def parse(self, code: str, file_path: str = "") -> ParsedFile:
        """Parse Java source code into structured representation."""
        errors: list[str] = []
        
        try:
            classes = self._parse_classes(code)
            imports = self._parse_imports(code)
            
            # Extract top-level methods (if any) - rare in Java
            functions: list[ParsedFunction] = []
            for cls in classes:
                functions.extend(cls.methods)
            
            return ParsedFile(
                path=file_path,
                language=self.language,
                functions=functions,
                classes=classes,
                imports=imports,
                global_variables=self._parse_fields(code),
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
        """Parse all class definitions from Java code."""
        classes: list[ParsedClass] = []
        lines = code.split("\n")
        
        # Match class declarations
        # public class Name extends Parent implements Interface {
        class_pattern = re.compile(
            r"^\s*(?P<modifiers>(?:public|private|protected|abstract|final|static|\s)+)?"
            r"\s*(?:class|interface|enum)\s+"
            r"(?P<name>\w+)"
            r"(?:\s*<[^>]+>)?"  # Generic type parameters
            r"(?:\s+extends\s+(?P<extends>[\w.<>,\s]+))?"
            r"(?:\s+implements\s+(?P<implements>[\w.<>,\s]+))?"
            r"\s*{"
        )
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = class_pattern.match(line)
            
            if match:
                name = match.group("name")
                extends = match.group("extends")
                implements = match.group("implements")
                
                line_start = i + 1
                
                # Find class body end
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
                
                # Parse methods in class body
                methods = self._parse_methods(body, line_offset=line_start)
                
                # Build base classes list
                base_classes: list[str] = []
                if extends:
                    base_classes.extend([b.strip() for b in extends.split(",")])
                if implements:
                    base_classes.extend([b.strip() for b in implements.split(",")])
                
                # Extract Javadoc
                docstring = self._extract_javadoc(lines, i)
                
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
        """Parse all methods from Java code."""
        methods: list[ParsedFunction] = []
        lines = code.split("\n")
        
        # Match method declarations
        method_pattern = re.compile(
            r"^\s*(?P<annotations>(?:@\w+(?:\([^)]*\))?\s+)*)"
            r"(?P<modifiers>(?:public|private|protected|abstract|final|static|synchronized|native|\s)+)?"
            r"\s*(?P<generics><[^>]+>\s+)?"
            r"(?P<return>[\w.<>,\[\]\s?]+)\s+"
            r"(?P<name>\w+)\s*"
            r"\((?P<params>[^)]*)\)"
            r"(?:\s+throws\s+(?P<throws>[\w,\s]+))?"
            r"\s*(?:{|;)"
        )
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = method_pattern.match(line)
            
            if match:
                name = match.group("name")
                return_type = match.group("return")
                params_str = match.group("params")
                annotations = match.group("annotations")
                modifiers = match.group("modifiers") or ""
                throws = match.group("throws")
                
                # Skip if this is a constructor (return type matches a class name pattern)
                if name[0].isupper() and not return_type:
                    i += 1
                    continue
                
                line_start = i + 1 + line_offset
                
                # Parse parameters
                parameters = self._parse_parameters(params_str)
                
                # Check if abstract/interface method (no body)
                is_abstract = "abstract" in modifiers or line.strip().endswith(";")
                
                body = ""
                if not is_abstract and "{" in line:
                    # Find method body end
                    brace_count = line.count("{") - line.count("}")
                    body_lines = []
                    j = i + 1
                    
                    while j < len(lines) and brace_count > 0:
                        body_lines.append(lines[j])
                        brace_count += lines[j].count("{")
                        brace_count -= lines[j].count("}")
                        j += 1
                    
                    line_end = j + line_offset
                    body = "\n".join(body_lines)
                else:
                    line_end = line_start
                    j = i + 1
                
                # Parse annotations as decorators
                decorators: list[str] = []
                if annotations:
                    for ann in re.findall(r"@(\w+)(?:\([^)]*\))?", annotations):
                        decorators.append(f"@{ann}")
                
                if throws:
                    decorators.append(f"throws: {throws}")
                
                # Extract Javadoc
                docstring = self._extract_javadoc(lines, i)
                
                # Calculate complexity
                complexity = self._calculate_complexity(body)
                
                # Extract function calls
                calls = self._extract_calls(body)
                
                # Extract conditions
                conditions = self._extract_conditions(body)
                
                methods.append(ParsedFunction(
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    parameters=parameters,
                    return_type=return_type.strip() if return_type else None,
                    docstring=docstring,
                    body=body,
                    is_async="async" in modifiers,  # Java doesn't have async keyword
                    decorators=decorators,
                    complexity=complexity,
                    calls=calls,
                    conditions=conditions,
                ))
                
                i = j if not is_abstract else i + 1
            else:
                i += 1
        
        return methods

    def _parse_parameters(self, params_str: str) -> list[ParsedParameter]:
        """Parse Java method parameters."""
        parameters: list[ParsedParameter] = []
        
        if not params_str.strip():
            return parameters
        
        # Java parameter format: Type name, Type name
        # Handle generics: List<String> items
        # Handle varargs: String... args
        # Handle annotations: @NotNull String name
        
        # Simple split by comma (doesn't handle all edge cases with generics)
        parts = self._split_parameters(params_str)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Remove annotations
            while part.startswith("@"):
                # Skip annotation
                match = re.match(r"@\w+(?:\([^)]*\))?\s*", part)
                if match:
                    part = part[match.end():]
                else:
                    break
            
            # Handle final modifier
            if part.startswith("final "):
                part = part[6:]
            
            tokens = part.rsplit(None, 1)  # Split from right to get name
            
            if len(tokens) >= 2:
                type_hint = tokens[0]
                name = tokens[1]
                
                # Handle varargs
                is_varargs = "..." in type_hint
                if is_varargs:
                    type_hint = type_hint.replace("...", "[]")
                
                parameters.append(ParsedParameter(
                    name=name,
                    type_hint=type_hint,
                    is_optional=False,
                ))
            elif len(tokens) == 1:
                # Just a type? Might be broken parameter
                parameters.append(ParsedParameter(
                    name=tokens[0],
                    type_hint=None,
                ))
        
        return parameters

    def _split_parameters(self, params_str: str) -> list[str]:
        """Split parameters handling nested generics."""
        parts: list[str] = []
        current = ""
        depth = 0
        
        for char in params_str:
            if char == "<":
                depth += 1
                current += char
            elif char == ">":
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

    def _parse_imports(self, code: str) -> list[ParsedImport]:
        """Parse Java import statements."""
        imports: list[ParsedImport] = []
        
        # import package.Class;
        # import package.*;
        # import static package.Class.method;
        import_pattern = re.compile(r"^\s*import\s+(static\s+)?([^;]+);")
        
        for line in code.split("\n"):
            match = import_pattern.match(line)
            if match:
                is_static = bool(match.group(1))
                import_path = match.group(2).strip()
                
                # Split into module and names
                parts = import_path.rsplit(".", 1)
                if len(parts) == 2:
                    module = parts[0]
                    name = parts[1]
                else:
                    module = import_path
                    name = "*"
                
                imports.append(ParsedImport(
                    module=module,
                    names=[name] if name != "*" else [],
                    is_from_import=is_static,
                ))
        
        return imports

    def _parse_fields(self, code: str) -> list[str]:
        """Parse class field declarations."""
        fields: list[str] = []
        
        # Match field declarations (simplified)
        field_pattern = re.compile(
            r"^\s*(?:public|private|protected|static|final|\s)+"
            r"[\w.<>,\[\]]+\s+"
            r"(\w+)\s*[;=]"
        )
        
        for line in code.split("\n"):
            match = field_pattern.match(line)
            if match:
                fields.append(match.group(1))
        
        return fields

    def _extract_javadoc(self, lines: list[str], method_line: int) -> str | None:
        """Extract Javadoc comment before a method."""
        comments: list[str] = []
        i = method_line - 1
        
        # Skip annotations
        while i >= 0 and lines[i].strip().startswith("@"):
            i -= 1
        
        # Look for Javadoc /** ... */
        in_javadoc = False
        
        while i >= 0:
            line = lines[i].strip()
            
            if line.endswith("*/"):
                in_javadoc = True
                comments.insert(0, line.rstrip("*/").strip())
            elif in_javadoc:
                if line.startswith("/**"):
                    comments.insert(0, line.lstrip("/**").strip())
                    break
                elif line.startswith("*"):
                    comments.insert(0, line.lstrip("* ").strip())
                else:
                    comments.insert(0, line)
            elif not line:
                i -= 1
                continue
            else:
                break
            
            i -= 1
        
        # Filter out empty lines and @param/@return tags for cleaner output
        filtered = [c for c in comments if c and not c.startswith("@")]
        return "\n".join(filtered) if filtered else None

    def _calculate_complexity(self, body: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        patterns = [
            r"\bif\b",
            r"\belse\s+if\b",
            r"\bfor\b",
            r"\bwhile\b",
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
        """Extract method calls from body."""
        calls: list[str] = []
        
        # Match method calls: name(...) or obj.name(...)
        call_pattern = re.compile(r"(\w+(?:\.\w+)*)\s*\(")
        
        keywords = {"if", "for", "while", "switch", "catch", "synchronized", "new", "return", "throw", "assert"}
        
        for match in call_pattern.finditer(body):
            call = match.group(1)
            # Get just the method name (last part)
            method_name = call.split(".")[-1]
            if method_name not in keywords:
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
