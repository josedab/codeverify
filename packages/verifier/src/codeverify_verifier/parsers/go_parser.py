"""Go language parser implementation."""

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


class GoParser(CodeParser):
    """Parser for Go source code.
    
    Uses regex-based parsing for simplicity.
    For production, consider using go/parser via subprocess or tree-sitter-go.
    """

    @property
    def language(self) -> str:
        return "go"

    @property
    def file_extensions(self) -> list[str]:
        return [".go"]

    def parse(self, code: str, file_path: str = "") -> ParsedFile:
        """Parse Go source code into structured representation."""
        errors: list[str] = []
        
        try:
            functions = self._parse_functions(code)
            structs = self._parse_structs(code)
            imports = self._parse_imports(code)
            
            return ParsedFile(
                path=file_path,
                language=self.language,
                functions=functions,
                classes=structs,  # Structs mapped to classes
                imports=imports,
                global_variables=self._parse_global_vars(code),
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
        """Parse all functions from Go code."""
        functions: list[ParsedFunction] = []
        lines = code.split("\n")
        
        # Match function declarations
        # func name(params) returnType { ... }
        # func (receiver Type) name(params) returnType { ... }
        func_pattern = re.compile(
            r"^(?P<indent>\s*)"
            r"func\s+"
            r"(?:\((?P<receiver>[^)]+)\)\s+)?"  # Optional receiver
            r"(?P<name>\w+)\s*"
            r"\((?P<params>[^)]*)\)\s*"
            r"(?P<return>[^{]*?)?\s*{"
        )
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = func_pattern.match(line)
            
            if match:
                name = match.group("name")
                params_str = match.group("params")
                return_type = match.group("return")
                receiver = match.group("receiver")
                
                line_start = i + 1
                
                # Parse parameters
                parameters = self._parse_parameters(params_str)
                
                # Find function body end (count braces)
                body_lines = []
                brace_count = 1
                j = i + 1
                
                # Count opening brace in first line
                brace_count = line.count("{") - line.count("}")
                
                while j < len(lines) and brace_count > 0:
                    body_lines.append(lines[j])
                    brace_count += lines[j].count("{")
                    brace_count -= lines[j].count("}")
                    j += 1
                
                line_end = j
                body = "\n".join(body_lines)
                
                # Extract docstring (comment before function)
                docstring = self._extract_docstring(lines, i)
                
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
                    is_async=False,  # Go uses goroutines differently
                    decorators=[f"receiver: {receiver}"] if receiver else [],
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
        """Parse Go function parameters."""
        parameters: list[ParsedParameter] = []
        
        if not params_str.strip():
            return parameters
        
        # Go parameter format: name type, name type, or name, name type
        # Examples: "x int", "x, y int", "ctx context.Context, data []byte"
        parts = params_str.split(",")
        
        current_names: list[str] = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Check if this part has a type
            tokens = part.split()
            
            if len(tokens) >= 2:
                # Has type: could be "x int" or "x []int" or "x *int"
                type_start = 1
                for idx, token in enumerate(tokens[1:], 1):
                    if token.startswith(("*", "[", "map", "chan", "func", "interface", "struct")):
                        type_start = idx
                        break
                    elif not token.isidentifier():
                        type_start = idx
                        break
                
                names = tokens[:type_start]
                type_hint = " ".join(tokens[type_start:])
                
                # Add pending names with this type
                for n in current_names:
                    parameters.append(ParsedParameter(
                        name=n,
                        type_hint=type_hint,
                        is_optional=False,
                    ))
                current_names = []
                
                # Add current names
                for n in names:
                    parameters.append(ParsedParameter(
                        name=n,
                        type_hint=type_hint,
                        is_optional=False,
                    ))
            else:
                # No type, just name - will get type from next part
                current_names.append(tokens[0])
        
        return parameters

    def _parse_structs(self, code: str) -> list[ParsedClass]:
        """Parse Go struct definitions."""
        structs: list[ParsedClass] = []
        lines = code.split("\n")
        
        # Match struct definitions
        struct_pattern = re.compile(r"^\s*type\s+(\w+)\s+struct\s*{")
        
        i = 0
        while i < len(lines):
            match = struct_pattern.match(lines[i])
            
            if match:
                name = match.group(1)
                line_start = i + 1
                
                # Find struct end
                brace_count = 1
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count("{")
                    brace_count -= lines[j].count("}")
                    j += 1
                
                line_end = j
                
                # Find methods for this struct
                methods = self._find_struct_methods(code, name)
                
                # Extract docstring
                docstring = self._extract_docstring(lines, i)
                
                structs.append(ParsedClass(
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    methods=methods,
                    base_classes=[],  # Go uses embedding, not inheritance
                    docstring=docstring,
                ))
                
                i = j
            else:
                i += 1
        
        return structs

    def _find_struct_methods(self, code: str, struct_name: str) -> list[ParsedFunction]:
        """Find methods associated with a struct."""
        methods: list[ParsedFunction] = []
        
        # Match methods with receiver of this struct type
        method_pattern = re.compile(
            rf"func\s+\([^)]*\*?{struct_name}\)\s+(\w+)"
        )
        
        for func in self._parse_functions(code):
            for decorator in func.decorators:
                if decorator.startswith("receiver:") and struct_name in decorator:
                    methods.append(func)
                    break
        
        return methods

    def _parse_imports(self, code: str) -> list[ParsedImport]:
        """Parse Go import statements."""
        imports: list[ParsedImport] = []
        
        # Single import: import "fmt"
        single_pattern = re.compile(r'^\s*import\s+"([^"]+)"')
        
        # Aliased import: import f "fmt"
        aliased_pattern = re.compile(r'^\s*import\s+(\w+)\s+"([^"]+)"')
        
        # Import block: import ( ... )
        block_pattern = re.compile(r'import\s*\((.*?)\)', re.DOTALL)
        
        for match in block_pattern.finditer(code):
            block = match.group(1)
            for line in block.split("\n"):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                
                # Check for alias
                alias_match = re.match(r'(\w+)\s+"([^"]+)"', line)
                if alias_match:
                    imports.append(ParsedImport(
                        module=alias_match.group(2),
                        alias=alias_match.group(1),
                    ))
                else:
                    # Plain import
                    pkg_match = re.match(r'"([^"]+)"', line)
                    if pkg_match:
                        imports.append(ParsedImport(
                            module=pkg_match.group(1),
                        ))
        
        # Single imports outside blocks
        for line in code.split("\n"):
            aliased = aliased_pattern.match(line)
            if aliased:
                imports.append(ParsedImport(
                    module=aliased.group(2),
                    alias=aliased.group(1),
                ))
            else:
                single = single_pattern.match(line)
                if single:
                    imports.append(ParsedImport(
                        module=single.group(1),
                    ))
        
        return imports

    def _parse_global_vars(self, code: str) -> list[str]:
        """Parse global variable declarations."""
        variables: list[str] = []
        
        # var name type
        var_pattern = re.compile(r"^\s*var\s+(\w+)")
        
        # const block
        const_pattern = re.compile(r"^\s*const\s+(\w+)")
        
        for line in code.split("\n"):
            var_match = var_pattern.match(line)
            if var_match:
                variables.append(var_match.group(1))
            
            const_match = const_pattern.match(line)
            if const_match:
                variables.append(const_match.group(1))
        
        return variables

    def _extract_docstring(self, lines: list[str], func_line: int) -> str | None:
        """Extract comment block before a function."""
        comments: list[str] = []
        i = func_line - 1
        
        while i >= 0:
            line = lines[i].strip()
            if line.startswith("//"):
                comments.insert(0, line[2:].strip())
                i -= 1
            elif line.startswith("/*") or line.endswith("*/"):
                # Multi-line comment - simplified handling
                comments.insert(0, line.strip("/* "))
                i -= 1
            elif not line:
                i -= 1
            else:
                break
        
        return "\n".join(comments) if comments else None

    def _calculate_complexity(self, body: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        # Count decision points
        patterns = [
            r"\bif\b",
            r"\belse\s+if\b",
            r"\bfor\b",
            r"\bswitch\b",
            r"\bcase\b",
            r"\bselect\b",
            r"\b&&\b",
            r"\b\|\|\b",
        ]
        
        for pattern in patterns:
            complexity += len(re.findall(pattern, body))
        
        return complexity

    def _extract_calls(self, body: str) -> list[str]:
        """Extract function calls from body."""
        calls: list[str] = []
        
        # Match function calls: name(...) or pkg.name(...)
        call_pattern = re.compile(r"(\w+(?:\.\w+)?)\s*\(")
        
        for match in call_pattern.finditer(body):
            call = match.group(1)
            # Filter out Go keywords
            if call not in ("if", "for", "switch", "select", "go", "defer", "return", "make", "new", "len", "cap", "append", "copy", "delete", "close"):
                calls.append(call)
        
        return list(set(calls))

    def _extract_conditions(self, body: str) -> list[str]:
        """Extract conditional expressions."""
        conditions: list[str] = []
        
        # Match conditions in if statements
        if_pattern = re.compile(r"if\s+(.+?)\s*{")
        
        for match in if_pattern.finditer(body):
            conditions.append(match.group(1).strip())
        
        return conditions
