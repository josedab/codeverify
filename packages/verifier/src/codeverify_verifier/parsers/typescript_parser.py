"""TypeScript/JavaScript code parser using regex-based parsing."""

import re
from typing import Any

from codeverify_verifier.parsers.base import (
    CodeParser,
    ParsedClass,
    ParsedFile,
    ParsedFunction,
    ParsedImport,
    ParsedParameter,
)


class TypeScriptParser(CodeParser):
    """TypeScript/JavaScript parser using regex patterns.

    Note: For production use, consider using tree-sitter-typescript
    for more accurate parsing. This implementation provides basic
    parsing capabilities without external dependencies.
    """

    @property
    def language(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> list[str]:
        return [".ts", ".tsx", ".js", ".jsx"]

    def parse(self, code: str, file_path: str = "") -> ParsedFile:
        """Parse TypeScript/JavaScript source code."""
        result = ParsedFile(path=file_path, language=self.language)

        # Parse imports
        result.imports = self._parse_imports(code)

        # Parse functions
        result.functions = self._parse_functions(code)

        # Parse classes
        result.classes = self._parse_classes(code)

        return result

    def parse_function(self, code: str) -> ParsedFunction | None:
        """Parse a single function."""
        functions = self._parse_functions(code)
        return functions[0] if functions else None

    def _parse_imports(self, code: str) -> list[ParsedImport]:
        """Parse import statements."""
        imports = []

        # ES6 imports: import { x } from 'module'
        es6_pattern = r"import\s+(?:{([^}]+)}|\*\s+as\s+(\w+)|(\w+))\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(es6_pattern, code):
            named = match.group(1)
            namespace = match.group(2)
            default = match.group(3)
            module = match.group(4)

            names = []
            alias = None

            if named:
                names = [n.strip().split(" as ")[0] for n in named.split(",")]
            if namespace:
                alias = namespace
            if default:
                names = [default]

            imports.append(
                ParsedImport(
                    module=module,
                    names=names,
                    alias=alias,
                    is_from_import=True,
                )
            )

        # CommonJS requires: const x = require('module')
        cjs_pattern = r"(?:const|let|var)\s+(\w+)\s*=\s*require\(['\"]([^'\"]+)['\"]\)"
        for match in re.finditer(cjs_pattern, code):
            imports.append(
                ParsedImport(
                    module=match.group(2),
                    names=[match.group(1)],
                    is_from_import=False,
                )
            )

        return imports

    def _parse_functions(self, code: str) -> list[ParsedFunction]:
        """Parse function declarations."""
        functions = []
        lines = code.split("\n")

        # Regular function pattern
        func_pattern = r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(<[^>]*>)?\s*\(([^)]*)\)\s*(?::\s*([^{]+))?\s*{"

        # Arrow function pattern
        arrow_pattern = r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\(?([^)]*)\)?\s*(?::\s*([^=]+))?\s*=>"

        for pattern, is_arrow in [(func_pattern, False), (arrow_pattern, True)]:
            for match in re.finditer(pattern, code):
                name = match.group(1)
                params_str = match.group(3) if not is_arrow else match.group(2)
                return_type = match.group(4) if not is_arrow else match.group(3)

                # Find line number
                start_pos = match.start()
                line_start = code[:start_pos].count("\n") + 1

                # Parse parameters
                parameters = self._parse_parameters(params_str)

                # Find function end (approximate)
                line_end = self._find_block_end(lines, line_start - 1)

                # Check if async
                is_async = "async" in match.group(0)

                functions.append(
                    ParsedFunction(
                        name=name,
                        line_start=line_start,
                        line_end=line_end,
                        parameters=parameters,
                        return_type=return_type.strip() if return_type else None,
                        is_async=is_async,
                        body="\n".join(lines[line_start - 1 : line_end]),
                    )
                )

        return functions

    def _parse_classes(self, code: str) -> list[ParsedClass]:
        """Parse class declarations."""
        classes = []
        lines = code.split("\n")

        class_pattern = r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*{"

        for match in re.finditer(class_pattern, code):
            name = match.group(1)
            extends = match.group(2)
            implements = match.group(3)

            start_pos = match.start()
            line_start = code[:start_pos].count("\n") + 1
            line_end = self._find_block_end(lines, line_start - 1)

            # Get class body and parse methods
            class_body = "\n".join(lines[line_start - 1 : line_end])
            methods = self._parse_methods(class_body, line_start)

            base_classes = []
            if extends:
                base_classes.append(extends)
            if implements:
                base_classes.extend([i.strip() for i in implements.split(",")])

            classes.append(
                ParsedClass(
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    methods=methods,
                    base_classes=base_classes,
                )
            )

        return classes

    def _parse_methods(self, class_body: str, class_start: int) -> list[ParsedFunction]:
        """Parse methods within a class."""
        methods = []

        # Method pattern: async? methodName(params): returnType {
        method_pattern = r"(?:public|private|protected|static|async|\s)*(\w+)\s*\(([^)]*)\)\s*(?::\s*([^{]+))?\s*{"

        for match in re.finditer(method_pattern, class_body):
            name = match.group(1)
            if name in ("constructor", "class", "if", "for", "while", "switch"):
                if name != "constructor":
                    continue

            params_str = match.group(2)
            return_type = match.group(3)

            start_pos = match.start()
            line_offset = class_body[:start_pos].count("\n")
            line_start = class_start + line_offset

            parameters = self._parse_parameters(params_str)
            is_async = "async" in match.group(0)

            methods.append(
                ParsedFunction(
                    name=name,
                    line_start=line_start,
                    line_end=line_start + 10,  # Approximate
                    parameters=parameters,
                    return_type=return_type.strip() if return_type else None,
                    is_async=is_async,
                )
            )

        return methods

    def _parse_parameters(self, params_str: str) -> list[ParsedParameter]:
        """Parse function parameters."""
        if not params_str.strip():
            return []

        parameters = []
        # Simple split - doesn't handle complex nested types well
        for param in params_str.split(","):
            param = param.strip()
            if not param:
                continue

            # Handle destructuring - skip for now
            if param.startswith("{") or param.startswith("["):
                parameters.append(ParsedParameter(name="destructured"))
                continue

            # Check for optional (?)
            is_optional = "?" in param
            param = param.replace("?", "")

            # Check for default value
            default_value = None
            if "=" in param:
                param, default_value = param.split("=", 1)
                default_value = default_value.strip()
                is_optional = True

            # Parse name and type
            if ":" in param:
                name, type_hint = param.split(":", 1)
                parameters.append(
                    ParsedParameter(
                        name=name.strip(),
                        type_hint=type_hint.strip(),
                        default_value=default_value,
                        is_optional=is_optional,
                    )
                )
            else:
                parameters.append(
                    ParsedParameter(
                        name=param.strip(),
                        default_value=default_value,
                        is_optional=is_optional,
                    )
                )

        return parameters

    def _find_block_end(self, lines: list[str], start_line: int) -> int:
        """Find the end of a code block by counting braces."""
        brace_count = 0
        in_string = False
        string_char = None

        for i, line in enumerate(lines[start_line:], start=start_line):
            j = 0
            while j < len(line):
                char = line[j]

                # Handle strings
                if char in ('"', "'", "`") and (j == 0 or line[j - 1] != "\\"):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None

                # Count braces outside strings
                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            return i + 1

                j += 1

        return len(lines)
