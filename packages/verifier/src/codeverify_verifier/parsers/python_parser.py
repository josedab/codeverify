"""Python code parser using AST."""

import ast
from typing import Any

from codeverify_verifier.parsers.base import (
    CodeParser,
    ParsedClass,
    ParsedFile,
    ParsedFunction,
    ParsedImport,
    ParsedParameter,
)


class PythonParser(CodeParser):
    """Python source code parser using the ast module."""

    @property
    def language(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> list[str]:
        return [".py", ".pyi"]

    def parse(self, code: str, file_path: str = "") -> ParsedFile:
        """Parse Python source code."""
        result = ParsedFile(path=file_path, language=self.language)

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.errors.append(f"Syntax error: {e}")
            return result

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if not self._is_method(node, tree):
                    func = self._parse_function_node(node, code)
                    if func:
                        result.functions.append(func)

            elif isinstance(node, ast.ClassDef):
                cls = self._parse_class_node(node, code)
                if cls:
                    result.classes.append(cls)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append(
                        ParsedImport(
                            module=alias.name,
                            alias=alias.asname,
                            is_from_import=False,
                        )
                    )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    result.imports.append(
                        ParsedImport(
                            module=node.module,
                            names=[alias.name for alias in node.names],
                            is_from_import=True,
                        )
                    )

        return result

    def parse_function(self, code: str) -> ParsedFunction | None:
        """Parse a single function."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    return self._parse_function_node(node, code)
        except SyntaxError:
            pass
        return None

    def _is_method(self, node: ast.FunctionDef | ast.AsyncFunctionDef, tree: ast.Module) -> bool:
        """Check if a function node is a method inside a class."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                for item in parent.body:
                    if item is node:
                        return True
        return False

    def _parse_function_node(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        code: str,
    ) -> ParsedFunction:
        """Parse a function AST node."""
        # Get parameters
        parameters = []
        args = node.args

        # Positional args
        defaults_start = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            param = ParsedParameter(
                name=arg.arg,
                type_hint=self._get_annotation(arg.annotation),
                is_optional=i >= defaults_start,
            )
            if i >= defaults_start:
                default_idx = i - defaults_start
                param.default_value = ast.unparse(args.defaults[default_idx])
            parameters.append(param)

        # Keyword-only args
        kw_defaults_map = dict(zip(args.kwonlyargs, args.kw_defaults))
        for kwarg in args.kwonlyargs:
            default = kw_defaults_map.get(kwarg)
            parameters.append(
                ParsedParameter(
                    name=kwarg.arg,
                    type_hint=self._get_annotation(kwarg.annotation),
                    is_optional=default is not None,
                    default_value=ast.unparse(default) if default else None,
                )
            )

        # Get docstring
        docstring = ast.get_docstring(node)

        # Get body as string
        body_lines = code.split("\n")[node.lineno - 1 : node.end_lineno]
        body = "\n".join(body_lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node)

        # Extract function calls
        calls = self._extract_calls(node)

        # Extract assignments
        assignments = self._extract_assignments(node)

        # Extract conditions
        conditions = self._extract_conditions(node)

        return ParsedFunction(
            name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            parameters=parameters,
            return_type=self._get_annotation(node.returns),
            docstring=docstring,
            body=body,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            complexity=complexity,
            calls=calls,
            assignments=assignments,
            conditions=conditions,
        )

    def _parse_class_node(self, node: ast.ClassDef, code: str) -> ParsedClass:
        """Parse a class AST node."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                methods.append(self._parse_function_node(item, code))

        return ParsedClass(
            name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            methods=methods,
            base_classes=[ast.unparse(base) for base in node.bases],
            docstring=ast.get_docstring(node),
        )

    def _get_annotation(self, annotation: ast.expr | None) -> str | None:
        """Get string representation of type annotation."""
        if annotation is None:
            return None
        return ast.unparse(annotation)

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name as string."""
        return ast.unparse(decorator)

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                if child.ifs:
                    complexity += len(child.ifs)

        return complexity

    def _extract_calls(self, node: ast.AST) -> list[str]:
        """Extract function calls from a node."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(f"*.{child.func.attr}")
        return calls

    def _extract_assignments(self, node: ast.AST) -> list[str]:
        """Extract variable assignments."""
        assignments = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        assignments.append(target.id)
            elif isinstance(child, ast.AnnAssign):
                if isinstance(child.target, ast.Name):
                    assignments.append(child.target.id)
        return assignments

    def _extract_conditions(self, node: ast.AST) -> list[str]:
        """Extract conditional expressions."""
        conditions = []
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                conditions.append(ast.unparse(child.test))
            elif isinstance(child, ast.While):
                conditions.append(ast.unparse(child.test))
            elif isinstance(child, ast.Assert):
                conditions.append(ast.unparse(child.test))
        return conditions
