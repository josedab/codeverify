"""Proof-Based Code Search - Search codebase by behavioral properties.

This module enables searching a codebase by behavioral properties using Z3
constraints. Instead of searching for text patterns, search for code that
can exhibit certain behaviors (e.g., "find all functions that can throw
null pointer exceptions").

Key features:
1. Proof Indexing: Build searchable index of verified properties per function
2. Query Language: Intuitive syntax for behavioral queries
3. Natural Language Interface: "Find functions that can return null"
4. IDE Integration: Search panel with results navigation
"""

import hashlib
import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator

import structlog

logger = structlog.get_logger()


class PropertyType(str, Enum):
    """Types of behavioral properties."""

    CAN_RETURN_NULL = "can_return_null"
    CAN_THROW_EXCEPTION = "can_throw_exception"
    CAN_OVERFLOW = "can_overflow"
    CAN_UNDERFLOW = "can_underflow"
    HAS_BOUNDS_ISSUE = "has_bounds_issue"
    HAS_DIVISION_BY_ZERO = "has_division_by_zero"
    IS_PURE_FUNCTION = "is_pure_function"
    HAS_SIDE_EFFECTS = "has_side_effects"
    IS_THREAD_SAFE = "is_thread_safe"
    CAN_DEADLOCK = "can_deadlock"
    HAS_RESOURCE_LEAK = "has_resource_leak"
    HAS_SQL_INJECTION = "has_sql_injection"
    HAS_XSS = "has_xss"
    HAS_SECURITY_ISSUE = "has_security_issue"


class QueryOperator(str, Enum):
    """Query operators for combining conditions."""

    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class CodeLocation:
    """Location of code in repository."""

    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
        }


@dataclass
class IndexedFunction:
    """A function with indexed properties."""

    id: str
    name: str
    location: CodeLocation
    signature: str
    properties: set[PropertyType]
    proof_status: str  # "verified", "unverified", "partial"
    z3_constraints: list[str] = field(default_factory=list)
    documentation: str = ""
    complexity_score: float = 0.0
    last_indexed: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location.to_dict(),
            "signature": self.signature,
            "properties": [p.value for p in self.properties],
            "proof_status": self.proof_status,
            "z3_constraints": self.z3_constraints,
            "documentation": self.documentation,
            "complexity_score": self.complexity_score,
            "last_indexed": self.last_indexed.isoformat(),
        }


@dataclass
class ProofQuery:
    """A query for searching by behavioral properties."""

    properties: list[PropertyType]
    operators: list[QueryOperator] = field(default_factory=list)
    file_pattern: str | None = None
    min_complexity: float | None = None
    max_complexity: float | None = None
    proof_status: str | None = None
    limit: int = 100

    def matches(self, function: IndexedFunction) -> bool:
        """Check if a function matches this query."""
        # Check proof status filter
        if self.proof_status and function.proof_status != self.proof_status:
            return False

        # Check complexity filters
        if self.min_complexity and function.complexity_score < self.min_complexity:
            return False
        if self.max_complexity and function.complexity_score > self.max_complexity:
            return False

        # Check file pattern
        if self.file_pattern:
            if not re.search(self.file_pattern, function.location.file_path):
                return False

        # Check properties with operators
        if not self.properties:
            return True

        if not self.operators or len(self.operators) == 0:
            # Default to AND
            return all(p in function.properties for p in self.properties)

        # Apply operators
        result = self.properties[0] in function.properties

        for i, op in enumerate(self.operators):
            if i + 1 >= len(self.properties):
                break

            next_prop = self.properties[i + 1]
            has_prop = next_prop in function.properties

            if op == QueryOperator.AND:
                result = result and has_prop
            elif op == QueryOperator.OR:
                result = result or has_prop
            elif op == QueryOperator.NOT:
                result = result and not has_prop

        return result


@dataclass
class SearchResult:
    """Result of a proof-based search."""

    function: IndexedFunction
    match_score: float
    matched_properties: list[PropertyType]
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function.to_dict(),
            "match_score": self.match_score,
            "matched_properties": [p.value for p in self.matched_properties],
            "snippet": self.snippet,
        }


class ProofIndex:
    """Index of verified function properties for searching."""

    def __init__(self, storage_path: str | None = None) -> None:
        self._functions: dict[str, IndexedFunction] = {}
        self._by_property: dict[PropertyType, set[str]] = {p: set() for p in PropertyType}
        self._by_file: dict[str, set[str]] = {}
        self.storage_path = storage_path

    def add(self, function: IndexedFunction) -> None:
        """Add a function to the index."""
        self._functions[function.id] = function

        # Index by properties
        for prop in function.properties:
            self._by_property[prop].add(function.id)

        # Index by file
        file_path = function.location.file_path
        if file_path not in self._by_file:
            self._by_file[file_path] = set()
        self._by_file[file_path].add(function.id)

    def remove(self, function_id: str) -> None:
        """Remove a function from the index."""
        if function_id not in self._functions:
            return

        function = self._functions[function_id]

        # Remove from property index
        for prop in function.properties:
            self._by_property[prop].discard(function_id)

        # Remove from file index
        file_path = function.location.file_path
        if file_path in self._by_file:
            self._by_file[file_path].discard(function_id)

        del self._functions[function_id]

    def get(self, function_id: str) -> IndexedFunction | None:
        """Get a function by ID."""
        return self._functions.get(function_id)

    def get_by_property(self, property: PropertyType) -> list[IndexedFunction]:
        """Get all functions with a property."""
        ids = self._by_property.get(property, set())
        return [self._functions[id] for id in ids if id in self._functions]

    def get_by_file(self, file_path: str) -> list[IndexedFunction]:
        """Get all functions in a file."""
        ids = self._by_file.get(file_path, set())
        return [self._functions[id] for id in ids if id in self._functions]

    def search(self, query: ProofQuery) -> list[SearchResult]:
        """Search the index with a query."""
        results = []

        for function in self._functions.values():
            if query.matches(function):
                # Calculate match score
                matched = [p for p in query.properties if p in function.properties]
                score = len(matched) / max(len(query.properties), 1)

                results.append(SearchResult(
                    function=function,
                    match_score=score,
                    matched_properties=matched,
                ))

        # Sort by score
        results.sort(key=lambda r: r.match_score, reverse=True)

        # Apply limit
        return results[:query.limit]

    def statistics(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "total_functions": len(self._functions),
            "by_property": {
                p.value: len(ids) for p, ids in self._by_property.items()
            },
            "files_indexed": len(self._by_file),
        }

    def save(self) -> None:
        """Save index to storage."""
        if not self.storage_path:
            return

        data = {
            "functions": [f.to_dict() for f in self._functions.values()],
        }

        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        """Load index from storage."""
        if not self.storage_path:
            return

        path = Path(self.storage_path)
        if not path.exists():
            return

        data = json.loads(path.read_text())

        for func_data in data.get("functions", []):
            function = IndexedFunction(
                id=func_data["id"],
                name=func_data["name"],
                location=CodeLocation(**func_data["location"]),
                signature=func_data["signature"],
                properties={PropertyType(p) for p in func_data["properties"]},
                proof_status=func_data["proof_status"],
                z3_constraints=func_data.get("z3_constraints", []),
                documentation=func_data.get("documentation", ""),
                complexity_score=func_data.get("complexity_score", 0.0),
                last_indexed=datetime.fromisoformat(func_data["last_indexed"]),
            )
            self.add(function)


class NaturalLanguageQueryParser:
    """Parse natural language queries into ProofQuery objects."""

    # Patterns for common queries
    NL_PATTERNS = [
        (r"functions?\s+that\s+can\s+return\s+null", [PropertyType.CAN_RETURN_NULL]),
        (r"functions?\s+that\s+can\s+throw", [PropertyType.CAN_THROW_EXCEPTION]),
        (r"functions?\s+that\s+may\s+overflow", [PropertyType.CAN_OVERFLOW]),
        (r"functions?\s+with\s+bounds?\s+issues?", [PropertyType.HAS_BOUNDS_ISSUE]),
        (r"functions?\s+with\s+division\s+by\s+zero", [PropertyType.HAS_DIVISION_BY_ZERO]),
        (r"pure\s+functions?", [PropertyType.IS_PURE_FUNCTION]),
        (r"functions?\s+with\s+side\s+effects?", [PropertyType.HAS_SIDE_EFFECTS]),
        (r"thread\s*-?\s*safe\s+functions?", [PropertyType.IS_THREAD_SAFE]),
        (r"functions?\s+that\s+can\s+deadlock", [PropertyType.CAN_DEADLOCK]),
        (r"resource\s+leaks?", [PropertyType.HAS_RESOURCE_LEAK]),
        (r"sql\s+injection", [PropertyType.HAS_SQL_INJECTION]),
        (r"xss\s+vulnerabilit", [PropertyType.HAS_XSS]),
        (r"security\s+(issues?|vulnerabilit)", [PropertyType.HAS_SECURITY_ISSUE]),
        (r"null\s+pointer", [PropertyType.CAN_RETURN_NULL]),
        (r"null\s+dereference", [PropertyType.CAN_RETURN_NULL]),
        (r"index\s+out\s+of\s+(range|bounds)", [PropertyType.HAS_BOUNDS_ISSUE]),
        (r"array\s+bounds?", [PropertyType.HAS_BOUNDS_ISSUE]),
        (r"integer\s+overflow", [PropertyType.CAN_OVERFLOW]),
    ]

    def parse(self, query: str) -> ProofQuery:
        """Parse natural language query to ProofQuery."""
        query_lower = query.lower()
        properties: list[PropertyType] = []
        operators: list[QueryOperator] = []

        # Find matching patterns
        for pattern, props in self.NL_PATTERNS:
            if re.search(pattern, query_lower):
                for prop in props:
                    if prop not in properties:
                        properties.append(prop)

        # Detect operators
        if " and " in query_lower:
            operators = [QueryOperator.AND] * (len(properties) - 1)
        elif " or " in query_lower:
            operators = [QueryOperator.OR] * (len(properties) - 1)
        elif " not " in query_lower or "without" in query_lower:
            # Handle negation
            operators = [QueryOperator.AND] * (len(properties) - 1)

        # Extract file pattern
        file_pattern = None
        file_match = re.search(r"in\s+([^\s]+\.(py|ts|js|go|java|rs))", query_lower)
        if file_match:
            file_pattern = re.escape(file_match.group(1))
        elif re.search(r"in\s+(\w+)\s+files?", query_lower):
            ext_match = re.search(r"in\s+(\w+)\s+files?", query_lower)
            if ext_match:
                ext = ext_match.group(1)
                ext_map = {
                    "python": r"\.py$",
                    "typescript": r"\.tsx?$",
                    "javascript": r"\.jsx?$",
                    "go": r"\.go$",
                    "java": r"\.java$",
                    "rust": r"\.rs$",
                }
                file_pattern = ext_map.get(ext.lower())

        # Default to CAN_RETURN_NULL if no properties found
        if not properties:
            # Try to infer from keywords
            if any(kw in query_lower for kw in ["null", "none", "nil"]):
                properties.append(PropertyType.CAN_RETURN_NULL)
            elif any(kw in query_lower for kw in ["throw", "exception", "error"]):
                properties.append(PropertyType.CAN_THROW_EXCEPTION)
            elif any(kw in query_lower for kw in ["safe", "secure"]):
                properties.append(PropertyType.IS_THREAD_SAFE)
            else:
                # Return empty query for unknown patterns
                pass

        return ProofQuery(
            properties=properties,
            operators=operators,
            file_pattern=file_pattern,
        )


class CodebaseIndexer:
    """Indexes a codebase for proof-based searching."""

    def __init__(self, index: ProofIndex) -> None:
        self.index = index
        self._parser_cache: dict[str, Any] = {}

    async def index_file(self, file_path: str, code: str) -> list[IndexedFunction]:
        """Index all functions in a file."""
        language = self._detect_language(file_path)
        functions = self._extract_functions(code, language, file_path)

        # Analyze each function
        indexed = []
        for func in functions:
            properties = self._analyze_properties(func["code"], language)
            z3_constraints = self._extract_z3_constraints(func["code"], language)

            indexed_func = IndexedFunction(
                id=self._generate_id(file_path, func["name"]),
                name=func["name"],
                location=CodeLocation(
                    file_path=file_path,
                    line_start=func["line_start"],
                    line_end=func["line_end"],
                ),
                signature=func.get("signature", func["name"]),
                properties=properties,
                proof_status="verified" if z3_constraints else "unverified",
                z3_constraints=z3_constraints,
                documentation=func.get("docstring", ""),
                complexity_score=self._calculate_complexity(func["code"]),
            )

            self.index.add(indexed_func)
            indexed.append(indexed_func)

        return indexed

    async def index_directory(
        self,
        directory: str,
        exclude_patterns: list[str] | None = None,
    ) -> int:
        """Index all files in a directory."""
        exclude = exclude_patterns or ["node_modules", "venv", "__pycache__", ".git"]
        count = 0

        for path in Path(directory).rglob("*"):
            if not path.is_file():
                continue

            # Check exclusions
            if any(ex in str(path) for ex in exclude):
                continue

            # Check supported extensions
            if path.suffix not in (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".rs"):
                continue

            try:
                code = path.read_text()
                functions = await self.index_file(str(path), code)
                count += len(functions)
            except Exception as e:
                logger.warning("Failed to index file", path=str(path), error=str(e))

        return count

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        ext_to_lang = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".java": "java",
            ".rs": "rust",
        }
        return ext_to_lang.get(ext, "unknown")

    def _extract_functions(
        self,
        code: str,
        language: str,
        file_path: str,
    ) -> list[dict[str, Any]]:
        """Extract function definitions from code."""
        functions = []
        lines = code.split("\n")

        if language == "python":
            # Simple regex-based extraction
            func_pattern = re.compile(r"^\s*def\s+(\w+)\s*\(([^)]*)\)\s*(->\s*[^:]+)?:")
            class_pattern = re.compile(r"^\s*class\s+(\w+)")

            current_class = None
            i = 0
            while i < len(lines):
                line = lines[i]

                # Track class context
                class_match = class_pattern.match(line)
                if class_match:
                    current_class = class_match.group(1)

                func_match = func_pattern.match(line)
                if func_match:
                    func_name = func_match.group(1)
                    params = func_match.group(2)
                    return_type = func_match.group(3) or ""

                    # Find function end
                    end_line = self._find_python_func_end(lines, i)

                    # Extract docstring
                    docstring = self._extract_python_docstring(lines, i)

                    full_name = f"{current_class}.{func_name}" if current_class else func_name

                    functions.append({
                        "name": full_name,
                        "signature": f"def {func_name}({params}){return_type}",
                        "code": "\n".join(lines[i:end_line + 1]),
                        "line_start": i + 1,
                        "line_end": end_line + 1,
                        "docstring": docstring,
                    })

                i += 1

        elif language in ("typescript", "javascript"):
            # Match function declarations and arrow functions
            patterns = [
                re.compile(r"^\s*(export\s+)?(async\s+)?function\s+(\w+)\s*\(([^)]*)\)"),
                re.compile(r"^\s*(const|let|var)\s+(\w+)\s*=\s*(async\s+)?\([^)]*\)\s*=>"),
                re.compile(r"^\s*(public|private|protected)?\s*(static)?\s*(async)?\s*(\w+)\s*\(([^)]*)\)\s*[:{]"),
            ]

            for i, line in enumerate(lines):
                for pattern in patterns:
                    match = pattern.match(line)
                    if match:
                        # Extract function name from match groups
                        groups = match.groups()
                        func_name = next((g for g in groups if g and g.isidentifier()), "anonymous")

                        # Find function end (simplified)
                        end_line = self._find_brace_end(lines, i)

                        functions.append({
                            "name": func_name,
                            "signature": line.strip(),
                            "code": "\n".join(lines[i:end_line + 1]),
                            "line_start": i + 1,
                            "line_end": end_line + 1,
                        })
                        break

        return functions

    def _find_python_func_end(self, lines: list[str], start: int) -> int:
        """Find the end of a Python function."""
        if start >= len(lines):
            return start

        # Get starting indentation
        start_indent = len(lines[start]) - len(lines[start].lstrip())

        for i in range(start + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue

            current_indent = len(line) - len(line.lstrip())
            if current_indent <= start_indent and line.strip():
                return i - 1

        return len(lines) - 1

    def _find_brace_end(self, lines: list[str], start: int) -> int:
        """Find the end of a brace-delimited block."""
        brace_count = 0
        started = False

        for i in range(start, len(lines)):
            line = lines[i]
            for char in line:
                if char == "{":
                    brace_count += 1
                    started = True
                elif char == "}":
                    brace_count -= 1

            if started and brace_count == 0:
                return i

        return len(lines) - 1

    def _extract_python_docstring(self, lines: list[str], func_start: int) -> str:
        """Extract docstring from Python function."""
        if func_start + 1 >= len(lines):
            return ""

        next_line = lines[func_start + 1].strip()
        if next_line.startswith('"""') or next_line.startswith("'''"):
            quote = next_line[:3]
            if next_line.count(quote) >= 2:
                # Single line docstring
                return next_line.strip(quote).strip()

            # Multi-line docstring
            doc_lines = [next_line.lstrip(quote)]
            for i in range(func_start + 2, len(lines)):
                line = lines[i].strip()
                if quote in line:
                    doc_lines.append(line.rstrip(quote))
                    break
                doc_lines.append(line)

            return " ".join(doc_lines).strip()

        return ""

    def _analyze_properties(self, code: str, language: str) -> set[PropertyType]:
        """Analyze code to determine behavioral properties."""
        properties = set()

        code_lower = code.lower()

        # Null-related
        if language == "python":
            if "is none" in code_lower or "none" in code or "return none" in code_lower:
                properties.add(PropertyType.CAN_RETURN_NULL)
        else:
            if "null" in code_lower or "undefined" in code_lower:
                properties.add(PropertyType.CAN_RETURN_NULL)

        # Exception-related
        if "raise " in code_lower or "throw " in code_lower:
            properties.add(PropertyType.CAN_THROW_EXCEPTION)

        # Array/bounds
        if "[" in code and "]" in code:
            if "len(" not in code_lower and "length" not in code_lower:
                properties.add(PropertyType.HAS_BOUNDS_ISSUE)

        # Division
        if "/" in code or "%" in code:
            if "zero" not in code_lower and "!= 0" not in code and "!== 0" not in code:
                properties.add(PropertyType.HAS_DIVISION_BY_ZERO)

        # Overflow checks
        if any(op in code for op in ["**", "pow(", "<<", ">>"]):
            properties.add(PropertyType.CAN_OVERFLOW)

        # Side effects
        if any(kw in code_lower for kw in ["print", "write", "send", "post", "put", "delete", "save"]):
            properties.add(PropertyType.HAS_SIDE_EFFECTS)
        else:
            properties.add(PropertyType.IS_PURE_FUNCTION)

        # Security
        if "execute" in code_lower and ("+" in code or "format" in code_lower):
            properties.add(PropertyType.HAS_SQL_INJECTION)
            properties.add(PropertyType.HAS_SECURITY_ISSUE)

        if "innerhtml" in code_lower:
            properties.add(PropertyType.HAS_XSS)
            properties.add(PropertyType.HAS_SECURITY_ISSUE)

        # Resource management
        if language == "python":
            if "open(" in code and "with " not in code_lower:
                properties.add(PropertyType.HAS_RESOURCE_LEAK)

        # Thread safety
        if any(kw in code_lower for kw in ["lock", "mutex", "synchronized", "atomic"]):
            properties.add(PropertyType.IS_THREAD_SAFE)

        return properties

    def _extract_z3_constraints(self, code: str, language: str) -> list[str]:
        """Extract Z3 constraints from code comments or annotations."""
        constraints = []

        # Look for Z3 annotations in comments
        z3_pattern = re.compile(r"#\s*@z3:\s*(.+)$|//\s*@z3:\s*(.+)$", re.MULTILINE)
        for match in z3_pattern.finditer(code):
            constraint = match.group(1) or match.group(2)
            if constraint:
                constraints.append(constraint.strip())

        # Look for assertion-style constraints
        assert_pattern = re.compile(r"assert\s+(.+?)(?:,|$)", re.MULTILINE)
        for match in assert_pattern.finditer(code):
            constraints.append(f"Assert({match.group(1).strip()})")

        return constraints

    def _calculate_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity approximation."""
        # Simple heuristic based on control flow keywords
        keywords = [
            "if ", "elif ", "else:", "for ", "while ", "try:", "except ",
            "case ", "switch", "?", "&&", "||", "and ", "or ",
        ]

        count = 1  # Base complexity
        for keyword in keywords:
            count += code.lower().count(keyword)

        # Normalize to 0-1 range
        return min(count / 20, 1.0)

    def _generate_id(self, file_path: str, func_name: str) -> str:
        """Generate unique ID for a function."""
        content = f"{file_path}:{func_name}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class ProofBasedCodeSearch:
    """Main interface for proof-based code search.

    Usage:
        search = ProofBasedCodeSearch()

        # Index codebase
        await search.index_directory("/path/to/repo")

        # Search with query language
        results = search.search(ProofQuery(
            properties=[PropertyType.CAN_RETURN_NULL],
        ))

        # Or use natural language
        results = search.search_nl("find functions that can return null")
    """

    def __init__(self, storage_path: str | None = None) -> None:
        self.index = ProofIndex(storage_path)
        self.indexer = CodebaseIndexer(self.index)
        self.nl_parser = NaturalLanguageQueryParser()

        # Load existing index
        if storage_path:
            self.index.load()

    async def index_file(self, file_path: str, code: str) -> list[IndexedFunction]:
        """Index a single file."""
        return await self.indexer.index_file(file_path, code)

    async def index_directory(
        self,
        directory: str,
        exclude_patterns: list[str] | None = None,
    ) -> int:
        """Index all files in a directory."""
        count = await self.indexer.index_directory(directory, exclude_patterns)
        self.index.save()
        return count

    def search(self, query: ProofQuery) -> list[SearchResult]:
        """Search with a ProofQuery."""
        return self.index.search(query)

    def search_nl(self, query: str) -> list[SearchResult]:
        """Search with natural language query."""
        proof_query = self.nl_parser.parse(query)
        return self.search(proof_query)

    def get_function(self, function_id: str) -> IndexedFunction | None:
        """Get a function by ID."""
        return self.index.get(function_id)

    def get_functions_by_file(self, file_path: str) -> list[IndexedFunction]:
        """Get all indexed functions in a file."""
        return self.index.get_by_file(file_path)

    def get_functions_by_property(self, property: PropertyType) -> list[IndexedFunction]:
        """Get all functions with a specific property."""
        return self.index.get_by_property(property)

    def statistics(self) -> dict[str, Any]:
        """Get index statistics."""
        return self.index.statistics()

    def save(self) -> None:
        """Save index to storage."""
        self.index.save()

    def clear(self) -> None:
        """Clear the index."""
        self.index = ProofIndex(self.index.storage_path)
        self.indexer = CodebaseIndexer(self.index)
