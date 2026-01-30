"""Base code parser interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedParameter:
    """A function parameter."""

    name: str
    type_hint: str | None = None
    default_value: str | None = None
    is_optional: bool = False


@dataclass
class ParsedFunction:
    """A parsed function from source code."""

    name: str
    line_start: int
    line_end: int
    parameters: list[ParsedParameter] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    body: str = ""
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    complexity: int = 1  # Cyclomatic complexity

    # Analysis metadata
    calls: list[str] = field(default_factory=list)  # Function calls made
    assignments: list[str] = field(default_factory=list)  # Variables assigned
    conditions: list[str] = field(default_factory=list)  # Conditional expressions


@dataclass
class ParsedClass:
    """A parsed class from source code."""

    name: str
    line_start: int
    line_end: int
    methods: list[ParsedFunction] = field(default_factory=list)
    base_classes: list[str] = field(default_factory=list)
    docstring: str | None = None


@dataclass
class ParsedImport:
    """A parsed import statement."""

    module: str
    names: list[str] = field(default_factory=list)
    alias: str | None = None
    is_from_import: bool = False


@dataclass
class ParsedFile:
    """A completely parsed source file."""

    path: str
    language: str
    functions: list[ParsedFunction] = field(default_factory=list)
    classes: list[ParsedClass] = field(default_factory=list)
    imports: list[ParsedImport] = field(default_factory=list)
    global_variables: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class CodeParser(ABC):
    """Abstract base class for code parsers."""

    @property
    @abstractmethod
    def language(self) -> str:
        """Get the language this parser handles."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """Get file extensions this parser handles."""
        pass

    @abstractmethod
    def parse(self, code: str, file_path: str = "") -> ParsedFile:
        """Parse source code into structured representation."""
        pass

    @abstractmethod
    def parse_function(self, code: str) -> ParsedFunction | None:
        """Parse a single function from code."""
        pass

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        return any(file_path.endswith(ext) for ext in self.file_extensions)
