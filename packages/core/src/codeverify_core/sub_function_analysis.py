"""Sub-function Granularity Analysis Engine.

This module provides fine-grained incremental analysis capabilities:
- Statement-level change detection
- Expression-level verification targeting
- Semantic block identification
- Dependency tracking at sub-function level
- Optimized re-verification scheduling
"""

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field


class GranularityLevel(str, Enum):
    """Granularity level for analysis."""
    
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    BLOCK = "block"  # if/for/while blocks
    STATEMENT = "statement"
    EXPRESSION = "expression"


class SemanticBlockType(str, Enum):
    """Type of semantic block."""
    
    FUNCTION_DEF = "function_def"
    CLASS_DEF = "class_def"
    METHOD_DEF = "method_def"
    IF_BLOCK = "if_block"
    FOR_LOOP = "for_loop"
    WHILE_LOOP = "while_loop"
    TRY_BLOCK = "try_block"
    WITH_BLOCK = "with_block"
    MATCH_BLOCK = "match_block"
    ASSIGNMENT = "assignment"
    RETURN_STMT = "return_stmt"
    CALL_EXPR = "call_expr"
    BINARY_EXPR = "binary_expr"
    GENERIC = "generic"


@dataclass
class Position:
    """Position in source code."""
    
    line: int
    column: int
    
    def __lt__(self, other: "Position") -> bool:
        if self.line != other.line:
            return self.line < other.line
        return self.column < other.column
    
    def __le__(self, other: "Position") -> bool:
        return self == other or self < other


@dataclass
class Span:
    """Span of source code."""
    
    start: Position
    end: Position
    
    def contains(self, pos: Position) -> bool:
        return self.start <= pos <= self.end
    
    def contains_line(self, line: int) -> bool:
        return self.start.line <= line <= self.end.line
    
    def overlaps(self, other: "Span") -> bool:
        return not (self.end < other.start or other.end < self.start)
    
    @property
    def line_count(self) -> int:
        return self.end.line - self.start.line + 1


@dataclass
class SemanticBlock:
    """A semantic block of code at any granularity level."""
    
    id: str
    block_type: SemanticBlockType
    level: GranularityLevel
    name: str | None
    span: Span
    content_hash: str
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    
    # Analysis state
    needs_verification: bool = True
    last_verified_hash: str | None = None
    last_verified_at: float | None = None
    verification_findings: list[dict[str, Any]] = field(default_factory=list)
    
    # Dependencies
    depends_on: set[str] = field(default_factory=set)  # Block IDs
    depended_by: set[str] = field(default_factory=set)  # Block IDs
    uses_symbols: set[str] = field(default_factory=set)  # Variable/function names
    defines_symbols: set[str] = field(default_factory=set)
    
    def mark_dirty(self) -> None:
        """Mark this block as needing re-verification."""
        self.needs_verification = True
    
    def mark_verified(self, findings: list[dict[str, Any]]) -> None:
        """Mark this block as verified."""
        self.needs_verification = False
        self.last_verified_hash = self.content_hash
        self.last_verified_at = time.time()
        self.verification_findings = findings


@dataclass
class SymbolDefinition:
    """A symbol definition (variable, function, class)."""
    
    name: str
    kind: str  # "variable", "function", "class", "parameter"
    block_id: str
    span: Span
    type_annotation: str | None = None


@dataclass
class SymbolReference:
    """A reference to a symbol."""
    
    name: str
    block_id: str
    position: Position
    is_write: bool = False


class SubFunctionParser:
    """Parser for extracting sub-function level semantic blocks."""
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.blocks: dict[str, SemanticBlock] = {}
        self.symbols: dict[str, SymbolDefinition] = {}
        self.references: list[SymbolReference] = []
        self._content: str = ""
        self._lines: list[str] = []
    
    def parse(self, content: str) -> dict[str, SemanticBlock]:
        """Parse content into semantic blocks."""
        self._content = content
        self._lines = content.split("\n")
        self.blocks.clear()
        self.symbols.clear()
        self.references.clear()
        
        if self.language == "python":
            self._parse_python()
        elif self.language in ("typescript", "javascript"):
            self._parse_typescript()
        else:
            self._parse_generic()
        
        # Build dependency graph
        self._build_dependencies()
        
        return self.blocks
    
    def _parse_python(self) -> None:
        """Parse Python code into semantic blocks."""
        # Create root module block
        root_id = str(uuid4())
        root_block = SemanticBlock(
            id=root_id,
            block_type=SemanticBlockType.GENERIC,
            level=GranularityLevel.FILE,
            name="module",
            span=Span(
                Position(0, 0),
                Position(len(self._lines) - 1, len(self._lines[-1]) if self._lines else 0),
            ),
            content_hash=self._hash(self._content),
        )
        self.blocks[root_id] = root_block
        
        i = 0
        while i < len(self._lines):
            line = self._lines[i]
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                i += 1
                continue
            
            # Function definition
            func_match = re.match(r"^(\s*)(async\s+)?def\s+(\w+)\s*\((.*?)\)", line)
            if func_match:
                func_indent = len(func_match.group(1))
                is_async = func_match.group(2) is not None
                func_name = func_match.group(3)
                params = func_match.group(4)
                
                end_line = self._find_python_block_end(i, func_indent)
                func_content = "\n".join(self._lines[i:end_line + 1])
                
                block = SemanticBlock(
                    id=str(uuid4()),
                    block_type=SemanticBlockType.FUNCTION_DEF,
                    level=GranularityLevel.FUNCTION,
                    name=func_name,
                    span=Span(Position(i, 0), Position(end_line, len(self._lines[end_line]))),
                    content_hash=self._hash(func_content),
                    parent_id=root_id,
                )
                block.defines_symbols.add(func_name)
                
                # Parse parameters
                for param in params.split(","):
                    param = param.strip()
                    if param:
                        param_name = param.split(":")[0].split("=")[0].strip()
                        if param_name and param_name != "*" and param_name != "**":
                            block.uses_symbols.add(param_name)
                
                self.blocks[block.id] = block
                root_block.children.append(block.id)
                
                # Parse sub-blocks within function
                self._parse_python_function_body(block, i + 1, end_line, func_indent)
                
                i = end_line + 1
                continue
            
            # Class definition
            class_match = re.match(r"^(\s*)class\s+(\w+)", line)
            if class_match:
                class_indent = len(class_match.group(1))
                class_name = class_match.group(2)
                
                end_line = self._find_python_block_end(i, class_indent)
                class_content = "\n".join(self._lines[i:end_line + 1])
                
                block = SemanticBlock(
                    id=str(uuid4()),
                    block_type=SemanticBlockType.CLASS_DEF,
                    level=GranularityLevel.CLASS,
                    name=class_name,
                    span=Span(Position(i, 0), Position(end_line, len(self._lines[end_line]))),
                    content_hash=self._hash(class_content),
                    parent_id=root_id,
                )
                block.defines_symbols.add(class_name)
                
                self.blocks[block.id] = block
                root_block.children.append(block.id)
                
                i = end_line + 1
                continue
            
            # Assignment statement
            assign_match = re.match(r"^(\s*)(\w+)\s*(?::\s*\w+)?\s*=", line)
            if assign_match and "==" not in line:
                var_name = assign_match.group(2)
                
                block = SemanticBlock(
                    id=str(uuid4()),
                    block_type=SemanticBlockType.ASSIGNMENT,
                    level=GranularityLevel.STATEMENT,
                    name=var_name,
                    span=Span(Position(i, 0), Position(i, len(line))),
                    content_hash=self._hash(line),
                    parent_id=root_id,
                )
                block.defines_symbols.add(var_name)
                
                # Extract used symbols from RHS
                rhs = line.split("=", 1)[1] if "=" in line else ""
                for match in re.finditer(r"\b([a-zA-Z_]\w*)\b", rhs):
                    name = match.group(1)
                    if not name[0].isupper() and name not in ("True", "False", "None"):
                        block.uses_symbols.add(name)
                
                self.blocks[block.id] = block
                root_block.children.append(block.id)
            
            i += 1
    
    def _parse_python_function_body(
        self,
        parent: SemanticBlock,
        start_line: int,
        end_line: int,
        base_indent: int,
    ) -> None:
        """Parse the body of a Python function for sub-blocks."""
        i = start_line
        while i <= end_line:
            if i >= len(self._lines):
                break
                
            line = self._lines[i]
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            if not stripped or stripped.startswith("#"):
                i += 1
                continue
            
            # Only parse blocks at function body level (base_indent + 4)
            if indent < base_indent + 4:
                i += 1
                continue
            
            # If block
            if stripped.startswith("if ") or stripped.startswith("elif ") or stripped == "else:":
                block_end = self._find_python_block_end(i, indent)
                block_content = "\n".join(self._lines[i:block_end + 1])
                
                block = SemanticBlock(
                    id=str(uuid4()),
                    block_type=SemanticBlockType.IF_BLOCK,
                    level=GranularityLevel.BLOCK,
                    name=None,
                    span=Span(Position(i, 0), Position(block_end, len(self._lines[block_end]))),
                    content_hash=self._hash(block_content),
                    parent_id=parent.id,
                )
                
                # Extract condition symbols
                if "if " in stripped or "elif " in stripped:
                    condition = stripped.split(":", 1)[0]
                    for match in re.finditer(r"\b([a-zA-Z_]\w*)\b", condition):
                        name = match.group(1)
                        if name not in ("if", "elif", "and", "or", "not", "in", "is"):
                            block.uses_symbols.add(name)
                
                self.blocks[block.id] = block
                parent.children.append(block.id)
                i = block_end + 1
                continue
            
            # For loop
            if stripped.startswith("for "):
                block_end = self._find_python_block_end(i, indent)
                block_content = "\n".join(self._lines[i:block_end + 1])
                
                block = SemanticBlock(
                    id=str(uuid4()),
                    block_type=SemanticBlockType.FOR_LOOP,
                    level=GranularityLevel.BLOCK,
                    name=None,
                    span=Span(Position(i, 0), Position(block_end, len(self._lines[block_end]))),
                    content_hash=self._hash(block_content),
                    parent_id=parent.id,
                )
                
                # Extract loop variable and iterable
                for_match = re.match(r"for\s+(\w+)\s+in\s+(.+):", stripped)
                if for_match:
                    block.defines_symbols.add(for_match.group(1))
                    for match in re.finditer(r"\b([a-zA-Z_]\w*)\b", for_match.group(2)):
                        block.uses_symbols.add(match.group(1))
                
                self.blocks[block.id] = block
                parent.children.append(block.id)
                i = block_end + 1
                continue
            
            # While loop
            if stripped.startswith("while "):
                block_end = self._find_python_block_end(i, indent)
                block_content = "\n".join(self._lines[i:block_end + 1])
                
                block = SemanticBlock(
                    id=str(uuid4()),
                    block_type=SemanticBlockType.WHILE_LOOP,
                    level=GranularityLevel.BLOCK,
                    name=None,
                    span=Span(Position(i, 0), Position(block_end, len(self._lines[block_end]))),
                    content_hash=self._hash(block_content),
                    parent_id=parent.id,
                )
                
                self.blocks[block.id] = block
                parent.children.append(block.id)
                i = block_end + 1
                continue
            
            # Try block
            if stripped.startswith("try:"):
                block_end = self._find_python_try_end(i, indent)
                block_content = "\n".join(self._lines[i:block_end + 1])
                
                block = SemanticBlock(
                    id=str(uuid4()),
                    block_type=SemanticBlockType.TRY_BLOCK,
                    level=GranularityLevel.BLOCK,
                    name=None,
                    span=Span(Position(i, 0), Position(block_end, len(self._lines[block_end]))),
                    content_hash=self._hash(block_content),
                    parent_id=parent.id,
                )
                
                self.blocks[block.id] = block
                parent.children.append(block.id)
                i = block_end + 1
                continue
            
            # Return statement
            if stripped.startswith("return ") or stripped == "return":
                block = SemanticBlock(
                    id=str(uuid4()),
                    block_type=SemanticBlockType.RETURN_STMT,
                    level=GranularityLevel.STATEMENT,
                    name=None,
                    span=Span(Position(i, 0), Position(i, len(line))),
                    content_hash=self._hash(line),
                    parent_id=parent.id,
                )
                
                # Extract returned symbols
                if "return " in stripped:
                    return_expr = stripped[7:]
                    for match in re.finditer(r"\b([a-zA-Z_]\w*)\b", return_expr):
                        name = match.group(1)
                        if name not in ("True", "False", "None"):
                            block.uses_symbols.add(name)
                
                self.blocks[block.id] = block
                parent.children.append(block.id)
            
            i += 1
    
    def _find_python_block_end(self, start_line: int, base_indent: int) -> int:
        """Find the end line of a Python block."""
        end_line = start_line
        
        for i in range(start_line + 1, len(self._lines)):
            line = self._lines[i]
            stripped = line.strip()
            
            if not stripped:
                continue
            
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent and stripped and not stripped.startswith("#"):
                break
            
            end_line = i
        
        return end_line
    
    def _find_python_try_end(self, start_line: int, base_indent: int) -> int:
        """Find the end of a try/except/finally block."""
        end_line = start_line
        
        for i in range(start_line + 1, len(self._lines)):
            line = self._lines[i]
            stripped = line.strip()
            
            if not stripped:
                continue
            
            indent = len(line) - len(line.lstrip())
            
            # Continue through except/else/finally at same indent
            if indent == base_indent and stripped.startswith(("except", "else:", "finally:")):
                end_line = i
                continue
            
            if indent <= base_indent and stripped and not stripped.startswith("#"):
                if not stripped.startswith(("except", "else:", "finally:")):
                    break
            
            end_line = i
        
        return end_line
    
    def _parse_typescript(self) -> None:
        """Parse TypeScript/JavaScript code into semantic blocks."""
        # Similar structure to Python but with brace-based blocks
        root_id = str(uuid4())
        root_block = SemanticBlock(
            id=root_id,
            block_type=SemanticBlockType.GENERIC,
            level=GranularityLevel.FILE,
            name="module",
            span=Span(
                Position(0, 0),
                Position(len(self._lines) - 1, len(self._lines[-1]) if self._lines else 0),
            ),
            content_hash=self._hash(self._content),
        )
        self.blocks[root_id] = root_block
        
        i = 0
        while i < len(self._lines):
            line = self._lines[i]
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                i += 1
                continue
            
            # Function definition
            func_match = re.match(
                r"^(\s*)(export\s+)?(async\s+)?function\s+(\w+)|"
                r"^(\s*)(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(async\s*)?\(",
                line,
            )
            if func_match:
                groups = func_match.groups()
                func_name = groups[3] or groups[7]
                
                end_line = self._find_brace_block_end(i)
                func_content = "\n".join(self._lines[i:end_line + 1])
                
                block = SemanticBlock(
                    id=str(uuid4()),
                    block_type=SemanticBlockType.FUNCTION_DEF,
                    level=GranularityLevel.FUNCTION,
                    name=func_name,
                    span=Span(Position(i, 0), Position(end_line, len(self._lines[end_line]))),
                    content_hash=self._hash(func_content),
                    parent_id=root_id,
                )
                if func_name:
                    block.defines_symbols.add(func_name)
                
                self.blocks[block.id] = block
                root_block.children.append(block.id)
                
                i = end_line + 1
                continue
            
            # Class definition
            class_match = re.match(r"^(\s*)(export\s+)?(abstract\s+)?class\s+(\w+)", line)
            if class_match:
                class_name = class_match.group(4)
                
                end_line = self._find_brace_block_end(i)
                class_content = "\n".join(self._lines[i:end_line + 1])
                
                block = SemanticBlock(
                    id=str(uuid4()),
                    block_type=SemanticBlockType.CLASS_DEF,
                    level=GranularityLevel.CLASS,
                    name=class_name,
                    span=Span(Position(i, 0), Position(end_line, len(self._lines[end_line]))),
                    content_hash=self._hash(class_content),
                    parent_id=root_id,
                )
                block.defines_symbols.add(class_name)
                
                self.blocks[block.id] = block
                root_block.children.append(block.id)
                
                i = end_line + 1
                continue
            
            i += 1
    
    def _find_brace_block_end(self, start_line: int) -> int:
        """Find the end of a brace-delimited block."""
        brace_count = 0
        found_first = False
        
        for i in range(start_line, len(self._lines)):
            line = self._lines[i]
            for char in line:
                if char == "{":
                    brace_count += 1
                    found_first = True
                elif char == "}":
                    brace_count -= 1
                    if found_first and brace_count == 0:
                        return i
        
        return len(self._lines) - 1
    
    def _parse_generic(self) -> None:
        """Generic parsing for unsupported languages."""
        root_id = str(uuid4())
        root_block = SemanticBlock(
            id=root_id,
            block_type=SemanticBlockType.GENERIC,
            level=GranularityLevel.FILE,
            name="module",
            span=Span(
                Position(0, 0),
                Position(len(self._lines) - 1, len(self._lines[-1]) if self._lines else 0),
            ),
            content_hash=self._hash(self._content),
        )
        self.blocks[root_id] = root_block
    
    def _build_dependencies(self) -> None:
        """Build dependency relationships between blocks."""
        # Map symbols to their defining blocks
        symbol_to_block: dict[str, str] = {}
        for block in self.blocks.values():
            for symbol in block.defines_symbols:
                symbol_to_block[symbol] = block.id
        
        # Build dependencies based on symbol usage
        for block in self.blocks.values():
            for symbol in block.uses_symbols:
                if symbol in symbol_to_block:
                    defining_block_id = symbol_to_block[symbol]
                    if defining_block_id != block.id:
                        block.depends_on.add(defining_block_id)
                        self.blocks[defining_block_id].depended_by.add(block.id)
    
    def _hash(self, content: str) -> str:
        """Hash content for change detection."""
        return hashlib.md5(content.encode()).hexdigest()


class IncrementalAnalysisEngine:
    """Engine for incremental analysis at sub-function granularity.
    
    Features:
    - Fine-grained change detection
    - Minimal re-verification
    - Dependency-aware propagation
    - Priority scheduling
    """
    
    def __init__(self):
        self.parser = SubFunctionParser()
        self.blocks: dict[str, SemanticBlock] = {}
        self._content: str = ""
        self._language: str = "python"
        self._verification_queue: list[str] = []
        self._callbacks: list[Callable[[SemanticBlock, list[dict]], None]] = []
    
    def initialize(self, content: str, language: str = "python") -> None:
        """Initialize with file content."""
        self._content = content
        self._language = language
        self.parser = SubFunctionParser(language)
        self.blocks = self.parser.parse(content)
    
    def on_verification_complete(
        self,
        callback: Callable[[SemanticBlock, list[dict]], None],
    ) -> None:
        """Register callback for verification completion."""
        self._callbacks.append(callback)
    
    def apply_change(
        self,
        start_line: int,
        end_line: int,
        new_text: str,
    ) -> list[str]:
        """Apply a change and return affected block IDs.
        
        Returns:
            List of block IDs that need re-verification
        """
        affected_ids: set[str] = set()
        
        # Find directly affected blocks
        for block_id, block in self.blocks.items():
            if block.span.contains_line(start_line) or block.span.contains_line(end_line):
                affected_ids.add(block_id)
                block.mark_dirty()
        
        # Propagate to dependent blocks
        for block_id in list(affected_ids):
            block = self.blocks[block_id]
            for dependent_id in block.depended_by:
                if dependent_id in self.blocks:
                    affected_ids.add(dependent_id)
                    self.blocks[dependent_id].mark_dirty()
        
        # Re-parse the affected region
        self._reparse_region(start_line, end_line, new_text)
        
        return list(affected_ids)
    
    def _reparse_region(self, start_line: int, end_line: int, new_text: str) -> None:
        """Re-parse an affected region of code."""
        lines = self._content.split("\n")
        new_lines = new_text.split("\n")
        
        # Replace affected lines
        lines[start_line:end_line + 1] = new_lines
        self._content = "\n".join(lines)
        
        # Re-parse entire content (could be optimized for partial re-parsing)
        self.blocks = self.parser.parse(self._content)
    
    def get_verification_queue(self) -> list[SemanticBlock]:
        """Get blocks that need verification, prioritized.
        
        Priority order:
        1. Statements (smallest, fastest)
        2. Blocks (if/for/while)
        3. Functions
        4. Classes
        5. Files
        """
        dirty_blocks = [
            block for block in self.blocks.values()
            if block.needs_verification
        ]
        
        # Sort by granularity level (smallest first) and then by line number
        level_order = {
            GranularityLevel.EXPRESSION: 0,
            GranularityLevel.STATEMENT: 1,
            GranularityLevel.BLOCK: 2,
            GranularityLevel.FUNCTION: 3,
            GranularityLevel.CLASS: 4,
            GranularityLevel.FILE: 5,
        }
        
        dirty_blocks.sort(key=lambda b: (level_order[b.level], b.span.start.line))
        
        return dirty_blocks
    
    def mark_verified(self, block_id: str, findings: list[dict[str, Any]]) -> None:
        """Mark a block as verified with findings."""
        if block_id in self.blocks:
            self.blocks[block_id].mark_verified(findings)
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(self.blocks[block_id], findings)
                except Exception:
                    pass
    
    def get_block_at_position(self, line: int, column: int = 0) -> SemanticBlock | None:
        """Get the most specific block at a position."""
        pos = Position(line, column)
        candidates = []
        
        for block in self.blocks.values():
            if block.span.contains(pos):
                candidates.append(block)
        
        if not candidates:
            return None
        
        # Return the most specific (smallest span)
        return min(candidates, key=lambda b: b.span.line_count)
    
    def get_blocks_in_range(self, start_line: int, end_line: int) -> list[SemanticBlock]:
        """Get all blocks that overlap with a line range."""
        result = []
        for block in self.blocks.values():
            if block.span.contains_line(start_line) or block.span.contains_line(end_line):
                result.append(block)
            elif start_line <= block.span.start.line <= end_line:
                result.append(block)
        return result
    
    def get_affected_by_change(self, block_id: str) -> set[str]:
        """Get all blocks affected by a change to the given block."""
        if block_id not in self.blocks:
            return set()
        
        affected = {block_id}
        block = self.blocks[block_id]
        
        # Add all dependents recursively
        to_process = list(block.depended_by)
        while to_process:
            dep_id = to_process.pop()
            if dep_id not in affected and dep_id in self.blocks:
                affected.add(dep_id)
                to_process.extend(self.blocks[dep_id].depended_by)
        
        return affected
    
    def get_statistics(self) -> dict[str, Any]:
        """Get analysis statistics."""
        level_counts: dict[str, int] = defaultdict(int)
        dirty_count = 0
        
        for block in self.blocks.values():
            level_counts[block.level.value] += 1
            if block.needs_verification:
                dirty_count += 1
        
        return {
            "total_blocks": len(self.blocks),
            "dirty_blocks": dirty_count,
            "by_level": dict(level_counts),
            "language": self._language,
        }


class RealTimeFeedbackModel(BaseModel):
    """Model for real-time feedback to the IDE."""
    
    block_id: str
    status: str  # "pending", "verifying", "verified", "error"
    findings: list[dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: int = 0
    from_cache: bool = False
    priority_score: float = 0.0
    
    @classmethod
    def from_block(cls, block: SemanticBlock, status: str = "pending") -> "RealTimeFeedbackModel":
        return cls(
            block_id=block.id,
            status=status,
            findings=block.verification_findings,
            priority_score=cls._calculate_priority(block),
        )
    
    @staticmethod
    def _calculate_priority(block: SemanticBlock) -> float:
        """Calculate verification priority score (higher = more urgent)."""
        score = 0.0
        
        # Smaller blocks are faster to verify
        if block.level == GranularityLevel.STATEMENT:
            score += 10.0
        elif block.level == GranularityLevel.BLOCK:
            score += 7.0
        elif block.level == GranularityLevel.FUNCTION:
            score += 5.0
        
        # Blocks with more dependents are more important
        score += len(block.depended_by) * 2.0
        
        # Blocks that haven't been verified recently
        if block.last_verified_at is None:
            score += 5.0
        elif time.time() - block.last_verified_at > 60:
            score += 2.0
        
        return score
