"""Diff-Aware Verification Scope.

Parses unified diffs to determine which functions were affected by changes,
allowing the verification engine to skip unchanged code and dramatically
reduce analysis time on typical PRs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DiffHunk:
    """A single hunk from a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FileDiff:
    """Parsed diff for a single file."""

    old_path: str
    new_path: str
    hunks: list[DiffHunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False

    @property
    def changed_lines_new(self) -> set[int]:
        """Return set of line numbers (1-indexed) changed in the new file."""
        lines: set[int] = set()
        for hunk in self.hunks:
            current_line = hunk.new_start
            for raw_line in hunk.lines:
                if raw_line.startswith("+"):
                    lines.add(current_line)
                    current_line += 1
                elif raw_line.startswith("-"):
                    pass  # Removed line doesn't exist in new file
                else:
                    current_line += 1
        return lines

    @property
    def total_additions(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.startswith("+"))

    @property
    def total_deletions(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.startswith("-"))


@dataclass
class FunctionScope:
    """A function extracted from source code with line range."""

    name: str
    line_start: int
    line_end: int
    language: str = ""


@dataclass
class VerificationScope:
    """The minimal set of functions to verify after a diff."""

    affected_functions: list[FunctionScope] = field(default_factory=list)
    new_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    total_changed_lines: int = 0
    total_source_lines: int = 0

    @property
    def scope_reduction_pct(self) -> float:
        """Percentage of code that can be skipped."""
        if self.total_source_lines == 0:
            return 0.0
        affected_lines = sum(f.line_end - f.line_start + 1 for f in self.affected_functions)
        return max(0.0, (1 - affected_lines / self.total_source_lines) * 100)


# =============================================================================
# Diff Parser
# =============================================================================

_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NEW_FILE_RE = re.compile(r"^new file mode")
_DELETED_FILE_RE = re.compile(r"^deleted file mode")
_RENAME_RE = re.compile(r"^rename (from|to) (.+)$")


def parse_unified_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff into structured FileDiff objects."""
    files: list[FileDiff] = []
    current_file: FileDiff | None = None
    current_hunk: DiffHunk | None = None

    for line in diff_text.split("\n"):
        file_match = _DIFF_FILE_RE.match(line)
        if file_match:
            if current_file is not None:
                files.append(current_file)
            current_file = FileDiff(
                old_path=file_match.group(1),
                new_path=file_match.group(2),
            )
            current_hunk = None
            continue

        if current_file is None:
            continue

        if _NEW_FILE_RE.match(line):
            current_file.is_new = True
            continue

        if _DELETED_FILE_RE.match(line):
            current_file.is_deleted = True
            continue

        rename_match = _RENAME_RE.match(line)
        if rename_match:
            current_file.is_renamed = True
            continue

        hunk_match = _HUNK_RE.match(line)
        if hunk_match:
            current_hunk = DiffHunk(
                old_start=int(hunk_match.group(1)),
                old_count=int(hunk_match.group(2) or "1"),
                new_start=int(hunk_match.group(3)),
                new_count=int(hunk_match.group(4) or "1"),
            )
            current_file.hunks.append(current_hunk)
            continue

        if current_hunk is not None and line and line[0] in ("+", "-", " "):
            current_hunk.lines.append(line)

    if current_file is not None:
        files.append(current_file)

    return files


# =============================================================================
# Function scope extraction (lightweight, regex-based)
# =============================================================================

_FUNC_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*(async\s+)?def\s+(\w+)\s*\("),
    "typescript": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[(<]"),
    "go": re.compile(r"^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\("),
    "java": re.compile(
        r"^\s*(?:public|private|protected|static|\s)+"
        r"[\w<>\[\]]+\s+(\w+)\s*\("
    ),
}


def extract_functions(source: str, language: str) -> list[FunctionScope]:
    """Extract function/method scopes from source code."""
    pattern = _FUNC_PATTERNS.get(language)
    if pattern is None:
        return []

    functions: list[FunctionScope] = []
    lines = source.split("\n")
    i = 0

    while i < len(lines):
        match = pattern.match(lines[i])
        if match:
            name = match.group(match.lastindex or 1)
            start = i + 1  # 1-indexed

            # Find function end via indentation / brace tracking
            if language == "python":
                end = _find_python_func_end(lines, i)
            else:
                end = _find_brace_func_end(lines, i)

            functions.append(
                FunctionScope(
                    name=name,
                    line_start=start,
                    line_end=end,
                    language=language,
                )
            )
            i = end
        else:
            i += 1

    return functions


def _find_python_func_end(lines: list[str], start: int) -> int:
    """Find end of a Python function using indentation."""
    if start >= len(lines):
        return start + 1

    base_indent = len(lines[start]) - len(lines[start].lstrip())
    i = start + 1
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped and (len(line) - len(stripped)) <= base_indent:
            return i  # 1-indexed
        i += 1
    return len(lines)


def _find_brace_func_end(lines: list[str], start: int) -> int:
    """Find end of a brace-delimited function."""
    depth = 0
    found_open = False
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if lines[i].count("{") > 0:
            found_open = True
        if found_open and depth <= 0:
            return i + 1  # 1-indexed
    return len(lines)


# =============================================================================
# Scope computation
# =============================================================================


def compute_verification_scope(
    diff_text: str,
    file_sources: dict[str, str],
    file_languages: dict[str, str],
) -> VerificationScope:
    """Compute the minimal verification scope from a diff.

    Args:
        diff_text: Unified diff output.
        file_sources: Map of file path → full source code (new version).
        file_languages: Map of file path → language ("python", "go", etc).

    Returns:
        VerificationScope listing affected functions.
    """
    file_diffs = parse_unified_diff(diff_text)
    scope = VerificationScope()

    total_source = 0
    for src in file_sources.values():
        total_source += src.count("\n") + 1
    scope.total_source_lines = total_source

    for fd in file_diffs:
        path = fd.new_path

        if fd.is_new:
            scope.new_files.append(path)
            continue
        if fd.is_deleted:
            scope.deleted_files.append(path)
            continue

        source = file_sources.get(path, "")
        language = file_languages.get(path, "")
        if not source or not language:
            continue

        changed = fd.changed_lines_new
        scope.total_changed_lines += len(changed)

        functions = extract_functions(source, language)
        for func in functions:
            if any(func.line_start <= ln <= func.line_end for ln in changed):
                scope.affected_functions.append(func)

    return scope
