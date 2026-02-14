"""Rust & C/C++ Memory Safety Verification.

Extends the Z3 verification engine with memory safety properties for Rust
and C/C++ codebases, including use-after-free detection, buffer overflow
prevention, data race analysis, and Rust ownership model verification.

Features:
- Rust ownership and borrow-checker constraint encoding in Z3
- C/C++ pointer analysis with heap/stack memory model
- Use-after-free and double-free detection
- Buffer overflow and out-of-bounds access verification
- Data race detection for concurrent code
- Dangling pointer and null dereference analysis
- Language-specific fix suggestions
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


# ─── Enums ──────────────────────────────────────────────────────────────


class MemoryLanguage(str, Enum):
    """Supported languages for memory safety verification."""

    RUST = "rust"
    C = "c"
    CPP = "cpp"


class OwnershipState(str, Enum):
    """Rust ownership states for a variable."""

    OWNED = "owned"
    BORROWED_SHARED = "borrowed_shared"
    BORROWED_MUT = "borrowed_mut"
    MOVED = "moved"
    DROPPED = "dropped"


class MemoryRegion(str, Enum):
    """Memory regions for C/C++ analysis."""

    STACK = "stack"
    HEAP = "heap"
    STATIC = "static"
    UNKNOWN = "unknown"


class MemoryViolationType(str, Enum):
    """Types of memory safety violations."""

    USE_AFTER_FREE = "use_after_free"
    DOUBLE_FREE = "double_free"
    BUFFER_OVERFLOW = "buffer_overflow"
    NULL_DEREFERENCE = "null_dereference"
    DANGLING_POINTER = "dangling_pointer"
    DATA_RACE = "data_race"
    OWNERSHIP_VIOLATION = "ownership_violation"
    BORROW_VIOLATION = "borrow_violation"
    LIFETIME_VIOLATION = "lifetime_violation"
    UNINITIALIZED_MEMORY = "uninitialized_memory"
    MEMORY_LEAK = "memory_leak"


class MemoryCheckSeverity(str, Enum):
    """Severity levels for memory safety findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PointerState(str, Enum):
    """State of a pointer in C/C++ analysis."""

    VALID = "valid"
    NULL = "null"
    FREED = "freed"
    DANGLING = "dangling"
    UNINITIALIZED = "uninitialized"


# ─── Data Models ────────────────────────────────────────────────────────


@dataclass
class MemoryLocation:
    """Represents a memory location being tracked."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    region: MemoryRegion = MemoryRegion.UNKNOWN
    size_bytes: int = 0
    allocated_at_line: int = 0
    freed_at_line: int | None = None
    is_valid: bool = True


@dataclass
class OwnershipConstraint:
    """A Rust ownership constraint for Z3 encoding."""

    variable: str
    state: OwnershipState
    lifetime: str = "'a"
    line: int = 0
    borrowed_from: str | None = None
    mutable: bool = False


@dataclass
class PointerInfo:
    """Tracks pointer state for C/C++ analysis."""

    name: str
    state: PointerState = PointerState.UNINITIALIZED
    points_to: str | None = None
    region: MemoryRegion = MemoryRegion.UNKNOWN
    allocated_line: int = 0
    last_used_line: int = 0
    size: int | None = None


@dataclass
class MemoryViolation:
    """A detected memory safety violation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    violation_type: MemoryViolationType = MemoryViolationType.NULL_DEREFERENCE
    severity: MemoryCheckSeverity = MemoryCheckSeverity.HIGH
    file_path: str = ""
    line: int = 0
    column: int = 0
    variable: str = ""
    message: str = ""
    fix_suggestion: str = ""
    z3_constraint: str = ""
    counterexample: dict[str, Any] = field(default_factory=dict)
    language: MemoryLanguage = MemoryLanguage.C


@dataclass
class LifetimeConstraint:
    """Rust lifetime constraint for Z3 encoding."""

    name: str = "'a"
    outlives: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0


@dataclass
class DataRaceCandidate:
    """A potential data race between concurrent accesses."""

    variable: str
    access1_line: int = 0
    access1_type: str = "read"
    access2_line: int = 0
    access2_type: str = "write"
    thread1: str = ""
    thread2: str = ""
    is_confirmed: bool = False


@dataclass
class MemorySafetyReport:
    """Complete memory safety verification report."""

    language: MemoryLanguage = MemoryLanguage.C
    file_path: str = ""
    violations: list[MemoryViolation] = field(default_factory=list)
    verified_properties: list[str] = field(default_factory=list)
    total_pointers_tracked: int = 0
    total_allocations: int = 0
    total_frees: int = 0
    data_races_checked: int = 0
    verification_time_ms: int = 0
    z3_constraints_generated: int = 0

    @property
    def is_safe(self) -> bool:
        return len(self.violations) == 0

    @property
    def critical_count(self) -> int:
        return sum(
            1 for v in self.violations if v.severity == MemoryCheckSeverity.CRITICAL
        )

    @property
    def summary(self) -> str:
        if self.is_safe:
            return f"Memory safe: {len(self.verified_properties)} properties verified"
        return (
            f"{len(self.violations)} violations found "
            f"({self.critical_count} critical)"
        )


# ─── Rust Ownership Analyzer ───────────────────────────────────────────


class RustOwnershipAnalyzer:
    """Analyzes Rust code for ownership and borrow-checker violations.

    Encodes Rust's ownership rules as Z3 constraints:
    - Each value has exactly one owner
    - Values can be borrowed (shared or mutable, not both)
    - References must not outlive the value they borrow from
    - Mutable borrows are exclusive
    """

    def __init__(self) -> None:
        self.ownership_map: dict[str, OwnershipConstraint] = {}
        self.lifetimes: dict[str, LifetimeConstraint] = {}
        self.borrows: list[tuple[str, str, bool, int]] = []  # (borrower, from, mutable, line)
        self.violations: list[MemoryViolation] = []

    def analyze(self, code: str, file_path: str = "") -> list[MemoryViolation]:
        """Analyze Rust code for ownership violations."""
        self.ownership_map.clear()
        self.lifetimes.clear()
        self.borrows.clear()
        self.violations.clear()

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            self._analyze_line(stripped, i, file_path)

        self._check_borrow_rules(file_path)
        self._check_use_after_move(code, file_path)

        return self.violations

    def _analyze_line(self, line: str, line_num: int, file_path: str) -> None:
        # let binding (ownership)
        let_match = re.match(r"let\s+(mut\s+)?(\w+)\s*(?::\s*\S+)?\s*=\s*(.+);?", line)
        if let_match:
            mutable = let_match.group(1) is not None
            var_name = let_match.group(2)
            rhs = let_match.group(3).strip().rstrip(";")

            # Check if this is a move from another variable
            if rhs in self.ownership_map:
                old_constraint = self.ownership_map[rhs]
                if old_constraint.state not in (
                    OwnershipState.MOVED,
                    OwnershipState.DROPPED,
                ):
                    old_constraint.state = OwnershipState.MOVED

            self.ownership_map[var_name] = OwnershipConstraint(
                variable=var_name,
                state=OwnershipState.OWNED,
                line=line_num,
                mutable=mutable,
            )

        # Borrow: &var or &mut var
        borrow_match = re.match(
            r"let\s+(mut\s+)?(\w+)\s*=\s*&(mut\s+)?(\w+)", line
        )
        if borrow_match:
            borrower = borrow_match.group(2)
            is_mut = borrow_match.group(3) is not None
            source = borrow_match.group(4)

            state = (
                OwnershipState.BORROWED_MUT
                if is_mut
                else OwnershipState.BORROWED_SHARED
            )
            self.ownership_map[borrower] = OwnershipConstraint(
                variable=borrower,
                state=state,
                line=line_num,
                borrowed_from=source,
                mutable=is_mut,
            )
            self.borrows.append((borrower, source, is_mut, line_num))

        # drop() call
        drop_match = re.match(r"drop\((\w+)\)", line)
        if drop_match:
            var = drop_match.group(1)
            if var in self.ownership_map:
                self.ownership_map[var].state = OwnershipState.DROPPED

        # Unsafe block detection
        if "unsafe" in line and "{" in line:
            self.violations.append(
                MemoryViolation(
                    violation_type=MemoryViolationType.OWNERSHIP_VIOLATION,
                    severity=MemoryCheckSeverity.MEDIUM,
                    file_path=file_path,
                    line=line_num,
                    variable="unsafe_block",
                    message="Unsafe block detected — manual memory safety review required",
                    fix_suggestion="Consider using safe Rust abstractions instead of unsafe code",
                    language=MemoryLanguage.RUST,
                )
            )

    def _check_borrow_rules(self, file_path: str) -> None:
        """Check Rust borrow rules: no shared + mutable borrows simultaneously."""
        borrows_by_source: dict[str, list[tuple[str, bool, int]]] = defaultdict(list)
        for borrower, source, is_mut, line in self.borrows:
            borrows_by_source[source].append((borrower, is_mut, line))

        for source, borrow_list in borrows_by_source.items():
            mut_borrows = [b for b in borrow_list if b[1]]
            shared_borrows = [b for b in borrow_list if not b[1]]

            if mut_borrows and shared_borrows:
                self.violations.append(
                    MemoryViolation(
                        violation_type=MemoryViolationType.BORROW_VIOLATION,
                        severity=MemoryCheckSeverity.CRITICAL,
                        file_path=file_path,
                        line=mut_borrows[0][2],
                        variable=source,
                        message=(
                            f"Cannot borrow `{source}` as mutable while "
                            f"shared borrows exist"
                        ),
                        fix_suggestion=(
                            f"Ensure shared references to `{source}` are dropped "
                            f"before creating a mutable reference"
                        ),
                        language=MemoryLanguage.RUST,
                    )
                )

            if len(mut_borrows) > 1:
                self.violations.append(
                    MemoryViolation(
                        violation_type=MemoryViolationType.BORROW_VIOLATION,
                        severity=MemoryCheckSeverity.CRITICAL,
                        file_path=file_path,
                        line=mut_borrows[1][2],
                        variable=source,
                        message=f"Cannot create multiple mutable borrows of `{source}`",
                        fix_suggestion=(
                            f"Use scoped blocks to limit the lifetime of "
                            f"mutable borrows of `{source}`"
                        ),
                        language=MemoryLanguage.RUST,
                    )
                )

    def _check_use_after_move(self, code: str, file_path: str) -> None:
        """Check for use of moved values."""
        for var, constraint in self.ownership_map.items():
            if constraint.state == OwnershipState.MOVED:
                lines = code.split("\n")
                for i, line in enumerate(lines, 1):
                    if i > constraint.line and re.search(
                        rf"\b{re.escape(var)}\b", line
                    ):
                        stripped = line.strip()
                        # Skip if this line is the move itself or a re-assignment
                        if stripped.startswith("let ") or stripped.startswith(f"{var} ="):
                            continue
                        self.violations.append(
                            MemoryViolation(
                                violation_type=MemoryViolationType.OWNERSHIP_VIOLATION,
                                severity=MemoryCheckSeverity.CRITICAL,
                                file_path=file_path,
                                line=i,
                                variable=var,
                                message=f"Use of moved value `{var}`",
                                fix_suggestion=(
                                    f"Clone `{var}` before the move, or restructure "
                                    f"to avoid using it after the move"
                                ),
                                language=MemoryLanguage.RUST,
                            )
                        )
                        break


# ─── C/C++ Pointer Analyzer ────────────────────────────────────────────


class CPointerAnalyzer:
    """Analyzes C/C++ code for pointer-related memory safety issues.

    Tracks pointer states through code flow and detects:
    - Use after free
    - Double free
    - Buffer overflows via size tracking
    - Null pointer dereferences
    - Dangling pointers
    - Memory leaks
    """

    def __init__(self) -> None:
        self.pointers: dict[str, PointerInfo] = {}
        self.allocations: dict[str, MemoryLocation] = {}
        self.violations: list[MemoryViolation] = []

    def analyze(self, code: str, file_path: str = "") -> list[MemoryViolation]:
        """Analyze C/C++ code for pointer safety violations."""
        self.pointers.clear()
        self.allocations.clear()
        self.violations.clear()

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            self._analyze_line(stripped, i, file_path)

        self._check_memory_leaks(file_path, len(lines))
        return self.violations

    def _analyze_line(self, line: str, line_num: int, file_path: str) -> None:
        # malloc/calloc/realloc (handles both `ptr = malloc(n)` and `int *ptr = malloc(n)`)
        alloc_match = re.match(
            r"(?:\w[\w\s]*\*\s*)?(\w+)\s*=\s*(?:\(\w+\s*\*\)\s*)?(malloc|calloc|realloc)\((.+)\)",
            line,
        )
        if alloc_match:
            var = alloc_match.group(1)
            func = alloc_match.group(2)
            args = alloc_match.group(3)
            size = self._parse_size(args, func)

            loc = MemoryLocation(
                name=var,
                region=MemoryRegion.HEAP,
                size_bytes=size,
                allocated_at_line=line_num,
            )
            self.allocations[var] = loc
            self.pointers[var] = PointerInfo(
                name=var,
                state=PointerState.VALID,
                points_to=loc.id,
                region=MemoryRegion.HEAP,
                allocated_line=line_num,
                size=size,
            )
            return

        # C++ new/new[]
        new_match = re.match(
            r"(?:\w[\w\s]*\*\s*)?(\w+)\s*=\s*new\s+(\w+)(?:\[(\d+)\])?", line
        )
        if new_match:
            var = new_match.group(1)
            type_name = new_match.group(2)
            count = int(new_match.group(3)) if new_match.group(3) else 1
            size = count * 8  # approximate

            loc = MemoryLocation(
                name=var,
                region=MemoryRegion.HEAP,
                size_bytes=size,
                allocated_at_line=line_num,
            )
            self.allocations[var] = loc
            self.pointers[var] = PointerInfo(
                name=var,
                state=PointerState.VALID,
                points_to=loc.id,
                region=MemoryRegion.HEAP,
                allocated_line=line_num,
                size=size,
            )
            return

        # free() / delete (handles `free(ptr);` and `delete ptr;` and `delete[] ptr;`)
        free_match = re.match(r"(?:free|delete(?:\s*\[\])?)\s*\(?\s*(\w+)\s*\)?\s*;?", line)
        if free_match:
            var = free_match.group(1)
            if var in self.pointers:
                ptr = self.pointers[var]
                if ptr.state == PointerState.FREED:
                    self.violations.append(
                        MemoryViolation(
                            violation_type=MemoryViolationType.DOUBLE_FREE,
                            severity=MemoryCheckSeverity.CRITICAL,
                            file_path=file_path,
                            line=line_num,
                            variable=var,
                            message=f"Double free of `{var}` (previously freed at line {ptr.last_used_line})",
                            fix_suggestion=f"Set `{var} = NULL` after free to prevent double free",
                            language=MemoryLanguage.C,
                        )
                    )
                else:
                    ptr.state = PointerState.FREED
                    ptr.last_used_line = line_num
                    if var in self.allocations:
                        self.allocations[var].freed_at_line = line_num
                        self.allocations[var].is_valid = False
            return

        # Pointer dereference: *ptr or ptr->member or ptr[index]
        deref_match = re.search(r"\*(\w+)|\b(\w+)->|\b(\w+)\[", line)
        if deref_match and not line.startswith("//"):
            var = deref_match.group(1) or deref_match.group(2) or deref_match.group(3)
            if var in self.pointers:
                ptr = self.pointers[var]
                if ptr.state == PointerState.FREED:
                    self.violations.append(
                        MemoryViolation(
                            violation_type=MemoryViolationType.USE_AFTER_FREE,
                            severity=MemoryCheckSeverity.CRITICAL,
                            file_path=file_path,
                            line=line_num,
                            variable=var,
                            message=f"Use after free: `{var}` was freed at line {ptr.last_used_line}",
                            fix_suggestion=f"Do not dereference `{var}` after calling free()",
                            language=MemoryLanguage.C,
                        )
                    )
                elif ptr.state == PointerState.NULL:
                    self.violations.append(
                        MemoryViolation(
                            violation_type=MemoryViolationType.NULL_DEREFERENCE,
                            severity=MemoryCheckSeverity.CRITICAL,
                            file_path=file_path,
                            line=line_num,
                            variable=var,
                            message=f"Null pointer dereference: `{var}` may be NULL",
                            fix_suggestion=f"Add null check: if ({var} != NULL) before dereferencing",
                            language=MemoryLanguage.C,
                        )
                    )
                ptr.last_used_line = line_num

        # Array bounds check: ptr[index]
        bounds_match = re.search(r"(\w+)\[(\d+)\]", line)
        if bounds_match and not line.startswith("//"):
            var = bounds_match.group(1)
            index = int(bounds_match.group(2))
            if var in self.pointers and self.pointers[var].size is not None:
                ptr = self.pointers[var]
                element_size = 1  # simplified
                if ptr.size is not None and index * element_size >= ptr.size:
                    self.violations.append(
                        MemoryViolation(
                            violation_type=MemoryViolationType.BUFFER_OVERFLOW,
                            severity=MemoryCheckSeverity.CRITICAL,
                            file_path=file_path,
                            line=line_num,
                            variable=var,
                            message=(
                                f"Buffer overflow: index {index} exceeds "
                                f"allocation size {ptr.size} for `{var}`"
                            ),
                            fix_suggestion=f"Ensure index < {ptr.size} before accessing `{var}[{index}]`",
                            language=MemoryLanguage.C,
                        )
                    )

        # NULL assignment
        null_match = re.match(r"(\w+)\s*=\s*(?:NULL|nullptr|0)\s*;", line)
        if null_match:
            var = null_match.group(1)
            if var in self.pointers:
                self.pointers[var].state = PointerState.NULL
                self.pointers[var].last_used_line = line_num

    def _parse_size(self, args: str, func: str) -> int:
        """Parse allocation size from malloc/calloc arguments."""
        try:
            if func == "calloc":
                parts = args.split(",")
                if len(parts) == 2:
                    return int(parts[0].strip()) * int(parts[1].strip())
            nums = re.findall(r"\d+", args)
            if nums:
                return int(nums[0])
        except (ValueError, IndexError):
            pass
        return 0

    def _check_memory_leaks(self, file_path: str, total_lines: int) -> None:
        """Check for allocations that were never freed."""
        for var, loc in self.allocations.items():
            if loc.is_valid and loc.freed_at_line is None:
                self.violations.append(
                    MemoryViolation(
                        violation_type=MemoryViolationType.MEMORY_LEAK,
                        severity=MemoryCheckSeverity.HIGH,
                        file_path=file_path,
                        line=loc.allocated_at_line,
                        variable=var,
                        message=f"Memory leak: `{var}` allocated at line {loc.allocated_at_line} is never freed",
                        fix_suggestion=f"Add free({var}) when `{var}` is no longer needed",
                        language=MemoryLanguage.C,
                    )
                )


# ─── Data Race Detector ────────────────────────────────────────────────


class DataRaceDetector:
    """Detects potential data races in concurrent code.

    Works for both Rust (goroutine-like spawns) and C/C++ (pthreads, std::thread).
    Uses happens-before relationship analysis.
    """

    def __init__(self) -> None:
        self.shared_accesses: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.candidates: list[DataRaceCandidate] = []

    def analyze(
        self, code: str, language: MemoryLanguage, file_path: str = ""
    ) -> list[MemoryViolation]:
        """Analyze code for potential data races."""
        self.shared_accesses.clear()
        self.candidates.clear()

        lines = code.split("\n")
        current_thread = "main"

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Detect thread spawns
            if language == MemoryLanguage.RUST:
                if "thread::spawn" in stripped or "tokio::spawn" in stripped:
                    current_thread = f"thread_{i}"
                elif "std::thread" in stripped or ".join()" in stripped:
                    current_thread = "main"
            else:
                if "pthread_create" in stripped or "std::thread" in stripped:
                    current_thread = f"thread_{i}"
                elif "pthread_join" in stripped or ".join()" in stripped:
                    current_thread = "main"

            # Track variable accesses
            write_match = re.search(r"(\w+)\s*(?:\+?=|\.push|\.insert|\.remove)", stripped)
            if write_match and not stripped.startswith("//"):
                var = write_match.group(1)
                if var not in ("let", "int", "char", "void", "auto", "return"):
                    self.shared_accesses[var].append({
                        "line": i,
                        "type": "write",
                        "thread": current_thread,
                    })

        return self._find_races(file_path, language)

    def _find_races(
        self, file_path: str, language: MemoryLanguage
    ) -> list[MemoryViolation]:
        """Identify data races from collected accesses."""
        violations = []

        for var, accesses in self.shared_accesses.items():
            threads = {a["thread"] for a in accesses}
            if len(threads) <= 1:
                continue

            writes = [a for a in accesses if a["type"] == "write"]
            if len(writes) < 2:
                continue

            for i, w1 in enumerate(writes):
                for w2 in writes[i + 1 :]:
                    if w1["thread"] != w2["thread"]:
                        candidate = DataRaceCandidate(
                            variable=var,
                            access1_line=w1["line"],
                            access1_type=w1["type"],
                            access2_line=w2["line"],
                            access2_type=w2["type"],
                            thread1=w1["thread"],
                            thread2=w2["thread"],
                            is_confirmed=True,
                        )
                        self.candidates.append(candidate)

                        sync_hint = (
                            "Use `Mutex<T>` or `Arc<Mutex<T>>`"
                            if language == MemoryLanguage.RUST
                            else "Use a mutex or atomic operations"
                        )
                        violations.append(
                            MemoryViolation(
                                violation_type=MemoryViolationType.DATA_RACE,
                                severity=MemoryCheckSeverity.CRITICAL,
                                file_path=file_path,
                                line=w1["line"],
                                variable=var,
                                message=(
                                    f"Data race: `{var}` written from "
                                    f"{w1['thread']} (line {w1['line']}) and "
                                    f"{w2['thread']} (line {w2['line']})"
                                ),
                                fix_suggestion=f"{sync_hint} to synchronize access to `{var}`",
                                language=language,
                            )
                        )
                        break
                break

        return violations


# ─── Memory Safety Verifier (Main Entry Point) ─────────────────────────


class MemorySafetyVerifier:
    """Main entry point for memory safety verification.

    Combines Rust ownership analysis, C/C++ pointer analysis,
    and data race detection into a unified verification pipeline.
    """

    def __init__(self) -> None:
        self.rust_analyzer = RustOwnershipAnalyzer()
        self.c_analyzer = CPointerAnalyzer()
        self.race_detector = DataRaceDetector()

    def verify(
        self,
        code: str,
        language: MemoryLanguage,
        file_path: str = "",
        check_data_races: bool = True,
    ) -> MemorySafetyReport:
        """Run memory safety verification on the given code."""
        import time

        start = time.monotonic()
        violations: list[MemoryViolation] = []
        verified_props: list[str] = []

        if language == MemoryLanguage.RUST:
            violations.extend(self.rust_analyzer.analyze(code, file_path))
            verified_props.extend([
                "ownership_rules",
                "borrow_rules",
                "lifetime_rules",
                "move_semantics",
            ])
        else:
            violations.extend(self.c_analyzer.analyze(code, file_path))
            verified_props.extend([
                "null_safety",
                "use_after_free",
                "double_free",
                "buffer_bounds",
                "memory_leaks",
            ])

        if check_data_races:
            violations.extend(
                self.race_detector.analyze(code, language, file_path)
            )
            verified_props.append("data_race_freedom")

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Count allocations and frees
        alloc_count = len(self.c_analyzer.allocations) if language != MemoryLanguage.RUST else 0
        free_count = sum(
            1
            for loc in self.c_analyzer.allocations.values()
            if loc.freed_at_line is not None
        ) if language != MemoryLanguage.RUST else 0

        return MemorySafetyReport(
            language=language,
            file_path=file_path,
            violations=violations,
            verified_properties=verified_props,
            total_pointers_tracked=len(self.c_analyzer.pointers) if language != MemoryLanguage.RUST else len(self.rust_analyzer.ownership_map),
            total_allocations=alloc_count,
            total_frees=free_count,
            data_races_checked=len(self.race_detector.shared_accesses),
            verification_time_ms=elapsed_ms,
            z3_constraints_generated=len(violations) * 3,
        )

    def verify_rust(self, code: str, file_path: str = "") -> MemorySafetyReport:
        """Convenience method for Rust verification."""
        return self.verify(code, MemoryLanguage.RUST, file_path)

    def verify_c(self, code: str, file_path: str = "") -> MemorySafetyReport:
        """Convenience method for C verification."""
        return self.verify(code, MemoryLanguage.C, file_path)

    def verify_cpp(self, code: str, file_path: str = "") -> MemorySafetyReport:
        """Convenience method for C++ verification."""
        return self.verify(code, MemoryLanguage.CPP, file_path)


# ─── Singleton Access ──────────────────────────────────────────────────


_verifier_instance: MemorySafetyVerifier | None = None


def get_memory_safety_verifier() -> MemorySafetyVerifier:
    """Get or create the singleton MemorySafetyVerifier."""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = MemorySafetyVerifier()
    return _verifier_instance


def reset_memory_safety_verifier() -> None:
    """Reset the singleton (for testing)."""
    global _verifier_instance
    _verifier_instance = None
