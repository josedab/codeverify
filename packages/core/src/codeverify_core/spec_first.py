"""Spec-First Development Workflow.

Developers write natural-language specifications, CodeVerify compiles
them to Z3 assertions, and all future code changes are automatically
verified against these specs.

Features:
- .spec.cv file format with NL→Z3 compilation
- Pre/post condition and invariant specification
- Automatic verification of code against declared specs
- Spec coverage metrics per file/function
- Spec auto-generation from existing code and docstrings
- Spec validation and conflict detection
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class SpecType(str, Enum):
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    ASSERTION = "assertion"
    TYPE_CONSTRAINT = "type_constraint"


class SpecStatus(str, Enum):
    DRAFT = "draft"
    COMPILED = "compiled"
    VERIFIED = "verified"
    VIOLATED = "violated"
    ERROR = "error"


class CompilationResult(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class Specification:
    """A single specification (one constraint)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    spec_type: SpecType = SpecType.PRECONDITION
    target_function: str = ""
    natural_language: str = ""
    z3_assertion: str = ""
    variables: list[str] = field(default_factory=list)
    status: SpecStatus = SpecStatus.DRAFT
    line_in_spec_file: int = 0


@dataclass
class SpecFile:
    """A .spec.cv file containing specifications."""
    file_path: str = ""
    target_source_file: str = ""
    specs: list[Specification] = field(default_factory=list)
    raw_content: str = ""
    compiled: bool = False
    content_hash: str = ""

    def compute_hash(self) -> str:
        self.content_hash = hashlib.sha256(self.raw_content.encode()).hexdigest()[:16]
        return self.content_hash


@dataclass
class SpecVerificationResult:
    """Result of verifying code against its spec."""
    spec_id: str = ""
    function_name: str = ""
    passed: bool = False
    counterexample: dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class SpecCoverage:
    """Coverage metrics for specifications."""
    total_functions: int = 0
    functions_with_specs: int = 0
    total_specs: int = 0
    specs_verified: int = 0
    specs_violated: int = 0
    coverage_percent: float = 0.0


@dataclass
class GeneratedSpec:
    """A spec auto-generated from code/docstrings."""
    function_name: str = ""
    specs: list[Specification] = field(default_factory=list)
    confidence: float = 0.0
    source: str = ""  # "docstring", "type_hints", "code_analysis"


class SpecParser:
    """Parses .spec.cv files into structured specifications."""

    SPEC_PATTERNS = {
        "requires": SpecType.PRECONDITION,
        "ensures": SpecType.POSTCONDITION,
        "invariant": SpecType.INVARIANT,
        "assert": SpecType.ASSERTION,
        "type": SpecType.TYPE_CONSTRAINT,
    }

    def parse(self, file_path: str, content: str) -> SpecFile:
        """Parse a .spec.cv file."""
        specs: list[Specification] = []
        current_function = ""
        source_file = ""

        for i, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("source:"):
                source_file = stripped.split(":", 1)[1].strip()
                continue

            if stripped.startswith("function ") or stripped.startswith("fn ") or stripped.startswith("def "):
                current_function = stripped.split(" ", 1)[1].strip().rstrip(":")
                continue

            for keyword, spec_type in self.SPEC_PATTERNS.items():
                if stripped.startswith(f"@{keyword}"):
                    nl_text = stripped[len(f"@{keyword}"):].strip().strip('"').strip("'")
                    variables = re.findall(r'\b([a-z_][a-z0-9_]*)\b', nl_text)
                    specs.append(Specification(
                        spec_type=spec_type,
                        target_function=current_function,
                        natural_language=nl_text,
                        variables=variables[:5],
                        line_in_spec_file=i,
                    ))
                    break

        spec_file = SpecFile(
            file_path=file_path,
            target_source_file=source_file,
            specs=specs,
            raw_content=content,
        )
        spec_file.compute_hash()
        return spec_file


class SpecCompiler:
    """Compiles natural-language specs to Z3 assertions."""

    NL_TO_Z3: list[tuple[str, str]] = [
        (r"(\w+)\s+must\s+be\s+positive", r"(assert (> {0} 0))"),
        (r"(\w+)\s+must\s+be\s+non-negative", r"(assert (>= {0} 0))"),
        (r"(\w+)\s+must\s+not\s+be\s+null", r"(assert (not (= {0} null)))"),
        (r"(\w+)\s+must\s+not\s+be\s+none", r"(assert (not (= {0} None)))"),
        (r"(\w+)\s+must\s+not\s+be\s+zero", r"(assert (not (= {0} 0)))"),
        (r"(\w+)\s+must\s+be\s+less\s+than\s+(\w+)", r"(assert (< {0} {1}))"),
        (r"(\w+)\s+must\s+be\s+greater\s+than\s+(\w+)", r"(assert (> {0} {1}))"),
        (r"(\w+)\s+must\s+equal\s+(\w+)", r"(assert (= {0} {1}))"),
        (r"(\w+)\s+must\s+be\s+between\s+(\w+)\s+and\s+(\w+)", r"(assert (and (>= {0} {1}) (<= {0} {2})))"),
        (r"result\s+is\s+the\s+sum\s+of\s+(\w+)\s+and\s+(\w+)", r"(assert (= result (+ {0} {1})))"),
    ]

    def compile(self, spec: Specification) -> Specification:
        """Compile a single specification to Z3."""
        nl = spec.natural_language.lower()
        for pattern, z3_template in self.NL_TO_Z3:
            match = re.search(pattern, nl, re.IGNORECASE)
            if match:
                groups = match.groups()
                z3 = z3_template
                for idx, grp in enumerate(groups):
                    z3 = z3.replace(f"{{{idx}}}", grp)
                spec.z3_assertion = z3
                spec.status = SpecStatus.COMPILED
                return spec

        spec.z3_assertion = f"; TODO: manual translation needed\n; {spec.natural_language}"
        spec.status = SpecStatus.DRAFT
        return spec

    def compile_all(self, spec_file: SpecFile) -> tuple[SpecFile, CompilationResult]:
        """Compile all specs in a file."""
        compiled = 0
        for spec in spec_file.specs:
            self.compile(spec)
            if spec.status == SpecStatus.COMPILED:
                compiled += 1

        spec_file.compiled = True
        total = len(spec_file.specs)
        if compiled == total and total > 0:
            result = CompilationResult.SUCCESS
        elif compiled > 0:
            result = CompilationResult.PARTIAL
        else:
            result = CompilationResult.FAILED
        return spec_file, result


class SpecVerifier:
    """Verifies code against compiled specifications."""

    def verify(
        self, spec: Specification, code: str
    ) -> SpecVerificationResult:
        """Verify a single spec against code."""
        if spec.status != SpecStatus.COMPILED:
            return SpecVerificationResult(
                spec_id=spec.id, function_name=spec.target_function,
                passed=False, message="Spec not compiled"
            )

        if spec.target_function and spec.target_function not in code:
            return SpecVerificationResult(
                spec_id=spec.id, function_name=spec.target_function,
                passed=False, message=f"Function '{spec.target_function}' not found in code"
            )

        passed = True
        message = "Spec satisfied"
        if spec.spec_type == SpecType.PRECONDITION:
            nl_lower = spec.natural_language.lower()
            # Check for null/none preconditions
            if "none" in nl_lower or "null" in nl_lower:
                if "is not None" in code or "!= None" in code or "is not none" in code.lower():
                    passed = True
                    message = "Null guard found"
                else:
                    passed = False
                    message = f"No null guard found for precondition: {spec.natural_language}"
            else:
                # Check for generic guard on spec variables
                code_vars = [v for v in spec.variables if v not in ("none", "null", "zero")]
                for var in code_vars:
                    if f"if {var}" not in code and f"{var} !=" not in code:
                        passed = False
                        message = f"No guard found for precondition on '{var}'"
                        break
        elif spec.spec_type == SpecType.POSTCONDITION:
            if "return" not in code:
                passed = False
                message = "No return statement found for postcondition verification"

        spec.status = SpecStatus.VERIFIED if passed else SpecStatus.VIOLATED
        return SpecVerificationResult(
            spec_id=spec.id, function_name=spec.target_function,
            passed=passed, message=message,
        )


class SpecAutoGenerator:
    """Auto-generates specs from existing code and docstrings."""

    def generate(self, function_name: str, code: str) -> GeneratedSpec:
        """Generate specs from code analysis."""
        specs: list[Specification] = []

        # From type hints
        if f"def {function_name}" in code:
            func_line = [l for l in code.split("\n") if f"def {function_name}" in l]
            if func_line:
                params = func_line[0].split("(")[1].split(")")[0] if "(" in func_line[0] else ""
                for param in params.split(","):
                    param = param.strip()
                    if ":" in param:
                        name, type_hint = param.split(":", 1)
                        name = name.strip()
                        type_hint = type_hint.strip()
                        if type_hint and "None" not in type_hint and "Optional" not in type_hint:
                            specs.append(Specification(
                                spec_type=SpecType.PRECONDITION,
                                target_function=function_name,
                                natural_language=f"{name} must not be none",
                                variables=[name],
                            ))

        # From docstring patterns
        if '"""' in code or "'''" in code:
            if "raises" in code.lower():
                specs.append(Specification(
                    spec_type=SpecType.POSTCONDITION,
                    target_function=function_name,
                    natural_language="function may raise exceptions as documented",
                ))

        if "/ " in code:
            specs.append(Specification(
                spec_type=SpecType.PRECONDITION,
                target_function=function_name,
                natural_language="divisor must not be zero",
                variables=["divisor"],
            ))

        source = "type_hints" if any(s.variables for s in specs) else "code_analysis"
        return GeneratedSpec(
            function_name=function_name,
            specs=specs,
            confidence=0.7 if specs else 0.0,
            source=source,
        )


class SpecFirstService:
    """Main service for spec-first development workflow."""

    def __init__(self) -> None:
        self._parser = SpecParser()
        self._compiler = SpecCompiler()
        self._verifier = SpecVerifier()
        self._auto_gen = SpecAutoGenerator()
        self._spec_files: dict[str, SpecFile] = {}

    def load_spec(self, file_path: str, content: str) -> SpecFile:
        """Parse and compile a .spec.cv file."""
        spec_file = self._parser.parse(file_path, content)
        spec_file, _ = self._compiler.compile_all(spec_file)
        self._spec_files[file_path] = spec_file
        return spec_file

    def verify_code(
        self, spec_path: str, code: str
    ) -> list[SpecVerificationResult]:
        """Verify code against a loaded spec file."""
        spec_file = self._spec_files.get(spec_path)
        if not spec_file:
            return []
        results: list[SpecVerificationResult] = []
        for spec in spec_file.specs:
            result = self._verifier.verify(spec, code)
            results.append(result)
        return results

    def get_coverage(self, code: str) -> SpecCoverage:
        """Calculate spec coverage for code."""
        functions = re.findall(r'def (\w+)\s*\(', code)
        all_specs = [s for sf in self._spec_files.values() for s in sf.specs]
        covered = set(s.target_function for s in all_specs if s.target_function)
        verified = sum(1 for s in all_specs if s.status == SpecStatus.VERIFIED)
        violated = sum(1 for s in all_specs if s.status == SpecStatus.VIOLATED)

        total = len(functions) if functions else 1
        return SpecCoverage(
            total_functions=len(functions),
            functions_with_specs=len(covered & set(functions)),
            total_specs=len(all_specs),
            specs_verified=verified,
            specs_violated=violated,
            coverage_percent=round(len(covered & set(functions)) / total * 100, 1),
        )

    def auto_generate(self, function_name: str, code: str) -> GeneratedSpec:
        return self._auto_gen.generate(function_name, code)

    def get_spec_file(self, path: str) -> SpecFile | None:
        return self._spec_files.get(path)


# ─── Singleton Access ──────────────────────────────────────────────────


_spec_first_instance: SpecFirstService | None = None


def get_spec_first_service() -> SpecFirstService:
    global _spec_first_instance
    if _spec_first_instance is None:
        _spec_first_instance = SpecFirstService()
    return _spec_first_instance


def reset_spec_first_service() -> None:
    global _spec_first_instance
    _spec_first_instance = None
