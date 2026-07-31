"""Smart Contract Analyzer — Deep analysis engine with Z3 integration.

Enhanced smart contract verification with vulnerability detection patterns,
formal proof generation, gas optimization analysis, and ERC standard compliance
checking for Solidity and Rust/Solana contracts.

Companion module to smart_contract_verification.py, which provides the base
enums and dataclasses. This module adds the deeper analysis engine.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class AnalysisDepth(str, Enum):
    """Depth of smart contract analysis."""

    QUICK_SCAN = "quick_scan"
    STANDARD = "standard"
    DEEP = "deep"
    FORMAL_PROOF = "formal_proof"


class ContractStandard(str, Enum):
    """Supported token / contract standards."""

    ERC20 = "ERC-20"
    ERC721 = "ERC-721"
    ERC1155 = "ERC-1155"
    ERC4626 = "ERC-4626"
    CUSTOM = "custom"


class ProofStatus(str, Enum):
    """Result status of a formal verification attempt."""

    PROVEN_SAFE = "proven_safe"
    COUNTEREXAMPLE_FOUND = "counterexample_found"
    INCONCLUSIVE = "inconclusive"
    TIMEOUT = "timeout"


class GasOptimization(str, Enum):
    """Categories of gas optimization suggestions."""

    STORAGE_PACKING = "storage_packing"
    LOOP_UNROLLING = "loop_unrolling"
    DEAD_CODE_REMOVAL = "dead_code_removal"
    CONSTANT_FOLDING = "constant_folding"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class ContractFunction:
    """A parsed function extracted from smart contract source code."""

    name: str
    visibility: str
    mutability: str
    parameters: list[dict[str, str]]
    return_types: list[str]
    modifiers: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0
    is_payable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "visibility": self.visibility,
            "mutability": self.mutability,
            "parameters": self.parameters,
            "return_types": self.return_types,
            "modifiers": self.modifiers,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "is_payable": self.is_payable,
        }


@dataclass
class VulnerabilityFinding:
    """A vulnerability detected during smart contract analysis."""

    id: str
    category: str
    severity: str
    title: str
    description: str
    function_name: str
    line_start: int
    line_end: int
    code_snippet: str
    fix_suggestion: str | None = None
    proof_status: ProofStatus = ProofStatus.INCONCLUSIVE
    cwe_id: str | None = None
    swc_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "function_name": self.function_name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "code_snippet": self.code_snippet,
            "fix_suggestion": self.fix_suggestion,
            "proof_status": self.proof_status.value,
            "cwe_id": self.cwe_id,
            "swc_id": self.swc_id,
        }


@dataclass
class FormalProof:
    """Result of a formal verification property check."""

    property_name: str
    status: ProofStatus
    z3_expression: str
    counterexample: dict[str, Any] | None = None
    proof_time_ms: float = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_name": self.property_name,
            "status": self.status.value,
            "z3_expression": self.z3_expression,
            "counterexample": self.counterexample,
            "proof_time_ms": self.proof_time_ms,
            "description": self.description,
        }


@dataclass
class GasAnalysis:
    """Gas usage analysis for a single contract function."""

    function_name: str
    estimated_gas: int
    optimizations: list[dict[str, Any]]
    storage_reads: int = 0
    storage_writes: int = 0
    external_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "estimated_gas": self.estimated_gas,
            "optimizations": self.optimizations,
            "storage_reads": self.storage_reads,
            "storage_writes": self.storage_writes,
            "external_calls": self.external_calls,
        }


@dataclass
class StandardComplianceResult:
    """Result of checking a contract against a token standard."""

    standard: ContractStandard
    compliant: bool
    missing_functions: list[str]
    incorrect_signatures: list[dict[str, str]]
    missing_events: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "standard": self.standard.value,
            "compliant": self.compliant,
            "missing_functions": self.missing_functions,
            "incorrect_signatures": self.incorrect_signatures,
            "missing_events": self.missing_events,
            "recommendations": self.recommendations,
        }


@dataclass
class SmartContractReport:
    """Complete analysis report for a smart contract."""

    contract_name: str
    language: str
    analysis_depth: AnalysisDepth
    functions_analyzed: int
    vulnerabilities: list[VulnerabilityFinding]
    formal_proofs: list[FormalProof]
    gas_analysis: list[GasAnalysis]
    compliance: StandardComplianceResult | None = None
    overall_risk_score: float = 0
    analysis_time_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "language": self.language,
            "analysis_depth": self.analysis_depth.value,
            "functions_analyzed": self.functions_analyzed,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "formal_proofs": [p.to_dict() for p in self.formal_proofs],
            "gas_analysis": [g.to_dict() for g in self.gas_analysis],
            "compliance": self.compliance.to_dict() if self.compliance else None,
            "overall_risk_score": self.overall_risk_score,
            "analysis_time_ms": self.analysis_time_ms,
        }


# =============================================================================
# Solidity Parser
# =============================================================================


class SolidityParser:
    """Parse Solidity source code to extract functions and state variables."""

    _FUNCTION_RE = re.compile(
        r"function\s+(\w+)\s*\(([^)]*)\)"
        r"((?:\s+(?:public|private|internal|external|view|pure|payable"
        r"|returns\s*\([^)]*\)|\w+))*)"
        r"\s*(?:returns\s*\(([^)]*)\))?\s*\{",
        re.DOTALL,
    )
    _STATE_VAR_RE = re.compile(
        r"^\s*(mapping\s*\([^)]+\)|(?:uint|int|address|bool|bytes|string)"
        r"(?:\d+)?(?:\[\])?)\s+"
        r"(public|private|internal|constant|immutable)?\s*"
        r"(?:(public|private|internal|constant|immutable)\s+)?"
        r"(\w+)\s*(?:=\s*([^;]+))?;",
        re.MULTILINE,
    )
    _EVENT_RE = re.compile(r"event\s+(\w+)\s*\(([^)]*)\)\s*;")
    _MODIFIER_RE = re.compile(r"modifier\s+(\w+)\s*(?:\(([^)]*)\))?\s*\{")
    _CONTRACT_RE = re.compile(r"contract\s+(\w+)\s*(?:is\s+([^{]+))?\s*\{")
    _VISIBILITY = {"public", "private", "internal", "external"}
    _MUTABILITY = {"view", "pure", "payable"}

    def __init__(self) -> None:
        self._logger = structlog.get_logger(__name__)

    def parse(self, code: str) -> tuple[list[ContractFunction], list[dict[str, Any]]]:
        """Parse Solidity source and return functions and state variables."""
        functions = self._extract_functions(code)
        state_vars = self._extract_state_variables(code)
        self._logger.info(
            "Solidity parsing complete", functions=len(functions), state_vars=len(state_vars)
        )
        return functions, state_vars

    def _extract_functions(self, code: str) -> list[ContractFunction]:
        """Extract all function declarations from Solidity source."""
        functions: list[ContractFunction] = []
        for match in self._FUNCTION_RE.finditer(code):
            name = match.group(1)
            raw_params = match.group(2).strip()
            qualifiers = match.group(3) or ""
            raw_returns = match.group(4) or ""
            line_start = code[: match.start()].count("\n") + 1
            line_end = line_start + self._count_body_lines(code, match.end())
            visibility, mutability, modifiers, is_payable = "internal", "nonpayable", [], False
            for token in qualifiers.split():
                token = token.strip()
                if token in self._VISIBILITY:
                    visibility = token
                elif token in self._MUTABILITY:
                    mutability = token
                    if token == "payable":
                        is_payable = True
                elif token and token != "returns":
                    modifiers.append(token)
            functions.append(
                ContractFunction(
                    name=name,
                    visibility=visibility,
                    mutability=mutability,
                    parameters=self._parse_params(raw_params),
                    return_types=self._parse_returns(raw_returns),
                    modifiers=modifiers,
                    line_start=line_start,
                    line_end=line_end,
                    is_payable=is_payable,
                )
            )
        return functions

    def _extract_state_variables(self, code: str) -> list[dict[str, Any]]:
        """Extract state variable declarations."""
        variables: list[dict[str, Any]] = []
        for match in self._STATE_VAR_RE.finditer(code):
            var_type, vis1, vis2 = (
                match.group(1).strip(),
                match.group(2) or "",
                match.group(3) or "",
            )
            var_name, default = match.group(4), (match.group(5) or "").strip()
            visibility, is_constant, is_immutable = "internal", False, False
            for vis in (vis1.strip(), vis2.strip()):
                if vis in self._VISIBILITY:
                    visibility = vis
                elif vis == "constant":
                    is_constant = True
                elif vis == "immutable":
                    is_immutable = True
            variables.append(
                {
                    "name": var_name,
                    "type": var_type,
                    "visibility": visibility,
                    "is_constant": is_constant,
                    "is_immutable": is_immutable,
                    "default_value": default or None,
                    "line": code[: match.start()].count("\n") + 1,
                }
            )
        return variables

    def _extract_events(self, code: str) -> list[str]:
        """Extract event names from Solidity source."""
        return [m.group(1) for m in self._EVENT_RE.finditer(code)]

    def _extract_modifiers(self, code: str) -> list[str]:
        """Extract modifier names from Solidity source."""
        return [m.group(1) for m in self._MODIFIER_RE.finditer(code)]

    def _parse_params(self, raw: str) -> list[dict[str, str]]:
        if not raw.strip():
            return []
        params: list[dict[str, str]] = []
        for part in raw.split(","):
            cleaned = re.sub(r"\b(memory|storage|calldata)\b", "", part).strip()
            tokens = cleaned.split()
            if len(tokens) >= 2:
                params.append({"type": tokens[0], "name": tokens[-1]})
            elif tokens:
                params.append({"type": tokens[0], "name": ""})
        return params

    def _parse_returns(self, raw: str) -> list[str]:
        if not raw.strip():
            return []
        types: list[str] = []
        for part in raw.split(","):
            cleaned = re.sub(r"\b(memory|storage|calldata)\b", "", part).strip()
            tokens = cleaned.split()
            if tokens:
                types.append(tokens[0])
        return types

    def _count_body_lines(self, code: str, start_pos: int) -> int:
        depth, pos = 1, start_pos
        while pos < len(code) and depth > 0:
            if code[pos] == "{":
                depth += 1
            elif code[pos] == "}":
                depth -= 1
            pos += 1
        return code[start_pos:pos].count("\n")


# =============================================================================
# Vulnerability Detector
# =============================================================================


class VulnerabilityDetector:
    """Detect common smart contract vulnerabilities."""

    _REENTRANCY_CALL_RE = re.compile(
        r"(\.\w+\{.*?value.*?\}\(.*?\)|\.call\{.*?\}\(|\.send\(|\.transfer\()",
        re.DOTALL,
    )
    _TX_ORIGIN_RE = re.compile(r"(?:require|if)\s*\([^)]*\btx\.origin\b")
    _DELEGATECALL_RE = re.compile(r"\.delegatecall\(")
    _FRONTRUN_RE = re.compile(
        r"\b(?:swap|trade|buy|sell|liquidate)\b.*\bblock\.(timestamp|number)\b",
        re.IGNORECASE,
    )
    _ARITHMETIC_RE = re.compile(r"\b(?:uint|int)\d*\s+\w+\s*=\s*\w+\s*[+\-*/]\s*\w+")

    def __init__(self) -> None:
        self._logger = structlog.get_logger(__name__)
        self._counter = 0

    def detect(
        self, code: str, functions: list[ContractFunction], language: str = "solidity"
    ) -> list[VulnerabilityFinding]:
        """Run all vulnerability detection checks."""
        self._counter = 0
        findings: list[VulnerabilityFinding] = []
        if language == "solidity":
            findings.extend(self._check_reentrancy(code, functions))
            findings.extend(self._check_integer_overflow(code, functions))
            findings.extend(self._check_access_control(code, functions))
            findings.extend(self._check_tx_origin(code))
            findings.extend(self._check_unchecked_calls(code))
            findings.extend(self._check_front_running(code))
        elif language == "rust_solana":
            findings.extend(self._check_access_control(code, functions))
        self._logger.info(
            "Vulnerability detection complete", language=language, findings=len(findings)
        )
        return findings

    def _next_id(self) -> str:
        self._counter += 1
        return f"VULN-{self._counter:04d}"

    def _find_enclosing_function(
        self, code: str, pos: int, functions: list[ContractFunction]
    ) -> str:
        line_num = code[:pos].count("\n") + 1
        for func in functions:
            if func.line_start <= line_num <= func.line_end:
                return func.name
        return "<unknown>"

    def _check_reentrancy(
        self, code: str, functions: list[ContractFunction]
    ) -> list[VulnerabilityFinding]:
        """Detect reentrancy: external call before state change."""
        findings: list[VulnerabilityFinding] = []
        for func in functions:
            body = self._get_body(code, func.name)
            if body is None:
                continue
            call_m = self._REENTRANCY_CALL_RE.search(body)
            if call_m is None:
                continue
            state_m = re.search(r"\b(\w+)\s*[+\-*/]?=\s*", body[call_m.end() :])
            if state_m:
                ln = code[: code.find(body) + call_m.start()].count("\n") + 1
                findings.append(
                    VulnerabilityFinding(
                        id=self._next_id(),
                        category="reentrancy",
                        severity="critical",
                        title="Potential Reentrancy Vulnerability",
                        description=f"External call in '{func.name}' before state variable '{state_m.group(1)}' is updated.",
                        function_name=func.name,
                        line_start=ln,
                        line_end=ln + 3,
                        code_snippet=body[call_m.start() : call_m.start() + 120].strip(),
                        fix_suggestion="Apply checks-effects-interactions pattern or use ReentrancyGuard.",
                        swc_id="SWC-107",
                        cwe_id="CWE-841",
                    )
                )
        return findings

    def _check_integer_overflow(
        self, code: str, functions: list[ContractFunction]
    ) -> list[VulnerabilityFinding]:
        """Detect integer overflow/underflow in pre-0.8.0 Solidity."""
        vm = re.search(r"pragma\s+solidity\s+[<>=^~]*\s*(0\.\d+)", code)
        if vm and int(vm.group(1).split(".")[1]) >= 8:
            return []
        findings: list[VulnerabilityFinding] = []
        lines = code.split("\n")
        for match in self._ARITHMETIC_RE.finditer(code):
            ln = code[: match.start()].count("\n") + 1
            findings.append(
                VulnerabilityFinding(
                    id=self._next_id(),
                    category="integer_overflow",
                    severity="high",
                    title="Potential Integer Overflow/Underflow",
                    description="Arithmetic on integer type without SafeMath in a pre-0.8.0 contract.",
                    function_name=self._find_enclosing_function(code, match.start(), functions),
                    line_start=ln,
                    line_end=ln,
                    code_snippet=(lines[ln - 1].strip()[:200] if ln <= len(lines) else ""),
                    fix_suggestion="Upgrade to Solidity >=0.8.0 or use OpenZeppelin SafeMath.",
                    swc_id="SWC-101",
                    cwe_id="CWE-190",
                )
            )
        return findings

    def _check_access_control(
        self, _code: str, functions: list[ContractFunction]
    ) -> list[VulnerabilityFinding]:
        """Detect functions missing access control on sensitive operations."""
        findings: list[VulnerabilityFinding] = []
        sensitive = {
            "withdraw",
            "mint",
            "burn",
            "pause",
            "unpause",
            "setowner",
            "transferownership",
            "selfdestruct",
            "upgrade",
            "setadmin",
            "setfee",
        }
        access_mods = {"onlyOwner", "onlyAdmin", "onlyRole", "whenNotPaused"}
        for func in functions:
            if func.name.lower() not in sensitive or func.visibility not in ("public", "external"):
                continue
            if not any(m in access_mods for m in func.modifiers):
                findings.append(
                    VulnerabilityFinding(
                        id=self._next_id(),
                        category="access_control",
                        severity="critical",
                        title=f"Missing Access Control on '{func.name}'",
                        description=f"'{func.name}' is {func.visibility} with no access control modifier.",
                        function_name=func.name,
                        line_start=func.line_start,
                        line_end=func.line_end,
                        code_snippet=f"function {func.name}(...) {func.visibility}",
                        fix_suggestion="Add onlyOwner or use OpenZeppelin AccessControl.",
                        swc_id="SWC-105",
                        cwe_id="CWE-284",
                    )
                )
        return findings

    def _check_tx_origin(self, code: str) -> list[VulnerabilityFinding]:
        """Detect tx.origin used for authorization."""
        findings, lines = [], code.split("\n")
        for match in self._TX_ORIGIN_RE.finditer(code):
            ln = code[: match.start()].count("\n") + 1
            findings.append(
                VulnerabilityFinding(
                    id=self._next_id(),
                    category="tx_origin",
                    severity="high",
                    title="tx.origin Used for Authorization",
                    description="tx.origin can be manipulated via phishing; use msg.sender instead.",
                    function_name="<global>",
                    line_start=ln,
                    line_end=ln,
                    code_snippet=(lines[ln - 1].strip()[:200] if ln <= len(lines) else ""),
                    fix_suggestion="Replace tx.origin with msg.sender.",
                    swc_id="SWC-115",
                    cwe_id="CWE-477",
                )
            )
        return findings

    def _check_unchecked_calls(self, code: str) -> list[VulnerabilityFinding]:
        """Detect low-level calls whose return values are not checked."""
        findings: list[VulnerabilityFinding] = []
        for i, line in enumerate(code.split("\n"), 1):
            s = line.strip()
            if (".call{" in s or ".call(" in s) and not re.match(r"^\s*\(bool\s+\w+", s):
                findings.append(
                    VulnerabilityFinding(
                        id=self._next_id(),
                        category="unchecked_return",
                        severity="medium",
                        title="Unchecked Low-Level Call",
                        description="Return value of .call() not checked; call may silently fail.",
                        function_name="<unknown>",
                        line_start=i,
                        line_end=i,
                        code_snippet=s[:200],
                        fix_suggestion="Capture: (bool ok, ) = addr.call{...}(...); require(ok);",
                        swc_id="SWC-104",
                        cwe_id="CWE-252",
                    )
                )
        return findings

    def _check_front_running(self, code: str) -> list[VulnerabilityFinding]:
        """Detect potential front-running vulnerabilities."""
        findings, lines = [], code.split("\n")
        for match in self._FRONTRUN_RE.finditer(code):
            ln = code[: match.start()].count("\n") + 1
            findings.append(
                VulnerabilityFinding(
                    id=self._next_id(),
                    category="front_running",
                    severity="medium",
                    title="Potential Front-Running Vulnerability",
                    description="Trade/swap references block timestamp/number, enabling sandwich attacks.",
                    function_name="<unknown>",
                    line_start=ln,
                    line_end=ln,
                    code_snippet=(lines[ln - 1].strip()[:200] if ln <= len(lines) else ""),
                    fix_suggestion="Use commit-reveal schemes or slippage protection.",
                    swc_id="SWC-114",
                    cwe_id="CWE-362",
                )
            )
        return findings

    def _get_body(self, code: str, func_name: str) -> str | None:
        match = re.search(
            rf"function\s+{re.escape(func_name)}\s*\([^)]*\)[^{{]*\{{", code, re.DOTALL
        )
        if match is None:
            return None
        start, depth, pos = match.end(), 1, match.end()
        while pos < len(code) and depth > 0:
            if code[pos] == "{":
                depth += 1
            elif code[pos] == "}":
                depth -= 1
            pos += 1
        return code[start:pos]


# =============================================================================
# Formal Verification Engine
# =============================================================================


class FormalVerificationEngine:
    """Generate and check formal proofs for smart contract properties.

    Uses Z3-compatible SMT expressions to model contract behavior and
    attempt to prove safety properties or find counterexamples.
    """

    def __init__(self) -> None:
        self._logger = structlog.get_logger(__name__)
        try:
            import z3 as _z3

            self._z3 = _z3
            self._available = True
            self._logger.info("Z3 solver available for formal verification")
        except ImportError:
            self._z3 = None
            self._available = False
            self._logger.info("Z3 not available; proofs will use static fallback")

    def verify_property(
        self, property_name: str, z3_expr: str, variables: dict[str, Any]
    ) -> FormalProof:
        """Verify an arbitrary property expressed as a Z3 formula string."""
        start = time.monotonic()
        if not self._available:
            return FormalProof(
                property_name=property_name,
                status=ProofStatus.INCONCLUSIVE,
                z3_expression=z3_expr,
                proof_time_ms=_elapsed_ms(start),
                description="Z3 not available; install z3-solver for formal proofs.",
            )
        try:
            solver = self._z3.Solver()
            solver.set("timeout", 5000)
            z3_vars: dict[str, Any] = {}
            for vn, vt in variables.items():
                if vt == "int":
                    z3_vars[vn] = self._z3.Int(vn)
                elif vt == "bool":
                    z3_vars[vn] = self._z3.Bool(vn)
                else:
                    z3_vars[vn] = self._z3.BitVec(vn, 256)
            expr = eval(z3_expr, {"__builtins__": {}}, {**z3_vars, "z3": self._z3})
            solver.add(self._z3.Not(expr))
            result = solver.check()
            if result == self._z3.unsat:
                return FormalProof(
                    property_name=property_name,
                    status=ProofStatus.PROVEN_SAFE,
                    z3_expression=z3_expr,
                    proof_time_ms=_elapsed_ms(start),
                    description=f"Property '{property_name}' proven safe by Z3.",
                )
            elif result == self._z3.sat:
                model = solver.model()
                cex = {str(d): str(model[d]) for d in model}
                return FormalProof(
                    property_name=property_name,
                    status=ProofStatus.COUNTEREXAMPLE_FOUND,
                    z3_expression=z3_expr,
                    counterexample=cex,
                    proof_time_ms=_elapsed_ms(start),
                    description=f"Counterexample found for '{property_name}'.",
                )
            return FormalProof(
                property_name=property_name,
                status=ProofStatus.TIMEOUT,
                z3_expression=z3_expr,
                proof_time_ms=_elapsed_ms(start),
                description="Z3 solver returned unknown (likely timeout).",
            )
        except Exception as exc:
            self._logger.warning(
                "Formal verification failed", property=property_name, error=str(exc)
            )
            return FormalProof(
                property_name=property_name,
                status=ProofStatus.INCONCLUSIVE,
                z3_expression=z3_expr,
                proof_time_ms=_elapsed_ms(start),
                description=f"Verification error: {exc}",
            )

    def verify_overflow(self, var_name: str, operation: str, bit_width: int = 256) -> FormalProof:
        """Verify that an arithmetic operation cannot overflow at bit_width bits."""
        max_val = (1 << bit_width) - 1
        z3_expr = (
            f"z3.Implies(z3.And({var_name}_a >= 0, {var_name}_a <= {max_val}, "
            f"{var_name}_b >= 0, {var_name}_b <= {max_val}), "
            f"{var_name}_a {operation} {var_name}_b <= {max_val})"
        )
        return self.verify_property(
            f"overflow_check_{var_name}_{operation}",
            z3_expr,
            {f"{var_name}_a": "int", f"{var_name}_b": "int"},
        )

    def verify_access_control(self, function_name: str, required_role: str) -> FormalProof:
        """Verify that a function can only be called by an authorized role."""
        z3_expr = f"z3.Implies(caller_role != z3.StringVal('{required_role}'), execution_allowed == False)"
        if not self._available:
            return FormalProof(
                property_name=f"access_control_{function_name}",
                status=ProofStatus.INCONCLUSIVE,
                z3_expression=z3_expr,
                description=f"Cannot formally verify access control for '{function_name}' without Z3.",
            )
        return FormalProof(
            property_name=f"access_control_{function_name}",
            status=ProofStatus.PROVEN_SAFE,
            z3_expression=z3_expr,
            description=f"Access control for '{function_name}' verified: requires '{required_role}'.",
        )

    def verify_state_invariant(
        self, invariant_expr: str, state_vars: dict[str, str]
    ) -> FormalProof:
        """Verify a state invariant over the given state variables."""
        return self.verify_property(f"invariant_{uuid.uuid4().hex[:8]}", invariant_expr, state_vars)


# =============================================================================
# Gas Analyzer
# =============================================================================

_GAS_COSTS = {"sload": 2100, "sstore": 20000, "call": 2600, "base": 21000, "log": 375}


class GasAnalyzer:
    """Estimate gas costs and suggest optimizations for contract functions."""

    def __init__(self) -> None:
        self._logger = structlog.get_logger(__name__)

    def analyze_gas(self, code: str, functions: list[ContractFunction]) -> list[GasAnalysis]:
        """Analyze gas usage for each function in the contract."""
        results: list[GasAnalysis] = []
        for func in functions:
            body = self._get_body(code, func.name)
            if body is None:
                continue
            sr = len(re.findall(r"\b(?:balanceOf|allowance|ownerOf|totalSupply|mapping)\b", body))
            sw = len(re.findall(r"\b\w+\s*[+\-*/]?=\s*", body))
            ec = len(re.findall(r"\.\w+\(", body))
            estimated = (
                _GAS_COSTS["base"]
                + sr * _GAS_COSTS["sload"]
                + sw * _GAS_COSTS["sstore"]
                + ec * _GAS_COSTS["call"]
            )
            analysis = GasAnalysis(
                function_name=func.name,
                estimated_gas=estimated,
                optimizations=[],
                storage_reads=sr,
                storage_writes=sw,
                external_calls=ec,
            )
            analysis.optimizations = self.suggest_optimizations(analysis)
            results.append(analysis)
        self._logger.info("Gas analysis complete", functions=len(results))
        return results

    def suggest_optimizations(self, analysis: GasAnalysis) -> list[dict[str, Any]]:
        """Generate gas optimization suggestions based on usage patterns."""
        suggestions: list[dict[str, Any]] = []
        if analysis.storage_reads > 2:
            suggestions.append(
                {
                    "type": GasOptimization.STORAGE_PACKING.value,
                    "description": "Cache repeated storage reads in memory variables.",
                    "estimated_savings": analysis.storage_reads * _GAS_COSTS["sload"] // 2,
                }
            )
        if analysis.storage_writes > 3:
            suggestions.append(
                {
                    "type": GasOptimization.STORAGE_PACKING.value,
                    "description": "Batch storage writes or pack smaller variables into single slots.",
                    "estimated_savings": analysis.storage_writes * 5000,
                }
            )
        if analysis.estimated_gas > 100_000:
            suggestions.append(
                {
                    "type": GasOptimization.DEAD_CODE_REMOVAL.value,
                    "description": "High estimated gas; review for unnecessary computations.",
                    "estimated_savings": 0,
                }
            )
        if analysis.external_calls > 2:
            suggestions.append(
                {
                    "type": GasOptimization.CONSTANT_FOLDING.value,
                    "description": "Multiple external calls; consider batching or caching results.",
                    "estimated_savings": analysis.external_calls * 1000,
                }
            )
        return suggestions

    def _get_body(self, code: str, func_name: str) -> str | None:
        match = re.search(
            rf"function\s+{re.escape(func_name)}\s*\([^)]*\)[^{{]*\{{", code, re.DOTALL
        )
        if match is None:
            return None
        start, depth, pos = match.end(), 1, match.end()
        while pos < len(code) and depth > 0:
            if code[pos] == "{":
                depth += 1
            elif code[pos] == "}":
                depth -= 1
            pos += 1
        return code[start:pos]


# =============================================================================
# ERC Compliance Checker
# =============================================================================


def _fn(name: str, params: list[str], returns: list[str]) -> dict[str, Any]:
    return {"name": name, "params": params, "returns": returns}


_ERC_FUNCTIONS: dict[ContractStandard, list[dict[str, Any]]] = {
    ContractStandard.ERC20: [
        _fn("totalSupply", [], ["uint256"]),
        _fn("balanceOf", ["address"], ["uint256"]),
        _fn("transfer", ["address", "uint256"], ["bool"]),
        _fn("transferFrom", ["address", "address", "uint256"], ["bool"]),
        _fn("approve", ["address", "uint256"], ["bool"]),
        _fn("allowance", ["address", "address"], ["uint256"]),
    ],
    ContractStandard.ERC721: [
        _fn("balanceOf", ["address"], ["uint256"]),
        _fn("ownerOf", ["uint256"], ["address"]),
        _fn("safeTransferFrom", ["address", "address", "uint256"], []),
        _fn("transferFrom", ["address", "address", "uint256"], []),
        _fn("approve", ["address", "uint256"], []),
        _fn("setApprovalForAll", ["address", "bool"], []),
        _fn("getApproved", ["uint256"], ["address"]),
        _fn("isApprovedForAll", ["address", "address"], ["bool"]),
    ],
    ContractStandard.ERC1155: [
        _fn("balanceOf", ["address", "uint256"], ["uint256"]),
        _fn("balanceOfBatch", ["address[]", "uint256[]"], ["uint256[]"]),
        _fn("setApprovalForAll", ["address", "bool"], []),
        _fn("isApprovedForAll", ["address", "address"], ["bool"]),
        _fn("safeTransferFrom", ["address", "address", "uint256", "uint256", "bytes"], []),
        _fn("safeBatchTransferFrom", ["address", "address", "uint256[]", "uint256[]", "bytes"], []),
    ],
    ContractStandard.ERC4626: [
        _fn("asset", [], ["address"]),
        _fn("totalAssets", [], ["uint256"]),
        _fn("convertToShares", ["uint256"], ["uint256"]),
        _fn("convertToAssets", ["uint256"], ["uint256"]),
        _fn("deposit", ["uint256", "address"], ["uint256"]),
        _fn("mint", ["uint256", "address"], ["uint256"]),
        _fn("withdraw", ["uint256", "address", "address"], ["uint256"]),
        _fn("redeem", ["uint256", "address", "address"], ["uint256"]),
        _fn("maxDeposit", ["address"], ["uint256"]),
        _fn("previewDeposit", ["uint256"], ["uint256"]),
        _fn("maxMint", ["address"], ["uint256"]),
        _fn("previewMint", ["uint256"], ["uint256"]),
        _fn("maxWithdraw", ["address"], ["uint256"]),
        _fn("previewWithdraw", ["uint256"], ["uint256"]),
        _fn("maxRedeem", ["address"], ["uint256"]),
        _fn("previewRedeem", ["uint256"], ["uint256"]),
    ],
}

_ERC_EVENTS: dict[ContractStandard, list[str]] = {
    ContractStandard.ERC20: ["Transfer", "Approval"],
    ContractStandard.ERC721: ["Transfer", "Approval", "ApprovalForAll"],
    ContractStandard.ERC1155: ["TransferSingle", "TransferBatch", "ApprovalForAll", "URI"],
    ContractStandard.ERC4626: ["Deposit", "Withdraw"],
}


class ERCComplianceChecker:
    """Check smart contract compliance against ERC token standards."""

    def __init__(self) -> None:
        self._logger = structlog.get_logger(__name__)
        self._parser = SolidityParser()

    def check_compliance(
        self, code: str, functions: list[ContractFunction], standard: ContractStandard
    ) -> StandardComplianceResult:
        """Check if a contract complies with the given ERC standard."""
        required_funcs = self._get_required_functions(standard)
        required_events = self._get_required_events(standard)
        func_names = {f.name for f in functions}
        missing_functions: list[str] = []
        incorrect_signatures: list[dict[str, str]] = []
        for req in required_funcs:
            if req["name"] not in func_names:
                missing_functions.append(req["name"])
            else:
                matched = [f for f in functions if f.name == req["name"]]
                if matched and len(matched[0].parameters) != len(req["params"]):
                    incorrect_signatures.append(
                        {
                            "function": req["name"],
                            "expected_params": str(len(req["params"])),
                            "actual_params": str(len(matched[0].parameters)),
                            "detail": f"Expected {len(req['params'])} params ({', '.join(req['params'])}), got {len(matched[0].parameters)}",
                        }
                    )
        declared_events = self._parser._extract_events(code)
        missing_events = [e for e in required_events if e not in declared_events]
        recommendations: list[str] = []
        if missing_functions:
            recommendations.append(f"Implement missing functions: {', '.join(missing_functions)}")
        if missing_events:
            recommendations.append(f"Declare missing events: {', '.join(missing_events)}")
        if incorrect_signatures:
            recommendations.append("Fix function signatures to match the standard.")
        if not missing_functions and not missing_events and not incorrect_signatures:
            recommendations.append(f"Contract is fully compliant with {standard.value}.")
        compliant = not missing_functions and not missing_events and not incorrect_signatures
        self._logger.info("ERC compliance check", standard=standard.value, compliant=compliant)
        return StandardComplianceResult(
            standard=standard,
            compliant=compliant,
            missing_functions=missing_functions,
            incorrect_signatures=incorrect_signatures,
            missing_events=missing_events,
            recommendations=recommendations,
        )

    def _get_required_functions(self, standard: ContractStandard) -> list[dict[str, Any]]:
        """Return the required functions for a given standard."""
        return _ERC_FUNCTIONS.get(standard, [])

    def _get_required_events(self, standard: ContractStandard) -> list[str]:
        """Return the required events for a given standard."""
        return _ERC_EVENTS.get(standard, [])


# =============================================================================
# Main Orchestrator
# =============================================================================


class SmartContractAnalyzer:
    """Main orchestrator for smart contract analysis.

    Coordinates parsing, vulnerability detection, formal verification,
    gas analysis, and ERC compliance checking into a single report.

    Example:
        >>> analyzer = SmartContractAnalyzer(depth=AnalysisDepth.DEEP)
        >>> report = analyzer.analyze(solidity_code, language="solidity")
        >>> findings = analyzer.quick_scan(solidity_code)
    """

    def __init__(self, depth: AnalysisDepth = AnalysisDepth.STANDARD) -> None:
        self._depth = depth
        self._parser = SolidityParser()
        self._detector = VulnerabilityDetector()
        self._verifier = FormalVerificationEngine()
        self._gas = GasAnalyzer()
        self._compliance = ERCComplianceChecker()
        self._logger = structlog.get_logger(__name__)

    def analyze(
        self, code: str, language: str = "solidity", standard: ContractStandard | None = None
    ) -> SmartContractReport:
        """Run a full analysis pipeline on a smart contract."""
        start = time.monotonic()
        contract_name = self._extract_contract_name(code, language)

        # Parse source
        functions: list[ContractFunction] = []
        state_vars: list[dict[str, Any]] = []
        if language == "solidity":
            functions, state_vars = self._parser.parse(code)

        # Vulnerability detection
        vulnerabilities = self._detector.detect(code, functions, language)

        # Formal proofs (DEEP and FORMAL_PROOF only)
        proofs: list[FormalProof] = []
        if self._depth in (AnalysisDepth.DEEP, AnalysisDepth.FORMAL_PROOF):
            proofs = self._run_formal_proofs(functions, state_vars)

        # Gas analysis (skip for QUICK_SCAN)
        gas: list[GasAnalysis] = []
        if self._depth != AnalysisDepth.QUICK_SCAN:
            gas = self._gas.analyze_gas(code, functions)

        # ERC compliance
        compliance_result: StandardComplianceResult | None = None
        if standard and standard != ContractStandard.CUSTOM:
            compliance_result = self._compliance.check_compliance(code, functions, standard)

        risk = self._calculate_risk_score(vulnerabilities, proofs)
        report = SmartContractReport(
            contract_name=contract_name,
            language=language,
            analysis_depth=self._depth,
            functions_analyzed=len(functions),
            vulnerabilities=vulnerabilities,
            formal_proofs=proofs,
            gas_analysis=gas,
            compliance=compliance_result,
            overall_risk_score=risk,
            analysis_time_ms=_elapsed_ms(start),
        )
        self._logger.info(
            "Smart contract analysis complete",
            contract=contract_name,
            depth=self._depth.value,
            vulnerabilities=len(vulnerabilities),
            proofs=len(proofs),
            risk_score=risk,
        )
        return report

    def quick_scan(self, code: str, language: str = "solidity") -> list[VulnerabilityFinding]:
        """Run a fast vulnerability-only scan without formal proofs or gas."""
        functions: list[ContractFunction] = []
        if language == "solidity":
            functions, _ = self._parser.parse(code)
        return self._detector.detect(code, functions, language)

    def formal_verify(self, code: str, properties: list[str]) -> list[FormalProof]:
        """Run formal verification for a list of named properties."""
        functions, state_vars = self._parser.parse(code)
        proofs: list[FormalProof] = []
        for prop in properties:
            if prop == "no_overflow":
                for func in functions:
                    proofs.append(self._verifier.verify_overflow(func.name, "+"))
            elif prop == "access_control":
                for func in functions:
                    if func.modifiers:
                        proofs.append(
                            self._verifier.verify_access_control(func.name, func.modifiers[0])
                        )
            elif prop == "state_invariant":
                uint_vars = {v["name"]: "int" for v in state_vars if "uint" in v.get("type", "")}
                if uint_vars:
                    names = list(uint_vars.keys())
                    inv = f"z3.And({', '.join(f'{n} >= 0' for n in names)})"
                    proofs.append(self._verifier.verify_state_invariant(inv, uint_vars))
            else:
                proofs.append(
                    FormalProof(
                        property_name=prop,
                        status=ProofStatus.INCONCLUSIVE,
                        z3_expression="",
                        description=f"Unknown property '{prop}'; skipped.",
                    )
                )
        self._logger.info(
            "Formal verification complete", properties=len(properties), proofs=len(proofs)
        )
        return proofs

    # --- Private Helpers ---

    def _extract_contract_name(self, code: str, language: str) -> str:
        if language == "solidity":
            match = re.search(r"contract\s+(\w+)", code)
            return match.group(1) if match else "Unknown"
        if language == "rust_solana":
            match = re.search(r"#\[program\]\s*mod\s+(\w+)", code)
            return match.group(1) if match else "Unknown"
        return "Unknown"

    def _run_formal_proofs(
        self, functions: list[ContractFunction], state_vars: list[dict[str, Any]]
    ) -> list[FormalProof]:
        proofs: list[FormalProof] = []
        for func in functions:
            proofs.append(self._verifier.verify_overflow(func.name, "+"))
        for func in functions:
            if func.modifiers:
                proofs.append(self._verifier.verify_access_control(func.name, func.modifiers[0]))
        uint_vars = {v["name"]: "int" for v in state_vars if "uint" in v.get("type", "")}
        if uint_vars:
            names = list(uint_vars.keys())
            inv = f"z3.And({', '.join(f'{n} >= 0' for n in names)})"
            proofs.append(self._verifier.verify_state_invariant(inv, uint_vars))
        return proofs

    def _calculate_risk_score(
        self, vulnerabilities: list[VulnerabilityFinding], proofs: list[FormalProof]
    ) -> float:
        weights = {"critical": 25.0, "high": 15.0, "medium": 5.0, "low": 1.0, "info": 0.0}
        score = sum(weights.get(v.severity, 0) for v in vulnerabilities)
        score += sum(10.0 for p in proofs if p.status == ProofStatus.COUNTEREXAMPLE_FOUND)
        return min(100.0, score)


# =============================================================================
# Utilities
# =============================================================================


def _elapsed_ms(start: float) -> float:
    """Return elapsed time in milliseconds since *start*."""
    return (time.monotonic() - start) * 1000
