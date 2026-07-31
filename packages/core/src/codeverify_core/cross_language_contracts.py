"""Cross-Language Contract Verification.

Verifies API contracts across language boundaries (e.g., Python calling
TypeScript API). Extracts types, maps between languages, detects incompatibilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()

# =============================================================================
# Enums
# =============================================================================


class ContractLanguage(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"


class TypeCompatibility(str, Enum):
    COMPATIBLE = "compatible"
    COERCIBLE = "coercible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class ContractViolationType(str, Enum):
    TYPE_MISMATCH = "type_mismatch"
    MISSING_FIELD = "missing_field"
    NULLABILITY = "nullability"
    RANGE_VIOLATION = "range_violation"
    ENUM_MISMATCH = "enum_mismatch"
    PROTOCOL_ERROR = "protocol_error"


class VerificationScope(str, Enum):
    API_BOUNDARY = "api_boundary"
    FUNCTION_CALL = "function_call"
    EVENT_PAYLOAD = "event_payload"
    DATABASE_SCHEMA = "database_schema"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class UniversalType:
    """Language-agnostic type representation."""

    name: str
    language: ContractLanguage
    native_type: str
    nullable: bool = False
    generic_params: list[UniversalType] = field(default_factory=list)
    properties: dict[str, UniversalType] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "language": self.language.value,
            "native_type": self.native_type,
            "nullable": self.nullable,
            "generic_params": [p.to_dict() for p in self.generic_params],
            "properties": {k: v.to_dict() for k, v in self.properties.items()},
            "constraints": self.constraints,
        }


@dataclass
class ContractEndpoint:
    """A single endpoint in a cross-language contract."""

    name: str
    language: ContractLanguage
    file_path: str
    line: int
    parameters: dict[str, UniversalType] = field(default_factory=dict)
    return_type: UniversalType | None = None
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "language": self.language.value,
            "file_path": self.file_path,
            "line": self.line,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "return_type": self.return_type.to_dict() if self.return_type else None,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
        }


@dataclass
class ContractViolation:
    """A detected violation between two contract endpoints."""

    violation_type: ContractViolationType
    source_endpoint: ContractEndpoint
    target_endpoint: ContractEndpoint
    description: str
    severity: str
    source_type: str
    target_type: str
    fix_suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "source_endpoint": self.source_endpoint.name,
            "target_endpoint": self.target_endpoint.name,
            "description": self.description,
            "severity": self.severity,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "fix_suggestion": self.fix_suggestion,
        }


@dataclass
class TypeMapping:
    """Mapping between types in two different languages."""

    source_lang: ContractLanguage
    target_lang: ContractLanguage
    source_type: str
    target_type: str
    compatibility: TypeCompatibility
    coercion_needed: bool = False
    coercion_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_lang": self.source_lang.value,
            "target_lang": self.target_lang.value,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "compatibility": self.compatibility.value,
            "coercion_needed": self.coercion_needed,
            "coercion_code": self.coercion_code,
        }


@dataclass
class CrossLanguageContractReport:
    """Summary report of cross-language contract verification."""

    contracts_checked: int
    violations_found: int
    violations: list[ContractViolation] = field(default_factory=list)
    type_mappings: list[TypeMapping] = field(default_factory=list)
    compatible_endpoints: int = 0
    total_endpoints: int = 0
    risk_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contracts_checked": self.contracts_checked,
            "violations_found": self.violations_found,
            "violations": [v.to_dict() for v in self.violations],
            "type_mappings": [m.to_dict() for m in self.type_mappings],
            "compatible_endpoints": self.compatible_endpoints,
            "total_endpoints": self.total_endpoints,
            "risk_score": self.risk_score,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Type Equivalence Tables
# =============================================================================

_PY = ContractLanguage.PYTHON
_TS = ContractLanguage.TYPESCRIPT
_GO = ContractLanguage.GO
_JV = ContractLanguage.JAVA
_RS = ContractLanguage.RUST

_TYPE_EQUIVALENTS: dict[ContractLanguage, dict[str, list[str]]] = {
    _PY: {
        "str": ["string", "String"],
        "int": ["number", "i32", "i64", "Integer"],
        "float": ["number", "f32", "f64", "double", "Double"],
        "bool": ["boolean", "Bool"],
        "list": ["Array", "Vec", "List", "ArrayList"],
        "dict": ["Record", "object", "Map", "HashMap"],
        "None": ["void", "undefined", "null", "()", "Void"],
        "bytes": ["Buffer", "Uint8Array", "[]byte", "Vec<u8>"],
        "Any": ["any", "unknown", "interface{}", "Object"],
    },
    _TS: {
        "string": ["str", "String"],
        "number": ["int", "float", "i32", "i64", "f64"],
        "boolean": ["bool", "Bool"],
        "Array": ["list", "Vec", "List", "ArrayList"],
        "Record": ["dict", "Map", "HashMap", "object"],
        "void": ["None", "null", "()", "Void"],
        "any": ["Any", "unknown", "interface{}", "Object"],
        "Buffer": ["bytes", "[]byte", "Vec<u8>", "Uint8Array"],
    },
    _GO: {
        "string": ["str", "String"],
        "int": ["int", "number", "Integer"],
        "float64": ["float", "number", "f64", "Double"],
        "bool": ["bool", "boolean", "Bool"],
        "[]byte": ["bytes", "Buffer", "Uint8Array"],
        "interface{}": ["Any", "any", "unknown", "Object"],
    },
    _JV: {
        "String": ["str", "string"],
        "int": ["int", "number", "i32"],
        "long": ["int", "number", "i64"],
        "double": ["float", "number", "f64"],
        "boolean": ["bool", "boolean", "Bool"],
        "List": ["list", "Array", "Vec", "ArrayList"],
        "Map": ["dict", "Record", "HashMap", "object"],
        "void": ["None", "void", "undefined", "()"],
    },
    _RS: {
        "String": ["str", "string"],
        "i32": ["int", "number", "Integer"],
        "i64": ["int", "number", "long"],
        "f64": ["float", "number", "double"],
        "bool": ["bool", "boolean", "Bool"],
        "Vec": ["list", "Array", "List", "ArrayList"],
        "HashMap": ["dict", "Record", "Map", "object"],
        "()": ["None", "void", "undefined", "Void"],
    },
}

_COERCIBLE_PAIRS: list[tuple[str, str]] = [
    ("int", "float"),
    ("int", "number"),
    ("float", "number"),
    ("i32", "f64"),
    ("i64", "f64"),
    ("int", "string"),
    ("str", "bytes"),
]

# =============================================================================
# Type Mapper
# =============================================================================


class TypeMapper:
    """Maps and checks compatibility of types across languages."""

    def __init__(self) -> None:
        self._equivalents = _TYPE_EQUIVALENTS
        self._coercible = _COERCIBLE_PAIRS

    def map_type(self, source_type: UniversalType, target_lang: ContractLanguage) -> TypeMapping:
        """Map a source type to its best equivalent in the target language."""
        target_equivs = self._equivalents.get(target_lang, {})
        for target_native, aliases in target_equivs.items():
            if source_type.native_type in aliases or source_type.native_type == target_native:
                return TypeMapping(
                    source_lang=source_type.language,
                    target_lang=target_lang,
                    source_type=source_type.native_type,
                    target_type=target_native,
                    compatibility=TypeCompatibility.COMPATIBLE,
                )
        # Check coercible pairs
        src_equivs = self._get_type_equivalents(source_type.native_type, source_type.language)
        for src, tgt in self._coercible:
            if source_type.native_type == src or src in src_equivs:
                for tn, ta in target_equivs.items():
                    if tn == tgt or tgt in ta:
                        return TypeMapping(
                            source_lang=source_type.language,
                            target_lang=target_lang,
                            source_type=source_type.native_type,
                            target_type=tn,
                            compatibility=TypeCompatibility.COERCIBLE,
                            coercion_needed=True,
                            coercion_code=f"Convert {source_type.native_type} to {tn}",
                        )
        logger.warning(
            "no_type_mapping",
            source_type=source_type.native_type,
            source_lang=source_type.language.value,
            target_lang=target_lang.value,
        )
        return TypeMapping(
            source_lang=source_type.language,
            target_lang=target_lang,
            source_type=source_type.native_type,
            target_type="unknown",
            compatibility=TypeCompatibility.UNKNOWN,
        )

    def check_compatibility(
        self, source: UniversalType, target: UniversalType
    ) -> TypeCompatibility:
        """Check whether two universal types are compatible."""
        if source.native_type == target.native_type:
            if not self._check_nullability_compat(source, target):
                return TypeCompatibility.INCOMPATIBLE
            if not self._check_generic_compat(source, target):
                return TypeCompatibility.INCOMPATIBLE
            return TypeCompatibility.COMPATIBLE

        src_eq = self._get_type_equivalents(source.native_type, source.language)
        tgt_eq = self._get_type_equivalents(target.native_type, target.language)
        if target.native_type in src_eq or source.native_type in tgt_eq:
            if not self._check_nullability_compat(source, target):
                return TypeCompatibility.INCOMPATIBLE
            if not self._check_generic_compat(source, target):
                return TypeCompatibility.INCOMPATIBLE
            return TypeCompatibility.COMPATIBLE

        for src, tgt in self._coercible:
            if (source.native_type == src or src in src_eq) and (
                target.native_type == tgt or tgt in tgt_eq
            ):
                return TypeCompatibility.COERCIBLE
        return TypeCompatibility.INCOMPATIBLE

    def _get_type_equivalents(self, native_type: str, lang: ContractLanguage) -> list[str]:
        lang_map = self._equivalents.get(lang, {})
        if native_type in lang_map:
            return lang_map[native_type]
        for base_type, aliases in lang_map.items():
            if native_type in aliases:
                return [base_type] + aliases
        return []

    def _check_nullability_compat(self, source: UniversalType, target: UniversalType) -> bool:
        return not (source.nullable and not target.nullable)

    def _check_generic_compat(self, source: UniversalType, target: UniversalType) -> bool:
        if not source.generic_params and not target.generic_params:
            return True
        if len(source.generic_params) != len(target.generic_params):
            return False
        return all(
            self.check_compatibility(sp, tp) != TypeCompatibility.INCOMPATIBLE
            for sp, tp in zip(source.generic_params, target.generic_params, strict=True)
        )


# =============================================================================
# Contract Extractor
# =============================================================================


class ContractExtractor:
    """Extracts contract endpoints from source code across languages."""

    def __init__(self) -> None:
        self._type_mapper = TypeMapper()

    def extract_contracts(
        self,
        code: str,
        language: ContractLanguage,
        file_path: str,
    ) -> list[ContractEndpoint]:
        """Extract contract endpoints from source code."""
        if language == ContractLanguage.PYTHON:
            return self._extract_python_contracts(code, file_path)
        if language == ContractLanguage.TYPESCRIPT:
            return self._extract_typescript_contracts(code, file_path)
        logger.info("unsupported_language_extraction", language=language.value)
        return []

    def _extract_python_contracts(self, code: str, file_path: str) -> list[ContractEndpoint]:
        endpoints: list[ContractEndpoint] = []
        pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?:async\s+)?def\s+(?P<name>\w+)\s*\("
            r"(?P<params>[^)]*)\)(?:\s*->\s*(?P<ret>[^:]+))?\s*:",
            re.MULTILINE,
        )
        for m in pattern.finditer(code):
            name = m.group("name")
            if name.startswith("_") or len(m.group("indent")) > 0:
                continue
            line = code[: m.start()].count("\n") + 1
            params: dict[str, UniversalType] = {}
            for part in self._split_params(m.group("params")):
                part = part.strip()
                if not part or part in ("self", "cls"):
                    continue
                if "=" in part:
                    part = part.split("=", 1)[0].strip()
                if ":" in part:
                    pn, pt = part.split(":", 1)
                    params[pn.strip()] = self._parse_type_annotation(pt.strip(), _PY)
                else:
                    params[part] = UniversalType(name=part, language=_PY, native_type="Any")
            ret_str = (m.group("ret") or "").strip()
            return_type = (
                self._parse_type_annotation(ret_str, _PY) if ret_str and ret_str != "None" else None
            )
            endpoints.append(
                ContractEndpoint(
                    name=name,
                    language=_PY,
                    file_path=file_path,
                    line=line,
                    parameters=params,
                    return_type=return_type,
                )
            )
        logger.debug("python_contracts_extracted", file_path=file_path, count=len(endpoints))
        return endpoints

    def _extract_typescript_contracts(self, code: str, file_path: str) -> list[ContractEndpoint]:
        endpoints: list[ContractEndpoint] = []
        pattern = re.compile(
            r"(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)\s*"
            r"(?:<[^>]*>)?\s*\((?P<params>[^)]*)\)\s*(?::\s*(?P<ret>[^{]+))?\s*\{",
            re.MULTILINE,
        )
        for m in pattern.finditer(code):
            name = m.group("name")
            line = code[: m.start()].count("\n") + 1
            params: dict[str, UniversalType] = {}
            for part in self._split_params(m.group("params")):
                part = part.strip()
                if not part:
                    continue
                optional = "?" in part
                part = part.replace("?", "")
                if "=" in part:
                    part = part.split("=", 1)[0].strip()
                if ":" in part:
                    pn, pt = part.split(":", 1)
                    ut = self._parse_type_annotation(pt.strip(), _TS)
                    ut.nullable = ut.nullable or optional
                    params[pn.strip()] = ut
                else:
                    params[part.strip()] = UniversalType(
                        name=part.strip(),
                        language=_TS,
                        native_type="any",
                        nullable=optional,
                    )
            ret_str = (m.group("ret") or "").strip()
            return_type = (
                self._parse_type_annotation(ret_str, _TS)
                if ret_str and ret_str not in ("void", "")
                else None
            )
            endpoints.append(
                ContractEndpoint(
                    name=name,
                    language=_TS,
                    file_path=file_path,
                    line=line,
                    parameters=params,
                    return_type=return_type,
                )
            )
        logger.debug("typescript_contracts_extracted", file_path=file_path, count=len(endpoints))
        return endpoints

    def _parse_type_annotation(self, annotation: str, language: ContractLanguage) -> UniversalType:
        """Parse a type annotation string into a UniversalType."""
        annotation = annotation.strip()
        nullable = False
        opt_match = re.match(r"Optional\[(.+)\]$", annotation)
        if opt_match:
            annotation, nullable = opt_match.group(1).strip(), True
        for token in (" | None", "None | ", " | null", " | undefined"):
            if token in annotation:
                annotation, nullable = annotation.replace(token, "").strip(), True
        generic_match = re.match(r"(\w+)[\[<](.+)[\]>]$", annotation)
        if generic_match:
            base = generic_match.group(1)
            gp = [
                self._parse_type_annotation(p.strip(), language)
                for p in self._split_params(generic_match.group(2))
            ]
            return UniversalType(
                name=base,
                language=language,
                native_type=base,
                nullable=nullable,
                generic_params=gp,
            )
        return UniversalType(
            name=annotation, language=language, native_type=annotation, nullable=nullable
        )

    def _split_params(self, params_str: str) -> list[str]:
        """Split parameter string respecting nested brackets."""
        parts: list[str] = []
        depth = 0
        current = ""
        for ch in params_str:
            if ch in "([{<":
                depth += 1
            elif ch in ")]}>":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(current)
                current = ""
                continue
            current += ch
        if current.strip():
            parts.append(current)
        return parts


# =============================================================================
# Cross-Language Verifier
# =============================================================================


class CrossLanguageVerifier:
    """Verifies compatibility between contract endpoints across languages."""

    def __init__(self) -> None:
        self._type_mapper = TypeMapper()
        self._extractor = ContractExtractor()

    def verify_contracts(
        self,
        source_contracts: list[ContractEndpoint],
        target_contracts: list[ContractEndpoint],
    ) -> CrossLanguageContractReport:
        """Verify all matched endpoint pairs and produce a report."""
        pairs = self._match_endpoints(source_contracts, target_contracts)
        all_violations: list[ContractViolation] = []
        all_mappings: list[TypeMapping] = []
        compatible_count = 0
        for source, target in pairs:
            violations = self.verify_endpoint_pair(source, target)
            all_violations.extend(violations)
            for stype in source.parameters.values():
                all_mappings.append(self._type_mapper.map_type(stype, target.language))
            if not violations:
                compatible_count += 1
        checked = len(pairs)
        crit = sum(1 for v in all_violations if v.severity == "critical")
        high = sum(1 for v in all_violations if v.severity == "high")
        med = sum(1 for v in all_violations if v.severity == "medium")
        risk = min(10.0, (crit * 3.0 + high * 2.0 + med) / max(checked, 1))
        report = CrossLanguageContractReport(
            contracts_checked=checked,
            violations_found=len(all_violations),
            violations=all_violations,
            type_mappings=all_mappings,
            compatible_endpoints=compatible_count,
            total_endpoints=len(source_contracts) + len(target_contracts),
            risk_score=round(risk, 2),
            recommendations=self._build_recommendations(all_violations, risk),
        )
        logger.info(
            "contract_verification_complete",
            contracts_checked=checked,
            violations_found=len(all_violations),
            risk_score=report.risk_score,
        )
        return report

    def verify_endpoint_pair(
        self,
        source: ContractEndpoint,
        target: ContractEndpoint,
    ) -> list[ContractViolation]:
        """Verify compatibility of a single source/target endpoint pair."""
        violations: list[ContractViolation] = []
        violations.extend(self._check_parameter_compatibility(source, target))
        violations.extend(self._check_return_type_compatibility(source, target))
        return violations

    def _match_endpoints(
        self,
        sources: list[ContractEndpoint],
        targets: list[ContractEndpoint],
    ) -> list[tuple[ContractEndpoint, ContractEndpoint]]:
        tmap = {ep.name: ep for ep in targets}
        pairs: list[tuple[ContractEndpoint, ContractEndpoint]] = []
        for src in sources:
            tgt = tmap.get(src.name)
            if tgt:
                pairs.append((src, tgt))
            else:
                logger.debug("unmatched_endpoint", name=src.name)
        return pairs

    def _check_parameter_compatibility(
        self,
        source: ContractEndpoint,
        target: ContractEndpoint,
    ) -> list[ContractViolation]:
        violations: list[ContractViolation] = []
        sl, tl = source.language.value, target.language.value

        for pname, stype in source.parameters.items():
            if pname not in target.parameters:
                violations.append(
                    ContractViolation(
                        violation_type=ContractViolationType.MISSING_FIELD,
                        source_endpoint=source,
                        target_endpoint=target,
                        description=f"Parameter '{pname}' in {sl} '{source.name}' missing in {tl} '{target.name}'.",
                        severity="high",
                        source_type=stype.native_type,
                        target_type="<missing>",
                        fix_suggestion=f"Add parameter '{pname}: {stype.native_type}' to {target.name}.",
                    )
                )
                continue
            ttype = target.parameters[pname]
            compat = self._type_mapper.check_compatibility(stype, ttype)
            if compat == TypeCompatibility.INCOMPATIBLE:
                violations.append(
                    ContractViolation(
                        violation_type=ContractViolationType.TYPE_MISMATCH,
                        source_endpoint=source,
                        target_endpoint=target,
                        description=f"Parameter '{pname}' type mismatch: {stype.native_type} ({sl}) vs {ttype.native_type} ({tl}).",
                        severity="critical",
                        source_type=stype.native_type,
                        target_type=ttype.native_type,
                        fix_suggestion=f"Change '{pname}' in {target.name} to match {stype.native_type}.",
                    )
                )
            if stype.nullable and not ttype.nullable:
                violations.append(
                    ContractViolation(
                        violation_type=ContractViolationType.NULLABILITY,
                        source_endpoint=source,
                        target_endpoint=target,
                        description=f"Parameter '{pname}' is nullable in {sl} but non-nullable in {tl}.",
                        severity="high",
                        source_type=f"{stype.native_type} (nullable)",
                        target_type=f"{ttype.native_type} (non-nullable)",
                        fix_suggestion=f"Mark '{pname}' as nullable/optional in {target.name}.",
                    )
                )
        return violations

    def _check_return_type_compatibility(
        self,
        source: ContractEndpoint,
        target: ContractEndpoint,
    ) -> list[ContractViolation]:
        violations: list[ContractViolation] = []
        sr, tr = source.return_type, target.return_type
        if sr is None and tr is None:
            return violations
        if sr is not None and tr is None:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.TYPE_MISMATCH,
                    source_endpoint=source,
                    target_endpoint=target,
                    description=f"{source.name} returns {sr.native_type} but {target.name} returns void.",
                    severity="critical",
                    source_type=sr.native_type,
                    target_type="void",
                )
            )
            return violations
        if sr is None and tr is not None:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.TYPE_MISMATCH,
                    source_endpoint=source,
                    target_endpoint=target,
                    description=f"{source.name} returns void but {target.name} returns {tr.native_type}.",
                    severity="high",
                    source_type="void",
                    target_type=tr.native_type,
                )
            )
            return violations

        assert sr is not None and tr is not None
        compat = self._type_mapper.check_compatibility(sr, tr)
        if compat == TypeCompatibility.INCOMPATIBLE:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.TYPE_MISMATCH,
                    source_endpoint=source,
                    target_endpoint=target,
                    description=f"Return type mismatch: {sr.native_type} ({source.language.value}) vs {tr.native_type} ({target.language.value}).",
                    severity="critical",
                    source_type=sr.native_type,
                    target_type=tr.native_type,
                    fix_suggestion=f"Align return type of {target.name} with {sr.native_type}.",
                )
            )
        if sr.nullable and not tr.nullable:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.NULLABILITY,
                    source_endpoint=source,
                    target_endpoint=target,
                    description=f"Return nullable in {source.language.value} but not in {target.language.value}.",
                    severity="medium",
                    source_type=f"{sr.native_type} (nullable)",
                    target_type=f"{tr.native_type} (non-nullable)",
                )
            )
        return violations

    def generate_interface_stubs(
        self,
        contracts: list[ContractEndpoint],
        target_lang: ContractLanguage,
    ) -> str:
        """Generate interface stub code for contracts in the target language."""
        lines: list[str] = []
        mt = self._type_mapper.map_type
        if target_lang == _TS:
            lines.append("// Auto-generated interface stubs\n")
            for ep in contracts:
                ps = [
                    f"{n}{'?' if t.nullable else ''}: {mt(t, target_lang).target_type}"
                    for n, t in ep.parameters.items()
                ]
                r = mt(ep.return_type, target_lang).target_type if ep.return_type else "void"
                lines.append(f"export function {ep.name}({', '.join(ps)}): {r};")
        elif target_lang == _PY:
            lines.append("# Auto-generated interface stubs\n")
            for ep in contracts:
                ps = [
                    f"{n}: {mt(t, target_lang).target_type + ' | None' if t.nullable else mt(t, target_lang).target_type}"
                    for n, t in ep.parameters.items()
                ]
                r = mt(ep.return_type, target_lang).target_type if ep.return_type else "None"
                lines.append(f"def {ep.name}({', '.join(ps)}) -> {r}: ...")
        elif target_lang == _GO:
            lines.append("// Auto-generated interface stubs\npackage stubs\n")
            for ep in contracts:
                ps = [
                    f"{n} {'*' + mt(t, target_lang).target_type if t.nullable else mt(t, target_lang).target_type}"
                    for n, t in ep.parameters.items()
                ]
                r = f" {mt(ep.return_type, target_lang).target_type}" if ep.return_type else ""
                lines.append(f"func {ep.name[0].upper() + ep.name[1:]}({', '.join(ps)}){r} {{}}")
        else:
            lines.append(f"// Stub generation not supported for {target_lang.value}")
        lines.append("")
        return "\n".join(lines)

    def _build_recommendations(
        self,
        violations: list[ContractViolation],
        risk_score: float,
    ) -> list[str]:
        recs: list[str] = []
        counts: dict[ContractViolationType, int] = {}
        for v in violations:
            counts[v.violation_type] = counts.get(v.violation_type, 0) + 1
        tm = counts.get(ContractViolationType.TYPE_MISMATCH, 0)
        ni = counts.get(ContractViolationType.NULLABILITY, 0)
        mf = counts.get(ContractViolationType.MISSING_FIELD, 0)
        if tm:
            recs.append(
                f"Resolve {tm} type mismatch(es). Consider shared schemas (OpenAPI, protobuf)."
            )
        if ni:
            recs.append(
                f"Fix {ni} nullability inconsistency(ies). Annotate nullable types on both sides."
            )
        if mf:
            recs.append(f"Add {mf} missing field(s) to target contracts.")
        if risk_score >= 7.0:
            recs.append("High risk score. Add integration tests for cross-language call sites.")
        if not violations:
            recs.append("All cross-language contracts are compatible.")
        return recs
