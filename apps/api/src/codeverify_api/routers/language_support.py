"""Language support API router for Go and Java expansion."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


class LanguageInfo(BaseModel):
    language: str
    file_extensions: list[str]
    null_type: str
    supports_generics: bool
    supports_null_safety: bool
    verification_checks: list[str]
    status: str = Field(description="support status: stable, beta, planned")


class FunctionExtractionRequest(BaseModel):
    code: str = Field(description="Source code to extract functions from")
    language: str = Field(description="Programming language")


class FunctionInfo(BaseModel):
    name: str
    params: list[str]
    return_type: str | None
    start_line: int
    end_line: int
    is_async: bool


class FunctionExtractionResponse(BaseModel):
    functions: list[FunctionInfo]
    language: str
    total_lines: int


SUPPORTED_LANGUAGES = {
    "python": LanguageInfo(
        language="python",
        file_extensions=[".py", ".pyi"],
        null_type="None",
        supports_generics=True,
        supports_null_safety=False,
        verification_checks=["null_safety", "array_bounds", "integer_overflow", "division_by_zero"],
        status="stable",
    ),
    "typescript": LanguageInfo(
        language="typescript",
        file_extensions=[".ts", ".tsx"],
        null_type="null | undefined",
        supports_generics=True,
        supports_null_safety=True,
        verification_checks=["null_safety", "array_bounds", "type_safety"],
        status="stable",
    ),
    "go": LanguageInfo(
        language="go",
        file_extensions=[".go"],
        null_type="nil",
        supports_generics=True,
        supports_null_safety=False,
        verification_checks=["null_safety", "array_bounds", "integer_overflow", "division_by_zero", "error_handling"],
        status="beta",
    ),
    "java": LanguageInfo(
        language="java",
        file_extensions=[".java"],
        null_type="null",
        supports_generics=True,
        supports_null_safety=False,
        verification_checks=["null_safety", "array_bounds", "integer_overflow", "division_by_zero", "exception_handling"],
        status="beta",
    ),
    "rust": LanguageInfo(
        language="rust",
        file_extensions=[".rs"],
        null_type="None (Option)",
        supports_generics=True,
        supports_null_safety=True,
        verification_checks=["integer_overflow", "lifetime_safety"],
        status="planned",
    ),
}


@router.get("", response_model=list[LanguageInfo])
async def list_languages(
    status_filter: str | None = Query(default=None, alias="status", description="Filter by status"),
) -> list[LanguageInfo]:
    """List all supported languages and their capabilities."""
    langs = list(SUPPORTED_LANGUAGES.values())
    if status_filter:
        langs = [l for l in langs if l.status == status_filter]
    return langs


@router.get("/{language}", response_model=LanguageInfo)
async def get_language(language: str) -> LanguageInfo:
    """Get details for a specific language."""
    info = SUPPORTED_LANGUAGES.get(language)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Language '{language}' not supported. Available: {list(SUPPORTED_LANGUAGES.keys())}",
        )
    return info


@router.get("/{language}/verification-rules")
async def get_verification_rules(language: str) -> dict[str, Any]:
    """Get language-specific verification rules and patterns."""
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Language not supported")

    rules: dict[str, Any] = {"language": language, "rules": {}}

    if language == "go":
        rules["rules"] = {
            "null_safety": {
                "null_type": "nil",
                "patterns": ["if err != nil", "if x == nil"],
                "common_issues": ["unchecked error returns", "nil pointer dereference", "nil map access"],
            },
            "integer_overflow": {
                "types": [
                    {"name": "int8", "bits": 8, "min": -128, "max": 127},
                    {"name": "int16", "bits": 16, "min": -32768, "max": 32767},
                    {"name": "int32", "bits": 32, "min": -2147483648, "max": 2147483647},
                    {"name": "int64", "bits": 64, "min": -9223372036854775808, "max": 9223372036854775807},
                    {"name": "uint8", "bits": 8, "min": 0, "max": 255},
                    {"name": "uint16", "bits": 16, "min": 0, "max": 65535},
                    {"name": "uint32", "bits": 32, "min": 0, "max": 4294967295},
                ],
            },
            "error_handling": {
                "patterns": ["if err != nil { return", "errors.New(", "fmt.Errorf("],
                "anti_patterns": ["_ = someFunc()", "ignoring error return value"],
            },
        }
    elif language == "java":
        rules["rules"] = {
            "null_safety": {
                "null_type": "null",
                "patterns": ["if (x != null)", "Objects.requireNonNull(", "@NonNull", "@Nullable"],
                "common_issues": ["NullPointerException", "unboxing null Integer/Long", "null collection iteration"],
            },
            "integer_overflow": {
                "types": [
                    {"name": "byte", "bits": 8, "min": -128, "max": 127},
                    {"name": "short", "bits": 16, "min": -32768, "max": 32767},
                    {"name": "int", "bits": 32, "min": -2147483648, "max": 2147483647},
                    {"name": "long", "bits": 64, "min": -9223372036854775808, "max": 9223372036854775807},
                ],
                "safe_methods": ["Math.addExact()", "Math.multiplyExact()", "Math.subtractExact()"],
            },
            "exception_handling": {
                "patterns": ["try {", "catch (", "finally {", "throws"],
                "anti_patterns": ["catch (Exception e) {}", "empty catch block", "catching Throwable"],
            },
        }
    elif language == "python":
        rules["rules"] = {
            "null_safety": {
                "null_type": "None",
                "patterns": ["if x is not None", "if x is None", "Optional["],
                "common_issues": ["AttributeError on None", "TypeError: NoneType"],
            },
            "integer_overflow": {
                "note": "Python integers have arbitrary precision, but numpy/ctypes integers can overflow",
                "types": [{"name": "numpy.int32", "bits": 32}, {"name": "numpy.int64", "bits": 64}],
            },
        }
    elif language == "typescript":
        rules["rules"] = {
            "null_safety": {
                "null_types": ["null", "undefined"],
                "patterns": ["if (x !== null)", "x?.prop", "x ?? default", "x!.prop"],
                "common_issues": ["Cannot read property of undefined", "non-null assertion misuse"],
            },
            "type_safety": {
                "patterns": ["as any", "any type", "// @ts-ignore"],
                "anti_patterns": ["type assertion to any", "disabling type checks"],
            },
        }

    return rules


@router.post("/extract-functions", response_model=FunctionExtractionResponse)
async def extract_functions(request: FunctionExtractionRequest) -> FunctionExtractionResponse:
    """Extract function signatures from source code."""
    import re

    functions: list[FunctionInfo] = []
    lines = request.code.splitlines()

    if request.language == "python":
        for i, line in enumerate(lines):
            match = re.match(r"^(\s*)(async\s+)?def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.+?))?:\s*$", line)
            if match:
                indent = len(match.group(1))
                is_async = match.group(2) is not None
                name = match.group(3)
                params = [p.strip().split(":")[0].strip() for p in match.group(4).split(",") if p.strip()]
                return_type = match.group(5).strip() if match.group(5) else None
                end = i + 1
                for j in range(i + 1, len(lines)):
                    stripped = lines[j].strip()
                    if stripped and not stripped.startswith("#"):
                        line_indent = len(lines[j]) - len(lines[j].lstrip())
                        if line_indent <= indent and stripped:
                            break
                        end = j + 1
                functions.append(FunctionInfo(name=name, params=params, return_type=return_type, start_line=i + 1, end_line=end, is_async=is_async))

    elif request.language == "go":
        for i, line in enumerate(lines):
            match = re.match(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\((.*?)\)(?:\s+(.+?))?\s*\{", line)
            if match:
                name = match.group(1)
                params = [p.strip().split(" ")[0] for p in match.group(2).split(",") if p.strip()]
                return_type = match.group(3)
                functions.append(FunctionInfo(name=name, params=params, return_type=return_type, start_line=i + 1, end_line=i + 1, is_async=False))

    elif request.language == "java":
        for i, line in enumerate(lines):
            match = re.match(
                r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:async\s+)?(\w+(?:<.*?>)?)\s+(\w+)\s*\((.*?)\)\s*(?:throws\s+\w+(?:,\s*\w+)*)?\s*\{",
                line,
            )
            if match:
                return_type = match.group(1)
                name = match.group(2)
                params = [p.strip().split(" ")[-1] for p in match.group(3).split(",") if p.strip()]
                functions.append(FunctionInfo(name=name, params=params, return_type=return_type, start_line=i + 1, end_line=i + 1, is_async=False))

    elif request.language in ("typescript", "javascript"):
        for i, line in enumerate(lines):
            match = re.match(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\((.*?)\)(?:\s*:\s*(.+?))?\s*\{", line)
            if match:
                name = match.group(1)
                params = [p.strip().split(":")[0].strip() for p in match.group(2).split(",") if p.strip()]
                return_type = match.group(3)
                functions.append(FunctionInfo(name=name, params=params, return_type=return_type, start_line=i + 1, end_line=i + 1, is_async="async" in line))

    return FunctionExtractionResponse(
        functions=functions,
        language=request.language,
        total_lines=len(lines),
    )
