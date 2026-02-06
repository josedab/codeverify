"""AI hallucination detection API router."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class HallucinationCheckRequest(BaseModel):
    code: str = Field(description="Source code to check for hallucinated APIs")
    language: str = Field(default="python", description="Programming language")
    check_registry: bool = Field(default=False, description="Validate against live package registries")


class HallucinationFinding(BaseModel):
    type: str = Field(description="Type: nonexistent_module, nonexistent_function, wrong_signature, deprecated_api")
    import_path: str
    symbol: str
    line_number: int
    confidence: float = Field(ge=0.0, le=1.0)
    suggestion: str
    evidence: str


class HallucinationCheckResponse(BaseModel):
    findings: list[HallucinationFinding]
    total_imports_checked: int
    hallucinations_found: int
    confidence_score: float = Field(description="Overall confidence that code is hallucination-free (0-1)")


# Known hallucinated APIs that AI models commonly generate
KNOWN_HALLUCINATIONS: dict[str, dict[str, str]] = {
    "python": {
        "requests.get_json": "Use response.json() method instead",
        "os.path.joinpath": "Use os.path.join() or pathlib.Path.joinpath()",
        "json.loads_file": "Use json.load(open(f)) or json.loads(f.read())",
        "json.dump_to_file": "Use json.dump(data, open(f, 'w'))",
        "collections.DefaultDict": "Use collections.defaultdict (lowercase)",
        "typing.StringType": "Use str or typing.AnyStr",
        "pathlib.Path.read": "Use pathlib.Path.read_text() or read_bytes()",
        "os.env": "Use os.environ",
        "os.get_env": "Use os.environ.get() or os.getenv()",
        "subprocess.execute": "Use subprocess.run() or subprocess.call()",
        "datetime.now": "Use datetime.datetime.now()",
        "re.match_all": "Use re.findall() or re.finditer()",
        "sys.args": "Use sys.argv",
        "logging.log": "Use logging.getLogger().info/error/etc",
        "hashlib.hash": "Use hashlib.sha256() or hashlib.md5()",
        "itertools.flatten": "Use itertools.chain.from_iterable()",
        "functools.memorize": "Use functools.lru_cache",
        "asyncio.async": "Use asyncio.create_task() (async was removed in 3.10)",
    },
    "typescript": {
        "Array.flat_map": "Use Array.flatMap() (camelCase)",
        "Promise.allResolved": "Use Promise.allSettled()",
        "Object.deep_copy": "Use structuredClone() or JSON parse/stringify",
        "String.contains": "Use String.includes()",
        "Array.unique": "Use [...new Set(array)]",
        "fs.readFileAsync": "Use fs.promises.readFile()",
        "path.combine": "Use path.join()",
        "console.write": "Use console.log()",
        "process.env.get": "Use process.env.KEY directly",
        "Buffer.from_string": "Use Buffer.from()",
    },
    "go": {
        "fmt.FormatString": "Use fmt.Sprintf()",
        "os.ReadFile": "Use os.ReadFile() (Go 1.16+) or ioutil.ReadFile()",
        "strings.Format": "Use fmt.Sprintf()",
        "errors.Newf": "Use fmt.Errorf()",
        "json.ParseString": "Use json.Unmarshal()",
        "http.Get_json": "Use http.Get() then json.NewDecoder().Decode()",
    },
    "java": {
        "String.format_string": "Use String.format()",
        "List.of_array": "Use Arrays.asList() or List.of()",
        "Map.merge_all": "Use Map.putAll()",
        "System.println": "Use System.out.println()",
        "Files.readString": "Available since Java 11, use Files.readString(Path)",
        "Collections.sort_by": "Use list.sort(Comparator.comparing(...))",
    },
}

# Known valid stdlib modules for quick validation
VALID_STDLIB: dict[str, set[str]] = {
    "python": {
        "os", "sys", "json", "re", "math", "datetime", "pathlib", "typing",
        "collections", "itertools", "functools", "hashlib", "uuid", "io",
        "dataclasses", "enum", "abc", "asyncio", "logging", "subprocess",
        "tempfile", "shutil", "glob", "fnmatch", "textwrap", "string",
        "struct", "copy", "pprint", "traceback", "unittest", "http",
        "urllib", "socket", "ssl", "email", "html", "xml", "csv",
        "sqlite3", "contextlib", "threading", "multiprocessing", "queue",
        "argparse", "configparser", "secrets", "hmac", "base64", "pickle",
        "shelve", "marshal", "time", "calendar", "random", "statistics",
        "decimal", "fractions", "operator", "inspect", "dis", "ast",
        "token", "tokenize", "pdb", "profile", "timeit", "heapq", "bisect",
        "array", "weakref", "types", "importlib", "pkgutil", "warnings",
    },
    "node": {
        "fs", "path", "http", "https", "crypto", "stream", "events",
        "util", "child_process", "os", "buffer", "url", "querystring",
        "net", "dns", "tls", "zlib", "readline", "cluster", "worker_threads",
        "assert", "console", "process", "timers", "v8", "vm", "wasi",
        "perf_hooks", "async_hooks", "string_decoder",
    },
}


def _extract_imports(code: str, language: str) -> list[dict[str, Any]]:
    """Extract import statements from code."""
    import re

    imports = []

    if language == "python":
        for i, line in enumerate(code.splitlines(), 1):
            stripped = line.strip()
            # from X import Y
            match = re.match(r"from\s+([\w.]+)\s+import\s+(.+)", stripped)
            if match:
                module = match.group(1)
                symbols = [s.strip().split(" as ")[0].strip() for s in match.group(2).split(",")]
                for sym in symbols:
                    imports.append({"module": module, "symbol": sym, "line": i})
                continue
            # import X
            match = re.match(r"import\s+([\w.]+)(?:\s+as\s+\w+)?", stripped)
            if match:
                imports.append({"module": match.group(1), "symbol": None, "line": i})

    elif language in ("typescript", "javascript"):
        for i, line in enumerate(code.splitlines(), 1):
            stripped = line.strip()
            match = re.match(r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]", stripped)
            if match:
                symbols = [s.strip().split(" as ")[0].strip() for s in match.group(1).split(",")]
                module = match.group(2)
                for sym in symbols:
                    imports.append({"module": module, "symbol": sym, "line": i})
                continue
            match = re.match(r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]", stripped)
            if match:
                imports.append({"module": match.group(2), "symbol": match.group(1), "line": i})

    elif language == "go":
        for i, line in enumerate(code.splitlines(), 1):
            match = re.match(r'\s*"([^"]+)"', line)
            if match:
                imports.append({"module": match.group(1), "symbol": None, "line": i})

    elif language == "java":
        for i, line in enumerate(code.splitlines(), 1):
            match = re.match(r"import\s+(?:static\s+)?([\w.]+)(?:\.\*)?;", line.strip())
            if match:
                full_path = match.group(1)
                parts = full_path.rsplit(".", 1)
                module = parts[0] if len(parts) > 1 else full_path
                symbol = parts[1] if len(parts) > 1 else None
                imports.append({"module": module, "symbol": symbol, "line": i})

    return imports


def _check_hallucinations(imports: list[dict[str, Any]], language: str) -> list[HallucinationFinding]:
    """Check imports against known hallucination database."""
    findings = []
    known = KNOWN_HALLUCINATIONS.get(language, {})

    for imp in imports:
        module = imp["module"]
        symbol = imp["symbol"]

        # Check full import path
        if symbol:
            full_path = f"{module}.{symbol}"
            if full_path in known:
                findings.append(HallucinationFinding(
                    type="nonexistent_function",
                    import_path=module,
                    symbol=symbol,
                    line_number=imp["line"],
                    confidence=0.95,
                    suggestion=known[full_path],
                    evidence=f"'{full_path}' is a commonly hallucinated API",
                ))

        # Check module-level hallucinations
        if module in known:
            findings.append(HallucinationFinding(
                type="nonexistent_module",
                import_path=module,
                symbol=symbol or "",
                line_number=imp["line"],
                confidence=0.9,
                suggestion=known[module],
                evidence=f"'{module}' is a commonly hallucinated module",
            ))

    return findings


@router.post("/check", response_model=HallucinationCheckResponse)
async def check_hallucinations(request: HallucinationCheckRequest) -> HallucinationCheckResponse:
    """Check code for hallucinated API calls."""
    imports = _extract_imports(request.code, request.language)
    findings = _check_hallucinations(imports, request.language)

    total = len(imports)
    hallucination_count = len(findings)
    confidence = 1.0 - (hallucination_count / max(total, 1))

    return HallucinationCheckResponse(
        findings=findings,
        total_imports_checked=total,
        hallucinations_found=hallucination_count,
        confidence_score=round(max(0.0, confidence), 3),
    )


@router.get("/known-hallucinations")
async def list_known_hallucinations(
    language: str | None = None,
) -> dict[str, Any]:
    """List all known hallucinated APIs in the database."""
    if language:
        if language not in KNOWN_HALLUCINATIONS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No hallucination data for language: {language}",
            )
        return {"language": language, "hallucinations": KNOWN_HALLUCINATIONS[language]}

    return {
        "languages": list(KNOWN_HALLUCINATIONS.keys()),
        "total_entries": sum(len(v) for v in KNOWN_HALLUCINATIONS.values()),
        "hallucinations": KNOWN_HALLUCINATIONS,
    }
