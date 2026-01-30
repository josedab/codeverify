"""Local code analyzer for CLI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


@dataclass
class AnalysisResults:
    """Results from local analysis."""
    
    files_analyzed: int = 0
    functions_found: int = 0
    classes_found: int = 0
    findings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "files_analyzed": self.files_analyzed,
            "functions_found": self.functions_found,
            "classes_found": self.classes_found,
            "findings": self.findings,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "summary": {
                "total": len(self.findings),
                "critical": sum(1 for f in self.findings if f.get("severity") == "critical"),
                "high": sum(1 for f in self.findings if f.get("severity") == "high"),
                "medium": sum(1 for f in self.findings if f.get("severity") == "medium"),
                "low": sum(1 for f in self.findings if f.get("severity") == "low"),
            }
        }


class LocalAnalyzer:
    """Local code analyzer that runs without API connection."""
    
    def __init__(self, config: Any, verbose: bool = False) -> None:
        self.config = config
        self.verbose = verbose
        self._parsers: list[Any] = []
        self._init_parsers()
    
    def _init_parsers(self) -> None:
        """Initialize language parsers."""
        try:
            from codeverify_verifier.parsers import (
                PythonParser, TypeScriptParser, GoParser, JavaParser
            )
            self._parsers = [
                PythonParser(),
                TypeScriptParser(),
                GoParser(),
                JavaParser(),
            ]
        except ImportError:
            if self.verbose:
                console.print("[yellow]Warning: Parsers not available[/yellow]")
    
    async def analyze_path(self, path: Path) -> AnalysisResults:
        """Analyze all files in a path."""
        results = AnalysisResults(started_at=datetime.utcnow())
        
        # Collect files
        files = self._collect_files(path)
        
        if self.verbose:
            console.print(f"[dim]Found {len(files)} files to analyze[/dim]")
        
        # Analyze each file
        for file_path in files:
            try:
                file_results = await self._analyze_file(file_path)
                results.files_analyzed += 1
                results.functions_found += file_results.get("functions", 0)
                results.classes_found += file_results.get("classes", 0)
                results.findings.extend(file_results.get("findings", []))
            except Exception as e:
                results.errors.append(f"{file_path}: {str(e)}")
        
        results.completed_at = datetime.utcnow()
        if results.started_at:
            results.duration_ms = (results.completed_at - results.started_at).total_seconds() * 1000
        
        return results
    
    async def analyze_files(self, files: list[Path]) -> AnalysisResults:
        """Analyze specific files."""
        results = AnalysisResults(started_at=datetime.utcnow())
        
        for file_path in files:
            if file_path.exists():
                try:
                    file_results = await self._analyze_file(file_path)
                    results.files_analyzed += 1
                    results.functions_found += file_results.get("functions", 0)
                    results.classes_found += file_results.get("classes", 0)
                    results.findings.extend(file_results.get("findings", []))
                except Exception as e:
                    results.errors.append(f"{file_path}: {str(e)}")
        
        results.completed_at = datetime.utcnow()
        if results.started_at:
            results.duration_ms = (results.completed_at - results.started_at).total_seconds() * 1000
        
        return results
    
    def _collect_files(self, path: Path) -> list[Path]:
        """Collect files matching configuration patterns."""
        files: list[Path] = []
        
        # Get extensions from config
        extensions = self._get_extensions()
        
        # Get include/exclude patterns
        include_patterns = getattr(self.config, "include", ["**/*"])
        exclude_patterns = getattr(self.config, "exclude", [])
        
        if path.is_file():
            return [path]
        
        for ext in extensions:
            for pattern in include_patterns:
                for file_path in path.glob(pattern):
                    if file_path.is_file() and file_path.suffix in extensions:
                        # Check exclusions
                        if not self._is_excluded(file_path, exclude_patterns):
                            files.append(file_path)
        
        # Also do simple recursive search
        for ext in extensions:
            for file_path in path.rglob(f"*{ext}"):
                if file_path not in files and not self._is_excluded(file_path, exclude_patterns):
                    files.append(file_path)
        
        return list(set(files))
    
    def _get_extensions(self) -> set[str]:
        """Get file extensions based on configured languages."""
        lang_extensions = {
            "python": {".py"},
            "typescript": {".ts", ".tsx"},
            "javascript": {".js", ".jsx"},
            "go": {".go"},
            "java": {".java"},
        }
        
        extensions: set[str] = set()
        languages = getattr(self.config, "languages", ["python", "typescript"])
        
        for lang in languages:
            extensions.update(lang_extensions.get(lang, set()))
        
        return extensions
    
    def _is_excluded(self, path: Path, patterns: list[str]) -> bool:
        """Check if path matches exclusion patterns."""
        path_str = str(path)
        
        default_excludes = [
            "node_modules", "__pycache__", ".git", "venv", 
            "dist", "build", ".tox", ".pytest_cache"
        ]
        
        for exclude in default_excludes:
            if exclude in path_str:
                return True
        
        for pattern in patterns:
            # Simple pattern matching
            if "*" in pattern:
                import fnmatch
                if fnmatch.fnmatch(path_str, pattern):
                    return True
            elif pattern in path_str:
                return True
        
        return False
    
    async def _analyze_file(self, file_path: Path) -> dict[str, Any]:
        """Analyze a single file."""
        results: dict[str, Any] = {
            "functions": 0,
            "classes": 0,
            "findings": [],
        }
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return results
        
        # Skip very large files
        if len(content) > 500000:
            return results
        
        # Parse with appropriate parser
        parser = self._get_parser(file_path)
        if parser:
            try:
                parsed = parser.parse(content, str(file_path))
                results["functions"] = len(parsed.functions)
                results["classes"] = len(parsed.classes)
                
                # Run verification checks
                findings = await self._run_checks(parsed, content, file_path)
                results["findings"] = findings
            except Exception as e:
                if self.verbose:
                    console.print(f"[yellow]Parse error for {file_path}: {e}[/yellow]")
        else:
            # Pattern-based analysis for unsupported files
            findings = self._pattern_analysis(content, file_path)
            results["findings"] = findings
        
        return results
    
    def _get_parser(self, file_path: Path) -> Any:
        """Get appropriate parser for file."""
        for parser in self._parsers:
            if parser.can_parse(str(file_path)):
                return parser
        return None
    
    async def _run_checks(self, parsed: Any, content: str, file_path: Path) -> list[dict[str, Any]]:
        """Run verification checks on parsed code."""
        findings: list[dict[str, Any]] = []
        
        # Check each function
        for func in parsed.functions:
            # Check complexity
            if func.complexity > 10:
                findings.append({
                    "category": "maintainability",
                    "severity": "medium" if func.complexity < 15 else "high",
                    "title": "High Cyclomatic Complexity",
                    "description": f"Function '{func.name}' has complexity {func.complexity}. Consider refactoring.",
                    "file_path": str(file_path),
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "confidence": 1.0,
                    "verification_type": "static",
                })
            
            # Check function length
            func_length = func.line_end - func.line_start
            if func_length > 50:
                findings.append({
                    "category": "maintainability",
                    "severity": "low",
                    "title": "Long Function",
                    "description": f"Function '{func.name}' is {func_length} lines. Consider breaking it up.",
                    "file_path": str(file_path),
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "confidence": 0.9,
                    "verification_type": "static",
                })
        
        # Run Z3 verification if available
        try:
            from codeverify_verifier.z3_verifier import Z3Verifier
            verifier = Z3Verifier()
            
            for func in parsed.functions:
                # Check for potential issues in function body
                for condition in func.conditions:
                    # Integer overflow checks
                    if any(op in condition for op in ["*", "+", "-"]):
                        # Simplified check - would be more sophisticated in production
                        if "int" in str(func.parameters):
                            findings.append({
                                "category": "verification",
                                "severity": "medium",
                                "title": "Potential Integer Overflow",
                                "description": f"Arithmetic operation in condition may overflow",
                                "file_path": str(file_path),
                                "line_start": func.line_start,
                                "confidence": 0.7,
                                "verification_type": "z3",
                            })
        except ImportError:
            pass
        
        # Pattern-based security checks
        security_findings = self._security_patterns(content, file_path)
        findings.extend(security_findings)
        
        return findings
    
    def _pattern_analysis(self, content: str, file_path: Path) -> list[dict[str, Any]]:
        """Run pattern-based analysis."""
        findings: list[dict[str, Any]] = []
        findings.extend(self._security_patterns(content, file_path))
        return findings
    
    def _security_patterns(self, content: str, file_path: Path) -> list[dict[str, Any]]:
        """Check for security patterns."""
        import re
        findings: list[dict[str, Any]] = []
        lines = content.split("\n")
        
        patterns = [
            # Hardcoded secrets
            (r"(?i)(password|secret|api_key|apikey|token)\s*=\s*['\"][^'\"]+['\"]", 
             "Potential Hardcoded Secret", "security", "high"),
            # SQL injection
            (r"(?i)(execute|query)\s*\([^)]*['\"].*%s.*['\"]", 
             "Potential SQL Injection", "security", "high"),
            # Eval usage
            (r"\beval\s*\(", 
             "Dangerous eval() Usage", "security", "critical"),
            # Shell injection
            (r"(?i)(subprocess|os\.system|shell)\s*\([^)]*\+", 
             "Potential Command Injection", "security", "high"),
            # Debug code
            (r"(?i)(console\.log|print|debugger)", 
             "Debug Statement", "maintainability", "low"),
            # TODO/FIXME
            (r"(?i)(TODO|FIXME|HACK|XXX):", 
             "TODO Comment", "maintainability", "info"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, title, category, severity in patterns:
                if re.search(pattern, line):
                    # Skip if it's a comment (basic check)
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith("//"):
                        if severity in ("critical", "high"):
                            continue
                    
                    findings.append({
                        "category": category,
                        "severity": severity,
                        "title": title,
                        "description": f"Pattern detected: {line.strip()[:80]}",
                        "file_path": str(file_path),
                        "line_start": i,
                        "line_end": i,
                        "confidence": 0.8,
                        "verification_type": "pattern",
                    })
        
        return findings
