"""Main analysis task - orchestrates the multi-agent verification pipeline."""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import structlog

from codeverify_core.vcs import (
    CheckConclusion,
    CheckRun,
    CheckRunAnnotation,
    CheckStatus,
    GitHubClient,
)
from codeverify_worker.main import app

logger = structlog.get_logger()


@dataclass
class AnalysisResult:
    """Result of a code analysis."""

    analysis_id: str
    status: str
    findings: list[dict[str, Any]]
    stages: list[dict[str, Any]]
    started_at: datetime
    completed_at: datetime
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Finding:
    """A single finding from the analysis."""

    category: str
    severity: str
    title: str
    description: str
    file_path: str
    line_start: int | None
    line_end: int | None
    code_snippet: str | None
    fix_suggestion: str | None
    confidence: float
    verification_type: str  # "formal" | "ai" | "pattern"
    verification_proof: str | None = None


def _create_github_client(installation_id: int) -> GitHubClient:
    """Create a GitHubClient authenticated as a GitHub App installation."""
    return GitHubClient.from_installation(
        installation_id=installation_id,
        app_id=os.environ.get("GITHUB_APP_ID", ""),
        private_key=os.environ.get("GITHUB_APP_PRIVATE_KEY", ""),
    )


class AnalysisPipeline:
    """Multi-stage analysis pipeline orchestrator."""

    def __init__(
        self,
        repo_full_name: str,
        pr_number: int,
        head_sha: str,
        base_sha: str | None,
        installation_id: int | None,
    ) -> None:
        self.repo_full_name = repo_full_name
        self.pr_number = pr_number
        self.head_sha = head_sha
        self.base_sha = base_sha
        self.installation_id = installation_id
        self.stages: list[dict[str, Any]] = []
        self.findings: list[Finding] = []
        self.pr_files: list[dict[str, Any]] = []
        self.pr_diff: str = ""
        self.file_contents: dict[str, str] = {}

    async def run(self) -> AnalysisResult:
        """Execute the full analysis pipeline."""
        started_at = datetime.utcnow()
        analysis_id = f"{self.repo_full_name}#{self.pr_number}@{self.head_sha[:8]}"

        logger.info(
            "Starting analysis pipeline",
            analysis_id=analysis_id,
            repo=self.repo_full_name,
            pr=self.pr_number,
        )

        try:
            # Stage 1: Fetch PR diff and files
            await self._run_stage("fetch", self._fetch_pr_data)

            # Stage 2: Parse code into AST
            await self._run_stage("parse", self._parse_code)

            # Stage 3: Semantic analysis with LLM
            await self._run_stage("semantic", self._semantic_analysis)

            # Stage 4: Formal verification with Z3
            await self._run_stage("verify", self._formal_verification)

            # Stage 5: Security analysis
            await self._run_stage("security", self._security_analysis)

            # Stage 6: Synthesize results
            await self._run_stage("synthesize", self._synthesize_results)

            completed_at = datetime.utcnow()

            # Calculate summary
            summary = self._calculate_summary()

            logger.info(
                "Analysis pipeline completed",
                analysis_id=analysis_id,
                findings_count=len(self.findings),
                duration_ms=(completed_at - started_at).total_seconds() * 1000,
            )

            return AnalysisResult(
                analysis_id=analysis_id,
                status="completed",
                findings=[self._finding_to_dict(f) for f in self.findings],
                stages=self.stages,
                started_at=started_at,
                completed_at=completed_at,
                summary=summary,
            )

        except Exception as e:
            logger.error("Analysis pipeline failed", analysis_id=analysis_id, error=str(e))
            return AnalysisResult(
                analysis_id=analysis_id,
                status="failed",
                findings=[],
                stages=self.stages,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                error=str(e),
            )

    async def _run_stage(self, name: str, func: Any) -> None:
        """Run a single pipeline stage with timing."""
        started = datetime.utcnow()
        stage_result = {"name": name, "started_at": started.isoformat(), "status": "running"}

        try:
            result = await func()
            completed = datetime.utcnow()
            stage_result.update(
                {
                    "status": "completed",
                    "completed_at": completed.isoformat(),
                    "duration_ms": (completed - started).total_seconds() * 1000,
                    "result": result,
                }
            )
            logger.info(f"Stage {name} completed", duration_ms=stage_result["duration_ms"])
        except Exception as e:
            completed = datetime.utcnow()
            stage_result.update(
                {
                    "status": "failed",
                    "completed_at": completed.isoformat(),
                    "duration_ms": (completed - started).total_seconds() * 1000,
                    "error": str(e),
                }
            )
            logger.error(f"Stage {name} failed", error=str(e))
            raise

        self.stages.append(stage_result)

    async def _fetch_pr_data(self) -> dict[str, Any]:
        """Fetch PR diff and changed files from GitHub."""
        logger.info("Fetching PR data", repo=self.repo_full_name, pr=self.pr_number)

        owner, repo = self.repo_full_name.split("/")

        # In production, would use GitHub service
        # For now, return mock data
        return {
            "files_changed": len(self.pr_files),
            "additions": sum(f.get("additions", 0) for f in self.pr_files),
            "deletions": sum(f.get("deletions", 0) for f in self.pr_files),
        }

    async def _parse_code(self) -> dict[str, Any]:
        """Parse changed files into AST representations."""
        logger.info("Parsing code")

        from codeverify_verifier.parsers import PythonParser, TypeScriptParser, GoParser, JavaParser

        # Initialize all available parsers
        parsers = [
            PythonParser(),
            TypeScriptParser(),
            GoParser(),
            JavaParser(),
        ]

        functions_found = 0
        classes_found = 0
        languages_parsed: dict[str, int] = {}

        for path, content in self.file_contents.items():
            for parser in parsers:
                if parser.can_parse(path):
                    parsed = parser.parse(content, path)
                    functions_found += len(parsed.functions)
                    classes_found += len(parsed.classes)
                    
                    # Track languages
                    lang = parser.language
                    languages_parsed[lang] = languages_parsed.get(lang, 0) + 1
                    break

        return {
            "files_parsed": len(self.file_contents),
            "functions_found": functions_found,
            "classes_found": classes_found,
            "languages": languages_parsed,
        }

    async def _semantic_analysis(self) -> dict[str, Any]:
        """Run semantic analysis using LLM to understand code intent."""
        logger.info("Running semantic analysis")

        # Import and use semantic agent
        try:
            from codeverify_agents.semantic import SemanticAgent

            agent = SemanticAgent()

            issues_found = 0
            for path, content in self.file_contents.items():
                if len(content) > 50000:  # Skip very large files
                    continue

                result = await agent.analyze(
                    code=content,
                    context={
                        "file_path": path,
                        "language": self._detect_language(path),
                        "diff": self.pr_diff,
                    },
                )

                if result.success and result.data:
                    # Extract concerns as findings
                    for func in result.data.get("functions", []):
                        for concern in func.get("concerns", []):
                            self.findings.append(
                                Finding(
                                    category="logic_error",
                                    severity="medium",
                                    title=f"Potential issue in {func.get('name', 'function')}",
                                    description=concern,
                                    file_path=path,
                                    line_start=None,
                                    line_end=None,
                                    code_snippet=None,
                                    fix_suggestion=None,
                                    confidence=0.7,
                                    verification_type="ai",
                                )
                            )
                            issues_found += 1

            return {"issues_found": issues_found}

        except ImportError:
            logger.warning("Semantic agent not available, skipping")
            return {"issues_found": 0, "skipped": True}

    async def _formal_verification(self) -> dict[str, Any]:
        """Run formal verification using Z3 SMT solver."""
        logger.info("Running formal verification")

        from codeverify_verifier import Z3Verifier
        from codeverify_verifier.parsers import PythonParser

        verifier = Z3Verifier(timeout_ms=30000)
        python_parser = PythonParser()

        proofs_attempted = 0
        proofs_succeeded = 0
        issues_found = 0

        for path, content in self.file_contents.items():
            if not python_parser.can_parse(path):
                continue

            parsed = python_parser.parse(content, path)

            for func in parsed.functions:
                # Check for potential integer overflow
                if any(call in func.calls for call in ["*", "+"]):
                    proofs_attempted += 1
                    result = verifier.check_integer_overflow(
                        var_name=func.name,
                        operation="mul",
                        operand1_range=(0, 1000000),
                        operand2_range=(0, 1000000),
                    )

                    if result.get("satisfiable"):
                        self.findings.append(
                            Finding(
                                category="overflow",
                                severity="high",
                                title=f"Potential integer overflow in {func.name}",
                                description=f"The function {func.name} may experience integer overflow. {result.get('message', '')}",
                                file_path=path,
                                line_start=func.line_start,
                                line_end=func.line_end,
                                code_snippet=func.body[:500],
                                fix_suggestion="Consider using checked arithmetic or larger integer types.",
                                confidence=0.95,
                                verification_type="formal",
                                verification_proof=str(result.get("counterexample")),
                            )
                        )
                        issues_found += 1
                    else:
                        proofs_succeeded += 1

                # Check for array bounds
                if "[" in func.body:
                    proofs_attempted += 1
                    result = verifier.check_array_bounds(
                        index_var="i",
                        index_range=None,  # Unknown range
                        array_length=100,  # Assumed
                    )

                    if result.get("satisfiable"):
                        self.findings.append(
                            Finding(
                                category="bounds",
                                severity="medium",
                                title=f"Potential array bounds issue in {func.name}",
                                description="Array access may go out of bounds.",
                                file_path=path,
                                line_start=func.line_start,
                                line_end=func.line_end,
                                code_snippet=None,
                                fix_suggestion="Add bounds checking before array access.",
                                confidence=0.8,
                                verification_type="formal",
                            )
                        )
                        issues_found += 1

        return {
            "proofs_attempted": proofs_attempted,
            "proofs_succeeded": proofs_succeeded,
            "issues_found": issues_found,
        }

    async def _security_analysis(self) -> dict[str, Any]:
        """Run security-focused analysis."""
        logger.info("Running security analysis")

        try:
            from codeverify_agents.security import SecurityAgent

            agent = SecurityAgent()
            vulnerabilities_found = 0

            for path, content in self.file_contents.items():
                if len(content) > 50000:
                    continue

                # Quick pattern-based secret scan
                secrets = await agent.scan_for_secrets(content)
                for secret in secrets:
                    self.findings.append(
                        Finding(
                            category="security",
                            severity=secret.get("severity", "high"),
                            title=f"Potential secret detected: {secret.get('type')}",
                            description=f"A potential {secret.get('type')} was detected in the code.",
                            file_path=path,
                            line_start=secret.get("line"),
                            line_end=secret.get("line"),
                            code_snippet=None,
                            fix_suggestion="Remove the secret and use environment variables instead.",
                            confidence=0.9,
                            verification_type="pattern",
                        )
                    )
                    vulnerabilities_found += 1

                # LLM-based security analysis
                result = await agent.analyze(
                    code=content,
                    context={
                        "file_path": path,
                        "language": self._detect_language(path),
                    },
                )

                if result.success and result.data:
                    for vuln in result.data.get("vulnerabilities", []):
                        self.findings.append(
                            Finding(
                                category="security",
                                severity=vuln.get("severity", "medium"),
                                title=vuln.get("title", "Security issue"),
                                description=vuln.get("description", ""),
                                file_path=path,
                                line_start=vuln.get("location", {}).get("line"),
                                line_end=None,
                                code_snippet=vuln.get("code_snippet"),
                                fix_suggestion=vuln.get("fix_code"),
                                confidence=vuln.get("confidence", 0.8),
                                verification_type="ai",
                            )
                        )
                        vulnerabilities_found += 1

            return {"vulnerabilities_found": vulnerabilities_found}

        except ImportError:
            logger.warning("Security agent not available, skipping")
            return {"vulnerabilities_found": 0, "skipped": True}

    async def _synthesize_results(self) -> dict[str, Any]:
        """Synthesize all results and generate fix suggestions."""
        logger.info("Synthesizing results")

        # Deduplicate findings
        unique_findings: list[Finding] = []
        seen = set()

        for finding in self.findings:
            key = (finding.file_path, finding.line_start, finding.title)
            if key not in seen:
                seen.add(key)
                unique_findings.append(finding)

        self.findings = unique_findings

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        self.findings.sort(key=lambda f: severity_order.get(f.severity, 5))

        return {"total_findings": len(self.findings)}

    def _calculate_summary(self) -> dict[str, Any]:
        """Calculate analysis summary."""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}

        for finding in self.findings:
            if finding.severity in severity_counts:
                severity_counts[finding.severity] += 1

        total = len(self.findings)
        passed = severity_counts["critical"] == 0 and severity_counts["high"] == 0

        return {
            "total_issues": total,
            "critical": severity_counts["critical"],
            "high": severity_counts["high"],
            "medium": severity_counts["medium"],
            "low": severity_counts["low"],
            "pass": passed,
        }

    def _detect_language(self, path: str) -> str:
        """Detect programming language from file path."""
        if path.endswith(".py"):
            return "python"
        elif path.endswith((".ts", ".tsx")):
            return "typescript"
        elif path.endswith((".js", ".jsx")):
            return "javascript"
        elif path.endswith(".go"):
            return "go"
        elif path.endswith(".java"):
            return "java"
        else:
            return "unknown"

    def _finding_to_dict(self, finding: Finding) -> dict[str, Any]:
        """Convert Finding dataclass to dictionary."""
        return {
            "category": finding.category,
            "severity": finding.severity,
            "title": finding.title,
            "description": finding.description,
            "file_path": finding.file_path,
            "line_start": finding.line_start,
            "line_end": finding.line_end,
            "code_snippet": finding.code_snippet,
            "fix_suggestion": finding.fix_suggestion,
            "confidence": finding.confidence,
            "verification_type": finding.verification_type,
            "verification_proof": finding.verification_proof,
        }


def format_github_comment(result: AnalysisResult) -> str:
    """Format analysis result as GitHub PR comment."""
    summary = result.summary
    findings = result.findings

    lines = [
        "## 🔍 CodeVerify Analysis",
        "",
        "### Summary",
        "",
        "| Total | Critical | High | Medium | Low |",
        "|:---:|:---:|:---:|:---:|:---:|",
        f"| {summary.get('total_issues', 0)} | {summary.get('critical', 0)} | "
        f"{summary.get('high', 0)} | {summary.get('medium', 0)} | {summary.get('low', 0)} |",
        "",
    ]

    if summary.get("pass", True):
        lines.append("✅ **Status: Passed**")
    else:
        lines.append("❌ **Status: Issues Found**")

    if findings:
        lines.extend(["", "### Findings", ""])

        severity_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🔵",
        }

        for finding in findings[:10]:
            emoji = severity_emoji.get(finding.get("severity", "low"), "⚪")
            title = finding.get("title", "Issue")
            file_path = finding.get("file_path", "")
            line = finding.get("line_start", "")

            lines.extend([
                f"<details>",
                f"<summary>{emoji} <b>{title}</b> ({file_path}:{line})</summary>",
                "",
                finding.get("description", ""),
                "",
            ])

            if fix := finding.get("fix_suggestion"):
                lines.extend([
                    "**Suggested fix:**",
                    f"```",
                    fix,
                    "```",
                    "",
                ])

            lines.extend(["</details>", ""])

        if len(findings) > 10:
            lines.append(f"*...and {len(findings) - 10} more findings*")

    lines.extend([
        "",
        "---",
        "*Powered by [CodeVerify](https://codeverify.dev)*",
    ])

    return "\n".join(lines)


def format_check_annotations(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format findings as GitHub Check Run annotations for inline display.
    
    These appear directly in the PR diff view as inline warnings/errors.
    """
    annotations = []
    
    level_map = {
        "critical": "failure",
        "high": "failure", 
        "medium": "warning",
        "low": "notice",
        "info": "notice",
    }
    
    for finding in findings[:50]:  # GitHub limit is 50 annotations per update
        annotation = {
            "path": finding.get("file_path", ""),
            "start_line": finding.get("line_start", 1),
            "end_line": finding.get("line_end") or finding.get("line_start", 1),
            "annotation_level": level_map.get(finding.get("severity", "low"), "notice"),
            "title": finding.get("title", "Issue found"),
            "message": finding.get("description", ""),
        }
        
        # Add raw details for GitHub's UI
        verification_type = finding.get("verification_type", "ai")
        confidence = finding.get("confidence", 0)
        
        raw_details = f"Confidence: {int(confidence * 100)}%\n"
        raw_details += f"Verification: {verification_type}\n"
        
        if fix := finding.get("fix_suggestion"):
            raw_details += f"\nSuggested fix:\n{fix}"
        
        annotation["raw_details"] = raw_details
        annotations.append(annotation)
    
    return annotations


async def post_results_to_github(
    result: AnalysisResult,
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
    installation_id: int,
) -> dict[str, Any]:
    """Post analysis results to GitHub with full integration.
    
    Creates:
    1. Check run with inline annotations
    2. PR comment with summary
    3. Review with suggested changes (one-click fixes)
    """
    owner, repo = repo_full_name.split("/")
    
    # Create GitHub client using shared VCS module
    github = _create_github_client(installation_id)
    
    results_posted = {
        "check_run_id": None,
        "comment_id": None,
        "review_id": None,
        "annotations_count": 0,
    }
    
    try:
        # 1. Create check run with in_progress status
        check_run_data = CheckRun(
            name="CodeVerify",
            status=CheckStatus.IN_PROGRESS,
        )
        created_check_run = await github.create_check_run(
            repo_full_name=repo_full_name,
            head_sha=head_sha,
            check_run=check_run_data,
        )
        results_posted["check_run_id"] = created_check_run.id
        
        # Format annotations for inline display
        raw_annotations = format_check_annotations(result.findings)
        results_posted["annotations_count"] = len(raw_annotations)
        
        # Convert to CheckRunAnnotation objects
        annotations = [
            CheckRunAnnotation(
                path=a["path"],
                start_line=a["start_line"],
                end_line=a["end_line"],
                annotation_level=a["annotation_level"],
                message=a["message"],
                title=a.get("title"),
            )
            for a in raw_annotations[:50]  # GitHub limits to 50 annotations
        ]
        
        # Determine conclusion
        passed = result.summary.get("pass", True)
        conclusion = CheckConclusion.SUCCESS if passed else CheckConclusion.FAILURE
        
        # Update check run with results
        summary_md = f"""## Analysis Complete

| Metric | Value |
|--------|-------|
| Total Issues | {result.summary.get('total_issues', 0)} |
| Critical | {result.summary.get('critical', 0)} |
| High | {result.summary.get('high', 0)} |
| Medium | {result.summary.get('medium', 0)} |
| Low | {result.summary.get('low', 0)} |
| Status | {'✅ Passed' if passed else '❌ Failed'} |
"""
        
        update_check_run = CheckRun(
            name="CodeVerify",
            status=CheckStatus.COMPLETED,
            conclusion=conclusion,
            title=f"CodeVerify: {result.summary.get('total_issues', 0)} issues found",
            summary=summary_md,
            annotations=annotations,
            completed_at=datetime.utcnow(),
        )
        
        await github.update_check_run(
            repo_full_name=repo_full_name,
            check_run_id=created_check_run.id,
            check_run=update_check_run,
        )
        
        # 2. Post PR comment with detailed findings
        comment_body = format_github_comment(result)
        comment = await github.create_pull_request_comment(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            body=comment_body,
        )
        results_posted["comment_id"] = comment.id
        
        # 3. Create review with suggested changes (one-click fixes)
        findings_with_fixes = [f for f in result.findings if f.get("fix_suggestion")]
        
        if findings_with_fixes:
            # Build review comments with suggestions
            review_comments = []
            for finding in findings_with_fixes[:10]:  # Limit suggestions
                review_comments.append({
                    "path": finding.get("file_path", ""),
                    "line": finding.get("line_end") or finding.get("line_start", 1),
                    "body": _format_suggestion_comment(finding),
                })
            
            review_body = f"## 🔧 CodeVerify Suggested Fixes\n\nFound {len(findings_with_fixes)} issues with suggested fixes. Click 'Apply suggestion' to fix with one click."
            
            # Note: In production, would use create_pr_review from GitHub client
            # For now, this shows the structure
            logger.info(
                "Would create review with suggestions",
                suggestions_count=len(review_comments),
            )
        
        logger.info(
            "Posted results to GitHub",
            check_run_id=results_posted["check_run_id"],
            annotations=results_posted["annotations_count"],
        )
        
    except Exception as e:
        logger.error(f"Failed to post results to GitHub: {e}")
        raise
    
    return results_posted


def _format_suggestion_comment(finding: dict[str, Any]) -> str:
    """Format a finding as a GitHub suggestion comment."""
    severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
    emoji = severity_emoji.get(finding.get("severity", "low"), "⚪")
    
    parts = [
        f"{emoji} **{finding.get('title', 'Issue')}**",
        "",
        finding.get("description", ""),
        "",
        f"*Confidence: {int(finding.get('confidence', 0) * 100)}%*",
        "",
        "**Apply the suggested fix:**",
        "",
        "```suggestion",
        finding.get("fix_suggestion", ""),
        "```",
    ]
    
    return "\n".join(parts)


@app.task(bind=True, name="analyze_pr")
def analyze_pr(
    self: Any,
    repo_full_name: str,
    repo_id: int,
    pr_number: int,
    pr_title: str | None,
    head_sha: str,
    base_sha: str | None,
    installation_id: int | None,
) -> dict[str, Any]:
    """
    Celery task to analyze a pull request.

    This is the main entry point for the analysis worker.
    """
    logger.info(
        "Starting PR analysis task",
        repo=repo_full_name,
        pr=pr_number,
        sha=head_sha[:8],
    )

    pipeline = AnalysisPipeline(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        head_sha=head_sha,
        base_sha=base_sha,
        installation_id=installation_id,
    )

    # Run async pipeline in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(pipeline.run())

        # Post results to GitHub if we have an installation ID
        github_results = None
        if installation_id:
            try:
                github_results = loop.run_until_complete(
                    post_results_to_github(
                        result=result,
                        repo_full_name=repo_full_name,
                        pr_number=pr_number,
                        head_sha=head_sha,
                        installation_id=installation_id,
                    )
                )
                logger.info(
                    "Posted results to GitHub",
                    check_run_id=github_results.get("check_run_id"),
                    comment_id=github_results.get("comment_id"),
                    annotations_count=github_results.get("annotations_count", 0),
                )
            except Exception as e:
                logger.error(f"Failed to post results to GitHub: {e}")
                # Don't fail the task if GitHub posting fails
                github_results = {"error": str(e)}

        # Store results in database via API
        db_results = None
        try:
            db_results = loop.run_until_complete(
                store_analysis_results(
                    result=result,
                    repo_id=repo_id,
                    repo_full_name=repo_full_name,
                    pr_number=pr_number,
                    pr_title=pr_title,
                    head_sha=head_sha,
                    base_sha=base_sha,
                )
            )
            logger.info(
                "Stored results in database",
                analysis_id=db_results.get("analysis_id"),
                findings_stored=db_results.get("findings_count", 0),
            )
        except Exception as e:
            logger.error(f"Failed to store results in database: {e}")
            db_results = {"error": str(e)}

    finally:
        loop.close()

    return {
        "analysis_id": result.analysis_id,
        "status": result.status,
        "findings_count": len(result.findings),
        "stages_count": len(result.stages),
        "summary": result.summary,
        "error": result.error,
        "github": github_results,
        "database": db_results,
    }


async def store_analysis_results(
    result: AnalysisResult,
    repo_id: int,
    repo_full_name: str,
    pr_number: int,
    pr_title: str | None,
    head_sha: str,
    base_sha: str | None,
) -> dict[str, Any]:
    """Store analysis results in the database via API call.
    
    This calls the internal API to persist results, allowing the worker
    to remain stateless and not require direct database access.
    """
    import os
    
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    internal_api_key = os.environ.get("INTERNAL_API_KEY", "")
    
    # Prepare analysis data
    analysis_data = {
        "repo_id": repo_id,
        "repo_full_name": repo_full_name,
        "pr_number": pr_number,
        "pr_title": pr_title,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "status": result.status,
        "started_at": result.started_at.isoformat(),
        "completed_at": result.completed_at.isoformat(),
        "error_message": result.error,
        "findings": result.findings,
        "stages": result.stages,
        "summary": result.summary,
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{api_url}/internal/analyses",
            json=analysis_data,
            headers={
                "Authorization": f"Bearer {internal_api_key}",
                "Content-Type": "application/json",
            },
        )
        
        if response.status_code == 201:
            data = response.json()
            return {
                "analysis_id": data.get("id"),
                "findings_count": len(result.findings),
                "stored": True,
            }
        else:
            logger.warning(
                "Failed to store analysis via API",
                status_code=response.status_code,
                response=response.text[:500],
            )
            # Fall back to direct storage notification
            return {
                "analysis_id": result.analysis_id,
                "findings_count": len(result.findings),
                "stored": False,
                "api_error": response.text[:200],
            }
