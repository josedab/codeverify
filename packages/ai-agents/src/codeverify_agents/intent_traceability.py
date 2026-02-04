"""Intent-to-Code Traceability - Jira/Linear/GitHub Issues Integration.

This module verifies that code changes match their linked tickets/requirements.
Catches scope creep and unauthorized changes for compliance.

Key insight: AI compares ticket intent vs. code behavior.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class IssueProvider(str, Enum):
    """Supported issue tracking providers."""
    
    JIRA = "jira"
    LINEAR = "linear"
    GITHUB = "github"
    GITLAB = "gitlab"
    AZURE_DEVOPS = "azure_devops"


class TraceabilityStatus(str, Enum):
    """Status of intent-to-code traceability check."""
    
    ALIGNED = "aligned"  # Code matches ticket intent
    PARTIAL = "partial"  # Some alignment, some deviation
    MISALIGNED = "misaligned"  # Code doesn't match ticket
    UNAUTHORIZED = "unauthorized"  # Changes not mentioned in ticket
    NO_TICKET = "no_ticket"  # No ticket linked to PR
    ERROR = "error"  # Failed to check


class ChangeScope(str, Enum):
    """Scope of code changes."""
    
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class IssueDetails:
    """Details of a linked issue/ticket."""
    
    id: str
    provider: IssueProvider
    key: str  # e.g., "PROJ-123"
    title: str
    description: str
    issue_type: str  # "bug", "feature", "task", etc.
    status: str
    assignee: str | None = None
    labels: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    url: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider.value,
            "key": self.key,
            "title": self.title,
            "issue_type": self.issue_type,
            "status": self.status,
            "labels": self.labels,
            "url": self.url,
        }


@dataclass
class ExtractedIntent:
    """Intent extracted from an issue description."""
    
    primary_intent: str
    expected_changes: list[str]
    affected_areas: list[str]  # Files, modules, components
    change_scope: ChangeScope
    keywords: list[str]
    acceptance_criteria: list[str]
    constraints: list[str]  # Things that should NOT change
    confidence: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_intent": self.primary_intent,
            "expected_changes": self.expected_changes,
            "affected_areas": self.affected_areas,
            "change_scope": self.change_scope.value,
            "keywords": self.keywords,
            "acceptance_criteria": self.acceptance_criteria,
            "constraints": self.constraints,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class CodeChangeSummary:
    """Summary of code changes in a PR."""
    
    files_modified: list[str]
    files_added: list[str]
    files_deleted: list[str]
    functions_modified: list[str]
    classes_modified: list[str]
    detected_scope: ChangeScope
    change_keywords: list[str]
    lines_added: int
    lines_deleted: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "files_modified": self.files_modified,
            "files_added": self.files_added,
            "files_deleted": self.files_deleted,
            "functions_modified": self.functions_modified,
            "detected_scope": self.detected_scope.value,
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
        }


@dataclass
class TraceabilityFinding:
    """A finding from traceability analysis."""
    
    type: str  # "scope_creep", "unauthorized_change", "missing_implementation"
    severity: str
    description: str
    location: str | None = None
    expected: str | None = None
    actual: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "description": self.description,
            "location": self.location,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass
class TraceabilityResult:
    """Result of intent-to-code traceability check."""
    
    status: TraceabilityStatus
    issue: IssueDetails | None
    extracted_intent: ExtractedIntent | None
    code_summary: CodeChangeSummary | None
    alignment_score: float  # 0-1
    findings: list[TraceabilityFinding]
    recommendations: list[str]
    checked_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "issue": self.issue.to_dict() if self.issue else None,
            "extracted_intent": self.extracted_intent.to_dict() if self.extracted_intent else None,
            "code_summary": self.code_summary.to_dict() if self.code_summary else None,
            "alignment_score": round(self.alignment_score, 3),
            "findings": [f.to_dict() for f in self.findings],
            "recommendations": self.recommendations,
            "checked_at": self.checked_at.isoformat(),
        }


class IntentExtractor:
    """Extracts intent from issue descriptions."""
    
    # Keywords for different change scopes
    SCOPE_KEYWORDS = {
        ChangeScope.FEATURE: [
            "add", "implement", "create", "new", "introduce", "enable",
            "feature", "functionality", "capability",
        ],
        ChangeScope.BUG_FIX: [
            "fix", "bug", "issue", "error", "crash", "broken",
            "regression", "defect", "problem", "incorrect",
        ],
        ChangeScope.REFACTOR: [
            "refactor", "clean", "improve", "optimize", "restructure",
            "simplify", "modernize", "migrate",
        ],
        ChangeScope.SECURITY: [
            "security", "vulnerability", "cve", "exploit", "patch",
            "auth", "permission", "access", "encrypt",
        ],
        ChangeScope.DOCUMENTATION: [
            "doc", "readme", "comment", "documentation", "guide",
        ],
        ChangeScope.TESTING: [
            "test", "spec", "coverage", "assertion", "mock",
        ],
        ChangeScope.CONFIGURATION: [
            "config", "setting", "environment", "deploy", "ci", "cd",
        ],
    }
    
    def extract_intent(self, issue: IssueDetails) -> ExtractedIntent:
        """Extract intent from issue details."""
        # Combine title and description
        full_text = f"{issue.title}\n{issue.description}"
        text_lower = full_text.lower()
        
        # Determine change scope
        change_scope = self._detect_scope(text_lower, issue.issue_type)
        
        # Extract expected changes
        expected_changes = self._extract_expected_changes(full_text)
        
        # Extract affected areas
        affected_areas = self._extract_affected_areas(full_text)
        
        # Extract keywords
        keywords = self._extract_keywords(text_lower)
        
        # Extract constraints (what should NOT change)
        constraints = self._extract_constraints(full_text)
        
        # Calculate confidence based on specificity
        confidence = self._calculate_confidence(
            expected_changes, affected_areas, issue.acceptance_criteria
        )
        
        return ExtractedIntent(
            primary_intent=issue.title,
            expected_changes=expected_changes,
            affected_areas=affected_areas,
            change_scope=change_scope,
            keywords=keywords,
            acceptance_criteria=issue.acceptance_criteria,
            constraints=constraints,
            confidence=confidence,
        )
    
    def _detect_scope(self, text: str, issue_type: str) -> ChangeScope:
        """Detect the scope of changes from text."""
        # Check issue type first
        type_mapping = {
            "bug": ChangeScope.BUG_FIX,
            "feature": ChangeScope.FEATURE,
            "story": ChangeScope.FEATURE,
            "task": ChangeScope.FEATURE,
            "security": ChangeScope.SECURITY,
            "documentation": ChangeScope.DOCUMENTATION,
        }
        
        if issue_type.lower() in type_mapping:
            return type_mapping[issue_type.lower()]
        
        # Check keywords
        scope_scores: dict[ChangeScope, int] = {}
        for scope, keywords in self.SCOPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scope_scores[scope] = score
        
        if scope_scores:
            return max(scope_scores, key=lambda k: scope_scores[k])
        
        return ChangeScope.UNKNOWN
    
    def _extract_expected_changes(self, text: str) -> list[str]:
        """Extract expected changes from description."""
        changes = []
        
        # Look for bullet points or numbered lists
        patterns = [
            r"[-*]\s+(.+?)(?=\n|$)",  # Bullet points
            r"\d+\.\s+(.+?)(?=\n|$)",  # Numbered lists
            r"(?:should|must|will)\s+(.+?)(?:\.|$)",  # Requirements
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            changes.extend(matches)
        
        # Deduplicate and clean
        changes = list(set(c.strip() for c in changes if len(c.strip()) > 5))
        
        return changes[:10]  # Limit to 10
    
    def _extract_affected_areas(self, text: str) -> list[str]:
        """Extract affected files/modules from description."""
        areas = []
        
        # Look for file paths
        file_pattern = r"[\w/]+\.\w{1,4}"
        files = re.findall(file_pattern, text)
        areas.extend(files)
        
        # Look for module/component names
        component_pattern = r"(?:component|module|service|api|endpoint|function)\s*[:\s]+(\w+)"
        components = re.findall(component_pattern, text, re.IGNORECASE)
        areas.extend(components)
        
        return list(set(areas))[:15]
    
    def _extract_keywords(self, text: str) -> list[str]:
        """Extract relevant keywords from text."""
        # Remove common stop words and find meaningful terms
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "this", "that", "these", "those", "i", "we", "you", "it",
            "they", "them", "their", "our", "your", "its", "and", "or",
            "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "as", "into", "through", "during", "before", "after",
        }
        
        words = re.findall(r"\b[a-z]{3,}\b", text)
        keywords = [w for w in words if w not in stop_words]
        
        # Count frequencies
        freq: dict[str, int] = {}
        for w in keywords:
            freq[w] = freq.get(w, 0) + 1
        
        # Return top keywords
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_keywords[:20]]
    
    def _extract_constraints(self, text: str) -> list[str]:
        """Extract constraints (what should NOT change)."""
        constraints = []
        
        # Look for negations
        patterns = [
            r"(?:do not|don't|should not|shouldn't|must not|mustn't)\s+(.+?)(?:\.|$)",
            r"(?:without|except|excluding)\s+(.+?)(?:\.|$)",
            r"(?:leave|keep)\s+(.+?)\s+(?:unchanged|as is)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            constraints.extend(matches)
        
        return list(set(c.strip() for c in constraints))
    
    def _calculate_confidence(
        self,
        changes: list[str],
        areas: list[str],
        criteria: list[str],
    ) -> float:
        """Calculate confidence in extracted intent."""
        confidence = 0.5  # Base confidence
        
        if changes:
            confidence += min(len(changes) * 0.05, 0.2)
        if areas:
            confidence += min(len(areas) * 0.05, 0.15)
        if criteria:
            confidence += min(len(criteria) * 0.05, 0.15)
        
        return min(confidence, 1.0)


class CodeChangeAnalyzer:
    """Analyzes code changes from PRs."""
    
    def analyze_diff(
        self,
        diff: str,
        files_changed: list[dict[str, Any]],
    ) -> CodeChangeSummary:
        """Analyze a PR diff to extract change summary."""
        files_modified = []
        files_added = []
        files_deleted = []
        functions_modified = []
        classes_modified = []
        lines_added = 0
        lines_deleted = 0
        
        for file_info in files_changed:
            filename = file_info.get("filename", "")
            status = file_info.get("status", "modified")
            
            if status == "added":
                files_added.append(filename)
            elif status == "removed":
                files_deleted.append(filename)
            else:
                files_modified.append(filename)
            
            lines_added += file_info.get("additions", 0)
            lines_deleted += file_info.get("deletions", 0)
        
        # Extract modified functions/classes from diff
        functions_modified = self._extract_modified_functions(diff)
        classes_modified = self._extract_modified_classes(diff)
        
        # Detect change scope from diff content
        detected_scope = self._detect_scope_from_diff(diff, files_modified + files_added)
        
        # Extract keywords from changes
        change_keywords = self._extract_change_keywords(diff)
        
        return CodeChangeSummary(
            files_modified=files_modified,
            files_added=files_added,
            files_deleted=files_deleted,
            functions_modified=functions_modified,
            classes_modified=classes_modified,
            detected_scope=detected_scope,
            change_keywords=change_keywords,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
        )
    
    def _extract_modified_functions(self, diff: str) -> list[str]:
        """Extract function names from diff."""
        functions = []
        
        # Python functions
        py_pattern = r"^\+.*def\s+(\w+)\s*\("
        functions.extend(re.findall(py_pattern, diff, re.MULTILINE))
        
        # JavaScript/TypeScript functions
        js_pattern = r"^\+.*(?:function|const|let|var)\s+(\w+)\s*[=\(]"
        functions.extend(re.findall(js_pattern, diff, re.MULTILINE))
        
        return list(set(functions))
    
    def _extract_modified_classes(self, diff: str) -> list[str]:
        """Extract class names from diff."""
        classes = []
        
        # Python classes
        py_pattern = r"^\+.*class\s+(\w+)"
        classes.extend(re.findall(py_pattern, diff, re.MULTILINE))
        
        # JavaScript/TypeScript classes
        js_pattern = r"^\+.*class\s+(\w+)"
        classes.extend(re.findall(js_pattern, diff, re.MULTILINE))
        
        return list(set(classes))
    
    def _detect_scope_from_diff(self, diff: str, files: list[str]) -> ChangeScope:
        """Detect change scope from diff content."""
        diff_lower = diff.lower()
        
        # Check for security-related changes
        security_patterns = ["password", "secret", "token", "auth", "credential", "encrypt"]
        if any(p in diff_lower for p in security_patterns):
            return ChangeScope.SECURITY
        
        # Check file types
        test_files = [f for f in files if "test" in f.lower() or "spec" in f.lower()]
        if len(test_files) == len(files):
            return ChangeScope.TESTING
        
        doc_files = [f for f in files if f.endswith((".md", ".rst", ".txt"))]
        if len(doc_files) == len(files):
            return ChangeScope.DOCUMENTATION
        
        config_files = [f for f in files if any(
            c in f.lower() for c in ["config", "setting", ".yml", ".yaml", ".json", ".env"]
        )]
        if len(config_files) == len(files):
            return ChangeScope.CONFIGURATION
        
        # Check for bug fix indicators
        if "fix" in diff_lower or "bug" in diff_lower:
            return ChangeScope.BUG_FIX
        
        return ChangeScope.FEATURE
    
    def _extract_change_keywords(self, diff: str) -> list[str]:
        """Extract keywords from diff content."""
        # Only look at added lines
        added_lines = "\n".join(
            line[1:] for line in diff.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        )
        
        words = re.findall(r"\b[a-z]{4,}\b", added_lines.lower())
        
        # Count and return top keywords
        freq: dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        
        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_kw[:15]]


class AlignmentChecker:
    """Checks alignment between intent and code changes."""
    
    def check_alignment(
        self,
        intent: ExtractedIntent,
        changes: CodeChangeSummary,
    ) -> tuple[float, list[TraceabilityFinding]]:
        """Check alignment between intent and code changes."""
        findings: list[TraceabilityFinding] = []
        scores: list[float] = []
        
        # Check scope alignment
        scope_score, scope_findings = self._check_scope_alignment(intent, changes)
        scores.append(scope_score)
        findings.extend(scope_findings)
        
        # Check file alignment
        file_score, file_findings = self._check_file_alignment(intent, changes)
        scores.append(file_score)
        findings.extend(file_findings)
        
        # Check keyword alignment
        keyword_score, keyword_findings = self._check_keyword_alignment(intent, changes)
        scores.append(keyword_score)
        findings.extend(keyword_findings)
        
        # Check for constraint violations
        constraint_findings = self._check_constraints(intent, changes)
        findings.extend(constraint_findings)
        if constraint_findings:
            scores.append(0.5)  # Penalize constraint violations
        
        # Calculate overall alignment score
        alignment_score = sum(scores) / len(scores) if scores else 0.0
        
        return alignment_score, findings
    
    def _check_scope_alignment(
        self,
        intent: ExtractedIntent,
        changes: CodeChangeSummary,
    ) -> tuple[float, list[TraceabilityFinding]]:
        """Check if change scope matches intent."""
        findings = []
        
        if intent.change_scope == changes.detected_scope:
            return 1.0, findings
        
        if intent.change_scope == ChangeScope.UNKNOWN:
            return 0.7, findings  # Can't verify
        
        # Scope mismatch
        findings.append(TraceabilityFinding(
            type="scope_mismatch",
            severity="warning",
            description=f"Ticket suggests {intent.change_scope.value}, but code changes look like {changes.detected_scope.value}",
            expected=intent.change_scope.value,
            actual=changes.detected_scope.value,
        ))
        
        return 0.5, findings
    
    def _check_file_alignment(
        self,
        intent: ExtractedIntent,
        changes: CodeChangeSummary,
    ) -> tuple[float, list[TraceabilityFinding]]:
        """Check if modified files match expected areas."""
        findings = []
        
        if not intent.affected_areas:
            return 0.7, findings  # No specific areas mentioned
        
        all_changed_files = (
            changes.files_modified +
            changes.files_added +
            changes.files_deleted
        )
        
        # Check for unexpected files
        expected_patterns = [a.lower() for a in intent.affected_areas]
        unexpected_files = []
        
        for f in all_changed_files:
            f_lower = f.lower()
            if not any(p in f_lower for p in expected_patterns):
                unexpected_files.append(f)
        
        if unexpected_files:
            findings.append(TraceabilityFinding(
                type="unexpected_files",
                severity="info",
                description=f"Files modified that weren't mentioned in ticket: {', '.join(unexpected_files[:5])}",
                location=unexpected_files[0] if unexpected_files else None,
            ))
        
        # Calculate score based on how many files match
        if all_changed_files:
            matched = len(all_changed_files) - len(unexpected_files)
            score = matched / len(all_changed_files)
        else:
            score = 1.0
        
        return max(score, 0.3), findings
    
    def _check_keyword_alignment(
        self,
        intent: ExtractedIntent,
        changes: CodeChangeSummary,
    ) -> tuple[float, list[TraceabilityFinding]]:
        """Check keyword overlap between intent and changes."""
        findings = []
        
        if not intent.keywords or not changes.change_keywords:
            return 0.7, findings
        
        intent_set = set(intent.keywords)
        change_set = set(changes.change_keywords)
        
        overlap = intent_set & change_set
        
        if not overlap:
            findings.append(TraceabilityFinding(
                type="keyword_mismatch",
                severity="warning",
                description="No keyword overlap between ticket and code changes",
            ))
            return 0.3, findings
        
        # Calculate Jaccard similarity
        score = len(overlap) / len(intent_set | change_set)
        
        return max(score, 0.3), findings
    
    def _check_constraints(
        self,
        intent: ExtractedIntent,
        changes: CodeChangeSummary,
    ) -> list[TraceabilityFinding]:
        """Check if constraints were violated."""
        findings = []
        
        all_changed = " ".join(
            changes.files_modified +
            changes.files_added +
            changes.files_deleted +
            changes.functions_modified +
            changes.classes_modified
        ).lower()
        
        for constraint in intent.constraints:
            constraint_lower = constraint.lower()
            # Check if constrained item was modified
            if any(word in all_changed for word in constraint_lower.split()):
                findings.append(TraceabilityFinding(
                    type="constraint_violation",
                    severity="high",
                    description=f"Constraint violated: '{constraint}' was modified",
                    expected="Should not modify",
                    actual="Was modified",
                ))
        
        return findings


class IssueProviderClient:
    """Base client for issue providers."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
    
    async def get_issue(self, issue_key: str) -> IssueDetails | None:
        """Get issue details from provider."""
        raise NotImplementedError
    
    async def extract_issue_from_pr(self, pr_data: dict[str, Any]) -> str | None:
        """Extract issue key from PR data."""
        title = pr_data.get("title", "")
        body = pr_data.get("body", "")
        branch = pr_data.get("head", {}).get("ref", "")
        
        # Look for common patterns
        patterns = [
            r"([A-Z]+-\d+)",  # JIRA-style: PROJ-123
            r"#(\d+)",  # GitHub-style: #123
            r"(?:fix|fixes|close|closes|resolve|resolves)\s+#?(\d+)",
        ]
        
        for text in [title, body, branch]:
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        return None


class JiraClient(IssueProviderClient):
    """Jira API client."""
    
    async def get_issue(self, issue_key: str) -> IssueDetails | None:
        """Get issue from Jira."""
        # In production, this would use the Jira REST API
        # For now, return a simulated response
        logger.info(f"Would fetch Jira issue: {issue_key}")
        
        return IssueDetails(
            id=issue_key,
            provider=IssueProvider.JIRA,
            key=issue_key,
            title=f"Sample Jira issue {issue_key}",
            description="Sample description",
            issue_type="feature",
            status="In Progress",
            url=f"https://jira.example.com/browse/{issue_key}",
        )


class LinearClient(IssueProviderClient):
    """Linear API client."""
    
    async def get_issue(self, issue_key: str) -> IssueDetails | None:
        """Get issue from Linear."""
        logger.info(f"Would fetch Linear issue: {issue_key}")
        
        return IssueDetails(
            id=issue_key,
            provider=IssueProvider.LINEAR,
            key=issue_key,
            title=f"Sample Linear issue {issue_key}",
            description="Sample description",
            issue_type="feature",
            status="In Progress",
        )


class GitHubIssueClient(IssueProviderClient):
    """GitHub Issues client."""
    
    async def get_issue(self, issue_key: str) -> IssueDetails | None:
        """Get issue from GitHub."""
        logger.info(f"Would fetch GitHub issue: {issue_key}")
        
        return IssueDetails(
            id=issue_key,
            provider=IssueProvider.GITHUB,
            key=f"#{issue_key}",
            title=f"Sample GitHub issue #{issue_key}",
            description="Sample description",
            issue_type="issue",
            status="open",
        )


class IntentTraceabilityEngine:
    """Main engine for intent-to-code traceability."""
    
    def __init__(
        self,
        provider_configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.provider_configs = provider_configs or {}
        
        self.intent_extractor = IntentExtractor()
        self.code_analyzer = CodeChangeAnalyzer()
        self.alignment_checker = AlignmentChecker()
        
        self._providers: dict[IssueProvider, IssueProviderClient] = {}
        self._init_providers()
    
    def _init_providers(self) -> None:
        """Initialize issue provider clients."""
        provider_classes = {
            IssueProvider.JIRA: JiraClient,
            IssueProvider.LINEAR: LinearClient,
            IssueProvider.GITHUB: GitHubIssueClient,
        }
        
        for provider, config in self.provider_configs.items():
            try:
                provider_enum = IssueProvider(provider)
                if provider_enum in provider_classes:
                    self._providers[provider_enum] = provider_classes[provider_enum](config)
            except ValueError:
                logger.warning(f"Unknown provider: {provider}")
    
    async def check_traceability(
        self,
        pr_data: dict[str, Any],
        diff: str,
        files_changed: list[dict[str, Any]],
    ) -> TraceabilityResult:
        """Check traceability between a PR and its linked ticket."""
        findings: list[TraceabilityFinding] = []
        recommendations: list[str] = []
        
        # Try to extract issue key from PR
        issue_key = None
        issue = None
        
        for provider in self._providers.values():
            issue_key = await provider.extract_issue_from_pr(pr_data)
            if issue_key:
                issue = await provider.get_issue(issue_key)
                if issue:
                    break
        
        if not issue_key:
            return TraceabilityResult(
                status=TraceabilityStatus.NO_TICKET,
                issue=None,
                extracted_intent=None,
                code_summary=None,
                alignment_score=0.0,
                findings=[TraceabilityFinding(
                    type="no_ticket",
                    severity="warning",
                    description="No ticket linked to this PR",
                )],
                recommendations=[
                    "Link a ticket to this PR for traceability",
                    "Use format: 'PROJ-123' in title or 'Fixes #123' in body",
                ],
            )
        
        if not issue:
            return TraceabilityResult(
                status=TraceabilityStatus.ERROR,
                issue=None,
                extracted_intent=None,
                code_summary=None,
                alignment_score=0.0,
                findings=[TraceabilityFinding(
                    type="ticket_not_found",
                    severity="warning",
                    description=f"Could not fetch ticket {issue_key}",
                )],
                recommendations=["Verify the ticket exists and is accessible"],
            )
        
        # Extract intent from issue
        extracted_intent = self.intent_extractor.extract_intent(issue)
        
        # Analyze code changes
        code_summary = self.code_analyzer.analyze_diff(diff, files_changed)
        
        # Check alignment
        alignment_score, alignment_findings = self.alignment_checker.check_alignment(
            extracted_intent, code_summary
        )
        findings.extend(alignment_findings)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            alignment_score, findings, issue, code_summary
        )
        
        # Determine status
        if alignment_score >= 0.8:
            status = TraceabilityStatus.ALIGNED
        elif alignment_score >= 0.5:
            status = TraceabilityStatus.PARTIAL
        else:
            status = TraceabilityStatus.MISALIGNED
        
        # Check for unauthorized changes
        if any(f.type == "constraint_violation" for f in findings):
            status = TraceabilityStatus.UNAUTHORIZED
        
        return TraceabilityResult(
            status=status,
            issue=issue,
            extracted_intent=extracted_intent,
            code_summary=code_summary,
            alignment_score=alignment_score,
            findings=findings,
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        score: float,
        findings: list[TraceabilityFinding],
        issue: IssueDetails,
        changes: CodeChangeSummary,
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if score < 0.5:
            recommendations.append(
                "Consider updating the ticket description to better match the changes"
            )
        
        has_scope_mismatch = any(f.type == "scope_mismatch" for f in findings)
        if has_scope_mismatch:
            recommendations.append(
                "PR scope differs from ticket. Verify this is intentional."
            )
        
        has_unexpected_files = any(f.type == "unexpected_files" for f in findings)
        if has_unexpected_files:
            recommendations.append(
                "Some files were modified that aren't mentioned in the ticket. "
                "Consider splitting into separate PRs if unrelated."
            )
        
        has_constraint_violation = any(f.type == "constraint_violation" for f in findings)
        if has_constraint_violation:
            recommendations.append(
                "⚠️ Changes violate constraints specified in the ticket. "
                "This may require approval."
            )
        
        return recommendations


# Convenience function
def create_traceability_engine(
    provider_configs: dict[str, dict[str, Any]] | None = None,
) -> IntentTraceabilityEngine:
    """Create an intent traceability engine."""
    return IntentTraceabilityEngine(provider_configs)
