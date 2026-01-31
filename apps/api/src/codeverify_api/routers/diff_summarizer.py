"""Diff Summarizer API endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class SummarizeDiffRequest(BaseModel):
    """Request to summarize a diff."""

    diff: str = Field(..., description="Git diff content")
    pr_title: str | None = Field(default=None, description="Pull request title")
    base_branch: str | None = Field(default=None, description="Target branch")
    author: str | None = Field(default=None, description="Author of the changes")
    commit_messages: list[str] = Field(default_factory=list, description="Commit messages")
    file_list: list[str] = Field(default_factory=list, description="List of changed files")


class RiskAssessment(BaseModel):
    """Risk assessment for changes."""

    level: str
    reasoning: str


class SummarizeDiffResponse(BaseModel):
    """Response from diff summarization."""

    summary: str
    behavioral_changes: list[str]
    risk_assessment: RiskAssessment
    pr_description: str
    changelog_entry: str
    affected_components: list[str]
    breaking_changes: list[str]
    review_suggestions: list[str] = Field(default_factory=list)


class GeneratePRDescriptionRequest(BaseModel):
    """Request to generate PR description."""

    diff: str = Field(..., description="Git diff content")
    title: str | None = Field(default=None, description="PR title")
    template: str | None = Field(default=None, description="PR description template")


class GeneratePRDescriptionResponse(BaseModel):
    """Response with generated PR description."""

    title_suggestion: str
    description: str
    summary: str
    changelog_entry: str
    labels_suggestion: list[str]
    reviewers_hint: list[str]


class GenerateChangelogRequest(BaseModel):
    """Request to generate changelog entries."""

    diffs: list[dict[str, str]] = Field(
        ...,
        description="List of {commit_sha, diff} objects",
    )


class ChangelogEntry(BaseModel):
    """A single changelog entry."""

    commit: str
    entry: str
    category: str


class GenerateChangelogResponse(BaseModel):
    """Response with changelog entries."""

    entries: list[ChangelogEntry]
    formatted_changelog: str


@router.post("/summarize", response_model=SummarizeDiffResponse)
async def summarize_diff(request: SummarizeDiffRequest) -> SummarizeDiffResponse:
    """
    Generate a comprehensive summary of a git diff.

    This endpoint analyzes the diff and produces:
    - A concise summary
    - List of behavioral changes
    - Risk assessment
    - Formatted PR description
    - Changelog entry
    - Affected components
    - Breaking changes
    """
    from codeverify_agents.diff_summarizer import DiffSummarizerAgent

    agent = DiffSummarizerAgent()

    context = {
        "pr_title": request.pr_title,
        "base_branch": request.base_branch,
        "author": request.author,
        "commit_messages": request.commit_messages,
        "file_list": request.file_list,
    }

    result = await agent.summarize_diff(request.diff, context)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {result.error}",
        )

    data = result.data

    # Handle risk assessment
    risk_data = data.get("risk_assessment", {})
    if isinstance(risk_data, str):
        risk_assessment = RiskAssessment(level="unknown", reasoning=risk_data)
    else:
        risk_assessment = RiskAssessment(
            level=risk_data.get("level", "unknown"),
            reasoning=risk_data.get("reasoning", ""),
        )

    return SummarizeDiffResponse(
        summary=data.get("summary", ""),
        behavioral_changes=data.get("behavioral_changes", []),
        risk_assessment=risk_assessment,
        pr_description=data.get("pr_description", ""),
        changelog_entry=data.get("changelog_entry", ""),
        affected_components=data.get("affected_components", []),
        breaking_changes=data.get("breaking_changes", []),
        review_suggestions=data.get("review_suggestions", []),
    )


@router.post("/pr-description", response_model=GeneratePRDescriptionResponse)
async def generate_pr_description(
    request: GeneratePRDescriptionRequest,
) -> GeneratePRDescriptionResponse:
    """
    Generate a formatted pull request description.

    Optionally uses a template to format the output.
    """
    from codeverify_agents.diff_summarizer import DiffSummarizerAgent

    agent = DiffSummarizerAgent()

    result = await agent.generate_pr_description(
        diff=request.diff,
        title=request.title,
        template=request.template,
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {result['error']}",
        )

    return GeneratePRDescriptionResponse(
        title_suggestion=result.get("title_suggestion", ""),
        description=result.get("description", ""),
        summary=result.get("summary", ""),
        changelog_entry=result.get("changelog_entry", ""),
        labels_suggestion=result.get("labels_suggestion", []),
        reviewers_hint=result.get("reviewers_hint", []),
    )


@router.post("/changelog", response_model=GenerateChangelogResponse)
async def generate_changelog(
    request: GenerateChangelogRequest,
) -> GenerateChangelogResponse:
    """
    Generate changelog entries for multiple commits.

    Groups entries by category (added, changed, fixed, removed, breaking).
    """
    from codeverify_agents.diff_summarizer import DiffSummarizerAgent

    agent = DiffSummarizerAgent()

    # Convert to list of tuples
    diffs = [
        (d.get("commit_sha", "unknown"), d.get("diff", ""))
        for d in request.diffs
    ]

    entries = await agent.generate_changelog_entries(diffs)

    # Format changelog
    categorized: dict[str, list[str]] = {
        "breaking": [],
        "added": [],
        "changed": [],
        "fixed": [],
        "removed": [],
    }

    for entry in entries:
        category = entry.get("category", "changed")
        if category in categorized:
            categorized[category].append(entry.get("entry", ""))

    # Build formatted changelog
    lines = ["# Changelog", ""]

    category_labels = {
        "breaking": "⚠️ Breaking Changes",
        "added": "✨ Added",
        "changed": "🔄 Changed",
        "fixed": "🐛 Fixed",
        "removed": "🗑️ Removed",
    }

    for category, label in category_labels.items():
        if categorized[category]:
            lines.append(f"## {label}")
            lines.append("")
            for entry in categorized[category]:
                lines.append(f"- {entry}")
            lines.append("")

    formatted_changelog = "\n".join(lines)

    return GenerateChangelogResponse(
        entries=[
            ChangelogEntry(
                commit=e.get("commit", ""),
                entry=e.get("entry", ""),
                category=e.get("category", "changed"),
            )
            for e in entries
        ],
        formatted_changelog=formatted_changelog,
    )


@router.get("/templates")
async def get_pr_templates() -> dict[str, Any]:
    """Get available PR description templates."""
    from codeverify_agents.diff_summarizer import DEFAULT_PR_TEMPLATE

    return {
        "templates": {
            "default": {
                "name": "Default Template",
                "description": "Standard PR description template",
                "template": DEFAULT_PR_TEMPLATE,
            },
            "minimal": {
                "name": "Minimal Template",
                "description": "Bare minimum PR description",
                "template": "## Summary\n\n{{summary}}\n\n## Changes\n\n{{changes}}",
            },
            "detailed": {
                "name": "Detailed Template",
                "description": "Comprehensive PR description with all sections",
                "template": """## Summary

{{summary}}

## Changes

{{changes}}

## Breaking Changes

{{breaking_changes}}

## Affected Components

{{components}}

## Risk Assessment

**Level:** {{risk}}

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] No sensitive data exposed
""",
            },
        }
    }
