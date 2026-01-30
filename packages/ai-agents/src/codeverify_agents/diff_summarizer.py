"""Diff Summarizer Agent - AI-powered diff summarization and PR descriptions."""

import json
import time
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


DIFF_SUMMARY_SYSTEM_PROMPT = """You are an expert code reviewer who creates concise, informative summaries of code changes. Your task is to analyze a git diff and produce:

1. **Summary**: A clear 2-3 sentence summary of what changed
2. **Behavioral Changes**: List of changes that affect program behavior
3. **Risk Assessment**: Quick assessment of risk level (low/medium/high) with reasoning
4. **PR Description**: A well-formatted pull request description
5. **Changelog Entry**: A one-line entry suitable for a changelog
6. **Affected Components**: Which parts of the system are affected
7. **Breaking Changes**: Any breaking changes that consumers should know about

Guidelines:
- Focus on WHAT changed and WHY it matters, not HOW (the diff shows how)
- Highlight anything that could cause issues in production
- Be specific about affected functionality
- Use clear, professional language
- Keep the PR description informative but concise

Respond in JSON format:
{
  "summary": "Brief summary of the change",
  "behavioral_changes": ["Change 1", "Change 2"],
  "risk_assessment": {
    "level": "low|medium|high",
    "reasoning": "Why this risk level"
  },
  "pr_description": "Formatted PR description with sections",
  "changelog_entry": "Single line for changelog",
  "affected_components": ["component1", "component2"],
  "breaking_changes": ["Breaking change 1"] or [],
  "review_suggestions": ["Suggestion 1", "Suggestion 2"]
}"""


class DiffSummarizerAgent(BaseAgent):
    """
    Agent for summarizing code diffs and generating PR descriptions.

    This agent analyzes git diffs and produces human-readable summaries,
    PR descriptions, and changelog entries.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize diff summarizer agent."""
        super().__init__(config)
        # Use GPT-4 for high-quality summaries
        if config is None:
            self.config.openai_model = "gpt-4-turbo-preview"

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Analyze a diff and generate summary.

        Args:
            code: The git diff to analyze
            context: Additional context including:
                - pr_title: Title of the PR
                - file_list: List of changed files
                - commit_messages: List of commit messages
                - base_branch: Target branch
                - author: PR author

        Returns:
            AgentResult with summary data
        """
        return await self.summarize_diff(code, context)

    async def summarize_diff(
        self,
        diff: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Generate a comprehensive summary of a git diff.

        Args:
            diff: The git diff content
            context: Optional context (PR title, commit messages, etc.)

        Returns:
            AgentResult with summary data
        """
        start_time = time.time()
        context = context or {}

        # Build the prompt
        user_prompt = self._build_prompt(diff, context)

        try:
            response = await self._call_llm(
                system_prompt=DIFF_SUMMARY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_mode=True,
            )

            # Parse JSON response
            try:
                summary_data = json.loads(response["content"])
            except json.JSONDecodeError:
                summary_data = {"raw_response": response["content"]}

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                "Diff summary generated",
                diff_lines=len(diff.split("\n")),
                latency_ms=elapsed_ms,
            )

            return AgentResult(
                success=True,
                data=summary_data,
                tokens_used=response.get("tokens", 0),
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("Diff summarization failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _build_prompt(self, diff: str, context: dict[str, Any]) -> str:
        """Build the summarization prompt."""
        prompt_parts = ["Analyze the following git diff:"]

        # Add context if available
        if context.get("pr_title"):
            prompt_parts.append(f"\nPR Title: {context['pr_title']}")

        if context.get("base_branch"):
            prompt_parts.append(f"Target Branch: {context['base_branch']}")

        if context.get("author"):
            prompt_parts.append(f"Author: {context['author']}")

        if context.get("commit_messages"):
            prompt_parts.append("\nCommit Messages:")
            for msg in context["commit_messages"][:5]:  # Limit to 5
                prompt_parts.append(f"  - {msg}")

        if context.get("file_list"):
            prompt_parts.append(f"\nChanged Files ({len(context['file_list'])} total):")
            for file in context["file_list"][:20]:  # Limit to 20
                prompt_parts.append(f"  - {file}")

        # Add the diff (truncate if too long)
        max_diff_length = 15000  # Leave room for response
        if len(diff) > max_diff_length:
            diff = diff[:max_diff_length] + "\n\n... [diff truncated] ..."

        prompt_parts.extend([
            "",
            "```diff",
            diff,
            "```",
            "",
            "Provide a comprehensive summary of these changes.",
        ])

        return "\n".join(prompt_parts)

    async def generate_pr_description(
        self,
        diff: str,
        title: str | None = None,
        template: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a formatted PR description.

        Args:
            diff: The git diff
            title: Optional PR title
            template: Optional template to follow

        Returns:
            Dictionary with generated description and metadata
        """
        context = {"pr_title": title} if title else {}

        result = await self.summarize_diff(diff, context)

        if not result.success:
            return {"error": result.error}

        data = result.data
        pr_description = data.get("pr_description", "")

        # If a template is provided, try to fill it
        if template:
            pr_description = self._fill_template(template, data)

        return {
            "title_suggestion": self._generate_title(data) if not title else title,
            "description": pr_description,
            "summary": data.get("summary", ""),
            "changelog_entry": data.get("changelog_entry", ""),
            "labels_suggestion": self._suggest_labels(data),
            "reviewers_hint": self._suggest_reviewers(data),
        }

    def _generate_title(self, data: dict[str, Any]) -> str:
        """Generate a PR title from summary data."""
        summary = data.get("summary", "")
        if summary:
            # Take first sentence, truncate if needed
            first_sentence = summary.split(".")[0]
            if len(first_sentence) > 72:
                first_sentence = first_sentence[:69] + "..."
            return first_sentence
        return "Code changes"

    def _fill_template(self, template: str, data: dict[str, Any]) -> str:
        """Fill a PR template with generated data."""
        replacements = {
            "{{summary}}": data.get("summary", ""),
            "{{changes}}": "\n".join(f"- {c}" for c in data.get("behavioral_changes", [])),
            "{{breaking_changes}}": "\n".join(f"- {c}" for c in data.get("breaking_changes", [])) or "None",
            "{{components}}": ", ".join(data.get("affected_components", [])) or "N/A",
            "{{risk}}": data.get("risk_assessment", {}).get("level", "unknown"),
        }

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def _suggest_labels(self, data: dict[str, Any]) -> list[str]:
        """Suggest labels based on the changes."""
        labels = []

        risk_level = data.get("risk_assessment", {}).get("level", "")
        if risk_level == "high":
            labels.append("high-risk")
        elif risk_level == "medium":
            labels.append("needs-review")

        if data.get("breaking_changes"):
            labels.append("breaking-change")

        components = data.get("affected_components", [])
        for component in components[:3]:  # Max 3 component labels
            labels.append(f"component:{component.lower().replace(' ', '-')}")

        return labels

    def _suggest_reviewers(self, data: dict[str, Any]) -> list[str]:
        """Suggest reviewers based on affected components."""
        # This would typically integrate with a team/codeowner mapping
        # For now, return hints about what expertise is needed
        hints = []

        risk_level = data.get("risk_assessment", {}).get("level", "")
        if risk_level == "high":
            hints.append("Senior engineer review recommended")

        components = data.get("affected_components", [])
        if "security" in [c.lower() for c in components]:
            hints.append("Security team review recommended")
        if "database" in [c.lower() for c in components]:
            hints.append("DBA review recommended")

        return hints

    async def generate_changelog_entries(
        self,
        diffs: list[tuple[str, str]],  # List of (commit_sha, diff)
    ) -> list[dict[str, Any]]:
        """
        Generate changelog entries for multiple commits.

        Args:
            diffs: List of (commit_sha, diff) tuples

        Returns:
            List of changelog entries
        """
        entries = []

        for commit_sha, diff in diffs:
            result = await self.summarize_diff(diff, {"commit_sha": commit_sha})

            if result.success:
                entries.append({
                    "commit": commit_sha[:8],
                    "entry": result.data.get("changelog_entry", ""),
                    "category": self._categorize_change(result.data),
                })

        return entries

    def _categorize_change(self, data: dict[str, Any]) -> str:
        """Categorize a change for changelog."""
        if data.get("breaking_changes"):
            return "breaking"

        summary = data.get("summary", "").lower()
        if any(word in summary for word in ["fix", "bug", "issue", "error"]):
            return "fixed"
        if any(word in summary for word in ["add", "new", "feature", "implement"]):
            return "added"
        if any(word in summary for word in ["update", "improve", "enhance", "refactor"]):
            return "changed"
        if any(word in summary for word in ["remove", "delete", "deprecate"]):
            return "removed"

        return "changed"


# Default PR template
DEFAULT_PR_TEMPLATE = """## Summary

{{summary}}

## Changes

{{changes}}

## Breaking Changes

{{breaking_changes}}

## Affected Components

{{components}}

## Risk Level

{{risk}}

---
*This description was auto-generated by CodeVerify.*
"""
