"""Factory functions for creating VCS clients."""

import re
from typing import Any

from codeverify_core.vcs.base import VCSClient, VCSConfig
from codeverify_core.vcs.bitbucket import BitbucketClient
from codeverify_core.vcs.github import GitHubClient
from codeverify_core.vcs.gitlab import GitLabClient


def get_provider_from_url(url: str) -> str:
    """
    Detect VCS provider from repository URL.

    Args:
        url: Repository URL (clone URL or web URL)

    Returns:
        Provider name: "github", "gitlab", or "bitbucket"

    Raises:
        ValueError: If provider cannot be determined
    """
    url_lower = url.lower()

    # GitHub patterns
    if "github.com" in url_lower or "github.io" in url_lower:
        return "github"

    # GitLab patterns (includes self-hosted with gitlab in domain)
    if "gitlab.com" in url_lower or "gitlab" in url_lower:
        return "gitlab"

    # Bitbucket patterns
    if "bitbucket.org" in url_lower or "bitbucket" in url_lower:
        return "bitbucket"

    # Try to detect from URL structure
    # GitHub Enterprise often uses /api/v3
    if "/api/v3" in url_lower:
        return "github"

    # GitLab often uses /api/v4
    if "/api/v4" in url_lower:
        return "gitlab"

    raise ValueError(f"Cannot determine VCS provider from URL: {url}")


def create_vcs_client(
    provider: str | None = None,
    url: str | None = None,
    token: str | None = None,
    base_url: str | None = None,
    webhook_secret: str | None = None,
    **kwargs: Any,
) -> VCSClient:
    """
    Create a VCS client for the specified provider.

    Args:
        provider: VCS provider name ("github", "gitlab", "bitbucket")
                  If not provided, will be detected from url
        url: Repository URL (used to detect provider if not specified)
        token: Authentication token
        base_url: Base URL for API (for self-hosted instances)
        webhook_secret: Secret for webhook signature verification
        **kwargs: Additional configuration options

    Returns:
        Configured VCS client

    Raises:
        ValueError: If provider cannot be determined or is unsupported

    Examples:
        >>> client = create_vcs_client(provider="github", token="ghp_xxx")
        >>> client = create_vcs_client(url="https://github.com/owner/repo")
        >>> client = create_vcs_client(
        ...     provider="gitlab",
        ...     base_url="https://gitlab.mycompany.com/api/v4",
        ...     token="glpat-xxx"
        ... )
    """
    # Determine provider
    if provider is None:
        if url is None:
            raise ValueError("Either 'provider' or 'url' must be specified")
        provider = get_provider_from_url(url)

    provider = provider.lower()

    # Create config
    config = VCSConfig(
        provider=provider,
        base_url=base_url,
        token=token or "",
        webhook_secret=webhook_secret,
        **kwargs,
    )

    # Create client
    client_classes = {
        "github": GitHubClient,
        "gitlab": GitLabClient,
        "bitbucket": BitbucketClient,
    }

    client_class = client_classes.get(provider)
    if client_class is None:
        raise ValueError(
            f"Unsupported VCS provider: {provider}. "
            f"Supported providers: {', '.join(client_classes.keys())}"
        )

    return client_class(config)


def parse_repo_url(url: str) -> dict[str, str]:
    """
    Parse a repository URL into components.

    Args:
        url: Repository URL

    Returns:
        Dictionary with 'provider', 'owner', 'repo', and 'full_name'
    """
    # Common patterns
    patterns = [
        # HTTPS URLs
        r"https?://(?P<host>[^/]+)/(?P<owner>[^/]+)/(?P<repo>[^/\.]+)",
        # SSH URLs
        r"git@(?P<host>[^:]+):(?P<owner>[^/]+)/(?P<repo>[^/\.]+)",
    ]

    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            groups = match.groupdict()
            host = groups["host"]
            owner = groups["owner"]
            repo = groups["repo"].rstrip(".git")

            # Determine provider from host
            if "github" in host:
                provider = "github"
            elif "gitlab" in host:
                provider = "gitlab"
            elif "bitbucket" in host:
                provider = "bitbucket"
            else:
                provider = "unknown"

            return {
                "provider": provider,
                "host": host,
                "owner": owner,
                "repo": repo,
                "full_name": f"{owner}/{repo}",
            }

    raise ValueError(f"Cannot parse repository URL: {url}")
