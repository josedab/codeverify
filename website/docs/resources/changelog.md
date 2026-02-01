---
sidebar_position: 4
---

# Changelog

All notable changes to CodeVerify.

## [1.3.0] - 2024-01-15

### Added
- **Semantic Diff** — Understand the real impact of code changes
- **Team Learning Dashboard** — Track patterns across your organization
- **Go language support** (Beta)
- Proof-carrying PRs for verified commits
- Custom rule testing framework
- Webhook support for Slack and Teams

### Changed
- Improved Trust Score calculation with historical accuracy
- Faster Z3 solving with incremental mode
- Better TypeScript generic inference
- Reduced memory usage for large files

### Fixed
- False positives in list comprehension analysis
- Timeout handling in CI environments
- VS Code extension memory leak
- GitHub App comment formatting

## [1.2.0] - 2023-12-01

### Added
- **Copilot Interceptor** — Verify AI suggestions in real-time
- Monorepo support with affected-only analysis
- Automated test generation from counterexamples
- GitLab and Bitbucket integrations
- SARIF export for GitHub Code Scanning

### Changed
- Trust Score now includes AI confidence component
- Improved null flow analysis for Python 3.10+ patterns
- Better handling of TypeScript optional chaining
- Configuration file validation on startup

### Fixed
- Race condition in parallel analysis
- Incorrect line numbers in multi-line expressions
- API rate limiting not respecting retry-after

## [1.1.0] - 2023-10-15

### Added
- **AI-powered explanations** for all findings
- Integer overflow verification
- VS Code extension with real-time analysis
- Self-hosting with Docker Compose
- Kubernetes Helm chart

### Changed
- Default timeout increased to 300 seconds
- Improved error messages with actionable suggestions
- Better support for Python dataclasses
- Reduced false positives in try/except blocks

### Fixed
- Division by zero not detected in floor division
- Array bounds false positive with negative indices
- Configuration inheritance in nested directories

## [1.0.0] - 2023-08-01

### Added
- Initial release
- **Z3 formal verification** for Python and TypeScript
- Null safety verification
- Array bounds verification
- Division by zero verification
- GitHub App integration
- REST API
- CLI tool
- Basic Trust Score

---

## Version Policy

CodeVerify follows [Semantic Versioning](https://semver.org/):

- **Major (X.0.0):** Breaking changes to API or configuration
- **Minor (0.X.0):** New features, backward compatible
- **Patch (0.0.X):** Bug fixes, backward compatible

## Upgrading

### From 1.2.x to 1.3.x

No breaking changes. Upgrade directly:

```bash
pip install --upgrade codeverify
```

### From 1.1.x to 1.2.x

Configuration change for monorepos:

```yaml
# Old
monorepo: true

# New
monorepo:
  enabled: true
  affected_analysis: true
```

### From 1.0.x to 1.1.x

API key format changed:

```bash
# Old
CODEVERIFY_KEY=abc123

# New
CODEVERIFY_API_KEY=cv_live_abc123
```

## Release Schedule

- **Major releases:** Annually
- **Minor releases:** Monthly
- **Patch releases:** As needed

## Beta Features

Features marked as "Beta" may change without notice:
- Go language support
- Proof-carrying PRs

## Deprecation Policy

- Deprecated features are announced one minor version before removal
- Deprecated API endpoints return `Deprecation` header
- Minimum 6-month notice for breaking changes

## Getting Updates

- **GitHub:** Watch [releases](https://github.com/codeverify/codeverify/releases)
- **Email:** Subscribe to release notifications in dashboard
- **RSS:** [releases.atom](https://github.com/codeverify/codeverify/releases.atom)
