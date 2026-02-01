# ADR-0001: Monorepo Structure

## Status

Accepted

## Context

CodeVerify consists of multiple components:
- Backend API (Python/FastAPI)
- Analysis workers (Python/Celery)
- Web dashboard (Next.js/React)
- GitHub App webhook handler (Node.js/Express)
- Shared packages (core, verifier, ai-agents)
- VS Code extension
- CLI tool

We needed to decide how to organize these components: as separate repositories (polyrepo) or a single repository (monorepo).

## Decision

We chose a **monorepo structure** with the following organization:

```
codeverify/
├── apps/           # Deployable applications
│   ├── api/
│   ├── worker/
│   ├── web/
│   └── github-app/
├── packages/       # Shared libraries
│   ├── core/
│   ├── verifier/
│   ├── ai-agents/
│   ├── cli/
│   ├── vscode-extension/
│   └── z3-mcp/
├── docker/         # Container definitions
├── deploy/         # Deployment configurations
├── docs/           # Documentation
└── tests/          # Integration and E2E tests
```

## Consequences

### Positive
- **Atomic changes**: Cross-cutting changes can be made in a single PR
- **Simplified dependency management**: Shared packages are always in sync
- **Unified CI/CD**: Single pipeline tests all components together
- **Easier onboarding**: Developers clone one repo to get everything
- **Consistent tooling**: Shared linting, formatting, and testing configurations

### Negative
- **Larger repository size**: Clone includes all components
- **Complex CI configuration**: Must handle multiple languages (Python, Node.js)
- **Permission granularity**: Can't restrict access to individual components easily

### Neutral
- Requires CODEOWNERS for review routing
- Need clear package boundaries to avoid circular dependencies

## Alternatives Considered

### Polyrepo (Separate Repositories)
- **Rejected because**: Would complicate cross-component changes, version synchronization, and developer experience

### Monorepo with Lerna/Nx/Turborepo
- **Partially adopted**: We use native tooling (pip editable installs, npm workspaces) rather than a monorepo framework, keeping complexity low while getting monorepo benefits

## References

- [Monorepo vs Polyrepo](https://github.com/joelparkerhenderson/monorepo-vs-polyrepo)
- [Python Monorepo Best Practices](https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html)
