# Architecture and Code Quality Audit

## 1. Summary

- The repository has a clear intended architecture (deployable `apps/`, shared `packages/`, specialized agents, Z3 verification) and mostly one-way runtime package dependencies; no runtime circular import cycle was found.
- The current baseline is not executable enough for safe refactoring: the API has a syntax error, Python test collection stops with 87 errors after 291 collected items, web/GitHub App tests do not run meaningful suites, and the VS Code extension does not compile.
- The highest-leverage structural change is to make composition roots thin and make cross-service contracts canonical: the CLI, worker, API, and VS Code entry points currently duplicate behavior or own business logic that should live behind typed package interfaces.
- Genuine duplication exists in the CLI commands, analysis payloads, formal-spec templates/fallbacks, and severity handling; language adapters and provider-specific integrations are only incidentally similar and should remain separate implementations.
- Phase 2 cannot start under the requested green-after-every-commit rule until the baseline defects are repaired separately; correctness changes identified below are explicitly deferred from the refactor phase.

### Existing conventions observed

| Area | Existing codebase idiom |
|---|---|
| Repository layout | ADR-0001 defines deployable services under `apps/` and reusable libraries under `packages/`; Python packages use `src/` layouts and Hatchling. |
| Python | Python 3.11+, pervasive type hints, dataclasses for internal domain records, Pydantic at API boundaries, async service methods, and module-level `structlog` loggers. |
| Agents | Specialized `BaseAgent` implementations return `AgentResult`; the documented architecture favors independent agents coordinated by a thin orchestrator. |
| Lifecycle | Many core features expose `get_*` / `reset_*` singleton accessors for application use and test isolation. Constructor injection is already used for meaningful boundaries such as storage backends. |
| TypeScript/UI | Strict TypeScript, Next.js client components, TanStack Query for server state, one central `ApiClient`, and shared UI primitives under `components/ui`. |
| Style | Ruff with a 100-character line target, four-space Python indentation, two-space web indentation, Conventional Commits, and colocated package/app tests. |
| Intended validation | Pytest + pytest-asyncio, a nominal 50% Python coverage gate, Ruff, strict mypy for core/verifier, Vitest for Node packages, Playwright for web E2E, and TypeScript compilation for the extension. |

### Audit basis

The ten largest non-test production files were read in full:

| Lines | File | Main observation |
|---:|---|---|
| 3,648 | `packages/core/src/codeverify_core/__init__.py` | Manually maintained public facade and lazy export registry. |
| 3,163 | `packages/cli/src/codeverify_cli/main.py` | Monolithic CLI despite an existing modular `commands/` tree. |
| 1,794 | `packages/ai-agents/src/codeverify_agents/cross_language_bridge.py` | Shared contracts plus five independent language adapters. |
| 1,454 | `packages/core/src/codeverify_core/dependency_visualizer.py` | Large but internally separated graph models, builder, analyzer, query, and exporter. |
| 1,380 | `packages/vscode-extension/src/extension.ts` | Activation, lifecycle, commands, state, formatting, and webview rendering. |
| 1,288 | `packages/core/src/codeverify_core/evidence_vault.py` | Storage, integrity, access, audit, export, reporting, and global lifecycle. |
| 1,247 | `packages/ai-agents/src/codeverify_agents/formal_spec_assistant.py` | Template catalog, parser, agent orchestration, validation, suggestions, and cache. |
| 1,244 | `apps/api/src/codeverify_api/routers/formal_specs.py` | HTTP models/endpoints plus parsing, inference, rendering, and verification logic. |
| 1,198 | `packages/core/src/codeverify_core/model_fine_tuning.py` | Explicit subcomponents, but the pipeline reaches into collaborator internals. |
| 1,197 | `packages/ai-agents/src/codeverify_agents/hallucination_detector.py` | Cohesive extraction/validation/detection pipeline with a large static catalog. |

### Test and coverage reality

- The repository contains 107 Python test files (about 40,000 lines), three VS Code unit-test files, and three web Playwright specifications.
- Python collection currently stops with 87 collection errors after discovering 291 items; examples include tests importing `SecurityAnalysisAgent` while production exposes `SecurityAgent`.
- The configured CI coverage command only targets `packages/ apps/`, omitting root integration/E2E/benchmark tests. Because collection fails, there is no trustworthy current coverage percentage and the 50% threshold is not an effective gate.
- `apps/github-app` has a Vitest command but no test files. `apps/web` has no unit tests; its Vitest command discovers the Playwright E2E specs and fails before executing them.
- The VS Code extension has three test files, but compilation currently reports nine TypeScript errors and its test precondition cannot complete. CI does not compile or test the extension.
- Ruff check and format validation are red, format checking is additionally blocked by the API syntax error, and strict mypy stops on duplicate top-level `tests` modules.

## 2. Findings table

| ID | Location | Category | Severity (P0/P1/P2) | Cost | Est. size (S/M/L) | Behavior risk (none/low/high) |
|---|---|---|---|---|---|---|
| CV-001 | `apps/api/src/codeverify_api/routers/public_api.py:374-383`; `apps/api/src/codeverify_api/main.py:61,141` | Correctness / API boundary | P0 | `api_list_analyses` places a required dependency parameter after defaulted parameters, so Python rejects the module and the FastAPI application, API tests, formatting, and builds cannot import. | S | none |
| CV-002 | `.github/workflows/ci.yml:79,130,142-163`; `Makefile:43-57`; `packages/ai-agents/tests/test_agents.py:8,114`; `apps/worker/tests/test_analysis.py:92,115-116`; `apps/web/package.json:11-13` | Test architecture / coverage | P1 | Validation covers inconsistent repository slices and contains stale or non-asserting tests, so the current 87 Python collection errors, absent GitHub App tests, misrouted web tests, and uncompiled extension leave refactors without a trustworthy regression or coverage signal. | M | none |
| CV-003 | `apps/api/pyproject.toml:10-29`; `apps/api/src/codeverify_api/routers/formal_specs.py:21`; `apps/api/src/codeverify_api/routers/debugger.py:80`; `docker/Dockerfile.api:14-23,42-46`; `packages/cli/pyproject.toml:9-17`; `packages/cli/src/codeverify_cli/main.py:2520,2595`; `packages/z3-mcp/pyproject.toml:10-15`; `packages/z3-mcp/src/z3_mcp/server.py:411,438,463` | Boundaries / dependency direction | P1 | Package manifests do not declare sibling packages imported at runtime and Docker/PYTHONPATH manually compensates, so isolated wheels or editable installs can succeed but fail when a feature path imports agents, verifier, or LSP code. | M | low |
| CV-004 | `packages/core/src/codeverify_core/__init__.py:24-3648`; `packages/core/src/codeverify_core/compliance_as_code.py:482-484`; `packages/core/src/codeverify_core/proof_service_api.py:30-32` | God module / leaky public abstraction | P1 | The core facade manually maps 1,689 lazy names and lists 1,791 exports (98 duplicates), advertises 11 names it cannot resolve, and can trigger environment-dependent import failures, so every feature addition edits one hotspot and external consumers cannot rely on the advertised API. | L | high |
| CV-005 | `packages/cli/src/codeverify_cli/main.py:39-3158`; `packages/cli/src/codeverify_cli/commands/analyze.py:22-643`; `packages/cli/src/codeverify_cli/commands/*.py`; `packages/cli/pyproject.toml:26` | Genuine duplication / SRP | P1 | The active 3,163-line entry point duplicates an already complete modular command tree that it never registers, so command fixes, help text, severity rules, and tests can drift between two implementations while only `main.py` is shipped. | L | high |
| CV-006 | `apps/worker/src/codeverify_worker/tasks/analysis.py:65-582,701-1021` | God object / hard-coded dependencies | P1 | `AnalysisPipeline` owns orchestration, mutable findings, parsing, LLM agents, Z3, auto-fix, formatting, GitHub delivery, and API persistence while constructing variable external collaborators internally, so stages cannot be tested or replaced independently and provider changes concentrate in one hot path. | L | high |
| CV-007 | `apps/worker/src/codeverify_worker/tasks/analysis.py:152-180,239-249,258-323,410-476` | Error handling / correctness | P1 | Missing adapters or mandatory semantic/security agent imports are converted into successful or skipped stage results, so an analysis can finish as `completed` while silently omitting promised checks and giving callers false assurance. | S | high |
| CV-008 | `packages/core/src/codeverify_core/models.py:112-181`; `apps/worker/src/codeverify_worker/tasks/analysis.py:24-52,567-582`; `apps/api/src/codeverify_api/routers/internal.py:20-66,137-181` | Contract duplication / shotgun surgery | P1 | Core, worker, and internal API define incompatible analysis/finding shapes and pass stages/findings as raw dictionaries, so adding or renaming a field requires coordinated edits across services and has already left tests constructing obsolete models. | M | high |
| CV-009 | `packages/ai-agents/src/codeverify_agents/formal_spec_assistant.py:210-483`; `apps/api/src/codeverify_api/routers/formal_specs.py:49-648,660-1244`; `packages/vscode-extension/src/client.ts:507-816`; `packages/vscode-extension/src/providers/__tests__/formalSpecAssistantProvider.test.ts:24-318` | Genuine duplication / layering / swallowed errors | P1 | Formal-spec templates, parsers, suggestions, and renderers have multiple divergent implementations, while the extension catches any API failure and returns a smaller local result whose tests duplicate rather than call production code, so outages look successful and a new pattern requires edits in several packages. | L | high |
| CV-010 | `apps/api/src/codeverify_api/routers/formal_specs.py:36-44`; `packages/ai-agents/src/codeverify_agents/formal_spec_assistant.py:830,853-893,1245-1247` | Mutable shared state / correctness | P1 | A process-global assistant owns an unbounded cache keyed only by natural-language text even though conversion accepts context, so one request can reuse another request's context-insensitive result and cache lifetime is tied implicitly to the API process. | S | high |
| CV-011 | `apps/api/src/codeverify_api/routers/webhooks.py:300-517` | Long function / mixed abstraction | P1 | The 217-line, deeply nested installation-event handler combines event dispatch, SQL queries, entity construction, state transitions, and response formatting, so adding an installation action risks inconsistent persistence behavior across branches and there are no direct action-matrix tests. | M | high |
| CV-012 | `packages/vscode-extension/src/extension.ts:28-177,287-1380` | God module / lifecycle ownership | P1 | The extension entry point owns global providers, timeouts, decorations, 30-plus command handlers, workspace listeners, fixes, and large webview templates, so unrelated feature changes collide in one file and disposal/command behavior cannot be characterized in isolation. | L | high |
| CV-013 | `apps/web/src/app/dashboard/analytics/page.tsx:31-58`; `apps/web/src/app/providers.tsx:4-20` | State altitude / hard-coded dependency | P2 | Analytics queries embed a fixed organization UUID while the application provider tree has no organization selection context, so the page cannot safely represent the active tenant and productionizing it would require edits across every organization-scoped query. | M | low |
| CV-014 | `apps/web/src/app/dashboard/debugger/page.tsx:34-113,303-607` | Component cohesion / missing hook seam | P2 | `DebuggerPage` combines session state, three mutations, polling, command orchestration, and all presentation subcomponents, so state-machine behavior has no unit-test seam and UI changes must understand API sequencing. | M | low |
| CV-015 | `packages/core/src/codeverify_core/evidence_vault.py:251-303,327-724,726-1267,1269-1288` | SRP / lifecycle / error handling | P2 | One module combines storage, cryptographic chaining, access tokens, audit logging, ZIP export, compliance assessment, HTML rendering, and a process-global default vault, while storage errors collapse to `False`/`None`, making persistence failure, absence, and lifecycle ownership hard to distinguish. | L | high |
| CV-016 | `packages/core/src/codeverify_core/severity.py:13-205`; `packages/cli/src/codeverify_cli/main.py:128-156`; `apps/worker/src/codeverify_worker/tasks/analysis.py:43,495-530`; `apps/api/src/codeverify_api/routers/stats.py:131` | Primitive obsession / duplication | P2 | A canonical `FindingSeverity` and ordering helper exist, but hot paths continue to use raw strings and independent maps, so aliases and the `info` level are accepted, sorted, counted, or rejected differently across products. | M | low |
| CV-017 | `packages/ai-agents/src/codeverify_agents/cross_language_bridge.py:234-1394` | Module cohesion | P2 | Five language adapters with independent parsing semantics share one 1,794-line module, so any language change loads and conflicts with all others even though their only necessary commonality is the existing `LanguageAdapter` interface. | M | low |
| CV-018 | `packages/core/src/codeverify_core/model_fine_tuning.py:1100,1176` | Feature envy / leaky abstraction | P2 | `FineTuningPipeline` writes `ModelRegistry._evaluations` and reads `ModelEvaluator._results` directly, so changing either collaborator to persistent storage or enforcing invariants requires modifying the pipeline rather than the owning type. | S | low |

## 3. Proposed refactor sequence

The sequence assumes a green baseline. `CV-001`, the baseline repairs in `CV-002`, and the manifest correctness work in `CV-003` require separate approval as defect/build-maintenance work before Phase 2; they must not be bundled into refactor commits.

1. **Add analysis-wire characterization tests** (`CV-008`, `CV-016`): pin the exact current worker JSON, internal API acceptance, enum/string spellings, optional fields, and stage payloads without changing production code.
2. **Add canonical wire DTOs to core, unused** (`CV-008`): introduce typed `FindingPayload`, `StagePayload`, and `AnalysisPayload` models whose serialized form exactly matches the characterized transport.
3. **Migrate worker serialization to the DTOs** (`CV-008`): keep the emitted JSON byte-for-byte equivalent and leave API models unchanged.
4. **Migrate the internal API to the DTOs and delete duplicate request records** (`CV-008`, `CV-016`): preserve HTTP schema and ORM mapping; this is the first high-leverage boundary cleanup.
5. **Add CLI command characterization tests** (`CV-005`): snapshot the command inventory, help, options, exit codes, and representative output from the currently active `main.py`.
6. **Register the existing modular CLI commands one command group per commit** (`CV-005`): leave the old implementations temporarily unreachable and verify parity after each group.
7. **Delete the unreachable CLI command bodies from `main.py`** (`CV-005`): retain only the Click root, shared context, and command registration.
8. **Add worker pipeline characterization tests** (`CV-006`, `CV-007`): pin stage order, current skipped-stage semantics, result ordering, summary calculation, and failure propagation, including behavior believed to be wrong.
9. **Introduce an `AnalysisDependencies` collaborator bundle with production defaults** (`CV-006`): inject agent factories, parser registry, verifier, result publisher, and persistence client; keep pure summary/formatting helpers direct rather than adding DI ceremony.
10. **Move pipeline stages and result delivery behind narrow interfaces** (`CV-006`): use a move-only/reference-update commit for each module relocation, followed by a separate wiring/edit commit; preserve the characterized skip behavior from `CV-007`.
11. **Add an installation-event action matrix** (`CV-011`): characterize `created`, `deleted`, `suspend`, `unsuspend`, `added`, `removed`, unknown actions, existing records, and transaction calls.
12. **Extract installation persistence and per-action handlers** (`CV-011`): keep the FastAPI/webhook function as transport normalization plus dispatch, with no response-shape changes.
13. **Add formal-spec parity characterization** (`CV-009`, `CV-010`): pin API, agent, and extension-local outputs separately, including current fallback and context-blind cache behavior.
14. **Move pure formal-spec parsing/rendering out of the API router** (`CV-009`): first relocate unchanged helpers with reference updates, then make the router depend on the service in a separate commit.
15. **Create one canonical, versioned specification-template data artifact** (`CV-009`): migrate the agent, legacy API endpoint, and VS Code offline fallback to it in separate commits while preserving each public response shape.
16. **Add VS Code activation/lifecycle characterization** (`CV-012`): assert registered command IDs, listener registration, configuration gates, and disposal against mocked VS Code APIs.
17. **Move VS Code command groups and webview renderers out of `extension.ts`** (`CV-012`): one move-only/reference commit per feature group, followed by an `ExtensionRuntime` ownership edit for providers, decorations, and timers.
18. **Add a core-facade export contract test** (`CV-004`): snapshot current resolvable names, aliases, ordering, duplicates, and failures without changing the public surface.
19. **Mechanically split the lazy export registry by feature** (`CV-004`): assemble the exact current facade from smaller registries; do not remove, rename, or redirect public names in the refactor phase.
20. **Extract `useDebuggerSession` and debugger presentation components** (`CV-014`): pin the current mutation/polling sequence first and keep the page as composition plus layout.
21. **Introduce an organization provider with the current UUID as its default** (`CV-013`): route analytics queries through the provider without changing visible behavior; real auth/selection wiring remains separate product work.
22. **Split evidence storage, vault operations, and report rendering by file** (`CV-015`): perform relocations without content edits first, then inject the vault at application boundaries while retaining the existing public getter until external-consumer status is known.
23. **Move each cross-language adapter to its own module** (`CV-017`): preserve the existing interface and registry; do not create a generic parser that erases language-specific behavior.
24. **Add registry/evaluator-owned evaluation accessors** (`CV-018`): migrate `FineTuningPipeline` off private dictionaries without changing keys or comparison results.
25. **Replace local Python severity ordering/counting maps one consumer at a time** (`CV-016`): use the existing core severity utilities while preserving each consumer's accepted input and serialized output.

### Correctness items explicitly deferred from Phase 2

- `CV-001`: repair the invalid FastAPI function signature in a dedicated bug-fix commit.
- `CV-002`: restore a green, repository-wide validation matrix in dedicated test/build-maintenance commits; do not mass-format as part of a refactor.
- `CV-003`: correct package metadata and add isolated install/import smoke tests in dedicated packaging commits.
- `CV-004`: remove invalid exports or change ambiguous public aliases only after identifying external consumers; that is a public API decision.
- `CV-007`: decide whether missing mandatory stages fail, degrade, or mark results partial; changing that policy is behavior, not refactoring.
- `CV-010`: include conversion context in cache identity and bound/own cache lifetime in a dedicated correctness change.
- `CV-013`: replace the default mock organization with authenticated organization selection as product work, after the seam exists.

## 4. Explicitly out of scope

- **No framework or monorepo-tool rewrite.** ADR-0001 deliberately uses native package tooling; adding Nx, Turborepo, Poetry, or a DI container would add migration cost without addressing the concrete boundary failures.
- **No generic language-adapter deduplication.** Python, TypeScript, Go, Rust, and Java adapters look structurally similar but change for different language semantics; only file separation is recommended.
- **No split-by-line-count campaign.** `dependency_visualizer.py` and `hallucination_detector.py` are large but have explicit internal roles and cohesive feature reasons to change; they should not be fragmented without a concrete change pressure.
- **No provider/client unification across GitHub, GitLab, and Bitbucket solely because method names align.** Their transport semantics, authentication, pagination, and failure modes differ.
- **No speculative React memoization or global state library.** No prop drilling beyond two levels or measured unrelated-render bottleneck was confirmed; the actionable UI issues are missing orchestration seams and organization state altitude.
- **No injection of pure helpers.** Severity comparisons, render formatting, hash functions, and deterministic parsers should remain direct calls; injection adds value for LLM, Z3, VCS, storage, persistence, and process-lifecycle boundaries only.
- **No removal of existing deprecated modules during this refactor.** `docs/module-consolidation.md` promises compatibility until v2.0.0, indicating external consumers may exist; removal needs a separately approved breaking release.
- **No prompt, verification algorithm, database schema, HTTP response, CLI output, or extension UX changes.** Those are observable behavior and require separate feature or bug-fix work.
