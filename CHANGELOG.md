# Changelog

All notable changes to CodeVerify will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-01-31

### Added

#### Next-Generation Features

- **Monorepo Intelligence**: Cross-package dependency analysis
  - Support for Nx, Turborepo, Lerna, pnpm, and Yarn workspaces
  - Dependency graph visualization with cycle detection
  - Affected package detection for incremental verification
  - Build order optimization
  - CLI command: `codeverify monorepo`

- **AI Regression Test Generator**: Auto-generate tests from Z3 counterexamples
  - Support for pytest, unittest, jest, vitest, and go_test frameworks
  - Converts formal verification counterexamples to runnable tests
  - Automatic test naming and organization
  - CLI command: `codeverify generate-tests`

- **Proof-Carrying PRs**: Cryptographic verification attestations
  - HMAC-SHA256 signed proofs
  - Proof embedding in commit messages and PR comments
  - Attestation expiry and validation
  - Compressed proof serialization for transport
  - CLI command: `codeverify attest`

- **IDE-Native Copilot Interceptor**: VS Code Copilot integration
  - Intercepts Copilot suggestions before insertion
  - Real-time verification with inline decorations
  - Status indicators (verifying, passed, warning, error)
  - Hover information with verification details

- **Natural Language Invariant Specs**: English to Z3 assertions
  - Pattern-based parsing for common constraints
  - LLM fallback for complex specifications
  - Generates both Z3 Python and SMT-LIB output
  - CLI command: `codeverify invariants`

- **Semantic Diff Visualization**: Behavioral change analysis
  - Detects signature, behavior, and exception changes
  - Generates Mermaid and DOT format diagrams
  - HTML visualization for PR reviews
  - Impact classification (breaking, backward-compatible)
  - CLI command: `codeverify semantic-diff`

- **Verification Budget Optimizer**: ML-based verification routing
  - Risk-based depth selection (pattern → static → AI → formal)
  - Cost estimation and budget tracking
  - Batch optimization with priority queueing
  - Outcome learning for improved routing
  - CLI command: `codeverify budget`

- **Team Learning Mode**: Organization-wide pattern detection
  - Systemic pattern identification across teams
  - Training recommendations based on recurring issues
  - Organization health reports with trends
  - Markdown export for stakeholder reporting
  - CLI command: `codeverify team-report`

- **Competing Model Arbitration**: Multi-LLM consensus with voting
  - Multiple voting strategies: Borda count, approval, ranked choice
  - Confidence-weighted arbitration
  - Model debate for disputed findings
  - Specialization-aware model weighting

- **Gradual Verification Ramp**: Warnings-only onboarding mode
  - Configurable baseline, observation, and transition phases
  - Progressive severity enforcement
  - Pause/resume/extend ramp controls
  - PR comments explaining current phase
  - CLI command: `codeverify ramp`

#### CLI Commands
- `codeverify monorepo <path>` - Analyze monorepo structure
- `codeverify monorepo affected <files>` - Get affected packages
- `codeverify generate-tests <file>` - Generate tests from counterexamples
- `codeverify attest <pr>` - Create verification attestation
- `codeverify attest verify <attestation>` - Verify an attestation
- `codeverify invariants <spec>` - Compile NL invariants to Z3
- `codeverify semantic-diff <old> <new>` - Visualize behavioral changes
- `codeverify budget estimate <files>` - Estimate verification cost
- `codeverify budget report` - Show usage report
- `codeverify team-report` - Generate team learning report
- `codeverify ramp start <repo>` - Start verification ramp
- `codeverify ramp status <repo>` - Show ramp progress
- `codeverify ramp pause|resume|end <repo>` - Control ramp

#### New Packages/Modules
- `codeverify_core.monorepo` - Monorepo analysis
- `codeverify_core.proof_carrying` - Proof attestations
- `codeverify_core.budget_optimizer` - Verification routing
- `codeverify_core.gradual_ramp` - Onboarding ramp
- `codeverify_agents.test_generator` - Test generation
- `codeverify_agents.nl_invariants` - NL to Z3
- `codeverify_agents.semantic_diff` - Behavioral diff
- `codeverify_agents.team_learning` - Team analytics
- `codeverify_agents.model_arbitrator` - Multi-model voting

#### VS Code Extension
- New provider: `copilotInterceptorProvider.ts`
- Copilot suggestion interception and verification

### Changed
- Extended `__init__.py` exports in core and ai-agents packages
- Added comprehensive unit tests for all new features

## [0.2.0] - 2026-01-29

### Added

#### New Features
- **Copilot Trust Score**: ML-powered scoring system for AI-generated code
  - Risk level assessment (low/medium/high/critical)
  - AI detection probability
  - Weighted scoring factors (complexity, patterns, history, verification, quality)
  - CLI command: `codeverify trust-score`

- **Multi-VCS Support**: Extended beyond GitHub
  - GitLab integration with merge request support
  - Bitbucket integration with pull request support
  - VCS abstraction layer for easy extension
  - Unified webhook handling for all providers

- **Verification Debugger**: Step-through Z3 constraint visualization
  - Interactive debugging sessions
  - Step-by-step trace output
  - Counterexample visualization
  - CLI command: `codeverify debug`

- **Custom Rule Builder**: No-code rule creation
  - Pattern-based rules (regex)
  - AST-based rules
  - Semantic rules
  - Composite rules with AND/OR logic
  - Built-in rule templates
  - CLI command: `codeverify rules`

- **AI Diff Summarizer**: Automatic PR descriptions
  - Change categorization
  - Risk assessment
  - Changelog entry generation
  - Suggested reviewers

- **Codebase-Wide Scanning**: Full repository analysis
  - Scheduled scans with cron expressions
  - Trend tracking over time
  - Configurable scan profiles
  - CLI command: `codeverify scan`

- **Slack/Teams Integration**: Real-time notifications
  - Slack Block Kit formatting
  - Microsoft Teams MessageCard formatting
  - Configurable event subscriptions
  - Finding alerts and scan summaries

- **Public API & Webhooks**: Programmatic access
  - RESTful API with API key authentication
  - Webhook subscriptions for events
  - Rate limiting (tiered by plan)
  - HMAC signature verification

- **VS Code Extension Enhancements**:
  - Real-time verification as you type
  - Trust score status bar
  - Verification decorations
  - Interactive debug panel

- **MCP Server Marketplace**: Open-source Z3 MCP
  - Template-based verification rules
  - New tools: check_overflow, check_bounds, check_div_zero
  - MIT licensed for community use

#### CLI Commands
- `codeverify trust-score <path>` - Calculate trust scores
- `codeverify rules <path>` - Evaluate custom rules
- `codeverify scan <path>` - Run full codebase scan
- `codeverify debug <file>` - Debug verification
- `codeverify list-rules` - List available rules

#### API Endpoints
- `POST /api/v1/trust-score/analyze` - Calculate trust score
- `GET/POST /api/v1/rules` - Manage custom rules
- `POST /api/v1/rules/test` - Test rule against code
- `POST /api/v1/scans` - Trigger codebase scan
- `POST /api/v1/scans/schedule` - Schedule recurring scan
- `POST /api/v1/notifications/slack` - Configure Slack
- `POST /api/v1/notifications/teams` - Configure Teams
- `POST /api/v1/debugger/trace` - Debug verification
- `POST /api/v1/diff/summarize` - Summarize PR diff
- `GET/POST /api/webhooks` - Manage webhooks
- `GET/POST /api/keys` - Manage API keys

#### Database
- New models: Webhook, WebhookDelivery, CodebaseScan, ScanSchedule, NotificationConfig, TrustScoreCache, DiffSummaryCache
- Migration: `002_next_gen_features`

### Changed
- Updated README with new features section
- Enhanced VS Code extension with real-time capabilities
- Expanded Z3 MCP server with marketplace features

### Documentation
- Added `docs/api/PUBLIC_API.md` - Complete API reference
- Added `docs/api/WEBHOOK_EVENTS.md` - Webhook event documentation

---

## [0.1.0] - 2026-01-15

### Added
- Initial release
- AI Semantic Analysis with LLM
- Formal Verification with Z3 SMT solver
- GitHub App integration
- PR checks and inline comments
- Python and TypeScript support
- Web dashboard
- `.codeverify.yml` configuration
- CLI tool for local analysis

### Core Features
- Null safety verification
- Array bounds checking
- Integer overflow detection
- Division by zero prevention
- Security vulnerability scanning
- Code quality analysis

---

## [Unreleased]

### Added

#### Next-Gen Features (v0.3.0)

- **AI Pair Reviewer**: Real-time code review as developers type
  - Sub-function granularity analysis with streaming feedback
  - Smart throttling (verify on pause, not keystroke)
  - Learning from user corrections to reduce false positives
  - VS Code integration with CodeLens and inline diagnostics
  - `PairReviewerAgent` in ai-agents package
  - `pairReviewerProvider.ts` for VS Code extension
  - New commands: `togglePairReviewer`, `reviewUnit`, `applyFixes`

- **Verification Memory Graph**: Persistent knowledge graph of verified code
  - Proof artifact serialization and storage
  - Knowledge graph with nodes for proofs, patterns, functions
  - Cross-project learning for proof reuse
  - Pattern similarity matching for proof suggestions
  - Privacy-preserving organization-level proof aggregation
  - `memory_graph.py` in core package with `VerificationKnowledgeGraph`

- **Formal Specification Generator**: LLM-powered auto-generation of specs
  - Pre/post condition inference from code + documentation
  - Loop and class invariant detection
  - Z3 validation with counterexample feedback
  - Interactive refinement based on validation failures
  - Specification coverage metrics
  - `SpecificationGeneratorAgent` in ai-agents package

- **Security Threat Modeling Agent**: AI-powered threat model generation
  - STRIDE threat categorization (Spoofing, Tampering, etc.)
  - OWASP Top 10 2021 mapping
  - Attack surface identification and risk scoring
  - Data flow diagram generation
  - `ThreatModelingAgent` in ai-agents package

- **Regression Oracle**: ML-powered bug prediction
  - Risk scoring based on change metrics and history
  - Author and file bug frequency tracking
  - Similar past bug detection
  - Verification priority assignment
  - Budget allocation for batch verification
  - `RegressionOracle` in ai-agents package

- **Multi-Model Consensus Verification**: Reduce false positives
  - Query multiple LLMs (GPT-5, Claude, GPT-4) in parallel
  - Configurable consensus strategies (unanimous, majority, weighted)
  - Finding similarity matching across models
  - Escalation from fast to consensus for uncertain findings
  - `MultiModelConsensus` in ai-agents package

- **Proof Artifact Repository**: Searchable proof library
  - Proof storage with category and language indexing
  - Pattern-based proof templates for common cases
  - Community proof sharing and voting
  - Automatic proof creation from verification results
  - `ProofArtifactRepository` in core package

- **Compliance Attestation Engine**: Auto-generate compliance reports
  - SOC2, HIPAA, PCI-DSS, GDPR, ISO 27001 frameworks
  - Control mapping from verification results
  - Evidence artifact linking
  - Multi-framework report generation
  - Attestation certificate generation
  - `ComplianceAttestationEngine` in ai-agents package

- **Verification Cost Optimizer**: Smart verification routing
  - Risk-based depth selection (pattern → static → AI → formal)
  - Budget constraint management
  - Cost model learning from outcomes
  - Batch optimization for multiple changes
  - Usage metrics and reporting
  - `VerificationCostOptimizer` in core package

- **Cross-Language Verification Bridge**: Polyglot codebase support
  - Language-agnostic type and function contracts
  - Python and TypeScript adapters
  - Contract inference from code
  - Cross-language compatibility checking
  - Stub generation in target languages
  - `CrossLanguageVerificationBridge` in ai-agents package

- **Sub-Function Analysis Engine**: Fine-grained incremental analysis
  - Statement-level change detection
  - Semantic block identification and dependency tracking
  - `SubFunctionParser` with Python and TypeScript support
  - `IncrementalAnalysisEngine` for targeted re-verification

- **Real-Time Copilot Sessions**: Streaming verification sessions
  - Session pooling for low latency
  - Context-aware prompting
  - Feedback loop for user corrections
  - `CopilotReviewSession` and `CopilotSessionPool`

### Planned
- Java and Go language support
- JetBrains IDE plugin
- Custom LLM model fine-tuning
- Self-hosted enterprise deployment
- SAML/SSO authentication
- Audit logging dashboard
- GraphQL API
