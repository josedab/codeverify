# Changelog

All notable changes to CodeVerify will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-02-15

### Added

#### Next-Generation Features (v1.2.0)

- **Rust & C/C++ Memory Safety Verification**: Extend Z3 verifier with memory safety
  - Rust ownership and borrow-checker constraint analysis
  - C/C++ pointer analysis with heap/stack memory model
  - Use-after-free, double-free, buffer overflow, null dereference detection
  - Data race detection for concurrent code (threads, goroutines)
  - Memory leak detection for unfreed heap allocations
  - Language-specific fix suggestions

- **GitHub Copilot Workspace Integration**: Deep integration with Copilot Workspace
  - Plan verification: intercept and verify generated code plans
  - Constraint injection into code generation loop
  - Trust score display for workspace-generated code
  - Session management with verification state
  - Security constraint enforcement (block on critical findings)

- **Zero-Config Onboarding (codeverify init)**: One-command setup
  - Language and framework auto-detection from project files
  - Automatic `.codeverify.yml` generation with smart defaults
  - GitHub Actions workflow generation for CI integration
  - Baseline scan with instant first findings
  - Monorepo detection and multi-language support

- **Autonomous Verification Agent**: Always-on verification teammate
  - Continuous repository monitoring for pushes and PRs
  - Automatic finding triage by severity and confidence
  - Fix candidate generation with formal verification loop
  - PR creation with proof attestations
  - Developer feedback learning loop (false positive reduction)
  - Configurable autonomy levels (suggest / warn / auto-fix)

- **Verification-as-a-Service API**: Hosted verification API
  - RESTful API for code submission and verification
  - Usage-based pricing with tiered plans (Free/Starter/Pro/Enterprise)
  - Verification result caching with content-addressed hashing
  - Rate limiting and quota management
  - SARIF output format support
  - Webhook delivery for async results

- **Interactive Proof Explorer**: Visual Z3 proof exploration
  - Z3 proof tree parsing and navigable visualization
  - Animated constraint propagation step-through
  - Multiple output formats (HTML, Mermaid, DOT, JSON)
  - Shareable proof URLs with embed support
  - Tiered detail levels (summary / detailed / full)

- **Cross-Repository Blast Radius Analysis**: Org-wide impact analysis
  - Organization-wide dependency graph with cycle detection
  - BFS transitive dependent discovery with path tracking
  - Impact level calculation (change type × distance × repo criticality)
  - Blast radius score (0-10) with actionable recommendations
  - Mermaid diagram generation for PR comments
  - Affected team notification

- **AI Code Review Benchmark Suite**: Open-source evaluation framework
  - Labeled code samples across multiple languages and bug categories
  - Benchmark runner with precision/recall/F1/latency metrics
  - Tool adapter interface for evaluating any review tool
  - Leaderboard tracking with rank scoring
  - Export to JSON and Markdown formats

- **Fine-Tuned Verification LLM**: Local model for verification
  - Training data pipeline from verification runs
  - Model configuration for fine-tuning (LoRA/QLoRA support)
  - Local inference engine with GGUF/ONNX backend support
  - Air-gapped deployment packaging with manifest
  - Cost comparison vs cloud APIs (90%+ savings)
  - Task-specific inference (vulnerability detection, fix suggestion, spec inference)

- **Developer Certification Program**: CodeVerify Certified Developer
  - 5-module foundations course curriculum
  - Hands-on lab exercises with verification tasks
  - Assessment engine with multiple question types
  - Digital badge and certificate issuance with cryptographic verification
  - Learner progress tracking and program analytics

#### Testing

- Added 75 comprehensive tests covering all 10 v1.2.0 features
- Total test count: 1514 passed, 7 skipped

#### New Modules

- `codeverify_core.memory_safety` - Rust/C/C++ memory safety verification
- `codeverify_core.copilot_workspace` - Copilot Workspace integration
- `codeverify_core.zero_config` - Zero-config onboarding
- `codeverify_core.autonomous_agent` - Autonomous verification agent
- `codeverify_core.vaas` - Verification-as-a-Service API
- `codeverify_core.proof_explorer_interactive` - Interactive proof explorer
- `codeverify_core.cross_repo_blast` - Cross-repo blast radius analysis
- `codeverify_core.benchmark` - AI code review benchmark suite
- `codeverify_core.fine_tuned_llm` - Fine-tuned verification LLM
- `codeverify_core.certification` - Developer certification program

## [1.1.0] - 2026-02-15

### Added

#### Next-Generation Features (v1.1.0)

- **Hosted SaaS Platform with Free Tier**: Multi-tenant cloud platform
  - Tenant provisioning with isolated configuration
  - Plan tiers (Free/Pro/Enterprise) with configurable limits
  - Usage metering (verifications, API calls, storage, AI tokens)
  - API key management with scoped permissions (read, write, admin, verify)
  - Rate limiting per tenant and plan
  - Trial management with auto-expiry

- **Go + Java Language Support**: Full verification for Go and Java
  - Advanced Go parser with struct, interface, goroutine, channel, and defer detection
  - Advanced Java parser with class, interface, enum, annotation, and generics support
  - Idiomatic pattern detection (error handling, null annotations, resource management)
  - Language-specific fix suggestions
  - Goroutine safety and nil pointer analysis for Go
  - Null annotation and try-with-resources detection for Java

- **Autofix Agent with PR Generation**: Automated fix generation pipeline
  - Template-based and heuristic fix generation for common categories
  - Verification loop: generate fix → verify correctness → rank candidates
  - PR metadata generation (branch, title, body with finding context)
  - Safety guardrails (diff size limits, confidence thresholds)
  - Batch fix support for multiple findings

- **GitHub Copilot Chat Extension**: Native Copilot Chat integration
  - Command routing: verify, explain, fix, scan, status, help
  - Context-aware responses using IDE selection and file info
  - Multi-turn session management
  - Follow-up command suggestions
  - Code action generation for one-click fixes

- **Incremental Verification Engine**: Content-addressed verification caching
  - Function/block-level content hashing for cache keys
  - Dependency graph with transitive invalidation
  - Cache hit/miss metrics with saved time tracking
  - Automatic invalidation on code or dependency changes
  - Configurable TTL for cached results

- **Organization Security Posture Dashboard**: Executive security visibility
  - Org-wide verification coverage aggregation
  - Risk heatmap across repositories with scoring
  - DORA metrics integration (deployment freq, lead time, MTTR, change failure rate)
  - Compliance tracking (SOC2, HIPAA, PCI-DSS, GDPR, ISO 27001)
  - Trend analysis with direction detection (improving/stable/declining)
  - Executive summary generation

- **CI/CD Pipeline Orchestrator**: Multi-platform quality gates
  - Support for GitHub Actions, GitLab CI, CircleCI, Jenkins, Azure DevOps, Bitbucket Pipelines
  - Configurable quality gates (default, strict, lenient, custom)
  - Merge blocking with warning-only mode
  - Pipeline configuration auto-generation per platform
  - Status reporting with findings summary

- **LLM-Powered Proof Explainer**: Human-readable Z3 explanations
  - Z3 counterexample parsing (assignment and SMT-LIB formats)
  - Tiered explanations (summary → detailed → full proof with raw output)
  - Category-specific fix suggestions with code examples
  - Educational resource links per check category
  - Markdown rendering for PR comments and dashboard
  - Batch explanation support

- **Supply Chain Verification**: Dependency security and compliance
  - Lockfile parsing (pip requirements, npm package-lock, go.sum)
  - CVE/GHSA vulnerability matching with severity scoring
  - License compliance checking with configurable policies
  - SBOM generation (CycloneDX-compatible with Package URLs)
  - Transitive dependency risk scoring
  - Dependency graph with reachability analysis

- **Self-Learning Rule Engine**: ML-powered false positive reduction
  - Feedback collection (accepted, dismissed, false positive, helpful)
  - Logistic regression false positive classifier
  - Severity calibration from historical feedback
  - Org-specific pattern learning (frequently dismissed rules, noisy categories)
  - Rule performance tracking with acceptance and FP rates
  - A/B testing support with confidence thresholds

#### Testing

- Added 83 comprehensive tests covering all 10 v1.1.0 features
- Total test count: 1439 passed, 7 skipped

#### New Modules

- `codeverify_core.saas_platform` - Multi-tenant SaaS management
- `codeverify_core.go_java_support` - Advanced Go/Java parsers
- `codeverify_core.autofix_pr_agent` - Autofix pipeline
- `codeverify_core.copilot_chat_extension` - Copilot Chat integration
- `codeverify_core.incremental_verification` - Verification caching
- `codeverify_core.org_security_dashboard` - Security posture dashboard
- `codeverify_core.cicd_orchestrator` - CI/CD quality gates
- `codeverify_core.proof_explainer` - Proof explanations
- `codeverify_core.supply_chain` - Supply chain verification
- `codeverify_core.self_learning_rules` - Self-learning engine

## [1.0.0] - 2026-02-15

### Added

#### Next-Generation Features (v1.0.0)

- **AI Code Insurance Underwriting Platform**: Risk assessment and insurance for AI-generated code
  - Actuarial risk scoring from code quality metrics (trust score, coverage, findings)
  - Premium calculation with industry factors, volume discounts, and coverage-based pricing
  - Policy lifecycle management (create, activate, suspend, cancel)
  - Claim submission with automated validation against verification proofs
  - Portfolio summary with loss ratio tracking

- **Cross-Repository Security Graph**: Organization-wide security knowledge graph
  - Security knowledge graph with repos, packages, vulnerabilities as nodes
  - Cross-repo vulnerability propagation via transitive dependencies
  - Blast radius analysis with path tracing across repositories
  - CVE correlation and organization-wide risk scoring
  - Similarity-based vulnerability prediction

- **Differential Privacy-Preserving Proof Marketplace**: Privacy-safe proof sharing
  - Proof anonymization with identifier stripping and Laplace noise injection
  - Differential privacy budget tracking (epsilon accounting)
  - Federated learning aggregation (FedAvg with noise)
  - Proof search, upvoting, and download tracking
  - Marketplace statistics and quality scoring

- **Blockchain-Verified Code Provenance**: Immutable verification audit trail
  - Attestation records with cryptographic hashes on local/simulated blockchain
  - Content-addressed storage (IPFS-like CID generation)
  - NFT-style verification badges (Bronze → Diamond) based on trust scores
  - Attestation verification and revocation
  - Multi-chain abstraction (Ethereum, Polygon, Hyperledger, local)

- **Natural Language Compliance Query Engine**: Plain English compliance verification
  - NL query parsing with intent classification and standard detection
  - 5+ built-in compliance templates (PII encryption, auth checks, audit logging)
  - Codebase search with evidence collection and strength scoring
  - Violation detection and remediation recommendations
  - Support for SOC2, HIPAA, PCI-DSS, GDPR, ISO 27001, EU AI Act

- **AI Model Bias & Fairness Verification**: Formal fairness verification for ML models
  - Demographic parity, equal opportunity, and equalized odds checks
  - Configurable fairness constraints with threshold tuning
  - Bias level classification (none → severe)
  - Remediation suggestions (reweighting, adversarial debiasing, threshold adjustment)
  - EU AI Act compliance reporting

- **IDE Copilot Undo with Proof Preservation**: Save points for AI suggestions
  - Save point creation on Copilot suggestion acceptance
  - One-click rollback with proof state preservation
  - Trust score trend tracking (improving/stable/declining)
  - Branching save points for comparing multiple suggestions
  - Automatic pruning of old save points

- **Predictive Code Quality Forecasting**: ML-based quality prediction
  - Time series trend analysis with linear regression
  - Anomaly detection using z-score statistical methods
  - 30-day and 90-day quality forecasting
  - What-if scenario analysis (status quo, increase verification, add team)
  - Executive summary generation with actionable recommendations

- **Verification-Driven Code Generation**: Provably correct code from specs
  - Natural language specification parsing with constraint definition
  - Multi-candidate generation with verification loop
  - Security, null safety, bounds check, and type safety constraint checking
  - Iterative refinement using counterexamples
  - Quality scoring and candidate ranking

- **Real-Time Collaborative Verification Sessions**: Google Docs-style collaboration
  - Session management with lobby, active, paused, and recording phases
  - Participant tracking with roles (host, editor, viewer)
  - Line-level verification status (verified/warning/error per line)
  - Chat messaging with proof context linking
  - Session recording and playback
  - Max 10 participants per session

#### Testing

- Added 67 comprehensive tests covering all 10 v1.0.0 features
- Total test count: 1356 passed, 7 skipped

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
