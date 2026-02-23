# Changelog

All notable changes to CodeVerify will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes._

## [1.6.0] - 2026-02-22

### Added

#### Next-Generation Features (v1.6.0)

- **AI Agent Marketplace**: Third-party agent publishing platform with review, install, revenue sharing
- **Verification Telemetry & Benchmarking**: Anonymized cross-org benchmarking with quarterly reports
- **LLM Output Verification Protocol**: Standardized protocol for any AI assistant to verify code
- **Predictive Defect Heatmap**: ML model predicting defect-prone files with Jira/Linear ticket suggestions
- **Self-Healing Codebase Agent**: Autonomous monitor→diagnose→fix→verify→PR pipeline
- **Verification-Native CI/CD**: .verify.yml pipeline DSL with proof-based gates and auto-rollback
- **Code Evolution Timeline**: Function behavioral contract evolution visualization with Mermaid
- **Verification Credit System**: Org-level gamification with credits, redemption, leaderboard
- **Multi-Modal Verification**: IaC (Terraform), DB migrations, API contracts, config file verification
- **Verification-Aware Code Search**: Semantic search by verification status, trust score, proof coverage

#### Testing

- Added 32 comprehensive tests covering all 10 v1.6.0 features
- Total test count: 1739 passed, 7 skipped

#### New Modules

- `codeverify_core.agent_marketplace` - AI agent marketplace
- `codeverify_core.verification_telemetry` - Telemetry and benchmarking
- `codeverify_core.verification_protocol` - LLM verification protocol
- `codeverify_core.defect_heatmap` - Predictive defect heatmap
- `codeverify_core.self_healing` - Self-healing codebase agent
- `codeverify_core.verification_cicd` - Verification-native CI/CD
- `codeverify_core.evolution_timeline` - Code evolution timeline
- `codeverify_core.credit_system` - Verification credit system
- `codeverify_core.multimodal_verify` - Multi-modal verification
- `codeverify_core.verification_search` - Verification-aware code search

## [1.5.0] - 2026-02-22

### Added

#### Next-Generation Features (v1.5.0)

- **Runtime Verification Bridge**: Static→runtime verification loop
  - Z3 constraint to runtime assertion translation (Python, TypeScript, Go)
  - Runtime violation capture with execution context serialization
  - Feedback loop: confirmed violations boost confidence, non-violations reduce FPs
  - Performance-aware instrumentation with sampling and decorator generation

- **Verification-Guided Fuzzing**: Counterexample-driven property testing
  - Z3 counterexample to test input translation with type classification
  - Input mutation generation for broader coverage
  - Execution classification: confirmed_bug, false_positive, inconclusive
  - Pytest test code generation from counterexamples

- **Intent-Preserving Refactoring**: Behavioral equivalence proofs
  - Pre/post-refactoring contract extraction (parameters, return types, exceptions)
  - Z3 equivalence proof generation across refactoring types
  - Safety scoring with divergent property detection

- **Context-Window Verification**: LLM analysis quality assurance
  - Context consistency checking (type conflicts, duplicate definitions, truncation)
  - Optimal snippet selection within token budgets
  - Truncation detection with quality scoring

- **Verification Replay & Regression**: Proof-based regression testing
  - Session recording with content-addressed snapshots
  - Replay engine detecting regressions in previously-proven properties
  - Trend tracking for proof regression rates over time

- **Natural Language Proof Explanation**: Human-readable Z3 explanations
  - Template-based counterexample narratives (brief/standard/detailed)
  - Proof success summarization
  - PR comment generation with Markdown formatting and caching

- **Verification-Aware Review Assignments**: Intelligent reviewer routing
  - PR risk classification from verification findings
  - Reviewer-expertise matching (security→AppSec, types→TS expert)
  - Load balancing with configurable caps

- **Multi-Repository Invariant Propagation**: Cross-repo guarantees
  - Central invariant registry with scope (repo/org/global)
  - Propagation engine checking invariants across dependent repos
  - Governance dashboard with compliance rates and violation tracking

- **Proof-Based Documentation Generation**: Verified API docs
  - Property extraction from code and specs with proof references
  - Multiple output formats (Markdown, HTML, OpenAPI)
  - Freshness tracking detecting stale documentation

- **Gamified Developer Security Training**: Learn from real findings
  - Personalized curriculum from individual finding history
  - Interactive fix-it challenges using real vulnerability patterns
  - Progress tracking, streak counting, badge earning
  - Team leaderboard with skill level progression

#### Testing

- Added 43 comprehensive tests covering all 10 v1.5.0 features
- Total test count: 1682 passed, 7 skipped

#### New Modules

- `codeverify_core.runtime_bridge` - Runtime verification bridge
- `codeverify_core.guided_fuzzing` - Verification-guided fuzzing
- `codeverify_core.intent_refactoring` - Intent-preserving refactoring
- `codeverify_core.context_window_verify` - Context-window verification
- `codeverify_core.verification_replay_regression` - Verification replay
- `codeverify_core.nl_proof_explanation` - NL proof explanation
- `codeverify_core.review_assignments` - Review assignments
- `codeverify_core.invariant_propagation` - Invariant propagation
- `codeverify_core.proof_docs` - Proof-based documentation
- `codeverify_core.gamified_training` - Gamified training

## [1.4.0] - 2026-02-22

### Added

#### Next-Generation Features (v1.4.0)

- **Agentic Review Orchestrator**: Autonomous agent pipeline
  - Planner agent decomposes PRs into typed verification tasks
  - Parallel dispatch of specialized sub-agents with timeout handling
  - Conflict resolution across agent findings via confidence-weighted voting
  - Circuit breaker pattern for agent reliability
  - Budget-aware task planning and cost estimation

- **Verification-Aware Code Generation**: Provably correct autofix
  - Counterexample-to-constraint translation for LLM prompting
  - Iterative generate→verify loop with Z3 proof certificates
  - Multi-candidate generation with parallel verification
  - Fallback to template fixes when loop exhausts iterations
  - Cost tracking per fix attempt

- **Privacy-Preserving Federated Verification**: Cross-org learning
  - Differential privacy with Laplace noise and epsilon budget tracking
  - Federated aggregation of verification patterns across organizations
  - k-anonymity enforcement (minimum org count per pattern)
  - Privacy level presets (strict/moderate/relaxed)
  - Pattern adoption tracking and contribution metrics

- **Live Verification Debugger**: Interactive proof exploration
  - Proof tree construction from Z3 constraint sets
  - Step-through constraint propagation with variable tracking
  - Variable override and re-evaluation
  - Export to Mermaid, HTML, DOT, JSON formats
  - Shareable proof URLs with permalink support

- **AI Drift & Regression Monitor**: Behavioral change detection
  - Behavioral fingerprinting per function (signature, content, calls, raises)
  - Drift detection on push via fingerprint comparison
  - Invariant monitoring for previously-verified properties
  - Signature, behavior, and exception change classification
  - Historical drift timeline with acknowledgment

- **Spec-First Development Workflow**: NL specifications → Z3 assertions
  - .spec.cv file format with @requires, @ensures, @invariant, @assert
  - NL→Z3 compilation (10+ pattern templates)
  - Automatic verification of code against declared specs
  - Spec auto-generation from type hints and docstrings
  - Spec coverage metrics per file/function

- **Multi-Language Polyglot Bridge**: Cross-language contract verification
  - Contract extraction from Python, TypeScript, Go, Java, Rust
  - Cross-language type compatibility checking with equivalence tables
  - Parameter, return type, and error contract comparison
  - Service boundary dependency graph generation
  - Type mismatch detection with severity classification

- **Organizational Learning Engine**: ML-powered false positive reduction
  - Feedback collection (accept/dismiss/false-positive/helpful)
  - Sigmoid-based false positive classifier with feature weights
  - Severity calibration from historical feedback patterns
  - Per-org noisy rule detection and auto-suppression
  - Predictive quality scoring for code changes

- **Verification Cost Optimizer (Smart Router)**: Budget-aware routing
  - Per-file risk scoring from change size, file criticality, author history
  - Multi-tier depth routing: pattern→static→AI→formal
  - Budget constraint enforcement with priority-based allocation
  - Savings estimation vs full-depth verification
  - Routing statistics and usage tracking

- **Embeddable Verification Widget**: Cross-platform visibility
  - SVG and HTML badge rendering with status colors
  - Trust score and finding summary widget types
  - Embed code generation: iframe, React, Web Component, Markdown
  - Widget data API with JSON responses
  - Configurable themes (light/dark/auto)

#### Testing

- Added 55 comprehensive tests covering all 10 v1.4.0 features
- Total test count: 1628 passed, 7 skipped

#### New Modules

- `codeverify_core.agentic_orchestrator` - Agentic review pipeline
- `codeverify_core.verified_codegen_loop` - Verification-aware code generation
- `codeverify_core.federated_verification` - Federated learning with privacy
- `codeverify_core.live_debugger` - Interactive proof debugger
- `codeverify_core.drift_monitor` - Behavioral drift detection
- `codeverify_core.spec_first` - Spec-first development workflow
- `codeverify_core.polyglot_bridge` - Cross-language contract verification
- `codeverify_core.org_learning` - Organizational learning engine
- `codeverify_core.smart_router` - Verification cost optimizer
- `codeverify_core.embed_widget` - Embeddable verification widget

## [1.3.0] - 2026-02-22

### Added

#### Next-Generation Features (v1.3.0)

- **GitHub Marketplace & One-Click Install**: Seamless marketplace onboarding
  - GitHub Marketplace listing metadata and pricing tiers (Free/Pro/Team/Enterprise)
  - One-click install flow with automatic `.codeverify.yml` generation
  - GitHub Actions workflow template generation
  - Free-tier usage metering and enforcement
  - Installation analytics and lifecycle management

- **Streaming IDE Verification**: Real-time verification as you type
  - Incremental function-level verification on file save
  - Content-addressed verification caching for sub-second responses
  - Inline proof status reporting (verified/warning/error per block)
  - Multi-file dependency tracking and transitive invalidation
  - Verification session management with debouncing

- **Verification Insights API (GraphQL)**: Public API for verification data
  - GraphQL schema for analyses, findings, trust scores, proofs, trends
  - Scoped API key management (read, write, admin, webhooks)
  - Tiered rate limiting (Free/Standard/Premium/Unlimited)
  - Webhook subscriptions for verification events
  - Query complexity analysis and depth limiting

- **AI Autofix with Verified Patches**: Auto-generate and verify fixes
  - Template-based and heuristic fix generation for common categories
  - Verification loop: generate → verify with Z3 → rank by confidence
  - One-click PR suggestion generation with confidence scoring
  - Batch fix mode for scan results
  - Safety guardrails (max diff lines, min confidence, allowed categories)

- **Organization Security Posture Score**: Org-wide health scoring
  - Composite posture score (0-100) from coverage, findings, fix rate, compliance
  - DORA metrics integration (deployment freq, lead time, MTTR, change failure rate)
  - Repository risk heatmap with weighted scoring
  - Trend detection (improving/stable/declining)
  - Executive digest generation with recommendations

- **Copilot Extension (Chat + Agent)**: Native Copilot Chat integration
  - Command routing: /verify, /explain, /fix, /trust-score, /scan, /help
  - Context-aware responses using IDE selection and file info
  - Multi-turn session management with conversation history
  - Code action generation for one-click fixes
  - Agent modes: passive, proactive, guardian

- **Multi-Tenant Hosted SaaS Platform**: Managed service with billing
  - Tenant provisioning with lifecycle management (provision/suspend/reactivate)
  - Plan management (Free/$0, Pro/$49, Enterprise/$199) with feature gates
  - Stripe-compatible billing with usage metering and overage charges
  - Invoice generation and payment tracking
  - Feature flags: SSO, audit log, custom models gated by plan

- **Proof Artifact Marketplace**: Community proof sharing
  - Proof artifact submission with metadata and categorization
  - Privacy-preserving anonymization pipeline (email/URL/path stripping)
  - Search by category, language, tags, and content
  - Community voting and quality scoring
  - Automatic proof reuse via content matching

- **Compliance-as-Code Engine**: NL compliance queries
  - Natural language compliance query parsing with intent classification
  - 10+ pre-built compliance checks (SOC2, HIPAA, PCI-DSS, GDPR, EU AI Act)
  - Automated evidence collection from codebase scanning
  - Gap analysis with remediation recommendations
  - Auditor-ready compliance report generation

- **Performance & Cost Dashboard**: Cost visibility and optimization
  - LLM token usage tracking per model and operation
  - Z3 solver time and resource metrics
  - Cost per review calculation with ROI metrics
  - Budget alerts with configurable thresholds (warning/critical)
  - Optimization recommendations (model downgrades, caching, batching)

#### Testing

- Added 59 comprehensive tests covering all 10 v1.3.0 features
- Total test count: 1573 passed, 7 skipped

#### New Modules

- `codeverify_core.marketplace_listing` - GitHub Marketplace integration
- `codeverify_core.streaming_ide` - Streaming IDE verification
- `codeverify_core.graphql_insights` - GraphQL Insights API
- `codeverify_core.autofix_verified_patches` - Verified autofix pipeline
- `codeverify_core.org_posture` - Organization security posture
- `codeverify_core.copilot_chat_agent` - Copilot Chat extension
- `codeverify_core.hosted_saas` - Hosted SaaS platform
- `codeverify_core.proof_artifact_marketplace` - Proof marketplace
- `codeverify_core.compliance_engine` - Compliance-as-Code engine
- `codeverify_core.perf_cost_dashboard` - Performance & cost dashboard

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
