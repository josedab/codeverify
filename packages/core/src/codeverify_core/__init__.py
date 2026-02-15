"""CodeVerify Core - Shared utilities and data models.

Quick start::

    from codeverify_core import LocalZ3Verifier
    verifier = LocalZ3Verifier()
    results = verifier.verify_all(code, "python")

Common imports:

    Data models:       Analysis, Finding, CodeLocation, VerificationType
    Verification:      LocalZ3Verifier, StreamingVerificationSession
    Rules engine:      CustomRule, RuleEvaluator, get_builtin_rules
    Severity:          FindingSeverity, compare_severity
    CI/CD gates:       PolicyEngine, PolicySet, get_default_policy_set
    CLI entry point:   cli_verify

See ``examples/`` in the repository root for runnable demos.
"""

# Import severity utilities directly from the severity module
from codeverify_core.agent_runtime import (
    AgentLoadError,
    AgentSandbox,
    IsolatedAgentRunner,
    ResourceLimitExceeded,
    SandboxConfig,
    SandboxError,
    SecurityViolation,
    run_agent,
)

# Plugin Marketplace & Agent SDK (v0.5.0)
from codeverify_core.agent_sdk import (
    AgentCapability,
    AgentCategory,
    AgentLanguage,
    AgentLifecycle,
    AgentManifest,
    AgentPackage,
)
from codeverify_core.agent_sdk import (
    AnalysisContext as SDKAnalysisContext,
)
from codeverify_core.agent_sdk import (
    AnalysisResult as SDKAnalysisResult,
)
from codeverify_core.agent_sdk import (
    BaseAgent as SDKBaseAgent,
)
from codeverify_core.agent_sdk import (
    Finding as SDKFinding,
)
from codeverify_core.agent_sdk import (
    SeverityLevel as SDKSeverityLevel,
)
from codeverify_core.agent_sdk import (
    agent as agent_decorator,
)

# Next-Gen Feature 7: Verification Budget Optimizer
from codeverify_core.budget_optimizer import (
    BatchOptimizationResult,
    Budget,
    CostEstimator,
    CostModel,
    DepthSelector,
    OutcomeLearner,
    RiskFactors,
    RiskScorer,
    VerificationBudgetOptimizer,
    VerificationDecision,
)
from codeverify_core.budget_optimizer import (
    VerificationDepth as BudgetVerificationDepth,
)

# Copilot Chat Integration (v0.5.0)
from codeverify_core.copilot_extension import (
    CodeSuggestion,
    CommandParser,
    CopilotChatParticipant,
    CopilotCommand,
    CopilotContext,
    CopilotMessage,
    CopilotMessageRole,
    CopilotResponse,
    CopilotWebhookHandler,
)
from codeverify_core.copilot_sessions import (
    CopilotReviewSession,
    CopilotSessionPool,
    RealTimeCopilotReviewer,
    SessionContext,
    SessionResult,
    SessionState,
    StreamingFinding,
    copilot_reviewer,
)

# Feature 9: Cost Optimizer
from codeverify_core.cost_optimizer import (
    BudgetConstraints,
    CostMetrics,
    RiskProfile,
    VerificationCost,
    VerificationCostOptimizer,
    VerificationDepth,
    VerificationPlan,
)
from codeverify_core.events import (
    AnalysisCompleteEvent,
    AnalysisFailedEvent,
    CriticalFindingEvent,
    DigestReadyEvent,
    Event,
    EventBus,
    EventPriority,
    ScanCompleteEvent,
    get_event_bus,
    on_event,
    reset_event_bus,
)

# New: Compliance Evidence Vault
from codeverify_core.evidence_vault import (
    ComplianceFramework,
    ComplianceReportGenerator,
    EvidenceType,
    EvidenceVault,
    StoredEvidence,
    get_evidence_vault,
    reset_evidence_vault,
)

# Next-Gen Feature 10: Gradual Verification Ramp
from codeverify_core.gradual_ramp import (
    BaselineCollector,
    BaselineMetrics,
    EnforcementDecision,
    EnforcementLevel,
    GradualVerificationRamp,
    RampPhase,
    RampProgress,
    RampSchedule,
    RampState,
)

# Organization Knowledge Graph (v0.5.0)
from codeverify_core.knowledge_graph import (
    EdgeType as KGEdgeType,
)
from codeverify_core.knowledge_graph import (
    GraphEdge as KGGraphEdge,
)
from codeverify_core.knowledge_graph import (
    GraphNode as KGGraphNode,
)
from codeverify_core.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphConfig,
    KnowledgeIngester,
    ProofReuseEngine,
    get_knowledge_graph,
    reset_knowledge_graph,
)
from codeverify_core.knowledge_graph import (
    NodeType as KGNodeType,
)

# Go + Java Language Support (v0.5.0)
from codeverify_core.language_support import (
    LanguageConfig as LangConfig,
)
from codeverify_core.language_support import (
    LanguageFeature,
    LanguageParser,
    LanguageRule,
    LanguageRuleRegistry,
    SupportedLanguage,
    Z3ConstraintGenerator,
    detect_language,
    get_language_registry,
    reset_language_registry,
)
from codeverify_core.memory_graph import (
    ConstraintKind,
    CrossProjectLearner,
    GraphEdgeType,
    GraphNodeType,
    InMemoryProofStorage,
    PatternFingerprint,
    ProofArtifact,
    ProofStatus,
    ProofStorageBackend,
    ProofType,
    SerializedConstraint,
    VerificationKnowledgeGraph,
    compute_code_hash,
    compute_pattern_hash,
    cross_project_learner,
    extract_pattern_fingerprint,
    verification_memory_graph,
)
from codeverify_core.models import (
    Analysis,
    AnalysisStatus,
    BoolResult,
    DataclassTimestampMixin,
    Finding,
    FindingCategory,
    OperationResult,
    # Result type pattern
    Result,
    StrResult,
    # Timestamp utilities
    TimestampMixin,
    VerificationType,
    parse_iso_datetime,
)

# Next-Gen Feature 1: Monorepo Intelligence
from codeverify_core.monorepo import (
    DependencyEdge,
    InterfaceContract,
    MonorepoAnalyzer,
    PackageInfo,
    WorkspaceType,
)

# New: Natural Language Bug Queries
from codeverify_core.nl_bug_queries import (
    BugCategory,
    FindingsIndex,
    NLQueryEngine,
    QueryIntent,
    QueryParser,
    QueryResponse,
    SemanticQuery,
    get_nl_query_engine,
    reset_nl_query_engine,
)
from codeverify_core.nl_bug_queries import (
    SearchResult as NLSearchResult,
)
from codeverify_core.notification_handlers import (
    ConfigProvider,
    InMemoryConfigProvider,
    NotificationEventHandler,
    setup_notification_handlers,
)
from codeverify_core.notifications import (
    NotificationChannel,
    NotificationConfig,
    NotificationSender,
    NotificationType,
    SlackFormatter,
    TeamsFormatter,
)

# New: Offline Mode
from codeverify_core.offline_mode import (
    LocalModelConfig,
    LocalModelType,
    LocalZ3Verifier,
    OfflineAnalysisResult,
    OfflineCapability,
    OfflineModeConfig,
    OfflineModeManager,
    OllamaClient,
    get_offline_manager,
    reset_offline_manager,
)
from codeverify_core.org_dependencies import (
    DependencyEdge as OrgDependencyEdge,
)

# New: Organization Dependencies
from codeverify_core.org_dependencies import (
    OrgDependencyAnalyzer,
    OrgDependencyGraph,
    OrgRepository,
    TransitiveRisk,
    get_org_dependency_analyzer,
)

# Policy Engine / CI-CD Gates (v0.5.0)
from codeverify_core.policy_engine import (
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyEvaluationResult,
    PolicyRule,
    PolicyScope,
    PolicySet,
    get_default_policy_set,
    parse_policy_yaml,
)

# Next-Gen Feature 3: Proof-Carrying PRs
from codeverify_core.proof_carrying import (
    ProofAttestation,
    ProofCarryingPRManager,
    ProofSerializer,
    VerificationProof,
)

# New: Proof Coverage Dashboard
from codeverify_core.proof_coverage import (
    CoverageTrend,
    DashboardData,
    FileCoverage,
    FunctionCoverage,
    LineCoverage,
    ProofCoverageCalculator,
    ProofCoverageDashboard,
    RepositoryCoverage,
    VerificationCategory,
    get_proof_coverage_dashboard,
    reset_proof_coverage_dashboard,
)
from codeverify_core.proof_coverage import (
    ProofStatus as ProofCoverageStatus,
)
from codeverify_core.proof_repository import (
    InMemoryProofStorage as InMemoryProofRepo,
)

# Feature 7: Proof Repository
from codeverify_core.proof_repository import (
    ProofArtifactRepository,
    ProofCategory,
    ProofStorage,
    ProofTemplate,
    SearchQuery,
    SearchResult,
)
from codeverify_core.proof_repository import (
    ProofStatus as ProofRepoStatus,
)

# Repository pattern
from codeverify_core.repositories import (
    InMemoryNotificationConfigRepository,
    InMemoryRepository,
    InMemoryScanResultRepository,
    InMemoryScheduledScanRepository,
    NotificationConfigRepository,
    Repository,
    ScanResultRepository,
    ScheduledScanRepository,
    get_notification_config_repository,
    get_scan_result_repository,
    get_scheduled_scan_repository,
    set_notification_config_repository,
    set_scan_result_repository,
    set_scheduled_scan_repository,
)

# ROI Dashboard & Cost Transparency
from codeverify_core.roi_dashboard import (
    BUG_COST_ESTIMATES,
    BugCaught,
    BugSeverity,
    BugValueCalculator,
    CostCategory,
    CostConfig,
    CostTracker,
    ROIDashboard,
    ROIMetrics,
    create_dashboard,
)
from codeverify_core.roi_dashboard import (
    VerificationCost as ROIVerificationCost,
)

# New feature exports
from codeverify_core.rules import (
    ASTRuleStrategy,
    CompositeRuleStrategy,
    CustomRule,
    PatternRuleStrategy,
    RuleBuilder,
    # Strategy pattern exports
    RuleEvaluationStrategy,
    RuleEvaluator,
    RuleType,
    RuleViolation,
    SemanticRuleStrategy,
    get_builtin_rules,
)

# SBOM + SLSA Provenance Integration
from codeverify_core.sbom import (
    SBOM,
    Component,
    ComponentHash,
    ComponentType,
    ExternalReference,
    LicenseType,
    SBOMFormat,
    SBOMGenerator,
    SLSAAttestationGenerator,
    SLSALevel,
    SLSAProvenance,
    VerificationAttestation,
    VerifiedSBOMExporter,
)
from codeverify_core.sbom import (
    Vulnerability as SBOMVulnerability,
)
from codeverify_core.scanning import (
    CodebaseScanResult,
    ScanConfiguration,
)
from codeverify_core.severity import (
    SEVERITY_EMOJI,
    SEVERITY_LABELS,
    SEVERITY_ORDER,
    FindingSeverity,
    compare_severity,
    get_severity_emoji,
    get_severity_label,
    is_above_threshold,
    is_blocking_severity,
    parse_severity,
    sort_by_severity,
)

# Streaming Verification API (v0.5.0)
from codeverify_core.streaming_verification import (
    IncrementalDiff,
    SessionStatus,
    StreamEvent,
    StreamEventType,
    StreamingSessionConfig,
    StreamingSessionPool,
    StreamingVerificationSession,
    get_streaming_pool,
    reset_streaming_pool,
)
from codeverify_core.streaming_verification import (
    VerificationStage as StreamingStage,
)
from codeverify_core.sub_function_analysis import (
    GranularityLevel,
    IncrementalAnalysisEngine,
    Position,
    RealTimeFeedbackModel,
    SemanticBlock,
    SemanticBlockType,
    Span,
    SubFunctionParser,
    SymbolDefinition,
    SymbolReference,
)

# Supply Chain Verification (v0.5.0)
from codeverify_core.supply_chain_verification import (
    DependencyParser as SCDependencyParser,
)
from codeverify_core.supply_chain_verification import (
    LockfileVerifier,
    NpmDependencyParser,
    PypiDependencyParser,
    SupplyChainThreat,
    SupplyChainVerifier,
    ThreatDetector,
    ThreatType,
)
from codeverify_core.supply_chain_verification import (
    PackageEcosystem as SCPackageEcosystem,
)
from codeverify_core.supply_chain_verification import (
    PackageInfo as SCPackageInfo,
)

# Telemetry & ROI Analytics (v0.5.0)
from codeverify_core.telemetry import (
    CostEstimator as TelemetryCostEstimator,
)
from codeverify_core.telemetry import (
    FindingLifecycle,
    FindingMetrics,
    ROIReport,
    TelemetryCollector,
    TelemetryEvent,
)
from codeverify_core.telemetry import (
    MetricType as TelemetryMetricType,
)
from codeverify_core.telemetry import (
    ROIDashboard as TelemetryROIDashboard,
)

# Universal Git Support
from codeverify_core.universal_git import (
    AirGappedConfig,
    AirGappedVerifier,
    AzureDevOpsAdapter,
    GenericGitAdapter,
    GerritAdapter,
    GitCredentials,
    GiteaAdapter,
    GitHubAdapter,
    GitLabAdapter,
    GitProvider,
    GitProviderAdapter,
    LocalGitOperations,
    UniversalGitSupport,
    WebhookEventType,
    WebhookPayload,
    WebhookReceiver,
    cli_verify,
)
from codeverify_core.universal_git import (
    PullRequest as GitPullRequest,
)
from codeverify_core.universal_git import (
    Repository as GitRepository,
)

# Next-Gen: Verified Auto-Fix Validation Engine (v0.6.0)
from codeverify_core.autofix_validation import (
    BatchFixProcessor,
    BatchFixResult,
    BatchFixStrategy,
    FixValidationConfig,
    FixValidationResult,
    FixValidationStatus,
    FixValidator,
    PRDescription,
    PRDescriptionGenerator,
    RegressionChecker,
    RegressionResult,
    RegressionType,
)

# Next-Gen: Compliance-as-Code Framework (v0.6.0)
from codeverify_core.compliance_as_code import (
    AttestationEngine,
    AttestationLevel,
    ComplianceAsCodeEngine,
    ComplianceAttestation,
)
from codeverify_core.compliance_as_code import (
    ComplianceReport as CACComplianceReport,
)
from codeverify_core.compliance_as_code import (
    ComplianceReportGenerator,
    ControlAssessment,
    ControlCategory,
)
from codeverify_core.compliance_as_code import (
    EvidenceArtifact as CACEvidenceArtifact,
)
from codeverify_core.compliance_as_code import (
    EvidenceType,
)
from codeverify_core.compliance_as_code import (
    EvidenceVault as CACEvidenceVault,
)
from codeverify_core.compliance_as_code import (
    ComplianceFrameworkType,
    FrameworkControl,
    FrameworkMapper,
)

# Next-Gen: Cross-Language Contract Verification (v0.6.0)
from codeverify_core.cross_language_contracts import (
    ContractEndpoint,
    ContractExtractor,
)
from codeverify_core.cross_language_contracts import (
    ContractLanguage as CLContractLanguage,
)
from codeverify_core.cross_language_contracts import (
    ContractViolation,
    ContractViolationType,
    CrossLanguageContractReport,
    CrossLanguageVerifier,
    TypeCompatibility,
    TypeMapper,
    TypeMapping,
    UniversalType,
    VerificationScope,
)

# Next-Gen: Multi-Repository Impact Analysis (v0.6.0)
from codeverify_core.multi_repo_impact import (
    BlastRadiusCalculator,
)
from codeverify_core.multi_repo_impact import (
    BlastRadiusResult as MultiRepoBlastRadiusResult,
)
from codeverify_core.multi_repo_impact import (
    ChangeScope,
    IndexStatus,
    MigrationPhase,
    MigrationPlan,
    MigrationPlanner,
    MultiRepoImpactAnalyzer,
    MultiRepoImpactReport,
    NotificationUrgency,
    OrgDependencyGraph,
    OrgRepositoryIndexer,
    RepositoryIndex,
)
from codeverify_core.multi_repo_impact import (
    TeamNotification as MultiRepoTeamNotification,
)
from codeverify_core.multi_repo_impact import (
    TeamNotifier as MultiRepoTeamNotifier,
)

# Next-Gen: Continuous Learning from Production (v0.6.0)
from codeverify_core.production_learning import (
    ABTest,
    ABTestManager,
    ABTestStatus,
    DetectionThreshold,
    IncidentCollector,
    IncidentCorrelation,
    IncidentSeverity,
    LearnedPattern,
    LearningReport,
    LearningStrategy,
    PatternExtractor,
    ProductionIncident,
    ProductionLearningEngine,
    RuleUpdate,
    RuleUpdateAction,
    ThresholdTuner,
)

# Next-Gen: Proof Marketplace V2 (v0.6.0)
from codeverify_core.proof_marketplace_v2 import (
    ContributionType,
    ContributorProfile,
    GamificationEngine,
    ProofCategory,
    ProofContent,
    ProofMarketplaceV2,
    ProofMetadata,
    ProofQualityManager,
    ProofQualityMetrics,
    ProofReview,
    ProofSearchEngine,
    ProofStorage,
    QualityTier,
    SearchQuery,
    SearchResult,
    SearchSortBy,
)
from codeverify_core.proof_marketplace_v2 import (
    LeaderboardEntry as MarketplaceLeaderboardEntry,
)

# Next-Gen: Real-Time Pair Programming (v0.6.0)
from codeverify_core.realtime_pair_programming import (
    AnalysisScope,
    CodeChangeEvent,
    CodeLensAnnotation,
    DebounceConfig,
    FeedbackAction,
    IncrementalAnalysisResult,
    IncrementalAnalyzer,
    InlineSuggestion,
    PersonalizationEngine,
    RealTimePairSession,
    SessionMetrics,
    SmartDebouncer,
    SuggestionEngine,
    SuggestionPriority,
    SuggestionType,
    UserPreferences,
)

# Next-Gen: Refactoring Engine (v0.6.0)
from codeverify_core.refactoring_engine import (
    CodeSmell,
    CodeSmellDetector,
    ComplexityAnalyzer,
    ComplexityMetrics,
    RefactoringEngine,
    RefactoringPlan,
    RefactoringPlanner,
    RefactoringReport,
    RefactoringRisk,
    RefactoringStatus,
    RefactoringStep,
    RefactoringType,
    SmellType,
)

# Next-Gen: Supply Chain Risk Scoring (v0.6.0)
from codeverify_core.supply_chain_risk import (
    CVECorrelator,
    CVERecord,
    CVESeverity,
    DependencyRiskProfile,
    ExploitMaturity,
    RemediationLevel,
    RiskCategory,
)
from codeverify_core.supply_chain_risk import (
    RiskScorer as SCRiskScorer,
)
from codeverify_core.supply_chain_risk import (
    SBOMComponent,
    SBOMDocument,
    SBOMFormat,
    SBOMGenerator,
    SupplyChainRiskAnalyzer,
    SupplyChainRiskReport,
)

# Next-Gen: Verification Performance Profiler (v0.6.0)
from codeverify_core.verification_profiler import (
    BottleneckDetector,
    BottleneckInfo,
    BottleneckType,
    BudgetAllocator,
    BudgetAllocation,
    FunctionProfile,
    OptimizationAdvisor,
    OptimizationRecommendation,
    OptimizationStrategy,
    PerformanceTrend,
    ProfileReport,
    ProfileStage,
    StageProfile,
    VerificationInstrumenter,
    VerificationProfiler,
)

# Verified Autofix Pipeline (v0.5.0)
from codeverify_core.verified_autofix import (
    AutofixConfig,
    AutofixPipeline,
    CodeIssue,
    DifferentialVerifier,
    FixAttemptStatus,
    FixCache,
    FixGenerator,
    GeneratedPatch,
    get_autofix_pipeline,
    reset_autofix_pipeline,
)
from codeverify_core.verified_autofix import (
    VerificationProof as AutofixVerificationProof,
)
from codeverify_core.verified_autofix import (
    VerifiedFix as AutofixVerifiedFix,
)

# Next-Gen Feature: Smart Contract Verification (v0.4.0)
from codeverify_core.smart_contract_verification import (
    AuditReport as SmartContractAuditReport,
    ComplianceResult as ERCComplianceResult,
    ContractFinding,
    ContractLanguage,
    ERCStandard,
    GasReport,
    RustSolanaAnalyzer,
    SmartContractVerifier,
    SolidityAnalyzer,
    VulnerabilityCategory,
)

# Next-Gen Feature: Enterprise Compliance Framework (v0.4.0)
from codeverify_core.compliance_framework import (
    AuditReport as ComplianceAuditReport,
    ComplianceControl,
    ComplianceException,
    ComplianceFramework,
    ComplianceStandard,
    ControlStatus,
    EvidenceArtifact,
    ExceptionStatus,
    Role as ComplianceRole,
    ROLE_PERMISSIONS,
    STANDARD_CONTROLS,
)

# Next-Gen Feature: Collaboration Sessions with Live Trust (v0.4.0)
from codeverify_core.collaboration_sessions import (
    CollaborationSession as LiveCollaborationSession,
    ConflictAlert,
    FileState as CollaborationFileState,
    LiveTrustScore,
    Participant as CollaborationParticipant,
    SessionManager as CollaborationSessionManager,
    SessionRole,
    SessionState,
    VerificationMode as CollaborationVerificationMode,
)

# Next-Gen Feature: Blast Radius Analysis (v0.4.0)
from codeverify_core.blast_radius import (
    AffectedService,
    BlastRadiusAnalyzer,
    BlastRadiusReport,
    ChangeType,
    DependencyGraph as BlastRadiusDependencyGraph,
    ImpactSeverity,
    TeamNotification,
)

# Next-Gen Feature: NL Conversational Verification Queries (v0.4.0)
from codeverify_core.nl_conversation import (
    AnswerConfidence,
    ConversationContext,
    ConversationSession,
    ConversationTurn,
    IntentClassifier,
    QueryIntent,
    ResponseGenerator,
)

# Next-Gen Feature: Proof Marketplace (v0.4.0)
from codeverify_core.proof_marketplace import (
    AuthorProfile,
    BadgeType,
    LeaderboardEntry,
    MarketplaceProof,
    PricingTier,
    ProofLicense,
    ProofMarketplace,
    Purchase,
)

# =============================================================================
# Next-Gen v0.7.0 Imports
# =============================================================================

from codeverify_core.ai_code_firewall import (
    SuggestionSource,
    FirewallAction,
    RiskLevel,
    SanitizationType,
    SuggestionInterception,
    FirewallPolicy,
    RiskAssessment,
    SanitizationAction,
    FirewallDecision,
    FirewallMetrics,
    SuggestionRiskAnalyzer,
    CodeSanitizer,
    AICodeFirewall,
)

from codeverify_core.saas_billing import (
    PlanType,
    BillingCycle,
    PaymentStatus,
    SubscriptionStatus,
    SSOProvider,
    UsageMetric,
    PricingPlan,
    Subscription,
    UsageRecord,
    Invoice,
    SSOConfig,
    BillingReport,
    PlanCatalog,
    SubscriptionManager,
    UsageMeter,
    InvoiceGenerator,
    SSOManager,
    SaaSBillingEngine,
)

from codeverify_core.proof_service_api import (
    ProofRequestStatus,
    VerificationCheck,
    ProofFormat,
    PricingModel,
    ProofRequest,
    ProofResult,
    APIKeyConfig,
    UsageBucket,
    PricingConfig,
    RateLimiter,
    APIKeyManager,
    UsageTracker,
    ProofRequestProcessor,
    ProofServiceAPI,
)

from codeverify_core.smart_contract_analyzer import (
    AnalysisDepth,
    ContractStandard,
    ProofStatus as SCAProofStatus,
    GasOptimization,
    ContractFunction,
    VulnerabilityFinding,
    FormalProof,
    GasAnalysis,
    StandardComplianceResult,
    SmartContractReport,
    SolidityParser,
    VulnerabilityDetector,
    FormalVerificationEngine,
    GasAnalyzer,
    ERCComplianceChecker,
    SmartContractAnalyzer,
)

from codeverify_core.model_fine_tuning import (
    TrainingStatus,
    ModelType,
    DatasetSplit,
    AdapterType,
    ModelVersion,
    TrainingExample,
    TrainingConfig,
    TrainingJob,
    ModelArtifact,
    EvaluationResult,
    ModelComparison,
    DatasetBuilder,
    TrainingOrchestrator,
    ModelRegistry,
    ModelEvaluator,
    FineTuningPipeline,
)

from codeverify_core.dependency_visualizer import (
    NodeType,
    EdgeType,
    LayoutAlgorithm,
    ExportFormat,
    GraphNode,
    GraphEdge,
    DependencyGraph as VisDependencyGraph,
    GraphQuery,
    ImpactPath,
    ClusterInfo,
    DependencyReport,
    GraphBuilder,
    GraphAnalyzer,
    GraphQueryEngine,
    GraphExporter,
    DependencyVisualizer,
)

from codeverify_core.autofix_test_generation import (
    TestType,
    TestFramework,
    FixConfidence,
    CoverageLevel,
    TestCase,
    FixWithTests,
    TestGenerationConfig,
    TestSuite,
    TestRunResult,
    FixValidationReport,
    TestGenerator,
    TestRunner,
    CoverageAnalyzer,
    AutoFixTestEngine,
)

from codeverify_core.ci_verification_agent import (
    CIProvider,
    PipelineStage,
    GateDecision,
    VerificationScope as CIVerificationScope,
    ChangeCategory,
    CIConfig,
    FileChange,
    ChangeImpact,
    PipelineRun,
    GatePolicy,
    PipelineReport,
    ProofCache,
    ChangeDetector,
    ProofCacheManager,
    PipelineOrchestrator,
    GateEvaluator,
    CIConfigGenerator,
    CIVerificationAgent,
)

from codeverify_core.compliance_dashboard import (
    ReportFormat,
    ComplianceStatus,
    ControlPriority,
    TrendDirection,
    AuditType,
    ControlStatus as CDControlStatus,
    ComplianceScore,
    DashboardWidget,
    DashboardView,
    AuditRecord,
    RemediationItem,
    ComplianceReport as CDComplianceReport,
    ComplianceScorer,
    DashboardBuilder,
    ReportGenerator,
    AuditManager,
    RemediationTracker,
    ComplianceDashboard,
)

from codeverify_core.marketplace_community import (
    ReviewStatus,
    ContributorRole,
    ReputationTier,
    VoteType,
    ChallengeType,
    AwardType,
    CommunityMember,
    ProofSubmission,
    ReviewComment,
    Vote,
    Challenge,
    Award,
    LeaderboardEntry as MCLeaderboardEntry,
    CommunityStats,
    ReputationEngine,
    ReviewWorkflow,
    VotingSystem,
    ChallengeManager,
    AwardSystem,
    MarketplaceCommunity,
)

# =============================================================================
# Next-Gen v0.8.0 Imports
# =============================================================================

# Feature 1: Language Expansion Engine (pluggable adapter framework)
from codeverify_core.language_adapter import (
    AdapterResult,
    GoAdapter,
    JavaAdapter,
    LanguageAdapter,
    LanguageAdapterRegistry,
    PythonAdapter,
    RustAdapter,
    TypeScriptAdapter,
    get_adapter_registry,
    reset_adapter_registry,
)

# Feature 2: Zero-Config Cloud SaaS
from codeverify_core.cloud_saas import (
    AuthProvider,
    OAuthManager,
    OAuthToken,
    Tenant,
    TenantLimits,
    TenantManager,
    TenantStatus,
    TenantTier,
    TenantUser,
    get_tenant_manager,
    reset_tenant_manager,
)
from codeverify_core.cloud_saas import (
    UsageRecord as CloudUsageRecord,
)

# Feature 5: Interactive Proof Explorer
from codeverify_core.proof_explorer import (
    Counterexample,
    ProofExplorer,
    ProofOutcome,
    ProofStep,
    ProofStepType,
    ProofTrace,
    ProofTraceSerializer,
    VariableBinding,
    VisualizationFormat,
    get_proof_explorer,
    reset_proof_explorer,
)

# Feature 6: Auto-Fix Pipeline with Verification Loop
from codeverify_core.autofix_loop import (
    AutoFixPipeline,
    FixAttempt,
    FixGenerator,
    FixVerifier,
    VerifiedFix,
    get_auto_fix_pipeline,
    reset_auto_fix_pipeline,
)
from codeverify_core.autofix_loop import (
    Finding as AutoFixLoopFinding,
)
from codeverify_core.autofix_loop import (
    FixConfidence as AutoFixLoopConfidence,
)
from codeverify_core.autofix_loop import (
    FixStatus as AutoFixLoopStatus,
)

# Feature 4: LLM Cost Optimizer with Local Models
from codeverify_core.llm_cost_optimizer import (
    CheckComplexity,
    ComplexityClassifier,
    CostOptimizer,
    CostRecord,
    ModelEndpoint,
    RoutingDecision,
    RoutingStrategy,
    get_cost_optimizer,
    reset_cost_optimizer,
)
from codeverify_core.llm_cost_optimizer import (
    CostBudget as LLMCostBudget,
)
from codeverify_core.llm_cost_optimizer import (
    CostReport as LLMCostReport,
)
from codeverify_core.llm_cost_optimizer import (
    ModelProvider as LLMModelProvider,
)

# Feature 7: CI/CD Native Actions & Plugins
from codeverify_core.cicd_actions import (
    CICDActionRunner,
    CIConfigGenerator as CICDConfigGenerator,
    CIPlatform,
    GateResult,
    QualityGateConfig,
    QualityGateEvaluator,
    SARIFReport,
    SARIFResult,
)

# Feature 8: Organization Intelligence Dashboard
from codeverify_core.org_dashboard import (
    MetricsAggregator,
    OrgDashboard,
    OrgDashboardData,
    RepoMetrics,
    RiskHeatmapEntry,
    TeamMetrics,
    TrendPoint,
)
from codeverify_core.org_dashboard import (
    MetricPeriod as OrgMetricPeriod,
)
from codeverify_core.org_dashboard import (
    RiskLevel as OrgRiskLevel,
)
from codeverify_core.org_dashboard import (
    ROIMetrics as OrgROIMetrics,
)

# Feature 9: Verification-Aware Code Generation
from codeverify_core.verification_codegen import (
    AssertionGenerator,
    AssertionType,
    ContextExtractor,
    GenerationContext,
    InlineAssertion,
    SuggestionRank,
    SuggestionRanker,
    VerificationAwareCodeGen,
    VerifiedSuggestion,
)

# Feature 10: Plugin Marketplace & SDK
from codeverify_core.plugin_marketplace import (
    PluginEntry,
    PluginManifest,
    PluginRegistry,
    PluginReview,
    PluginSDK,
    PluginSearchResult,
    PluginStatus,
    PluginType,
    get_plugin_registry,
    reset_plugin_registry,
)

from codeverify_core.redis_backends import (
    RedisCostStore,
    RedisPluginStore,
    RedisTenantStore,
)

# v0.9.0 modules
from codeverify_core.verification_completion import (
    CompletionCandidate,
    CompletionMiddleware,
    CompletionSource,
    CompletionVerifier,
    CompletionVerifierStats,
    VerificationRule as CompletionVerificationRule,
    VerificationStatus as CompletionVerificationStatus,
    VerifiedCompletion,
    get_completion_verifier,
    reset_completion_verifier,
)

from codeverify_core.multi_repo_graph import (
    BlastRadius,
    BreakingChange,
    ChangeType,
    CompatibilityResult,
    ContractCompatibilityChecker,
    ContractExtractor,
    ContractField,
    ContractType,
    Endpoint as ContractEndpoint,
    FieldType,
    ServiceContract,
    ServiceDependency,
    VerificationGraph,
    get_verification_graph,
    reset_verification_graph,
)

from codeverify_core.test_generation import (
    CodePath,
    GeneratedTest,
    MutantStatus,
    MutationReport,
    MutationTester,
    PathCondition,
    PathConditionType,
    SymbolicPathDiscoverer,
    TestFramework,
    TestInput,
    TestInputGenerator,
    TestSuiteGenerator,
    get_test_generator,
    reset_test_generator,
)

from codeverify_core.adversarial_testing import (
    AdversarialReport,
    AdversarialTester,
    AttackCategory,
    AttackVector,
    AttackVectorGenerator,
    Exploit,
    ExploitSeverity,
    FuzzStrategy,
    VulnerabilityScanner,
    get_adversarial_tester,
    reset_adversarial_tester,
)

from codeverify_core.live_review import (
    AssertionSource,
    AssertionStatus,
    CompiledAssertion,
    FindingConsensus,
    LiveReviewManager,
    LiveReviewSession,
    NLAssertion,
    NLAssertionCompiler,
    ReviewVote,
    VoteType,
    get_live_review_manager,
    reset_live_review_manager,
)

from codeverify_core.audit_trail import (
    AICodeDetector,
    AIFingerprint,
    ComplianceFramework,
    ComplianceReport,
    ProvenanceRecord,
    ProvenanceSource,
    ProvenanceTracker,
    RiskLevel as AuditRiskLevel,
    get_provenance_tracker,
    reset_provenance_tracker,
)

from codeverify_core.explainable_reports import (
    CounterExample,
    ExplainableReport,
    ExplainedFinding,
    ExplanationLevel,
    FindingExplainer,
    VerificationFinding,
    VerificationOutcome,
    get_finding_explainer,
    reset_finding_explainer,
)

from codeverify_core.clone_detector import (
    CloneCluster,
    CloneDetector,
    ClonePair,
    CloneType,
    DeduplicationReport,
    FunctionExtractor,
    FunctionSignature,
    RefactoringSuggestion,
    RefactoringStatus,
    get_clone_detector,
    reset_clone_detector,
)

from codeverify_core.verification_as_code import (
    DriftReport,
    DriftStatus,
    PolicyDSLParser,
    PolicyEngine,
    PolicyModule,
    PolicyRule,
    PolicyScope,
    PolicySeverity,
    PolicyViolation,
    get_policy_engine,
    reset_policy_engine,
)

from codeverify_core.budget_marketplace import (
    CapacityLease,
    MarketOrder,
    MarketStats,
    OrderSide,
    OrderStatus,
    Trade,
    TransactionType,
    VerificationCredit,
    VerificationMarketplace,
    Wallet,
    WalletTransaction,
    get_marketplace,
    reset_marketplace,
)

# =============================================================================
# Next-Gen v1.0.0 Imports
# =============================================================================

# Feature 1: AI Code Insurance Underwriting Platform
from codeverify_core.insurance_underwriting import (
    ClaimRejectionReason,
    ClaimStatus,
    ClaimValidator,
    CoverageType,
    InsuranceClaim,
    InsurancePolicy,
    InsuranceUnderwriter,
    PolicyStatus,
    PremiumCalculation,
    PremiumCalculator,
    RiskTier,
    get_insurance_underwriter,
    reset_insurance_underwriter,
)
from codeverify_core.insurance_underwriting import (
    RiskProfile as InsuranceRiskProfile,
)

# Feature 2: Cross-Repository Security Graph
from codeverify_core.cross_repo_security_graph import (
    BlastRadiusResult as SecurityBlastRadiusResult,
)
from codeverify_core.cross_repo_security_graph import (
    ScanStatus,
    SecurityEdge,
    SecurityEdgeType,
    SecurityKnowledgeGraph,
    SecurityNode,
    SecurityNodeType,
    SimilarityMatch,
    VulnSeverity,
    VulnerabilityRecord,
    get_security_graph,
    reset_security_graph,
)

# Feature 3: Differential Privacy-Preserving Proof Marketplace
from codeverify_core.privacy_preserving_proofs import (
    AnonymizedProof,
    ContributionStatus,
    FederatedAggregator,
    FederatedUpdate,
    PrivacyBudget,
    PrivacyLevel,
    PrivacyPreservingProofMarketplace,
    ProofAnonymizer,
    get_privacy_marketplace,
    reset_privacy_marketplace,
)
from codeverify_core.privacy_preserving_proofs import (
    ProofCategory as PrivacyProofCategory,
)

# Feature 4: Blockchain-Verified Code Provenance
from codeverify_core.blockchain_provenance import (
    AttestationStatus,
    BadgeLevel,
    BlockchainAttestation,
    BlockchainProvenanceEngine,
    ChainType,
    ContentAddress,
    LocalBlockchain,
    VerificationBadge,
    get_blockchain_provenance,
    reset_blockchain_provenance,
)

# Feature 5: Natural Language Compliance Query Engine
from codeverify_core.nl_compliance_engine import (
    BUILTIN_TEMPLATES,
    ComplianceEvidence,
    ComplianceQuery,
    ComplianceQueryExecutor,
    ComplianceQueryResult,
    ComplianceTemplate,
    EvidenceStrength,
    NLQueryParser,
    QueryStatus,
    QueryType,
    get_compliance_query_engine,
    reset_compliance_query_engine,
)
from codeverify_core.nl_compliance_engine import (
    ComplianceStandard as NLComplianceStandard,
)

# Feature 6: AI Model Bias & Fairness Verification
from codeverify_core.fairness_verification import (
    BiasDetectionResult,
    BiasDetector,
    BiasLevel,
    FairnessConstraint,
    FairnessMetric,
    FairnessReport,
    FairnessVerifier,
    GroupMetrics,
    ProtectedAttribute,
    RemediationAdvisor,
    RemediationSuggestion,
    RemediationType,
    get_fairness_verifier,
    reset_fairness_verifier,
)
from codeverify_core.fairness_verification import (
    ComplianceStatus as FairnessComplianceStatus,
)

# Feature 7: IDE Copilot Undo with Proof Preservation
from codeverify_core.copilot_undo import (
    CodeDiff,
    CopilotUndoManager,
    ProofSnapshot,
    SavePoint,
    SavePointStatus,
    SavePointType,
    TrustScoreTrend,
    get_copilot_undo_manager,
    reset_copilot_undo_manager,
)
from codeverify_core.copilot_undo import (
    VerificationState as UndoVerificationState,
)

# Feature 8: Predictive Code Quality Forecasting
from codeverify_core.quality_forecasting import (
    AnomalyDetector,
    QualityAlert,
    QualityDataPoint,
    QualityForecast,
    QualityForecaster,
    ScenarioResult,
    TrendAnalysis,
    TrendCalculator,
    get_quality_forecaster,
    reset_quality_forecaster,
)
from codeverify_core.quality_forecasting import (
    AlertSeverity as ForecastAlertSeverity,
)
from codeverify_core.quality_forecasting import (
    ForecastConfidence,
)
from codeverify_core.quality_forecasting import (
    MetricType as ForecastMetricType,
)
from codeverify_core.quality_forecasting import (
    TrendDirection as ForecastTrendDirection,
)

# Feature 9: Verification-Driven Code Generation
from codeverify_core.verified_codegen import (
    CodeGenerator,
    CodeSpec,
    ConstraintChecker,
    ConstraintType,
    FormalConstraint,
    GeneratedCandidate,
    GenerationResult,
    GenerationStatus,
    SpecLanguage,
    VerifiedCodeGenerator,
    get_verified_codegen,
    reset_verified_codegen,
)
from codeverify_core.verified_codegen import (
    VerificationResult as CodegenVerificationResult,
)

# Feature 10: Real-Time Collaborative Verification Sessions
from codeverify_core.collaborative_verification_sessions import (
    CollaborativeSessionManager,
    CollaborativeVerificationSession,
    LineStatus,
    LineVerification,
    ParticipantRole,
    SessionMessage,
    SessionPhase,
    SessionRecording,
    SessionStats,
    get_collab_session_manager,
    reset_collab_session_manager,
)
from codeverify_core.collaborative_verification_sessions import (
    MessageType as CollabMessageType,
)
from codeverify_core.collaborative_verification_sessions import (
    SessionParticipant as CollabSessionParticipant,
)

# Hosted SaaS Platform (v1.1.0)
from codeverify_core.saas_platform import (
    ApiKey,
    ApiKeyScope,
    PlanLimits,
    PlanTier,
    RateLimiter,
    SaaSPlatform,
    Tenant,
    TenantStatus,
    UsageMetricType,
    UsageRecord,
    UsageSummary,
    UsageTracker,
    get_saas_platform,
    reset_saas_platform,
)

# Go + Java Language Support (v1.1.0)
from codeverify_core.go_java_support import (
    AdvancedGoParser,
    AdvancedJavaParser,
    GoJavaLanguageSupport,
    GoJavaNode,
    GoJavaNodeType,
    GoJavaParseResult,
    IdiomaticPattern,
    PatternMatch,
    get_go_java_support,
    reset_go_java_support,
)

# Autofix Agent with PR Generation (v1.1.0)
from codeverify_core.autofix_pr_agent import (
    AutofixAgent,
    FixCandidate,
    FixCategory,
    FixConfidence,
    FixGenerator,
    FixResult,
    FixStatus,
    FixVerifier,
    get_autofix_agent,
    reset_autofix_agent,
)

# GitHub Copilot Chat Extension (v1.1.0)
from codeverify_core.copilot_chat_extension import (
    ChatContext,
    ChatMessage,
    ChatResponse,
    CommandRouter,
    CopilotCommand,
    CopilotExtensionHandler,
    CopilotSession,
    ResponseFormat,
    get_copilot_extension_handler,
    reset_copilot_extension_handler,
)

# Incremental Verification Engine (v1.1.0)
from codeverify_core.incremental_verification import (
    CacheMetrics,
    CacheStatus,
    CachedResult,
    CodeUnit,
    DependencyGraph,
    IncrementalVerificationEngine,
    VerificationStatus as IncrementalVerificationStatus,
    get_incremental_engine,
    reset_incremental_engine,
)

# Organization Security Posture Dashboard (v1.1.0)
from codeverify_core.org_security_dashboard import (
    ComplianceFramework as DashboardComplianceFramework,
    ComplianceRecord,
    ComplianceStatus as DashboardComplianceStatus,
    DORAMetrics,
    OrgSecurityDashboard,
    OrgSecurityPosture,
    RepoMetrics,
    RiskLevel as DashboardRiskLevel,
    TrendDataPoint,
    TrendDirection,
    TrendSeries,
    get_org_security_dashboard,
    reset_org_security_dashboard,
)

# CI/CD Pipeline Orchestrator (v1.1.0)
from codeverify_core.cicd_orchestrator import (
    CICDOrchestrator,
    CICDPlatform,
    GateEvaluation,
    GateMode,
    GateResult,
    PipelineConfig,
    PipelineConfigGenerator,
    PipelineStatus,
    QualityGate,
    QualityGateEvaluator,
    QualityThresholds,
    StatusState,
    get_cicd_orchestrator,
    reset_cicd_orchestrator,
)

# LLM-Powered Proof Explainer (v1.1.0)
from codeverify_core.proof_explainer import (
    CheckCategory,
    CounterexampleParser,
    CounterexampleValue,
    ExplanationDetail,
    ParsedCounterexample,
    ProofExplanation,
    ProofExplainerEngine,
    ProofOutcome,
    get_proof_explainer,
    reset_proof_explainer,
)

# Supply Chain Verification (v1.1.0)
from codeverify_core.supply_chain import (
    Dependency,
    DependencyRisk,
    LicenseCategory,
    LicensePolicy,
    LockfileParser,
    PackageManager,
    SBOMEntry,
    SupplyChainReport,
    SupplyChainVerifier,
    Vulnerability,
    VulnerabilityDatabase,
    VulnerabilitySeverity,
    get_supply_chain_verifier,
    reset_supply_chain_verifier,
)
from codeverify_core.supply_chain import (
    ComplianceStatus as SupplyChainComplianceStatus,
)

# Self-Learning Rule Engine (v1.1.0)
from codeverify_core.self_learning_rules import (
    ClassificationResult,
    DismissReason,
    FalsePositiveClassifier,
    FeedbackCollector,
    FeedbackType,
    FeatureVector,
    FindingFeedback,
    LearnedPattern,
    PatternLearner,
    RulePerformance,
    SelfLearningRuleEngine,
    SeverityAdjustment,
    SeverityCalibrator,
    get_self_learning_engine,
    reset_self_learning_engine,
)

# Rust & C/C++ Memory Safety Verification (v1.2.0)
from codeverify_core.memory_safety import (
    CPointerAnalyzer,
    DataRaceCandidate,
    DataRaceDetector,
    LifetimeConstraint,
    MemoryCheckSeverity,
    MemoryLanguage,
    MemoryLocation,
    MemoryRegion,
    MemorySafetyReport,
    MemorySafetyVerifier,
    MemoryViolation,
    MemoryViolationType,
    OwnershipConstraint,
    OwnershipState,
    PointerInfo,
    PointerState,
    RustOwnershipAnalyzer,
    get_memory_safety_verifier,
    reset_memory_safety_verifier,
)

# GitHub Copilot Workspace Integration (v1.2.0)
from codeverify_core.copilot_workspace import (
    ConstraintType,
    CopilotWorkspaceIntegration,
    PlanFinding,
    PlanVerificationResult,
    VerificationConstraint,
    VerificationGate,
    WorkspaceEventType,
    WorkspaceFile,
    WorkspacePlan,
    WorkspacePlanStatus,
    WorkspaceSession,
    get_copilot_workspace_integration,
    reset_copilot_workspace_integration,
)

# Zero-Config Onboarding (v1.2.0)
from codeverify_core.zero_config import (
    BaselineScanResult,
    ConfigGenerator,
    DetectedFramework,
    DetectedLanguage,
    GeneratedConfig,
    GeneratedWorkflow,
    OnboardingResult,
    OnboardingStep,
    ProjectAnalysis,
    ProjectDetector,
    ProjectType,
    WorkflowGenerator,
    ZeroConfigOnboarder,
    get_zero_config_onboarder,
    reset_zero_config_onboarder,
)

# Autonomous Verification Agent (v1.2.0)
from codeverify_core.autonomous_agent import (
    AgentState,
    AutonomousPR,
    AutonomousVerificationAgent,
    AutonomyLevel,
    FeedbackEntry,
    FindingTriage,
    FindingTriager,
    FixCandidate,
    FixGenerator,
    FixOutcome,
    MonitoredChange,
    TriagedFinding,
    get_autonomous_agent,
    reset_autonomous_agent,
)

# Verification-as-a-Service API (v1.2.0)
from codeverify_core.vaas import (
    OutputFormat,
    TierLimits,
    VaaSApiKey,
    VaaSService,
    VerificationRequest,
    VerificationResponse,
    VerificationStatus,
    VerificationTier,
    WebhookConfig,
    WebhookEventType,
    get_vaas_service,
    reset_vaas_service,
)

# Interactive Proof Explorer (v1.2.0)
from codeverify_core.proof_explorer_interactive import (
    ConstraintAnimator,
    ConstraintStep,
    DetailLevel,
    ExportFormat,
    InteractiveProofExplorer,
    ProofAnimation,
    ProofNode,
    ProofNodeType,
    ProofRenderer,
    ProofStatus,
    ProofTreeParser,
    ShareableProof,
    get_proof_explorer,
    reset_proof_explorer,
)

# Cross-Repository Blast Radius (v1.2.0)
from codeverify_core.cross_repo_blast import (
    CrossRepoBlastAnalyzer,
    CrossRepoBlastReport,
    CrossRepoChange,
    CrossRepoChangeImpact,
    CrossRepoDependency,
    CrossRepoDependencyType,
    CrossRepoImpactLevel,
    CrossRepoImpactedRepo,
    OrgDependencyGraph,
    OrgRepository,
    get_cross_repo_blast_analyzer,
    reset_cross_repo_blast_analyzer,
)

# AI Code Review Benchmark Suite (v1.2.0)
from codeverify_core.benchmark import (
    BenchmarkDataset,
    BenchmarkMetrics,
    BenchmarkRunner,
    BenchmarkSample,
    BugCategory,
    BuiltinBenchmarkAdapter,
    CategoryMetrics,
    LeaderboardEntry,
    SampleDifficulty,
    SampleLanguage,
    ToolDetection,
    get_benchmark_runner,
    reset_benchmark_runner,
)

# Fine-Tuned Verification LLM (v1.2.0)
from codeverify_core.fine_tuned_llm import (
    AirGapPackage,
    AirGapPackager,
    CostComparison,
    FineTunedVerificationLLM,
    InferenceConfig,
    InferenceResult,
    LocalInferenceEngine,
    ModelConfig,
    ModelFormat,
    ModelSize,
    TaskType,
    TrainingDataPipeline,
    TrainingDataset,
    TrainingJob,
    TrainingMetrics,
    TrainingSample,
    TrainingStatus,
    get_fine_tuned_llm,
    reset_fine_tuned_llm,
)

# Developer Certification Program (v1.2.0)
from codeverify_core.certification import (
    Assessment,
    AssessmentGrader,
    AssessmentQuestion,
    AssessmentSubmission,
    AssessmentType,
    BadgeType,
    Certificate,
    CertificationLevel,
    CertificationProgram,
    CourseBuilder,
    CourseModule,
    CredentialIssuer,
    DigitalBadge,
    LabExercise,
    LabStatus,
    LearnerProgress,
    ModuleStatus,
    get_certification_program,
    reset_certification_program,
)

__all__ = [
    # Models
    "Analysis",
    "AnalysisStatus",
    "Finding",
    "FindingCategory",
    "FindingSeverity",
    "VerificationType",
    # Severity utilities (from severity module)
    "SEVERITY_ORDER",
    "SEVERITY_EMOJI",
    "SEVERITY_LABELS",
    "parse_severity",
    "compare_severity",
    "is_blocking_severity",
    "is_above_threshold",
    "get_severity_emoji",
    "get_severity_label",
    "sort_by_severity",
    # Timestamp utilities
    "TimestampMixin",
    "DataclassTimestampMixin",
    "parse_iso_datetime",
    # Result type pattern
    "Result",
    "StrResult",
    "BoolResult",
    "OperationResult",
    # Rules
    "CustomRule",
    "RuleType",
    "RuleEvaluator",
    "RuleBuilder",
    "RuleViolation",
    "get_builtin_rules",
    # Rule Evaluation Strategies
    "RuleEvaluationStrategy",
    "PatternRuleStrategy",
    "CompositeRuleStrategy",
    "ASTRuleStrategy",
    "SemanticRuleStrategy",
    # Scanning
    "ScanConfiguration",
    "CodebaseScanResult",
    # Notifications
    "SlackFormatter",
    "TeamsFormatter",
    "NotificationSender",
    # Sub-function Analysis
    "GranularityLevel",
    "IncrementalAnalysisEngine",
    "Position",
    "RealTimeFeedbackModel",
    "SemanticBlock",
    "SemanticBlockType",
    "Span",
    "SubFunctionParser",
    "SymbolDefinition",
    "SymbolReference",
    # Copilot Sessions
    "CopilotReviewSession",
    "CopilotSessionPool",
    "RealTimeCopilotReviewer",
    "SessionContext",
    "SessionResult",
    "SessionState",
    "StreamingFinding",
    "copilot_reviewer",
    # Memory Graph
    "ConstraintKind",
    "CrossProjectLearner",
    "GraphEdgeType",
    "GraphNodeType",
    "InMemoryProofStorage",
    "PatternFingerprint",
    "ProofArtifact",
    "ProofStatus",
    "ProofStorageBackend",
    "ProofType",
    "SerializedConstraint",
    "VerificationKnowledgeGraph",
    "compute_code_hash",
    "compute_pattern_hash",
    "cross_project_learner",
    "extract_pattern_fingerprint",
    "verification_memory_graph",
    # Proof Repository (Feature 7)
    "ProofArtifactRepository",
    "ProofCategory",
    "ProofRepoStatus",
    "ProofStorage",
    "ProofTemplate",
    "SearchQuery",
    "SearchResult",
    "InMemoryProofRepo",
    # Cost Optimizer (Feature 9)
    "BudgetConstraints",
    "CostMetrics",
    "RiskProfile",
    "VerificationCost",
    "VerificationCostOptimizer",
    "VerificationDepth",
    "VerificationPlan",
    # Next-Gen Feature 1: Monorepo Intelligence
    "DependencyGraph",
    "MonorepoAnalyzer",
    "MonorepoType",
    "Package",
    "PackageDependency",
    # Next-Gen Feature 3: Proof-Carrying PRs
    "ProofAttestation",
    "ProofCarryingManager",
    "ProofSerializer",
    "VerificationProof",
    # Next-Gen Feature 7: Verification Budget Optimizer
    "BatchOptimizationResult",
    "Budget",
    "BudgetVerificationDepth",
    "CostEstimator",
    "CostModel",
    "DepthSelector",
    "OutcomeLearner",
    "RiskFactors",
    "RiskScorer",
    "VerificationBudgetOptimizer",
    "VerificationDecision",
    # Next-Gen Feature 10: Gradual Verification Ramp
    "BaselineCollector",
    "BaselineMetrics",
    "EnforcementDecision",
    "EnforcementLevel",
    "GradualVerificationRamp",
    "RampPhase",
    "RampProgress",
    "RampSchedule",
    "RampState",
    # Repository Pattern
    "Repository",
    "InMemoryRepository",
    "ScanResultRepository",
    "ScheduledScanRepository",
    "NotificationConfigRepository",
    "InMemoryScanResultRepository",
    "InMemoryScheduledScanRepository",
    "InMemoryNotificationConfigRepository",
    "get_scan_result_repository",
    "get_scheduled_scan_repository",
    "get_notification_config_repository",
    "set_scan_result_repository",
    "set_scheduled_scan_repository",
    "set_notification_config_repository",
    # Compliance Evidence Vault
    "ComplianceFramework",
    "ComplianceReportGenerator",
    "EvidenceType",
    "EvidenceVault",
    "StoredEvidence",
    "get_evidence_vault",
    "reset_evidence_vault",
    # Organization Dependencies
    "OrgDependencyAnalyzer",
    "OrgDependencyGraph",
    "OrgRepository",
    "OrgDependencyEdge",
    "TransitiveRisk",
    "get_org_dependency_analyzer",
    # Offline Mode
    "LocalModelConfig",
    "LocalModelType",
    "LocalZ3Verifier",
    "OfflineAnalysisResult",
    "OfflineCapability",
    "OfflineModeConfig",
    "OfflineModeManager",
    "OllamaClient",
    "get_offline_manager",
    "reset_offline_manager",
    # Natural Language Bug Queries
    "BugCategory",
    "FindingsIndex",
    "NLQueryEngine",
    "QueryIntent",
    "QueryParser",
    "QueryResponse",
    "NLSearchResult",
    "SemanticQuery",
    "get_nl_query_engine",
    "reset_nl_query_engine",
    # Proof Coverage Dashboard
    "CoverageTrend",
    "DashboardData",
    "FileCoverage",
    "FunctionCoverage",
    "LineCoverage",
    "ProofCoverageCalculator",
    "ProofCoverageDashboard",
    "ProofCoverageStatus",
    "RepositoryCoverage",
    "VerificationCategory",
    "get_proof_coverage_dashboard",
    "reset_proof_coverage_dashboard",
    # SBOM + SLSA Provenance
    "Component",
    "ComponentHash",
    "ComponentType",
    "ExternalReference",
    "LicenseType",
    "SBOM",
    "SBOMFormat",
    "SBOMGenerator",
    "SLSAAttestationGenerator",
    "SLSALevel",
    "SLSAProvenance",
    "VerificationAttestation",
    "VerifiedSBOMExporter",
    "SBOMVulnerability",
    # ROI Dashboard & Cost Transparency
    "BUG_COST_ESTIMATES",
    "BugCaught",
    "BugSeverity",
    "BugValueCalculator",
    "CostCategory",
    "CostConfig",
    "CostTracker",
    "ROIDashboard",
    "ROIMetrics",
    "ROIVerificationCost",
    "create_dashboard",
    # Universal Git Support
    "AirGappedConfig",
    "AirGappedVerifier",
    "AzureDevOpsAdapter",
    "GenericGitAdapter",
    "GerritAdapter",
    "GitCredentials",
    "GiteaAdapter",
    "GitHubAdapter",
    "GitLabAdapter",
    "GitProvider",
    "GitProviderAdapter",
    "GitPullRequest",
    "GitRepository",
    "LocalGitOperations",
    "UniversalGitSupport",
    "WebhookEventType",
    "WebhookPayload",
    "WebhookReceiver",
    "cli_verify",
    # Streaming Verification API (v0.5.0)
    "IncrementalDiff",
    "SessionStatus",
    "StreamEvent",
    "StreamEventType",
    "StreamingSessionConfig",
    "StreamingSessionPool",
    "StreamingVerificationSession",
    "StreamingStage",
    "get_streaming_pool",
    "reset_streaming_pool",
    # Verified Autofix Pipeline (v0.5.0)
    "AutofixConfig",
    "AutofixPipeline",
    "CodeIssue",
    "DifferentialVerifier",
    "FixAttemptStatus",
    "FixCache",
    "FixGenerator",
    "GeneratedPatch",
    "AutofixVerifiedFix",
    "AutofixVerificationProof",
    "get_autofix_pipeline",
    "reset_autofix_pipeline",
    # Organization Knowledge Graph (v0.5.0)
    "KGEdgeType",
    "KGGraphEdge",
    "KGGraphNode",
    "KnowledgeGraph",
    "KnowledgeGraphConfig",
    "KnowledgeIngester",
    "KGNodeType",
    "ProofReuseEngine",
    "get_knowledge_graph",
    "reset_knowledge_graph",
    # Go + Java Language Support (v0.5.0)
    "LangConfig",
    "LanguageFeature",
    "LanguageParser",
    "LanguageRule",
    "LanguageRuleRegistry",
    "SupportedLanguage",
    "Z3ConstraintGenerator",
    "detect_language",
    "get_language_registry",
    "reset_language_registry",
    # Supply Chain Verification (v0.5.0)
    "SCDependencyParser",
    "LockfileVerifier",
    "NpmDependencyParser",
    "SCPackageEcosystem",
    "SCPackageInfo",
    "PypiDependencyParser",
    "SupplyChainThreat",
    "SupplyChainVerifier",
    "ThreatDetector",
    "ThreatType",
    # Copilot Chat Integration (v0.5.0)
    "CodeSuggestion",
    "CommandParser",
    "CopilotChatParticipant",
    "CopilotCommand",
    "CopilotContext",
    "CopilotMessage",
    "CopilotMessageRole",
    "CopilotResponse",
    "CopilotWebhookHandler",
    # Policy Engine / CI-CD Gates (v0.5.0)
    "PolicyAction",
    "PolicyCondition",
    "PolicyEngine",
    "PolicyEvaluationResult",
    "PolicyRule",
    "PolicyScope",
    "PolicySet",
    "get_default_policy_set",
    "parse_policy_yaml",
    # Telemetry & ROI Analytics (v0.5.0)
    "TelemetryCostEstimator",
    "FindingLifecycle",
    "FindingMetrics",
    "TelemetryMetricType",
    "TelemetryROIDashboard",
    "ROIReport",
    "TelemetryCollector",
    "TelemetryEvent",
    # Plugin Marketplace & Agent SDK (v0.5.0)
    "AgentCapability",
    "AgentCategory",
    "AgentLanguage",
    "AgentManifest",
    "AgentPackage",
    "AgentLifecycle",
    "SDKAnalysisContext",
    "SDKAnalysisResult",
    "SDKBaseAgent",
    "SDKFinding",
    "SDKSeverityLevel",
    "agent_decorator",
    "AgentLoadError",
    "AgentSandbox",
    "IsolatedAgentRunner",
    "ResourceLimitExceeded",
    "SandboxConfig",
    "SandboxError",
    "SecurityViolation",
    "run_agent",
    # Smart Contract Verification
    "SmartContractVerifier",
    "SmartContractAuditReport",
    "ContractFinding",
    "ContractLanguage",
    "ERCStandard",
    "ERCComplianceResult",
    "GasReport",
    "SolidityAnalyzer",
    "RustSolanaAnalyzer",
    "VulnerabilityCategory",
    # Enterprise Compliance Framework
    "ComplianceFramework",
    "ComplianceStandard",
    "ComplianceControl",
    "ComplianceException",
    "ComplianceAuditReport",
    "ComplianceRole",
    "ControlStatus",
    "ExceptionStatus",
    "EvidenceArtifact",
    "ROLE_PERMISSIONS",
    "STANDARD_CONTROLS",
    # Collaboration Sessions
    "LiveCollaborationSession",
    "CollaborationSessionManager",
    "CollaborationParticipant",
    "CollaborationFileState",
    "CollaborationVerificationMode",
    "ConflictAlert",
    "LiveTrustScore",
    "SessionRole",
    "SessionState",
    # Blast Radius Analysis
    "BlastRadiusAnalyzer",
    "BlastRadiusReport",
    "BlastRadiusDependencyGraph",
    "AffectedService",
    "ChangeType",
    "ImpactSeverity",
    "TeamNotification",
    # NL Conversational Queries
    "ConversationSession",
    "ConversationContext",
    "ConversationTurn",
    "IntentClassifier",
    "QueryIntent",
    "AnswerConfidence",
    "ResponseGenerator",
    # Proof Marketplace
    "ProofMarketplace",
    "MarketplaceProof",
    "AuthorProfile",
    "BadgeType",
    "LeaderboardEntry",
    "PricingTier",
    "ProofLicense",
    "Purchase",
    # Next-Gen: Verified Auto-Fix Validation (v0.6.0)
    "FixValidationStatus",
    "RegressionType",
    "BatchFixStrategy",
    "FixValidationConfig",
    "FixValidationResult",
    "RegressionResult",
    "BatchFixResult",
    "PRDescription",
    "FixValidator",
    "RegressionChecker",
    "BatchFixProcessor",
    "PRDescriptionGenerator",
    # Next-Gen: Continuous Learning from Production (v0.6.0)
    "IncidentSeverity",
    "LearningStrategy",
    "RuleUpdateAction",
    "ABTestStatus",
    "ProductionIncident",
    "IncidentCorrelation",
    "DetectionThreshold",
    "LearnedPattern",
    "RuleUpdate",
    "ABTest",
    "LearningReport",
    "IncidentCollector",
    "ThresholdTuner",
    "PatternExtractor",
    "ABTestManager",
    "ProductionLearningEngine",
    # Next-Gen: Real-Time Pair Programming (v0.6.0)
    "AnalysisScope",
    "SuggestionType",
    "SuggestionPriority",
    "FeedbackAction",
    "CodeChangeEvent",
    "IncrementalAnalysisResult",
    "InlineSuggestion",
    "CodeLensAnnotation",
    "UserPreferences",
    "DebounceConfig",
    "SessionMetrics",
    "IncrementalAnalyzer",
    "SmartDebouncer",
    "SuggestionEngine",
    "PersonalizationEngine",
    "RealTimePairSession",
    # Next-Gen: Cross-Language Contract Verification (v0.6.0)
    "CLContractLanguage",
    "TypeCompatibility",
    "ContractViolationType",
    "VerificationScope",
    "UniversalType",
    "ContractEndpoint",
    "ContractViolation",
    "TypeMapping",
    "CrossLanguageContractReport",
    "TypeMapper",
    "ContractExtractor",
    "CrossLanguageVerifier",
    # Next-Gen: Supply Chain Risk Scoring (v0.6.0)
    "CVESeverity",
    "ExploitMaturity",
    "RemediationLevel",
    "SBOMFormat",
    "RiskCategory",
    "CVERecord",
    "DependencyRiskProfile",
    "SBOMComponent",
    "SBOMDocument",
    "SupplyChainRiskReport",
    "CVECorrelator",
    "SCRiskScorer",
    "SBOMGenerator",
    "SupplyChainRiskAnalyzer",
    # Next-Gen: Multi-Repository Impact Analysis (v0.6.0)
    "IndexStatus",
    "ChangeScope",
    "NotificationUrgency",
    "MigrationPhase",
    "RepositoryIndex",
    "OrgDependencyGraph",
    "MultiRepoBlastRadiusResult",
    "MultiRepoTeamNotification",
    "MigrationPlan",
    "MultiRepoImpactReport",
    "OrgRepositoryIndexer",
    "BlastRadiusCalculator",
    "MultiRepoTeamNotifier",
    "MigrationPlanner",
    "MultiRepoImpactAnalyzer",
    # Next-Gen: Refactoring Engine (v0.6.0)
    "SmellType",
    "RefactoringType",
    "RefactoringRisk",
    "RefactoringStatus",
    "CodeSmell",
    "ComplexityMetrics",
    "RefactoringStep",
    "RefactoringPlan",
    "RefactoringReport",
    "CodeSmellDetector",
    "ComplexityAnalyzer",
    "RefactoringPlanner",
    "RefactoringEngine",
    # Next-Gen: Compliance-as-Code Framework (v0.6.0)
    "ComplianceFrameworkType",
    "EvidenceType",
    "AttestationLevel",
    "ControlCategory",
    "FrameworkControl",
    "CACEvidenceArtifact",
    "ComplianceAttestation",
    "ControlAssessment",
    "CACComplianceReport",
    "FrameworkMapper",
    "CACEvidenceVault",
    "AttestationEngine",
    "ComplianceReportGenerator",
    "ComplianceAsCodeEngine",
    # Next-Gen: Proof Marketplace V2 (v0.6.0)
    "ProofCategory",
    "QualityTier",
    "ContributionType",
    "SearchSortBy",
    "ProofMetadata",
    "ProofContent",
    "ProofQualityMetrics",
    "ProofReview",
    "ContributorProfile",
    "SearchQuery",
    "SearchResult",
    "MarketplaceLeaderboardEntry",
    "ProofStorage",
    "ProofSearchEngine",
    "ProofQualityManager",
    "GamificationEngine",
    "ProofMarketplaceV2",
    # Next-Gen: Verification Performance Profiler (v0.6.0)
    "ProfileStage",
    "BottleneckType",
    "OptimizationStrategy",
    "StageProfile",
    "BottleneckInfo",
    "FunctionProfile",
    "OptimizationRecommendation",
    "BudgetAllocation",
    "PerformanceTrend",
    "ProfileReport",
    "VerificationInstrumenter",
    "BottleneckDetector",
    "OptimizationAdvisor",
    "BudgetAllocator",
    "VerificationProfiler",
    # Next-Gen: AI Code Generation Firewall (v0.7.0)
    "SuggestionSource",
    "FirewallAction",
    "RiskLevel",
    "SanitizationType",
    "SuggestionInterception",
    "FirewallPolicy",
    "RiskAssessment",
    "SanitizationAction",
    "FirewallDecision",
    "FirewallMetrics",
    "SuggestionRiskAnalyzer",
    "CodeSanitizer",
    "AICodeFirewall",
    # Next-Gen: Multi-Tenant SaaS Billing (v0.7.0)
    "PlanType",
    "BillingCycle",
    "PaymentStatus",
    "SubscriptionStatus",
    "SSOProvider",
    "UsageMetric",
    "PricingPlan",
    "Subscription",
    "UsageRecord",
    "Invoice",
    "SSOConfig",
    "BillingReport",
    "PlanCatalog",
    "SubscriptionManager",
    "UsageMeter",
    "InvoiceGenerator",
    "SSOManager",
    "SaaSBillingEngine",
    # Next-Gen: Proof-as-a-Service API (v0.7.0)
    "ProofRequestStatus",
    "VerificationCheck",
    "ProofFormat",
    "PricingModel",
    "ProofRequest",
    "ProofResult",
    "APIKeyConfig",
    "UsageBucket",
    "PricingConfig",
    "RateLimiter",
    "APIKeyManager",
    "UsageTracker",
    "ProofRequestProcessor",
    "ProofServiceAPI",
    # Next-Gen: Smart Contract Analyzer (v0.7.0)
    "AnalysisDepth",
    "ContractStandard",
    "SCAProofStatus",
    "GasOptimization",
    "ContractFunction",
    "VulnerabilityFinding",
    "FormalProof",
    "GasAnalysis",
    "StandardComplianceResult",
    "SmartContractReport",
    "SolidityParser",
    "VulnerabilityDetector",
    "FormalVerificationEngine",
    "GasAnalyzer",
    "ERCComplianceChecker",
    "SmartContractAnalyzer",
    # Next-Gen: AI Model Fine-Tuning Pipeline (v0.7.0)
    "TrainingStatus",
    "ModelType",
    "DatasetSplit",
    "AdapterType",
    "ModelVersion",
    "TrainingExample",
    "TrainingConfig",
    "TrainingJob",
    "ModelArtifact",
    "EvaluationResult",
    "ModelComparison",
    "DatasetBuilder",
    "TrainingOrchestrator",
    "ModelRegistry",
    "ModelEvaluator",
    "FineTuningPipeline",
    # Next-Gen: Cross-Repo Dependency Visualizer (v0.7.0)
    "NodeType",
    "EdgeType",
    "LayoutAlgorithm",
    "ExportFormat",
    "GraphNode",
    "GraphEdge",
    "VisDependencyGraph",
    "GraphQuery",
    "ImpactPath",
    "ClusterInfo",
    "DependencyReport",
    "GraphBuilder",
    "GraphAnalyzer",
    "GraphQueryEngine",
    "GraphExporter",
    "DependencyVisualizer",
    # Next-Gen: Auto-Fix with Test Generation (v0.7.0)
    "TestType",
    "TestFramework",
    "FixConfidence",
    "CoverageLevel",
    "TestCase",
    "FixWithTests",
    "TestGenerationConfig",
    "TestSuite",
    "TestRunResult",
    "FixValidationReport",
    "TestGenerator",
    "TestRunner",
    "CoverageAnalyzer",
    "AutoFixTestEngine",
    # Next-Gen: CI/CD Verification Agent (v0.7.0)
    "CIProvider",
    "PipelineStage",
    "GateDecision",
    "CIVerificationScope",
    "ChangeCategory",
    "CIConfig",
    "FileChange",
    "ChangeImpact",
    "PipelineRun",
    "GatePolicy",
    "PipelineReport",
    "ProofCache",
    "ChangeDetector",
    "ProofCacheManager",
    "PipelineOrchestrator",
    "GateEvaluator",
    "CIConfigGenerator",
    "CIVerificationAgent",
    # Next-Gen: Compliance Dashboard & Reporter (v0.7.0)
    "ReportFormat",
    "ComplianceStatus",
    "ControlPriority",
    "TrendDirection",
    "AuditType",
    "CDControlStatus",
    "ComplianceScore",
    "DashboardWidget",
    "DashboardView",
    "AuditRecord",
    "RemediationItem",
    "CDComplianceReport",
    "ComplianceScorer",
    "DashboardBuilder",
    "ReportGenerator",
    "AuditManager",
    "RemediationTracker",
    "ComplianceDashboard",
    # Next-Gen: Verification Marketplace & Community (v0.7.0)
    "ReviewStatus",
    "ContributorRole",
    "ReputationTier",
    "VoteType",
    "ChallengeType",
    "AwardType",
    "CommunityMember",
    "ProofSubmission",
    "ReviewComment",
    "Vote",
    "Challenge",
    "Award",
    "MCLeaderboardEntry",
    "CommunityStats",
    "ReputationEngine",
    "ReviewWorkflow",
    "VotingSystem",
    "ChallengeManager",
    "AwardSystem",
    "MarketplaceCommunity",
    # Next-Gen v0.8.0: Language Expansion Engine
    "LanguageAdapter",
    "LanguageAdapterRegistry",
    "AdapterResult",
    "PythonAdapter",
    "TypeScriptAdapter",
    "GoAdapter",
    "JavaAdapter",
    "RustAdapter",
    "get_adapter_registry",
    "reset_adapter_registry",
    # Next-Gen v0.8.0: Cloud SaaS
    "AuthProvider",
    "TenantTier",
    "TenantStatus",
    "TenantLimits",
    "Tenant",
    "OAuthToken",
    "TenantUser",
    "TenantManager",
    "OAuthManager",
    "CloudUsageRecord",
    "get_tenant_manager",
    "reset_tenant_manager",
    # Next-Gen v0.8.0: Interactive Proof Explorer
    "ProofStepType",
    "ProofOutcome",
    "VisualizationFormat",
    "VariableBinding",
    "ProofStep",
    "Counterexample",
    "ProofTrace",
    "ProofTraceSerializer",
    "ProofExplorer",
    "get_proof_explorer",
    "reset_proof_explorer",
    # Next-Gen v0.8.0: Auto-Fix Pipeline
    "AutoFixPipeline",
    "FixAttempt",
    "FixGenerator",
    "FixVerifier",
    "VerifiedFix",
    "AutoFixLoopFinding",
    "AutoFixLoopStatus",
    "AutoFixLoopConfidence",
    "get_auto_fix_pipeline",
    "reset_auto_fix_pipeline",
    # Next-Gen v0.8.0: LLM Cost Optimizer
    "LLMModelProvider",
    "RoutingStrategy",
    "CheckComplexity",
    "ModelEndpoint",
    "RoutingDecision",
    "CostRecord",
    "LLMCostBudget",
    "LLMCostReport",
    "ComplexityClassifier",
    "CostOptimizer",
    "get_cost_optimizer",
    "reset_cost_optimizer",
    # Next-Gen v0.8.0: CI/CD Actions
    "CIPlatform",
    "GateResult",
    "QualityGateConfig",
    "SARIFResult",
    "SARIFReport",
    "QualityGateEvaluator",
    "CICDConfigGenerator",
    "CICDActionRunner",
    # Next-Gen v0.8.0: Organization Dashboard
    "OrgMetricPeriod",
    "OrgRiskLevel",
    "TeamMetrics",
    "RepoMetrics",
    "RiskHeatmapEntry",
    "OrgROIMetrics",
    "TrendPoint",
    "OrgDashboardData",
    "MetricsAggregator",
    "OrgDashboard",
    # Next-Gen v0.8.0: Verification-Aware Code Generation
    "AssertionType",
    "SuggestionRank",
    "InlineAssertion",
    "VerifiedSuggestion",
    "GenerationContext",
    "ContextExtractor",
    "AssertionGenerator",
    "SuggestionRanker",
    "VerificationAwareCodeGen",
    # Next-Gen v0.8.0: Plugin Marketplace
    "PluginType",
    "PluginStatus",
    "PluginManifest",
    "PluginEntry",
    "PluginReview",
    "PluginSearchResult",
    "PluginSDK",
    "PluginRegistry",
    "get_plugin_registry",
    "reset_plugin_registry",
    # Redis backends (v0.8.0)
    "RedisTenantStore",
    "RedisPluginStore",
    "RedisCostStore",
    # Verification-First Code Completion (v0.9.0)
    "CompletionCandidate",
    "CompletionMiddleware",
    "CompletionSource",
    "CompletionVerifier",
    "CompletionVerifierStats",
    "CompletionVerificationRule",
    "CompletionVerificationStatus",
    "VerifiedCompletion",
    "get_completion_verifier",
    "reset_completion_verifier",
    # Multi-Repo Verification Graph (v0.9.0)
    "BlastRadius",
    "BreakingChange",
    "ChangeType",
    "CompatibilityResult",
    "ContractCompatibilityChecker",
    "ContractExtractor",
    "ContractField",
    "ContractType",
    "ContractEndpoint",
    "FieldType",
    "ServiceContract",
    "ServiceDependency",
    "VerificationGraph",
    "get_verification_graph",
    "reset_verification_graph",
    # Verification-Driven Test Generation (v0.9.0)
    "CodePath",
    "GeneratedTest",
    "MutantStatus",
    "MutationReport",
    "MutationTester",
    "PathCondition",
    "PathConditionType",
    "SymbolicPathDiscoverer",
    "TestFramework",
    "TestInput",
    "TestInputGenerator",
    "TestSuiteGenerator",
    "get_test_generator",
    "reset_test_generator",
    # Adversarial Testing Copilot (v0.9.0)
    "AdversarialReport",
    "AdversarialTester",
    "AttackCategory",
    "AttackVector",
    "AttackVectorGenerator",
    "Exploit",
    "ExploitSeverity",
    "FuzzStrategy",
    "VulnerabilityScanner",
    "get_adversarial_tester",
    "reset_adversarial_tester",
    # Live Verification During Code Review (v0.9.0)
    "AssertionSource",
    "AssertionStatus",
    "CompiledAssertion",
    "FindingConsensus",
    "LiveReviewManager",
    "LiveReviewSession",
    "NLAssertion",
    "NLAssertionCompiler",
    "ReviewVote",
    "VoteType",
    "get_live_review_manager",
    "reset_live_review_manager",
    # AI Code Audit Trail & Provenance (v0.9.0)
    "AICodeDetector",
    "AIFingerprint",
    "ComplianceFramework",
    "ComplianceReport",
    "ProvenanceRecord",
    "ProvenanceSource",
    "ProvenanceTracker",
    "AuditRiskLevel",
    "get_provenance_tracker",
    "reset_provenance_tracker",
    # Explainable AI Verification Reports (v0.9.0)
    "CounterExample",
    "ExplainableReport",
    "ExplainedFinding",
    "ExplanationLevel",
    "FindingExplainer",
    "VerificationFinding",
    "VerificationOutcome",
    "get_finding_explainer",
    "reset_finding_explainer",
    # Semantic Code Clone Detector (v0.9.0)
    "CloneCluster",
    "CloneDetector",
    "ClonePair",
    "CloneType",
    "DeduplicationReport",
    "FunctionExtractor",
    "FunctionSignature",
    "RefactoringSuggestion",
    "RefactoringStatus",
    "get_clone_detector",
    "reset_clone_detector",
    # Verification-as-Code Infrastructure (v0.9.0)
    "DriftReport",
    "DriftStatus",
    "PolicyDSLParser",
    "PolicyEngine",
    "PolicyModule",
    "PolicyRule",
    "PolicyScope",
    "PolicySeverity",
    "PolicyViolation",
    "get_policy_engine",
    "reset_policy_engine",
    # Budget Marketplace (v0.9.0)
    "CapacityLease",
    "MarketOrder",
    "MarketStats",
    "OrderSide",
    "OrderStatus",
    "Trade",
    "TransactionType",
    "VerificationCredit",
    "VerificationMarketplace",
    "Wallet",
    "WalletTransaction",
    "get_marketplace",
    "reset_marketplace",
    # AI Code Insurance Underwriting Platform (v1.0.0)
    "ClaimRejectionReason",
    "ClaimStatus",
    "ClaimValidator",
    "CoverageType",
    "InsuranceClaim",
    "InsurancePolicy",
    "InsuranceRiskProfile",
    "InsuranceUnderwriter",
    "PolicyStatus",
    "PremiumCalculation",
    "PremiumCalculator",
    "RiskTier",
    "get_insurance_underwriter",
    "reset_insurance_underwriter",
    # Cross-Repository Security Graph (v1.0.0)
    "ScanStatus",
    "SecurityBlastRadiusResult",
    "SecurityEdge",
    "SecurityEdgeType",
    "SecurityKnowledgeGraph",
    "SecurityNode",
    "SecurityNodeType",
    "SimilarityMatch",
    "VulnSeverity",
    "VulnerabilityRecord",
    "get_security_graph",
    "reset_security_graph",
    # Differential Privacy-Preserving Proof Marketplace (v1.0.0)
    "AnonymizedProof",
    "ContributionStatus",
    "FederatedAggregator",
    "FederatedUpdate",
    "PrivacyBudget",
    "PrivacyLevel",
    "PrivacyPreservingProofMarketplace",
    "PrivacyProofCategory",
    "ProofAnonymizer",
    "get_privacy_marketplace",
    "reset_privacy_marketplace",
    # Blockchain-Verified Code Provenance (v1.0.0)
    "AttestationStatus",
    "BadgeLevel",
    "BlockchainAttestation",
    "BlockchainProvenanceEngine",
    "ChainType",
    "ContentAddress",
    "LocalBlockchain",
    "VerificationBadge",
    "get_blockchain_provenance",
    "reset_blockchain_provenance",
    # Natural Language Compliance Query Engine (v1.0.0)
    "BUILTIN_TEMPLATES",
    "ComplianceEvidence",
    "ComplianceQuery",
    "ComplianceQueryExecutor",
    "ComplianceQueryResult",
    "ComplianceTemplate",
    "EvidenceStrength",
    "NLComplianceStandard",
    "NLQueryParser",
    "QueryStatus",
    "QueryType",
    "get_compliance_query_engine",
    "reset_compliance_query_engine",
    # AI Model Bias & Fairness Verification (v1.0.0)
    "BiasDetectionResult",
    "BiasDetector",
    "BiasLevel",
    "FairnessComplianceStatus",
    "FairnessConstraint",
    "FairnessMetric",
    "FairnessReport",
    "FairnessVerifier",
    "GroupMetrics",
    "ProtectedAttribute",
    "RemediationAdvisor",
    "RemediationSuggestion",
    "RemediationType",
    "get_fairness_verifier",
    "reset_fairness_verifier",
    # IDE Copilot Undo with Proof Preservation (v1.0.0)
    "CodeDiff",
    "CopilotUndoManager",
    "ProofSnapshot",
    "SavePoint",
    "SavePointStatus",
    "SavePointType",
    "TrustScoreTrend",
    "UndoVerificationState",
    "get_copilot_undo_manager",
    "reset_copilot_undo_manager",
    # Predictive Code Quality Forecasting (v1.0.0)
    "AnomalyDetector",
    "ForecastAlertSeverity",
    "ForecastConfidence",
    "ForecastMetricType",
    "ForecastTrendDirection",
    "QualityAlert",
    "QualityDataPoint",
    "QualityForecast",
    "QualityForecaster",
    "ScenarioResult",
    "TrendAnalysis",
    "TrendCalculator",
    "get_quality_forecaster",
    "reset_quality_forecaster",
    # Verification-Driven Code Generation (v1.0.0)
    "CodeGenerator",
    "CodeSpec",
    "CodegenVerificationResult",
    "ConstraintChecker",
    "ConstraintType",
    "FormalConstraint",
    "GeneratedCandidate",
    "GenerationResult",
    "GenerationStatus",
    "SpecLanguage",
    "VerifiedCodeGenerator",
    "get_verified_codegen",
    "reset_verified_codegen",
    # Real-Time Collaborative Verification Sessions (v1.0.0)
    "CollabMessageType",
    "CollabSessionParticipant",
    "CollaborativeSessionManager",
    "CollaborativeVerificationSession",
    "LineStatus",
    "LineVerification",
    "ParticipantRole",
    "SessionMessage",
    "SessionPhase",
    "SessionRecording",
    "SessionStats",
    "get_collab_session_manager",
    "reset_collab_session_manager",
    # Hosted SaaS Platform (v1.1.0)
    "ApiKey",
    "ApiKeyScope",
    "PlanLimits",
    "PlanTier",
    "RateLimiter",
    "SaaSPlatform",
    "Tenant",
    "TenantStatus",
    "UsageMetricType",
    "UsageRecord",
    "UsageSummary",
    "UsageTracker",
    "get_saas_platform",
    "reset_saas_platform",
    # Go + Java Language Support (v1.1.0)
    "AdvancedGoParser",
    "AdvancedJavaParser",
    "GoJavaLanguageSupport",
    "GoJavaNodeType",
    "GoJavaNode",
    "GoJavaParseResult",
    "IdiomaticPattern",
    "PatternMatch",
    "get_go_java_support",
    "reset_go_java_support",
    # Autofix Agent with PR Generation (v1.1.0)
    "AutofixAgent",
    "FixCandidate",
    "FixCategory",
    "FixConfidence",
    "FixGenerator",
    "FixResult",
    "FixStatus",
    "FixVerifier",
    "get_autofix_agent",
    "reset_autofix_agent",
    # GitHub Copilot Chat Extension (v1.1.0)
    "ChatContext",
    "ChatMessage",
    "ChatResponse",
    "CommandRouter",
    "CopilotCommand",
    "CopilotExtensionHandler",
    "CopilotSession",
    "ResponseFormat",
    "get_copilot_extension_handler",
    "reset_copilot_extension_handler",
    # Incremental Verification Engine (v1.1.0)
    "CacheMetrics",
    "CacheStatus",
    "CachedResult",
    "CodeUnit",
    "DependencyGraph",
    "IncrementalVerificationEngine",
    "get_incremental_engine",
    "reset_incremental_engine",
    # Organization Security Posture Dashboard (v1.1.0)
    "ComplianceRecord",
    "DORAMetrics",
    "OrgSecurityDashboard",
    "OrgSecurityPosture",
    "RepoMetrics",
    "TrendDataPoint",
    "TrendDirection",
    "TrendSeries",
    "get_org_security_dashboard",
    "reset_org_security_dashboard",
    # CI/CD Pipeline Orchestrator (v1.1.0)
    "CICDOrchestrator",
    "CICDPlatform",
    "GateEvaluation",
    "GateMode",
    "GateResult",
    "PipelineConfig",
    "PipelineConfigGenerator",
    "PipelineStatus",
    "QualityGate",
    "QualityGateEvaluator",
    "QualityThresholds",
    "StatusState",
    "get_cicd_orchestrator",
    "reset_cicd_orchestrator",
    # LLM-Powered Proof Explainer (v1.1.0)
    "CheckCategory",
    "CounterexampleParser",
    "CounterexampleValue",
    "ExplanationDetail",
    "ParsedCounterexample",
    "ProofExplanation",
    "ProofExplainerEngine",
    "ProofOutcome",
    "get_proof_explainer",
    "reset_proof_explainer",
    # Supply Chain Verification (v1.1.0)
    "ComplianceStatus",
    "Dependency",
    "DependencyRisk",
    "LicenseCategory",
    "LicensePolicy",
    "LockfileParser",
    "PackageManager",
    "SBOMEntry",
    "SupplyChainReport",
    "SupplyChainVerifier",
    "Vulnerability",
    "VulnerabilityDatabase",
    "VulnerabilitySeverity",
    "get_supply_chain_verifier",
    "reset_supply_chain_verifier",
    # Self-Learning Rule Engine (v1.1.0)
    "ClassificationResult",
    "DismissReason",
    "FalsePositiveClassifier",
    "FeedbackCollector",
    "FeedbackType",
    "FeatureVector",
    "FindingFeedback",
    "LearnedPattern",
    "PatternLearner",
    "RulePerformance",
    "SelfLearningRuleEngine",
    "SeverityAdjustment",
    "SeverityCalibrator",
    "get_self_learning_engine",
    "reset_self_learning_engine",
    # Rust & C/C++ Memory Safety Verification (v1.2.0)
    "CPointerAnalyzer",
    "DataRaceCandidate",
    "DataRaceDetector",
    "LifetimeConstraint",
    "MemoryCheckSeverity",
    "MemoryLanguage",
    "MemoryLocation",
    "MemoryRegion",
    "MemorySafetyReport",
    "MemorySafetyVerifier",
    "MemoryViolation",
    "MemoryViolationType",
    "OwnershipConstraint",
    "OwnershipState",
    "PointerInfo",
    "PointerState",
    "RustOwnershipAnalyzer",
    "get_memory_safety_verifier",
    "reset_memory_safety_verifier",
    # GitHub Copilot Workspace Integration (v1.2.0)
    "ConstraintType",
    "CopilotWorkspaceIntegration",
    "PlanFinding",
    "PlanVerificationResult",
    "VerificationConstraint",
    "VerificationGate",
    "WorkspaceEventType",
    "WorkspaceFile",
    "WorkspacePlan",
    "WorkspacePlanStatus",
    "WorkspaceSession",
    "get_copilot_workspace_integration",
    "reset_copilot_workspace_integration",
    # Zero-Config Onboarding (v1.2.0)
    "BaselineScanResult",
    "ConfigGenerator",
    "DetectedFramework",
    "DetectedLanguage",
    "GeneratedConfig",
    "GeneratedWorkflow",
    "OnboardingResult",
    "OnboardingStep",
    "ProjectAnalysis",
    "ProjectDetector",
    "ProjectType",
    "WorkflowGenerator",
    "ZeroConfigOnboarder",
    "get_zero_config_onboarder",
    "reset_zero_config_onboarder",
    # Autonomous Verification Agent (v1.2.0)
    "AutonomousPR",
    "AutonomousVerificationAgent",
    "AutonomyLevel",
    "AgentState",
    "FeedbackEntry",
    "FindingTriage",
    "FindingTriager",
    "FixCandidate",
    "FixGenerator",
    "FixOutcome",
    "MonitoredChange",
    "TriagedFinding",
    "get_autonomous_agent",
    "reset_autonomous_agent",
    # Verification-as-a-Service API (v1.2.0)
    "OutputFormat",
    "TierLimits",
    "VaaSApiKey",
    "VaaSService",
    "VerificationRequest",
    "VerificationResponse",
    "VerificationStatus",
    "VerificationTier",
    "WebhookConfig",
    "WebhookEventType",
    "get_vaas_service",
    "reset_vaas_service",
    # Interactive Proof Explorer (v1.2.0)
    "ConstraintAnimator",
    "ConstraintStep",
    "DetailLevel",
    "ExportFormat",
    "InteractiveProofExplorer",
    "ProofAnimation",
    "ProofNode",
    "ProofNodeType",
    "ProofRenderer",
    "ProofStatus",
    "ProofTreeParser",
    "ShareableProof",
    # Cross-Repository Blast Radius (v1.2.0)
    "CrossRepoBlastAnalyzer",
    "CrossRepoBlastReport",
    "CrossRepoChange",
    "CrossRepoChangeImpact",
    "CrossRepoDependency",
    "CrossRepoDependencyType",
    "CrossRepoImpactLevel",
    "CrossRepoImpactedRepo",
    "OrgDependencyGraph",
    "OrgRepository",
    "get_cross_repo_blast_analyzer",
    "reset_cross_repo_blast_analyzer",
    # AI Code Review Benchmark Suite (v1.2.0)
    "BenchmarkDataset",
    "BenchmarkMetrics",
    "BenchmarkRunner",
    "BenchmarkSample",
    "BugCategory",
    "BuiltinBenchmarkAdapter",
    "CategoryMetrics",
    "LeaderboardEntry",
    "SampleDifficulty",
    "SampleLanguage",
    "ToolDetection",
    "get_benchmark_runner",
    "reset_benchmark_runner",
    # Fine-Tuned Verification LLM (v1.2.0)
    "AirGapPackage",
    "AirGapPackager",
    "CostComparison",
    "FineTunedVerificationLLM",
    "InferenceConfig",
    "InferenceResult",
    "LocalInferenceEngine",
    "ModelConfig",
    "ModelFormat",
    "ModelSize",
    "TaskType",
    "TrainingDataPipeline",
    "TrainingDataset",
    "TrainingJob",
    "TrainingMetrics",
    "TrainingSample",
    "TrainingStatus",
    "get_fine_tuned_llm",
    "reset_fine_tuned_llm",
    # Developer Certification Program (v1.2.0)
    "Assessment",
    "AssessmentGrader",
    "AssessmentQuestion",
    "AssessmentSubmission",
    "AssessmentType",
    "BadgeType",
    "Certificate",
    "CertificationLevel",
    "CertificationProgram",
    "CourseBuilder",
    "CourseModule",
    "CredentialIssuer",
    "DigitalBadge",
    "LabExercise",
    "LabStatus",
    "LearnerProgress",
    "ModuleStatus",
    "get_certification_program",
    "reset_certification_program",
]
