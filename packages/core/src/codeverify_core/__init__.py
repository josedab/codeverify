"""CodeVerify Core - Shared utilities and data models."""

# Import severity utilities directly from the severity module
from codeverify_core.severity import (
    FindingSeverity,
    SEVERITY_ORDER,
    SEVERITY_EMOJI,
    SEVERITY_LABELS,
    parse_severity,
    compare_severity,
    is_blocking_severity,
    is_above_threshold,
    get_severity_emoji,
    get_severity_label,
    sort_by_severity,
)

from codeverify_core.models import (
    Analysis,
    AnalysisStatus,
    Finding,
    FindingCategory,
    VerificationType,
    # Timestamp utilities
    TimestampMixin,
    DataclassTimestampMixin,
    parse_iso_datetime,
    # Result type pattern
    Result,
    StrResult,
    BoolResult,
    OperationResult,
)

# New feature exports
from codeverify_core.rules import (
    CustomRule,
    RuleType,
    RuleEvaluator,
    RuleBuilder,
    RuleViolation,
    get_builtin_rules,
    # Strategy pattern exports
    RuleEvaluationStrategy,
    PatternRuleStrategy,
    CompositeRuleStrategy,
    ASTRuleStrategy,
    SemanticRuleStrategy,
)
from codeverify_core.scanning import (
    ScanConfiguration,
    CodebaseScanResult,
)
from codeverify_core.notifications import (
    SlackFormatter,
    TeamsFormatter,
    NotificationSender,
    NotificationChannel,
    NotificationConfig,
    NotificationType,
)
from codeverify_core.events import (
    Event,
    EventBus,
    EventPriority,
    AnalysisCompleteEvent,
    AnalysisFailedEvent,
    CriticalFindingEvent,
    ScanCompleteEvent,
    DigestReadyEvent,
    get_event_bus,
    reset_event_bus,
    on_event,
)
from codeverify_core.notification_handlers import (
    NotificationEventHandler,
    ConfigProvider,
    InMemoryConfigProvider,
    setup_notification_handlers,
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

# Feature 7: Proof Repository
from codeverify_core.proof_repository import (
    ProofArtifactRepository,
    ProofCategory,
    ProofStatus as ProofRepoStatus,
    ProofStorage,
    ProofTemplate,
    SearchQuery,
    SearchResult,
    InMemoryProofStorage as InMemoryProofRepo,
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

# Next-Gen Feature 1: Monorepo Intelligence
from codeverify_core.monorepo import (
    DependencyGraph,
    MonorepoAnalyzer,
    MonorepoType,
    Package,
    PackageDependency,
)

# Next-Gen Feature 3: Proof-Carrying PRs
from codeverify_core.proof_carrying import (
    ProofAttestation,
    ProofCarryingManager,
    ProofSerializer,
    VerificationProof,
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
    VerificationDepth as BudgetVerificationDepth,
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

# Repository pattern
from codeverify_core.repositories import (
    Repository,
    InMemoryRepository,
    ScanResultRepository,
    ScheduledScanRepository,
    NotificationConfigRepository,
    InMemoryScanResultRepository,
    InMemoryScheduledScanRepository,
    InMemoryNotificationConfigRepository,
    get_scan_result_repository,
    get_scheduled_scan_repository,
    get_notification_config_repository,
    set_scan_result_repository,
    set_scheduled_scan_repository,
    set_notification_config_repository,
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
]
