"""CodeVerify AI Agents - LLM-powered code analysis agents."""

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent, CodeContext, ParsedResponse
from codeverify_agents.diff_summarizer import DiffSummarizerAgent
from codeverify_agents.factory import (
    AgentFactory,
    DefaultLLMClientProvider,
    LLMClientProvider,
    MockLLMClientProvider,
    get_llm_provider,
    reset_llm_provider,
    set_llm_provider,
)
from codeverify_agents.pair_reviewer import (
    CodeRegion,
    FeedbackLearner,
    InlineFinding,
    PairReviewerAgent,
    ReviewCategory,
    ReviewContext,
    ReviewFeedback,
    ReviewPriority,
    SmartThrottler,
)
from codeverify_agents.retry import (
    RetryConfig,
    async_retry,
    retry,
    with_llm_retry,
    DEFAULT_LLM_RETRY_CONFIG,
)
from codeverify_agents.security import SecurityAgent
from codeverify_agents.semantic import SemanticAgent
from codeverify_agents.spec_generator import (
    ClassInvariant,
    FunctionContract as SpecFunctionContract,
    GeneratedSpec,
    SpecificationGeneratorAgent,
    SpecificationSource,
    SpecificationType,
    TypeInfo,
)
from codeverify_agents.synthesis import SynthesisAgent
from codeverify_agents.trust_score import (
    TrustScoreAgent,
    TrustScoreResult,
    TrustScoreFactors,
    # Decomposed components (SRP compliance)
    AIDetector,
    PatternMatcher,
    ComplexityAnalyzer,
    VerificationCoverageCalculator,
    HistoricalAccuracyTracker,
    TrustScoreCalculator,
    RecommendationGenerator,
    calculate_code_hash,
)

# Feature 4: Threat Modeling Agent
from codeverify_agents.threat_modeling import (
    AttackSurface,
    OWASPCategory,
    STRIDECategory,
    Threat,
    ThreatModel,
    ThreatModelingAgent,
)

# Feature 5: Regression Oracle
from codeverify_agents.regression_oracle import (
    BugRecord,
    ChangeMetrics,
    HistoricalSignal,
    RegressionOracle,
    RiskLevel,
    RiskPrediction,
)

# Feature 6: Multi-Model Consensus
from codeverify_agents.multi_model_consensus import (
    ConsensusFinding,
    ConsensusResult,
    ConsensusStrategy,
    ModelConfig,
    ModelFinding,
    ModelProvider,
    MultiModelConsensus,
)

# Feature 8: Compliance Attestation Engine
from codeverify_agents.compliance_attestation import (
    ComplianceAttestationEngine,
    ComplianceFramework,
    ComplianceReport,
    ControlMapping,
    ControlStatus,
    EvidenceItem,
)

# Feature 10: Cross-Language Verification Bridge
from codeverify_agents.cross_language_bridge import (
    CrossLanguageVerificationBridge,
    CrossLanguageVerificationResult,
    FunctionContract,
    InterfaceContract,
    Language,
    LanguageBinding,
    TypeContract,
)

# Next-Gen Feature 2: AI Regression Test Generator
from codeverify_agents.test_generator import (
    Counterexample as CounterexampleToTest,  # Alias for backwards compatibility
    GeneratedTest,
    TestFramework,
    TestGeneratorAgent,
    TestGenerationResult as TestSuite,  # Alias for backwards compatibility
)

# Next-Gen Feature 5: Natural Language Invariant Specs
from codeverify_agents.nl_invariants import (
    NaturalLanguageInvariant as InvariantSpec,  # Alias for backwards compatibility
    NaturalLanguageInvariantAgent as NaturalLanguageInvariantsAgent,  # Alias
    ParsedConstraint,
    Z3Compiler as Z3Assertion,  # Alias for backwards compatibility
)

# Next-Gen Feature 6: Semantic Diff Visualization
from codeverify_agents.semantic_diff import (
    BehaviorChange,
    ChangeType,
    SemanticDiffAgent,
    SemanticDiffResult,
)

# Next-Gen Feature 8: Team Learning Mode
from codeverify_agents.team_learning import (
    OrgHealthReport,
    PatternOccurrence,
    SystemicPattern,
    TeamLearningAgent,
    TeamMetrics,
    TrainingRecommendation,
)

# Next-Gen Feature 9: Competing Model Arbitration
from codeverify_agents.model_arbitrator import (
    ArbitratedFinding,
    ArbitrationResult,
    ArbitrationVote,
    CompetingModelArbitrator,
    ModelProfile,
    ModelSpecialization,
    VotingMethod,
)

# Feature: LLM Fine-Tuning Pipeline
from codeverify_agents.fine_tuning import (
    DataCollector,
    DataSourceType,
    FineTunedModel,
    FineTuningManager,
    ModelServer,
    ModelType,
    TrainingConfig,
    TrainingDataset,
    TrainingExample,
    TrainingJob,
    TrainingMetrics,
    TrainingPipeline,
    TrainingStatus,
    get_fine_tuning_manager,
    reset_fine_tuning_manager,
)

# Feature: Self-Healing Code Suggestions
from codeverify_agents.self_healing import (
    BugReport,
    FixCategory,
    FixGenerationResult,
    FixVerifier,
    ProofStatus,
    SelfHealingAgent,
    SelfHealingManager,
    VerifiedFix,
    get_self_healing_manager,
    reset_self_healing_manager,
)

# Killer Feature 1: AI Code Fingerprinting & Origin Detection
from codeverify_agents.ai_fingerprint import (
    AIClassifier,
    AIFingerprintAgent,
    AIModel,
    CodeMetrics,
    FeatureExtractor,
    FingerprintResult,
    compute_code_hash,
)

# Killer Feature 4: Agentic Auto-Fix PRs
from codeverify_agents.agentic_autofix import (
    AgenticAutoFix,
    AutoFixResult,
    Finding as AutoFixFinding,
    FixCategory as AutoFixCategory,
    FixGenerator,
    FixPR,
    FixStatus,
    FixTemplates,
    FixVerifier as AutoFixVerifier,
    GeneratedFix,
    TestGenerator as AutoFixTestGenerator,
)

# Killer Feature 6: Codebase Intelligence Engine
from codeverify_agents.codebase_intelligence import (
    BugCorrelation,
    BugTracker,
    CodebaseContext,
    CodebaseIntelligenceEngine,
    CodePattern,
    ComponentCriticality,
    ComponentInfo,
    DependencyAnalyzer,
    PatternDetector,
    PatternType,
)

# Killer Feature 8: Intent-to-Code Traceability
from codeverify_agents.intent_traceability import (
    AlignmentChecker,
    ChangeScope,
    CodeChangeAnalyzer,
    CodeChangeSummary,
    ExtractedIntent,
    IntentExtractor,
    IntentTraceabilityEngine,
    IssueDetails,
    IssueProvider,
    TraceabilityFinding,
    TraceabilityResult,
    TraceabilityStatus,
    create_traceability_engine,
)

# Follow-up Item 4: AI Model Integration for Fingerprinting
from codeverify_agents.model_integration import (
    AICodeDetector,
    FeatureExtractor as MLFeatureExtractor,
    HuggingFaceModelBackend,
    MockModelBackend,
    ModelBackend,
    ModelConfig as MLModelConfig,
    ModelEnsemble,
    ModelRegistry,
    ONNXModelBackend,
    PredictionResult,
    detect_ai_code,
    get_detector,
)

# Next-Gen Feature: Formal Specification Assistant
from codeverify_agents.formal_spec_assistant import (
    ConversionResult,
    FormalSpecAssistant,
    NLSpecParser,
    ParsedSpec,
    SpecComplexity,
    SpecDomain,
    SpecLibrary,
    SpecTemplate,
)

# Next-Gen Feature: AI Drift Detector
from codeverify_agents.ai_drift_detector import (
    AIDriftDetector,
    AICodeSnapshot,
    AlertType,
    DriftAlert,
    DriftCategory,
    DriftMetrics,
    DriftReport,
    DriftSeverity,
)

# Next-Gen Feature: Verification Debugger
from codeverify_agents.verification_debugger import (
    Constraint,
    ConstraintType,
    Counterexample,
    DebugResult,
    DebugSession,
    ProofStep,
    ProofStepType,
    Variable,
    VerificationDebugger,
    VerificationStatus,
)

# Next-Gen Feature: Smart Code Search
from codeverify_agents.smart_code_search import (
    CodeCluster,
    CodeType,
    CodeUnit,
    DuplicateGroup,
    SearchMode,
    SearchQuery,
    SearchResult,
    SmartCodeSearch,
)

# Next-Gen Feature: Continuous Learning Engine
from codeverify_agents.continuous_learning import (
    ContinuousLearningEngine,
    FeedbackCollector,
    FeedbackRecord,
    FeedbackType,
    FindingCategory,
    LearnedPattern,
    LearningMetrics,
    LearningStatus,
    PatternLearner,
    TrainingJob,
    TrainingManager,
    TrainingTrigger,
)

# Next-Gen Feature: Context-Aware Analysis
from codeverify_agents.context_aware_analysis import (
    ArchitectureDetector,
    ArchitectureType,
    ContextAwareAnalyzer,
    ContextualFinding,
    Convention,
    ConventionDetector,
    ConventionType,
    DetectedArchitecture,
    DetectedPattern,
    PatternExtractor,
    ProjectContext,
    ProjectType,
    Severity,
    SeverityAdjuster,
    SeverityAdjustment,
)

# Next-Gen Feature: Code Evolution Tracker
from codeverify_agents.code_evolution import (
    CodeEvolutionTracker,
    CommitSnapshot,
    DetectedRegression,
    EvolutionReport,
    MetricTrend,
    MetricType,
    RegressionDetector,
    RegressionSeverity,
    TrendAnalyzer,
    TrendDirection,
)

# Next-Gen Feature: Automated Fix Verification
from codeverify_agents.fix_verification import (
    Fix,
    FixStatus,
    FixVerificationEngine,
    Issue,
    IssueResolver,
    IssueType,
    RegressionCheck,
    RegressionChecker,
    SafetyCheck,
    SafetyLevel,
    SafetyVerifier,
    VerificationMethod,
    VerificationResult,
)

# Next-Gen Feature: Dependency Vulnerability Scanner
from codeverify_agents.dependency_scanner import (
    CVE,
    CVEDatabase,
    DependencyNode,
    DependencyParser,
    DependencyType,
    DependencyVulnerabilityScanner,
    Package,
    PackageEcosystem,
    ScanResult,
    UpgradeAdvisor,
    UpgradePath,
    UpgradeRisk,
    VulnerablePackage,
    VulnerabilitySeverity,
)

__all__ = [
    # Base
    "AgentConfig",
    "AgentResult",
    "BaseAgent",
    "CodeContext",
    "ParsedResponse",
    # Dependency Injection / Factory
    "AgentFactory",
    "DefaultLLMClientProvider",
    "LLMClientProvider",
    "MockLLMClientProvider",
    "get_llm_provider",
    "reset_llm_provider",
    "set_llm_provider",
    # Retry utilities
    "RetryConfig",
    "async_retry",
    "retry",
    "with_llm_retry",
    "DEFAULT_LLM_RETRY_CONFIG",
    # Pair Reviewer
    "ClassInvariant",
    "CodeRegion",
    "FeedbackLearner",
    "GeneratedSpec",
    "InlineFinding",
    "PairReviewerAgent",
    "ReviewCategory",
    "ReviewContext",
    "ReviewFeedback",
    "ReviewPriority",
    "SmartThrottler",
    # Existing Agents
    "DiffSummarizerAgent",
    "SemanticAgent",
    "SecurityAgent",
    "SynthesisAgent",
    "TrustScoreAgent",
    "TrustScoreResult",
    "TrustScoreFactors",
    # Trust Score Components (SRP compliance)
    "AIDetector",
    "PatternMatcher",
    "ComplexityAnalyzer",
    "VerificationCoverageCalculator",
    "HistoricalAccuracyTracker",
    "TrustScoreCalculator",
    "RecommendationGenerator",
    "calculate_code_hash",
    # Spec Generator
    "SpecFunctionContract",
    "SpecificationGeneratorAgent",
    "SpecificationSource",
    "SpecificationType",
    "TypeInfo",
    # Threat Modeling (Feature 4)
    "AttackSurface",
    "OWASPCategory",
    "STRIDECategory",
    "Threat",
    "ThreatModel",
    "ThreatModelingAgent",
    # Regression Oracle (Feature 5)
    "BugRecord",
    "ChangeMetrics",
    "HistoricalSignal",
    "RegressionOracle",
    "RiskLevel",
    "RiskPrediction",
    # Multi-Model Consensus (Feature 6)
    "ConsensusFinding",
    "ConsensusResult",
    "ConsensusStrategy",
    "ModelConfig",
    "ModelFinding",
    "ModelProvider",
    "MultiModelConsensus",
    # Compliance Attestation (Feature 8)
    "ComplianceAttestationEngine",
    "ComplianceFramework",
    "ComplianceReport",
    "ControlMapping",
    "ControlStatus",
    "EvidenceItem",
    # Cross-Language Bridge (Feature 10)
    "CrossLanguageVerificationBridge",
    "CrossLanguageVerificationResult",
    "FunctionContract",
    "InterfaceContract",
    "Language",
    "LanguageBinding",
    "TypeContract",
    # Next-Gen Feature 2: AI Regression Test Generator
    "CounterexampleToTest",
    "GeneratedTest",
    "TestFramework",
    "TestGeneratorAgent",
    "TestSuite",
    # Next-Gen Feature 5: Natural Language Invariant Specs
    "InvariantSpec",
    "NaturalLanguageInvariantsAgent",
    "ParsedConstraint",
    "Z3Assertion",
    # Next-Gen Feature 6: Semantic Diff Visualization
    "BehaviorChange",
    "ChangeType",
    "SemanticDiffAgent",
    "SemanticDiffResult",
    # Next-Gen Feature 8: Team Learning Mode
    "OrgHealthReport",
    "PatternOccurrence",
    "SystemicPattern",
    "TeamLearningAgent",
    "TeamMetrics",
    "TrainingRecommendation",
    # Next-Gen Feature 9: Competing Model Arbitration
    "ArbitratedFinding",
    "ArbitrationResult",
    "ArbitrationVote",
    "CompetingModelArbitrator",
    "ModelProfile",
    "ModelSpecialization",
    "VotingMethod",
    # LLM Fine-Tuning Pipeline
    "DataCollector",
    "DataSourceType",
    "FineTunedModel",
    "FineTuningManager",
    "ModelServer",
    "ModelType",
    "TrainingConfig",
    "TrainingDataset",
    "TrainingExample",
    "TrainingJob",
    "TrainingMetrics",
    "TrainingPipeline",
    "TrainingStatus",
    "get_fine_tuning_manager",
    "reset_fine_tuning_manager",
    # Self-Healing Code Suggestions
    "BugReport",
    "FixCategory",
    "FixGenerationResult",
    "FixVerifier",
    "ProofStatus",
    "SelfHealingAgent",
    "SelfHealingManager",
    "VerifiedFix",
    "get_self_healing_manager",
    "reset_self_healing_manager",
    # Killer Feature 1: AI Code Fingerprinting
    "AIClassifier",
    "AIFingerprintAgent",
    "AIModel",
    "CodeMetrics",
    "FeatureExtractor",
    "FingerprintResult",
    "compute_code_hash",
    # Killer Feature 4: Agentic Auto-Fix PRs
    "AgenticAutoFix",
    "AutoFixResult",
    "AutoFixFinding",
    "AutoFixCategory",
    "FixGenerator",
    "FixPR",
    "FixStatus",
    "FixTemplates",
    "AutoFixVerifier",
    "GeneratedFix",
    "AutoFixTestGenerator",
    # Killer Feature 6: Codebase Intelligence Engine
    "BugCorrelation",
    "BugTracker",
    "CodebaseContext",
    "CodebaseIntelligenceEngine",
    "CodePattern",
    "ComponentCriticality",
    "ComponentInfo",
    "DependencyAnalyzer",
    "PatternDetector",
    "PatternType",
    # Killer Feature 8: Intent-to-Code Traceability
    "AlignmentChecker",
    "ChangeScope",
    "CodeChangeAnalyzer",
    "CodeChangeSummary",
    "ExtractedIntent",
    "IntentExtractor",
    "IntentTraceabilityEngine",
    "IssueDetails",
    "IssueProvider",
    "TraceabilityFinding",
    "TraceabilityResult",
    "TraceabilityStatus",
    "create_traceability_engine",
    # Follow-up Item 4: AI Model Integration
    "AICodeDetector",
    "MLFeatureExtractor",
    "HuggingFaceModelBackend",
    "MockModelBackend",
    "ModelBackend",
    "MLModelConfig",
    "ModelEnsemble",
    "ModelRegistry",
    "ONNXModelBackend",
    "PredictionResult",
    "detect_ai_code",
    "get_detector",
    # Next-Gen Feature: Formal Specification Assistant
    "ConversionResult",
    "FormalSpecAssistant",
    "NLSpecParser",
    "ParsedSpec",
    "SpecComplexity",
    "SpecDomain",
    "SpecLibrary",
    "SpecTemplate",
    # Next-Gen Feature: AI Drift Detector
    "AIDriftDetector",
    "AICodeSnapshot",
    "AlertType",
    "DriftAlert",
    "DriftCategory",
    "DriftMetrics",
    "DriftReport",
    "DriftSeverity",
    # Next-Gen Feature: Verification Debugger
    "Constraint",
    "ConstraintType",
    "Counterexample",
    "DebugResult",
    "DebugSession",
    "ProofStep",
    "ProofStepType",
    "Variable",
    "VerificationDebugger",
    "VerificationStatus",
    # Next-Gen Feature: Smart Code Search
    "CodeCluster",
    "CodeType",
    "CodeUnit",
    "DuplicateGroup",
    "SearchMode",
    "SearchQuery",
    "SearchResult",
    "SmartCodeSearch",
    # Next-Gen Feature: Continuous Learning Engine
    "ContinuousLearningEngine",
    "FeedbackCollector",
    "FeedbackRecord",
    "FeedbackType",
    "FindingCategory",
    "LearnedPattern",
    "LearningMetrics",
    "LearningStatus",
    "PatternLearner",
    "TrainingJob",
    "TrainingManager",
    "TrainingTrigger",
    # Next-Gen Feature: Context-Aware Analysis
    "ArchitectureDetector",
    "ArchitectureType",
    "ContextAwareAnalyzer",
    "ContextualFinding",
    "Convention",
    "ConventionDetector",
    "ConventionType",
    "DetectedArchitecture",
    "DetectedPattern",
    "PatternExtractor",
    "ProjectContext",
    "ProjectType",
    "Severity",
    "SeverityAdjuster",
    "SeverityAdjustment",
    # Next-Gen Feature: Code Evolution Tracker
    "CodeEvolutionTracker",
    "CommitSnapshot",
    "DetectedRegression",
    "EvolutionReport",
    "MetricTrend",
    "MetricType",
    "RegressionDetector",
    "RegressionSeverity",
    "TrendAnalyzer",
    "TrendDirection",
    # Next-Gen Feature: Automated Fix Verification
    "Fix",
    "FixStatus",
    "FixVerificationEngine",
    "Issue",
    "IssueResolver",
    "IssueType",
    "RegressionCheck",
    "RegressionChecker",
    "SafetyCheck",
    "SafetyLevel",
    "SafetyVerifier",
    "VerificationMethod",
    "VerificationResult",
    # Next-Gen Feature: Dependency Vulnerability Scanner
    "CVE",
    "CVEDatabase",
    "DependencyNode",
    "DependencyParser",
    "DependencyType",
    "DependencyVulnerabilityScanner",
    "Package",
    "PackageEcosystem",
    "ScanResult",
    "UpgradeAdvisor",
    "UpgradePath",
    "UpgradeRisk",
    "VulnerablePackage",
    "VulnerabilitySeverity",
]
