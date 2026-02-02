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
    CounterexampleToTest,
    GeneratedTest,
    TestFramework,
    TestGeneratorAgent,
    TestSuite,
)

# Next-Gen Feature 5: Natural Language Invariant Specs
from codeverify_agents.nl_invariants import (
    InvariantSpec,
    NaturalLanguageInvariantsAgent,
    ParsedConstraint,
    Z3Assertion,
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
]
