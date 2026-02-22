"""Focused unit tests for previously untested modules.

Covers 16 modules that had zero test references in the test suite.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestAICollaboration:
    def test_import(self):
        from codeverify_core.ai_collaboration import CollaborationSession, MessageType
        assert CollaborationSession is not None
        assert MessageType.CONSTRAINT is not None


class TestContinuousVerification:
    def test_import(self):
        from codeverify_core.continuous_verification import ChangeType, ASTNode
        node = ASTNode(id="n1", node_type="function", name="foo", range=(0, 10), content_hash="abc")
        assert node.name == "foo"


class TestCopilotSessions:
    def test_import(self):
        from codeverify_core.copilot_sessions import CopilotSessionPool, SessionState
        pool = CopilotSessionPool()
        assert pool is not None
        assert SessionState.IDLE is not None


class TestDistributedNetwork:
    def test_import(self):
        from codeverify_core.distributed_network import LoadBalancer, NodeStatus
        lb = LoadBalancer()
        assert lb is not None
        assert NodeStatus.ONLINE is not None


class TestFormalSpecs:
    def test_import(self):
        from codeverify_core.formal_specs import ClassInvariant, SpecificationType
        inv = ClassInvariant(class_name="User", invariants=["age >= 0"])
        assert inv.class_name == "User"
        assert SpecificationType.PRECONDITION is not None


class TestImpactAnalysis:
    def test_import(self):
        from codeverify_core.impact_analysis import CrossRepoImpactAnalyzer, ImpactSeverity
        analyzer = CrossRepoImpactAnalyzer()
        assert analyzer is not None
        assert ImpactSeverity.HIGH is not None


class TestMemoryGraph:
    def test_import(self):
        from codeverify_core.memory_graph import VerificationKnowledgeGraph, GraphNodeType
        graph = VerificationKnowledgeGraph()
        assert graph is not None
        assert GraphNodeType.PROOF is not None


class TestMultiTenancy:
    def test_import(self):
        from codeverify_core.multi_tenancy import TenantConfig, TenantTier
        config = TenantConfig(
            id="t1", name="Test", slug="test", tier=TenantTier.FREE,
            max_repos=5, max_analyses_per_month=100, max_users=3,
        )
        assert config.name == "Test"


class TestNLQueries:
    def test_import(self):
        from codeverify_core.nl_queries import NaturalLanguageQueryParser, QueryType
        parser = NaturalLanguageQueryParser()
        assert parser is not None
        assert QueryType.NULL_CHECK is not None


class TestPerformanceOptimization:
    def test_import(self):
        from codeverify_core.performance_optimization import BackgroundVerificationQueue, VerificationDepth
        queue = BackgroundVerificationQueue()
        assert queue is not None


class TestProofRepository:
    def test_import(self):
        from codeverify_core.proof_repository import ProofArtifactRepository, ProofCategory
        repo = ProofArtifactRepository()
        assert repo is not None
        assert ProofCategory.NULL_SAFETY is not None


class TestRedisBackends:
    def test_import(self):
        from codeverify_core.redis_backends import RedisTenantStore
        assert RedisTenantStore is not None


class TestRegressionLearning:
    def test_import(self):
        from codeverify_core.regression_learning import BugType, CodeFeatureExtractor
        extractor = CodeFeatureExtractor()
        assert extractor is not None
        assert BugType.NULL_POINTER is not None


class TestRulesLegacy:
    def test_import(self):
        from codeverify_core.rules_legacy import RuleSeverity, RuleType, ASTRuleStrategy
        strategy = ASTRuleStrategy()
        assert strategy is not None
        assert RuleType.PATTERN is not None


class TestRuntimeProbes:
    def test_import(self):
        from codeverify_core.runtime_probes import InstrumentationEngine, ProbeType
        engine = InstrumentationEngine(language="python")
        assert engine is not None
        assert ProbeType.NULL_CHECK is not None


class TestSubFunctionAnalysis:
    def test_import(self):
        from codeverify_core.sub_function_analysis import SubFunctionParser, GranularityLevel
        parser = SubFunctionParser()
        assert parser is not None
        assert GranularityLevel.FUNCTION is not None
