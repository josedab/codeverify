"""Integration tests for next-gen features (Features 4-10)."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from datetime import datetime


class TestThreatModelingIntegration:
    """Integration tests for threat modeling agent."""

    @pytest.fixture
    def sample_api_code(self):
        return """
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import text

app = FastAPI()

@app.post("/users")
async def create_user(username: str, password: str, db = Depends(get_db)):
    query = f"INSERT INTO users (username, password) VALUES ('{username}', '{password}')"
    await db.execute(text(query))
    return {"status": "created"}

@app.get("/users/{user_id}")
async def get_user(user_id: int, db = Depends(get_db)):
    result = await db.execute(text(f"SELECT * FROM users WHERE id = {user_id}"))
    return result.fetchone()
"""

    @pytest.mark.asyncio
    async def test_threat_model_generation(self, sample_api_code):
        """Test threat model generation from code."""
        from codeverify_agents import ThreatModelingAgent

        agent = ThreatModelingAgent()
        
        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "content": """{
                    "attack_surfaces": [
                        {"name": "POST /users", "type": "api_endpoint", "entry_points": ["username", "password"], "trust_level": "untrusted"}
                    ],
                    "threats": [
                        {"id": "T1", "title": "SQL Injection", "stride_category": "tampering", "owasp_category": "A03:2021", "attack_surface": "POST /users", "likelihood": "high", "impact": "critical", "risk_score": 9.5, "mitigations": ["Use parameterized queries"]}
                    ],
                    "trust_boundaries": [],
                    "data_flows": [],
                    "recommendations": ["Implement input validation"]
                }""",
                "tokens": 500,
            }
            
            result = await agent.analyze(sample_api_code, {
                "system_name": "User API",
                "language": "python",
                "framework": "fastapi",
            })
            
            assert result.success
            assert "attack_surfaces" in result.data
            assert "threats" in result.data
            assert len(result.data["threats"]) > 0

    @pytest.mark.asyncio
    async def test_stride_categorization(self, sample_api_code):
        """Test STRIDE threat categorization."""
        from codeverify_agents import ThreatModelingAgent, STRIDECategory

        agent = ThreatModelingAgent()
        
        # Verify all STRIDE categories are valid
        categories = [c.value for c in STRIDECategory]
        assert "spoofing" in categories
        assert "tampering" in categories
        assert "repudiation" in categories
        assert "information_disclosure" in categories
        assert "denial_of_service" in categories
        assert "elevation_of_privilege" in categories


class TestRegressionOracleIntegration:
    """Integration tests for regression oracle."""

    @pytest.fixture
    def sample_diff(self):
        return """
diff --git a/src/auth.py b/src/auth.py
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,20 @@ def authenticate(user, password):
+def validate_token(token):
+    if not token:
+        return False
+    try:
+        decoded = jwt.decode(token, SECRET_KEY)
+        if decoded['exp'] < time.time():
+            return False
+        return True
+    except Exception:
+        return False
"""

    @pytest.mark.asyncio
    async def test_risk_prediction(self, sample_diff):
        """Test risk prediction for code changes."""
        from codeverify_agents import RegressionOracle

        oracle = RegressionOracle()
        
        with patch.object(oracle, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "content": '{"risk_score": 45, "risk_factors": [{"factor": "New authentication code", "severity": "medium", "confidence": 0.8}], "recommended_tests": ["Test token expiration"]}',
                "tokens": 200,
            }
            
            result = await oracle.analyze(sample_diff, {
                "change_id": "test-123",
                "file_paths": ["src/auth.py"],
                "author": "developer@example.com",
                "commit_message": "Add token validation",
            })
            
            assert result.success
            assert "risk_score" in result.data
            assert "risk_level" in result.data
            assert result.data["risk_score"] >= 0
            assert result.data["risk_score"] <= 100

    def test_bug_recording(self):
        """Test bug recording for training."""
        from codeverify_agents import RegressionOracle, BugRecord, ChangeMetrics
        from datetime import datetime

        oracle = RegressionOracle()
        
        bug = BugRecord(
            bug_id="BUG-123",
            timestamp=datetime.utcnow(),
            file_path="src/auth.py",
            function_name="validate_token",
            change_metrics=ChangeMetrics(lines_added=20, lines_deleted=0),
            author="developer@example.com",
            severity="high",
            root_cause="Missing null check",
            fix_complexity="low",
        )
        
        oracle.record_bug(bug)
        
        # Bug should be recorded in history
        assert len(oracle._bug_history) == 1
        assert oracle._file_bug_counts.get("src/auth.py") == 1


class TestMultiModelConsensusIntegration:
    """Integration tests for multi-model consensus."""

    @pytest.fixture
    def sample_code(self):
        return """
def process_payment(amount, card_number):
    # No input validation
    charge = stripe.Charge.create(
        amount=amount,
        card=card_number,
    )
    return charge
"""

    @pytest.mark.asyncio
    async def test_consensus_verification(self, sample_code):
        """Test consensus verification with multiple models."""
        from codeverify_agents import MultiModelConsensus, ConsensusStrategy

        consensus = MultiModelConsensus(
            consensus_strategy=ConsensusStrategy.MAJORITY,
        )
        
        with patch.object(consensus, "_query_all_models", new_callable=AsyncMock) as mock_query:
            from codeverify_agents.multi_model_consensus import ModelProvider, ModelFinding
            
            # Simulate findings from multiple models
            mock_query.return_value = {
                ModelProvider.OPENAI_GPT5: [
                    ModelFinding(
                        model=ModelProvider.OPENAI_GPT5,
                        finding_id="f1",
                        severity="high",
                        category="security",
                        title="Missing input validation",
                        description="Card number not validated",
                        location={"line": 3},
                        confidence=0.9,
                    )
                ],
                ModelProvider.ANTHROPIC_CLAUDE: [
                    ModelFinding(
                        model=ModelProvider.ANTHROPIC_CLAUDE,
                        finding_id="f2",
                        severity="high",
                        category="security",
                        title="No input validation",
                        description="Amount and card not validated",
                        location={"line": 3},
                        confidence=0.85,
                    )
                ],
            }
            
            result = await consensus.analyze(sample_code, {
                "file_path": "payment.py",
                "language": "python",
            })
            
            assert result.success
            assert "consensus_findings" in result.data
            assert "overall_confidence" in result.data

    def test_consensus_strategies(self):
        """Test different consensus strategies."""
        from codeverify_agents import ConsensusStrategy

        strategies = [s.value for s in ConsensusStrategy]
        assert "unanimous" in strategies
        assert "majority" in strategies
        assert "weighted" in strategies
        assert "any" in strategies


class TestProofRepositoryIntegration:
    """Integration tests for proof artifact repository."""

    @pytest.mark.asyncio
    async def test_proof_storage_and_search(self):
        """Test proof storage and retrieval."""
        from codeverify_core import ProofArtifactRepository
        from codeverify_core.proof_repository import ProofArtifact, ProofCategory, ProofStatus
        from datetime import datetime

        repo = ProofArtifactRepository()
        
        proof = ProofArtifact(
            proof_id="",
            category=ProofCategory.NULL_SAFETY,
            status=ProofStatus.VERIFIED,
            pattern_name="null_check_pattern",
            pattern_description="Null check before dereference",
            language="python",
            code_template="if x is not None: x.method()",
            code_hash="abc123",
            constraints=["(not (= x null))"],
            tags=["null", "safety"],
            keywords=["null_check", "dereference"],
        )
        
        proof_id = await repo.store_proof(proof)
        assert proof_id
        
        # Search for the proof
        results = await repo.search_proofs(
            keywords=["null_check"],
            categories=[ProofCategory.NULL_SAFETY],
            language="python",
        )
        
        assert len(results) > 0
        assert results[0].proof.pattern_name == "null_check_pattern"

    @pytest.mark.asyncio
    async def test_proof_templates(self):
        """Test built-in proof templates."""
        from codeverify_core import ProofArtifactRepository
        from codeverify_core.proof_repository import ProofCategory

        repo = ProofArtifactRepository()
        
        templates = repo.list_templates(
            category=ProofCategory.NULL_SAFETY,
            language="python",
        )
        
        assert len(templates) > 0
        assert templates[0].name == "Null Check Before Use"


class TestComplianceAttestationIntegration:
    """Integration tests for compliance attestation engine."""

    @pytest.fixture
    def sample_verification_results(self):
        return [
            {
                "file_path": "src/auth.py",
                "status": "verified",
                "findings": [],
                "verified_properties": ["authentication", "authorization"],
            },
            {
                "file_path": "src/db.py",
                "status": "partial",
                "findings": [{"category": "security", "severity": "medium"}],
                "verified_properties": ["data_integrity"],
            },
        ]

    @pytest.mark.asyncio
    async def test_soc2_report_generation(self, sample_verification_results):
        """Test SOC2 compliance report generation."""
        from codeverify_agents import ComplianceAttestationEngine, ComplianceFramework

        engine = ComplianceAttestationEngine()
        
        result = await engine.analyze("", {
            "framework": ComplianceFramework.SOC2,
            "verification_results": sample_verification_results,
            "scope": "User Authentication System",
            "organization": "TestCorp",
        })
        
        assert result.success
        assert result.data["framework"] == "soc2"
        assert "compliance_score" in result.data
        assert "controls" in result.data
        assert len(result.data["controls"]) > 0

    @pytest.mark.asyncio
    async def test_multi_framework_report(self, sample_verification_results):
        """Test multi-framework compliance report."""
        from codeverify_agents import ComplianceAttestationEngine, ComplianceFramework

        engine = ComplianceAttestationEngine()
        
        result = await engine.generate_multi_framework_report(
            frameworks=[ComplianceFramework.SOC2, ComplianceFramework.HIPAA],
            verification_results=sample_verification_results,
            scope="Healthcare API",
            organization="HealthCorp",
        )
        
        assert "soc2" in result["reports"]
        assert "hipaa" in result["reports"]
        assert "overall_summary" in result

    def test_supported_frameworks(self):
        """Test all supported compliance frameworks."""
        from codeverify_agents import ComplianceAttestationEngine, ComplianceFramework

        engine = ComplianceAttestationEngine()
        frameworks = engine.get_supported_frameworks()
        
        assert "soc2" in frameworks
        assert "hipaa" in frameworks
        assert "pci_dss" in frameworks
        assert "gdpr" in frameworks


class TestCostOptimizerIntegration:
    """Integration tests for verification cost optimizer."""

    def test_verification_planning(self):
        """Test verification depth planning."""
        from codeverify_core import VerificationCostOptimizer, VerificationDepth

        optimizer = VerificationCostOptimizer()
        
        # Low-risk code
        low_risk_code = "x = 1 + 2"
        plan = optimizer.plan_verification(low_risk_code)
        
        assert plan.selected_depth in [VerificationDepth.PATTERN, VerificationDepth.STATIC]
        assert plan.estimated_cost_usd < 0.01
        
        # High-risk code with security patterns
        high_risk_code = """
def authenticate(password, api_key):
    if password == "admin123":
        return True
    eval(api_key)  # Dangerous!
"""
        plan = optimizer.plan_verification(high_risk_code)
        
        assert plan.selected_depth in [VerificationDepth.AI, VerificationDepth.FORMAL]
        assert "Security-sensitive" in " ".join(plan.rationale)

    def test_budget_constraints(self):
        """Test budget constraint handling."""
        from codeverify_core import VerificationCostOptimizer, BudgetConstraints, VerificationDepth

        optimizer = VerificationCostOptimizer()
        
        # Very tight budget
        budget = BudgetConstraints(
            max_cost_usd=0.001,
            max_time_seconds=1,
        )
        
        code = "def risky(): eval(input())"
        plan = optimizer.plan_verification(code, budget=budget)
        
        # Should downgrade due to budget
        assert plan.estimated_cost_usd <= budget.max_cost_usd

    def test_batch_optimization(self):
        """Test batch verification optimization."""
        from codeverify_core import VerificationCostOptimizer, BudgetConstraints

        optimizer = VerificationCostOptimizer()
        
        code_items = [
            ("def safe(): return 1", None),
            ("def risky(): eval(x)", None),
            ("x = 1", None),
        ]
        
        budget = BudgetConstraints(max_cost_usd=0.1)
        plans = optimizer.optimize_batch(code_items, budget)
        
        assert len(plans) == 3
        # Total cost should be within budget
        total_cost = sum(p.estimated_cost_usd for p in plans)
        assert total_cost <= budget.max_cost_usd * 1.5  # Allow some flexibility


class TestCrossLanguageBridgeIntegration:
    """Integration tests for cross-language verification bridge."""

    @pytest.fixture
    def python_function(self):
        return """
def calculate_discount(price: float, discount: float) -> float:
    '''Calculate discounted price.'''
    if discount < 0 or discount > 100:
        raise ValueError("Invalid discount")
    return price * (1 - discount / 100)
"""

    @pytest.fixture
    def typescript_function(self):
        return """
function calculateDiscount(price: number, discount: number): number {
    if (discount < 0 || discount > 100) {
        throw new Error("Invalid discount");
    }
    return price * (1 - discount / 100);
}
"""

    @pytest.mark.asyncio
    async def test_contract_inference_python(self, python_function):
        """Test contract inference from Python code."""
        from codeverify_agents import CrossLanguageVerificationBridge, Language

        bridge = CrossLanguageVerificationBridge()
        
        result = await bridge.analyze(python_function, {
            "language": "python",
            "symbol_name": "calculate_discount",
            "symbol_type": "function",
        })
        
        assert result.success
        assert "contract" in result.data
        assert result.data["contract"]["name"] == "calculate_discount"
        assert len(result.data["contract"]["parameters"]) == 2

    @pytest.mark.asyncio
    async def test_contract_inference_typescript(self, typescript_function):
        """Test contract inference from TypeScript code."""
        from codeverify_agents import CrossLanguageVerificationBridge, Language

        bridge = CrossLanguageVerificationBridge()
        
        result = await bridge.analyze(typescript_function, {
            "language": "typescript",
            "symbol_name": "calculateDiscount",
            "symbol_type": "function",
        })
        
        assert result.success
        assert "contract" in result.data
        assert result.data["contract"]["name"] == "calculateDiscount"

    def test_type_mapping(self):
        """Test type mapping between languages."""
        from codeverify_agents import CrossLanguageVerificationBridge, Language

        bridge = CrossLanguageVerificationBridge()
        
        # Python to TypeScript
        assert bridge.get_type_mapping("int", Language.PYTHON, Language.TYPESCRIPT) == "number"
        assert bridge.get_type_mapping("string", Language.PYTHON, Language.TYPESCRIPT) == "string"
        assert bridge.get_type_mapping("bool", Language.PYTHON, Language.TYPESCRIPT) == "boolean"

    def test_stub_generation(self):
        """Test stub generation in target language."""
        from codeverify_agents import CrossLanguageVerificationBridge, Language
        from codeverify_agents.cross_language_bridge import FunctionContract, TypeContract

        bridge = CrossLanguageVerificationBridge()
        
        # Create a contract
        contract = FunctionContract(
            contract_id="test_func",
            name="process_data",
            description="Process input data",
            parameters=[
                ("data", TypeContract(
                    contract_id="t1",
                    name="string",
                    description="",
                    base_type="string",
                )),
            ],
            return_type=TypeContract(
                contract_id="t2",
                name="bool",
                description="",
                base_type="bool",
            ),
        )
        
        bridge.register_contract(contract)
        
        stub = bridge.generate_stub("test_func", Language.PYTHON)
        assert stub is not None
        assert "def process_data" in stub
        assert "str" in stub

        stub_ts = bridge.generate_stub("test_func", Language.TYPESCRIPT)
        assert stub_ts is not None
        assert "function process_data" in stub_ts
