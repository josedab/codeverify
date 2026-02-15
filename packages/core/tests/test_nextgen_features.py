"""Tests for all 10 next-gen features."""

import sys
from pathlib import Path

import pytest

# Add package src dirs to path so tests can import regardless of installation
_repo = Path(__file__).resolve().parents[3]
for pkg_src in [
    _repo / "packages" / "ai-agents" / "src",
    _repo / "packages" / "lsp-server" / "src",
]:
    if str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))

# Pre-existing __init__.py import errors in codeverify_agents prevent
# importing the package directly. We import modules directly instead.
_AGENTS_IMPORT_NOTE = "Direct module import (bypasses broken __init__.py)"

# Direct module loaders to bypass the pre-existing import error in
# codeverify_agents/__init__.py (SemanticDiffResult missing)
import importlib.util as _ilu
import types as _types


def _load_module(name: str, path: str):
    """Load a Python module directly from file path."""
    spec = _ilu.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"Cannot find module at {path}")
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_agents_src = _repo / "packages" / "ai-agents" / "src" / "codeverify_agents"
_lsp_src = _repo / "packages" / "lsp-server" / "src" / "codeverify_lsp"

# Create namespace packages and load only the modules we need
if "codeverify_agents" not in sys.modules:
    _pkg = _types.ModuleType("codeverify_agents")
    _pkg.__path__ = [str(_agents_src)]
    _pkg.__package__ = "codeverify_agents"
    sys.modules["codeverify_agents"] = _pkg

for _mod_name, _file_name in [
    ("codeverify_agents.retry", "retry.py"),
    ("codeverify_agents.base", "base.py"),
    ("codeverify_agents.fix_approval", "fix_approval.py"),
    ("codeverify_agents.verification_budget", "verification_budget.py"),
    ("codeverify_agents.model_comparison", "model_comparison.py"),
]:
    if _mod_name not in sys.modules:
        _load_module(_mod_name, str(_agents_src / _file_name))

if "codeverify_lsp" not in sys.modules:
    _pkg2 = _types.ModuleType("codeverify_lsp")
    _pkg2.__path__ = [str(_lsp_src)]
    _pkg2.__package__ = "codeverify_lsp"
    sys.modules["codeverify_lsp"] = _pkg2

if "codeverify_lsp.ai_interceptor" not in sys.modules:
    _load_module("codeverify_lsp.ai_interceptor", str(_lsp_src / "ai_interceptor.py"))


# =============================================================================
# Feature 1: Fix Approval Workflow
# =============================================================================

class TestFixApproval:
    """Tests for the ApprovalNotifier and related classes."""

    def test_create_approval_request(self):
        from codeverify_agents.fix_approval import (
            ApprovalNotifier,
            ApprovalPolicy,
            ApprovalStatus,
        )
        notifier = ApprovalNotifier()
        request = notifier.create_request(
            fix_id="fix-1",
            repository="owner/repo",
            title="Fix null dereference",
            description="Adds null check before .name access",
            diff_summary="+ if user is not None:",
            confidence=0.85,
            severity="high",
        )
        assert request.status == ApprovalStatus.PENDING
        assert request.confidence == 0.85
        assert request.fix_id == "fix-1"
        assert not request.is_resolved

    def test_auto_approve_high_confidence(self):
        from codeverify_agents.fix_approval import (
            ApprovalNotifier,
            ApprovalPolicy,
            ApprovalStatus,
        )
        policy = ApprovalPolicy(auto_approve_confidence_threshold=0.9)
        notifier = ApprovalNotifier(default_policy=policy)
        request = notifier.create_request(
            fix_id="fix-2",
            repository="owner/repo",
            title="Simple typo fix",
            description="Fix variable name typo",
            diff_summary="- naem\n+ name",
            confidence=0.95,
        )
        assert request.status == ApprovalStatus.APPROVED
        assert request.is_resolved
        assert "Auto-approved" in (request.resolution_comment or "")

    def test_record_approval_decision(self):
        from codeverify_agents.fix_approval import (
            ApprovalNotifier,
            ApprovalStatus,
        )
        notifier = ApprovalNotifier()
        request = notifier.create_request(
            fix_id="fix-3",
            repository="owner/repo",
            title="Security fix",
            description="Patch SQL injection",
            diff_summary="- query = f'SELECT...'",
            confidence=0.75,
            severity="critical",
        )
        result = notifier.record_decision(request.id, "alice@co.com", "approved")
        assert result is not None
        assert result.status == ApprovalStatus.APPROVED
        assert result.approval_count == 1

    def test_rejection_immediately_resolves(self):
        from codeverify_agents.fix_approval import (
            ApprovalNotifier,
            ApprovalStatus,
        )
        notifier = ApprovalNotifier()
        request = notifier.create_request(
            fix_id="fix-4",
            repository="owner/repo",
            title="Questionable fix",
            description="Removes error handling",
            diff_summary="- try/except removed",
            confidence=0.5,
        )
        result = notifier.record_decision(request.id, "bob@co.com", "rejected", "Too risky")
        assert result is not None
        assert result.status == ApprovalStatus.REJECTED

    def test_get_pending_requests(self):
        from codeverify_agents.fix_approval import ApprovalNotifier
        notifier = ApprovalNotifier()
        notifier.create_request(
            fix_id="fix-5", repository="r", title="t",
            description="d", diff_summary="s", confidence=0.5,
        )
        notifier.create_request(
            fix_id="fix-6", repository="r", title="t2",
            description="d", diff_summary="s", confidence=0.5,
        )
        pending = notifier.get_pending_requests()
        assert len(pending) == 2

    def test_stats(self):
        from codeverify_agents.fix_approval import ApprovalNotifier
        notifier = ApprovalNotifier()
        r1 = notifier.create_request(
            fix_id="f1", repository="r", title="t",
            description="d", diff_summary="s", confidence=0.5,
        )
        notifier.record_decision(r1.id, "user", "approved")
        stats = notifier.get_stats()
        assert stats["total"] == 1
        assert stats["approved"] == 1


# =============================================================================
# Feature 2: Smart Contract Verification
# =============================================================================

class TestSmartContractVerification:
    """Tests for Solidity and Rust/Solana analyzers."""

    SOLIDITY_CODE = """
    pragma solidity ^0.7.0;

    contract Vulnerable {
        mapping(address => uint) balances;

        function withdraw(uint amount) public {
            require(balances[msg.sender] >= amount);
            msg.sender.call{value: amount}("");
            balances[msg.sender] -= amount;
        }

        function authenticate() public {
            require(tx.origin == owner);
        }

        function destroy() public {
            selfdestruct(payable(msg.sender));
        }

        function lottery() public {
            if (block.timestamp % 2 == 0) {
                winner = msg.sender;
            }
        }
    }
    """

    def test_solidity_reentrancy_detection(self):
        from codeverify_core.smart_contract_verification import (
            SolidityAnalyzer,
            VulnerabilityCategory,
        )
        analyzer = SolidityAnalyzer()
        report = analyzer.analyze(self.SOLIDITY_CODE, "Vulnerable")
        categories = [f.category for f in report.findings]
        assert VulnerabilityCategory.REENTRANCY in categories

    def test_solidity_tx_origin_detection(self):
        from codeverify_core.smart_contract_verification import (
            SolidityAnalyzer,
            VulnerabilityCategory,
        )
        analyzer = SolidityAnalyzer()
        report = analyzer.analyze(self.SOLIDITY_CODE)
        categories = [f.category for f in report.findings]
        assert VulnerabilityCategory.TX_ORIGIN in categories

    def test_solidity_selfdestruct_detection(self):
        from codeverify_core.smart_contract_verification import (
            SolidityAnalyzer,
            VulnerabilityCategory,
        )
        analyzer = SolidityAnalyzer()
        report = analyzer.analyze(self.SOLIDITY_CODE)
        categories = [f.category for f in report.findings]
        assert VulnerabilityCategory.SELFDESTRUCT in categories

    def test_solidity_timestamp_detection(self):
        from codeverify_core.smart_contract_verification import (
            SolidityAnalyzer,
            VulnerabilityCategory,
        )
        analyzer = SolidityAnalyzer()
        report = analyzer.analyze(self.SOLIDITY_CODE)
        categories = [f.category for f in report.findings]
        assert VulnerabilityCategory.TIMESTAMP_DEPENDENCY in categories

    def test_solidity_overflow_pre_08(self):
        from codeverify_core.smart_contract_verification import (
            SolidityAnalyzer,
            VulnerabilityCategory,
        )
        # Explicit uint arithmetic in pre-0.8 contract
        code = """
        pragma solidity ^0.7.0;
        contract Overflow {
            uint256 total;
            function add(uint256 amount) public {
                uint256 result = total + amount;
                total = result;
            }
        }
        """
        analyzer = SolidityAnalyzer()
        report = analyzer.analyze(code)
        categories = [f.category for f in report.findings]
        assert VulnerabilityCategory.INTEGER_OVERFLOW in categories

    def test_solidity_no_overflow_post_08(self):
        from codeverify_core.smart_contract_verification import SolidityAnalyzer
        code = "pragma solidity ^0.8.0;\ncontract Safe { uint x = 1 + 2; }"
        analyzer = SolidityAnalyzer()
        report = analyzer.analyze(code)
        overflow_findings = [
            f for f in report.findings if f.category.value == "integer_overflow"
        ]
        assert len(overflow_findings) == 0

    def test_erc20_compliance_check(self):
        from codeverify_core.smart_contract_verification import (
            ERCStandard,
            SolidityAnalyzer,
        )
        code = """
        function totalSupply() external view returns (uint256) {}
        function balanceOf(address) external view returns (uint256) {}
        function transfer(address, uint256) external returns (bool) {}
        function transferFrom(address, address, uint256) external returns (bool) {}
        function approve(address, uint256) external returns (bool) {}
        function allowance(address, address) external view returns (uint256) {}
        event Transfer(address indexed, address indexed, uint256);
        event Approval(address indexed, address indexed, uint256);
        """
        analyzer = SolidityAnalyzer()
        result = analyzer.check_erc_compliance(code, ERCStandard.ERC20)
        assert result.compliant is True

    def test_erc20_non_compliant(self):
        from codeverify_core.smart_contract_verification import (
            ERCStandard,
            SolidityAnalyzer,
        )
        code = "contract Empty {}"
        analyzer = SolidityAnalyzer()
        result = analyzer.check_erc_compliance(code, ERCStandard.ERC20)
        assert result.compliant is False
        assert len(result.missing_functions) > 0

    def test_gas_analysis(self):
        from codeverify_core.smart_contract_verification import SolidityAnalyzer
        code = """
        function transfer(address to, uint amount) public {
            balanceOf[msg.sender] -= amount;
            balanceOf[to] += amount;
        }
        """
        analyzer = SolidityAnalyzer()
        report = analyzer.analyze(code)
        assert len(report.gas_reports) > 0
        assert report.gas_reports[0].function_name == "transfer"

    def test_smart_contract_verifier(self):
        from codeverify_core.smart_contract_verification import (
            ContractLanguage,
            SmartContractVerifier,
        )
        verifier = SmartContractVerifier()
        report = verifier.verify(
            self.SOLIDITY_CODE, "Vulnerable", ContractLanguage.SOLIDITY
        )
        assert report.contract_name == "Vulnerable"
        assert len(report.findings) > 0
        assert report.risk_score > 0

    def test_risk_score_calculation(self):
        from codeverify_core.smart_contract_verification import (
            SmartContractVerifier,
        )
        verifier = SmartContractVerifier()
        report = verifier.verify(self.SOLIDITY_CODE, "Vulnerable")
        assert 0 <= report.risk_score <= 100

    def test_audit_report_to_dict(self):
        from codeverify_core.smart_contract_verification import SmartContractVerifier
        verifier = SmartContractVerifier()
        report = verifier.verify(self.SOLIDITY_CODE)
        d = report.to_dict()
        assert "findings" in d
        assert "summary" in d
        assert "risk_score" in d["summary"]

    def test_rust_solana_missing_signer(self):
        from codeverify_core.smart_contract_verification import (
            ContractLanguage,
            RustSolanaAnalyzer,
            VulnerabilityCategory,
        )
        code = """
        #[derive(Accounts)]
        pub struct Initialize<'info> {
            pub authority: AccountInfo<'info>,
            pub data: Account<'info, MyData>,
        }
        """
        analyzer = RustSolanaAnalyzer()
        report = analyzer.analyze(code, "TestProgram")
        categories = [f.category for f in report.findings]
        assert VulnerabilityCategory.ACCESS_CONTROL in categories


# =============================================================================
# Feature 3: AI Code Interceptor
# =============================================================================

class TestAIInterceptor:
    """Tests for AI code detection and risk scoring."""

    def test_detect_human_code(self):
        from codeverify_lsp.ai_interceptor import AICodeInterceptor, AISource
        interceptor = AICodeInterceptor()
        result = interceptor.intercept("x = 1")
        # Very short code is skipped
        assert result.source == AISource.HUMAN

    def test_detect_copilot_from_context(self):
        from codeverify_lsp.ai_interceptor import AICodeInterceptor, AISource
        interceptor = AICodeInterceptor()
        result = interceptor.intercept(
            "def calculate_total(items):\n    return sum(item.price for item in items)",
            context={"completion_source": "copilot"},
        )
        assert result.source == AISource.COPILOT
        assert result.ai_confidence >= 0.9

    def test_risk_scoring_eval(self):
        from codeverify_lsp.ai_interceptor import AICodeInterceptor, RiskLevel
        interceptor = AICodeInterceptor()
        result = interceptor.intercept(
            "user_input = input()\nresult = eval(user_input)\nprint(result)",
        )
        assert result.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        assert any(f.category == "code_injection" for f in result.risk_factors)

    def test_risk_scoring_sql_injection(self):
        from codeverify_lsp.ai_interceptor import AICodeInterceptor
        interceptor = AICodeInterceptor()
        result = interceptor.intercept(
            'query = f"SELECT * FROM users WHERE id = {user_id}"\ncursor.execute(query)',
        )
        assert any(f.category == "sql_injection" for f in result.risk_factors)

    def test_risk_scoring_hardcoded_secret(self):
        from codeverify_lsp.ai_interceptor import AICodeInterceptor
        interceptor = AICodeInterceptor()
        result = interceptor.intercept(
            'password = "SuperSecretP@ss123"\ndb.connect(password)',
        )
        assert any(f.category == "hardcoded_secret" for f in result.risk_factors)

    def test_cache_hit(self):
        from codeverify_lsp.ai_interceptor import AICodeInterceptor
        interceptor = AICodeInterceptor()
        code = "def foo():\n    return bar()\n    # some code here"
        r1 = interceptor.intercept(code)
        r2 = interceptor.intercept(code)
        assert r2.cached is True

    def test_disabled_interceptor(self):
        from codeverify_lsp.ai_interceptor import (
            AICodeInterceptor,
            InterceptionConfig,
            RiskLevel,
        )
        config = InterceptionConfig(enabled=False)
        interceptor = AICodeInterceptor(config=config)
        result = interceptor.intercept("eval(input())")
        assert result.risk_level == RiskLevel.SAFE

    def test_result_to_dict(self):
        from codeverify_lsp.ai_interceptor import AICodeInterceptor
        interceptor = AICodeInterceptor()
        result = interceptor.intercept(
            "import os\nos.system(user_input)\nprint('done')",
        )
        d = result.to_dict()
        assert "risk_level" in d
        assert "risk_score" in d
        assert "risk_factors" in d


# =============================================================================
# Feature 4: Enterprise Compliance Framework
# =============================================================================

class TestComplianceFramework:
    """Tests for compliance mapping, exceptions, and RBAC."""

    def test_load_soc2_standard(self):
        from codeverify_core.compliance_framework import (
            ComplianceFramework,
            ComplianceStandard,
        )
        fw = ComplianceFramework(organization="acme")
        count = fw.load_standard(ComplianceStandard.SOC2)
        assert count == 5

    def test_load_hipaa_standard(self):
        from codeverify_core.compliance_framework import (
            ComplianceFramework,
            ComplianceStandard,
        )
        fw = ComplianceFramework(organization="acme")
        count = fw.load_standard(ComplianceStandard.HIPAA)
        assert count == 4

    def test_load_pci_dss_standard(self):
        from codeverify_core.compliance_framework import (
            ComplianceFramework,
            ComplianceStandard,
        )
        fw = ComplianceFramework(organization="acme")
        count = fw.load_standard(ComplianceStandard.PCI_DSS)
        assert count == 4

    def test_rbac_permission_check(self):
        from codeverify_core.compliance_framework import (
            ComplianceFramework,
            Role,
        )
        fw = ComplianceFramework(organization="acme")
        fw.set_user_role("alice", Role.ADMIN)
        fw.set_user_role("bob", Role.DEVELOPER)

        assert fw.check_permission("alice", "generate_report") is True
        assert fw.check_permission("bob", "generate_report") is False
        assert fw.check_permission("bob", "view_controls") is True

    def test_assess_controls(self):
        from codeverify_core.compliance_framework import (
            ComplianceFramework,
            ComplianceStandard,
            ControlStatus,
        )
        fw = ComplianceFramework(organization="acme")
        fw.load_standard(ComplianceStandard.SOC2)

        results = fw.assess("my-repo", [
            {"rule_id": "access_control_check", "passed": True},
            {"rule_id": "auth_verification", "passed": True},
            {"rule_id": "security_scan", "passed": False},
        ])

        assert results["CC6.1"] == ControlStatus.PASSING
        assert results["CC7.2"] in (ControlStatus.PARTIALLY_PASSING, ControlStatus.FAILING)

    def test_exception_workflow(self):
        from codeverify_core.compliance_framework import (
            ComplianceFramework,
            ComplianceStandard,
            ExceptionStatus,
            Role,
        )
        fw = ComplianceFramework(organization="acme")
        fw.load_standard(ComplianceStandard.SOC2)
        fw.set_user_role("dev", Role.DEVELOPER)
        fw.set_user_role("sec-lead", Role.SECURITY_LEAD)

        exc = fw.request_exception("dev", "CC6.1", "my-repo", "Legacy system")
        assert exc is not None
        assert exc.status == ExceptionStatus.PENDING

        approved = fw.approve_exception("sec-lead", exc.id, True, "Temporary OK")
        assert approved is not None
        assert approved.status == ExceptionStatus.APPROVED
        assert approved.is_active

    def test_exception_denied_without_permission(self):
        from codeverify_core.compliance_framework import (
            ComplianceFramework,
            ComplianceStandard,
            Role,
        )
        fw = ComplianceFramework(organization="acme")
        fw.load_standard(ComplianceStandard.SOC2)
        fw.set_user_role("dev", Role.DEVELOPER)

        exc = fw.request_exception("dev", "CC6.1", "my-repo", "Reason")
        result = fw.approve_exception("dev", exc.id, True)  # Dev can't approve
        assert result is None

    def test_generate_report(self):
        from codeverify_core.compliance_framework import (
            ComplianceFramework,
            ComplianceStandard,
            Role,
        )
        fw = ComplianceFramework(organization="acme")
        fw.load_standard(ComplianceStandard.SOC2)
        fw.set_user_role("officer", Role.COMPLIANCE_OFFICER)

        fw.assess("my-repo", [
            {"rule_id": "access_control_check", "passed": True},
            {"rule_id": "auth_verification", "passed": True},
        ])

        report = fw.generate_report(ComplianceStandard.SOC2, "officer")
        assert report is not None
        d = report.to_dict()
        assert "summary" in d
        assert "controls" in d
        assert d["summary"]["total_controls"] == 5

    def test_exempted_control_in_assessment(self):
        from codeverify_core.compliance_framework import (
            ComplianceFramework,
            ComplianceStandard,
            ControlStatus,
            Role,
        )
        fw = ComplianceFramework(organization="acme")
        fw.load_standard(ComplianceStandard.SOC2)
        fw.set_user_role("dev", Role.DEVELOPER)
        fw.set_user_role("lead", Role.SECURITY_LEAD)

        exc = fw.request_exception("dev", "CC6.1", "my-repo", "Temporary")
        fw.approve_exception("lead", exc.id, True)

        results = fw.assess("my-repo", [
            {"rule_id": "access_control_check", "passed": False},
        ])
        assert results["CC6.1"] == ControlStatus.EXEMPTED


# =============================================================================
# Feature 5: Collaboration Sessions
# =============================================================================

class TestCollaborationSessions:
    """Tests for live collaboration with trust scoring."""

    def test_create_session(self):
        from codeverify_core.collaboration_sessions import (
            CollaborationSession,
            SessionState,
        )
        session = CollaborationSession(host_name="alice")
        assert session.state == SessionState.ACTIVE
        assert len(session._participants) == 1

    def test_add_participant(self):
        from codeverify_core.collaboration_sessions import (
            CollaborationSession,
            SessionRole,
        )
        session = CollaborationSession(host_name="alice")
        bob = session.add_participant("bob-id", "bob", SessionRole.PARTICIPANT)
        assert bob.name == "bob"
        assert len(session._participants) == 2

    def test_record_edit_and_conflict(self):
        from codeverify_core.collaboration_sessions import CollaborationSession
        session = CollaborationSession(host_name="alice")
        bob = session.add_participant("bob-id", "bob")
        carol = session.add_participant("carol-id", "carol")

        session.record_edit("bob-id", "main.py", 10, 20)
        conflict = session.record_edit("carol-id", "main.py", 15, 25)
        assert conflict is not None
        assert conflict.severity in ("warning", "conflict")

    def test_no_conflict_different_lines(self):
        from codeverify_core.collaboration_sessions import CollaborationSession
        session = CollaborationSession(host_name="alice")
        session.add_participant("bob-id", "bob")
        session.add_participant("carol-id", "carol")

        session.record_edit("bob-id", "main.py", 1, 10)
        conflict = session.record_edit("carol-id", "main.py", 50, 60)
        assert conflict is None

    def test_live_trust_score(self):
        from codeverify_core.collaboration_sessions import CollaborationSession
        session = CollaborationSession(host_name="alice")
        session.record_verification("main.py", 85.0, 3)
        trust = session.get_live_trust_score()
        assert trust.score == 85.0

    def test_session_end_summary(self):
        from codeverify_core.collaboration_sessions import CollaborationSession
        session = CollaborationSession(host_name="alice")
        session.add_participant("bob-id", "bob")
        session.record_edit("bob-id", "main.py", 1, 5)
        summary = session.end()
        assert summary["participants"] == 2
        assert summary["files_edited"] == 1

    def test_session_manager(self):
        from codeverify_core.collaboration_sessions import SessionManager
        manager = SessionManager()
        session = manager.create_session(host_name="alice")
        bob = manager.join_session(session.session_id, "bob", "Bob")
        assert bob is not None
        active = manager.list_active_sessions()
        assert len(active) == 1

    def test_verification_mode_change(self):
        from codeverify_core.collaboration_sessions import (
            CollaborationSession,
            VerificationMode,
        )
        session = CollaborationSession()
        events = []
        session.on_event(lambda e: events.append(e))
        session.set_mode(VerificationMode.BRAINSTORMING)
        assert session.verification_mode == VerificationMode.BRAINSTORMING
        assert any(e.type == "mode_changed" for e in events)


# =============================================================================
# Feature 6: Blast Radius Analysis
# =============================================================================

class TestBlastRadius:
    """Tests for blast radius analysis and team notifications."""

    def _build_graph(self):
        from codeverify_core.blast_radius import DependencyGraph
        graph = DependencyGraph()
        graph.add_dependency("api-gateway", "auth-service")
        graph.add_dependency("web-app", "api-gateway")
        graph.add_dependency("mobile-app", "api-gateway")
        graph.add_dependency("analytics", "api-gateway")
        graph.set_team_owner("web-app", "frontend")
        graph.set_team_owner("mobile-app", "mobile")
        graph.set_team_owner("analytics", "data")
        graph.set_team_contacts("frontend", ["fe@co.com"])
        graph.set_team_contacts("mobile", ["mob@co.com"])
        return graph

    def test_direct_dependents(self):
        graph = self._build_graph()
        deps = graph.get_dependents("auth-service")
        names = [name for name, _ in deps]
        assert "api-gateway" in names

    def test_transitive_dependents(self):
        graph = self._build_graph()
        deps = graph.get_dependents("auth-service")
        names = [name for name, _ in deps]
        assert "web-app" in names
        assert "mobile-app" in names

    def test_blast_radius_analysis(self):
        from codeverify_core.blast_radius import (
            BlastRadiusAnalyzer,
            ChangeType,
        )
        graph = self._build_graph()
        analyzer = BlastRadiusAnalyzer(graph)
        report = analyzer.analyze(
            "auth-service", ["src/auth.py"], ChangeType.BREAKING
        )
        assert report.total_affected >= 3
        assert report.radius_score > 0

    def test_blast_radius_breaking_vs_additive(self):
        from codeverify_core.blast_radius import (
            BlastRadiusAnalyzer,
            ChangeType,
        )
        graph = self._build_graph()
        analyzer = BlastRadiusAnalyzer(graph)

        breaking = analyzer.analyze("auth-service", ["a.py"], ChangeType.BREAKING)
        additive = analyzer.analyze("auth-service", ["a.py"], ChangeType.ADDITIVE)
        assert breaking.radius_score > additive.radius_score

    def test_team_notifications(self):
        from codeverify_core.blast_radius import (
            BlastRadiusAnalyzer,
            ChangeType,
        )
        graph = self._build_graph()
        analyzer = BlastRadiusAnalyzer(graph)
        report = analyzer.analyze("auth-service", ["a.py"], ChangeType.BREAKING)
        notifications = analyzer.generate_notifications(report)
        teams = [n.team for n in notifications]
        assert "frontend" in teams

    def test_report_markdown(self):
        from codeverify_core.blast_radius import (
            BlastRadiusAnalyzer,
            ChangeType,
        )
        graph = self._build_graph()
        analyzer = BlastRadiusAnalyzer(graph)
        report = analyzer.analyze("auth-service", ["a.py"], ChangeType.BREAKING)
        md = report.to_markdown()
        assert "Blast Radius Report" in md
        assert "auth-service" in md

    def test_report_to_dict(self):
        from codeverify_core.blast_radius import (
            BlastRadiusAnalyzer,
            ChangeType,
        )
        graph = self._build_graph()
        analyzer = BlastRadiusAnalyzer(graph)
        report = analyzer.analyze("auth-service", ["a.py"], ChangeType.BREAKING)
        d = report.to_dict()
        assert "summary" in d
        assert "affected_services" in d


# =============================================================================
# Feature 7: Verification Budget Optimizer
# =============================================================================

class TestVerificationBudget:
    """Tests for budget allocation and cost tracking."""

    def test_allocate_by_risk(self):
        from codeverify_agents.verification_budget import (
            VerificationBudgetOptimizer,
            VerificationDepth,
        )
        optimizer = VerificationBudgetOptimizer()
        allocations = optimizer.allocate([
            {"file_path": "auth.py", "risk_score": 0.9},
            {"file_path": "readme.md", "risk_score": 0.05},
        ])
        assert len(allocations) == 2
        assert allocations[0].allocated_depth == VerificationDepth.FORMAL_VERIFICATION
        assert allocations[1].allocated_depth == VerificationDepth.PATTERN_ONLY

    def test_force_depth_override(self):
        from codeverify_agents.verification_budget import (
            VerificationBudgetOptimizer,
            VerificationDepth,
        )
        optimizer = VerificationBudgetOptimizer()
        allocations = optimizer.allocate(
            [{"file_path": "a.py", "risk_score": 0.1}],
            force_depth=VerificationDepth.FULL_PIPELINE,
        )
        assert allocations[0].allocated_depth == VerificationDepth.FULL_PIPELINE
        assert allocations[0].override is True

    def test_budget_tracking(self):
        from codeverify_agents.verification_budget import VerificationBudgetOptimizer
        optimizer = VerificationBudgetOptimizer()
        optimizer.allocate([
            {"file_path": "a.py", "risk_score": 0.5},
        ])
        usage = optimizer.get_usage()
        assert usage["today"]["consumed"] > 0

    def test_cost_report(self):
        from codeverify_agents.verification_budget import VerificationBudgetOptimizer
        optimizer = VerificationBudgetOptimizer()
        optimizer.allocate([
            {"file_path": "a.py", "risk_score": 0.9},
            {"file_path": "b.py", "risk_score": 0.1},
        ])
        report = optimizer.get_cost_report()
        assert report.total_files == 2
        assert report.savings_vs_full > 0

    def test_budget_downgrade(self):
        from codeverify_agents.verification_budget import (
            BudgetConfig,
            VerificationBudgetOptimizer,
            VerificationDepth,
        )
        config = BudgetConfig(daily_budget_units=5.0, reserve_fraction=0.0)
        optimizer = VerificationBudgetOptimizer(config=config)
        # 10 low-risk files should cause downgrading
        allocations = optimizer.allocate([
            {"file_path": f"f{i}.py", "risk_score": 0.2}
            for i in range(20)
        ])
        depths = [a.allocated_depth for a in allocations]
        # Some should be downgraded
        assert VerificationDepth.PATTERN_ONLY in depths


# =============================================================================
# Feature 8: NL Conversational Queries
# =============================================================================

class TestNLConversation:
    """Tests for multi-turn conversational verification."""

    def test_classify_null_query(self):
        from codeverify_core.nl_conversation import IntentClassifier, QueryIntent
        classifier = IntentClassifier()
        assert classifier.classify("Can this be null?") == QueryIntent.NULL_CHECK

    def test_classify_bounds_query(self):
        from codeverify_core.nl_conversation import IntentClassifier, QueryIntent
        classifier = IntentClassifier()
        assert classifier.classify("Is the index out of bounds?") == QueryIntent.BOUNDS_CHECK

    def test_classify_overflow_query(self):
        from codeverify_core.nl_conversation import IntentClassifier, QueryIntent
        classifier = IntentClassifier()
        assert classifier.classify("Can this integer overflow?") == QueryIntent.OVERFLOW_CHECK

    def test_classify_followup(self):
        from codeverify_core.nl_conversation import IntentClassifier, QueryIntent
        classifier = IntentClassifier()
        assert classifier.classify("What about the other parameter?", has_history=True) == QueryIntent.FOLLOWUP

    def test_classify_clarification(self):
        from codeverify_core.nl_conversation import IntentClassifier, QueryIntent
        classifier = IntentClassifier()
        assert classifier.classify("Why is that unsafe?") == QueryIntent.CLARIFICATION

    def test_conversation_session_basic(self):
        from codeverify_core.nl_conversation import ConversationSession
        code = "def divide(a, b):\n    return a / b"
        session = ConversationSession(code=code)
        turn = session.ask("Can this throw an exception?")
        assert turn.response
        assert turn.intent.value == "exception_analysis"

    def test_conversation_multi_turn(self):
        from codeverify_core.nl_conversation import ConversationSession
        code = "def get_name(user):\n    return user.name"
        session = ConversationSession(code=code)
        t1 = session.ask("Can user be null?")
        t2 = session.ask("What about the other parameter?")
        assert len(session.get_history()) == 2

    def test_conversation_context_accumulation(self):
        from codeverify_core.nl_conversation import ConversationSession
        code = "def foo(x):\n    return x[0]"
        session = ConversationSession(code=code)
        session.ask("Can this be null?")
        session.ask("Is the index out of bounds?")
        summary = session.get_summary()
        assert "null_safety" in summary["properties_checked"]
        assert "bounds_safety" in summary["properties_checked"]

    def test_extract_variables(self):
        from codeverify_core.nl_conversation import IntentClassifier
        classifier = IntentClassifier()
        vars = classifier.extract_variables("Is variable `user_id` ever null?")
        assert "user_id" in vars

    def test_suggestions_provided(self):
        from codeverify_core.nl_conversation import ConversationSession
        session = ConversationSession(code="x = 1")
        turn = session.ask("Tell me about this code")
        assert len(turn.suggestions) > 0


# =============================================================================
# Feature 9: Model Comparison Engine
# =============================================================================

class TestModelComparison:
    """Tests for A/B testing and cost optimization."""

    def test_add_sample_and_record_result(self):
        from codeverify_agents.model_comparison import (
            BenchmarkSample,
            ModelComparisonEngine,
            ModelProvider,
            ModelResult,
        )
        engine = ModelComparisonEngine()
        sample = BenchmarkSample(
            id="s1", code="eval(input())", language="python",
            expected_findings=[{"category": "security", "title": "eval_usage"}],
            category="security",
        )
        engine.add_sample(sample)

        result = ModelResult(
            model=ModelProvider.GPT4, sample_id="s1",
            findings=[{"category": "security", "title": "eval_usage"}],
            latency_ms=1500, tokens_used=500, cost=0.03,
        )
        engine.record_result(result)
        report = engine.get_comparison_report()
        assert report["models_tested"] == 1

    def test_accuracy_metrics(self):
        from codeverify_agents.model_comparison import AccuracyMetrics
        m = AccuracyMetrics(true_positives=8, false_positives=2, false_negatives=1)
        assert m.precision == pytest.approx(0.8, abs=0.01)
        assert m.recall == pytest.approx(0.889, abs=0.01)
        assert m.f1_score > 0

    def test_comparison_report_rankings(self):
        from codeverify_agents.model_comparison import (
            BenchmarkSample,
            ModelComparisonEngine,
            ModelProvider,
            ModelResult,
        )
        engine = ModelComparisonEngine()
        sample = BenchmarkSample(
            id="s1", code="x = 1", language="python",
            expected_findings=[{"category": "test", "title": "issue_1"}],
            category="test",
        )
        engine.add_sample(sample)

        # GPT-4 finds the issue
        engine.record_result(ModelResult(
            model=ModelProvider.GPT4, sample_id="s1",
            findings=[{"category": "test", "title": "issue_1"}],
            latency_ms=2000, cost=0.05,
        ))
        # Claude misses it but is cheaper
        engine.record_result(ModelResult(
            model=ModelProvider.CLAUDE_SONNET, sample_id="s1",
            findings=[], latency_ms=500, cost=0.01,
        ))

        report = engine.get_comparison_report()
        assert report["models_tested"] == 2
        assert report["rankings"]["by_accuracy"][0]["model"] == "gpt-4"

    def test_routing_recommendations(self):
        from codeverify_agents.model_comparison import (
            BenchmarkSample,
            ModelComparisonEngine,
            ModelProvider,
            ModelResult,
        )
        engine = ModelComparisonEngine()
        sample = BenchmarkSample(
            id="s1", code="x = 1", language="python",
            expected_findings=[{"category": "t", "title": "i1"}],
            category="test",
        )
        engine.add_sample(sample)
        engine.record_result(ModelResult(
            model=ModelProvider.GPT4, sample_id="s1",
            findings=[{"category": "t", "title": "i1"}],
            latency_ms=2000, cost=0.05,
        ))
        engine.record_result(ModelResult(
            model=ModelProvider.CLAUDE_SONNET, sample_id="s1",
            findings=[{"category": "t", "title": "i1"}],
            latency_ms=500, cost=0.003,
        ))

        recs = engine.get_routing_recommendations()
        assert len(recs) > 0
        contexts = [r.context for r in recs]
        assert "security_critical" in contexts
        assert "real_time_ide" in contexts

    def test_monthly_cost_estimate(self):
        from codeverify_agents.model_comparison import (
            ModelComparisonEngine,
            ModelProvider,
        )
        engine = ModelComparisonEngine()
        estimate = engine.estimate_monthly_cost(ModelProvider.GPT4, daily_analyses=100)
        assert estimate["monthly_cost"] > 0
        assert estimate["daily_analyses"] == 100


# =============================================================================
# Feature 10: Proof Marketplace
# =============================================================================

class TestProofMarketplace:
    """Tests for marketplace with monetization and reputation."""

    def test_register_author(self):
        from codeverify_core.proof_marketplace import ProofMarketplace
        mp = ProofMarketplace()
        author = mp.register_author("alice", "Alice Smith")
        assert author.username == "alice"
        assert author.reputation_score == 0.0

    def test_publish_proof(self):
        from codeverify_core.proof_marketplace import ProofMarketplace
        mp = ProofMarketplace()
        author = mp.register_author("alice", "Alice")
        proof = mp.publish_proof(
            author_id=author.id,
            title="Null Safety Check",
            description="Verifies null safety",
            category="null_safety",
            language="python",
            proof_content="(assert (not (= x None)))",
        )
        assert proof is not None
        assert author.proofs_published == 1
        assert author.reputation_score == 10

    def test_vote_on_proof(self):
        from codeverify_core.proof_marketplace import ProofMarketplace
        mp = ProofMarketplace()
        author = mp.register_author("alice", "Alice")
        proof = mp.publish_proof(
            author_id=author.id, title="Test", description="d",
            category="test", language="python", proof_content="(assert true)",
        )
        mp.vote(proof.id, "bob-id", "up")
        mp.vote(proof.id, "carol-id", "up")
        assert proof.upvotes == 2
        assert author.total_upvotes == 2

    def test_download_free_proof(self):
        from codeverify_core.proof_marketplace import ProofMarketplace
        mp = ProofMarketplace()
        author = mp.register_author("alice", "Alice")
        proof = mp.publish_proof(
            author_id=author.id, title="Free Proof", description="d",
            category="test", language="python", proof_content="(assert true)",
        )
        content = mp.download(proof.id, "bob-id")
        assert content == "(assert true)"
        assert proof.downloads == 1

    def test_purchase_paid_proof(self):
        from codeverify_core.proof_marketplace import (
            PricingTier,
            ProofMarketplace,
        )
        mp = ProofMarketplace(platform_fee_percent=30.0)
        author = mp.register_author("alice", "Alice")
        proof = mp.publish_proof(
            author_id=author.id, title="Premium Proof", description="d",
            category="security", language="python", proof_content="secure",
            pricing_tier=PricingTier.PROFESSIONAL, price=25.0,
        )

        # Can't download without purchasing
        content = mp.download(proof.id, "bob-id")
        assert content is None

        # Purchase
        purchase = mp.purchase(proof.id, "bob-id")
        assert purchase is not None
        assert purchase.price == 25.0
        assert purchase.author_revenue == 17.5  # 70%
        assert purchase.platform_fee == 7.5  # 30%

        # Now can download
        content = mp.download(proof.id, "bob-id")
        assert content == "secure"

    def test_no_double_purchase(self):
        from codeverify_core.proof_marketplace import (
            PricingTier,
            ProofMarketplace,
        )
        mp = ProofMarketplace()
        author = mp.register_author("alice", "Alice")
        proof = mp.publish_proof(
            author_id=author.id, title="P", description="d",
            category="t", language="python", proof_content="c",
            pricing_tier=PricingTier.BASIC, price=5.0,
        )
        mp.purchase(proof.id, "bob")
        second = mp.purchase(proof.id, "bob")
        assert second is None

    def test_search_marketplace(self):
        from codeverify_core.proof_marketplace import ProofMarketplace
        mp = ProofMarketplace()
        author = mp.register_author("alice", "Alice")
        mp.publish_proof(
            author_id=author.id, title="Null Safety", description="null check",
            category="null_safety", language="python", proof_content="c",
            tags=["null", "safety"],
        )
        mp.publish_proof(
            author_id=author.id, title="Bounds Check", description="array bounds",
            category="bounds", language="python", proof_content="c",
            tags=["bounds"],
        )
        results = mp.search(query="null")
        assert len(results) == 1
        assert results[0].title == "Null Safety"

    def test_search_by_category(self):
        from codeverify_core.proof_marketplace import ProofMarketplace
        mp = ProofMarketplace()
        author = mp.register_author("alice", "Alice")
        mp.publish_proof(
            author_id=author.id, title="A", description="d",
            category="security", language="python", proof_content="c",
        )
        mp.publish_proof(
            author_id=author.id, title="B", description="d",
            category="null_safety", language="python", proof_content="c",
        )
        results = mp.search(category="security")
        assert len(results) == 1

    def test_leaderboard(self):
        from codeverify_core.proof_marketplace import ProofMarketplace
        mp = ProofMarketplace()
        alice = mp.register_author("alice", "Alice")
        bob = mp.register_author("bob", "Bob")

        # Alice publishes more
        for i in range(5):
            mp.publish_proof(
                author_id=alice.id, title=f"Proof {i}", description="d",
                category="test", language="python", proof_content="c",
            )
        mp.publish_proof(
            author_id=bob.id, title="Bob's Proof", description="d",
            category="test", language="python", proof_content="c",
        )

        leaderboard = mp.get_leaderboard()
        assert leaderboard[0].author.username == "alice"
        assert leaderboard[0].rank == 1

    def test_reputation_levels(self):
        from codeverify_core.proof_marketplace import AuthorProfile, BadgeType
        author = AuthorProfile(id="1", username="test", display_name="Test")
        assert author.rank == BadgeType.NEWCOMER
        author.reputation_score = 150
        assert author.rank == BadgeType.CONTRIBUTOR
        author.reputation_score = 1500
        assert author.rank == BadgeType.EXPERT
        author.reputation_score = 6000
        assert author.rank == BadgeType.MASTER

    def test_marketplace_stats(self):
        from codeverify_core.proof_marketplace import ProofMarketplace
        mp = ProofMarketplace()
        author = mp.register_author("alice", "Alice")
        mp.publish_proof(
            author_id=author.id, title="Test", description="d",
            category="test", language="python", proof_content="c",
        )
        stats = mp.get_marketplace_stats()
        assert stats["total_proofs"] == 1
        assert stats["total_authors"] == 1
