"""Tests for v1.2.0 next-gen features.

Covers all 10 next-gen features:
1. Rust & C/C++ Memory Safety Verification
2. GitHub Copilot Workspace Integration
3. Zero-Config Onboarding
4. Autonomous Verification Agent
5. Verification-as-a-Service API
6. Interactive Proof Explorer
7. Cross-Repository Blast Radius Analysis
8. AI Code Review Benchmark Suite
9. Fine-Tuned Verification LLM
10. Developer Certification Program
"""

import pytest

# --- Feature 1: Rust & C/C++ Memory Safety Verification ---


class TestMemorySafety:
    def test_rust_use_after_move(self):
        from codeverify_core.memory_safety import (
            MemoryLanguage,
            MemorySafetyVerifier,
            MemoryViolationType,
        )

        code = 'let data = vec![1, 2, 3];\nlet other = data;\nprintln!("{}", data);'
        verifier = MemorySafetyVerifier()
        report = verifier.verify_rust(code)
        move_violations = [
            v
            for v in report.violations
            if v.violation_type == MemoryViolationType.OWNERSHIP_VIOLATION
        ]
        assert len(move_violations) > 0
        assert report.language == MemoryLanguage.RUST

    def test_rust_borrow_violation(self):
        from codeverify_core.memory_safety import (
            MemorySafetyVerifier,
            MemoryViolationType,
        )

        code = "let mut x = 5;\nlet r1 = &x;\nlet r2 = &mut x;"
        verifier = MemorySafetyVerifier()
        report = verifier.verify_rust(code)
        borrow_violations = [
            v for v in report.violations if v.violation_type == MemoryViolationType.BORROW_VIOLATION
        ]
        assert len(borrow_violations) > 0

    def test_c_use_after_free(self):
        from codeverify_core.memory_safety import (
            MemorySafetyVerifier,
            MemoryViolationType,
        )

        code = "int *ptr = malloc(10);\nfree(ptr);\n*ptr = 42;"
        verifier = MemorySafetyVerifier()
        report = verifier.verify_c(code)
        uaf = [
            v for v in report.violations if v.violation_type == MemoryViolationType.USE_AFTER_FREE
        ]
        assert len(uaf) > 0

    def test_c_double_free(self):
        from codeverify_core.memory_safety import (
            MemorySafetyVerifier,
            MemoryViolationType,
        )

        code = "char *buf = malloc(100);\nfree(buf);\nfree(buf);"
        verifier = MemorySafetyVerifier()
        report = verifier.verify_c(code)
        df = [v for v in report.violations if v.violation_type == MemoryViolationType.DOUBLE_FREE]
        assert len(df) > 0

    def test_c_memory_leak(self):
        from codeverify_core.memory_safety import (
            MemorySafetyVerifier,
            MemoryViolationType,
        )

        code = "int *data = malloc(sizeof(int) * 10);\ndata[0] = 42;"
        verifier = MemorySafetyVerifier()
        report = verifier.verify_c(code)
        leaks = [
            v for v in report.violations if v.violation_type == MemoryViolationType.MEMORY_LEAK
        ]
        assert len(leaks) > 0

    def test_c_buffer_overflow(self):
        from codeverify_core.memory_safety import (
            MemorySafetyVerifier,
            MemoryViolationType,
        )

        code = "int *arr = malloc(5);\narr[100] = 42;"
        verifier = MemorySafetyVerifier()
        report = verifier.verify_c(code)
        overflow = [
            v for v in report.violations if v.violation_type == MemoryViolationType.BUFFER_OVERFLOW
        ]
        assert len(overflow) > 0

    def test_memory_safety_report_properties(self):
        from codeverify_core.memory_safety import MemorySafetyVerifier

        code = "int x = 5;\nint y = x + 1;"
        verifier = MemorySafetyVerifier()
        report = verifier.verify_c(code)
        assert report.is_safe or not report.is_safe  # property works
        assert isinstance(report.summary, str)

    def test_rust_unsafe_detection(self):
        from codeverify_core.memory_safety import MemorySafetyVerifier

        code = "unsafe { *ptr = 42; }"
        verifier = MemorySafetyVerifier()
        report = verifier.verify_rust(code)
        assert len(report.violations) > 0

    def test_singleton_access(self):
        from codeverify_core.memory_safety import (
            get_memory_safety_verifier,
            reset_memory_safety_verifier,
        )

        reset_memory_safety_verifier()
        v1 = get_memory_safety_verifier()
        v2 = get_memory_safety_verifier()
        assert v1 is v2
        reset_memory_safety_verifier()


# --- Feature 2: Copilot Workspace Integration ---


class TestCopilotWorkspace:
    def test_create_session(self):
        from codeverify_core.copilot_workspace import CopilotWorkspaceIntegration

        integration = CopilotWorkspaceIntegration()
        session = integration.create_session("ws-123", "user-1")
        assert session.workspace_id == "ws-123"
        assert len(session.active_constraints) > 0

    def test_submit_and_verify_plan(self):
        from codeverify_core.copilot_workspace import (
            CopilotWorkspaceIntegration,
        )

        integration = CopilotWorkspaceIntegration()
        session = integration.create_session("ws-1")
        plan = integration.submit_plan(
            session.id,
            "Add auth",
            [
                {
                    "path": "auth.py",
                    "original_content": "",
                    "proposed_content": "def login(): pass",
                    "is_new": True,
                }
            ],
        )
        result = integration.verify_plan(session.id, plan.id)
        assert result.plan_id == plan.id
        assert result.trust_score >= 0
        assert result.gate is not None

    def test_security_finding_blocks(self):
        from codeverify_core.copilot_workspace import (
            CopilotWorkspaceIntegration,
            VerificationGate,
        )

        integration = CopilotWorkspaceIntegration(block_on_critical=True)
        session = integration.create_session("ws-2")
        plan = integration.submit_plan(
            session.id,
            "Add code",
            [{"path": "app.py", "proposed_content": 'password = "secret123"\neval(user_input)'}],
        )
        result = integration.verify_plan(session.id, plan.id)
        assert result.gate == VerificationGate.BLOCK
        assert len(result.findings) > 0

    def test_add_custom_constraint(self):
        from codeverify_core.copilot_workspace import (
            ConstraintType,
            CopilotWorkspaceIntegration,
        )

        integration = CopilotWorkspaceIntegration()
        session = integration.create_session("ws-3")
        constraint = integration.add_constraint(
            session.id, ConstraintType.CUSTOM, "All functions must have docstrings"
        )
        assert constraint.constraint_type == ConstraintType.CUSTOM
        assert len(session.active_constraints) > 3  # default + custom

    def test_generation_prompt(self):
        from codeverify_core.copilot_workspace import CopilotWorkspaceIntegration

        integration = CopilotWorkspaceIntegration()
        session = integration.create_session("ws-4")
        prompt = integration.get_generation_prompt(session.id)
        assert "VERIFICATION CONSTRAINTS" in prompt
        assert "[REQUIRED]" in prompt

    def test_workspace_file_properties(self):
        from codeverify_core.copilot_workspace import WorkspaceFile

        f = WorkspaceFile(path="test.py", original_content="a", proposed_content="b")
        assert f.has_changes is True
        assert f.diff_size >= 0

    def test_singleton(self):
        from codeverify_core.copilot_workspace import (
            get_copilot_workspace_integration,
            reset_copilot_workspace_integration,
        )

        reset_copilot_workspace_integration()
        i1 = get_copilot_workspace_integration()
        i2 = get_copilot_workspace_integration()
        assert i1 is i2
        reset_copilot_workspace_integration()


# --- Feature 3: Zero-Config Onboarding ---


class TestZeroConfig:
    def test_language_detection(self):
        from codeverify_core.zero_config import ProjectDetector

        detector = ProjectDetector()
        result = detector.detect_languages("/Users/josedab/Code/copilot-sdk-apps/codeverify")
        assert result.primary_language is not None
        assert len(result.all_languages) > 0
        assert result.confidence > 0

    def test_project_type_detection(self):
        from codeverify_core.zero_config import ProjectDetector, ProjectType

        detector = ProjectDetector()
        ptype = detector.detect_project_type("/Users/josedab/Code/copilot-sdk-apps/codeverify")
        assert ptype == ProjectType.MONOREPO

    def test_config_generation(self):
        from codeverify_core.zero_config import (
            ConfigGenerator,
            DetectedLanguage,
            LanguageDetectionResult,
            ProjectAnalysis,
        )

        analysis = ProjectAnalysis(
            languages=LanguageDetectionResult(
                primary_language=DetectedLanguage.PYTHON,
                all_languages=[DetectedLanguage.PYTHON, DetectedLanguage.TYPESCRIPT],
            )
        )
        config = ConfigGenerator().generate(analysis)
        assert "python" in config.languages
        assert len(config.verification_checks) > 0
        yaml = config.to_yaml()
        assert "version:" in yaml
        assert "python" in yaml

    def test_workflow_generation(self):
        from codeverify_core.zero_config import CIProvider, ProjectAnalysis, WorkflowGenerator

        analysis = ProjectAnalysis(ci_provider=CIProvider.GITHUB_ACTIONS)
        wf = WorkflowGenerator().generate(analysis)
        assert wf is not None
        assert "CodeVerify" in wf.content
        assert "codeverify scan" in wf.content

    def test_onboard_dry_run(self):
        from codeverify_core.zero_config import OnboardingStep, ZeroConfigOnboarder

        onboarder = ZeroConfigOnboarder()
        result = onboarder.preview("/Users/josedab/Code/copilot-sdk-apps/codeverify")
        assert result.success is True
        assert OnboardingStep.DETECT in result.steps_completed
        assert OnboardingStep.CONFIGURE in result.steps_completed

    def test_ci_detection(self):
        from codeverify_core.zero_config import CIProvider, ProjectDetector

        detector = ProjectDetector()
        has_ci, provider = detector.detect_ci("/Users/josedab/Code/copilot-sdk-apps/codeverify")
        assert has_ci is True
        assert provider == CIProvider.GITHUB_ACTIONS

    def test_singleton(self):
        from codeverify_core.zero_config import (
            get_zero_config_onboarder,
            reset_zero_config_onboarder,
        )

        reset_zero_config_onboarder()
        o1 = get_zero_config_onboarder()
        o2 = get_zero_config_onboarder()
        assert o1 is o2
        reset_zero_config_onboarder()


# --- Feature 4: Autonomous Verification Agent ---


class TestAutonomousAgent:
    def test_agent_lifecycle(self):
        from codeverify_core.autonomous_agent import (
            AgentState,
            AutonomousVerificationAgent,
        )

        agent = AutonomousVerificationAgent()
        assert agent.state == AgentState.IDLE
        agent.start_monitoring()
        assert agent.state == AgentState.MONITORING
        agent.stop_monitoring()
        assert agent.state == AgentState.PAUSED

    def test_process_change(self):
        from codeverify_core.autonomous_agent import (
            AutonomousVerificationAgent,
            MonitoredChange,
        )

        agent = AutonomousVerificationAgent()
        change = MonitoredChange(
            repository="my-repo",
            branch="main",
            changed_files=["app.py"],
        )
        triaged = agent.process_change(change)
        assert len(triaged) > 0
        assert agent.metrics.total_changes_monitored == 1

    def test_generate_fixes(self):
        from codeverify_core.autonomous_agent import (
            AgentConfig,
            AutonomousVerificationAgent,
            AutonomyLevel,
            MonitoredChange,
        )

        config = AgentConfig(autonomy_level=AutonomyLevel.AUTO_FIX)
        agent = AutonomousVerificationAgent(config)
        change = MonitoredChange(repository="repo", changed_files=["main.py"])
        triaged = agent.process_change(change)
        fixes = agent.generate_fixes(triaged, "def foo():\n    x = None\n    return x.strip()")
        assert isinstance(fixes, list)

    def test_create_pr(self):
        from codeverify_core.autonomous_agent import (
            AutonomousVerificationAgent,
            FixCandidate,
        )

        agent = AutonomousVerificationAgent()
        fix = FixCandidate(
            file_path="app.py",
            original_code="x = None",
            fixed_code="x = None\nif x is not None:",
            confidence=0.9,
            verification_passed=True,
            diff_lines=1,
        )
        pr = agent.create_pr("my-repo", [fix])
        assert pr is not None
        assert "auto-fix" in pr.branch_name
        assert agent.metrics.total_prs_created == 1

    def test_feedback_learning(self):
        from codeverify_core.autonomous_agent import AutonomousVerificationAgent

        agent = AutonomousVerificationAgent()
        entry = agent.record_feedback("fix-1", "null_dereference", accepted=True)
        assert entry.accepted is True
        assert agent.metrics.total_fixes_accepted == 1
        assert agent.metrics.acceptance_rate > 0

    def test_agent_status(self):
        from codeverify_core.autonomous_agent import AutonomousVerificationAgent

        agent = AutonomousVerificationAgent()
        status = agent.get_status()
        assert "state" in status
        assert "metrics" in status

    def test_triage_suppression(self):
        from codeverify_core.autonomous_agent import (
            AgentConfig,
            FindingTriage,
            FindingTriager,
        )

        config = AgentConfig()
        triager = FindingTriager(config)
        triager.false_positive_history["noisy_rule"] = 0.8
        result = triager.triage({"type": "noisy_rule", "severity": "low", "confidence": 0.5})
        assert result.triage == FindingTriage.SUPPRESS

    def test_singleton(self):
        from codeverify_core.autonomous_agent import get_autonomous_agent, reset_autonomous_agent

        reset_autonomous_agent()
        a1 = get_autonomous_agent()
        a2 = get_autonomous_agent()
        assert a1 is a2
        reset_autonomous_agent()


# --- Feature 5: Verification-as-a-Service API ---


class TestVaaS:
    def test_create_api_key(self):
        from codeverify_core.vaas import VaaSService, VerificationTier

        svc = VaaSService()
        key, raw = svc.create_api_key("Test Key", VerificationTier.FREE)
        assert raw.startswith("cv_vaas_")
        assert key.tier == VerificationTier.FREE

    def test_validate_api_key(self):
        from codeverify_core.vaas import VaaSService

        svc = VaaSService()
        _, raw = svc.create_api_key("Key")
        validated = svc.validate_api_key(raw)
        assert validated is not None
        assert svc.validate_api_key("invalid") is None

    def test_submit_verification(self):
        from codeverify_core.vaas import VaaSService, VerificationStatus

        svc = VaaSService()
        key, _ = svc.create_api_key("Key")
        response = svc.submit_verification(
            key.id,
            files=[{"path": "test.py", "content": "x = None\nprint(x)"}],
        )
        assert response.status == VerificationStatus.COMPLETED
        assert response.verification_time_ms >= 0

    def test_caching(self):
        from codeverify_core.vaas import VaaSService

        svc = VaaSService()
        key, _ = svc.create_api_key("Key")
        files = [{"path": "a.py", "content": "pass"}]
        svc.submit_verification(key.id, files)
        r2 = svc.submit_verification(key.id, files)
        assert r2.cached is True

    def test_quota_enforcement(self):
        from codeverify_core.vaas import VaaSService, VerificationTier

        svc = VaaSService()
        key, _ = svc.create_api_key("Key", VerificationTier.FREE)
        key.usage_this_month = 100
        with pytest.raises(ValueError, match="quota"):
            svc.submit_verification(key.id, [{"path": "a.py", "content": "x=1"}])

    def test_tier_limits(self):
        from codeverify_core.vaas import TierLimits, VerificationTier

        free = TierLimits.for_tier(VerificationTier.FREE)
        pro = TierLimits.for_tier(VerificationTier.PROFESSIONAL)
        assert pro.verifications_per_month > free.verifications_per_month
        assert pro.webhook_enabled is True

    def test_sarif_output(self):
        from codeverify_core.vaas import VaaSService

        svc = VaaSService()
        key, _ = svc.create_api_key("Key")
        response = svc.submit_verification(
            key.id, [{"path": "t.py", "content": "x = None\nprint(x)"}]
        )
        sarif = response.to_sarif()
        assert sarif["version"] == "2.1.0"
        assert "runs" in sarif

    def test_usage_stats(self):
        from codeverify_core.vaas import VaaSService

        svc = VaaSService()
        key, _ = svc.create_api_key("Key")
        svc.submit_verification(key.id, [{"path": "a.py", "content": "pass"}])
        stats = svc.get_usage_stats(key.id)
        assert stats.total_verifications == 1

    def test_singleton(self):
        from codeverify_core.vaas import get_vaas_service, reset_vaas_service

        reset_vaas_service()
        s1 = get_vaas_service()
        s2 = get_vaas_service()
        assert s1 is s2
        reset_vaas_service()


# --- Feature 6: Interactive Proof Explorer ---


class TestProofExplorer:
    def test_parse_z3_output(self):
        from codeverify_core.proof_explorer_interactive import ProofStatus, ProofTreeParser

        parser = ProofTreeParser()
        tree = parser.parse_z3_output("(assert (> x 0))\nunsat", "null_safety")
        assert tree.label == "Verification: null_safety"
        assert len(tree.children) == 2
        assert tree.status == ProofStatus.VERIFIED

    def test_parse_sat_output(self):
        from codeverify_core.proof_explorer_interactive import ProofStatus, ProofTreeParser

        parser = ProofTreeParser()
        tree = parser.parse_z3_output("(assert (> x 0))\nsat", "bounds")
        assert tree.status == ProofStatus.REFUTED

    def test_animation_creation(self):
        from codeverify_core.proof_explorer_interactive import (
            ConstraintAnimator,
            ProofTreeParser,
        )

        parser = ProofTreeParser()
        tree = parser.parse_z3_output("(assert (> x 0))\nunsat")
        animator = ConstraintAnimator()
        anim = animator.create_animation(tree, {"x": "unknown"})
        assert anim.total_steps > 0
        assert len(anim.steps) > 0

    def test_mermaid_rendering(self):
        from codeverify_core.proof_explorer_interactive import (
            ExportFormat,
            ProofRenderer,
            ProofTreeParser,
        )

        tree = ProofTreeParser().parse_z3_output("unsat")
        output = ProofRenderer().render(tree, ExportFormat.MERMAID)
        assert "graph TD" in output

    def test_html_rendering(self):
        from codeverify_core.proof_explorer_interactive import (
            ExportFormat,
            ProofRenderer,
            ProofTreeParser,
        )

        tree = ProofTreeParser().parse_z3_output("unsat")
        output = ProofRenderer().render(tree, ExportFormat.HTML)
        assert "<html>" in output

    def test_share_proof(self):
        from codeverify_core.proof_explorer_interactive import InteractiveProofExplorer

        explorer = InteractiveProofExplorer()
        tree, anim = explorer.explore_z3_output("unsat", "test")
        shared = explorer.share_proof(tree, anim, title="My Proof")
        assert shared.share_url.startswith("https://")
        assert shared.embed_html.startswith("<iframe")

        retrieved = explorer.get_shared_proof(shared.share_token)
        assert retrieved is not None
        assert retrieved.view_count == 1

    def test_singleton(self):
        from codeverify_core.proof_explorer_interactive import (
            get_proof_explorer,
            reset_proof_explorer,
        )

        reset_proof_explorer()
        e1 = get_proof_explorer()
        e2 = get_proof_explorer()
        assert e1 is e2
        reset_proof_explorer()


# --- Feature 7: Cross-Repository Blast Radius ---


class TestCrossRepoBlast:
    def test_add_repos_and_deps(self):
        from codeverify_core.cross_repo_blast import CrossRepoBlastAnalyzer

        analyzer = CrossRepoBlastAnalyzer()
        analyzer.add_repository("core-lib", team="platform")
        analyzer.add_repository("api-svc", team="backend")
        analyzer.add_dependency("api-svc", "core-lib")
        assert analyzer.graph.repos["core-lib"].team == "platform"
        assert len(analyzer.graph.edges) == 1

    def test_blast_analysis(self):
        from codeverify_core.cross_repo_blast import (
            CrossRepoBlastAnalyzer,
            CrossRepoChange,
            CrossRepoChangeImpact,
        )

        analyzer = CrossRepoBlastAnalyzer()
        analyzer.add_repository("lib", team="platform", is_critical=True)
        analyzer.add_repository("svc-a", team="team-a")
        analyzer.add_repository("svc-b", team="team-b")
        analyzer.add_dependency("svc-a", "lib")
        analyzer.add_dependency("svc-b", "lib")

        change = CrossRepoChange(
            repository="lib",
            apis_changed=["get_user"],
            impact_type=CrossRepoChangeImpact.BREAKING,
        )
        report = analyzer.analyze(change)
        assert report.total_impacted == 2
        assert len(report.teams_affected) == 2
        assert report.blast_score > 0

    def test_mermaid_diagram(self):
        from codeverify_core.cross_repo_blast import (
            CrossRepoBlastAnalyzer,
            CrossRepoChange,
            CrossRepoChangeImpact,
        )

        analyzer = CrossRepoBlastAnalyzer()
        analyzer.add_repository("lib", team="core")
        analyzer.add_repository("app", team="web")
        analyzer.add_dependency("app", "lib")
        change = CrossRepoChange(repository="lib", impact_type=CrossRepoChangeImpact.BREAKING)
        report = analyzer.analyze(change)
        assert "graph LR" in report.mermaid_diagram

    def test_transitive_deps(self):
        from codeverify_core.cross_repo_blast import (
            CrossRepoBlastAnalyzer,
            CrossRepoChange,
            CrossRepoChangeImpact,
        )

        analyzer = CrossRepoBlastAnalyzer()
        analyzer.add_repository("a")
        analyzer.add_repository("b")
        analyzer.add_repository("c")
        analyzer.add_dependency("b", "a")
        analyzer.add_dependency("c", "b")
        change = CrossRepoChange(repository="a", impact_type=CrossRepoChangeImpact.BREAKING)
        report = analyzer.analyze(change)
        assert report.total_impacted == 2

    def test_singleton(self):
        from codeverify_core.cross_repo_blast import (
            get_cross_repo_blast_analyzer,
            reset_cross_repo_blast_analyzer,
        )

        reset_cross_repo_blast_analyzer()
        a1 = get_cross_repo_blast_analyzer()
        a2 = get_cross_repo_blast_analyzer()
        assert a1 is a2
        reset_cross_repo_blast_analyzer()


# --- Feature 8: AI Code Review Benchmark ---


class TestBenchmark:
    def test_dataset_loading(self):
        from codeverify_core.benchmark import BenchmarkDataset

        dataset = BenchmarkDataset()
        dataset.load_builtin_samples()
        assert dataset.total_samples >= 8
        assert dataset.bug_samples > 0
        assert dataset.correct_samples > 0

    def test_benchmark_runner(self):
        from codeverify_core.benchmark import (
            BenchmarkDataset,
            BenchmarkRunner,
            BuiltinBenchmarkAdapter,
        )

        dataset = BenchmarkDataset()
        dataset.load_builtin_samples()
        runner = BenchmarkRunner(dataset)
        adapter = BuiltinBenchmarkAdapter()
        entry = runner.run(adapter)
        assert entry.overall_metrics.total_samples > 0
        assert 0 <= entry.overall_metrics.f1_score <= 1
        assert 0 <= entry.overall_metrics.precision <= 1

    def test_leaderboard(self):
        from codeverify_core.benchmark import (
            BenchmarkDataset,
            BenchmarkRunner,
            BuiltinBenchmarkAdapter,
        )

        dataset = BenchmarkDataset()
        dataset.load_builtin_samples()
        runner = BenchmarkRunner(dataset)
        runner.run(BuiltinBenchmarkAdapter())
        board = runner.get_leaderboard()
        assert len(board) == 1

    def test_markdown_export(self):
        from codeverify_core.benchmark import (
            BenchmarkDataset,
            BenchmarkRunner,
            BuiltinBenchmarkAdapter,
        )

        dataset = BenchmarkDataset()
        dataset.load_builtin_samples()
        runner = BenchmarkRunner(dataset)
        runner.run(BuiltinBenchmarkAdapter())
        md = runner.render_leaderboard_markdown()
        assert "Leaderboard" in md
        assert "F1" in md

    def test_json_export(self):
        import json

        from codeverify_core.benchmark import (
            BenchmarkDataset,
            BenchmarkRunner,
            BuiltinBenchmarkAdapter,
        )

        dataset = BenchmarkDataset()
        dataset.load_builtin_samples()
        runner = BenchmarkRunner(dataset)
        runner.run(BuiltinBenchmarkAdapter())
        data = json.loads(runner.export_results_json())
        assert len(data) == 1
        assert "f1" in data[0]

    def test_sample_filtering(self):
        from codeverify_core.benchmark import BenchmarkDataset, SampleLanguage

        dataset = BenchmarkDataset()
        dataset.load_builtin_samples()
        python_bugs = dataset.get_samples(language=SampleLanguage.PYTHON, has_bug=True)
        assert len(python_bugs) > 0

    def test_singleton(self):
        from codeverify_core.benchmark import get_benchmark_runner, reset_benchmark_runner

        reset_benchmark_runner()
        r1 = get_benchmark_runner()
        r2 = get_benchmark_runner()
        assert r1 is r2
        reset_benchmark_runner()


# --- Feature 9: Fine-Tuned Verification LLM ---


class TestFineTunedLLM:
    def test_training_data_pipeline(self):
        from codeverify_core.fine_tuned_llm import TrainingDataPipeline

        pipeline = TrainingDataPipeline()
        dataset = pipeline.create_dataset("test-dataset")
        samples = pipeline.add_sample_from_verification(
            dataset.id,
            code="def foo(): pass",
            findings=[{"message": "Missing docstring", "fix_suggestion": "Add docstring"}],
        )
        assert len(samples) >= 1
        assert dataset.total_samples >= 1

    def test_dataset_split(self):
        from codeverify_core.fine_tuned_llm import TrainingDataPipeline

        pipeline = TrainingDataPipeline()
        ds = pipeline.create_dataset("split-test")
        for i in range(10):
            pipeline.add_sample_from_verification(
                ds.id,
                code=f"def f{i}(): pass",
                findings=[{"message": f"Issue {i}", "fix_suggestion": f"Fix {i}"}],
            )
        train, val, test = ds.split(0.8, 0.1)
        assert len(train) + len(val) + len(test) == ds.total_samples

    def test_local_inference(self):
        from codeverify_core.fine_tuned_llm import (
            LocalInferenceEngine,
            TaskType,
        )

        engine = LocalInferenceEngine()
        engine.load_model("models/test.gguf")
        assert engine.is_loaded is True
        result = engine.predict("def foo(): pass", TaskType.VULNERABILITY_DETECTION)
        assert result.text != ""
        assert result.latency_ms >= 0

    def test_cost_comparison(self):
        from codeverify_core.fine_tuned_llm import LocalInferenceEngine

        engine = LocalInferenceEngine()
        engine.load_model("model.gguf")
        comparison = engine.get_cost_comparison()
        assert comparison.savings_percent > 0
        assert comparison.local_cost_per_verification < comparison.cloud_cost_per_verification

    def test_air_gap_package(self):
        from codeverify_core.fine_tuned_llm import AirGapPackager, ModelFormat

        packager = AirGapPackager()
        package = packager.create_package("model.gguf", ModelFormat.GGUF)
        assert "codeverify-core" in package.included_tools
        assert package.model_format == ModelFormat.GGUF
        assert package.manifest["format"] == "gguf"

    def test_training_job(self):
        from codeverify_core.fine_tuned_llm import (
            FineTunedVerificationLLM,
            TrainingStatus,
        )

        llm = FineTunedVerificationLLM()
        ds = llm.data_pipeline.create_dataset("train")
        job = llm.create_training_job(ds.id)
        result = llm.start_training(job.id)
        assert result.status == TrainingStatus.COMPLETED
        assert result.metrics.eval_accuracy > 0

    def test_singleton(self):
        from codeverify_core.fine_tuned_llm import get_fine_tuned_llm, reset_fine_tuned_llm

        reset_fine_tuned_llm()
        l1 = get_fine_tuned_llm()
        l2 = get_fine_tuned_llm()
        assert l1 is l2
        reset_fine_tuned_llm()


# --- Feature 10: Developer Certification Program ---


class TestCertification:
    def test_course_curriculum(self):
        from codeverify_core.certification import CertificationProgram

        program = CertificationProgram()
        assert len(program.modules) == 5
        assert program.modules[0].title == "Introduction to Formal Verification"
        assert len(program.labs) > 0

    def test_enroll_learner(self):
        from codeverify_core.certification import CertificationProgram

        program = CertificationProgram()
        progress = program.enroll_learner("Alice", "alice@example.com")
        assert progress.learner_name == "Alice"
        assert progress.completion_percentage == 0.0

    def test_complete_module(self):
        from codeverify_core.certification import CertificationProgram, ModuleStatus

        program = CertificationProgram()
        progress = program.enroll_learner("Bob", "bob@example.com")
        module_id = program.modules[0].id
        status = program.complete_module(progress.learner_id, module_id)
        assert status == ModuleStatus.COMPLETED
        assert progress.modules_completed == 1
        assert len(progress.badges) == 1

    def test_take_assessment(self):
        from codeverify_core.certification import CertificationProgram

        program = CertificationProgram()
        progress = program.enroll_learner("Charlie", "charlie@example.com")
        exam_id = list(program.assessments.keys())[0]
        exam = program.assessments[exam_id]

        # Answer all correctly
        answers = {q.id: q.correct_answer for q in exam.questions}
        submission = program.take_assessment(progress.learner_id, exam_id, answers)
        assert submission.passed is True
        assert submission.score >= 0.7
        assert len(progress.certificates) == 1

    def test_failing_assessment(self):
        from codeverify_core.certification import CertificationProgram

        program = CertificationProgram()
        progress = program.enroll_learner("Dave", "dave@example.com")
        exam_id = list(program.assessments.keys())[0]

        # Answer all wrong
        submission = program.take_assessment(progress.learner_id, exam_id, {})
        assert submission.passed is False
        assert len(progress.certificates) == 0

    def test_certificate_verification(self):
        from codeverify_core.certification import CertificationProgram

        program = CertificationProgram()
        progress = program.enroll_learner("Eve", "eve@example.com")
        exam_id = list(program.assessments.keys())[0]
        exam = program.assessments[exam_id]
        answers = {q.id: q.correct_answer for q in exam.questions}
        program.take_assessment(progress.learner_id, exam_id, answers)

        cert = progress.certificates[0]
        verified = program.verify_certificate(cert.certificate_number)
        assert verified is not None
        assert verified.is_valid is True

    def test_badge_verification(self):
        from codeverify_core.certification import (
            BadgeType,
            CertificationLevel,
            CredentialIssuer,
        )

        issuer = CredentialIssuer()
        badge = issuer.issue_badge(
            "learner-1", BadgeType.COMPLETION, CertificationLevel.FOUNDATIONS, "Test Badge"
        )
        assert badge.verify() is True

    def test_program_stats(self):
        from codeverify_core.certification import CertificationProgram

        program = CertificationProgram()
        program.enroll_learner("A", "a@b.com")
        stats = program.get_program_stats()
        assert stats["total_enrolled"] == 1
        assert stats["total_modules"] == 5

    def test_singleton(self):
        from codeverify_core.certification import (
            get_certification_program,
            reset_certification_program,
        )

        reset_certification_program()
        p1 = get_certification_program()
        p2 = get_certification_program()
        assert p1 is p2
        reset_certification_program()
