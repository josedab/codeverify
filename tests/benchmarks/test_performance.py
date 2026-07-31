"""Performance benchmarks for CodeVerify.

Run with: pytest tests/benchmarks/ -v --benchmark-enable
"""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pytest_benchmark") is None,
    reason="pytest-benchmark is not installed",
)


class TestImportPerformance:
    def test_core_import_time(self, benchmark):
        import importlib

        def do_import():
            import codeverify_core

            importlib.reload(codeverify_core)

        benchmark(do_import)

    def test_single_module_import(self, benchmark):
        import importlib

        def do_import():
            mod = importlib.import_module("codeverify_core.agentic_orchestrator")
            importlib.reload(mod)

        benchmark(do_import)


class TestVerificationLatency:
    def test_protocol_verify(self, benchmark):
        from codeverify_core.verification_protocol import VerificationProtocolServer, VerifyRequest

        server = VerificationProtocolServer()
        req = VerifyRequest(files=[{"path": "a.py", "content": "def f(x): return x + 1\n"}])
        benchmark(server.verify, req)

    def test_orchestrator_review(self, benchmark):
        from codeverify_core.agentic_orchestrator import AgenticReviewOrchestrator, PRContext

        orch = AgenticReviewOrchestrator(budget_cents=50.0)
        ctx = PRContext(pr_id="1", changed_files=[{"path": "app.py"}])
        benchmark(orch.review, ctx)

    def test_streaming_ide_verify(self, benchmark):
        from codeverify_core.streaming_ide import StreamingIDEVerificationService

        svc = StreamingIDEVerificationService()
        benchmark(svc.verify_file, "t.py", "def f():\n    return 1\n", "python")

    def test_drift_scan(self, benchmark):
        from codeverify_core.drift_monitor import DriftMonitorService

        svc = DriftMonitorService()
        code = {"a.py": "def f(x): return x\n"}
        svc.set_baseline("r", code)
        benchmark(svc.scan, "r", code)


class TestThroughput:
    def test_search_throughput(self, benchmark):
        from codeverify_core.verification_search import (
            CodeEntity,
            VerificationSearchService,
            VerificationStatus,
        )

        svc = VerificationSearchService()
        for i in range(100):
            svc.index_entity(
                CodeEntity(
                    file_path=f"f{i}.py",
                    function_name=f"fn{i}",
                    verification_status=VerificationStatus.VERIFIED
                    if i % 2 == 0
                    else VerificationStatus.UNVERIFIED,
                )
            )
        benchmark(svc.search, "unverified functions in f")

    def test_multimodal_batch(self, benchmark):
        from codeverify_core.multimodal_verify import MultiModalVerificationService

        svc = MultiModalVerificationService()
        files = {f"c{i}.yml": f"password=s{i}\n" for i in range(10)}
        benchmark(svc.verify_batch, files)
