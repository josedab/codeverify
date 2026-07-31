"""Tests for incremental verification cache enhancements.

Tests batch verification, dependency-aware invalidation, and Prometheus metrics.
"""

import pytest

from codeverify_core.verification_cache import (
    ASTFingerprinter,
    BatchCachedVerifier,
    BatchVerificationItem,
    CacheConfig,
    CachePrometheusMetrics,
    DependencyAwareCacheInvalidator,
    VerificationCache,
)


class FakeVerifier:
    """Minimal fake verifier for testing."""

    def check_null_dereference(self, _var_name, can_be_null, null_check_exists):
        return {
            "check": "null_safety",
            "satisfiable": can_be_null and not null_check_exists,
            "proof_time_ms": 42.0,
        }

    def check_array_bounds(self, _index_var, _index_range, _array_length):
        return {"check": "array_bounds", "satisfiable": False, "proof_time_ms": 30.0}

    def check_integer_overflow(self, _var_name, _op, _r1, _r2, _bw):
        return {"check": "integer_overflow", "satisfiable": False, "proof_time_ms": 55.0}

    def check_division_by_zero(self, _divisor_var, _divisor_range):
        return {"check": "division_by_zero", "satisfiable": True, "proof_time_ms": 20.0}

    def verify_condition(self, _condition, _description=""):
        return {"check": "custom", "satisfiable": False, "proof_time_ms": 10.0}


# =============================================================================
# Batch Verification Tests
# =============================================================================


class TestBatchCachedVerifier:
    @pytest.fixture
    def batch_verifier(self):
        cache = VerificationCache(CacheConfig(enabled=True))
        return BatchCachedVerifier(FakeVerifier(), cache)

    def test_batch_all_cache_misses(self, batch_verifier):
        items = [
            BatchVerificationItem(
                "def f(): pass",
                "f",
                "a.py",
                "python",
                "null_safety",
                {"var_name": "x", "can_be_null": True, "null_check_exists": False},
            ),
            BatchVerificationItem(
                "def g(): pass",
                "g",
                "b.py",
                "python",
                "array_bounds",
                {"index_var": "i", "array_length": 5},
            ),
        ]
        result = batch_verifier.verify_batch(items)
        assert result.cache_misses == 2
        assert result.cache_hits == 0
        assert len(result.results) == 2
        assert result.results[0]["cached"] is False

    def test_batch_with_cache_hits(self, batch_verifier):
        items = [
            BatchVerificationItem(
                "def f(): pass",
                "f",
                "a.py",
                "python",
                "null_safety",
                {"var_name": "x", "can_be_null": True, "null_check_exists": False},
            ),
        ]
        # First run populates cache
        batch_verifier.verify_batch(items)
        # Second run hits cache
        result = batch_verifier.verify_batch(items)
        assert result.cache_hits == 1
        assert result.cache_misses == 0
        assert result.results[0]["cached"] is True

    def test_batch_force_bypasses_cache(self, batch_verifier):
        items = [
            BatchVerificationItem(
                "def f(): pass",
                "f",
                "a.py",
                "python",
                "null_safety",
                {"var_name": "x", "can_be_null": True, "null_check_exists": False},
            ),
        ]
        batch_verifier.verify_batch(items)
        result = batch_verifier.verify_batch(items, force=True)
        assert result.cache_misses == 1
        assert result.cache_hits == 0

    def test_batch_mixed_verification_types(self, batch_verifier):
        items = [
            BatchVerificationItem(
                "def f(): pass",
                "f",
                "a.py",
                "python",
                "null_safety",
                {"var_name": "x", "can_be_null": True, "null_check_exists": False},
            ),
            BatchVerificationItem(
                "def f(): pass", "f", "a.py", "python", "division_by_zero", {"divisor_var": "d"}
            ),
            BatchVerificationItem(
                "def g(): pass",
                "g",
                "a.py",
                "python",
                "custom",
                {"condition": "(assert true)", "description": "test"},
            ),
        ]
        result = batch_verifier.verify_batch(items)
        assert len(result.results) == 3
        assert result.items_verified == 3

    def test_batch_hit_rate(self, batch_verifier):
        items = [
            BatchVerificationItem(
                "def f(): pass",
                "f",
                "a.py",
                "python",
                "null_safety",
                {"var_name": "x", "can_be_null": True, "null_check_exists": False},
            ),
            BatchVerificationItem(
                "def g(): pass",
                "g",
                "b.py",
                "python",
                "null_safety",
                {"var_name": "y", "can_be_null": False, "null_check_exists": True},
            ),
        ]
        batch_verifier.verify_batch(items)
        # Re-run with one new, one cached
        items2 = [
            items[0],  # cached
            BatchVerificationItem(
                "def h(): pass",
                "h",
                "c.py",
                "python",
                "null_safety",
                {"var_name": "z", "can_be_null": True, "null_check_exists": False},
            ),
        ]
        result = batch_verifier.verify_batch(items2)
        assert result.cache_hits == 1
        assert result.cache_misses == 1
        assert result.hit_rate == 0.5


# =============================================================================
# Dependency-Aware Invalidation Tests
# =============================================================================


class TestDependencyAwareCacheInvalidator:
    @pytest.fixture
    def setup(self):
        cache = VerificationCache(CacheConfig(enabled=True))
        inv = DependencyAwareCacheInvalidator(cache)
        fp = ASTFingerprinter()
        return cache, inv, fp

    def test_register_and_invalidate_single(self, setup):
        cache, inv, fp = setup
        fingerprint = fp.fingerprint_function("def f(): pass", "f", "python")
        cache.put(fingerprint, "f", "a.py", "null_safety", {"ok": True}, 10.0)
        inv.register_function("mod.f", fingerprint)

        count = inv.invalidate_with_dependents("mod.f")
        assert count >= 1
        assert cache.get(fingerprint, "null_safety") is None

    def test_transitive_invalidation(self, setup):
        cache, inv, fp = setup
        fp_a = fp.fingerprint_function("def a(): pass", "a", "python")
        fp_b = fp.fingerprint_function("def b(): a()", "b", "python")
        fp_c = fp.fingerprint_function("def c(): b()", "c", "python")

        cache.put(fp_a, "a", "f.py", "null_safety", {"ok": True}, 5.0)
        cache.put(fp_b, "b", "f.py", "null_safety", {"ok": True}, 5.0)
        cache.put(fp_c, "c", "f.py", "null_safety", {"ok": True}, 5.0)

        inv.register_function("mod.a", fp_a)
        inv.register_function("mod.b", fp_b, dependencies=["mod.a"])
        inv.register_function("mod.c", fp_c, dependencies=["mod.b"])

        # Changing a should invalidate b and c
        dependents = inv.get_transitive_dependents("mod.a")
        assert "mod.b" in dependents
        assert "mod.c" in dependents

        count = inv.invalidate_with_dependents("mod.a")
        assert count >= 3

    def test_dependency_graph_export(self, setup):
        _, inv, _ = setup
        inv.register_dependency("mod.b", "mod.a")
        inv.register_dependency("mod.c", "mod.b")
        graph = inv.get_dependency_graph()
        assert "mod.b" in graph
        assert "mod.a" in graph["mod.b"]

    def test_on_file_changed(self, setup):
        cache, inv, fp = setup
        fp_a = fp.fingerprint_function("def a(): pass", "a", "python")
        cache.put(fp_a, "a", "f.py", "null_safety", {"ok": True}, 5.0)
        inv.register_function("mod.a", fp_a)

        count = inv.on_file_changed("f.py", ["mod.a"])
        assert count >= 1


# =============================================================================
# Prometheus Metrics Tests
# =============================================================================


class TestCachePrometheusMetrics:
    def test_export_text_format(self):
        cache = VerificationCache(CacheConfig(enabled=True))
        cache.put("fp1", "f", "a.py", "null_safety", {"ok": True}, 50.0)
        cache.get("fp1", "null_safety")  # hit
        cache.get("fp_miss", "null_safety")  # miss

        metrics = CachePrometheusMetrics(cache)
        text = metrics.export()

        assert "codeverify_verification_cache_hits_total 1" in text
        assert "codeverify_verification_cache_misses_total 1" in text
        assert "codeverify_verification_cache_hit_rate 0.5" in text
        assert "codeverify_verification_cache_entries 1" in text
        assert "# TYPE" in text

    def test_export_as_dict(self):
        cache = VerificationCache(CacheConfig(enabled=True))
        metrics = CachePrometheusMetrics(cache)
        d = metrics.export_as_dict()

        assert d["namespace"] == "codeverify_verification_cache"
        assert "hits_total" in d["metrics"]
        assert "backend" in d["metrics"]

    def test_metrics_after_operations(self):
        cache = VerificationCache(CacheConfig(enabled=True))
        cache.put("fp1", "f", "a.py", "null_safety", {"ok": True}, 100.0)
        cache.get("fp1", "null_safety")
        cache.get("fp1", "null_safety")

        metrics = CachePrometheusMetrics(cache)
        d = metrics.export_as_dict()
        assert d["metrics"]["hits_total"] == 2
        assert d["metrics"]["time_saved_ms_total"] == 200.0
