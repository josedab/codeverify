"""Tests for verification_queue module."""

from __future__ import annotations

import time

from codeverify_core.verification_queue import (
    JobPriority,
    JobStatus,
    RateLimitConfig,
    VerificationJob,
    VerificationQueue,
)


class TestVerificationJob:
    def test_dedup_key_same_for_same_repo_commit(self):
        j1 = VerificationJob(priority=0, repo_id="r1", commit_sha="abc")
        j2 = VerificationJob(priority=0, repo_id="r1", commit_sha="abc")
        assert j1.dedup_key == j2.dedup_key

    def test_dedup_key_differs_for_different_commit(self):
        j1 = VerificationJob(priority=0, repo_id="r1", commit_sha="abc")
        j2 = VerificationJob(priority=0, repo_id="r1", commit_sha="def")
        assert j1.dedup_key != j2.dedup_key


class TestVerificationQueue:
    def test_enqueue_and_dequeue(self):
        q = VerificationQueue()
        jid = q.enqueue(VerificationJob(priority=JobPriority.NORMAL, repo_id="r1", commit_sha="a"))
        job = q.dequeue()
        assert job is not None
        assert job.job_id == jid
        assert job.status == JobStatus.RUNNING

    def test_priority_ordering(self):
        q = VerificationQueue()
        q.enqueue(VerificationJob(priority=JobPriority.LOW, repo_id="r1", commit_sha="low"))
        q.enqueue(VerificationJob(priority=JobPriority.CRITICAL, repo_id="r2", commit_sha="crit"))
        q.enqueue(VerificationJob(priority=JobPriority.NORMAL, repo_id="r3", commit_sha="norm"))
        j1 = q.dequeue()
        q.dequeue()
        j3 = q.dequeue()
        assert j1.commit_sha == "crit"
        assert j3.commit_sha == "low"

    def test_deduplication(self):
        q = VerificationQueue()
        id1 = q.enqueue(VerificationJob(priority=0, repo_id="r1", commit_sha="same"))
        id2 = q.enqueue(VerificationJob(priority=0, repo_id="r1", commit_sha="same"))
        assert id1 == id2
        assert q.stats.deduplicated == 1

    def test_complete_job(self):
        q = VerificationQueue()
        q.enqueue(VerificationJob(priority=0, repo_id="r1", commit_sha="a"))
        job = q.dequeue()
        q.complete(job.job_id, result={"findings": []})
        assert job.status == JobStatus.COMPLETED
        assert q.stats.completed == 1

    def test_failed_job(self):
        q = VerificationQueue()
        q.enqueue(VerificationJob(priority=0, repo_id="r1", commit_sha="a"))
        job = q.dequeue()
        q.complete(job.job_id, failed=True)
        assert job.status == JobStatus.FAILED
        assert q.stats.failed == 1

    def test_timeout_detection(self):
        q = VerificationQueue()
        q.enqueue(VerificationJob(priority=0, repo_id="r1", commit_sha="a", timeout_seconds=0))
        q.dequeue()
        time.sleep(0.01)
        timed_out = q.check_timeouts()
        assert len(timed_out) == 1
        assert q.stats.timed_out == 1

    def test_rate_limiting_concurrent(self):
        limit = RateLimitConfig(max_concurrent=1, max_per_minute=100)
        q = VerificationQueue(default_rate_limit=limit)
        q.enqueue(VerificationJob(priority=0, org_id="org1", repo_id="r1", commit_sha="a"))
        q.enqueue(VerificationJob(priority=0, org_id="org1", repo_id="r2", commit_sha="b"))
        j1 = q.dequeue()
        j2 = q.dequeue()
        assert j1 is not None
        assert j2 is None  # Rate limited

    def test_rate_limited_job_available_after_completion(self):
        limit = RateLimitConfig(max_concurrent=1, max_per_minute=100)
        q = VerificationQueue(default_rate_limit=limit)
        q.enqueue(VerificationJob(priority=0, org_id="org1", repo_id="r1", commit_sha="a"))
        q.enqueue(VerificationJob(priority=0, org_id="org1", repo_id="r2", commit_sha="b"))
        j1 = q.dequeue()
        q.complete(j1.job_id)
        j2 = q.dequeue()
        assert j2 is not None

    def test_different_orgs_not_rate_limited(self):
        limit = RateLimitConfig(max_concurrent=1, max_per_minute=100)
        q = VerificationQueue(default_rate_limit=limit)
        q.enqueue(VerificationJob(priority=0, org_id="org1", repo_id="r1", commit_sha="a"))
        q.enqueue(VerificationJob(priority=0, org_id="org2", repo_id="r2", commit_sha="b"))
        j1 = q.dequeue()
        j2 = q.dequeue()
        assert j1 is not None and j2 is not None

    def test_get_job(self):
        q = VerificationQueue()
        jid = q.enqueue(VerificationJob(priority=0, repo_id="r1", commit_sha="a"))
        job = q.get_job(jid)
        assert job is not None
        assert job.repo_id == "r1"

    def test_pending_count(self):
        q = VerificationQueue()
        q.enqueue(VerificationJob(priority=0, repo_id="r1", commit_sha="a"))
        q.enqueue(VerificationJob(priority=0, repo_id="r2", commit_sha="b"))
        assert q.pending_count == 2
        q.dequeue()
        assert q.pending_count == 1

    def test_empty_dequeue(self):
        q = VerificationQueue()
        assert q.dequeue() is None

    def test_per_org_rate_limit(self):
        q = VerificationQueue()
        q.set_rate_limit("org1", RateLimitConfig(max_concurrent=2))
        q.set_rate_limit("org2", RateLimitConfig(max_concurrent=1))
        q.enqueue(VerificationJob(priority=0, org_id="org1", repo_id="r1", commit_sha="a"))
        q.enqueue(VerificationJob(priority=0, org_id="org1", repo_id="r2", commit_sha="b"))
        q.enqueue(VerificationJob(priority=0, org_id="org2", repo_id="r3", commit_sha="c"))
        q.enqueue(VerificationJob(priority=0, org_id="org2", repo_id="r4", commit_sha="d"))
        j1 = q.dequeue()
        j2 = q.dequeue()
        j3 = q.dequeue()
        j4 = q.dequeue()
        # org1 allows 2 concurrent, org2 allows 1
        running = [j for j in [j1, j2, j3, j4] if j is not None]
        assert len(running) == 3
