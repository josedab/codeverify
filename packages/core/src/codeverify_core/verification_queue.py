"""Verification Job Queue.

Priority queue with fair scheduling across tenants, job deduplication,
rate limiting per organization, and timeout handling.
"""

from __future__ import annotations

import hashlib
import heapq
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobPriority(int, Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    DEDUPLICATED = "deduplicated"


@dataclass(order=True)
class VerificationJob:
    """A verification job in the queue."""

    priority: int = field(compare=True)
    enqueued_at: float = field(compare=True, default_factory=time.time)
    job_id: str = field(compare=False, default="")
    org_id: str = field(compare=False, default="")
    repo_id: str = field(compare=False, default="")
    commit_sha: str = field(compare=False, default="")
    pr_number: int = field(compare=False, default=0)
    timeout_seconds: int = field(compare=False, default=300)
    status: JobStatus = field(compare=False, default=JobStatus.QUEUED)
    started_at: float | None = field(compare=False, default=None)
    completed_at: float | None = field(compare=False, default=None)
    result: dict[str, Any] | None = field(compare=False, default=None)

    @property
    def dedup_key(self) -> str:
        """Key for deduplication — same repo+commit = same job."""
        raw = f"{self.repo_id}|{self.commit_sha}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class RateLimitConfig:
    """Rate limiting configuration per organization."""

    max_concurrent: int = 5
    max_per_minute: int = 30
    burst_size: int = 10


@dataclass
class QueueStats:
    """Statistics for the verification queue."""

    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    deduplicated: int = 0


class VerificationQueue:
    """Priority queue with fair scheduling and rate limiting.

    Usage:
        queue = VerificationQueue()
        job_id = queue.enqueue(VerificationJob(
            priority=JobPriority.NORMAL,
            org_id="org-1",
            repo_id="repo-1",
            commit_sha="abc123",
        ))
        job = queue.dequeue()
        queue.complete(job.job_id, result={"findings": []})
    """

    def __init__(self, default_rate_limit: RateLimitConfig | None = None):
        self._heap: list[VerificationJob] = []
        self._jobs: dict[str, VerificationJob] = {}
        self._dedup_index: dict[str, str] = {}  # dedup_key → job_id
        self._org_running: dict[str, int] = defaultdict(int)
        self._org_timestamps: dict[str, list[float]] = defaultdict(list)
        self._rate_limits: dict[str, RateLimitConfig] = {}
        self._default_limit = default_rate_limit or RateLimitConfig()
        self._stats = QueueStats()
        self._counter = 0  # Tie-breaker for heap ordering

    def set_rate_limit(self, org_id: str, config: RateLimitConfig) -> None:
        self._rate_limits[org_id] = config

    def enqueue(self, job: VerificationJob) -> str:
        """Add a job to the queue. Returns job_id."""
        if not job.job_id:
            self._counter += 1
            job.job_id = f"vj-{self._counter}-{int(time.time() * 1000)}"

        # Deduplication
        dk = job.dedup_key
        if dk in self._dedup_index:
            existing_id = self._dedup_index[dk]
            existing = self._jobs.get(existing_id)
            if existing and existing.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                self._stats.deduplicated += 1
                return existing_id  # Return existing job

        self._jobs[job.job_id] = job
        self._dedup_index[dk] = job.job_id
        heapq.heappush(self._heap, job)
        self._stats.queued += 1

        return job.job_id

    def dequeue(self) -> VerificationJob | None:
        """Get the next job to process, respecting rate limits."""
        skipped: list[VerificationJob] = []

        while self._heap:
            job = heapq.heappop(self._heap)

            # Skip completed/cancelled jobs still in heap
            if job.status != JobStatus.QUEUED:
                continue

            # Check rate limits
            if not self._check_rate_limit(job.org_id):
                skipped.append(job)
                continue

            # Mark as running
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._org_running[job.org_id] += 1
            self._record_org_activity(job.org_id)
            self._stats.queued -= 1
            self._stats.running += 1

            # Re-add skipped jobs
            for s in skipped:
                heapq.heappush(self._heap, s)

            return job

        # Re-add skipped jobs
        for s in skipped:
            heapq.heappush(self._heap, s)

        return None

    def complete(
        self,
        job_id: str,
        result: dict[str, Any] | None = None,
        failed: bool = False,
    ) -> None:
        """Mark a job as completed or failed."""
        job = self._jobs.get(job_id)
        if not job:
            return

        job.completed_at = time.time()
        job.result = result
        job.status = JobStatus.FAILED if failed else JobStatus.COMPLETED

        self._org_running[job.org_id] = max(0, self._org_running[job.org_id] - 1)
        self._stats.running -= 1

        if failed:
            self._stats.failed += 1
        else:
            self._stats.completed += 1

    def check_timeouts(self) -> list[str]:
        """Check for timed-out jobs and mark them. Returns list of timed-out job IDs."""
        now = time.time()
        timed_out: list[str] = []

        for job_id, job in self._jobs.items():
            if job.status == JobStatus.RUNNING and job.started_at:
                elapsed = now - job.started_at
                if elapsed > job.timeout_seconds:
                    job.status = JobStatus.TIMED_OUT
                    job.completed_at = now
                    self._org_running[job.org_id] = max(0, self._org_running[job.org_id] - 1)
                    self._stats.running -= 1
                    self._stats.timed_out += 1
                    timed_out.append(job_id)

        return timed_out

    def get_job(self, job_id: str) -> VerificationJob | None:
        return self._jobs.get(job_id)

    @property
    def stats(self) -> QueueStats:
        return self._stats

    @property
    def pending_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)

    def _check_rate_limit(self, org_id: str) -> bool:
        limit = self._rate_limits.get(org_id, self._default_limit)

        # Check concurrent limit
        if self._org_running[org_id] >= limit.max_concurrent:
            return False

        # Check per-minute limit
        now = time.time()
        timestamps = self._org_timestamps.get(org_id, [])
        recent = [t for t in timestamps if now - t < 60]
        self._org_timestamps[org_id] = recent

        return len(recent) < limit.max_per_minute

    def _record_org_activity(self, org_id: str) -> None:
        self._org_timestamps[org_id].append(time.time())
