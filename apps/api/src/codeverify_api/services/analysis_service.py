"""Analysis service for queuing and managing code analysis jobs."""

import json
import uuid
from typing import Any

import redis.asyncio as redis
import structlog

from codeverify_api.config import settings

logger = structlog.get_logger()


class AnalysisService:
    """Service for managing code analysis jobs."""

    def __init__(self) -> None:
        self.redis_url = settings.REDIS_URL

    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection."""
        return redis.from_url(self.redis_url, decode_responses=True)

    async def queue_analysis(
        self,
        repo_full_name: str,
        repo_id: int,
        pr_number: int,
        pr_title: str | None,
        head_sha: str,
        base_sha: str | None,
        installation_id: int | None,
    ) -> str:
        """Queue a new analysis job."""
        job_id = str(uuid.uuid4())

        job_data = {
            "job_id": job_id,
            "repo_full_name": repo_full_name,
            "repo_id": repo_id,
            "pr_number": pr_number,
            "pr_title": pr_title,
            "head_sha": head_sha,
            "base_sha": base_sha,
            "installation_id": installation_id,
            "status": "queued",
        }

        logger.info("Queuing analysis job", job_id=job_id, repo=repo_full_name, pr=pr_number)

        # Store job data and push to queue
        r = await self._get_redis()
        try:
            # Store job details
            await r.set(f"job:{job_id}", json.dumps(job_data), ex=86400)  # 24h expiry

            # Push to analysis queue
            await r.lpush("codeverify:analysis:queue", job_id)

            logger.info("Analysis job queued successfully", job_id=job_id)
        finally:
            await r.aclose()

        return job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the status of an analysis job."""
        r = await self._get_redis()
        try:
            job_data = await r.get(f"job:{job_id}")
            if job_data:
                return json.loads(job_data)
            return None
        finally:
            await r.aclose()

    async def update_job_status(self, job_id: str, status: str, **kwargs: Any) -> None:
        """Update job status."""
        r = await self._get_redis()
        try:
            job_data = await r.get(f"job:{job_id}")
            if job_data:
                data = json.loads(job_data)
                data["status"] = status
                data.update(kwargs)
                await r.set(f"job:{job_id}", json.dumps(data), ex=86400)
        finally:
            await r.aclose()
