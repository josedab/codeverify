"""CodeVerify Worker - Main Celery Application"""

import os

from celery import Celery

# Configure Celery
app = Celery(
    "codeverify_worker",
    broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    include=["codeverify_worker.tasks.analysis"],
)

# Celery configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # Soft limit at 9 minutes
    worker_prefetch_multiplier=1,  # One task at a time for memory management
    task_acks_late=True,  # Acknowledge after completion
    task_reject_on_worker_lost=True,
)

if __name__ == "__main__":
    app.start()
