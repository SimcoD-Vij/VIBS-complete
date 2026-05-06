"""
celery_app.py
=============
Celery application configured to use Redis as broker and result backend.
Workers are started with --pool=threads to allow PyTorch/faster-whisper
to share GPU context across tasks without re-loading models.
"""
from celery import Celery
from app.config import settings

celery_app = Celery(
    "vibs",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,           # only ack after task completes (no lost jobs)
    worker_prefetch_multiplier=1,  # don't prefetch — each task is heavy
    task_soft_time_limit=3600,     # 1-hour soft limit
    task_time_limit=4000,          # hard kill at 66 min
    broker_connection_retry_on_startup=True,
)
