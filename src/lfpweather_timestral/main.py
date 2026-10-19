"""Process entrypoint: load the model, start the hourly scheduler, serve both ports."""

import asyncio
import logging
import sys
import threading
from datetime import UTC, datetime

import httpx
import psycopg
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from lfpweather_timestral.api import LockedModel, create_api, create_metrics_app
from lfpweather_timestral.config import Settings
from lfpweather_timestral.model import CudaUnavailable, Model
from lfpweather_timestral.run import run_once

log = logging.getLogger("lfpweather_timestral")


def connect(settings: Settings, *, statement_timeout_ms: int = 60_000) -> psycopg.Connection:
    """Fresh autocommit connection with bounded connect and statement time."""
    return psycopg.connect(
        settings.database_url,
        autocommit=True,
        connect_timeout=10,
        options=f"-c statement_timeout={statement_timeout_ms}",
    )


async def _scheduled(settings: Settings, model) -> None:
    # Fresh autocommit connection per run so a caught DB error cannot poison later work.
    with connect(settings) as conn:
        async with httpx.AsyncClient(timeout=30) as http:
            result = await run_once(
                datetime.now(UTC), conn=conn, http=http, model=model, settings=settings
            )
    log.info("run complete written=%s failed=%s", result.written, result.failed)


def scheduled_job(settings: Settings, model) -> None:
    """Runs in an APScheduler worker thread; never raises."""
    try:
        asyncio.run(_scheduled(settings, model))
    except Exception:
        log.exception("scheduled run failed")


def build_scheduler(settings: Settings, model) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        scheduled_job,
        "cron",
        minute=35,
        timezone="UTC",
        args=[settings, model],
        id="hourly-forecast",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )
    return scheduler


def _serve_metrics(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    )
    server.install_signal_handlers = lambda: None
    threading.Thread(target=server.run, name="metrics", daemon=True).start()
    return server


def main() -> int:
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    ready = threading.Event()
    metrics_server = _serve_metrics(create_metrics_app(ready.is_set), settings.metrics_port)
    try:
        raw = Model.load(settings.model_id, settings.require_cuda)
    except CudaUnavailable:
        log.exception("CUDA unavailable")
        metrics_server.should_exit = True
        return 1
    model = LockedModel(raw)
    ready.set()
    log.info("model loaded: %s", settings.model_id)

    scheduler = None
    if settings.schedule_enabled:
        scheduler = build_scheduler(settings, model)
        scheduler.start()
        log.info("scheduler started (minute=35 UTC)")
    try:
        uvicorn.run(
            create_api(model, settings.api_key),
            host="0.0.0.0",
            port=settings.api_port,
            log_level="warning",
        )
    finally:
        if scheduler:
            scheduler.shutdown(wait=False)
        metrics_server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
