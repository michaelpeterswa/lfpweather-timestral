"""Hourly run orchestration with partial-run semantics."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
import numpy as np

from lfpweather_timestral import metrics
from lfpweather_timestral.config import Settings
from lfpweather_timestral.history import hour_grid, load_history
from lfpweather_timestral.model import Model
from lfpweather_timestral.nws import NwsError, fetch_hourly, nws_target
from lfpweather_timestral.store import (
    HORIZON,
    SourceForecast,
    load_nws_covariate_past,
    write_run,
)

log = logging.getLogger(__name__)

MIN_RECENT_POINTS = 24
RECENT_WINDOW = 72


@dataclass
class RunResult:
    issued_at: datetime
    written: dict[str, int] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)


def floor_hour(now: datetime) -> datetime:
    return now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def _fail(result: RunResult, stage: str, sources: list[str], reason: str) -> None:
    metrics.RUN_FAILURES.labels(stage=stage).inc()
    for s in sources:
        result.failed[s] = reason
    log.error("run stage failed", extra={"stage": stage, "sources": sources, "reason": reason})


def _require_finite(source: str, q) -> None:
    if not (np.isfinite(q.median).all() and np.isfinite(q.quantiles).all()):
        raise ValueError(f"{source}: non-finite model output")


async def run_once(
    now: datetime,
    *,
    conn,
    http: httpx.AsyncClient,
    model: Model,
    settings: Settings,
) -> RunResult:
    issued_at = floor_hour(now)
    result = RunResult(issued_at=issued_at)
    forecasts: list[SourceForecast] = []

    # NWS: written whenever it succeeds, independent of the model.
    nws_values: np.ndarray | None = None
    try:
        hourly = await fetch_hourly(http, settings.grid_point, settings.nws_user_agent)
        nws_values = nws_target(hourly, issued_at)
        forecasts.append(SourceForecast("nws", nws_values))
    except NwsError as exc:
        _fail(result, "nws", ["nws", "timestral_nws"], f"nws: {exc}")

    history: np.ndarray | None = None
    try:
        history = load_history(conn, issued_at, settings.history_days)
    except Exception as exc:  # noqa: BLE001 - DB error, partial run
        _fail(result, "history", ["timestral", "timestral_nws"], f"history: {exc!r}")
    if (
        history is not None
        and np.count_nonzero(~np.isnan(history[-RECENT_WINDOW:])) < MIN_RECENT_POINTS
    ):
        _fail(
            result,
            "insufficient_history",
            ["timestral", "timestral_nws"],
            f"fewer than {MIN_RECENT_POINTS} readings in last {RECENT_WINDOW}h",
        )
        history = None

    if history is not None:
        try:
            q = model.forecast(history, HORIZON)
            _require_finite("timestral", q)
            forecasts.append(
                SourceForecast("timestral", q.median, q.quantiles, None, settings.model_id)
            )
        except Exception as exc:  # noqa: BLE001 - model failure must not abort run
            _fail(result, "model", ["timestral"], f"model: {exc!r}")

        if nws_values is not None:
            try:
                grid = hour_grid(issued_at, settings.history_days * 24)
                past = load_nws_covariate_past(conn, grid)
                coverage = float(np.mean(~np.isnan(past)))
                q = model.forecast(
                    history, HORIZON, covariate_past=past, covariate_future=nws_values
                )
                _require_finite("timestral_nws", q)
                forecasts.append(
                    SourceForecast(
                        "timestral_nws", q.median, q.quantiles, coverage, settings.model_id
                    )
                )
            except Exception as exc:  # noqa: BLE001 - model failure must not abort run
                _fail(result, "model", ["timestral_nws"], f"model: {exc!r}")

    if forecasts:
        try:
            write_run(conn, issued_at, forecasts)
        except Exception:
            metrics.RUN_FAILURES.labels(stage="write").inc()
            log.exception("write failed")
            raise
        ts = time.time()
        for f in forecasts:
            result.written[f.source] = HORIZON
            metrics.RUN_LAST_SUCCESS.labels(source=f.source).set(ts)
    return result
