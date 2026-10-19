from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pytest
import respx
import torch

from lfpweather_timestral import metrics
from lfpweather_timestral.config import Settings
from lfpweather_timestral.model import QuantileForecast
from lfpweather_timestral.run import run_once
from lfpweather_timestral.store import SourceForecast, write_run

URL = "https://api.weather.gov/gridpoints/SEW/127,75/forecast/hourly"
NOW = datetime(2026, 11, 1, 10, 35, 12, tzinfo=UTC)
ISSUED = datetime(2026, 11, 1, 10, tzinfo=UTC)

pytestmark = pytest.mark.db


class FakeModel:
    def __init__(self, exc: Exception | None = None):
        self.exc = exc
        self.calls: list[dict] = []

    def forecast(self, history, horizon, covariate_past=None, covariate_future=None):
        self.calls.append(
            {"past": covariate_past, "future": covariate_future, "history": history}
        )
        if self.exc is not None:
            raise self.exc
        med = np.full(horizon, 55.0)
        q = np.tile(np.linspace(50, 60, 21)[:, None], (1, horizon))
        return QuantileForecast(med, q, tuple(np.linspace(0.02, 0.98, 21)))


@pytest.fixture
def settings():
    return Settings(
        database_url="postgresql://x", nws_user_agent="test", history_days=3
    )


def _readings(conn, hours=80, end=ISSUED):
    conn.execute(
        """
        INSERT INTO sensors.vantagepro2plus (time, device_id, temperature)
        SELECT %s - make_interval(hours => g), 'test-station', 50.0
        FROM generate_series(0, %s) g
        """,
        (end, hours),
    )


def _nws_body(issued=ISSUED):
    periods = [
        {
            "startTime": (issued + timedelta(hours=h)).isoformat(),
            "temperature": 60 + h,
            "temperatureUnit": "F",
        }
        for h in range(0, 30)
    ]
    return {"properties": {"periods": periods}}


def _counts(conn):
    return dict(
        conn.execute(
            "SELECT source, count(*) FROM sensors.temperature_forecast GROUP BY source"
        ).fetchall()
    )


def _failures(stage):
    return metrics.RUN_FAILURES.labels(stage=stage)._value.get()


@respx.mock
async def test_all_sources_written(db_conn, settings):
    _readings(db_conn)
    respx.get(URL).respond(200, json=_nws_body())
    async with httpx.AsyncClient() as http:
        res = await run_once(NOW, conn=db_conn, http=http, model=FakeModel(), settings=settings)
    assert _counts(db_conn) == {"nws": 24, "timestral": 24, "timestral_nws": 24}
    assert res.failed == {}
    assert set(res.written) == {"nws", "timestral", "timestral_nws"}
    models = dict(
        db_conn.execute(
            "SELECT DISTINCT source, model FROM sensors.temperature_forecast"
        ).fetchall()
    )
    assert models == {"nws": None, "timestral": settings.model_id, "timestral_nws": settings.model_id}


@respx.mock
async def test_nws_down_writes_only_timestral(db_conn, settings):
    _readings(db_conn)
    respx.get(URL).respond(503)
    async with httpx.AsyncClient() as http:
        res = await run_once(NOW, conn=db_conn, http=http, model=FakeModel(), settings=settings)
    assert _counts(db_conn) == {"timestral": 24}
    assert set(res.failed) == {"nws", "timestral_nws"}


@respx.mock
async def test_model_oom_still_writes_nws(db_conn, settings):
    _readings(db_conn)
    respx.get(URL).respond(200, json=_nws_body())
    before = _failures("model")
    model = FakeModel(torch.cuda.OutOfMemoryError("oom"))
    async with httpx.AsyncClient() as http:
        res = await run_once(NOW, conn=db_conn, http=http, model=model, settings=settings)
    assert _counts(db_conn) == {"nws": 24}
    assert _failures("model") > before
    assert set(res.failed) == {"timestral", "timestral_nws"}


@respx.mock
async def test_station_outage_skips_timestral(db_conn, settings):
    respx.get(URL).respond(200, json=_nws_body())
    before = _failures("insufficient_history")
    model = FakeModel()
    async with httpx.AsyncClient() as http:
        res = await run_once(NOW, conn=db_conn, http=http, model=model, settings=settings)
    assert _counts(db_conn) == {"nws": 24}
    assert _failures("insufficient_history") > before
    assert model.calls == []
    assert set(res.failed) == {"timestral", "timestral_nws"}


@respx.mock
async def test_issued_at_floors_to_hour(db_conn, settings):
    _readings(db_conn)
    respx.get(URL).respond(200, json=_nws_body())
    async with httpx.AsyncClient() as http:
        res = await run_once(NOW, conn=db_conn, http=http, model=FakeModel(), settings=settings)
    assert res.issued_at == ISSUED


@respx.mock
async def test_covariate_coverage_recorded(db_conn, settings):
    _readings(db_conn)
    s1 = Settings(database_url="postgresql://x", nws_user_agent="t", history_days=1)
    # archived NWS issued 36h earlier covers I-35..I-12 -> 12 of the 24 grid hours
    write_run(
        db_conn,
        ISSUED - timedelta(hours=36),
        [SourceForecast("nws", np.full(24, 40.0))],
    )
    respx.get(URL).respond(200, json=_nws_body())
    async with httpx.AsyncClient() as http:
        await run_once(NOW, conn=db_conn, http=http, model=FakeModel(), settings=s1)
    cov = db_conn.execute(
        "SELECT DISTINCT covariate_coverage FROM sensors.temperature_forecast"
        " WHERE source = 'timestral_nws'"
    ).fetchall()
    assert len(cov) == 1 and cov[0][0] == pytest.approx(0.5)
