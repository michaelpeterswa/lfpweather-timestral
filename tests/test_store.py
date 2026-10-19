from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lfpweather_timestral.history import hour_grid
from lfpweather_timestral.store import SourceForecast, load_nws_covariate_past, write_run

pytestmark = pytest.mark.db

ISSUED = datetime(2026, 10, 19, 12, tzinfo=UTC)


def _forecasts():
    q = np.tile(np.linspace(40, 60, 21)[:, None], (1, 24))
    return [
        SourceForecast("nws", np.arange(24, dtype=float) + 50),
        SourceForecast("timestral", np.full(24, 51.0), quantiles=q, model="m1"),
        SourceForecast(
            "timestral_nws", np.full(24, 52.0), quantiles=q,
            covariate_coverage=0.9, model="m1",
        ),
    ]


def _count(conn, issued=ISSUED):
    return conn.execute(
        "SELECT count(*) FROM sensors.temperature_forecast WHERE issued_at=%s",
        (issued,),
    ).fetchone()[0]


def test_write_run_inserts_24_rows_per_source(db_conn):
    assert write_run(db_conn, ISSUED, _forecasts()) == 72
    assert _count(db_conn) == 72
    leads = db_conn.execute(
        "SELECT array_agg(lead_hours ORDER BY lead_hours) FROM sensors.temperature_forecast "
        "WHERE source='nws'"
    ).fetchone()[0]
    assert leads == list(range(1, 25))
    q = db_conn.execute(
        "SELECT quantiles FROM sensors.temperature_forecast WHERE source='timestral' LIMIT 1"
    ).fetchone()[0]
    assert len(q) == 21
    nq = db_conn.execute(
        "SELECT quantiles FROM sensors.temperature_forecast WHERE source='nws' LIMIT 1"
    ).fetchone()[0]
    assert nq is None


def test_write_run_is_idempotent(db_conn):
    assert write_run(db_conn, ISSUED, _forecasts()) == 72
    assert write_run(db_conn, ISSUED, _forecasts()) == 0
    assert _count(db_conn) == 72


def test_write_run_is_atomic(db_conn):
    bad = _forecasts()
    bad[2] = SourceForecast("timestral_nws", np.zeros(23))
    with pytest.raises(ValueError):
        write_run(db_conn, ISSUED, bad)
    assert _count(db_conn) == 0


def test_covariate_past_uses_latest_issue(db_conn):
    t = ISSUED
    for issued, val in [(t - timedelta(hours=5), 50.0), (t - timedelta(hours=1), 52.0)]:
        db_conn.execute(
            "INSERT INTO sensors.temperature_forecast "
            "(issued_at, source, valid_at, lead_hours, temperature) "
            "VALUES (%s,'nws',%s,%s,%s)",
            (issued, t, int((t - issued).total_seconds() // 3600), val),
        )
    grid = hour_grid(t, 3)
    out = load_nws_covariate_past(db_conn, grid)
    assert out[2] == 52.0
    assert np.isnan(out[0]) and np.isnan(out[1])
