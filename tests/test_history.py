import itertools
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lfpweather_timestral.history import hour_grid, load_history


def test_hour_grid_is_utc_and_consecutive_across_dst():
    end = datetime(2026, 11, 1, 12, tzinfo=UTC)
    grid = hour_grid(end, 48)
    assert len(grid) == 48
    assert grid[-1] == end
    assert all(g.utcoffset() == timedelta(0) for g in grid)
    assert all(b - a == timedelta(hours=1) for a, b in itertools.pairwise(grid))


@pytest.mark.slow
def test_load_history_aligns_and_fills_nan(db_conn):
    issued = datetime(2026, 11, 1, 12, tzinfo=UTC)
    # readings around hours -4, -3, -1 (hours -2 and 0 missing); centered window
    for hours_back, temps in {4: (50.0, 52.0), 3: (53.0,), 1: (60.0, 62.0)}.items():
        top = issued - timedelta(hours=hours_back)
        for i, t in enumerate(temps):
            db_conn.execute(
                "INSERT INTO sensors.vantagepro2plus (time, device_id, temperature)"
                " VALUES (%s, 'test-station', %s)",
                (top + timedelta(minutes=10 * i - 10), t),
            )
    arr = load_history(db_conn, issued, 1)
    assert arr.dtype == np.float64
    assert arr.shape == (24,)
    assert np.isnan(arr[:19]).all()
    tail = arr[19:]
    assert tail[0] == 51.0 and tail[1] == 53.0 and tail[3] == 61.0
    assert np.isnan(tail[2]) and np.isnan(tail[4])
