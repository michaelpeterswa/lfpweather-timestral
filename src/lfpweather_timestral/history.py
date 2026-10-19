from datetime import UTC, datetime, timedelta

import numpy as np


def hour_grid(end: datetime, hours: int) -> list[datetime]:
    """`hours` consecutive UTC hours ending at `end` inclusive, oldest first."""
    end = end.astimezone(UTC)
    return [end - timedelta(hours=hours - 1 - i) for i in range(hours)]


def load_history(conn, issued_at: datetime, days: int) -> np.ndarray:
    """Hourly station temperatures aligned to the grid; NaN where no readings."""
    grid = hour_grid(issued_at, days * 24)
    out = np.full(len(grid), np.nan, dtype=np.float64)
    index = {t: i for i, t in enumerate(grid)}
    rows = conn.execute(
        "SELECT valid_at, temperature FROM sensors.temperature_hourly(%s, %s)",
        (grid[0], grid[-1]),
    ).fetchall()
    for valid_at, temp in rows:
        i = index.get(valid_at.astimezone(UTC))
        if i is not None and temp is not None:
            out[i] = temp
    return out
