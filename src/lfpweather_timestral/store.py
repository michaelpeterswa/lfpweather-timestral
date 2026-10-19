from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

HORIZON = 24
N_QUANTILES = 21

_INSERT = """
INSERT INTO sensors.temperature_forecast
    (issued_at, source, valid_at, lead_hours, temperature, quantiles,
     covariate_coverage, model)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT DO NOTHING
"""


@dataclass
class SourceForecast:
    source: str
    values: np.ndarray  # shape (24,)
    quantiles: np.ndarray | None = None  # shape (21, 24), ascending levels
    covariate_coverage: float | None = None
    model: str | None = None


def _validate(f: SourceForecast) -> None:
    if f.values.shape != (HORIZON,):
        raise ValueError(f"{f.source}: expected {HORIZON} values, got {f.values.shape}")
    if f.quantiles is not None and f.quantiles.shape != (N_QUANTILES, HORIZON):
        raise ValueError(f"{f.source}: bad quantiles shape {f.quantiles.shape}")


def write_run(conn, issued_at: datetime, forecasts: list[SourceForecast]) -> int:
    """Insert all forecasts in one transaction; returns rows actually inserted."""
    for f in forecasts:
        _validate(f)
    issued_at = issued_at.astimezone(UTC)
    inserted = 0
    with conn.transaction(), conn.cursor() as cur:
        for f in forecasts:
            for k in range(HORIZON):
                lead = k + 1
                q = (
                    [float(x) for x in f.quantiles[:, k]]
                    if f.quantiles is not None
                    else None
                )
                cur.execute(
                    _INSERT,
                    (
                        issued_at,
                        f.source,
                        issued_at + timedelta(hours=lead),
                        lead,
                        float(f.values[k]),
                        q,
                        f.covariate_coverage,
                        f.model,
                    ),
                )
                inserted += cur.rowcount
    return inserted


def load_nws_covariate_past(conn, grid: list[datetime]) -> np.ndarray:
    """Latest archived NWS forecast (issued before the hour) per grid hour; NaN if none."""
    out = np.full(len(grid), np.nan, dtype=np.float64)
    index = {t.astimezone(UTC): i for i, t in enumerate(grid)}
    rows = conn.execute(
        """
        SELECT DISTINCT ON (valid_at) valid_at, temperature
        FROM sensors.temperature_forecast
        WHERE source = 'nws' AND valid_at = ANY(%s) AND issued_at < valid_at
        ORDER BY valid_at, issued_at DESC
        """,
        (list(index),),
    ).fetchall()
    for valid_at, temp in rows:
        i = index.get(valid_at.astimezone(UTC))
        if i is not None:
            out[i] = temp
    return out
