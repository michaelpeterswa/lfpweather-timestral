"""NWS hourly forecast client (api.weather.gov)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import numpy as np

BASE_URL = "https://api.weather.gov/gridpoints"
TIMEOUT_S = 30.0
HORIZON_HOURS = 24


class NwsError(Exception):
    """Any failure fetching or interpreting the NWS hourly forecast."""


def _to_fahrenheit(period: dict) -> float | None:
    temp = period["temperature"]
    if isinstance(temp, dict):
        value, unit = temp["value"], temp["unitCode"]
        celsius = unit.endswith("degC")
        if not celsius and not unit.endswith("degF"):
            raise ValueError(f"unknown unitCode {unit!r}")
    else:
        value, unit = temp, period.get("temperatureUnit", "F")
        celsius = unit == "C"
        if not celsius and unit != "F":
            raise ValueError(f"unknown temperatureUnit {unit!r}")
    if value is None:
        return None
    value = float(value)
    return value * 9.0 / 5.0 + 32.0 if celsius else value


def _parse(body: dict) -> dict[datetime, float]:
    out: dict[datetime, float] = {}
    for period in body["properties"]["periods"]:
        start = datetime.fromisoformat(period["startTime"])
        if start.tzinfo is None:
            raise ValueError("naive startTime")
        temp = _to_fahrenheit(period)
        if temp is None:
            # NWS nulls the far tail of its horizon; nws_target checks the hours we need
            continue
        out[start.astimezone(UTC)] = temp
    if not out:
        raise ValueError("no periods")
    return out


async def fetch_hourly(
    client: httpx.AsyncClient, grid_point: str, user_agent: str
) -> dict[datetime, float]:
    """Return UTC period start -> temperature in degrees F."""
    url = f"{BASE_URL}/{grid_point}/forecast/hourly"
    headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}
    last: Exception | None = None
    for _ in range(2):  # one retry on 5xx
        try:
            resp = await client.get(url, headers=headers, timeout=TIMEOUT_S)
        except httpx.HTTPError as exc:
            raise NwsError(f"nws request failed: {exc!r}") from exc
        if resp.status_code >= 500:
            last = NwsError(f"nws returned {resp.status_code}")
            continue
        if resp.status_code != 200:
            raise NwsError(f"nws returned {resp.status_code}")
        try:
            return _parse(resp.json())
        except (ValueError, KeyError, TypeError) as exc:
            raise NwsError(f"nws response malformed: {exc!r}") from exc
    assert last is not None
    raise last


def nws_target(hourly: dict[datetime, float], issued_at: datetime) -> np.ndarray:
    """Degrees F for issued_at+1h .. +24h; NwsError if any hour is missing."""
    base = issued_at.astimezone(UTC)
    vals: list[float] = []
    for h in range(1, HORIZON_HOURS + 1):
        t = base + timedelta(hours=h)
        if t not in hourly:
            raise NwsError(f"nws forecast missing {t.isoformat()}")
        vals.append(hourly[t])
    return np.asarray(vals, dtype=float)
