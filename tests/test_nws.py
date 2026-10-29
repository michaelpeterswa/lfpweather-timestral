import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from lfpweather_timestral.nws import NwsError, fetch_hourly, nws_target

FIX = Path(__file__).parent / "fixtures"
URL = "https://api.weather.gov/gridpoints/SEW/127,75/forecast/hourly"
UA = "lfpweather-timestral (test)"


def _payload(name):
    return json.loads((FIX / name).read_text())


async def _fetch():
    async with httpx.AsyncClient() as client:
        return await fetch_hourly(client, "SEW/127,75", UA)


@respx.mock
async def test_fetch_parses_offsets_to_utc():
    payload = _payload("nws_hourly_f.json")
    payload["properties"]["periods"][0]["startTime"] = "2026-10-19T11:00:00-07:00"
    route = respx.get(URL).respond(200, json=payload)
    out = await _fetch()
    assert datetime(2026, 10, 19, 18, tzinfo=UTC) in out
    assert all(k.tzinfo is not None and k.utcoffset() == timedelta(0) for k in out)
    assert route.calls.last.request.headers["user-agent"] == UA


@respx.mock
async def test_fahrenheit_passthrough():
    payload = _payload("nws_hourly_f.json")
    expected = payload["properties"]["periods"][1]["temperature"]
    respx.get(URL).respond(200, json=payload)
    out = await _fetch()
    start = datetime.fromisoformat(payload["properties"]["periods"][1]["startTime"])
    assert out[start.astimezone(UTC)] == float(expected)


@respx.mock
async def test_celsius_converted():
    payload = _payload("nws_hourly_c.json")
    respx.get(URL).respond(200, json=payload)
    out = await _fetch()
    first = datetime.fromisoformat(payload["properties"]["periods"][0]["startTime"])
    assert out[first.astimezone(UTC)] == pytest.approx(50.0)


@respx.mock
async def test_quantitative_value_celsius():
    payload = _payload("nws_hourly_f.json")
    payload["properties"]["periods"][0]["temperature"] = {
        "unitCode": "wmoUnit:degC",
        "value": 10,
    }
    respx.get(URL).respond(200, json=payload)
    out = await _fetch()
    first = datetime.fromisoformat(payload["properties"]["periods"][0]["startTime"])
    assert out[first.astimezone(UTC)] == pytest.approx(50.0)


@respx.mock
async def test_null_temperature_periods_skipped():
    # NWS leaves the far end of its 156h horizon null; the 24h we use is still valid
    payload = _payload("nws_hourly_f.json")
    periods = payload["properties"]["periods"]
    periods[-1]["temperature"] = None
    respx.get(URL).respond(200, json=payload)
    out = await _fetch()
    last = datetime.fromisoformat(periods[-1]["startTime"]).astimezone(UTC)
    first = datetime.fromisoformat(periods[0]["startTime"]).astimezone(UTC)
    assert last not in out
    assert first in out


@respx.mock
async def test_all_null_temperatures_raise_nwserror():
    payload = _payload("nws_hourly_f.json")
    for period in payload["properties"]["periods"]:
        period["temperature"] = None
    respx.get(URL).respond(200, json=payload)
    with pytest.raises(NwsError):
        await _fetch()


def _hourly(start, n, skip=()):
    return {start + timedelta(hours=i): 50.0 + i for i in range(n) if i not in skip}


def test_target_requires_all_24():
    issued = datetime(2026, 10, 19, 18, tzinfo=UTC)
    hourly = _hourly(issued, 30, skip={7})
    with pytest.raises(NwsError):
        nws_target(hourly, issued)


def test_target_ignores_past_periods():
    issued = datetime(2026, 10, 19, 18, tzinfo=UTC)
    hourly = _hourly(issued - timedelta(hours=5), 40)
    out = nws_target(hourly, issued)
    assert out.shape == (24,)
    assert out[0] == hourly[issued + timedelta(hours=1)]
    assert out[-1] == hourly[issued + timedelta(hours=24)]


@respx.mock
async def test_http_error_raises_nwserror():
    respx.get(URL).respond(503)
    with pytest.raises(NwsError):
        await _fetch()


@respx.mock
async def test_retries_once_on_5xx():
    respx.get(URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=_payload("nws_hourly_f.json")),
        ]
    )
    assert await _fetch()


@respx.mock
async def test_transport_error_raises_nwserror():
    respx.get(URL).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(NwsError):
        await _fetch()


@respx.mock
async def test_malformed_body_raises_nwserror():
    respx.get(URL).respond(200, json={"properties": {}})
    with pytest.raises(NwsError):
        await _fetch()
