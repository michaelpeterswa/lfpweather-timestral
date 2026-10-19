import numpy as np
from fastapi.testclient import TestClient

from lfpweather_timestral.api import create_api, create_metrics_app
from lfpweather_timestral.model import QuantileForecast


class FakeModel:
    def forecast(self, history, horizon, covariate_past=None, covariate_future=None):
        return QuantileForecast(
            median=np.zeros(horizon),
            quantiles=np.zeros((21, horizon)),
            levels=tuple(np.linspace(0.02, 0.98, 21)),
        )


def test_forecast_endpoint_shapes():
    client = TestClient(create_api(FakeModel()))
    r = client.post("/forecast", json={"history": [1.0] * 48, "horizon": 24})
    assert r.status_code == 200
    body = r.json()
    assert len(body["levels"]) == 21
    assert len(body["median"]) == 24
    assert len(body["quantiles"]) == 21
    assert all(len(q) == 24 for q in body["quantiles"])


def test_forecast_rejects_short_history():
    client = TestClient(create_api(FakeModel()))
    r = client.post("/forecast", json={"history": [1.0] * 10, "horizon": 24})
    assert r.status_code == 422


def test_forecast_accepts_null_history_and_covariates():
    client = TestClient(create_api(FakeModel()))
    r = client.post(
        "/forecast",
        json={
            "history": [None] + [1.0] * 47,
            "horizon": 4,
            "covariate_past": [None] * 48,
            "covariate_future": [1.0] * 4,
        },
    )
    assert r.status_code == 200


def test_healthcheck_ok_after_load():
    client = TestClient(create_metrics_app(lambda: True))
    assert client.get("/healthcheck").status_code == 200
    assert "timestral_run_failures_total" in client.get("/metrics").text


def test_healthcheck_503_before_load():
    client = TestClient(create_metrics_app(lambda: False))
    assert client.get("/healthcheck").status_code == 503
