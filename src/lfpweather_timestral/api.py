"""HTTP surfaces: the forecast API (8080) and the metrics/health app (8081)."""

import hmac
import threading
from collections.abc import Callable

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, model_validator


class LockedModel:
    """Serialises GPU use: API and scheduled runs share one model and one lock."""

    def __init__(self, model):
        self._model = model
        self._lock = threading.Lock()

    def forecast(self, *args, **kwargs):
        with self._lock:
            return self._model.forecast(*args, **kwargs)


class ForecastRequest(BaseModel):
    history: list[float | None] = Field(min_length=24, max_length=32768)
    horizon: int = Field(ge=1, le=168)
    covariate_past: list[float | None] | None = Field(default=None, max_length=32768)
    covariate_future: list[float] | None = Field(default=None, max_length=168)

    @model_validator(mode="after")
    def _check_covariates(self):
        if self.covariate_past is not None:
            if self.covariate_future is None:
                raise ValueError("covariate_past requires covariate_future")
            if len(self.covariate_past) != len(self.history):
                raise ValueError("covariate_past must have the same length as history")
        if self.covariate_future is not None and len(self.covariate_future) != self.horizon:
            raise ValueError("covariate_future must have length equal to horizon")
        return self


class ForecastResponse(BaseModel):
    levels: list[float]
    median: list[float]
    quantiles: list[list[float]]


def _arr(values: list[float | None] | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.array([np.nan if v is None else v for v in values], dtype=np.float32)


def create_api(model, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="lfpweather-timestral")

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key is None:
            return
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="invalid api key")

    # Sync endpoint: FastAPI runs it in a threadpool, off the event loop.
    @app.post("/forecast", response_model=ForecastResponse, dependencies=[Depends(require_key)])
    def forecast(req: ForecastRequest) -> ForecastResponse:
        fc = model.forecast(
            _arr(req.history),
            req.horizon,
            _arr(req.covariate_past),
            _arr(req.covariate_future),
        )
        return ForecastResponse(
            levels=[float(x) for x in fc.levels],
            median=np.asarray(fc.median, dtype=float).tolist(),
            quantiles=np.asarray(fc.quantiles, dtype=float).tolist(),
        )

    return app


def create_metrics_app(is_ready: Callable[[], bool]) -> FastAPI:
    app = FastAPI(title="lfpweather-timestral-metrics")

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthcheck")
    def healthcheck(response: Response) -> dict:
        if not is_ready():
            response.status_code = 503
            return {"status": "loading"}
        return {"status": "ok"}

    return app
