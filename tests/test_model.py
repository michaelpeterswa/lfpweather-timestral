import numpy as np
import pytest
import torch

from lfpweather_timestral.model import CudaUnavailable, Model


def test_load_raises_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CudaUnavailable):
        Model.load("mistralai/Timestral-5M", require_cuda=True)


def _history() -> np.ndarray:
    t = np.arange(720)
    return 15.0 + 5.0 * np.sin(2 * np.pi * t / 24)


def _check(fc) -> None:
    assert fc.median.shape == (24,)
    assert fc.quantiles.shape == (21, 24)
    assert len(fc.levels) == 21
    assert np.all(np.diff(fc.quantiles, axis=0) >= -1e-6)
    assert np.isfinite(fc.quantiles).all()


@pytest.fixture(scope="module")
def model() -> Model:
    return Model.load("mistralai/Timestral-5M", require_cuda=False)


@pytest.mark.slow
def test_forecast_shapes(model):
    _check(model.forecast(_history(), 24))


@pytest.mark.slow
def test_forecast_with_covariate_shapes(model):
    t = np.arange(720)
    past = 15.0 + 5.0 * np.sin(2 * np.pi * t / 24)
    past[:360] = np.nan
    future = 15.0 + 5.0 * np.sin(2 * np.pi * np.arange(720, 744) / 24)
    _check(model.forecast(_history(), 24, covariate_past=past, covariate_future=future))
