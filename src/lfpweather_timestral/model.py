"""Timestral model adapter. The only module that imports `timestral`.

Pinned API (timestral v1.0.0, commit a534bdb):

- ``TimestralForecaster.from_pretrained(path_or_repo_id, device="cpu", ...)``
- ``forecaster.forecast(context, prediction_length, observation_mask=None,
  past_covariates=None, known_covariates=None, ..., symmetric_averaging=True,
  clip_nonnegative=True) -> Forecast``
- ``context``: ``(T,)``, ``(B, T)`` or ``(B, V, T)``; NaN marks a missing value.
- ``known_covariates``: ``(B, C, T + prediction_length)``, i.e. ONE array covering the
  context window followed by the horizon, so ``covariate_past`` and ``covariate_future``
  are concatenated here. NaN is accepted anywhere: NaN positions in the context part are
  masked as unobserved (a mostly-NaN past window works), NaN in the future part is masked.
  ``past_covariates`` (``(B, C, T)``) is for covariates unknown in the future; not used.
- ``Forecast.quantiles``: ``(B, V, H, Q)`` sorted along Q; ``Forecast.quantile_levels``:
  the model's levels (21 for the released checkpoints); ``Forecast.median`` is ``(B, V, H)``.
- Covariate rows are only usable if the checkpoint ``supports_known_covariates``.
- ``clip_nonnegative`` defaults to True and clamps the forecast at 0 whenever the
  observed context is all non-negative. That is wrong for temperature (a warm history
  followed by a sub-zero forecast), so it is disabled here.
"""

from dataclasses import dataclass

import numpy as np
import torch
from timestral import TimestralForecaster

from lfpweather_timestral.metrics import FORECAST_SECONDS

N_LEVELS = 21


class CudaUnavailable(RuntimeError):
    """CUDA was required but torch cannot see a GPU."""


@dataclass
class QuantileForecast:
    """median `(H,)`, quantiles `(Q, H)`, levels `(Q,)`."""

    median: np.ndarray
    quantiles: np.ndarray
    levels: tuple[float, ...]


class Model:
    def __init__(self, forecaster: TimestralForecaster):
        self._forecaster = forecaster

    @classmethod
    def load(cls, model_id: str, require_cuda: bool) -> "Model":
        if require_cuda and not torch.cuda.is_available():
            raise CudaUnavailable("CUDA is required but torch.cuda.is_available() is False")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        forecaster = TimestralForecaster.from_pretrained(model_id, device=device)
        assert len(forecaster.quantile_levels) == N_LEVELS, forecaster.quantile_levels
        return cls(forecaster)

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        covariate_past: np.ndarray | None = None,
        covariate_future: np.ndarray | None = None,
    ) -> QuantileForecast:
        known = None
        if covariate_future is not None:
            past = (
                np.full(len(history), np.nan)
                if covariate_past is None
                else np.asarray(covariate_past, dtype=np.float32)
            )
            assert len(past) == len(history), "covariate_past must match history length"
            assert len(covariate_future) == horizon, "covariate_future must match horizon"
            known = np.concatenate([past, np.asarray(covariate_future, dtype=np.float32)])[
                None, None
            ]
        with FORECAST_SECONDS.time(), torch.inference_mode():
            fc = self._forecaster.forecast(
                np.asarray(history, dtype=np.float32),
                horizon,
                known_covariates=known,
                clip_nonnegative=False,
            )
        levels = tuple(fc.quantile_levels)
        assert len(levels) == N_LEVELS
        q = fc.quantiles[0, 0].float().cpu().numpy().T  # (Q, H)
        return QuantileForecast(
            median=fc.median[0, 0].float().cpu().numpy(), quantiles=q, levels=levels
        )
