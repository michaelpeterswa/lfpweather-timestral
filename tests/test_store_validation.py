import numpy as np
import pytest

from lfpweather_timestral.store import SourceForecast, write_run


def _q():
    return np.tile(np.linspace(50, 60, 21)[:, None], (1, 24))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_write_run_rejects_non_finite_values(bad):
    v = np.full(24, 55.0)
    v[3] = bad
    with pytest.raises(ValueError, match="non-finite"):
        write_run(None, __import__("datetime").datetime.now(), [SourceForecast("x", v)])


def test_write_run_rejects_non_finite_quantiles():
    q = _q()
    q[5, 7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write_run(
            None,
            __import__("datetime").datetime.now(),
            [SourceForecast("x", np.full(24, 55.0), q)],
        )
