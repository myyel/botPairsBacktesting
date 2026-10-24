"""ZScoreModel (RAPOR §3.2): causal EWMA z-score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_bt.models.zscore import ZScoreModel


def test_matches_pandas_ewm(ou_spread: np.ndarray) -> None:
    hl = 60
    path = ZScoreModel(hl).run(ou_spread)
    s = pd.Series(ou_spread)
    mean = s.ewm(halflife=hl, adjust=False).mean().to_numpy()
    var = s.ewm(halflife=hl, adjust=False).var(bias=True).to_numpy()
    np.testing.assert_allclose(path.mean, mean, rtol=1e-10)
    np.testing.assert_allclose(path.std[1:], np.sqrt(var[1:]), rtol=1e-8)
    np.testing.assert_allclose(path.z[1:], ((s - mean) / np.sqrt(var)).to_numpy()[1:], rtol=1e-8)


def test_first_bar_and_zero_vol_are_nan() -> None:
    path = ZScoreModel(10).run(np.full(50, 3.0))
    assert np.all(np.isnan(path.z))
    assert np.all(path.std == 0.0)


def test_single_bar_and_empty() -> None:
    assert np.isnan(ZScoreModel(10).run(np.array([1.0])).z[0])
    assert ZScoreModel(10).run(np.array([])).z.shape == (0,)


def test_nan_input_keeps_state(ou_spread: np.ndarray) -> None:
    x = ou_spread[:200].copy()
    x[100] = np.nan
    path = ZScoreModel(20).run(x)
    assert np.isnan(path.z[100])
    assert path.mean[100] == path.mean[99]
    clean = ZScoreModel(20).run(np.delete(x, 100))
    np.testing.assert_allclose(path.z[101:], clean.z[100:])


def test_step_equals_run(ou_spread: np.ndarray) -> None:
    model = ZScoreModel(30)
    path = model.run(ou_spread[:300])
    state = None
    for i, v in enumerate(ou_spread[:300]):
        state, z = model.step(state, float(v))
        assert z == pytest.approx(path.z[i], nan_ok=True)


def test_invalid_halflife() -> None:
    with pytest.raises(ValueError, match="halflife"):
        ZScoreModel(0)
