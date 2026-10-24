"""research.stats — OLS, ADF, Engle-Granger, Hurst (RAPOR §8.2). statsmodels referans alınır."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller, coint

from pairs_bt.research.stats import adf_tstat, engle_granger, hurst_exponent, ols_fit
from tests.conftest import make_cointegrated_pair, make_ou_spread


def test_ols_matches_polyfit(rng: np.random.Generator) -> None:
    x = rng.standard_normal(500)
    y = 0.3 + 1.7 * x + 0.1 * rng.standard_normal(500)
    fit = ols_fit(y, x)
    beta, alpha = np.polyfit(x, y, 1)
    assert fit.alpha == pytest.approx(alpha)
    assert fit.beta == pytest.approx(beta)
    np.testing.assert_allclose(fit.resid, y - alpha - beta * x, atol=1e-12)


@pytest.mark.parametrize("lags", [0, 1, 3])
def test_adf_matches_statsmodels(lags: int, ou_spread: np.ndarray, random_walk: np.ndarray) -> None:
    for series in (ou_spread[:2_000], random_walk[:2_000]):
        ref = adfuller(series, maxlag=lags, autolag=None, regression="n", result_object=False)[0]
        assert adf_tstat(series, lags) == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize("lags", [0, 1, 2])
def test_engle_granger_matches_statsmodels_coint(lags: int, rng: np.random.Generator) -> None:
    log_a, log_b = make_cointegrated_pair(rng, 2_000, sigma_b=0.01)
    res = engle_granger(log_a, log_b, lags)
    stat, pval, _ = coint(log_a, log_b, trend="c", maxlag=lags, autolag=None)
    assert res.adf_stat == pytest.approx(stat, rel=1e-9)
    assert res.p_value == pytest.approx(pval, rel=1e-9)
    assert res.n_obs == 2_000


def test_cointegrated_pair_rejects_unit_root(rng: np.random.Generator) -> None:
    log_a, log_b = make_cointegrated_pair(rng, 2_000, beta=1.3, alpha=0.1, sigma_b=0.01)
    res = engle_granger(log_a, log_b, 1)
    assert res.p_value < 0.01
    assert res.beta == pytest.approx(1.3, abs=0.05)


def test_independent_random_walks_not_cointegrated(rng: np.random.Generator) -> None:
    pvals = [
        engle_granger(
            np.cumsum(rng.standard_normal(2_000)), np.cumsum(rng.standard_normal(2_000)), 1
        ).p_value
        for _ in range(20)
    ]
    assert np.median(pvals) > 0.2
    assert sum(p < 0.05 for p in pvals) <= 3


def test_engle_granger_drops_nan_rows(rng: np.random.Generator) -> None:
    log_a, log_b = make_cointegrated_pair(rng, 1_000, sigma_b=0.01)
    a = log_a.copy()
    a[[10, 500]] = np.nan
    res = engle_granger(a, log_b, 1)
    assert res.n_obs == 998
    assert np.isfinite(res.p_value)


@pytest.mark.parametrize("kind", ["constant_x", "too_short"])
def test_engle_granger_degenerate_returns_nan(kind: str) -> None:
    if kind == "constant_x":
        y, x = np.linspace(0, 1, 100), np.full(100, 2.0)
    else:
        y, x = np.arange(5.0), np.arange(5.0) * 2
    res = engle_granger(y, x, 1)
    assert np.isnan(res.p_value)


def test_hurst_random_walk_near_half(random_walk: np.ndarray) -> None:
    assert hurst_exponent(random_walk, 2, 20) == pytest.approx(0.5, abs=0.05)


def test_hurst_ou_is_mean_reverting(rng: np.random.Generator) -> None:
    x = make_ou_spread(rng, 5_000, kappa=0.1, sigma=0.01)
    assert hurst_exponent(x, 2, 20) < 0.45  # config max_hurst


@pytest.mark.parametrize("series", [np.full(500, 1.0), np.arange(10.0), np.array([])])
def test_hurst_degenerate_is_nan(series: np.ndarray) -> None:
    # sabit seri (tau=0), max_lag'a göre çok kısa seri, boş seri
    assert np.isnan(hurst_exponent(series, 2, 20))


# ---------------------------------------------------------------- sıçrama filtresi


def test_robust_scale_matches_normal_std(rng: np.random.Generator) -> None:
    from pairs_bt.research.stats import robust_scale

    assert robust_scale(rng.standard_normal(20_000) * 3.0) == pytest.approx(3.0, rel=0.03)
    assert robust_scale(np.array([1.0, 1.0, 1.0])) == 0.0
    assert np.isnan(robust_scale(np.array([])))


def test_winsorize_increments_clean_series_unchanged(ou_spread: np.ndarray) -> None:
    from pairs_bt.research.stats import winsorize_increments

    out, is_jump = winsorize_increments(ou_spread, k=8.0)
    np.testing.assert_allclose(out, ou_spread)
    assert not is_jump.any()


def test_winsorize_increments_clips_wick(ou_spread: np.ndarray) -> None:
    """Çöküş fitili: tek bar'da +0.5 sonra geri dönüş. Kırpılır, seviye kalıcı kaymaz."""
    from pairs_bt.research.stats import robust_scale, winsorize_increments

    x = ou_spread.copy()
    x[2_000] += 0.5
    out, is_jump = winsorize_increments(x, k=8.0)
    assert is_jump.sum() == 2  # yukarı ve aşağı artış
    assert np.flatnonzero(is_jump).tolist() == [2_000, 2_001]
    scale = robust_scale(np.diff(x))
    assert np.max(np.abs(np.diff(out))) <= 8.0 * scale + 1e-12
    assert out[0] == x[0]
    # fitil simetrik → kalıcı kayma en fazla fitil bar'larının kendi (kırpılan) artışı kadar
    assert out[-1] == pytest.approx(x[-1], abs=abs(x[2_001] - x[1_999]) + 1e-12)


def test_winsorize_increments_nan_and_constant() -> None:
    from pairs_bt.research.stats import winsorize_increments

    x = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
    out, is_jump = winsorize_increments(x, k=5.0)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(x))
    assert not is_jump.any()
    assert winsorize_increments(np.array([]), k=5.0)[0].size == 0
