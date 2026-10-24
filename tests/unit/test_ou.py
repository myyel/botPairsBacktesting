"""OUCalibrator (RAPOR §3.3): AR(1) ile kappa ve half-life."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pairs_bt.research.ou import OUCalibrator
from tests.conftest import make_ou_spread


@pytest.mark.parametrize("kappa", [0.05, 0.2, 0.5])
def test_recovers_known_half_life(kappa: float, rng: np.random.Generator) -> None:
    x = make_ou_spread(rng, 5_000, kappa=kappa, sigma=0.01, mu=0.3)
    p = OUCalibrator().fit(x, bar_minutes=5.0)
    expected_min = math.log(2) / -math.log(1 - kappa) * 5.0
    assert p.valid
    assert p.half_life_min == pytest.approx(expected_min, rel=0.15)
    assert p.mu == pytest.approx(0.3, abs=0.01)
    # durağan std: sigma / sqrt(1 - phi^2)
    assert p.sigma_eq == pytest.approx(0.01 / math.sqrt(1 - (1 - kappa) ** 2), rel=0.1)


def test_random_walk_has_no_useful_half_life(random_walk: np.ndarray) -> None:
    p = OUCalibrator().fit(random_walk, bar_minutes=1.0)
    assert not p.valid or p.half_life_min > 500


def test_alternating_series_invalid() -> None:
    x = np.tile([1.0, -1.0], 200) + np.linspace(0, 1e-3, 400)
    assert not OUCalibrator().fit(x, bar_minutes=1.0).valid


@pytest.mark.parametrize("series", [np.array([1.0, 2.0]), np.full(100, 1.0), np.array([])])
def test_degenerate_invalid(series: np.ndarray) -> None:
    p = OUCalibrator().fit(series, bar_minutes=1.0)
    assert not p.valid
    assert math.isnan(p.half_life_min)


def test_nan_values_skipped(rng: np.random.Generator) -> None:
    x = make_ou_spread(rng, 3_000, kappa=0.1, sigma=0.01)
    x[[5, 6, 1000]] = np.nan
    assert OUCalibrator().fit(x, bar_minutes=1.0).valid


# ---------------------------------------------------------------- EWMA sapması


def test_random_walk_deviation_half_life_equals_ewma_half_life(rng: np.random.Generator) -> None:
    """Sınır durumu: RW'nin EWMA sapması da AR(1) gibi davranır, HL = EWMA HL.

    Bu yüzden ewma_deviation yönteminde half_life_max < EWMA HL olmalı (config validator).
    """
    from pairs_bt.research.ou import ewma_deviation

    x = np.cumsum(rng.standard_normal(50_000))
    p = OUCalibrator().fit(ewma_deviation(x, halflife_bars=12), bar_minutes=5.0)
    assert p.half_life_min == pytest.approx(60.0, rel=0.05)


def test_fast_ou_deviation_half_life_below_ewma(rng: np.random.Generator) -> None:
    from pairs_bt.research.ou import ewma_deviation

    x = make_ou_spread(rng, 20_000, kappa=0.25, sigma=0.01)  # HL ≈ 2.4 bar
    p = OUCalibrator().fit(ewma_deviation(x, halflife_bars=12), bar_minutes=5.0)
    assert p.half_life_min < 20.0


def test_slow_ou_with_noise_deviation_is_shorter_than_level(rng: np.random.Generator) -> None:
    """Yavaş OU seviye + hızlı OU gürültü: seviye HL uzun, sapma HL kısa (sinyalin ölçtüğü)."""
    from pairs_bt.research.ou import ewma_deviation

    x = make_ou_spread(rng, 20_000, kappa=0.02, sigma=0.004) + make_ou_spread(
        rng, 20_000, kappa=0.3, sigma=0.004
    )
    level = OUCalibrator().fit(x, bar_minutes=5.0)
    dev = OUCalibrator().fit(ewma_deviation(x, halflife_bars=12), bar_minutes=5.0)
    assert level.half_life_min > 45.0
    assert dev.half_life_min < 45.0


def test_ewma_deviation_nan_and_causal(ou_spread: np.ndarray) -> None:
    from pairs_bt.research.ou import ewma_deviation

    x = ou_spread[:500].copy()
    x[100] = np.nan
    d = ewma_deviation(x, halflife_bars=12)
    assert np.isnan(d[100])
    y = x.copy()
    y[300:] += 5.0
    np.testing.assert_allclose(ewma_deviation(y, 12)[:300], d[:300], equal_nan=True)
