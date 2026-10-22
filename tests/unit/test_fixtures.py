"""Sentetik fixture'ların kendisini doğrula — sonraki testlerin dayanağı bunlar."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller


def test_ou_spread_is_stationary(ou_spread: np.ndarray) -> None:
    assert adfuller(ou_spread, result_object=False)[1] < 0.01


def test_random_walk_not_stationary(random_walk: np.ndarray) -> None:
    assert adfuller(random_walk, result_object=False)[1] > 0.05


def test_ou_half_life_recoverable(ou_spread: np.ndarray) -> None:
    dx = np.diff(ou_spread)
    x_lag = ou_spread[:-1]
    slope = np.polyfit(x_lag, dx, 1)[0]
    half_life = np.log(2) / -slope
    assert half_life == pytest.approx(np.log(2) / 0.05, rel=0.25)


def test_cointegrated_pair_ols_beta(cointegrated_pair: tuple[np.ndarray, np.ndarray]) -> None:
    log_a, log_b = cointegrated_pair
    beta, _ = np.polyfit(log_b, log_a, 1)
    assert beta == pytest.approx(1.3, rel=0.1)
