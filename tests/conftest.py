"""Ortak fixture'lar: sentetik, deterministik seriler (CLAUDE.md §6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pairs_bt.config import DataConfig, StrategyConfig, load_config, load_data_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def strategy_config() -> StrategyConfig:
    return load_config(ROOT / "config" / "strategy_v1.yaml")


@pytest.fixture
def data_config() -> DataConfig:
    return load_data_config(ROOT / "config" / "data_v1.yaml")


def make_ou_spread(
    rng: np.random.Generator, n: int, kappa: float, sigma: float, mu: float = 0.0
) -> np.ndarray:
    """Ayrık OU: x_{t+1} = x_t + kappa*(mu - x_t) + sigma*eps.

    Beklenen half-life = ln2/kappa (bar).
    """
    x = np.empty(n)
    x[0] = mu
    eps = rng.standard_normal(n - 1)
    for t in range(n - 1):
        x[t + 1] = x[t] + kappa * (mu - x[t]) + sigma * eps[t]
    return x


def make_cointegrated_pair(
    rng: np.random.Generator,
    n: int,
    beta: float = 1.3,
    alpha: float = 0.1,
    kappa: float = 0.05,
    sigma_spread: float = 0.002,
    sigma_b: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """log P_B random walk; log P_A = alpha + beta*log P_B + OU spread. (log fiyat döner)."""
    log_b = np.cumsum(sigma_b * rng.standard_normal(n)) + np.log(100.0)
    spread = make_ou_spread(rng, n, kappa=kappa, sigma=sigma_spread)
    log_a = alpha + beta * log_b + spread
    return log_a, log_b


@pytest.fixture
def ou_spread(rng: np.random.Generator) -> np.ndarray:
    return make_ou_spread(rng, 5_000, kappa=0.05, sigma=0.01)


@pytest.fixture
def random_walk(rng: np.random.Generator) -> np.ndarray:
    return np.cumsum(rng.standard_normal(5_000))


@pytest.fixture
def cointegrated_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return make_cointegrated_pair(rng, 5_000)
