"""Look-ahead koruma test KALIBI (CLAUDE.md invariant #1, §6).

Her yeni feature/sinyal fonksiyonu `CAUSAL_FUNCTIONS` listesine eklenmelidir:
t sonrasındaki veriyi bozmak, t ve öncesindeki çıktıyı DEĞİŞTİRMEMELİDİR.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pairs_bt.data.normalize import KLINES_SCHEMA
from pairs_bt.data.synchronizer import TimeSynchronizer
from pairs_bt.models import KalmanHedgeModel, KalmanState, ZScoreModel

SeriesFn = Callable[[np.ndarray], np.ndarray]


def _ewma_zscore_reference(x: np.ndarray, halflife: int = 60) -> np.ndarray:
    """Referans causal EWMA z-score (RAPOR §3.2). Gerçek implementasyon models/ altına gelince
    bu listeye o fonksiyon da eklenir."""
    s = pd.Series(x)
    mean = s.ewm(halflife=halflife, adjust=False).mean()
    std = s.ewm(halflife=halflife, adjust=False).std()
    return ((s - mean) / std).to_numpy()


def _leaky_centered_mean(x: np.ndarray) -> np.ndarray:
    """KASITLI sızıntılı örnek: center=True geleceği kullanır. Guard bunu yakalamalı."""
    return pd.Series(x).rolling(21, center=True, min_periods=1).mean().to_numpy()


def _synchronizer_close(x: np.ndarray) -> np.ndarray:
    """TimeSynchronizer: her 7. bar eksik (ffill yolu), 3. sembol ileride başlıyor."""
    m = 60_000
    keep = np.arange(x.size) % 7 != 3
    ts = (np.arange(x.size, dtype=np.int64) + 1) * m

    def bars(values: np.ndarray, mask: np.ndarray) -> pl.DataFrame:
        n = int(mask.sum())
        v = values[mask]
        cols = dict.fromkeys(("open", "high", "low", "close"), v)
        return pl.DataFrame(
            {
                "ts_ms": ts[mask],
                "open_time_ms": ts[mask] - m,
                **cols,
                "volume": np.ones(n),
                "quote_volume": np.ones(n),
                "trades": np.ones(n, dtype=np.int64),
                "taker_buy_volume": np.zeros(n),
                "taker_buy_quote_volume": np.zeros(n),
            },
            schema=KLINES_SCHEMA,
        )

    late = keep & (np.arange(x.size) >= 50)
    panel = TimeSynchronizer(m, max_ffill_bars=2).align(
        {"A": bars(x, keep), "B": bars(x * 2, late)}, int(ts[0]), int(ts[-1]) + 1
    )
    return panel.close.ravel()[::2] + np.nan_to_num(panel.close.ravel()[1::2])


def _zscore_model(x: np.ndarray) -> np.ndarray:
    return ZScoreModel(60).run(x).z


def _kalman_spread_z(x: np.ndarray) -> np.ndarray:
    """x = ln A; ln B sabit bir random walk. Kalman inovasyonu + beta + z-score zinciri."""
    log_b = np.cumsum(np.full(x.size, 0.001)) + 4.0
    path = KalmanHedgeModel(1e-4, 1e-3).run(
        x, log_b, KalmanState(0.0, 1.0, p_aa=1.0, p_ab=0.0, p_bb=1.0)
    )
    return path.beta + ZScoreModel(30).run(path.innovation).z


CAUSAL_FUNCTIONS: list[SeriesFn] = [
    _ewma_zscore_reference,
    _synchronizer_close,
    _zscore_model,
    _kalman_spread_z,
]


def is_causal(fn: SeriesFn, x: np.ndarray, cut: int, rng: np.random.Generator) -> bool:
    perturbed = x.copy()
    perturbed[cut + 1 :] += rng.standard_normal(len(x) - cut - 1) * 10.0
    a, b = fn(x)[: cut + 1], fn(perturbed)[: cut + 1]
    return bool(np.allclose(a, b, equal_nan=True))


@pytest.mark.parametrize("fn", CAUSAL_FUNCTIONS, ids=lambda f: f.__name__)
def test_function_is_causal(fn: SeriesFn, ou_spread: np.ndarray, rng: np.random.Generator) -> None:
    for cut in (100, 1_000, 4_000):
        assert is_causal(fn, ou_spread, cut, rng), f"{fn.__name__} t={cut}'de look-ahead içeriyor"


def test_guard_detects_leak(ou_spread: np.ndarray, rng: np.random.Generator) -> None:
    assert not is_causal(_leaky_centered_mean, ou_spread, 1_000, rng)
