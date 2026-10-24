"""Çok sembollü sentetik 5m panel + lifecycle + exchange-info (Faz 2 research testleri).

Panel doğrudan `SyncedPanel` olarak kurulur (parquet/synchronizer katmanını atlar).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

import numpy as np

from pairs_bt.data.exchange_info import ExchangeInfoSnapshot, ExchangeInfoStore, SymbolMeta
from pairs_bt.data.lifecycle import LifecycleEntry, SymbolLifecycle
from pairs_bt.data.synchronizer import SyncedPanel
from pairs_bt.domain.instruments import SymbolFilters

M5 = 300_000
DAY = 86_400_000
T0 = 1_756_684_800_000  # 2025-09-01 00:00 UTC
BARS_PER_DAY = DAY // M5


def make_panel(
    log_prices: Mapping[str, np.ndarray],
    *,
    start_ms: int = T0,
    interval_ms: int = M5,
    quote_volume: float | Mapping[str, float] = 1e6,
) -> SyncedPanel:
    """Bar kapanışları `start_ms + interval, start_ms + 2·interval, ...`; NaN log fiyat = stale."""
    symbols = tuple(sorted(log_prices))
    n = len(next(iter(log_prices.values())))
    ts = start_ms + interval_ms * (np.arange(n, dtype=np.int64) + 1)
    close = np.column_stack([np.exp(log_prices[s]) for s in symbols])
    stale = ~np.isfinite(close)
    qv_map = (
        quote_volume if isinstance(quote_volume, Mapping) else dict.fromkeys(symbols, quote_volume)
    )
    qvol = np.column_stack([np.full(n, qv_map[s]) for s in symbols])
    qvol[stale] = 0.0
    return SyncedPanel(ts, symbols, close, qvol, ~stale, stale)


def make_lifecycle(
    symbols: Sequence[str],
    *,
    first_ms: int = T0,
    last_ms: int = T0 + 400 * DAY,
    overrides: Mapping[str, LifecycleEntry] | None = None,
) -> SymbolLifecycle:
    entries = {
        s: LifecycleEntry(
            symbol=s,
            first_data_ms=first_ms,
            last_data_ms=last_ms,
            contract_type="PERPETUAL",
            status="TRADING",
            onboard_ms=first_ms - 365 * DAY,
            delivery_ms=None,
        )
        for s in symbols
    }
    entries.update(overrides or {})
    return SymbolLifecycle(tuple(entries.values()))


def make_store(
    symbols: Sequence[str], *, server_time_ms: int = T0 + 500 * DAY
) -> ExchangeInfoStore:
    filters = {
        s: SymbolFilters(
            symbol=s,
            tick_size=Decimal("0.0001"),
            step_size=Decimal("0.001"),
            min_qty=Decimal("0.001"),
            min_notional=Decimal("5"),
            status="TRADING",
            valid_from_ms=server_time_ms,
        )
        for s in symbols
    }
    meta = {s: SymbolMeta(s, "PERPETUAL", "USDT", "TRADING", None, None) for s in symbols}
    return ExchangeInfoStore([ExchangeInfoSnapshot(server_time_ms, filters, meta)])


def make_pair_market(
    rng: np.random.Generator, n: int, *, extra_b_pair: bool = False
) -> dict[str, np.ndarray]:
    """Ortak piyasa faktörlü semboller: iki cointegrated çift + korelasyonlu ama cointegrated
    OLMAYAN 4 sembol.

    - AAAUSDT ~ 0.3 + 1.2·BBBUSDT + OU(kappa 0.25)   (HL ≈ 2.4 bar ≈ 12 dk @5m)
    - CCCUSDT ~ -0.1 + 1.0·DDDUSDT + OU(kappa 0.20)  (HL ≈ 3.1 bar ≈ 16 dk @5m)
    - EEE..HHH: piyasa faktörü + bağımsız random walk (spread'leri unit-root)
    - `extra_b_pair`: IIIUSDT ~ 0.1 + 0.9·BBBUSDT + OU — BBB iki pair'de merkezi olur
    """
    market = np.cumsum(0.004 * rng.standard_normal(n))

    def ou(kappa: float, sigma: float) -> np.ndarray:
        x = np.zeros(n)
        eps = rng.standard_normal(n)
        for t in range(1, n):
            x[t] = (1 - kappa) * x[t - 1] + sigma * eps[t]
        return x

    def rw(sigma: float) -> np.ndarray:
        return np.cumsum(sigma * rng.standard_normal(n))

    log_b = 3.0 + market + rw(0.0008)
    log_d = 1.0 + market + rw(0.0008)
    out = {
        "AAAUSDT": 0.3 + 1.2 * log_b + ou(0.25, 0.004),
        "BBBUSDT": log_b,
        "CCCUSDT": -0.1 + 1.0 * log_d + ou(0.20, 0.004),
        "DDDUSDT": log_d,
        "EEEUSDT": 2.0 + market + rw(0.0012),
        "FFFUSDT": 4.0 + market + rw(0.0012),
        "GGGUSDT": 0.5 + 1.1 * market + rw(0.0012),
        "HHHUSDT": 1.5 + 0.9 * market + rw(0.0012),
    }
    if extra_b_pair:
        out["IIIUSDT"] = 0.1 + 0.9 * log_b + ou(0.25, 0.004)
    return out
