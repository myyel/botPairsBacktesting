"""UniverseFilter (RAPOR §8.1, CLAUDE.md invariant #1, #3, #9)."""

from __future__ import annotations

import numpy as np
import pytest

from pairs_bt.config import MarketConfig, StrategyConfig
from pairs_bt.data.lifecycle import LifecycleEntry
from pairs_bt.research.universe import UniverseFilter, UniverseReason
from tests.fixtures.synthetic_panel import (
    BARS_PER_DAY,
    DAY,
    T0,
    make_lifecycle,
    make_panel,
    make_store,
)

WINDOW_DAYS = 28
N_BARS = (WINDOW_DAYS + 5) * BARS_PER_DAY
CUTOFF = T0 + (WINDOW_DAYS + 2) * DAY  # 2 gün sonrası da veri var (look-ahead testi için)


def _market(cfg: StrategyConfig, **kw: object) -> MarketConfig:
    return cfg.market.model_copy(update=kw)


def _prices(rng: np.random.Generator, symbols: list[str]) -> dict[str, np.ndarray]:
    return {s: 1.0 + np.cumsum(0.002 * rng.standard_normal(N_BARS)) for s in symbols}


def _reasons(result: object) -> dict[str, UniverseReason]:
    return {r.symbol: r.reason for r in result.rows}  # type: ignore[attr-defined]


@pytest.fixture
def base(rng: np.random.Generator, strategy_config: StrategyConfig):  # type: ignore[no-untyped-def]
    syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    prices = _prices(rng, syms)
    flt = UniverseFilter(_market(strategy_config, min_quote_volume_24h_usdt=1e6), WINDOW_DAYS)
    return syms, prices, flt


def test_healthy_symbols_eligible(base) -> None:  # type: ignore[no-untyped-def]
    syms, prices, flt = base
    res = flt.evaluate(make_panel(prices), CUTOFF, make_lifecycle(syms), make_store(syms))
    assert res.eligible == tuple(syms)
    row = res.rows[0]
    assert row.coverage == pytest.approx(1.0)
    assert row.zero_trade_frac == 0.0
    assert row.median_quote_volume_24h_usdt == pytest.approx(1e6 * BARS_PER_DAY)


def test_excluded_symbol(base, strategy_config: StrategyConfig) -> None:  # type: ignore[no-untyped-def]
    syms, prices, _ = base
    flt = UniverseFilter(
        _market(strategy_config, min_quote_volume_24h_usdt=1e6, exclude_symbols=("BBBUSDT",)),
        WINDOW_DAYS,
    )
    res = flt.evaluate(make_panel(prices), CUTOFF, make_lifecycle(syms), make_store(syms))
    assert _reasons(res)["BBBUSDT"] == UniverseReason.EXCLUDED


def test_not_active_after_delivery(base) -> None:  # type: ignore[no-untyped-def]
    syms, prices, flt = base
    dead = LifecycleEntry(
        "BBBUSDT", T0, CUTOFF - DAY, "PERPETUAL", "SETTLING", T0 - 400 * DAY, None
    )
    lc = make_lifecycle(syms, overrides={"BBBUSDT": dead})
    res = flt.evaluate(make_panel(prices), CUTOFF, lc, make_store(syms))
    assert _reasons(res)["BBBUSDT"] == UniverseReason.NOT_ACTIVE


def test_future_delisting_does_not_exclude(base) -> None:  # type: ignore[no-untyped-def]
    """Survivorship (#3): cutoff'tan SONRA delist olacak sembol cutoff anında universe'te."""
    syms, prices, flt = base
    later = LifecycleEntry(
        "BBBUSDT", T0, CUTOFF + DAY, "PERPETUAL", "SETTLING", T0 - 400 * DAY, CUTOFF + DAY
    )
    lc = make_lifecycle(syms, overrides={"BBBUSDT": later})
    res = flt.evaluate(make_panel(prices), CUTOFF, lc, make_store(syms))
    assert "BBBUSDT" in res.eligible


def test_missing_lifecycle_entry(base) -> None:  # type: ignore[no-untyped-def]
    syms, prices, flt = base
    res = flt.evaluate(make_panel(prices), CUTOFF, make_lifecycle(syms[:2]), make_store(syms))
    assert _reasons(res)["CCCUSDT"] == UniverseReason.NOT_ACTIVE


def test_non_perpetual_excluded(base) -> None:  # type: ignore[no-untyped-def]
    syms, prices, flt = base
    tradifi = LifecycleEntry(
        "CCCUSDT", T0, T0 + 400 * DAY, "TRADIFI_PERPETUAL", "TRADING", T0 - 400 * DAY, None
    )
    lc = make_lifecycle(syms, overrides={"CCCUSDT": tradifi})
    res = flt.evaluate(make_panel(prices), CUTOFF, lc, make_store(syms))
    assert _reasons(res)["CCCUSDT"] == UniverseReason.CONTRACT_TYPE


def test_no_filters_excluded(base) -> None:  # type: ignore[no-untyped-def]
    """Invariant #9: emir kuralı bilinmeyen sembolde emir üretilemez."""
    syms, prices, flt = base
    res = flt.evaluate(make_panel(prices), CUTOFF, make_lifecycle(syms), make_store(syms[:2]))
    assert _reasons(res)["CCCUSDT"] == UniverseReason.NO_FILTERS


def test_young_listing_excluded(base) -> None:  # type: ignore[no-untyped-def]
    syms, prices, flt = base
    young = LifecycleEntry(
        "AAAUSDT", T0, T0 + 400 * DAY, "PERPETUAL", "TRADING", CUTOFF - DAY, None
    )
    lc = make_lifecycle(syms, overrides={"AAAUSDT": young})
    res = flt.evaluate(make_panel(prices), CUTOFF, lc, make_store(syms))
    assert _reasons(res)["AAAUSDT"] == UniverseReason.LISTING_AGE


def test_low_coverage_excluded(base) -> None:  # type: ignore[no-untyped-def]
    syms, prices, flt = base
    p = dict(prices)
    p["AAAUSDT"] = prices["AAAUSDT"].copy()
    cut_idx = (CUTOFF - T0) // 300_000
    p["AAAUSDT"][cut_idx - 3 * BARS_PER_DAY : cut_idx - 2 * BARS_PER_DAY] = np.nan  # 1 gün stale
    res = flt.evaluate(make_panel(p), CUTOFF, make_lifecycle(syms), make_store(syms))
    assert _reasons(res)["AAAUSDT"] == UniverseReason.LOW_COVERAGE


def test_short_history_is_low_coverage(base) -> None:  # type: ignore[no-untyped-def]
    """Pencereden kısa veri: eksik günler stale sayılır (kısa geçmişle seçim yapılmaz)."""
    syms, prices, flt = base
    p = {s: v[-20 * BARS_PER_DAY :] for s, v in prices.items()}
    panel = make_panel(p, start_ms=T0 + (N_BARS - 20 * BARS_PER_DAY) * 300_000)
    res = flt.evaluate(panel, CUTOFF, make_lifecycle(syms), make_store(syms))
    assert set(_reasons(res).values()) == {UniverseReason.LOW_COVERAGE}


def test_zero_trade_and_low_volume(base) -> None:  # type: ignore[no-untyped-def]
    syms, prices, flt = base
    panel = make_panel(prices, quote_volume={"AAAUSDT": 1e6, "BBBUSDT": 1.0, "CCCUSDT": 1e6})
    panel.quote_volume[: N_BARS // 2, 2] = 0.0  # CCC: yarı yarıya işlemsiz
    res = flt.evaluate(panel, CUTOFF, make_lifecycle(syms), make_store(syms))
    r = _reasons(res)
    assert r["BBBUSDT"] == UniverseReason.LOW_VOLUME
    assert r["CCCUSDT"] == UniverseReason.ZERO_TRADE


def test_data_after_cutoff_is_ignored(base, rng: np.random.Generator) -> None:  # type: ignore[no-untyped-def]
    """Invariant #1: cutoff sonrası veriyi bozmak sonucu değiştirmez."""
    syms, prices, flt = base
    panel = make_panel(prices)
    res = flt.evaluate(panel, CUTOFF, make_lifecycle(syms), make_store(syms))
    after = panel.ts_ms > CUTOFF
    panel.close[after] = np.nan
    panel.stale[after] = True
    panel.quote_volume[after] = 0.0
    res2 = flt.evaluate(panel, CUTOFF, make_lifecycle(syms), make_store(syms))
    assert res == res2
