from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from pairs_bt.data.exchange_info import (
    ExchangeInfoSnapshot,
    ExchangeInfoStore,
    derive_tick_size,
)
from tests.fixtures.binance_archive import ei_symbol, exchange_info_json

SNAP_MS = 1_792_137_674_765


def _snap(symbols: list[dict[str, object]], ts: int = SNAP_MS) -> ExchangeInfoSnapshot:
    return ExchangeInfoSnapshot.from_json(json.loads(exchange_info_json(symbols, ts)))


def test_parse_snapshot() -> None:
    snap = _snap([ei_symbol("BTCUSDT", tick="0.10", step="0.001", min_notional="50")])
    f = snap.filters["BTCUSDT"]
    assert f.tick_size == Decimal("0.10")
    assert f.step_size == Decimal("0.001")
    assert f.min_notional == Decimal("50")
    assert f.valid_from_ms == SNAP_MS
    assert not f.approximate
    meta = snap.meta["BTCUSDT"]
    assert meta.contract_type == "PERPETUAL"
    assert meta.status == "TRADING"


def test_filters_after_snapshot_are_exact() -> None:
    store = ExchangeInfoStore([_snap([ei_symbol("BTCUSDT")])])
    f = store.filters_at("BTCUSDT", SNAP_MS + 1)
    assert f is not None
    assert not f.approximate


def test_filters_before_snapshot_are_approximate_with_derived_tick() -> None:
    store = ExchangeInfoStore(
        [_snap([ei_symbol("BTCUSDT", tick="0.10")])],
        derived_ticks={"BTCUSDT": [(1_000, Decimal("0.01")), (5_000, Decimal("0.1"))]},
    )
    f_early = store.filters_at("BTCUSDT", 2_000)
    assert f_early is not None
    assert f_early.approximate
    assert f_early.tick_size == Decimal("0.01")  # o tarihteki fiyat grid'i
    assert f_early.step_size == Decimal("0.001")  # en erken snapshot'tan
    f_later = store.filters_at("BTCUSDT", 6_000)
    assert f_later is not None
    assert f_later.tick_size == Decimal("0.1")


def test_before_snapshot_without_derived_tick_uses_snapshot_tick_approx() -> None:
    store = ExchangeInfoStore([_snap([ei_symbol("BTCUSDT", tick="0.10")])])
    f = store.filters_at("BTCUSDT", 0)
    assert f is not None
    assert f.approximate
    assert f.tick_size == Decimal("0.10")


def test_latest_snapshot_before_ts_wins() -> None:
    store = ExchangeInfoStore(
        [
            _snap([ei_symbol("BTCUSDT", tick="0.5")], ts=2_000),
            _snap([ei_symbol("BTCUSDT", tick="0.1")], ts=1_000),
        ]
    )
    assert store.filters_at("BTCUSDT", 1_500).tick_size == Decimal("0.1")  # type: ignore[union-attr]
    assert store.filters_at("BTCUSDT", 2_500).tick_size == Decimal("0.5")  # type: ignore[union-attr]


def test_unknown_symbol_returns_none() -> None:
    assert ExchangeInfoStore([_snap([ei_symbol("BTCUSDT")])]).filters_at("OLDUSDT", 0) is None


def test_store_load_from_directory(tmp_path: Path) -> None:
    d = tmp_path / "exchange_info"
    d.mkdir()
    (d / f"exchangeInfo_{SNAP_MS}.json").write_bytes(
        exchange_info_json([ei_symbol("ETHUSDT")], SNAP_MS)
    )
    store = ExchangeInfoStore.load(d)
    assert store.filters_at("ETHUSDT", SNAP_MS) is not None
    assert store.meta("ETHUSDT") is not None


@pytest.mark.parametrize(
    ("prices", "tick"),
    [
        ([108221.4, 108168.7, 108290.0, 108208.3], "0.1"),
        ([0.012345, 0.012351, 0.012362], "0.000001"),
        ([100.0, 105.0, 110.0, 95.0], "5"),
        ([1.25, 1.5, 1.75], "0.25"),
    ],
)
def test_derive_tick_size(prices: list[float], tick: str) -> None:
    assert derive_tick_size(np.array(prices)) == Decimal(tick)


def test_derive_tick_size_empty_or_nan() -> None:
    assert derive_tick_size(np.array([])) is None
    assert derive_tick_size(np.array([np.nan, 1.5, 2.0])) == Decimal("0.5")


def test_derive_tick_size_robust_to_outliers() -> None:
    rng = np.random.default_rng(0)
    prices = np.round(200 + rng.integers(0, 5000, size=10_000) * 0.01, 2)
    prices[:2] = [203.491, 219.467]  # gerçek veride görülen alt-tick aykırılar (SOLUSDT 2025-09)
    assert derive_tick_size(prices) == Decimal("0.001")
    assert derive_tick_size(prices, min_fraction=0.999) == Decimal("0.01")


def test_derive_tick_size_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="min_fraction"):
        derive_tick_size(np.array([1.0]), min_fraction=0.0)
