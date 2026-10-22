from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from pairs_bt.data.exchange_info import ExchangeInfoSnapshot, ExchangeInfoStore
from pairs_bt.data.lifecycle import LifecycleEntry, SymbolLifecycle, build_lifecycle
from pairs_bt.data.normalize import normalize_klines
from pairs_bt.data.repository import DataRepository
from tests.fixtures.binance_archive import (
    day_ms,
    ei_symbol,
    exchange_info_json,
    klines_csv,
    make_klines,
    zip_bytes,
)

D1 = day_ms(date(2025, 9, 1))
D2 = day_ms(date(2025, 9, 2))
D3 = day_ms(date(2025, 9, 3))


def _entry(sym: str, first: int, last: int, **kw: object) -> LifecycleEntry:
    base: dict[str, object] = {
        "symbol": sym,
        "first_data_ms": first,
        "last_data_ms": last,
        "contract_type": "PERPETUAL",
        "status": "TRADING",
        "onboard_ms": None,
        "delivery_ms": None,
    }
    base.update(kw)
    return LifecycleEntry(**base)  # type: ignore[arg-type]


def test_active_symbols_respects_data_range_and_delivery() -> None:
    lc = SymbolLifecycle(
        (
            _entry("ZUSDT", D1, D3),
            _entry("AUSDT", D1, D3),
            _entry("DELUSDT", D1, D3, status="SETTLING", delivery_ms=D2),
            _entry("NEWUSDT", D2, D3),
        )
    )
    assert lc.active_symbols(D1) == ("AUSDT", "DELUSDT", "ZUSDT")
    assert lc.active_symbols(D2) == ("AUSDT", "NEWUSDT", "ZUSDT")  # DELUSDT delivery'de çıkar
    assert lc.active_symbols(D3 + 1) == ()


def test_delisted_symbol_is_kept() -> None:
    lc = SymbolLifecycle((_entry("DELUSDT", D1, D2, status="SETTLING", delivery_ms=D2),))
    assert lc.get("DELUSDT") is not None
    assert lc.delisted() == ("DELUSDT",)


def test_listing_age_prefers_onboard_date() -> None:
    e = _entry("AUSDT", D2, D3, onboard_ms=D1)
    assert e.listing_age_days(D3) == 2.0
    e2 = _entry("OLDUSDT", D2, D3)  # onboard bilinmiyor -> ilk veri (alt sınır)
    assert e2.listing_age_days(D3) == 1.0


def test_json_roundtrip(tmp_path: Path) -> None:
    lc = SymbolLifecycle((_entry("AUSDT", D1, D3), _entry("BUSDT", D2, D3, contract_type=None)))
    path = lc.write(tmp_path / "lifecycle.json")
    assert SymbolLifecycle.read(path) == lc


def test_build_lifecycle(rng: np.random.Generator, tmp_path: Path) -> None:
    repo = DataRepository(tmp_path)
    for sym, start, n in (("AUSDT", D1, 10), ("OLDUSDT", D2, 5)):
        df = normalize_klines(zip_bytes("a.csv", klines_csv(make_klines(rng, start, n))), 60_000).df
        repo.write_symbol("klines_1m", sym, df)
    snap = ExchangeInfoSnapshot.from_json(
        json.loads(exchange_info_json([ei_symbol("AUSDT", onboard_ms=D1 - 1000)], 99))
    )
    lc = build_lifecycle(repo, ExchangeInfoStore([snap]))
    a, old = lc.get("AUSDT"), lc.get("OLDUSDT")
    assert a is not None
    assert old is not None
    assert a.first_data_ms == D1 + 60_000
    assert a.last_data_ms == D1 + 10 * 60_000
    assert a.onboard_ms == D1 - 1000
    assert a.contract_type == "PERPETUAL"
    assert old.contract_type is None  # exchangeInfo'da yok ama silinmedi


def test_build_lifecycle_uses_traded_bars(rng: np.random.Generator, tmp_path: Path) -> None:
    """Delist sonrası sentetik (trades=0) bar'lar son veri zamanını uzatmamalı."""
    import polars as pl

    repo = DataRepository(tmp_path)
    df = normalize_klines(zip_bytes("a.csv", klines_csv(make_klines(rng, D1, 10))), 60_000).df
    df = df.with_columns(
        pl.when(pl.col("ts_ms") > D1 + 6 * 60_000)
        .then(0)
        .otherwise(pl.col("trades"))
        .alias("trades")
    )
    repo.write_symbol("klines_1m", "DELUSDT", df)
    e = build_lifecycle(repo, ExchangeInfoStore([])).get("DELUSDT")
    assert e is not None
    assert e.last_data_ms == D1 + 6 * 60_000
