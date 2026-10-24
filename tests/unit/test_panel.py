"""research.panel — repository'den hizalı panel yükleme ve memmap önbellek."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pairs_bt.data.normalize import KLINES_SCHEMA
from pairs_bt.data.repository import DataRepository
from pairs_bt.research.panel import load_panel, open_panel, save_panel

M5 = 300_000
T0 = 1_756_684_800_000


def _bars(ts: np.ndarray, close: np.ndarray, trades: int = 1) -> pl.DataFrame:
    n = ts.size
    return pl.DataFrame(
        {
            "ts_ms": ts,
            "open_time_ms": ts - M5,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(n),
            "quote_volume": np.full(n, 100.0 * trades),
            "trades": np.full(n, trades, dtype=np.int64),
            "taker_buy_volume": np.zeros(n),
            "taker_buy_quote_volume": np.zeros(n),
        },
        schema=KLINES_SCHEMA,
    )


@pytest.fixture
def repo(tmp_path: Path) -> DataRepository:
    r = DataRepository(tmp_path / "parquet")
    ts = T0 + M5 * np.arange(1, 101, dtype=np.int64)
    r.write_symbol("klines_5m", "AAAUSDT", _bars(ts, np.linspace(1, 2, 100)))
    keep = np.ones(100, dtype=bool)
    keep[40:50] = False  # 10 bar boşluk
    r.write_symbol("klines_5m", "BBBUSDT", _bars(ts[keep], np.linspace(5, 6, 100)[keep]))
    return r


def test_load_panel_aligns_and_marks_stale(repo: DataRepository) -> None:
    p = load_panel(repo, "klines_5m", ["BBBUSDT", "AAAUSDT"], T0 + M5, T0 + 101 * M5, M5, 3)
    assert p.symbols == ("AAAUSDT", "BBBUSDT")
    assert p.ts_ms.size == 100
    assert not p.stale[:, 0].any()
    b_stale = np.flatnonzero(p.stale[:, 1])
    assert b_stale.tolist() == list(range(43, 50))  # 3 bar ffill, sonra stale
    assert p.quote_volume[40, 1] == 0.0


def test_load_panel_missing_symbol_all_stale(repo: DataRepository) -> None:
    p = load_panel(repo, "klines_5m", ["AAAUSDT", "ZZZUSDT"], T0 + M5, T0 + 101 * M5, M5, 3)
    assert p.stale[:, 1].all()


def test_save_and_open_roundtrip(repo: DataRepository, tmp_path: Path) -> None:
    p = load_panel(repo, "klines_5m", ["AAAUSDT", "BBBUSDT"], T0 + M5, T0 + 101 * M5, M5, 3)
    d = save_panel(p, tmp_path / "cache")
    q = open_panel(d)
    assert q.symbols == p.symbols
    for name in ("ts_ms", "close", "quote_volume", "observed", "stale"):
        np.testing.assert_array_equal(getattr(q, name), getattr(p, name))
    assert isinstance(q.close, np.memmap)
