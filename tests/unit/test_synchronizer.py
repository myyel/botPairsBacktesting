from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pairs_bt.data.normalize import KLINES_SCHEMA
from pairs_bt.data.synchronizer import TimeSynchronizer

M = 60_000
T0 = 1_756_684_800_000  # 2025-09-01 00:00 UTC


def _bars(ts: list[int], close: list[float]) -> pl.DataFrame:
    n = len(ts)
    return pl.DataFrame(
        {
            "ts_ms": ts,
            "open_time_ms": [t - M for t in ts],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1.0] * n,
            "quote_volume": [10.0] * n,
            "trades": [1] * n,
            "taker_buy_volume": [0.5] * n,
            "taker_buy_quote_volume": [5.0] * n,
        },
        schema=KLINES_SCHEMA,
    )


def test_aligns_symbols_sorted_on_common_grid() -> None:
    sync = TimeSynchronizer(M, max_ffill_bars=3)
    frames = {
        "ZUSDT": _bars([T0 + M, T0 + 2 * M], [1.0, 2.0]),
        "AUSDT": _bars([T0 + M, T0 + 2 * M], [10.0, 20.0]),
    }
    p = sync.align(frames, T0 + M, T0 + 3 * M)
    assert p.symbols == ("AUSDT", "ZUSDT")
    assert p.ts_ms.tolist() == [T0 + M, T0 + 2 * M]
    np.testing.assert_array_equal(p.close, [[10.0, 1.0], [20.0, 2.0]])
    assert not p.stale.any()
    assert p.observed.all()


def test_ffill_limited_then_stale() -> None:
    sync = TimeSynchronizer(M, max_ffill_bars=2)
    frames = {"AUSDT": _bars([T0 + M], [5.0])}
    p = sync.align(frames, T0 + M, T0 + 5 * M)  # grid: 1..4
    col = p.close[:, 0]
    assert col[:3].tolist() == [5.0, 5.0, 5.0]  # 1 gözlem + 2 ffill
    assert np.isnan(col[3])
    assert p.stale[:, 0].tolist() == [False, False, False, True]
    assert p.observed[:, 0].tolist() == [True, False, False, False]
    # ffill edilen barın hacmi 0 (işlem olmadı)
    assert p.quote_volume[:, 0].tolist() == [10.0, 0.0, 0.0, 0.0]


def test_no_backfill_before_first_bar() -> None:
    sync = TimeSynchronizer(M, max_ffill_bars=5)
    p = sync.align({"AUSDT": _bars([T0 + 3 * M], [7.0])}, T0 + M, T0 + 4 * M)
    assert np.isnan(p.close[0, 0])
    assert np.isnan(p.close[1, 0])
    assert p.close[2, 0] == 7.0
    assert p.stale[:, 0].tolist() == [True, True, False]


def test_zero_ffill_marks_every_missing_bar_stale() -> None:
    sync = TimeSynchronizer(M, max_ffill_bars=0)
    p = sync.align({"AUSDT": _bars([T0 + M, T0 + 3 * M], [1.0, 3.0])}, T0 + M, T0 + 4 * M)
    assert p.stale[:, 0].tolist() == [False, True, False]


def test_empty_frame_and_empty_range() -> None:
    sync = TimeSynchronizer(M, max_ffill_bars=3)
    p = sync.align({"AUSDT": pl.DataFrame(schema=KLINES_SCHEMA)}, T0 + M, T0 + 3 * M)
    assert p.stale.all()
    empty = sync.align({"AUSDT": _bars([T0 + M], [1.0])}, T0 + M, T0 + M)
    assert empty.ts_ms.size == 0
    assert empty.close.shape == (0, 1)


def test_unaligned_start_rounded_up() -> None:
    sync = TimeSynchronizer(M, max_ffill_bars=3)
    p = sync.align({"AUSDT": _bars([T0 + M, T0 + 2 * M], [1.0, 2.0])}, T0 + M + 1, T0 + 3 * M)
    assert p.ts_ms.tolist() == [T0 + 2 * M]


def test_misaligned_bars_rejected() -> None:
    sync = TimeSynchronizer(M, max_ffill_bars=3)
    with pytest.raises(ValueError, match="hizalı"):
        sync.align({"AUSDT": _bars([T0 + M + 5], [1.0])}, T0, T0 + 3 * M)


def test_invalid_params() -> None:
    with pytest.raises(ValueError, match="max_ffill_bars"):
        TimeSynchronizer(M, max_ffill_bars=-1)
    with pytest.raises(ValueError, match="interval_ms"):
        TimeSynchronizer(0, max_ffill_bars=1)
