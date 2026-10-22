from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from pairs_bt.data.normalize import (
    AGG_TRADES_SCHEMA,
    FUNDING_SCHEMA,
    KLINES_SCHEMA,
    normalize_agg_trades,
    normalize_file,
    normalize_funding,
    normalize_klines,
)
from tests.fixtures.binance_archive import (
    KlineRow,
    agg_trades_csv,
    day_ms,
    funding_csv,
    klines_csv,
    make_klines,
    zip_bytes,
)

T0 = day_ms(date(2025, 9, 1))


@pytest.fixture
def rows(
    rng: np.random.Generator,
) -> list[KlineRow]:
    return make_klines(rng, T0, 5)


@pytest.mark.parametrize("header", [True, False])
def test_klines_header_detection(rows: list[KlineRow], header: bool) -> None:
    res = normalize_klines(zip_bytes("a.csv", klines_csv(rows, header=header)), interval_ms=60_000)
    assert res.df.schema == KLINES_SCHEMA
    assert res.df.height == 5
    assert res.df["open_time_ms"].to_list() == [T0 + i * 60_000 for i in range(5)]


def test_klines_ts_is_bar_close(rows: list[KlineRow]) -> None:
    """ts_ms = open_time + interval (barın tamamen kapandığı an) — karar en erken bu anda."""
    res = normalize_klines(zip_bytes("a.csv", klines_csv(rows)), interval_ms=60_000)
    assert (res.df["ts_ms"] - res.df["open_time_ms"]).to_list() == [60_000] * 5


def test_klines_values_preserved(rows: list[KlineRow]) -> None:
    res = normalize_klines(zip_bytes("a.csv", klines_csv(rows)), interval_ms=60_000)
    first = res.df.row(0, named=True)
    assert first["open"] == pytest.approx(rows[0][1])
    assert first["high"] == pytest.approx(rows[0][2])
    assert first["trades"] == rows[0][7]


def test_klines_microseconds_converted(rows: list[KlineRow]) -> None:
    res = normalize_klines(
        zip_bytes("a.csv", klines_csv(rows, ts_unit_us=True)), interval_ms=60_000
    )
    assert res.df["open_time_ms"][0] == T0
    assert any("mikrosaniye" in c for c in res.corrections)


def test_klines_exact_duplicates_dropped_and_logged(rows: list[KlineRow]) -> None:
    res = normalize_klines(zip_bytes("a.csv", klines_csv(rows + rows[:2])), interval_ms=60_000)
    assert res.df.height == 5
    assert res.rows_in == 7
    assert any("tekrar eden" in c for c in res.corrections)


def test_klines_unsorted_input_sorted(rows: list[KlineRow]) -> None:
    res = normalize_klines(zip_bytes("a.csv", klines_csv(rows[::-1])), interval_ms=60_000)
    assert res.df["ts_ms"].is_sorted()


def test_klines_close_time_mismatch_logged(rows: list[KlineRow]) -> None:
    csv = klines_csv(rows).replace(str(T0 + 59_999).encode(), str(T0 + 30_000).encode())
    res = normalize_klines(zip_bytes("a.csv", csv), interval_ms=60_000)
    assert any("close_time" in c for c in res.corrections)


def test_empty_klines_file() -> None:
    from tests.fixtures.binance_archive import KLINE_HEADER

    res = normalize_klines(zip_bytes("a.csv", (KLINE_HEADER + "\n").encode()), interval_ms=60_000)
    assert res.df.height == 0
    assert res.df.schema == KLINES_SCHEMA


def test_funding() -> None:
    data = funding_csv([(T0, 8, 0.0000466), (T0 + 8 * 3_600_000, 8, -0.0001)])
    res = normalize_funding(zip_bytes("f.csv", data))
    assert res.df.schema == FUNDING_SCHEMA
    assert res.df["ts_ms"].to_list() == [T0, T0 + 8 * 3_600_000]
    assert res.df["rate"].to_list() == pytest.approx([0.0000466, -0.0001])
    assert res.df["interval_hours"].to_list() == [8, 8]


def test_funding_headerless() -> None:
    res = normalize_funding(zip_bytes("f.csv", funding_csv([(T0, 4, 0.0001)], header=False)))
    assert res.df.height == 1


def test_agg_trades() -> None:
    data = agg_trades_csv([(T0 + 5, 100.1, 0.5, 1, True), (T0 + 7, 100.2, 1.5, 2, False)])
    res = normalize_agg_trades(zip_bytes("t.csv", data))
    assert res.df.schema == AGG_TRADES_SCHEMA
    assert res.df["ts_ms"].to_list() == [T0 + 5, T0 + 7]
    assert res.df["is_buyer_maker"].to_list() == [True, False]


def test_agg_trades_same_ms_keeps_id_order() -> None:
    data = agg_trades_csv([(T0, 1.0, 1.0, 2, True), (T0, 2.0, 1.0, 1, True)])
    res = normalize_agg_trades(zip_bytes("t.csv", data))
    assert res.df["agg_trade_id"].to_list() == [1, 2]


def test_normalize_file_dispatch(rows: list[KlineRow]) -> None:
    assert normalize_file("klines_5m", zip_bytes("a.csv", klines_csv(rows, 300_000))).df.height == 5
    assert (
        normalize_file("fundingRate", zip_bytes("a.csv", funding_csv([(T0, 8, 0.0)]))).df.height
        == 1
    )


def test_zip_must_have_single_csv() -> None:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", "x")
        zf.writestr("b.csv", "y")
    with pytest.raises(ValueError, match="tek"):
        normalize_funding(buf.getvalue())


def test_result_frame_is_polars(rows: list[KlineRow]) -> None:
    assert isinstance(
        normalize_klines(zip_bytes("a.csv", klines_csv(rows)), 60_000).df, pl.DataFrame
    )
