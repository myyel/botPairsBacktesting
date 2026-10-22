from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from pairs_bt.config import ValidationConfig
from pairs_bt.data.normalize import FUNDING_SCHEMA, KLINES_SCHEMA, normalize_klines
from pairs_bt.data.validator import DataValidator, ValidationReport
from tests.fixtures.binance_archive import (
    KlineRow,
    day_ms,
    klines_csv,
    make_klines,
    resample_rows,
    zip_bytes,
)

T0 = day_ms(date(2025, 9, 1))
H = 3_600_000


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator(ValidationConfig(max_examples=3))


@pytest.fixture
def rows_1m(rng: np.random.Generator) -> list[KlineRow]:
    return make_klines(rng, T0, 60)


def _k(rows: list[KlineRow], interval_ms: int = 60_000) -> pl.DataFrame:
    return normalize_klines(zip_bytes("a.csv", klines_csv(rows, interval_ms)), interval_ms).df


def _checks(v: object) -> set[str]:
    return {i.check for i in v.issues}  # type: ignore[attr-defined]


def test_clean_klines_pass(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    v = validator.validate_klines(_k(rows_1m), "BTCUSDT", "klines_1m", 60_000)
    assert v.issues == ()
    assert v.rows == 60
    assert v.coverage == pytest.approx(1.0)
    assert v.ok


def test_gap_detected_as_warning(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    df = _k(rows_1m[:10] + rows_1m[15:])
    v = validator.validate_klines(df, "BTCUSDT", "klines_1m", 60_000)
    gap = next(i for i in v.issues if i.check == "gap")
    assert gap.severity == "warning"
    assert gap.count == 5  # eksik bar sayısı
    assert v.coverage == pytest.approx(55 / 60)
    assert v.ok  # gap tek başına hata değil


def test_duplicate_and_order_errors(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    df = _k(rows_1m)
    dup = pl.concat([df, df.slice(3, 1).with_columns(pl.col("close") + 1)])
    v = validator.validate_klines(dup, "BTCUSDT", "klines_1m", 60_000)
    assert {"duplicate_ts", "unsorted"} <= _checks(v)
    assert not v.ok


def test_ohlc_inconsistency(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    df = (
        _k(rows_1m)
        .with_row_index()
        .with_columns(
            pl.when(pl.col("index") == 7)
            .then(pl.col("low") - 1)
            .otherwise(pl.col("high"))
            .alias("high")
        )
        .drop("index")
    )
    v = validator.validate_klines(df, "BTCUSDT", "klines_1m", 60_000)
    issue = next(i for i in v.issues if i.check == "ohlc")
    assert issue.count == 1
    assert issue.severity == "error"


def test_nonpositive_price_and_negative_volume(
    validator: DataValidator, rows_1m: list[KlineRow]
) -> None:
    df = _k(rows_1m).with_row_index()
    df = df.with_columns(
        pl.when(pl.col("index") == 1).then(0.0).otherwise(pl.col("low")).alias("low"),
        pl.when(pl.col("index") == 2).then(-1.0).otherwise(pl.col("volume")).alias("volume"),
    ).drop("index")
    assert {"nonpositive_price", "negative_volume"} <= _checks(
        validator.validate_klines(df, "X", "klines_1m", 60_000)
    )


def test_nan_detected(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    df = (
        _k(rows_1m)
        .with_row_index()
        .with_columns(
            pl.when(pl.col("index") == 4)
            .then(float("nan"))
            .otherwise(pl.col("close"))
            .alias("close")
        )
        .drop("index")
    )
    assert "nan" in _checks(validator.validate_klines(df, "X", "klines_1m", 60_000))


def test_misaligned_ts(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    df = _k(rows_1m).with_columns(pl.col("ts_ms") + 7)
    assert "misaligned_ts" in _checks(validator.validate_klines(df, "X", "klines_1m", 60_000))


def test_empty_and_single_bar(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    empty = validator.validate_klines(pl.DataFrame(schema=KLINES_SCHEMA), "X", "klines_1m", 60_000)
    assert "empty" in _checks(empty)
    assert not empty.ok
    single = validator.validate_klines(_k(rows_1m[:1]), "X", "klines_1m", 60_000)
    assert single.ok
    assert single.coverage == pytest.approx(1.0)


def test_crosscheck_5m_matches(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    df5 = _k(resample_rows(rows_1m, 5), 300_000)
    v = validator.crosscheck_5m(_k(rows_1m), df5, "BTCUSDT")
    assert v.issues == ()
    assert v.rows == 12


def test_crosscheck_5m_detects_mismatch(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    rs = resample_rows(rows_1m, 5)
    bad = list(rs[2])
    bad[2] = bad[2] + 1.0  # high
    rs[2] = tuple(bad)  # type: ignore[call-overload]
    v = validator.crosscheck_5m(_k(rows_1m), _k(rs, 300_000), "BTCUSDT")
    issue = next(i for i in v.issues if i.check == "crosscheck_mismatch")
    assert issue.count == 1
    assert issue.severity == "error"


def test_crosscheck_incomplete_bucket_skipped(
    validator: DataValidator, rows_1m: list[KlineRow]
) -> None:
    df1 = _k(rows_1m[:7] + rows_1m[8:])  # 2. 5m kovası eksik
    v = validator.crosscheck_5m(df1, _k(resample_rows(rows_1m, 5), 300_000), "BTCUSDT")
    assert "crosscheck_mismatch" not in _checks(v)
    skipped = next(i for i in v.issues if i.check == "crosscheck_incomplete")
    assert skipped.count == 1


def _funding(times: list[int], hours: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"ts_ms": times, "interval_hours": hours, "rate": [0.0001] * len(times)},
        schema=FUNDING_SCHEMA,
    )


def test_funding_clean(validator: DataValidator) -> None:
    v = validator.validate_funding(_funding([T0, T0 + 8 * H, T0 + 16 * H], [8, 8, 8]), "BTCUSDT")
    assert v.ok
    assert v.issues == ()


def test_funding_interval_change_reported(validator: DataValidator) -> None:
    times = [T0, T0 + 8 * H, T0 + 12 * H, T0 + 16 * H]
    v = validator.validate_funding(_funding(times, [8, 4, 4, 4]), "BTCUSDT")
    change = next(i for i in v.issues if i.check == "funding_interval_change")
    assert change.severity == "warning"
    assert v.ok


def test_funding_spacing_mismatch(validator: DataValidator) -> None:
    v = validator.validate_funding(_funding([T0, T0 + 8 * H, T0 + 24 * H], [8, 8, 8]), "BTCUSDT")
    assert "funding_spacing" in _checks(v)


def test_tick_consistency(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    df = _k(rows_1m)
    assert validator.check_tick(df, "X", Decimal("0.01")) is None
    issue = validator.check_tick(df, "X", Decimal("1"))
    assert issue is not None
    assert issue.check == "tick_mismatch"


def test_report_summary_is_turkish(validator: DataValidator, rows_1m: list[KlineRow]) -> None:
    ok = validator.validate_klines(_k(rows_1m), "AUSDT", "klines_1m", 60_000)
    bad = validator.validate_klines(
        pl.DataFrame(schema=KLINES_SCHEMA), "BUSDT", "klines_1m", 60_000
    )
    report = ValidationReport((bad, ok))
    assert [r.symbol for r in report.results] == ["AUSDT", "BUSDT"]  # sıralı
    assert report.error_count == 1
    text = report.summary_markdown()
    assert "Doğrulama Raporu" in text
    assert "BUSDT" in text
    assert report.to_dict()["error_count"] == 1


def test_crosscheck_ignores_synthetic_zero_trade_bars(validator: DataValidator) -> None:
    """Gerçek veride (BSWUSDT 2025-09-01 07:30): işlemsiz 1m bar'lar önceki close'u taşır,
    5m open'ı ise ilk gerçek trade'dir."""
    t = T0
    rows_1m = [
        (t, 0.01722, 0.01722, 0.01722, 0.01722, 0.0, 0.0, 0, 0.0, 0.0),
        (t + 60_000, 0.01722, 0.01722, 0.01722, 0.01722, 0.0, 0.0, 0, 0.0, 0.0),
        (t + 120_000, 0.0172, 0.01722, 0.0172, 0.01722, 63093.0, 1.0, 5, 0.0, 0.0),
        (t + 180_000, 0.01722, 0.01722, 0.01721, 0.01722, 190772.0, 1.0, 5, 0.0, 0.0),
        (t + 240_000, 0.01723, 0.01725, 0.01723, 0.01724, 230199.0, 1.0, 5, 0.0, 0.0),
    ]
    row_5m = [(t, 0.0172, 0.01725, 0.0172, 0.01724, 484064.0, 3.0, 15, 0.0, 0.0)]
    df1 = normalize_klines(zip_bytes("a.csv", klines_csv(rows_1m, tick=0.00001)), 60_000).df
    df5 = normalize_klines(
        zip_bytes("a.csv", klines_csv(row_5m, 300_000, tick=0.00001)), 300_000
    ).df
    assert validator.crosscheck_5m(df1, df5, "BSWUSDT").issues == ()
    v = validator.validate_klines(df1, "BSWUSDT", "klines_1m", 60_000)
    zero = next(i for i in v.issues if i.check == "zero_trade_bars")
    assert zero.count == 2
    assert "en uzun seri 2" in zero.message


def _trades(ids: list[int], prices: list[float]) -> pl.DataFrame:
    from pairs_bt.data.normalize import AGG_TRADES_SCHEMA

    n = len(ids)
    return pl.DataFrame(
        {
            "ts_ms": [T0 + i for i in range(n)],
            "agg_trade_id": ids,
            "price": prices,
            "qty": [1.0] * n,
            "first_trade_id": ids,
            "last_trade_id": ids,
            "is_buyer_maker": [True] * n,
        },
        schema=AGG_TRADES_SCHEMA,
    )


def test_agg_trades_clean_and_defects(validator: DataValidator) -> None:
    assert validator.validate_agg_trades(_trades([1, 2, 3], [1.0, 1.1, 1.2]), "X").issues == ()
    v = validator.validate_agg_trades(_trades([1, 1, 5], [1.0, 0.0, 1.2]), "X")
    assert {"duplicate_id", "nonpositive_trade", "id_gap"} <= _checks(v)
    assert not v.ok
