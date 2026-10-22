"""Ham arşiv zip -> sabit şemalı polars DataFrame (CLAUDE.md §7 normalize adımı).

Zaman konvansiyonu (CLAUDE.md invariant #1, #8):
- klines: `ts_ms = open_time_ms + interval_ms` — barın tamamen kapandığı an. Bar'a dayalı karar
  en erken bu anda verilebilir. (Binance `close_time = open_time + interval - 1` saklanmaz;
  tutarsızlığı düzeltme kaydına yazılır.)
- fundingRate: `ts_ms = calc_time`.
- aggTrades: `ts_ms = transact_time`.
Tüm ts'ler UTC epoch-ms int64'tür; mikro-saniye gelirse ms'e çevrilir ve loglanır.

Düzeltmeler (birebir tekrar eden satırların atılması, sıralama, birim dönüşümü) raw veriyi
değiştirmez; `NormalizationResult.corrections` içinde kaydedilir.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import polars as pl
import structlog

from pairs_bt.config import DatasetName

log = structlog.get_logger(__name__)

KLINES_SCHEMA = pl.Schema(
    {
        "ts_ms": pl.Int64,
        "open_time_ms": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "quote_volume": pl.Float64,
        "trades": pl.Int64,
        "taker_buy_volume": pl.Float64,
        "taker_buy_quote_volume": pl.Float64,
    }
)
FUNDING_SCHEMA = pl.Schema({"ts_ms": pl.Int64, "interval_hours": pl.Int64, "rate": pl.Float64})
AGG_TRADES_SCHEMA = pl.Schema(
    {
        "ts_ms": pl.Int64,
        "agg_trade_id": pl.Int64,
        "price": pl.Float64,
        "qty": pl.Float64,
        "first_trade_id": pl.Int64,
        "last_trade_id": pl.Int64,
        "is_buyer_maker": pl.Boolean,
    }
)

_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
_FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
_AGG_COLS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]
# epoch-ms ~1.7e12; epoch-µs ~1.7e15. Eşik: yıl ~5138'e denk gelen ms değeri.
_US_THRESHOLD = 100_000_000_000_000


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    df: pl.DataFrame
    rows_in: int
    corrections: tuple[str, ...]


def _read_single_csv(zip_data: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"zip tek bir CSV içermeli, bulunan: {names}")
        return zf.read(names[0])


def _read_csv(raw: bytes, cols: list[str]) -> pl.DataFrame:
    first = raw.lstrip()[:1]
    has_header = bool(first) and not first.isdigit() and first != b"-"
    if not raw.strip():
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in cols})
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=has_header,
        new_columns=cols,
        infer_schema=False,  # her şey Utf8 okunur; tipler aşağıda açıkça verilir
    )
    if df.width != len(cols):
        raise ValueError(f"beklenen {len(cols)} kolon, bulunan {df.width}")
    return df


def _to_ms(df: pl.DataFrame, cols: list[str], corrections: list[str]) -> pl.DataFrame:
    if df.height == 0:
        return df
    if int(df[cols[0]].max()) >= _US_THRESHOLD:  # type: ignore[arg-type]
        corrections.append(f"zaman damgası mikrosaniye -> ms çevrildi ({', '.join(cols)})")
        df = df.with_columns([pl.col(c) // 1000 for c in cols])
    return df


def _dedupe_sort(df: pl.DataFrame, sort_cols: list[str], corrections: list[str]) -> pl.DataFrame:
    n = df.height
    df = df.unique(maintain_order=True)
    if df.height != n:
        corrections.append(f"birebir tekrar eden {n - df.height} satır atıldı")
    out = df.sort(sort_cols, maintain_order=True)
    if not out.equals(df):
        corrections.append("satırlar zamana göre yeniden sıralandı")
    return out


def normalize_klines(zip_data: bytes, interval_ms: int) -> NormalizationResult:
    raw = _read_csv(_read_single_csv(zip_data), _KLINE_COLS)
    corrections: list[str] = []
    df = raw.select(
        pl.col("open_time").cast(pl.Int64),
        pl.col("close_time").cast(pl.Int64),
        *(pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume")),
        pl.col("quote_volume").cast(pl.Float64),
        pl.col("count").cast(pl.Int64).alias("trades"),
        pl.col("taker_buy_volume").cast(pl.Float64),
        pl.col("taker_buy_quote_volume").cast(pl.Float64),
    )
    df = _to_ms(df, ["open_time", "close_time"], corrections)
    if df.height:
        bad_close = df.filter(pl.col("close_time") != pl.col("open_time") + interval_ms - 1).height
        if bad_close:
            corrections.append(f"{bad_close} satırda close_time != open_time + interval - 1")
    df = df.drop("close_time").rename({"open_time": "open_time_ms"})
    df = df.with_columns((pl.col("open_time_ms") + interval_ms).alias("ts_ms"))
    df = _dedupe_sort(df.select(KLINES_SCHEMA.names()), ["ts_ms"], corrections)
    return _result(df.cast(KLINES_SCHEMA), raw.height, corrections)


def normalize_funding(zip_data: bytes) -> NormalizationResult:
    raw = _read_csv(_read_single_csv(zip_data), _FUNDING_COLS)
    corrections: list[str] = []
    df = raw.select(
        pl.col("calc_time").cast(pl.Int64).alias("ts_ms"),
        pl.col("funding_interval_hours").cast(pl.Int64).alias("interval_hours"),
        pl.col("last_funding_rate").cast(pl.Float64).alias("rate"),
    )
    df = _to_ms(df, ["ts_ms"], corrections)
    df = _dedupe_sort(df, ["ts_ms"], corrections)
    return _result(df.cast(FUNDING_SCHEMA), raw.height, corrections)


def normalize_agg_trades(zip_data: bytes) -> NormalizationResult:
    raw = _read_csv(_read_single_csv(zip_data), _AGG_COLS)
    corrections: list[str] = []
    df = raw.select(
        pl.col("transact_time").cast(pl.Int64).alias("ts_ms"),
        pl.col("agg_trade_id").cast(pl.Int64),
        pl.col("price").cast(pl.Float64),
        pl.col("quantity").cast(pl.Float64).alias("qty"),
        pl.col("first_trade_id").cast(pl.Int64),
        pl.col("last_trade_id").cast(pl.Int64),
        (pl.col("is_buyer_maker").str.to_lowercase() == "true").alias("is_buyer_maker"),
    )
    df = _to_ms(df, ["ts_ms"], corrections)
    df = _dedupe_sort(df, ["ts_ms", "agg_trade_id"], corrections)
    return _result(df.cast(AGG_TRADES_SCHEMA), raw.height, corrections)


def normalize_file(dataset: DatasetName, zip_data: bytes) -> NormalizationResult:
    if dataset == "klines_1m":
        return normalize_klines(zip_data, 60_000)
    if dataset == "klines_5m":
        return normalize_klines(zip_data, 300_000)
    if dataset == "fundingRate":
        return normalize_funding(zip_data)
    return normalize_agg_trades(zip_data)


def _result(df: pl.DataFrame, rows_in: int, corrections: list[str]) -> NormalizationResult:
    for c in corrections:
        log.info("normalize_correction", detail=c)
    return NormalizationResult(df, rows_in, tuple(corrections))
