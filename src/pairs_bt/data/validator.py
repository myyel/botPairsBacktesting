"""DataValidator — normalize edilmiş verinin şema/zaman/tutarlılık kontrolleri (RAPOR §4, §15.4).

Validator veriyi DEĞİŞTİRMEZ; yalnız rapor üretir. Şiddet:
- `error`: veri o haliyle backtest'te kullanılmamalı (tekrar eden ts, sırasızlık, OHLC ihlali, NaN,
  1m↔5m uyuşmazlığı ...).
- `warning`: bilinmesi gereken ama tek başına diskalifiye etmeyen durumlar (gap, funding aralığı
  değişimi, tick farkı). Gap'ler sinyal tarafında `max_ffill_bars` + stale mantığıyla ele alınır.
Rapor metinleri Türkçedir.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import numpy as np
import polars as pl

from pairs_bt.config import DatasetName, ValidationConfig

Severity = Literal["error", "warning"]
_HOUR_MS = 3_600_000


def fmt_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M")


@dataclass(frozen=True, slots=True)
class Issue:
    check: str
    severity: Severity
    count: int
    message: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SymbolValidation:
    dataset: str
    symbol: str
    rows: int
    first_ts_ms: int | None
    last_ts_ms: int | None
    expected_rows: int | None
    issues: tuple[Issue, ...]

    @property
    def ok(self) -> bool:
        return all(i.severity != "error" for i in self.issues)

    @property
    def coverage(self) -> float | None:
        if not self.expected_rows:
            return None
        return self.rows / self.expected_rows


class DataValidator:
    def __init__(self, cfg: ValidationConfig) -> None:
        self._cfg = cfg

    def _ex(self, ts: pl.Series) -> tuple[str, ...]:
        return tuple(fmt_ts(int(t)) for t in ts.head(self._cfg.max_examples).to_list())

    # ------------------------------------------------------------ klines
    def validate_klines(
        self, df: pl.DataFrame, symbol: str, dataset: DatasetName, interval_ms: int
    ) -> SymbolValidation:
        if df.height == 0:
            empty = Issue("empty", "error", 0, "veri yok")
            return SymbolValidation(dataset, symbol, 0, None, None, None, (empty,))
        issues: list[Issue] = []
        ts = df["ts_ms"]
        dup = df.filter(ts.is_duplicated())
        if dup.height:
            n = dup["ts_ms"].n_unique()
            issues.append(
                Issue(
                    "duplicate_ts",
                    "error",
                    n,
                    f"{n} zaman damgası birden fazla kez var",
                    self._ex(dup["ts_ms"]),
                )
            )
        diffs = ts.diff().drop_nulls()
        if (diffs < 0).any():
            n = int((diffs < 0).sum())
            issues.append(Issue("unsorted", "error", n, f"{n} yerde zaman geriye gidiyor"))
        mis = df.filter(pl.col("ts_ms") % interval_ms != 0)
        if mis.height:
            issues.append(
                Issue(
                    "misaligned_ts",
                    "error",
                    mis.height,
                    f"{mis.height} bar {interval_ms // 1000}s grid'ine hizalı değil",
                    self._ex(mis["ts_ms"]),
                )
            )
        uniq = ts.unique().sort()
        gaps = pl.DataFrame({"ts": uniq.slice(1), "d": uniq.diff().slice(1)}).filter(
            pl.col("d") > interval_ms
        )
        if gaps.height:
            missing = int(((gaps["d"] // interval_ms) - 1).sum())
            longest = int(gaps["d"].max()) // interval_ms - 1  # type: ignore[arg-type]
            issues.append(
                Issue(
                    "gap",
                    "warning",
                    missing,
                    f"{gaps.height} boşlukta toplam {missing} eksik bar (en uzun {longest} bar)",
                    self._ex(gaps["ts"] - gaps["d"] + interval_ms),
                )
            )
        price_cols = ["open", "high", "low", "close"]
        num_cols = [*price_cols, "volume", "quote_volume", "taker_buy_volume"]
        nan = df.filter(
            pl.any_horizontal([pl.col(c).is_nan() | pl.col(c).is_null() for c in num_cols])
        )
        if nan.height:
            issues.append(
                Issue(
                    "nan",
                    "error",
                    nan.height,
                    f"{nan.height} barda NaN/boş değer",
                    self._ex(nan["ts_ms"]),
                )
            )
        clean = df.filter(~pl.any_horizontal([pl.col(c).is_nan() for c in price_cols]))
        bad_ohlc = clean.filter(
            (pl.col("high") < pl.max_horizontal("open", "close"))
            | (pl.col("low") > pl.min_horizontal("open", "close"))
            | (pl.col("low") > pl.col("high"))
        )
        if bad_ohlc.height:
            issues.append(
                Issue(
                    "ohlc",
                    "error",
                    bad_ohlc.height,
                    f"{bad_ohlc.height} barda OHLC tutarsız (high/low aralığı)",
                    self._ex(bad_ohlc["ts_ms"]),
                )
            )
        nonpos = clean.filter(pl.any_horizontal([pl.col(c) <= 0 for c in price_cols]))
        if nonpos.height:
            issues.append(
                Issue(
                    "nonpositive_price",
                    "error",
                    nonpos.height,
                    f"{nonpos.height} barda fiyat <= 0",
                    self._ex(nonpos["ts_ms"]),
                )
            )
        negvol = df.filter(
            pl.any_horizontal(
                [pl.col(c) < 0 for c in ("volume", "quote_volume", "taker_buy_volume")]
            )
        )
        if negvol.height:
            issues.append(
                Issue(
                    "negative_volume",
                    "error",
                    negvol.height,
                    f"{negvol.height} barda negatif hacim",
                    self._ex(negvol["ts_ms"]),
                )
            )
        zero = df["trades"] == 0
        n_zero = int(zero.sum())
        if n_zero:
            # en uzun ardışık işlemsiz bar serisi
            runs = zero.rle().struct.unnest().filter(pl.col("value"))["len"]
            longest_zero = int(runs.max())  # type: ignore[arg-type]
            issues.append(
                Issue(
                    "zero_trade_bars",
                    "warning",
                    n_zero,
                    f"{n_zero} işlemsiz (sentetik) bar, en uzun seri {longest_zero} bar",
                    self._ex(df.filter(zero)["ts_ms"]),
                )
            )
        taker = df.filter(pl.col("taker_buy_volume") > pl.col("volume") * (1 + 1e-9))
        if taker.height:
            issues.append(
                Issue(
                    "taker_gt_volume",
                    "warning",
                    taker.height,
                    f"{taker.height} barda taker_buy_volume > volume",
                    self._ex(taker["ts_ms"]),
                )
            )
        lo, hi = int(uniq.min()), int(uniq.max())  # type: ignore[arg-type]
        expected = (hi - lo) // interval_ms + 1
        return SymbolValidation(dataset, symbol, df.height, lo, hi, expected, tuple(issues))

    # ------------------------------------------------------------ 1m <-> 5m
    def crosscheck_5m(
        self, df_1m: pl.DataFrame, df_5m: pl.DataFrame, symbol: str
    ) -> SymbolValidation:
        """1m'den yeniden üretilen 5m barları arşivdeki 5m ile karşılaştırır.

        Yalnız tam (5 bar'lık) kovalar karşılaştırılır.
        """
        agg = (
            df_1m.with_columns(((pl.col("open_time_ms") // 300_000) * 300_000).alias("bucket"))
            .group_by("bucket")
            .agg(
                pl.len().alias("n"),
                # Binance işlemsiz 1m bar'ları sentetik üretir (OHLC = önceki close); 5m OHLC ise
                # yalnız gerçek trade'lerden gelir. Bu yüzden open/high/low işlem görmüş bar'lardan.
                pl.col("open")
                .filter(pl.col("trades") > 0)
                .first()
                .fill_null(pl.col("open").first())
                .alias("open_r"),
                pl.col("high")
                .filter(pl.col("trades") > 0)
                .max()
                .fill_null(pl.col("high").max())
                .alias("high_r"),
                pl.col("low")
                .filter(pl.col("trades") > 0)
                .min()
                .fill_null(pl.col("low").min())
                .alias("low_r"),
                pl.col("close").last().alias("close_r"),
                pl.col("volume").sum().alias("volume_r"),
            )
        )
        joined = df_5m.join(agg, left_on="open_time_ms", right_on="bucket", how="left")
        complete = joined.filter(pl.col("n") == 5)
        incomplete = joined.height - complete.height
        ptol, vtol = self._cfg.crosscheck_price_rel_tol, self._cfg.crosscheck_volume_rel_tol

        def _bad(col: str, tol: float) -> pl.Expr:
            ref = pl.col(f"{col}_r")
            return (pl.col(col) - ref).abs() > tol * pl.max_horizontal(
                ref.abs(), pl.lit(1e-12)
            ) + 1e-12

        bad = complete.filter(
            pl.any_horizontal(
                [_bad(c, ptol) for c in ("open", "high", "low", "close")] + [_bad("volume", vtol)]
            )
        )
        issues: list[Issue] = []
        if bad.height:
            issues.append(
                Issue(
                    "crosscheck_mismatch",
                    "error",
                    bad.height,
                    f"{bad.height} adet 5m bar 1m'den üretilenle uyuşmuyor",
                    self._ex(bad["ts_ms"]),
                )
            )
        if incomplete:
            issues.append(
                Issue(
                    "crosscheck_incomplete",
                    "warning",
                    incomplete,
                    f"{incomplete} adet 5m bar için 1m kovası eksik, karşılaştırılmadı",
                )
            )
        first = int(df_5m["ts_ms"].min()) if df_5m.height else None  # type: ignore[arg-type]
        last = int(df_5m["ts_ms"].max()) if df_5m.height else None  # type: ignore[arg-type]
        return SymbolValidation(
            "crosscheck_5m", symbol, complete.height, first, last, None, tuple(issues)
        )

    # ------------------------------------------------------------ funding
    def validate_funding(self, df: pl.DataFrame, symbol: str) -> SymbolValidation:
        if df.height == 0:
            return SymbolValidation(
                "fundingRate",
                symbol,
                0,
                None,
                None,
                None,
                (Issue("empty", "error", 0, "funding verisi yok"),),
            )
        issues: list[Issue] = []
        dup = df.filter(pl.col("ts_ms").is_duplicated())
        if dup.height:
            issues.append(
                Issue(
                    "duplicate_ts", "error", dup.height, f"{dup.height} tekrar eden funding zamanı"
                )
            )
        if not df["ts_ms"].is_sorted():
            issues.append(Issue("unsorted", "error", 1, "funding zamanları sıralı değil"))
        hours = df["interval_hours"]
        changes = df.filter(hours.diff().fill_null(0) != 0)
        if changes.height:
            transitions = tuple(
                f"{fmt_ts(int(t))}: {int(h)}s"
                for t, h in zip(changes["ts_ms"], changes["interval_hours"], strict=True)
            )[: self._cfg.max_examples]
            issues.append(
                Issue(
                    "funding_interval_change",
                    "warning",
                    changes.height,
                    f"funding aralığı {changes.height} kez değişti",
                    transitions,
                )
            )
        # ardışık funding zamanları arası, sonraki kaydın bildirdiği aralıkla uyumlu mu? (±1 dk)
        spacing = df.with_columns(pl.col("ts_ms").diff().alias("d")).filter(
            pl.col("d").is_not_null()
            & ((pl.col("d") - pl.col("interval_hours") * _HOUR_MS).abs() > 60_000)
        )
        if spacing.height:
            issues.append(
                Issue(
                    "funding_spacing",
                    "warning",
                    spacing.height,
                    f"{spacing.height} funding kaydı bildirilen aralıkla uyumsuz "
                    "(eksik/ekstra kayıt)",
                    self._ex(spacing["ts_ms"]),
                )
            )
        off_hour = df.filter((pl.col("ts_ms") % _HOUR_MS) > 60_000)
        if off_hour.height:
            issues.append(
                Issue(
                    "funding_off_hour",
                    "warning",
                    off_hour.height,
                    f"{off_hour.height} funding zamanı saat başına hizalı değil",
                    self._ex(off_hour["ts_ms"]),
                )
            )
        lo, hi = int(df["ts_ms"].min()), int(df["ts_ms"].max())  # type: ignore[arg-type]
        return SymbolValidation("fundingRate", symbol, df.height, lo, hi, None, tuple(issues))

    # ------------------------------------------------------------ aggTrades
    def validate_agg_trades(self, df: pl.DataFrame, symbol: str) -> SymbolValidation:
        if df.height == 0:
            empty = Issue("empty", "error", 0, "aggTrades verisi yok")
            return SymbolValidation("aggTrades", symbol, 0, None, None, None, (empty,))
        issues: list[Issue] = []
        dup = df.filter(pl.col("agg_trade_id").is_duplicated())
        if dup.height:
            issues.append(
                Issue(
                    "duplicate_id",
                    "error",
                    dup.height,
                    f"{dup.height} tekrar eden agg_trade_id",
                    self._ex(dup["ts_ms"]),
                )
            )
        if not df["ts_ms"].is_sorted():
            issues.append(Issue("unsorted", "error", 1, "trade zamanları sıralı değil"))
        bad = df.filter((pl.col("price") <= 0) | (pl.col("qty") <= 0) | pl.col("price").is_nan())
        if bad.height:
            issues.append(
                Issue(
                    "nonpositive_trade",
                    "error",
                    bad.height,
                    f"{bad.height} trade'de fiyat/miktar <= 0 veya NaN",
                    self._ex(bad["ts_ms"]),
                )
            )
        id_gaps = df["agg_trade_id"].sort().diff().drop_nulls()
        missing = int((id_gaps.filter(id_gaps > 1) - 1).sum())
        if missing:
            issues.append(
                Issue("id_gap", "warning", missing, f"agg_trade_id dizisinde {missing} eksik id")
            )
        lo, hi = int(df["ts_ms"].min()), int(df["ts_ms"].max())  # type: ignore[arg-type]
        return SymbolValidation("aggTrades", symbol, df.height, lo, hi, None, tuple(issues))

    # ------------------------------------------------------------ tick
    def check_tick(self, df: pl.DataFrame, symbol: str, tick: Decimal) -> Issue | None:
        """Fiyatların verilen tickSize grid'inde olup olmadığını kontrol eder (warning)."""
        prices = (
            df.select(pl.concat_list("open", "high", "low", "close").explode())
            .to_series()
            .to_numpy()
        )
        finite = np.isfinite(prices)
        prices = prices[finite]
        if prices.size == 0:
            return None
        t = float(tick)
        ratio = prices / t
        off = np.abs(ratio - np.rint(ratio)) > 1e-9 * np.maximum(1.0, np.abs(ratio))
        n = int(off.sum())
        if n == 0:
            return None
        ts = df.select(pl.col("ts_ms").repeat_by(4).explode()).to_series().to_numpy()
        ts = ts[finite]
        return Issue(
            "tick_mismatch",
            "warning",
            n,
            f"{symbol}: {n} fiyat tickSize={tick} grid'inde değil (tick değişmiş olabilir)",
            tuple(fmt_ts(int(t)) for t in ts[off][: self._cfg.max_examples]),
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    results: tuple[SymbolValidation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "results", tuple(sorted(self.results, key=lambda r: (r.dataset, r.symbol)))
        )

    @property
    def error_count(self) -> int:
        return sum(not r.ok for r in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "error_count": self.error_count,
            "results": [
                {
                    "dataset": r.dataset,
                    "symbol": r.symbol,
                    "rows": r.rows,
                    "ok": r.ok,
                    "first_ts_ms": r.first_ts_ms,
                    "last_ts_ms": r.last_ts_ms,
                    "expected_rows": r.expected_rows,
                    "coverage": r.coverage,
                    "issues": [
                        {
                            "check": i.check,
                            "severity": i.severity,
                            "count": i.count,
                            "message": i.message,
                            "examples": list(i.examples),
                        }
                        for i in r.issues
                    ],
                }
                for r in self.results
            ],
        }

    def summary_markdown(self) -> str:
        lines = ["# Doğrulama Raporu", ""]
        by_ds: dict[str, list[SymbolValidation]] = {}
        for r in self.results:
            by_ds.setdefault(r.dataset, []).append(r)
        lines += [
            "| Dataset | Sembol | Hatalı sembol | Satır | Medyan kapsama |",
            "|---|---|---|---|---|",
        ]
        for ds in sorted(by_ds):
            rs = by_ds[ds]
            cov = [r.coverage for r in rs if r.coverage is not None]
            med = f"%{100 * float(np.median(cov)):.2f}" if cov else "-"
            lines.append(
                f"| {ds} | {len(rs)} | {sum(not r.ok for r in rs)} "
                f"| {sum(r.rows for r in rs):,} | {med} |"
            )
        lines += ["", f"Toplam hatalı (sembol, dataset): **{self.error_count}**", ""]
        counts: dict[tuple[str, str, str], int] = {}
        for r in self.results:
            for i in r.issues:
                key = (r.dataset, i.check, i.severity)
                counts[key] = counts.get(key, 0) + 1
        if counts:
            lines += [
                "## Kontrol bazında",
                "",
                "| Dataset | Kontrol | Şiddet | Etkilenen sembol |",
                "|---|---|---|---|",
            ]
            lines += [f"| {d} | {c} | {s} | {n} |" for (d, c, s), n in sorted(counts.items())]
            lines.append("")
        errors = [r for r in self.results if not r.ok]
        if errors:
            lines += ["## Hatalar", ""]
            for r in errors:
                for i in r.issues:
                    if i.severity == "error":
                        ex = f" (örn. {', '.join(i.examples[:3])})" if i.examples else ""
                        lines.append(f"- `{r.dataset}` **{r.symbol}**: {i.message}{ex}")
        return "\n".join(lines) + "\n"
