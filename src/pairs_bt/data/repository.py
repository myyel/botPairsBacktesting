"""DataRepository — normalize edilmiş veri için hive-partition parquet deposu (CLAUDE.md §7).

Yerleşim: `<root>/dataset=<ad>/symbol=<SEMBOL>/month=<YYYY-MM>/part.parquet`.
Partition anahtarı `ts_ms`'in UTC ayıdır (sembol/ay; gün bazlı okuma `read_range` ile ts filtresi).
Bir sembol her zaman bütün olarak yazılır: `write_symbol` sembolün eski partition'larını değiştirir.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from pairs_bt.config import DatasetName
from pairs_bt.data.normalize import AGG_TRADES_SCHEMA, FUNDING_SCHEMA, KLINES_SCHEMA

SCHEMAS: dict[DatasetName, pl.Schema] = {
    "klines_1m": KLINES_SCHEMA,
    "klines_5m": KLINES_SCHEMA,
    "fundingRate": FUNDING_SCHEMA,
    "aggTrades": AGG_TRADES_SCHEMA,
}
_PART = "part.parquet"


def month_of_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m")


def _month_bounds(month: str) -> tuple[int, int]:
    y, m = map(int, month.split("-"))
    start = datetime(y, m, 1, tzinfo=UTC)
    end = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=UTC)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


class DataRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def symbol_dir(self, dataset: DatasetName, symbol: str) -> Path:
        return self.root / f"dataset={dataset}" / f"symbol={symbol}"

    def write_symbol(self, dataset: DatasetName, symbol: str, df: pl.DataFrame) -> list[Path]:
        """Sembolün tüm verisini yazar (önceki partition'lar silinir).

        df ts_ms'e göre kesin artan sıralı olmalı.
        """
        schema = SCHEMAS[dataset]
        if df.schema != schema:
            raise ValueError(f"{dataset}: şema uyuşmuyor: {df.schema} != {schema}")
        sort_cols = ["ts_ms", "agg_trade_id"] if dataset == "aggTrades" else ["ts_ms"]
        ts = df.select(sort_cols)
        if df.height and (ts.is_duplicated().any() or not ts.equals(ts.sort(sort_cols))):
            raise ValueError(f"{dataset}/{symbol}: ts_ms kesin artan sıralı (tekrarsız) olmalı")
        target = self.symbol_dir(dataset, symbol)
        tmp = target.with_name(target.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        written: list[Path] = []
        months = df.with_columns(
            pl.from_epoch("ts_ms", time_unit="ms").dt.strftime("%Y-%m").alias("_month")
        )
        for (month,), part in months.group_by(["_month"], maintain_order=True):
            out = tmp / f"month={month}" / _PART
            out.parent.mkdir(parents=True, exist_ok=True)
            part.drop("_month").write_parquet(out, compression="zstd", statistics=True)
            written.append(target / f"month={month}" / _PART)
        if target.exists():
            shutil.rmtree(target)
        if written:
            tmp.rename(target)
        return sorted(written)

    def read_range(
        self, dataset: DatasetName, symbol: str, start_ms: int, end_ms: int
    ) -> pl.DataFrame:
        """`[start_ms, end_ms)` aralığındaki satırlar, ts sıralı."""
        schema = SCHEMAS[dataset]
        files = [
            p
            for p in self._symbol_files(dataset, symbol)
            if _overlaps(p.parent.name.removeprefix("month="), start_ms, end_ms)
        ]
        if not files:
            return pl.DataFrame(schema=schema)
        lf = pl.scan_parquet(files).filter(
            (pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms)
        )
        return lf.collect().cast(schema)

    def read_symbol(self, dataset: DatasetName, symbol: str) -> pl.DataFrame:
        files = self._symbol_files(dataset, symbol)
        if not files:
            return pl.DataFrame(schema=SCHEMAS[dataset])
        return pl.read_parquet(files).cast(SCHEMAS[dataset])

    def symbols(self, dataset: DatasetName) -> tuple[str, ...]:
        base = self.root / f"dataset={dataset}"
        if not base.exists():
            return ()
        return tuple(
            sorted(
                p.name.removeprefix("symbol=")
                for p in base.iterdir()
                if p.is_dir() and p.name.startswith("symbol=") and not p.name.endswith(".tmp")
            )
        )

    def partition_files(self, dataset: DatasetName) -> list[Path]:
        return [f for s in self.symbols(dataset) for f in self._symbol_files(dataset, s)]

    def partition_files_for(self, dataset: DatasetName, symbol: str) -> list[Path]:
        return self._symbol_files(dataset, symbol)

    def _symbol_files(self, dataset: DatasetName, symbol: str) -> list[Path]:
        d = self.symbol_dir(dataset, symbol)
        return sorted(d.glob(f"month=*/{_PART}")) if d.exists() else []


def _overlaps(month: str, start_ms: int, end_ms: int) -> bool:
    lo, hi = _month_bounds(month)
    return lo < end_ms and start_ms < hi
