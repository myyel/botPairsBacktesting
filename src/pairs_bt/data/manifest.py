"""Dataset manifest: dosya, SHA-256, satır sayısı, zaman aralığı (CLAUDE.md §7, §8).

`manifest_hash()` yalnız içerik alanlarından (yol, hash, satır, aralık) hesaplanır; aynı veri
her zaman aynı hash'i verir. Backtest run'ları bu hash'i artifact'a yazar.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from pairs_bt.config import DatasetName
from pairs_bt.data.downloader import sha256_file
from pairs_bt.data.repository import DataRepository


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    path: str  # repository köküne göre, POSIX
    symbol: str
    month: str
    sha256: str
    rows: int
    min_ts_ms: int | None
    max_ts_ms: int | None


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset: DatasetName
    entries: tuple[ManifestEntry, ...]

    @property
    def total_rows(self) -> int:
        return sum(e.rows for e in self.entries)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted({e.symbol for e in self.entries}))

    def to_dict(self) -> dict[str, object]:
        return {"dataset": self.dataset, "entries": [asdict(e) for e in self.entries]}

    def manifest_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.dataset}.json"
        body = {
            **self.to_dict(),
            "manifest_hash": self.manifest_hash(),
            "total_rows": self.total_rows,
        }
        path.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> DatasetManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = tuple(ManifestEntry(**e) for e in raw["entries"])
        m = cls(raw["dataset"], entries)
        if "manifest_hash" in raw and raw["manifest_hash"] != m.manifest_hash():
            raise ValueError(f"{path}: manifest hash tutarsız (dosya elle değiştirilmiş olabilir)")
        return m


def build_manifest(repo: DataRepository, dataset: DatasetName) -> DatasetManifest:
    entries: list[ManifestEntry] = []
    for f in repo.partition_files(dataset):
        stats = (
            pl.scan_parquet(f)
            .select(
                pl.len().alias("n"),
                pl.col("ts_ms").min().alias("lo"),
                pl.col("ts_ms").max().alias("hi"),
            )
            .collect()
        )
        n, lo, hi = stats.row(0)
        entries.append(
            ManifestEntry(
                path=f.relative_to(repo.root).as_posix(),
                symbol=f.parent.parent.name.removeprefix("symbol="),
                month=f.parent.name.removeprefix("month="),
                sha256=sha256_file(f),
                rows=int(n),
                min_ts_ms=None if lo is None else int(lo),
                max_ts_ms=None if hi is None else int(hi),
            )
        )
    entries.sort(key=lambda e: e.path)
    return DatasetManifest(dataset, tuple(entries))
