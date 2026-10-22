from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pairs_bt.data.manifest import DatasetManifest, build_manifest
from pairs_bt.data.normalize import KLINES_SCHEMA, normalize_klines
from pairs_bt.data.repository import DataRepository, month_of_ms
from tests.fixtures.binance_archive import day_ms, klines_csv, make_klines, zip_bytes

T_AUG31 = day_ms(date(2025, 8, 31))
T_SEP1 = day_ms(date(2025, 9, 1))


def _klines(rng: np.random.Generator, start_ms: int, n: int) -> pl.DataFrame:
    return normalize_klines(
        zip_bytes("a.csv", klines_csv(make_klines(rng, start_ms, n))), 60_000
    ).df


def test_month_of_ms() -> None:
    assert month_of_ms(T_SEP1) == "2025-09"
    assert month_of_ms(T_SEP1 - 1) == "2025-08"


def test_write_read_roundtrip(rng: np.random.Generator, tmp_path: Path) -> None:
    repo = DataRepository(tmp_path)
    df = _klines(rng, T_AUG31, 3 * 1440)  # 31 Ağu, 1 Eyl, 2 Eyl (iki ay partition'ı)
    repo.write_symbol("klines_1m", "BTCUSDT", df)
    out = repo.read_range("klines_1m", "BTCUSDT", df["ts_ms"].min(), df["ts_ms"].max() + 1)  # type: ignore[arg-type]
    assert out.schema == KLINES_SCHEMA
    assert out.equals(df)


def test_partitioned_by_ts_month(rng: np.random.Generator, tmp_path: Path) -> None:
    repo = DataRepository(tmp_path)
    repo.write_symbol("klines_1m", "BTCUSDT", _klines(rng, T_AUG31, 1440))
    files = repo.partition_files("klines_1m")
    rel = sorted(p.relative_to(tmp_path).as_posix() for p in files)
    # 31 Ağu'nun son barının ts'i (= 1 Eyl 00:00) Eylül partition'ına düşer
    assert rel == [
        "dataset=klines_1m/symbol=BTCUSDT/month=2025-08/part.parquet",
        "dataset=klines_1m/symbol=BTCUSDT/month=2025-09/part.parquet",
    ]
    sep = pl.read_parquet(files[1])
    assert sep.height == 1
    assert sep["ts_ms"][0] == T_SEP1


def test_read_range_is_half_open(rng: np.random.Generator, tmp_path: Path) -> None:
    repo = DataRepository(tmp_path)
    repo.write_symbol("klines_1m", "BTCUSDT", _klines(rng, T_SEP1, 10))
    start, end = T_SEP1 + 2 * 60_000, T_SEP1 + 5 * 60_000
    out = repo.read_range("klines_1m", "BTCUSDT", start, end)
    assert out["ts_ms"].to_list() == [start, start + 60_000, start + 120_000]


def test_read_range_missing_symbol_returns_empty(tmp_path: Path) -> None:
    out = DataRepository(tmp_path).read_range("klines_1m", "NOPE", 0, 10)
    assert out.height == 0
    assert out.schema == KLINES_SCHEMA


def test_rewrite_replaces_symbol(rng: np.random.Generator, tmp_path: Path) -> None:
    repo = DataRepository(tmp_path)
    repo.write_symbol("klines_1m", "BTCUSDT", _klines(rng, T_AUG31, 10))
    repo.write_symbol("klines_1m", "BTCUSDT", _klines(rng, T_SEP1, 10))
    assert [p.parent.name for p in repo.partition_files("klines_1m")] == ["month=2025-09"]


def test_write_rejects_bad_schema_and_duplicates(rng: np.random.Generator, tmp_path: Path) -> None:
    repo = DataRepository(tmp_path)
    df = _klines(rng, T_SEP1, 5)
    with pytest.raises(ValueError, match="şema"):
        repo.write_symbol("klines_1m", "BTCUSDT", df.drop("trades"))
    with pytest.raises(ValueError, match="sıralı"):
        repo.write_symbol("klines_1m", "BTCUSDT", pl.concat([df, df]))


def test_symbols_sorted(rng: np.random.Generator, tmp_path: Path) -> None:
    repo = DataRepository(tmp_path)
    for s in ("ZUSDT", "AUSDT"):
        repo.write_symbol("klines_1m", s, _klines(rng, T_SEP1, 2))
    assert repo.symbols("klines_1m") == ("AUSDT", "ZUSDT")
    assert repo.symbols("klines_5m") == ()


def test_manifest_deterministic(rng: np.random.Generator, tmp_path: Path) -> None:
    repo = DataRepository(tmp_path)
    repo.write_symbol("klines_1m", "BTCUSDT", _klines(np.random.default_rng(1), T_AUG31, 1440))
    m1 = build_manifest(repo, "klines_1m")
    assert len(m1.entries) == 2
    e = m1.entries[0]
    assert e.symbol == "BTCUSDT"
    assert e.rows == 1439
    assert e.min_ts_ms == T_AUG31 + 60_000
    assert m1.total_rows == 1440
    # Aynı veri tekrar yazılınca hash aynı
    repo.write_symbol("klines_1m", "BTCUSDT", _klines(np.random.default_rng(1), T_AUG31, 1440))
    assert build_manifest(repo, "klines_1m").manifest_hash() == m1.manifest_hash()
    # Farklı veri -> farklı hash
    repo.write_symbol("klines_1m", "BTCUSDT", _klines(np.random.default_rng(2), T_AUG31, 1440))
    assert build_manifest(repo, "klines_1m").manifest_hash() != m1.manifest_hash()


def test_manifest_json_roundtrip(rng: np.random.Generator, tmp_path: Path) -> None:
    repo = DataRepository(tmp_path / "pq")
    repo.write_symbol("klines_1m", "BTCUSDT", _klines(rng, T_SEP1, 5))
    m = build_manifest(repo, "klines_1m")
    path = m.write(tmp_path / "manifests")
    assert path.name == "klines_1m.json"
    loaded = DatasetManifest.read(path)
    assert loaded == m
    assert loaded.manifest_hash() == m.manifest_hash()
