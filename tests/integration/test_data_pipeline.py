"""Uçtan uca veri pipeline'ı: sahte arşiv -> download -> build -> validate -> manifest (ağsız)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from tests.fixtures.binance_archive import (
    FakeArchive,
    day_ms,
    ei_symbol,
    exchange_info_json,
    funding_csv,
    klines_csv,
    make_klines,
    resample_rows,
)

from pairs_bt.config import DataConfig, DataPathsConfig, DownloadConfig, UniverseConfig
from pairs_bt.data.archive import ArchiveFile
from pairs_bt.data.lifecycle import SymbolLifecycle
from pairs_bt.data.manifest import DatasetManifest
from pairs_bt.data.pipeline import run_build, run_download
from pairs_bt.data.repository import DataRepository

pytestmark = pytest.mark.slow

SEP1 = day_ms(date(2025, 9, 1))
OCT1 = day_ms(date(2025, 10, 1))
SNAP_MS = day_ms(date(2026, 10, 18))
H = 3_600_000


def _add_symbol(
    fa: FakeArchive, rng: np.random.Generator, sym: str, start_ms: int, n_1m: int
) -> None:
    """Eylül monthly + 1-2 Ekim daily (Ekim monthly 'henüz yayınlanmamış')."""
    rows = make_klines(rng, start_ms, n_1m)
    sep = [r for r in rows if r[0] < OCT1]
    octo = [r for r in rows if r[0] >= OCT1]
    if sep:
        fa.add_zip(ArchiveFile("klines_1m", sym, "monthly", "2025-09").key, klines_csv(sep))
        fa.add_zip(
            ArchiveFile("klines_5m", sym, "monthly", "2025-09").key,
            klines_csv(resample_rows(sep, 5), 300_000),
        )
        fund = [(t, 8, 0.0001) for t in range(SEP1, OCT1, 8 * H) if t >= start_ms]
        fa.add_zip(ArchiveFile("fundingRate", sym, "monthly", "2025-09").key, funding_csv(fund))
    for day in ("2025-10-01", "2025-10-02"):
        d0 = day_ms(date.fromisoformat(day))
        part = [r for r in octo if d0 <= r[0] < d0 + 86_400_000]
        if part:
            fa.add_zip(ArchiveFile("klines_1m", sym, "daily", day).key, klines_csv(part))
            fa.add_zip(
                ArchiveFile("klines_5m", sym, "daily", day).key,
                klines_csv(resample_rows(part, 5), 300_000),
            )


@pytest.fixture
def setup(rng: np.random.Generator, tmp_path: Path) -> tuple[FakeArchive, DataConfig]:
    fa = FakeArchive(page_size=3)
    day_bars = 1440
    late = OCT1 - 3 * 86_400_000  # 28 Eylül'de listelenen sembol
    _add_symbol(fa, rng, "AUSDT", SEP1, 32 * day_bars)
    _add_symbol(fa, rng, "BUSDT", late, 5 * day_bars)
    _add_symbol(fa, rng, "DELUSDT", SEP1, 10 * day_bars)  # 11 Eylül'de delist
    _add_symbol(fa, rng, "OLDUSDT", SEP1, 2 * day_bars)  # exchangeInfo'da hiç yok
    _add_symbol(fa, rng, "XAUUSDT", SEP1, 2 * day_bars)  # TradFi -> hariç
    fa.exchange_info = exchange_info_json(
        [
            ei_symbol("AUSDT", onboard_ms=SEP1 - 400 * 86_400_000),
            ei_symbol("BUSDT", onboard_ms=late),
            ei_symbol("DELUSDT", status="SETTLING", delivery_ms=SEP1 + 10 * 86_400_000),
            ei_symbol("XAUUSDT", contract_type="TRADIFI_PERPETUAL"),
        ],
        SNAP_MS,
    )
    cfg = DataConfig(
        start_date=date(2025, 9, 1),
        end_date=date(2025, 10, 2),
        datasets=("klines_1m", "klines_5m", "fundingRate"),
        universe=UniverseConfig(),
        paths=DataPathsConfig(
            raw_dir=tmp_path / "raw",
            normalized_dir=tmp_path / "normalized",
            parquet_dir=tmp_path / "parquet",
            manifests_dir=tmp_path / "manifests",
        ),
        download=DownloadConfig(backoff_base_seconds=0.0, max_workers=4),
    )
    return fa, cfg


def test_end_to_end(setup: tuple[FakeArchive, DataConfig]) -> None:
    fa, cfg = setup
    dl = run_download(cfg, fa.client())
    assert dl.universe == ("AUSDT", "BUSDT", "DELUSDT", "OLDUSDT")  # TradFi hariç, delist dahil
    assert dl.failed == ()
    assert (
        (cfg.paths.manifests_dir / "download_report.md")
        .read_text(encoding="utf-8")
        .startswith("# İndirme Raporu")
    )

    build = run_build(cfg)
    repo = DataRepository(cfg.paths.parquet_dir)
    assert repo.symbols("klines_1m") == ("AUSDT", "BUSDT", "DELUSDT", "OLDUSDT")
    a = repo.read_symbol("klines_1m", "AUSDT")
    # Eylül monthly + 2 günlük Ekim daily birleşik, tekrar yok, tam kapsama
    assert a.height == 32 * 1440
    assert a["ts_ms"].is_sorted()
    assert build.report.error_count == 0

    # Manifestler + hash
    m = DatasetManifest.read(cfg.paths.manifests_dir / "klines_1m.json")
    assert m.total_rows == sum(
        repo.read_symbol("klines_1m", s).height for s in repo.symbols("klines_1m")
    )
    index = json.loads((cfg.paths.manifests_dir / "dataset_index.json").read_text(encoding="utf-8"))
    assert index["data_manifest_hash"] == build.data_manifest_hash
    assert set(index["datasets"]) == {"klines_1m", "klines_5m", "fundingRate"}

    # Lifecycle: delist edilen sembol korunur, delivery sonrası aktif değil
    lc = SymbolLifecycle.read(cfg.paths.manifests_dir / "symbol_lifecycle.json")
    assert {"DELUSDT", "OLDUSDT"} <= set(lc.delisted())
    mid_sep = SEP1 + 15 * 86_400_000
    assert "DELUSDT" not in lc.active_symbols(mid_sep)
    assert "AUSDT" in lc.active_symbols(mid_sep)
    assert "BUSDT" not in lc.active_symbols(mid_sep)  # henüz listelenmedi

    # Türkçe rapor
    summary = (cfg.paths.manifests_dir / "build_summary.md").read_text(encoding="utf-8")
    assert "Veri Seti Özeti" in summary
    assert build.data_manifest_hash in summary
    assert (cfg.paths.manifests_dir / "validation_report.md").exists()
    # normalized katmanı: kaynak dosya bazlı normalizasyon kayıtları
    assert (cfg.paths.normalized_dir / "klines_1m" / "AUSDT.json").exists()


def test_build_is_deterministic(setup: tuple[FakeArchive, DataConfig]) -> None:
    fa, cfg = setup
    run_download(cfg, fa.client())
    h1 = run_build(cfg).data_manifest_hash
    h2 = run_build(cfg).data_manifest_hash
    assert h1 == h2


def test_download_is_idempotent(setup: tuple[FakeArchive, DataConfig]) -> None:
    fa, cfg = setup
    first = run_download(cfg, fa.client())
    second = run_download(cfg, fa.client())
    assert first.downloaded > 0
    assert second.downloaded == 0
    assert second.skipped == first.downloaded


def test_checksum_failure_is_reported(setup: tuple[FakeArchive, DataConfig]) -> None:
    fa, cfg = setup
    bad = ArchiveFile("klines_1m", "AUSDT", "monthly", "2025-09").key
    fa.corrupt.add(bad)
    dl = run_download(cfg, fa.client())
    assert [r.file.key for r in dl.failed] == [bad]
    report = (cfg.paths.manifests_dir / "download_report.md").read_text(encoding="utf-8")
    assert "checksum_mismatch" in report


def test_apply_overrides(setup: tuple[FakeArchive, DataConfig]) -> None:
    from pairs_bt.data.pipeline import apply_overrides

    _, cfg = setup
    out = apply_overrides(
        cfg, symbols="BUSDT,AUSDT", start="2025-09-05", end="2025-09-06", datasets="aggTrades"
    )
    assert out.universe.include_symbols == ("AUSDT", "BUSDT")
    assert (out.start_date, out.end_date) == (date(2025, 9, 5), date(2025, 9, 6))
    assert out.datasets == ("aggTrades",)
    assert apply_overrides(cfg) == cfg


def test_partial_build_keeps_other_datasets(setup: tuple[FakeArchive, DataConfig]) -> None:
    """Sadece fundingRate yeniden build edilse de kline manifestleri/tick'leri korunur."""
    from pairs_bt.data.pipeline import apply_overrides

    fa, cfg = setup
    run_download(cfg, fa.client())
    full = run_build(cfg)
    partial = run_build(apply_overrides(cfg, datasets="fundingRate"))
    assert set(partial.manifests) == {"klines_1m", "klines_5m", "fundingRate"}
    assert partial.data_manifest_hash == full.data_manifest_hash
    ticks = json.loads((cfg.paths.manifests_dir / "derived_ticks.json").read_text(encoding="utf-8"))
    assert "AUSDT" in ticks
