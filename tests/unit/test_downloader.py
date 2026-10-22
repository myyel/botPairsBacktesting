from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from pairs_bt.config import DownloadConfig, UniverseConfig
from pairs_bt.data.archive import ArchiveFile
from pairs_bt.data.downloader import BinanceArchiveDownloader, select_universe
from tests.fixtures.binance_archive import (
    FakeArchive,
    day_ms,
    ei_symbol,
    exchange_info_json,
    funding_csv,
    klines_csv,
    make_klines,
)

KEY_SEP = ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2025-09").key


@pytest.fixture
def archive(rng: np.random.Generator) -> FakeArchive:
    fa = FakeArchive()
    rows = make_klines(rng, day_ms(date(2025, 9, 1)), 10)
    fa.add_zip(KEY_SEP, klines_csv(rows))
    return fa


def _dl(fa: FakeArchive, tmp_path: Path, **cfg: object) -> BinanceArchiveDownloader:
    config = DownloadConfig(backoff_base_seconds=0.0, max_workers=2, **cfg)  # type: ignore[arg-type]
    return BinanceArchiveDownloader(fa.client(), tmp_path, config, sleep=lambda _s: None)


def test_download_verifies_and_writes(archive: FakeArchive, tmp_path: Path) -> None:
    f = ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2025-09")
    res = _dl(archive, tmp_path).download(f)
    assert res.status == "downloaded"
    local = tmp_path / f.key
    assert local.read_bytes() == archive.files[f.key]
    assert (tmp_path / f.checksum_key).exists()
    assert not list(tmp_path.rglob("*.part"))
    assert res.sha256 is not None
    assert len(res.sha256) == 64


def test_second_download_is_skipped(archive: FakeArchive, tmp_path: Path) -> None:
    f = ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2025-09")
    dl = _dl(archive, tmp_path)
    dl.download(f)
    n_requests = len(archive.requests)
    assert dl.download(f).status == "skipped_verified"
    assert len(archive.requests) == n_requests  # ağa hiç çıkmadı


def test_locally_corrupted_file_is_redownloaded(archive: FakeArchive, tmp_path: Path) -> None:
    f = ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2025-09")
    dl = _dl(archive, tmp_path)
    dl.download(f)
    (tmp_path / f.key).write_bytes(b"bozuk")
    assert dl.download(f).status == "downloaded"
    assert (tmp_path / f.key).read_bytes() == archive.files[f.key]


def test_checksum_mismatch_writes_nothing(archive: FakeArchive, tmp_path: Path) -> None:
    archive.corrupt.add(KEY_SEP)
    f = ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2025-09")
    res = _dl(archive, tmp_path).download(f)
    assert res.status == "checksum_mismatch"
    assert not (tmp_path / f.key).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_missing_checksum_is_an_error(rng: np.random.Generator, tmp_path: Path) -> None:
    fa = FakeArchive()
    fa.add_zip(KEY_SEP, klines_csv(make_klines(rng, 0, 3)), with_checksum=False)
    res = _dl(fa, tmp_path).download(ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2025-09"))
    assert res.status == "checksum_missing"
    assert not (tmp_path / KEY_SEP).exists()


def test_404_reported_as_missing(archive: FakeArchive, tmp_path: Path) -> None:
    res = _dl(archive, tmp_path).download(ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2025-10"))
    assert res.status == "missing"


def test_transient_5xx_is_retried(archive: FakeArchive, tmp_path: Path) -> None:
    archive.fail_once.add(KEY_SEP)
    res = _dl(archive, tmp_path).download(ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2025-09"))
    assert res.status == "downloaded"
    assert sum(r.endswith(KEY_SEP) for r in archive.requests) == 2


def test_retries_exhausted_is_error(archive: FakeArchive, tmp_path: Path) -> None:
    archive.fail_once.add(KEY_SEP)
    res = _dl(archive, tmp_path, max_retries=0).download(
        ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2025-09")
    )
    assert res.status == "error"
    assert not (tmp_path / KEY_SEP).exists()


def test_download_many_is_sorted(rng: np.random.Generator, tmp_path: Path) -> None:
    fa = FakeArchive()
    files = [ArchiveFile("klines_1m", s, "monthly", "2025-09") for s in ("ZUSDT", "AUSDT", "MUSDT")]
    for f in files:
        fa.add_zip(f.key, klines_csv(make_klines(rng, 0, 3)))
    results = _dl(fa, tmp_path).download_many(files)
    assert [r.file.symbol for r in results] == ["AUSDT", "MUSDT", "ZUSDT"]
    assert {r.status for r in results} == {"downloaded"}


def test_list_symbols_paginates(rng: np.random.Generator, tmp_path: Path) -> None:
    fa = FakeArchive(page_size=2)
    for s in ("AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"):
        fa.add_zip(ArchiveFile("klines_1m", s, "monthly", "2025-09").key, b"x")
    fa.add_zip(ArchiveFile("klines_1m", "FUSDT", "daily", "2026-09-01").key, b"x")  # sadece daily
    assert _dl(fa, tmp_path).list_symbols() == (
        "AUSDT",
        "BUSDT",
        "CUSDT",
        "DUSDT",
        "EUSDT",
        "FUSDT",
    )


def test_plan_uses_listing(rng: np.random.Generator, tmp_path: Path) -> None:
    fa = FakeArchive()
    fa.add_zip(ArchiveFile("klines_1m", "AUSDT", "monthly", "2025-09").key, b"x")
    fa.add_zip(ArchiveFile("klines_1m", "AUSDT", "daily", "2025-10-01").key, b"x")
    fa.add_zip(ArchiveFile("klines_1m", "AUSDT", "daily", "2025-10-02").key, b"x")
    fa.add_zip(ArchiveFile("fundingRate", "AUSDT", "monthly", "2025-09").key, funding_csv([]))
    dl = _dl(fa, tmp_path)
    plan = dl.plan("klines_1m", "AUSDT", date(2025, 9, 1), date(2025, 10, 1))
    assert [(f.period, f.label) for f in plan] == [("monthly", "2025-09"), ("daily", "2025-10-01")]
    assert [
        f.label for f in dl.plan("fundingRate", "AUSDT", date(2025, 9, 1), date(2025, 10, 31))
    ] == ["2025-09"]


def test_exchange_info_snapshot_saved_raw(tmp_path: Path) -> None:
    payload = exchange_info_json([ei_symbol("BTCUSDT")], server_time_ms=1_792_137_674_765)
    fa = FakeArchive(exchange_info=payload)
    path = _dl(fa, tmp_path).fetch_exchange_info_snapshot()
    assert path == tmp_path / "exchange_info" / "exchangeInfo_1792137674765.json"
    assert path.read_bytes() == payload  # ham bayt, değiştirilmeden


def test_select_universe_filters() -> None:
    ei = {
        "BTCUSDT": ("PERPETUAL", "USDT"),
        "XAUUSDT": ("TRADIFI_PERPETUAL", "USDT"),
        "BTCUSDC": ("PERPETUAL", "USDC"),
    }
    archive_symbols = ("BTCUSDT", "XAUUSDT", "BTCUSDC", "OLDUSDT", "BTCUSDT_250926", "ETHBUSD")
    cfg = UniverseConfig(exclude_symbols=("ETHUSDT",))
    # OLDUSDT exchangeInfo'da yok (eskiden delist) -> USDT soneki ile dahil (survivorship)
    assert select_universe(archive_symbols, ei, cfg) == ("BTCUSDT", "OLDUSDT")
    cfg2 = UniverseConfig(contract_types=("PERPETUAL", "TRADIFI_PERPETUAL"))
    assert select_universe(archive_symbols, ei, cfg2) == ("BTCUSDT", "OLDUSDT", "XAUUSDT")
    cfg3 = UniverseConfig(include_symbols=("XAUUSDT", "BTCUSDT"))
    assert select_universe(archive_symbols, ei, cfg3) == ("BTCUSDT",)  # include de filtreden geçer


def test_contract_map_from_exchange_info(tmp_path: Path) -> None:
    payload = json.loads(
        exchange_info_json(
            [ei_symbol("BTCUSDT"), ei_symbol("XAUUSDT", contract_type="TRADIFI_PERPETUAL")], 1
        )
    )
    from pairs_bt.data.downloader import contract_map

    assert contract_map(payload) == {
        "BTCUSDT": ("PERPETUAL", "USDT"),
        "XAUUSDT": ("TRADIFI_PERPETUAL", "USDT"),
    }
