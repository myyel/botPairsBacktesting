"""Gerçek data.binance.vision'a küçük smoke test (varsayılan çalıştırılmaz: -m network)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from pairs_bt.config import DownloadConfig
from pairs_bt.data.archive import ArchiveFile
from pairs_bt.data.downloader import BinanceArchiveDownloader
from pairs_bt.data.normalize import normalize_funding


@pytest.mark.network
def test_real_funding_file_downloads_and_normalizes(tmp_path: Path) -> None:
    f = ArchiveFile("fundingRate", "BTCUSDT", "monthly", "2025-09")
    with httpx.Client() as client:
        res = BinanceArchiveDownloader(client, tmp_path, DownloadConfig()).download(f)
    assert res.status == "downloaded"
    df = normalize_funding((tmp_path / f.key).read_bytes()).df
    assert df.height >= 89  # 30 gün x 3 (8 saatlik)
