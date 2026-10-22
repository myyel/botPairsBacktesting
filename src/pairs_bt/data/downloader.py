"""BinanceArchiveDownloader — data.binance.vision indirme + .CHECKSUM doğrulama.

RAPOR §4, CLAUDE.md §7.

- Dosyalar `raw_root/<arşiv anahtarı>` altına arşivle aynı yapıda, bayt bayt kaydedilir.
- Yazım atomiktir (`.part` -> rename) ve yalnızca SHA-256 doğrulandıktan sonra yapılır.
- Doğrulanmış yerel dosya tekrar indirilmez (idempotent).
- `data/raw` içeriği asla düzenlenmez; yalnız eksik/bozuk dosya yeniden indirilir.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog

from pairs_bt.config import DatasetName, DownloadConfig, UniverseConfig
from pairs_bt.data.archive import (
    DATASETS,
    ROOT_PREFIX,
    ArchiveFile,
    file_from_key,
    listing_url,
    months_between,
    parse_s3_listing,
    plan_files,
    symbol_prefix,
)

log = structlog.get_logger(__name__)

DownloadStatus = Literal[
    "downloaded", "skipped_verified", "missing", "checksum_mismatch", "checksum_missing", "error"
]
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class DownloadResult:
    file: ArchiveFile
    status: DownloadStatus
    sha256: str | None = None
    size_bytes: int = 0
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("downloaded", "skipped_verified")


class _NotFoundError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksum(text: bytes) -> str:
    """`.CHECKSUM` içeriği: `<sha256 hex>  <dosya adı>`."""
    token = text.decode("ascii", errors="replace").strip().split()[0] if text.strip() else ""
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token.lower()):
        raise ValueError(f"geçersiz CHECKSUM içeriği: {text[:80]!r}")
    return token.lower()


def contract_map(exchange_info: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """exchangeInfo JSON -> {symbol: (contractType, quoteAsset)}."""
    return {
        str(s["symbol"]): (str(s.get("contractType", "")), str(s.get("quoteAsset", "")))
        for s in exchange_info.get("symbols", [])
    }


def select_universe(
    archive_symbols: Iterable[str],
    contracts: Mapping[str, tuple[str, str]],
    cfg: UniverseConfig,
) -> tuple[str, ...]:
    """İndirilecek sembolleri seçer (sıralı).

    exchangeInfo'da bulunan semboller contractType/quoteAsset ile filtrelenir. Bulunmayanlar
    (çok önce delist edilmiş) survivorship bias'ı önlemek için yalnız quote sonekiyle dahil edilir;
    `_` içeren (vadeli) semboller her durumda hariçtir.
    """
    include = set(cfg.include_symbols)
    exclude = set(cfg.exclude_symbols)
    out: list[str] = []
    for sym in sorted(set(archive_symbols)):
        if "_" in sym or sym in exclude or (include and sym not in include):
            continue
        if sym in contracts:
            ctype, quote = contracts[sym]
            if ctype in cfg.contract_types and quote == cfg.quote_asset:
                out.append(sym)
        elif sym.endswith(cfg.quote_asset):
            out.append(sym)
    return tuple(out)


class BinanceArchiveDownloader:
    def __init__(
        self,
        client: httpx.Client,
        raw_root: Path,
        cfg: DownloadConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._raw = Path(raw_root)
        self._cfg = cfg
        self._sleep = sleep

    # ------------------------------------------------------------------ HTTP
    def _get(self, url: str) -> bytes:
        attempts = self._cfg.max_retries + 1
        last: str = ""
        for attempt in range(attempts):
            try:
                resp = self._client.get(url, timeout=self._cfg.timeout_seconds)
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code == 404:
                    raise _NotFoundError(url)
                if resp.status_code not in _RETRY_STATUS:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                last = f"HTTP {resp.status_code}"
            if attempt + 1 < attempts:
                self._sleep(self._cfg.backoff_base_seconds * 2**attempt)
        raise ConnectionError(f"{url}: {attempts} denemede başarısız ({last})")

    # --------------------------------------------------------------- listing
    def list_prefix(
        self, prefix: str, *, delimiter: bool
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """S3 listing'i sayfalayarak (prefixes, keys) döner."""
        prefixes: list[str] = []
        keys: list[str] = []
        marker = ""
        while True:
            url = listing_url(self._cfg.s3_listing_url, prefix, marker=marker, delimiter=delimiter)
            page = parse_s3_listing(self._get(url))
            prefixes.extend(page.prefixes)
            keys.extend(page.keys)
            if not page.is_truncated or page.next_marker is None:
                return tuple(prefixes), tuple(keys)
            marker = page.next_marker

    def list_symbols(self) -> tuple[str, ...]:
        """Arşivde (monthly ∪ daily klines) bulunan TÜM semboller — delist edilenler dahil."""
        found: set[str] = set()
        for period in ("monthly", "daily"):
            base = f"{ROOT_PREFIX}/{period}/klines/"
            prefixes, _ = self.list_prefix(base, delimiter=True)
            found.update(p[len(base) :].rstrip("/") for p in prefixes)
        return tuple(sorted(s for s in found if s))

    def _available_labels(self, dataset: DatasetName, symbol: str, period: str) -> set[str]:
        _, keys = self.list_prefix(symbol_prefix(dataset, symbol, period), delimiter=False)  # type: ignore[arg-type]
        labels: set[str] = set()
        for k in keys:
            f = file_from_key(k)
            if f is not None and f.dataset == dataset and f.symbol == symbol and f.period == period:
                labels.add(f.label)
        return labels

    def plan(self, dataset: DatasetName, symbol: str, start: date, end: date) -> list[ArchiveFile]:
        monthly = self._available_labels(dataset, symbol, "monthly")
        daily: set[str] = set()
        months = months_between(start, end)
        partial = start.day != 1 or (end + timedelta(days=1)).day != 1
        if DATASETS[dataset].daily_available and (partial or not set(months) <= monthly):
            daily = self._available_labels(dataset, symbol, "daily")
        return plan_files(dataset, symbol, start, end, monthly, daily)

    # -------------------------------------------------------------- download
    def local_path(self, f: ArchiveFile) -> Path:
        return self._raw / f.key

    def _verified_local(self, f: ArchiveFile) -> str | None:
        path, ck = self.local_path(f), self._raw / f.checksum_key
        if not (path.exists() and ck.exists()):
            return None
        try:
            expected = parse_checksum(ck.read_bytes())
        except ValueError:
            return None
        actual = sha256_file(path)
        return actual if actual == expected else None

    def download(self, f: ArchiveFile) -> DownloadResult:
        verified = self._verified_local(f)
        if verified is not None:
            return DownloadResult(
                f, "skipped_verified", verified, self.local_path(f).stat().st_size
            )
        base = self._cfg.archive_base_url
        try:
            data = self._get(f.url(base))
        except _NotFoundError:
            return DownloadResult(f, "missing")
        except (ConnectionError, httpx.HTTPError) as exc:
            log.warning("download_failed", key=f.key, error=str(exc))
            return DownloadResult(f, "error", message=str(exc))
        try:
            ck_bytes = self._get(base.rstrip("/") + "/" + f.checksum_key)
            expected = parse_checksum(ck_bytes)
        except _NotFoundError:
            return DownloadResult(f, "checksum_missing")
        except (ConnectionError, httpx.HTTPError, ValueError) as exc:
            log.warning("checksum_failed", key=f.key, error=str(exc))
            return DownloadResult(f, "error", message=str(exc))
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            log.error("checksum_mismatch", key=f.key, expected=expected, actual=actual)
            return DownloadResult(f, "checksum_mismatch", actual, len(data))
        self._atomic_write(self._raw / f.checksum_key, ck_bytes)
        self._atomic_write(self.local_path(f), data)
        return DownloadResult(f, "downloaded", actual, len(data))

    def download_many(self, files: Sequence[ArchiveFile]) -> list[DownloadResult]:
        """Paralel indirir; sonuç sırası anahtara göre deterministiktir (invariant #7)."""
        ordered = sorted(set(files), key=lambda f: f.key)
        with ThreadPoolExecutor(max_workers=self._cfg.max_workers) as pool:
            return list(pool.map(self.download, ordered))

    def fetch_exchange_info_snapshot(self) -> Path:
        """Güncel exchangeInfo'yu ham haliyle yazar.

        Hedef: `exchange_info/exchangeInfo_<serverTime>.json`.
        """
        data = self._get(self._cfg.exchange_info_url)
        server_time = int(json.loads(data)["serverTime"])
        path = self._raw / "exchange_info" / f"exchangeInfo_{server_time}.json"
        if not path.exists():
            self._atomic_write(path, data)
        return path

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)
