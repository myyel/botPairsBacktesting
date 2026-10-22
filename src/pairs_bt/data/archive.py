"""data.binance.vision USDⓈ-M arşiv şeması: anahtarlar, S3 listing, indirme planı (RAPOR §4.1).

Bu modül ağa çıkmaz; saf fonksiyonlardan oluşur. Ağ erişimi `data.downloader`'dadır.
"""

from __future__ import annotations

import calendar
import re
import xml.etree.ElementTree as ET
from collections.abc import Set
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal
from urllib.parse import urlencode

from pairs_bt.config import DatasetName

Period = Literal["monthly", "daily"]
ArchiveKind = Literal["klines", "aggTrades", "fundingRate"]

ROOT_PREFIX = "data/futures/um"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: DatasetName
    kind: ArchiveKind
    interval: str | None  # klines için "1m"/"5m"
    interval_ms: int | None
    daily_available: bool


DATASETS: dict[DatasetName, DatasetSpec] = {
    "klines_1m": DatasetSpec("klines_1m", "klines", "1m", 60_000, True),
    "klines_5m": DatasetSpec("klines_5m", "klines", "5m", 300_000, True),
    "aggTrades": DatasetSpec("aggTrades", "aggTrades", None, None, True),
    "fundingRate": DatasetSpec("fundingRate", "fundingRate", None, None, False),
}

_KEY_RE = re.compile(
    rf"^{ROOT_PREFIX}/(?P<period>monthly|daily)/(?P<kind>klines|aggTrades|fundingRate)/"
    r"(?P<symbol>[A-Z0-9_]+)/(?:(?P<interval>\w+)/)?"
    r"(?P=symbol)-(?P<tag>\w+)-(?P<label>\d{4}-\d{2}(?:-\d{2})?)\.zip$"
)


@dataclass(frozen=True, slots=True)
class ArchiveFile:
    dataset: DatasetName
    symbol: str
    period: Period
    label: str  # monthly: YYYY-MM, daily: YYYY-MM-DD

    def __post_init__(self) -> None:
        spec = DATASETS[self.dataset]
        if self.period == "daily" and not spec.daily_available:
            raise ValueError(f"{self.dataset} için daily arşiv yok")
        expected_len = 7 if self.period == "monthly" else 10
        if len(self.label) != expected_len:
            raise ValueError(f"{self.period} label formatı hatalı: {self.label}")

    @property
    def spec(self) -> DatasetSpec:
        return DATASETS[self.dataset]

    @property
    def filename(self) -> str:
        tag = self.spec.interval or self.spec.kind
        return f"{self.symbol}-{tag}-{self.label}.zip"

    @property
    def key(self) -> str:
        return symbol_prefix(self.dataset, self.symbol, self.period) + self.filename

    @property
    def checksum_key(self) -> str:
        return self.key + ".CHECKSUM"

    def url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/{self.key}"

    @property
    def first_day(self) -> date:
        if self.period == "daily":
            return date.fromisoformat(self.label)
        y, m = map(int, self.label.split("-"))
        return date(y, m, 1)

    @property
    def last_day(self) -> date:
        if self.period == "daily":
            return self.first_day
        y, m = map(int, self.label.split("-"))
        return date(y, m, calendar.monthrange(y, m)[1])


def symbol_prefix(dataset: DatasetName, symbol: str, period: Period) -> str:
    spec = DATASETS[dataset]
    base = f"{ROOT_PREFIX}/{period}/{spec.kind}/{symbol}/"
    return base + (f"{spec.interval}/" if spec.interval else "")


def file_from_key(key: str) -> ArchiveFile | None:
    """Arşiv anahtarını `ArchiveFile`'a çevirir; desteklenmeyen anahtar için None."""
    m = _KEY_RE.match(key)
    if m is None:
        return None
    kind, interval, tag = m["kind"], m["interval"], m["tag"]
    for spec in DATASETS.values():
        if spec.kind == kind and spec.interval == interval and tag == (interval or kind):
            try:
                return ArchiveFile(spec.name, m["symbol"], m["period"], m["label"])  # type: ignore[arg-type]
            except ValueError:
                return None
    return None


def months_between(start: date, end: date) -> list[str]:
    out: list[str] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def days_between(start: date, end: date) -> list[str]:
    n = (end - start).days
    return [(start + timedelta(days=i)).isoformat() for i in range(n + 1)]


def plan_files(
    dataset: DatasetName,
    symbol: str,
    start: date,
    end: date,
    monthly_available: Set[str],
    daily_available: Set[str],
) -> list[ArchiveFile]:
    """[start, end] (dahil) aralığını kapsayan dosya listesi.

    Ay aralığın tamamen içindeyse ve monthly dosya varsa monthly kullanılır. Kısmi aylarda
    (aralık kenarları) veya monthly henüz yayınlanmamışsa aralıktaki daily dosyalar kullanılır;
    daily de yoksa monthly'e düşülür. Sonuç kronolojik sıradadır (CLAUDE.md invariant #7).
    """
    spec = DATASETS[dataset]
    out: list[ArchiveFile] = []
    for month in months_between(start, end):
        y, m = map(int, month.split("-"))
        month_first, month_last = date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
        full = start <= month_first and month_last <= end
        lo, hi = max(start, month_first), min(end, month_last)
        dailies = [
            ArchiveFile(dataset, symbol, "daily", d)
            for d in days_between(lo, hi)
            if spec.daily_available and d in daily_available
        ]
        if month in monthly_available and (full or not dailies):
            out.append(ArchiveFile(dataset, symbol, "monthly", month))
        else:
            out.extend(dailies)
    return out


@dataclass(frozen=True, slots=True)
class S3ListingPage:
    prefixes: tuple[str, ...]
    keys: tuple[str, ...]
    is_truncated: bool
    next_marker: str | None


def parse_s3_listing(xml: bytes) -> S3ListingPage:
    root = ET.fromstring(xml)
    prefixes = tuple(
        (e.text or "") for e in root.iterfind(f"{_S3_NS}CommonPrefixes/{_S3_NS}Prefix")
    )
    keys = tuple((e.text or "") for e in root.iterfind(f"{_S3_NS}Contents/{_S3_NS}Key"))
    truncated = (root.findtext(f"{_S3_NS}IsTruncated") or "false").lower() == "true"
    marker = root.findtext(f"{_S3_NS}NextMarker")
    if truncated and not marker:
        candidates = keys + prefixes
        marker = max(candidates) if candidates else None
    return S3ListingPage(prefixes, keys, truncated, marker if truncated else None)


def listing_url(s3_base_url: str, prefix: str, *, marker: str, delimiter: bool) -> str:
    params = {"prefix": prefix}
    if marker:
        params["marker"] = marker
    if delimiter:
        params["delimiter"] = "/"
    return f"{s3_base_url}?{urlencode(params)}"
