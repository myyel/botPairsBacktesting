"""Symbol lifecycle manifest — survivorship-bias'sız universe (CLAUDE.md invariant #3, RAPOR §12.3).

Kaynaklar: indirilen 1m kline verisinin işlem görmüş (trades > 0) ilk/son bar'ı +
exchangeInfo meta'sı
(contractType, status, onboardDate, deliveryDate). Delist edilen semboller manifest'ten silinmez.

Look-ahead notu: `active_symbols(t)` bir sembolü yalnız verisi bittikten / delivery zamanı geçtikten
SONRA dışlar; "ileride delist edilecek" bilgisi t'deki seçimi etkilemez. Universe filtresi
(Faz 2) bu bilgiyi kullanamaz.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from pairs_bt.data.exchange_info import ExchangeInfoStore
from pairs_bt.data.repository import DataRepository

_DAY_MS = 86_400_000
# exchangeInfo'da aktif perpetual'lar için deliveryDate uzak gelecek bir sentinel'dir (2100 yılı).
_FAR_FUTURE_MS = 4_000_000_000_000


@dataclass(frozen=True, slots=True)
class LifecycleEntry:
    symbol: str
    first_data_ms: int
    last_data_ms: int
    contract_type: str | None  # None: hiçbir snapshot'ta yok (çok önce delist)
    status: str | None
    onboard_ms: int | None
    delivery_ms: int | None

    def active_at(self, ts_ms: int) -> bool:
        if not self.first_data_ms <= ts_ms <= self.last_data_ms:
            return False
        return self.delivery_ms is None or ts_ms < self.delivery_ms

    def listing_age_days(self, ts_ms: int) -> float:
        """onboardDate biliniyorsa ondan, yoksa ilk veriden (alt sınır) itibaren gün."""
        origin = self.onboard_ms if self.onboard_ms is not None else self.first_data_ms
        return (ts_ms - origin) / _DAY_MS


@dataclass(frozen=True, slots=True)
class SymbolLifecycle:
    entries: tuple[LifecycleEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(sorted(self.entries, key=lambda e: e.symbol)))

    def get(self, symbol: str) -> LifecycleEntry | None:
        for e in self.entries:
            if e.symbol == symbol:
                return e
        return None

    def active_symbols(self, ts_ms: int) -> tuple[str, ...]:
        return tuple(e.symbol for e in self.entries if e.active_at(ts_ms))

    def delisted(self) -> tuple[str, ...]:
        return tuple(
            e.symbol
            for e in self.entries
            if e.contract_type is None or (e.status is not None and e.status != "TRADING")
        )

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"entries": [asdict(e) for e in self.entries]}
        path.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> SymbolLifecycle:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(tuple(LifecycleEntry(**e) for e in raw["entries"]))


def build_lifecycle(repo: DataRepository, store: ExchangeInfoStore) -> SymbolLifecycle:
    entries: list[LifecycleEntry] = []
    for sym in repo.symbols("klines_1m"):
        files = repo.partition_files_for("klines_1m", sym)
        # İşlemsiz bar'lar sentetiktir (ör. delist sonrası sabit fiyat); aralık işlem görenlerden.
        traded = pl.col("ts_ms").filter(pl.col("trades") > 0)
        stats = (
            pl.scan_parquet(files)
            .select(traded.min().alias("lo"), traded.max().alias("hi"))
            .collect()
        )
        lo, hi = stats.row(0)
        if lo is None:
            continue
        meta = store.meta(sym)
        delivery = meta.delivery_ms if meta is not None else None
        entries.append(
            LifecycleEntry(
                symbol=sym,
                first_data_ms=int(lo),
                last_data_ms=int(hi),
                contract_type=meta.contract_type if meta else None,
                status=meta.status if meta else None,
                onboard_ms=meta.onboard_ms if meta else None,
                delivery_ms=delivery
                if delivery is not None and delivery < _FAR_FUTURE_MS
                else None,
            )
        )
    return SymbolLifecycle(tuple(entries))
