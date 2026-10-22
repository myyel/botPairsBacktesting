"""Faz 1 veri pipeline'ı: download -> normalize -> parquet -> validate -> manifest.

RAPOR §4, CLAUDE.md §7.

Katmanlar:
- `raw/`        : data.binance.vision ile aynı yapıda, checksum doğrulanmış zip'ler +
                  exchangeInfo snapshot'ları
- `normalized/` : kaynak dosya bazlı normalizasyon kayıtları (satır sayısı, düzeltmeler) —
                  <dataset>/<sembol>.json
- `parquet/`    : `DataRepository` (dataset/sembol/ay partition)
- `manifests/`  : dataset manifestleri, lifecycle, türetilmiş tick'ler,
                  doğrulama + özet raporları (Türkçe)
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import polars as pl
import structlog

from pairs_bt.config import DataConfig, DatasetName
from pairs_bt.data.archive import DATASETS, ArchiveFile, file_from_key
from pairs_bt.data.downloader import (
    BinanceArchiveDownloader,
    DownloadResult,
    contract_map,
    select_universe,
)
from pairs_bt.data.exchange_info import ExchangeInfoStore, derive_tick_size
from pairs_bt.data.lifecycle import SymbolLifecycle, build_lifecycle
from pairs_bt.data.manifest import DatasetManifest, build_manifest
from pairs_bt.data.normalize import normalize_file
from pairs_bt.data.repository import DataRepository
from pairs_bt.data.validator import DataValidator, SymbolValidation, ValidationReport, fmt_ts

log = structlog.get_logger(__name__)


def _day_start_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)


def window_ms(cfg: DataConfig) -> tuple[int, int]:
    """Config penceresi `[start_date 00:00, end_date+1 00:00)` UTC, ms."""
    return _day_start_ms(cfg.start_date), _day_start_ms(cfg.end_date + timedelta(days=1))


def apply_overrides(
    cfg: DataConfig,
    *,
    symbols: str | None = None,
    start: str | None = None,
    end: str | None = None,
    datasets: str | None = None,
) -> DataConfig:
    """CLI override'larını (virgülle ayrılmış listeler, ISO tarih) uygulayıp yeniden doğrular."""
    data = cfg.model_dump()
    if symbols:
        data["universe"]["include_symbols"] = sorted(
            s.strip() for s in symbols.split(",") if s.strip()
        )
    if start:
        data["start_date"] = date.fromisoformat(start)
    if end:
        data["end_date"] = date.fromisoformat(end)
    if datasets:
        data["datasets"] = [d.strip() for d in datasets.split(",") if d.strip()]
    return DataConfig.model_validate(data)


# ====================================================================== download


@dataclass(frozen=True, slots=True)
class DownloadSummary:
    universe: tuple[str, ...]
    results: tuple[DownloadResult, ...]
    exchange_info_path: Path

    @property
    def downloaded(self) -> int:
        return sum(r.status == "downloaded" for r in self.results)

    @property
    def skipped(self) -> int:
        return sum(r.status == "skipped_verified" for r in self.results)

    @property
    def failed(self) -> tuple[DownloadResult, ...]:
        return tuple(r for r in self.results if not r.ok)


def run_download(cfg: DataConfig, client: httpx.Client) -> DownloadSummary:
    dl = BinanceArchiveDownloader(client, cfg.paths.raw_dir, cfg.download)
    ei_path = dl.fetch_exchange_info_snapshot()
    contracts = contract_map(json.loads(ei_path.read_text(encoding="utf-8")))
    universe = select_universe(dl.list_symbols(), contracts, cfg.universe)
    log.info("universe_selected", n=len(universe))
    jobs = [(ds, sym) for ds in cfg.datasets for sym in universe]
    with ThreadPoolExecutor(max_workers=cfg.download.max_workers) as pool:
        plans = list(pool.map(lambda j: dl.plan(j[0], j[1], cfg.start_date, cfg.end_date), jobs))
    files = [f for plan in plans for f in plan]
    log.info("download_planned", files=len(files))
    results = tuple(dl.download_many(files))
    summary = DownloadSummary(universe, results, ei_path)
    _write_download_report(cfg, summary)
    return summary


def _write_download_report(cfg: DataConfig, s: DownloadSummary) -> None:
    d = cfg.paths.manifests_dir
    d.mkdir(parents=True, exist_ok=True)
    by_status: dict[str, int] = defaultdict(int)
    by_ds: dict[str, int] = defaultdict(int)
    for r in s.results:
        by_status[r.status] += 1
        if r.ok:
            by_ds[r.file.dataset] += 1
    total_bytes = sum(r.size_bytes for r in s.results if r.ok)
    lines = [
        "# İndirme Raporu",
        "",
        f"- Pencere: {cfg.start_date} → {cfg.end_date} (UTC, dahil)",
        f"- Universe: {len(s.universe)} sembol",
        f"- exchangeInfo snapshot: `{s.exchange_info_path.name}`",
        f"- Toplam dosya: {len(s.results)} · boyut: {total_bytes / 1e9:.2f} GB",
        "",
        "| Durum | Dosya |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in sorted(by_status.items())),
        "",
        "| Dataset | Başarılı dosya |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in sorted(by_ds.items())),
        "",
    ]
    if s.failed:
        lines += ["## Başarısız dosyalar", ""]
        lines += [f"- `{r.file.key}`: {r.status} {r.message}".rstrip() for r in s.failed]
    (d / "download_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw = [
        {"key": r.file.key, "status": r.status, "sha256": r.sha256, "size_bytes": r.size_bytes}
        for r in s.results
    ]
    (d / "raw_manifest.json").write_text(
        json.dumps({"universe": list(s.universe), "files": raw}, indent=1), encoding="utf-8"
    )


# ========================================================================= build


@dataclass(frozen=True, slots=True)
class BuildSummary:
    report: ValidationReport
    manifests: dict[DatasetName, DatasetManifest]
    lifecycle: SymbolLifecycle
    data_manifest_hash: str
    approximate_filter_symbols: tuple[str, ...]
    missing_filter_symbols: tuple[str, ...]


def raw_files(cfg: DataConfig) -> dict[tuple[DatasetName, str], list[ArchiveFile]]:
    """raw/ altındaki, config penceresi ve dataset'leriyle örtüşen dosyalar.

    Sembol bazında gruplanır, kronolojik sıralıdır.
    """
    out: dict[tuple[DatasetName, str], list[ArchiveFile]] = defaultdict(list)
    root = cfg.paths.raw_dir
    if not root.exists():
        return {}
    for p in root.rglob("*.zip"):
        f = file_from_key(p.relative_to(root).as_posix())
        if f is None or f.dataset not in cfg.datasets:
            continue
        if f.last_day < cfg.start_date or f.first_day > cfg.end_date:
            continue
        out[(f.dataset, f.symbol)].append(f)
    return {k: sorted(v, key=lambda f: (f.first_day, f.period)) for k, v in sorted(out.items())}


def _build_symbol(
    cfg: DataConfig, dataset: DatasetName, symbol: str, files: Sequence[ArchiveFile]
) -> pl.DataFrame:
    start_ms, end_ms = window_ms(cfg)
    frames: list[pl.DataFrame] = []
    records: list[dict[str, object]] = []
    for f in files:
        res = normalize_file(dataset, (cfg.paths.raw_dir / f.key).read_bytes())
        frames.append(res.df)
        records.append(
            {
                "source": f.key,
                "rows_in": res.rows_in,
                "rows_out": res.df.height,
                "corrections": list(res.corrections),
            }
        )
    df = pl.concat(frames) if frames else pl.DataFrame()
    # pencere filtresi: kline'lar open_time'a göre, diğerleri ts'e göre
    tcol = "open_time_ms" if DATASETS[dataset].kind == "klines" else "ts_ms"
    df = df.filter((pl.col(tcol) >= start_ms) & (pl.col(tcol) < end_ms))
    keys = ["ts_ms", "agg_trade_id"] if dataset == "aggTrades" else ["ts_ms"]
    n = df.height
    df = df.unique(subset=keys, keep="first", maintain_order=True).sort(keys)
    overlap = n - df.height
    if overlap:
        log.warning("overlapping_rows_dropped", dataset=dataset, symbol=symbol, n=overlap)
    out = cfg.paths.normalized_dir / dataset / f"{symbol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "dataset": dataset,
        "symbol": symbol,
        "files": records,
        "kaynaklar_arasi_cakisan_satir_atildi": overlap,
    }
    out.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
    return df


def run_build(cfg: DataConfig) -> BuildSummary:
    """1) `cfg.datasets` için raw -> normalize -> parquet. 2) Repository'deki TÜM dataset'ler için
    doğrulama, tick, lifecycle, manifest ve raporları yeniden üretir
    (kısmi build manifestleri silmez).
    """
    repo = DataRepository(cfg.paths.parquet_dir)
    for (dataset, symbol), files in raw_files(cfg).items():
        df = _build_symbol(cfg, dataset, symbol, files)
        if df.height:
            repo.write_symbol(dataset, symbol, df)
    return finalize(cfg, repo)


def present_datasets(repo: DataRepository) -> tuple[DatasetName, ...]:
    return tuple(ds for ds in DATASETS if repo.symbols(ds))


def finalize(cfg: DataConfig, repo: DataRepository) -> BuildSummary:
    validator = DataValidator(cfg.validation)
    results: list[SymbolValidation] = []
    derived_ticks: dict[str, list[tuple[int, Decimal]]] = {}
    datasets = present_datasets(repo)
    for dataset in datasets:
        spec = DATASETS[dataset]
        for symbol in repo.symbols(dataset):
            df = repo.read_symbol(dataset, symbol)
            if spec.kind == "klines" and spec.interval_ms is not None:
                results.append(validator.validate_klines(df, symbol, dataset, spec.interval_ms))
            elif dataset == "fundingRate":
                results.append(validator.validate_funding(df, symbol))
            else:
                results.append(validator.validate_agg_trades(df, symbol))
            if dataset == "klines_1m":
                derived_ticks[symbol] = _derive_ticks(df, cfg.validation.tick_grid_min_fraction)

    for symbol in sorted(set(repo.symbols("klines_1m")) & set(repo.symbols("klines_5m"))):
        results.append(
            validator.crosscheck_5m(
                repo.read_symbol("klines_1m", symbol), repo.read_symbol("klines_5m", symbol), symbol
            )
        )

    store = ExchangeInfoStore.load(cfg.paths.raw_dir / "exchange_info", derived_ticks)
    approx: list[str] = []
    missing: list[str] = []
    for symbol in repo.symbols("klines_1m"):
        df = repo.read_symbol("klines_1m", symbol)
        first_ts = int(df["ts_ms"].min())  # type: ignore[arg-type]
        f = store.filters_at(symbol, first_ts)
        if f is None:
            missing.append(symbol)
            continue
        if f.approximate:
            approx.append(symbol)
        snap = store.filters_at(symbol, 2**62)  # en güncel snapshot (kesin)
        if snap is not None:
            issue = validator.check_tick(df, symbol, snap.tick_size)
            if issue is not None:
                results.append(
                    SymbolValidation("tick", symbol, df.height, first_ts, None, None, (issue,))
                )

    report = ValidationReport(tuple(results))
    lifecycle = build_lifecycle(repo, store)
    manifests = {ds: build_manifest(repo, ds) for ds in datasets}
    data_hash = _write_manifests(cfg, manifests, lifecycle, derived_ticks, report)
    summary = BuildSummary(report, manifests, lifecycle, data_hash, tuple(approx), tuple(missing))
    _write_build_summary(cfg, summary)
    return summary


def _derive_ticks(df: pl.DataFrame, min_fraction: float) -> list[tuple[int, Decimal]]:
    """Ay bazında fiyat grid'inden tick tahmini: [(ay başlangıcı ms, tick)]."""
    out: list[tuple[int, Decimal]] = []
    by_month = df.with_columns(
        pl.from_epoch("open_time_ms", time_unit="ms").dt.strftime("%Y-%m").alias("_m")
    )
    for (month,), part in by_month.group_by(["_m"], maintain_order=True):
        prices = part.select(pl.concat_list("open", "high", "low", "close").explode()).to_series()
        tick = derive_tick_size(prices.to_numpy(), min_fraction)
        if tick is not None:
            y, m = map(int, str(month).split("-"))
            out.append((_day_start_ms(date(y, m, 1)), tick))
    return sorted(out)


def _write_manifests(
    cfg: DataConfig,
    manifests: dict[DatasetName, DatasetManifest],
    lifecycle: SymbolLifecycle,
    derived_ticks: dict[str, list[tuple[int, Decimal]]],
    report: ValidationReport,
) -> str:
    d = cfg.paths.manifests_dir
    d.mkdir(parents=True, exist_ok=True)
    hashes = {ds: m.manifest_hash() for ds, m in sorted(manifests.items())}
    for m in manifests.values():
        m.write(d)
    lifecycle.write(d / "symbol_lifecycle.json")
    ticks = {s: [[ts, str(t)] for ts, t in v] for s, v in sorted(derived_ticks.items())}
    (d / "derived_ticks.json").write_text(json.dumps(ticks, indent=1), encoding="utf-8")
    exch = sorted(p.name for p in (cfg.paths.raw_dir / "exchange_info").glob("exchangeInfo_*.json"))
    combined = json.dumps(
        {"datasets": hashes, "exchange_info": exch},  # yalnız içerik; config hash ayrı
        sort_keys=True,
    )
    data_hash = hashlib.sha256(combined.encode()).hexdigest()
    index = {
        "data_manifest_hash": data_hash,
        "datasets": hashes,
        "exchange_info_snapshots": exch,
        "data_config_hash": cfg.config_hash(),
    }
    (d / "dataset_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    (d / "validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (d / "validation_report.md").write_text(report.summary_markdown(), encoding="utf-8")
    return data_hash


def _manifest_row(ds: str, m: DatasetManifest) -> str:
    los = [e.min_ts_ms for e in m.entries if e.min_ts_ms is not None]
    his = [e.max_ts_ms for e in m.entries if e.max_ts_ms is not None]
    first = fmt_ts(min(los)) if los else "-"
    last = fmt_ts(max(his)) if his else "-"
    return (
        f"| {ds} | {len(m.symbols)} | {len(m.entries)} | {m.total_rows:,} | {first} | {last} "
        f"| `{m.manifest_hash()[:16]}` |"
    )


def _write_build_summary(cfg: DataConfig, s: BuildSummary) -> None:
    lc = s.lifecycle
    # aktif sayımları 1m verisinin kapsadığı aralığın uçlarında
    # (son build çağrısının penceresi değil)
    start_ms = min((e.first_data_ms for e in lc.entries), default=0)
    end_ms = max((e.last_data_ms for e in lc.entries), default=0)
    k1 = s.manifests.get("klines_1m")
    n_repo = len(k1.symbols) if k1 is not None else 0  # repository'deki 1m sembolleri
    n_life = len(lc.entries)  # en az bir işlem görmüş bar'ı olanlar
    delisted = lc.delisted()
    n_approx, n_missing = len(s.approximate_filter_symbols), len(s.missing_filter_symbols)
    lines = [
        "# Veri Seti Özeti",
        "",
        f"- Son build çağrısı: {cfg.start_date} → {cfg.end_date} (UTC) · {', '.join(cfg.datasets)}",
        f"- `data_manifest_hash`: `{s.data_manifest_hash}`",
        f"- Sembol (1m kline): **{n_repo}** · pencerede işlem görmüş (lifecycle): **{n_life}** · "
        f"hiç işlem görmemiş (yalnız sentetik bar): **{n_repo - n_life}**",
        f"- Lifecycle içinde delist/listede olmayan: **{len(delisted)}**",
        f"- Veri başında ({fmt_ts(start_ms)}) aktif: {len(lc.active_symbols(start_ms))} · "
        f"veri sonunda ({fmt_ts(end_ms)}) aktif: {len(lc.active_symbols(end_ms))}",
        f"- Emir kuralları (ilk bar anında): kesin (snapshot) {n_repo - n_approx - n_missing}"
        f" · yaklaşık (türetilmiş tick) {n_approx} · hiç yok {n_missing}",
        "",
        "| Dataset | Sembol | Dosya | Satır | İlk | Son | manifest_hash |",
        "|---|---|---|---|---|---|---|",
        *(_manifest_row(ds, m) for ds, m in sorted(s.manifests.items())),
        "",
        f"Doğrulama: hatalı (dataset, sembol) sayısı **{s.report.error_count}** — "
        "ayrıntı `validation_report.md`.",
        "",
    ]
    if s.missing_filter_symbols:
        lines += [
            "## Emir kuralı bulunamayan semboller",
            "",
            "exchangeInfo snapshot'larında yoklar (çok önce delist). Faz 3'te bu semboller için "
            "emir üretilemez; universe'ten dışlanmaları gerekir.",
            "",
            ", ".join(s.missing_filter_symbols),
            "",
        ]
    if delisted:
        firsts = {e.symbol: e for e in lc.entries}
        lines += [
            "## Delist / listede olmayan semboller",
            "",
            "| Sembol | İlk veri | Son veri | Durum |",
            "|---|---|---|---|",
        ]
        for sym in delisted:
            e = firsts[sym]
            lines.append(
                f"| {sym} | {fmt_ts(e.first_data_ms)} | {fmt_ts(e.last_data_ms)} "
                f"| {e.status or 'bilinmiyor'} |"
            )
        lines.append("")
    (cfg.paths.manifests_dir / "build_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
