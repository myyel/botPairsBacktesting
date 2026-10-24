"""Pair universe run'ı için Türkçe özet raporu (`summary.md`)."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pairs_bt.research.pair_selector import RejectReason
from pairs_bt.research.universe import UniverseReason
from pairs_bt.research.universe_run import UniverseRunResult

_REJECT_TR = {
    RejectReason.SELECTED: "seçildi",
    RejectReason.COINT_LONG: "uzun pencere coint (EG) geçmedi",
    RejectReason.BETA_BOUNDS: "beta sınır dışı",
    RejectReason.COINT_SHORT: "kısa pencere coint geçmedi",
    RejectReason.JUMP: "kısa pencerede sıçrama oranı yüksek",
    RejectReason.HALF_LIFE: "half-life aralık dışı / OU geçersiz",
    RejectReason.HURST: "Hurst > eşik",
    RejectReason.EDGE_COST: "beklenen edge / maliyet yetersiz",
    RejectReason.DIVERSIFICATION: "varlık kapasitesi dolu",
    RejectReason.RANK_CUTOFF: "active_pairs doldu",
}
_UNI_TR = {
    UniverseReason.OK: "uygun",
    UniverseReason.EXCLUDED: "config ile hariç",
    UniverseReason.NOT_ACTIVE: "cutoff'ta aktif değil",
    UniverseReason.CONTRACT_TYPE: "kontrat tipi",
    UniverseReason.NO_FILTERS: "emir kuralı yok",
    UniverseReason.LISTING_AGE: "listing yaşı kısa",
    UniverseReason.LOW_COVERAGE: "kapsama düşük",
    UniverseReason.ZERO_TRADE: "işlemsiz bar oranı yüksek",
    UniverseReason.LOW_VOLUME: "hacim düşük",
}


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _pct(values: pl.Series | np.ndarray, qs: tuple[int, ...] = (10, 50, 90)) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan")] * len(qs)
    return [float(np.percentile(arr, q)) for q in qs]


def _fmt(x: float, nd: int = 2) -> str:
    return "-" if not np.isfinite(x) else f"{x:,.{nd}f}"


def turnover(selected: pl.DataFrame) -> pl.DataFrame:
    """Ardışık refresh'lerde önceki seçimin korunan oranı (|S_t ∩ S_{t-1}| / |S_{t-1}|)."""
    sets: list[tuple[int, set[str]]] = []
    if selected.height:
        grouped = selected.group_by("cutoff_ms", maintain_order=True).agg(
            (pl.col("a") + "|" + pl.col("b")).alias("p")
        )
        sets = [(int(c), set(p)) for c, p in grouped.sort("cutoff_ms").iter_rows()]
    rows = [
        (cur[0], len(cur[1] & prev[1]) / len(prev[1]) if prev[1] else float("nan"))
        for prev, cur in itertools.pairwise(sets)
    ]
    return pl.DataFrame(rows, schema={"cutoff_ms": pl.Int64, "retained": pl.Float64}, orient="row")


def write_summary(result: UniverseRunResult, out_dir: Path, meta: Mapping[str, Any]) -> Path:
    s = result.refresh_summary
    d = result.diagnostics
    sel = result.selected
    lines = ["# Pair Universe Özeti (Faz 2)", ""]
    lines += [f"- {k}: `{v}`" for k, v in meta.items()]
    if s.height == 0:
        lines += ["", "Hiç refresh işlenmedi."]
        path = out_dir / "summary.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    lines += [
        f"- refresh sayısı: **{s.height}** ({_utc(int(s['cutoff_ms'].min()))} → "  # type: ignore[arg-type]
        f"{_utc(int(s['cutoff_ms'].max()))})",  # type: ignore[arg-type]
        "",
        "## Refresh başına sayılar (P10 / P50 / P90)",
        "",
        "| Metrik | P10 | P50 | P90 |",
        "|---|---|---|---|",
    ]
    for col, label in [
        ("n_symbols", "universe sembol"),
        ("n_pairs_total", "olası pair"),
        ("n_corr_pass", "korelasyon ≥ eşik"),
        ("n_candidates", "coint test edilen aday"),
        ("n_selected", "seçilen pair"),
    ]:
        p = _pct(s[col])
        lines.append(f"| {label} | {_fmt(p[0], 0)} | {_fmt(p[1], 0)} | {_fmt(p[2], 0)} |")
    zero = int((s["n_selected"] == 0).sum())
    lines += ["", f"Hiç pair seçilemeyen refresh: **{zero}** / {s.height}", ""]

    lines += ["## Universe red nedenleri (refresh başına medyan sembol)", ""]
    lines += ["| Neden | Medyan |", "|---|---|"]
    for u in UniverseReason:
        lines.append(f"| {_UNI_TR[u]} | {_fmt(float(s[f'uni_{u.value}'].median()), 0)} |")  # type: ignore[arg-type]

    lines += ["", "## Aday red nedenleri (tüm refresh'ler)", ""]
    total = max(int(s["n_candidates"].sum()), 1)
    lines += ["| Neden | Adet | Oran |", "|---|---|---|"]
    for r in RejectReason:
        n = int(s[f"rej_{r.value}"].sum())
        lines.append(f"| {_REJECT_TR[r]} | {n:,} | %{100 * n / total:.2f} |")

    if sel.height:
        lines += ["", "## Seçilen pair metrikleri (P10 / P50 / P90)", ""]
        lines += ["| Metrik | P10 | P50 | P90 |", "|---|---|---|---|"]
        for col, label in [
            ("half_life_min", "half-life (dk, seçim yöntemi)"),
            ("half_life_level_min", "half-life (dk, spread seviyesi)"),
            ("jump_frac", "sıçrama bar oranı"),
            ("hurst", "Hurst"),
            ("p_long", "EG p (uzun)"),
            ("p_short", "EG p (kısa)"),
            ("beta", "beta"),
            ("corr", "getiri korelasyonu"),
            ("expected_gross_bps", "beklenen brüt (bps)"),
            ("expected_cost_bps", "round-trip maliyet (bps)"),
            ("expected_net_bps", "beklenen net (bps)"),
            ("edge_cost_ratio", "edge / maliyet"),
            ("coint_stability", "coint stabilitesi"),
            ("beta_cv", "beta CV"),
            ("score", "skor"),
        ]:
            p = _pct(sel[col])
            nd = 4 if col in ("p_long", "p_short", "jump_frac") else 2
            lines.append(f"| {label} | {_fmt(p[0], nd)} | {_fmt(p[1], nd)} | {_fmt(p[2], nd)} |")
        n_sp = int(sel["spurious_risk"].sum())
        lines += [
            "",
            f"**spurious_risk** (spread seviyesinin HL'i işlem ufkunu aşıyor; kabul edilen risk): "
            f"{n_sp:,} / {sel.height:,} seçim (%{100 * n_sp / sel.height:.1f})",
        ]
        to = turnover(sel)
        if to.height:
            tp = _pct(to["retained"])
            lines += [
                "",
                f"Ardışık refresh'te korunan pair oranı (P10/P50/P90): "
                f"{_fmt(tp[0])} / {_fmt(tp[1])} / {_fmt(tp[2])}",
            ]
        freq = (
            sel.with_columns((pl.col("a") + "|" + pl.col("b")).alias("pair"))
            .group_by("pair")
            .len()
            .sort(["len", "pair"], descending=[True, False])
            .head(20)
        )
        lines += ["", "## En sık seçilen 20 pair", "", "| Pair | Refresh | Oran |", "|---|---|---|"]
        for pair, n in freq.iter_rows():
            lines.append(f"| {pair} | {n} | %{100 * n / s.height:.1f} |")
        assets = (
            pl.concat([sel.select(pl.col("a").alias("s")), sel.select(pl.col("b").alias("s"))])
            .group_by("s")
            .len()
            .sort(["len", "s"], descending=[True, False])
            .head(15)
        )
        lines += ["", "## En sık görülen 15 varlık", "", "| Varlık | Pair-refresh |", "|---|---|"]
        lines += [f"| {a} | {n} |" for a, n in assets.iter_rows()]

    lines += [
        "",
        "## Notlar",
        "",
        "- Seçim yalnız `(cutoff − 28 gün, cutoff]` verisini kullanır (invariant #1, #2); skor "
        "ağırlıkları RAPOR §8.3 başlangıç değerleridir, test dönemine göre optimize edilmemiştir.",
        "- Maliyet tahmini konservatiftir: her bacak taker fee + liquidity-scaled slippage + "
        "leg-risk rezervi; funding dahil değildir (Faz 3 FundingModel).",
        "- Pencere içindeki tüm emir kuralları `approximate` (exchange-info snapshot'ları "
        "pencereden sonra alındı, CLAUDE.md §7.2).",
        f"- Tüm adaylar (red nedenleriyle): `diagnostics.parquet` ({d.height:,} satır).",
    ]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
