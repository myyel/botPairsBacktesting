"""PairSelector — formation verisinden pair universe üretimi (RAPOR §8.2, §8.3).

Bir refresh (cutoff) için adımlar:
1. Universe (UniverseFilter) sembollerinin uzun pencere log-getiri korelasyonu; `>= min_corr`
   olanlar korelasyona göre sıralanıp en fazla `max_coint_candidates` aday alınır.
2. Uzun pencerede Engle-Granger iki yönde; düşük p'li yön seçilir (A = bağımlı bacak, kullanıcı
   kararı Faz 2). `p_long < adf_p_long`, beta `beta_bounds` içinde.
3. Kısa pencerede aynı yönde EG: `p_short < adf_p_short` (iki pencerede birlikte geçerlilik).
4. Spread (uzun pencere alpha/beta ile): sıçrama filtresi — kısa pencerede sıçrama oranı
   > `max_jump_frac` ise red, değilse artışlar ±`jump_mad_k`·robust std'de kırpılır; sonra
   kısa pencerede OU half-life ∈ `half_life_minutes` (`half_life_method`: spread seviyesi veya
   EWMA sapması) ve Hurst <= `max_hurst`.
5. Uzun pencere `stability_subwindows` alt pencerede beta değişim katsayısı ve coint geçme oranı.
6. Beklenen brüt yakınsama (entry_z − exit_z)·sigma_eq/(1+|beta|) ile round-trip maliyet tahmini
   (costs.estimates — Faz 3 cost gate ile aynı formül); oran >= `min_edge_cost_ratio`.
7. PairQualityScore (§8.3) + NetworkDiversifier; en fazla `active_pairs` pair.
Seçilen pair'ler için Kalman başlangıç state'i (uzun pencere OLS → kısa pencerede filtre) üretilir.

half_life_method=ewma_deviation iken seçim kısa vadeli göreli değer sapmasını ölçer; spread
seviyesi yavaş (saatler) dönebilir veya unit-root olabilir. Bu kabul edilen risktir
(kullanıcı kararı B, Faz 2) ve `spurious_risk` ile işaretlenir.

Yalnız `(cutoff − long_window_days, cutoff]` verisi kullanılır (invariant #1, #2). Skor
normalizasyonu (yüzdelik rank) yalnız bu refresh'in adayları içindedir. Semboller sıralı işlenir,
eşitlikler pair adıyla çözülür (invariant #7).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import rankdata

from pairs_bt.config import StrategyConfig
from pairs_bt.costs.estimates import liquidity_scaled_slippage_bps, pair_round_trip_cost_bps
from pairs_bt.data.exchange_info import ExchangeInfoStore
from pairs_bt.data.lifecycle import SymbolLifecycle
from pairs_bt.data.synchronizer import SyncedPanel
from pairs_bt.domain.events import PairId
from pairs_bt.models.kalman import KalmanHedgeModel, KalmanState
from pairs_bt.research.diversifier import NetworkDiversifier, ScoredCandidate
from pairs_bt.research.ou import OUCalibrator, ewma_deviation
from pairs_bt.research.stats import (
    EGResult,
    engle_granger,
    hurst_exponent,
    ols_fit,
    winsorize_increments,
)
from pairs_bt.research.universe import UniverseFilter, UniverseResult
from pairs_bt.research.window import formation_window

FloatArr = npt.NDArray[np.float64]
_MINUTE_MS = 60_000
_INTERVAL_MINUTES = {"1m": 1.0, "5m": 5.0}
NAN = math.nan


class RejectReason(StrEnum):
    SELECTED = "selected"
    COINT_LONG = "coint_long"
    BETA_BOUNDS = "beta_bounds"
    COINT_SHORT = "coint_short"
    JUMP = "jump"  # kısa pencerede sıçrama oranı yüksek (çöküş fitili vb.)
    HALF_LIFE = "half_life"
    HURST = "hurst"
    EDGE_COST = "edge_cost"
    DIVERSIFICATION = "diversification"  # varlık kapasitesi dolu
    RANK_CUTOFF = "rank_cutoff"  # active_pairs doldu


@dataclass(frozen=True, slots=True)
class PairDiagnostics:
    cutoff_ms: int
    a: str  # bağımlı bacak: ln A = alpha + beta·ln B + S
    b: str
    corr: float  # uzun pencere log-getiri korelasyonu
    p_long: float
    p_short: float
    adf_long: float
    alpha: float  # uzun pencere OLS
    beta: float
    half_life_min: float  # half_life_method'a göre (seçimde kullanılan)
    half_life_level_min: float  # spread seviyesinin HL'i (bilgi)
    jump_frac: float  # kısa pencerede sıçrama bar'ı oranı
    hurst: float
    sigma_eq: float  # kısa pencerede half_life_method serisinin OU durağan std'si (log)
    beta_cv: float  # alt pencere beta'larının değişim katsayısı
    coint_stability: float  # alt pencerelerde p < adf_p_long oranı
    liquidity_usdt: float  # iki bacağın medyan 24s quote volume'ünün küçüğü
    expected_gross_bps: float  # brüt notional'a göre, işlem başı
    expected_cost_bps: float  # round-trip
    expected_net_bps: float
    edge_cost_ratio: float
    c_coint_stability: float
    c_mean_reversion: float
    c_net_edge: float
    c_liquidity: float
    c_beta_stability: float
    c_diversification: float
    score: float
    rank: int  # 0 = seçilmedi
    selected: bool
    reject_reason: RejectReason
    # Kabul edilen risk (kullanıcı kararı B, Faz 2): spread SEVİYESİNİN half-life'ı işlem
    # ufkunu (half_life_minutes üst sınırı) aşıyor → yavaş/olası unit-root seviye + hızlı
    # gürültü; EG bu yapıyı sahte cointegration olarak geçirebilir. Faz 3 bu bayrakla raporlar.
    spurious_risk: bool
    kalman_alpha: float
    kalman_beta: float
    kalman_p_aa: float
    kalman_p_ab: float
    kalman_p_bb: float
    kalman_beta_std: float  # kısa pencere boyunca Kalman beta'sının std'si

    @property
    def pair(self) -> PairId:
        return PairId(self.a, self.b)


_POLARS_TYPES: dict[str, type[pl.DataType]] = {
    "int": pl.Int64,
    "str": pl.String,
    "float": pl.Float64,
    "bool": pl.Boolean,
    "RejectReason": pl.String,
}
DIAGNOSTICS_SCHEMA = pl.Schema(
    {f.name: _POLARS_TYPES[str(f.type)] for f in fields(PairDiagnostics)}
)


@dataclass(frozen=True, slots=True)
class PairUniverse:
    cutoff_ms: int
    universe: UniverseResult
    n_symbols: int  # universe'ten geçen sembol
    n_pairs_total: int
    n_corr_pass: int  # korelasyon eşiğini geçen (üst sınır öncesi)
    diagnostics: tuple[PairDiagnostics, ...]  # seçilenler rank sırasıyla, sonra diğerleri

    @property
    def selected(self) -> tuple[PairDiagnostics, ...]:
        return tuple(d for d in self.diagnostics if d.selected)

    def to_frame(self) -> pl.DataFrame:
        rows = [{**asdict(d), "reject_reason": str(d.reject_reason)} for d in self.diagnostics]
        return pl.DataFrame(rows, schema=DIAGNOSTICS_SCHEMA, orient="row")


@dataclass(slots=True)
class _Candidate:
    a: int  # eligible indeksleri
    b: int
    corr: float
    values: dict[str, Any]
    reason: RejectReason | None = None


class PairSelector:
    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.ps = cfg.pair_selection
        self.ou = OUCalibrator()
        k = cfg.models.kalman
        self.kalman = KalmanHedgeModel(delta=k.delta, obs_var=k.obs_var)
        self.diversifier = NetworkDiversifier(
            self.ps.max_pairs_per_asset, self.ps.score_weights.diversification
        )

    # ------------------------------------------------------------------ public
    def select(self, panel: SyncedPanel, cutoff_ms: int, universe: UniverseResult) -> PairUniverse:
        ps = self.ps
        syms = universe.eligible
        n = len(syms)
        n_total = n * (n - 1) // 2
        if n < 2:
            return PairUniverse(cutoff_ms, universe, n, n_total, 0, ())
        interval = int(panel.ts_ms[1] - panel.ts_ms[0])
        bar_min = interval / _MINUTE_MS
        cols = np.array([panel.symbols.index(s) for s in syms])
        long_w = formation_window(panel.ts_ms, cutoff_ms, ps.long_window_days, interval)
        short_w = formation_window(panel.ts_ms, cutoff_ms, ps.short_window_days, interval)
        log_long = np.log(panel.close[long_w.rows][:, cols])
        n_short = short_w.stop - short_w.start
        log_short = log_long[log_long.shape[0] - n_short :]
        qv_short = panel.quote_volume[short_w.rows][:, cols]

        corr = _return_corr(log_long)
        iu, ju = np.triu_indices(n, k=1)
        cvals = corr[iu, ju]
        keep = np.isfinite(cvals) & (cvals >= ps.min_corr)
        order = sorted(
            np.flatnonzero(keep).tolist(), key=lambda k: (-cvals[k], syms[iu[k]], syms[ju[k]])
        )
        n_corr_pass = len(order)
        cands = [
            _Candidate(int(iu[k]), int(ju[k]), float(cvals[k]), {})
            for k in order[: ps.max_coint_candidates]
        ]

        leg_stats = _leg_stats(log_short, qv_short, bar_min, self._exec_minutes())
        for c in cands:
            self._evaluate(c, log_long, log_short, leg_stats, bar_min, universe, syms)
        passed = [c for c in cands if c.reason is None]
        self._score(passed)
        selections = self.diversifier.select(
            [ScoredCandidate(PairId(syms[c.a], syms[c.b]), c.values["base"]) for c in passed],
            ps.active_pairs,
        )
        by_pair = {s.pair: s for s in selections}
        for c in passed:
            pid = PairId(syms[c.a], syms[c.b])
            sel = by_pair.get(pid)
            if sel is None:
                blocked = self.diversifier.is_blocked(pid, selections)
                c.reason = RejectReason.DIVERSIFICATION if blocked else RejectReason.RANK_CUTOFF
                continue
            c.reason = RejectReason.SELECTED
            c.values.update(c_diversification=sel.diversification, score=sel.score, rank=sel.rank)
            c.values.update(self._kalman_init(log_long[:, c.a], log_long[:, c.b], n_short))
        diags = [self._diagnostics(cutoff_ms, c, syms) for c in cands]
        diags.sort(key=lambda d: (not d.selected, d.rank, d.a, d.b))
        return PairUniverse(cutoff_ms, universe, n, n_total, n_corr_pass, tuple(diags))

    # ------------------------------------------------------------------ adımlar
    def _evaluate(
        self,
        c: _Candidate,
        log_long: FloatArr,
        log_short: FloatArr,
        leg_stats: dict[str, FloatArr],
        bar_min: float,
        universe: UniverseResult,
        syms: tuple[str, ...],
    ) -> None:
        ps, sig, v = self.ps, self.cfg.signal, c.values
        fwd = engle_granger(log_long[:, c.a], log_long[:, c.b], ps.adf_lags)
        rev = engle_granger(log_long[:, c.b], log_long[:, c.a], ps.adf_lags)
        if _p(rev) < _p(fwd):
            c.a, c.b = c.b, c.a
            fwd = rev
        eg = fwd
        v.update(p_long=eg.p_value, adf_long=eg.adf_stat, alpha=eg.alpha, beta=eg.beta)
        if not _p(eg) < ps.adf_p_long:
            c.reason = RejectReason.COINT_LONG
            return
        lo, hi = ps.beta_bounds
        if not lo <= eg.beta <= hi:
            c.reason = RejectReason.BETA_BOUNDS
            return
        ya, xb = log_short[:, c.a], log_short[:, c.b]
        short = engle_granger(ya, xb, ps.adf_lags)
        v["p_short"] = short.p_value
        if not _p(short) < ps.adf_p_short:
            c.reason = RejectReason.COINT_SHORT
            return
        # spread uzun pencerede kurulur (EWMA ısınması için), tahminler kısa pencerede
        spread_long = log_long[:, c.a] - eg.alpha - eg.beta * log_long[:, c.b]
        clean_long, is_jump = winsorize_increments(spread_long, ps.jump_mad_k)
        start = spread_long.size - ya.size
        v["jump_frac"] = float(is_jump[start:].mean()) if ya.size else NAN
        if not v["jump_frac"] <= ps.max_jump_frac:
            c.reason = RejectReason.JUMP
            return
        spread = clean_long[start:]
        ou_level = self.ou.fit(spread, bar_min)
        if ps.half_life_method == "ewma_deviation":
            hl_bars = self.cfg.models.zscore.ewma_halflife_bars * self._exec_minutes() / bar_min
            ou = self.ou.fit(ewma_deviation(clean_long, hl_bars)[start:], bar_min)
        else:
            ou = ou_level
        v.update(
            half_life_min=ou.half_life_min,
            half_life_level_min=ou_level.half_life_min,
            sigma_eq=ou.sigma_eq,
        )
        hl_lo, hl_hi = ps.half_life_minutes
        if not (ou.valid and hl_lo <= ou.half_life_min <= hl_hi):
            c.reason = RejectReason.HALF_LIFE
            return
        h = hurst_exponent(spread, *ps.hurst_lags)
        v["hurst"] = h
        if not h <= ps.max_hurst:
            c.reason = RejectReason.HURST
            return
        # alt pencere stabilitesi
        subs = [
            engle_granger(log_long[idx, c.a], log_long[idx, c.b], ps.adf_lags)
            for idx in np.array_split(np.arange(log_long.shape[0]), ps.stability_subwindows)
        ]
        betas = np.array([s.beta for s in subs])
        mean_beta = float(np.mean(betas))
        v["beta_cv"] = (
            float(np.std(betas) / abs(mean_beta))
            if np.all(np.isfinite(betas)) and mean_beta != 0
            else math.inf
        )
        v["coint_stability"] = float(np.mean([_p(s) < ps.adf_p_long for s in subs]))
        # edge / maliyet
        beta = abs(eg.beta)
        gross_bps = 1e4 * (sig.entry_z - sig.exit_z) * ou.sigma_eq / (1.0 + beta)
        cost_bps = self._cost_bps(c, beta, leg_stats)
        liq_a, liq_b = (universe.get(syms[i]) for i in (c.a, c.b))
        v.update(
            expected_gross_bps=gross_bps,
            expected_cost_bps=cost_bps,
            expected_net_bps=gross_bps - cost_bps,
            edge_cost_ratio=gross_bps / cost_bps if cost_bps > 0 else math.inf,
            liquidity_usdt=min(
                liq_a.median_quote_volume_24h_usdt if liq_a else 0.0,
                liq_b.median_quote_volume_24h_usdt if liq_b else 0.0,
            ),
        )
        if not v["edge_cost_ratio"] >= ps.min_edge_cost_ratio:
            c.reason = RejectReason.EDGE_COST

    def _cost_bps(self, c: _Candidate, beta: float, leg_stats: dict[str, FloatArr]) -> float:
        costs = self.cfg.costs
        gross = self.ps.cost_est_gross_notional_usdt
        notional = {c.a: gross / (1.0 + beta), c.b: beta * gross / (1.0 + beta)}  # RAPOR §11.1
        slip = {}
        for leg, n_usdt in notional.items():
            sc = costs.slippage
            if sc.model == "fixed":
                slip[leg] = sc.base_bps + sc.stress_addon_bps
                continue
            slip[leg] = liquidity_scaled_slippage_bps(
                base_bps=sc.base_bps,
                k1=sc.k1,
                k2=sc.k2,
                order_notional_usdt=n_usdt,
                bar_quote_volume_usdt=float(leg_stats["exec_bar_qv"][leg]),
                volatility_1m_bps=float(leg_stats["vol_1m_bps"][leg]),
                stress_addon_bps=sc.stress_addon_bps,
            )
        return pair_round_trip_cost_bps(
            notional_a_usdt=notional[c.a],
            notional_b_usdt=notional[c.b],
            fee_bps=costs.fees.taker_bps,  # konservatif: her bacak taker
            slippage_a_bps=slip[c.a],
            slippage_b_bps=slip[c.b],
            leg_risk_reserve_bps=costs.leg_risk.reserve_bps,
        )

    def _score(self, passed: list[_Candidate]) -> None:
        """RAPOR §8.3 bileşenleri, [0, 1]. Sınırsız metrikler bu refresh içinde yüzdelik rank."""
        if not passed:
            return
        w = self.ps.score_weights
        hurst = _pct_rank([-c.values["hurst"] for c in passed])
        hl = _pct_rank([-c.values["half_life_min"] for c in passed])
        net = _pct_rank([c.values["expected_net_bps"] for c in passed])
        liq = _pct_rank([c.values["liquidity_usdt"] for c in passed])
        for i, c in enumerate(passed):
            v = c.values
            v["c_coint_stability"] = v["coint_stability"]
            v["c_mean_reversion"] = 0.5 * (hurst[i] + hl[i])
            v["c_net_edge"] = net[i]
            v["c_liquidity"] = liq[i]
            v["c_beta_stability"] = 1.0 / (1.0 + v["beta_cv"])
            v["base"] = (
                w.coint_stability * v["c_coint_stability"]
                + w.mean_reversion_quality * v["c_mean_reversion"]
                + w.expected_net_edge * v["c_net_edge"]
                + w.liquidity * v["c_liquidity"]
                + w.beta_stability * v["c_beta_stability"]
            )

    def _kalman_init(self, log_a: FloatArr, log_b: FloatArr, n_short: int) -> dict[str, float]:
        """Uzun pencerenin kısa pencere öncesi kısmında OLS → P0; kısa pencerede Kalman filtresi.

        Böylece Kalman state'i formation sonunda (cutoff) hazırdır; Faz 3 buradan devam eder.
        """
        split = log_a.size - n_short
        mask = np.isfinite(log_a[:split]) & np.isfinite(log_b[:split])
        y, x = log_a[:split][mask], log_b[:split][mask]
        fit = ols_fit(y, x)
        dof = y.size - 2
        out = dict.fromkeys(
            ("kalman_alpha", "kalman_beta", "kalman_p_aa", "kalman_p_ab", "kalman_p_bb"), NAN
        )
        out["kalman_beta_std"] = NAN
        if dof <= 0 or not math.isfinite(fit.beta):
            return out
        s2 = float(fit.resid @ fit.resid) / dof
        xtx_inv = np.linalg.inv(np.array([[y.size, x.sum()], [x.sum(), x @ x]]))
        p0 = s2 * xtx_inv
        init = KalmanState(fit.alpha, fit.beta, float(p0[0, 0]), float(p0[0, 1]), float(p0[1, 1]))
        path = self.kalman.run(log_a[split:], log_b[split:], init)
        f = path.final
        return {
            "kalman_alpha": f.alpha,
            "kalman_beta": f.beta,
            "kalman_p_aa": f.p_aa,
            "kalman_p_ab": f.p_ab,
            "kalman_p_bb": f.p_bb,
            "kalman_beta_std": float(np.std(path.beta)) if path.beta.size else NAN,
        }

    def _diagnostics(self, cutoff_ms: int, c: _Candidate, syms: tuple[str, ...]) -> PairDiagnostics:
        v = c.values
        reason = c.reason if c.reason is not None else RejectReason.RANK_CUTOFF
        selected = reason is RejectReason.SELECTED
        kw: dict[str, Any] = {
            f.name: v.get(f.name, NAN)
            for f in fields(PairDiagnostics)
            if str(f.type) == "float" and f.name != "corr"
        }
        if not selected:
            kw["score"] = v.get("base", NAN) if "base" in v else NAN
        level_hl = kw.get("half_life_level_min", NAN)
        spurious = bool(level_hl > self.ps.half_life_minutes[1])  # NaN -> False (ölçülmedi)
        return PairDiagnostics(
            cutoff_ms=cutoff_ms,
            a=syms[c.a],
            b=syms[c.b],
            corr=c.corr,
            rank=int(v.get("rank", 0)),
            selected=selected,
            reject_reason=reason,
            spurious_risk=spurious,
            **kw,
        )

    def _exec_minutes(self) -> float:
        return _INTERVAL_MINUTES[self.cfg.market.execution_interval]


# ---------------------------------------------------------------------- yardımcılar
def build_pair_universe(
    panel: SyncedPanel,
    cutoff_ms: int,
    lifecycle: SymbolLifecycle,
    exchange_info: ExchangeInfoStore,
    cfg: StrategyConfig,
) -> PairUniverse:
    """UniverseFilter + PairSelector (tek refresh)."""
    universe = UniverseFilter(cfg.market, cfg.pair_selection.long_window_days).evaluate(
        panel, cutoff_ms, lifecycle, exchange_info
    )
    return PairSelector(cfg).select(panel, cutoff_ms, universe)


def _p(eg: EGResult) -> float:
    """NaN p-değeri = en kötü (reddedilir)."""
    return eg.p_value if math.isfinite(eg.p_value) else math.inf


def _pct_rank(values: list[float]) -> FloatArr:
    """(0, 1] yüzdelik rank (ortalama rank / n); büyük değer = yüksek rank."""
    arr = np.asarray(values, dtype=np.float64)
    return np.asarray(rankdata(arr, method="average"), dtype=np.float64) / arr.size


def _return_corr(log_prices: FloatArr) -> FloatArr:
    """Log-getiri korelasyon matrisi. Stale (NaN) getiriler 0 sayılır (ucuz ön filtre)."""
    r = np.diff(log_prices, axis=0)
    r = np.where(np.isfinite(r), r, 0.0)
    if r.shape[0] < 2:
        return np.full((log_prices.shape[1],) * 2, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.corrcoef(r, rowvar=False), dtype=np.float64).reshape(
            log_prices.shape[1], log_prices.shape[1]
        )


def _leg_stats(
    log_short: FloatArr, qv_short: FloatArr, bar_min: float, exec_min: float
) -> dict[str, FloatArr]:
    """Kısa pencere bacak istatistikleri (slippage tahmini için, execution bar ölçeğinde).

    - exec_bar_qv: medyan bar quote volume × (exec_min / bar_min)
    - vol_1m_bps: bar log-getiri std'si × sqrt(1 / bar_min) × 1e4  (1 dakikalık ölçek)
    """
    r = np.diff(log_short, axis=0)
    with np.errstate(invalid="ignore"):
        vol = np.nanstd(r, axis=0) if r.shape[0] > 1 else np.full(log_short.shape[1], np.nan)
    qv = np.median(qv_short, axis=0) if qv_short.shape[0] else np.zeros(qv_short.shape[1])
    return {
        "exec_bar_qv": qv * (exec_min / bar_min),
        "vol_1m_bps": np.nan_to_num(vol, nan=0.0) * math.sqrt(1.0 / bar_min) * 1e4,
    }
