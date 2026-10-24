"""İstatistiksel testler: OLS, ADF, Engle-Granger, Hurst (RAPOR §8.2).

statsmodels `coint(trend="c", autolag=None)` ile sayısal olarak eşdeğerdir (bkz. test_stats) ama
binlerce aday çift için numpy ile hızlıdır. p-değeri MacKinnon (2010) yüzey yaklaşımıdır (N=2).
Dejenere girdilerde (sabit seri, çok kısa seri, tam colinear) istisna değil NaN döner; çağıran
taraf bunu red nedeni olarak işler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from statsmodels.tsa.adfvalues import mackinnonp

FloatArr = npt.NDArray[np.float64]

_MIN_OBS_PER_PARAM = 3  # EG için asgari gözlem: 3·(lags + 2)


@dataclass(frozen=True, slots=True)
class OLSFit:
    alpha: float
    beta: float
    resid: FloatArr


@dataclass(frozen=True, slots=True)
class EGResult:
    alpha: float
    beta: float
    adf_stat: float
    p_value: float
    n_obs: int


def ols_fit(y: FloatArr, x: FloatArr) -> OLSFit:
    """y = alpha + beta·x + e (sabitli OLS). x sabitse beta = NaN."""
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    sxx = float(dx @ dx)
    if sxx <= 0 or not math.isfinite(sxx):
        return OLSFit(math.nan, math.nan, np.full_like(y, np.nan))
    beta = float(dx @ (y - ym)) / sxx
    alpha = float(ym - beta * xm)
    return OLSFit(alpha, beta, y - alpha - beta * x)


def adf_tstat(e: FloatArr, lags: int) -> float:
    """Sabitsiz ADF t-istatistiği: Δe_t = γ·e_{t−1} + Σ_{i=1..lags} φ_i·Δe_{t−i} + u_t.

    statsmodels `adfuller(regression="n", maxlag=lags, autolag=None)` ile aynı örneklem.
    """
    de = np.diff(e)
    nobs = de.size - lags
    if lags < 0 or nobs <= lags + 1:
        return math.nan
    cols = [e[lags : lags + nobs]] + [de[lags - i : lags - i + nobs] for i in range(1, lags + 1)]
    xmat = np.column_stack(cols)
    yv = de[lags:]
    try:
        xtx_inv = np.linalg.inv(xmat.T @ xmat)
    except np.linalg.LinAlgError:
        return math.nan
    coef = xtx_inv @ (xmat.T @ yv)
    resid = yv - xmat @ coef
    s2 = float(resid @ resid) / (nobs - xmat.shape[1])
    se = math.sqrt(s2 * xtx_inv[0, 0]) if s2 > 0 else 0.0
    return float(coef[0]) / se if se > 0 else math.nan


def engle_granger(y: FloatArr, x: FloatArr, lags: int) -> EGResult:
    """İki adımlı Engle-Granger: y ~ alpha + beta·x, residual ADF, MacKinnon p (N=2, "c").

    y = bağımlı bacak (A), x = B. İki değişkenden birinde NaN olan satırlar atılır.
    """
    mask = np.isfinite(y) & np.isfinite(x)
    yv, xv = y[mask], x[mask]
    n = int(yv.size)
    nan = EGResult(math.nan, math.nan, math.nan, math.nan, n)
    if n < _MIN_OBS_PER_PARAM * (lags + 2):
        return nan
    fit = ols_fit(yv, xv)
    if not math.isfinite(fit.beta):
        return nan
    stat = adf_tstat(fit.resid, lags)
    if not math.isfinite(stat):
        return EGResult(fit.alpha, fit.beta, math.nan, math.nan, n)
    p = float(mackinnonp(stat, regression="c", N=2))
    return EGResult(fit.alpha, fit.beta, stat, p, n)


def hurst_exponent(x: FloatArr, min_lag: int, max_lag: int) -> float:
    """Gecikmeli fark yöntemi: std(x_{t+τ} − x_t) ∝ τ^H, τ ∈ [min_lag, max_lag].

    Random walk ≈ 0.5, mean-reverting < 0.5, trending > 0.5. Dejenere ise NaN.
    """
    if not 1 <= min_lag < max_lag:
        raise ValueError("1 <= min_lag < max_lag olmalı")
    if x.size < 2 * max_lag:
        return math.nan
    lags = np.arange(min_lag, max_lag + 1)
    tau = np.array([np.nanstd(x[k:] - x[:-k]) for k in lags])
    if not np.all(np.isfinite(tau)) or np.any(tau <= 0):
        return math.nan
    slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return float(slope)


# ---------------------------------------------------------------------- sıçrama filtresi
_MAD_TO_SIGMA = 1.4826  # normal dağılımda MAD -> std tutarlılık sabiti (matematiksel, eşik değil)


def robust_scale(x: FloatArr) -> float:
    """MAD tabanlı robust std tahmini (NaN'lar atlanır). Boş seri NaN, sabit seri 0."""
    v = x[np.isfinite(x)]
    if v.size == 0:
        return math.nan
    return float(_MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def winsorize_increments(x: FloatArr, k: float) -> tuple[FloatArr, npt.NDArray[np.bool_]]:
    """Seri artışlarını ±k·robust_scale(artışlar) aralığına kırpıp seviyeyi yeniden kurar.

    Çöküş fitilleri gibi tek bar'lık sıçramaların OU/Hurst/sigma tahminini bozmasını engeller
    (Faz 2 kullanıcı kararı). `is_jump[t]`: x[t − 1] → x[t] artışı kırpıldı (NaN'lar atlanarak
    ardışık gözlemler arasında). NaN pozisyonları korunur; ilk gözlem değişmez.
    Robust ölçek 0/NaN ise seri değişmeden döner.
    """
    if k <= 0:
        raise ValueError("k pozitif olmalı")
    out = x.astype(np.float64, copy=True)
    is_jump = np.zeros(x.size, dtype=bool)
    idx = np.flatnonzero(np.isfinite(x))
    if idx.size < 2:
        return out, is_jump
    v = x[idx]
    dv = np.diff(v)
    scale = robust_scale(dv)
    if not scale > 0:
        return out, is_jump
    lim = k * scale
    is_jump[idx[1:]] = np.abs(dv) > lim
    out[idx] = v[0] + np.concatenate(([0.0], np.cumsum(np.clip(dv, -lim, lim))))
    return out, is_jump
