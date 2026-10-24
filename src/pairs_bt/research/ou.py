"""OUCalibrator — spread'in Ornstein-Uhlenbeck parametreleri (RAPOR §3.3).

AR(1): s_{t+1} = c + φ·s_t + e.  Geçerli yalnız 0 < φ < 1:
    kappa (bar başına) = −ln φ,  HalfLife = ln2 / kappa  (bar → dakika: × bar_minutes)
    mu = c / (1 − φ),  durağan std sigma_eq = std(e) / sqrt(1 − φ²)
NaN içeren ardışık çiftler atlanır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pairs_bt.models.zscore import ZScoreModel

FloatArr = npt.NDArray[np.float64]

_MIN_PAIRS = 3


@dataclass(frozen=True, slots=True)
class OUParams:
    phi: float
    kappa_per_bar: float
    mu: float
    sigma_eq: float
    half_life_min: float
    valid: bool


_INVALID = OUParams(math.nan, math.nan, math.nan, math.nan, math.nan, False)


class OUCalibrator:
    def fit(self, spread: FloatArr, bar_minutes: float) -> OUParams:
        if bar_minutes <= 0:
            raise ValueError("bar_minutes pozitif olmalı")
        prev, nxt = spread[:-1], spread[1:]
        mask = np.isfinite(prev) & np.isfinite(nxt)
        prev, nxt = prev[mask], nxt[mask]
        if prev.size < _MIN_PAIRS:
            return _INVALID
        pm, nm = prev.mean(), nxt.mean()
        dp = prev - pm
        var = float(dp @ dp)
        if var <= 0:
            return _INVALID
        phi = float(dp @ (nxt - nm)) / var
        if not 0.0 < phi < 1.0:
            return _INVALID
        c = nm - phi * pm
        resid = nxt - c - phi * prev
        kappa = -math.log(phi)
        return OUParams(
            phi=phi,
            kappa_per_bar=kappa,
            mu=float(c / (1.0 - phi)),
            sigma_eq=float(resid.std() / math.sqrt(1.0 - phi * phi)),
            half_life_min=math.log(2.0) / kappa * bar_minutes,
            valid=True,
        )


def ewma_deviation(spread: FloatArr, halflife_bars: float) -> FloatArr:
    """Spread'in causal EWMA ortalamasından sapması: d_t = S_t − EWMA_mean_t (RAPOR §3.2).

    Sinyal Z = d / EWMA_std olduğundan, half-life bu seri üzerinde ölçülünce sinyalin trade ettiği
    geri dönüşü ölçer. DİKKAT: saf random walk için de d bir AR(1)'dir ve HL = EWMA half-life'ı
    olur; bu yüzden half-life üst sınırı EWMA half-life'ından küçük olmalıdır (config validator).
    """
    return spread - ZScoreModel(halflife_bars).run(spread).mean
