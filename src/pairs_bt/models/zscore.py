"""ZScoreModel — causal EWMA z-score (RAPOR §3.2): Z_t = (S_t − EWMA_mean) / EWMA_std.

EWMA `adjust=False` özyinelemesi (pandas `ewm(halflife, adjust=False)` ile aynı; std `bias=True`):
    a = 1 − exp(−ln2 / halflife),  d = x − m_{t−1}
    m_t = m_{t−1} + a·d,  v_t = (1 − a)·(v_{t−1} + a·d²)
İlk gözlemde v = 0 → z = NaN. std = 0 iken z = NaN. NaN girdi state'i değiştirmez, z = NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EwmaState:
    mean: float
    var: float
    n: int  # işlenen (NaN olmayan) gözlem sayısı


@dataclass(frozen=True, slots=True)
class ZScorePath:
    mean: FloatArr
    std: FloatArr
    z: FloatArr


class ZScoreModel:
    def __init__(self, halflife_bars: float) -> None:
        if halflife_bars <= 0:
            raise ValueError("halflife_bars pozitif olmalı")
        self.halflife_bars = halflife_bars
        self.alpha = 1.0 - math.exp(-math.log(2.0) / halflife_bars)

    def step(self, state: EwmaState | None, x: float) -> tuple[EwmaState | None, float]:
        if math.isnan(x):
            return state, math.nan
        if state is None:
            new = EwmaState(x, 0.0, 1)
        else:
            d = x - state.mean
            a = self.alpha
            new = EwmaState(state.mean + a * d, (1.0 - a) * (state.var + a * d * d), state.n + 1)
        z = (x - new.mean) / math.sqrt(new.var) if new.var > 0 else math.nan
        return new, z

    def run(self, x: FloatArr) -> ZScorePath:
        n = x.size
        mean, std, z = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        state: EwmaState | None = None
        for t, v in enumerate(x.tolist()):
            state, z[t] = self.step(state, v)
            if state is not None:
                mean[t], std[t] = state.mean, math.sqrt(state.var)
        return ZScorePath(mean, std, z)
