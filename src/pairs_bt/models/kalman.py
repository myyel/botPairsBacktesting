"""KalmanHedgeModel — dinamik alpha/beta (RAPOR §3.1).

Gözlem: ln A_t = alpha_t + beta_t · ln B_t + e_t,  e_t ~ N(0, obs_var)
Durum:  [alpha, beta]_t = [alpha, beta]_{t-1} + w_t,  w_t ~ N(0, q·I),  q = delta / (1 − delta)

Causal ve artımlı: t'deki çıktı yalnız ≤ t gözlemlerine bağlıdır. İnovasyon, t gözlemi işlenmeden
ÖNCEKİ (prior) state ile hesaplanır. NaN gözlem (stale bar) yalnız predict adımı uygular.
2×2 kovaryans skaler olarak tutulur (hot loop'ta küçük numpy matrisi yok, CLAUDE.md §5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class KalmanState:
    alpha: float
    beta: float
    p_aa: float  # Var(alpha)
    p_ab: float  # Cov(alpha, beta)
    p_bb: float  # Var(beta)


@dataclass(frozen=True, slots=True)
class KalmanStep:
    state: KalmanState  # posterior (gözlem işlendikten sonra)
    innovation: float  # y − (alpha_prior + beta_prior · x); NaN gözlemde NaN
    innovation_var: float


@dataclass(frozen=True, slots=True)
class KalmanPath:
    alpha: FloatArr  # posterior, (T,)
    beta: FloatArr
    innovation: FloatArr
    innovation_var: FloatArr
    final: KalmanState


class KalmanHedgeModel:
    def __init__(self, delta: float, obs_var: float) -> None:
        if not 0 < delta < 1:
            raise ValueError("delta (0, 1) aralığında olmalı")
        if obs_var <= 0:
            raise ValueError("obs_var pozitif olmalı")
        self.delta = delta
        self.obs_var = obs_var
        self.q = delta / (1.0 - delta)

    def step(self, s: KalmanState, y: float, x: float) -> KalmanStep:
        p_aa, p_ab, p_bb = s.p_aa + self.q, s.p_ab, s.p_bb + self.q
        if math.isnan(y) or math.isnan(x):
            return KalmanStep(KalmanState(s.alpha, s.beta, p_aa, p_ab, p_bb), math.nan, math.nan)
        e = y - (s.alpha + s.beta * x)
        var = p_aa + 2.0 * x * p_ab + x * x * p_bb + self.obs_var
        k_a = (p_aa + x * p_ab) / var
        k_b = (p_ab + x * p_bb) / var
        post = KalmanState(
            alpha=s.alpha + k_a * e,
            beta=s.beta + k_b * e,
            p_aa=p_aa - k_a * k_a * var,
            p_ab=p_ab - k_a * k_b * var,
            p_bb=p_bb - k_b * k_b * var,
        )
        return KalmanStep(post, e, var)

    def run(self, y: FloatArr, x: FloatArr, init: KalmanState) -> KalmanPath:
        if y.shape != x.shape or y.ndim != 1:
            raise ValueError("y ve x aynı uzunlukta 1-boyutlu diziler olmalı")
        n = y.size
        alpha, beta = np.empty(n), np.empty(n)
        innov, ivar = np.empty(n), np.empty(n)
        state = init
        for t, (yt, xt) in enumerate(zip(y.tolist(), x.tolist(), strict=True)):
            st = self.step(state, yt, xt)
            state = st.state
            alpha[t], beta[t] = state.alpha, state.beta
            innov[t], ivar[t] = st.innovation, st.innovation_var
        return KalmanPath(alpha, beta, innov, ivar, state)
