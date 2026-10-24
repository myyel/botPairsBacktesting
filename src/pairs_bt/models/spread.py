"""SpreadModel — log-fiyat spread'i (RAPOR §3.1): S_t = ln P_A − alpha_t − beta_t · ln P_B."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.float64]
FloatLike = float | FloatArr


def log_spread(log_a: FloatLike, log_b: FloatLike, alpha: FloatLike, beta: FloatLike) -> FloatLike:
    """Skaler veya eleman bazlı (dinamik alpha/beta) spread. NaN yayılır."""
    return log_a - alpha - beta * log_b
