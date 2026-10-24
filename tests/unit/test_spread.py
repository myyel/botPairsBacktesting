"""SpreadModel (RAPOR §3.1): S = ln A − alpha − beta·ln B."""

from __future__ import annotations

import numpy as np
import pytest

from pairs_bt.models.spread import log_spread


def test_scalar() -> None:
    assert log_spread(2.0, 1.5, 0.1, 1.2) == pytest.approx(2.0 - 0.1 - 1.2 * 1.5)


def test_vectorized_with_dynamic_params() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 1.0, 1.5])
    alpha = np.array([0.0, 0.1, 0.2])
    beta = np.array([1.0, 1.1, 1.2])
    np.testing.assert_allclose(log_spread(a, b, alpha, beta), a - alpha - beta * b)


def test_nan_propagates() -> None:
    out = log_spread(np.array([1.0, np.nan]), np.array([1.0, 1.0]), 0.0, 1.0)
    assert out[0] == 0.0
    assert np.isnan(out[1])
