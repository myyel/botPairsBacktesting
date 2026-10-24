"""research.window — formation dilimi ve refresh takvimi (invariant #1, #2)."""

from __future__ import annotations

import numpy as np

from pairs_bt.research.window import DAY_MS, formation_window, refresh_schedule

M5 = 300_000


def test_window_includes_cutoff_bar_excludes_older_edge() -> None:
    ts = np.arange(1, 1_000, dtype=np.int64) * M5
    cutoff = 600 * M5
    w = formation_window(ts, cutoff, days=100 * M5 / DAY_MS, interval_ms=M5)
    sel = ts[w.rows]
    assert sel[-1] == cutoff
    assert sel[0] == cutoff - 99 * M5  # (cutoff − span, cutoff]
    assert sel.size == w.expected_bars == 100


def test_window_cutoff_between_bars() -> None:
    ts = np.arange(1, 100, dtype=np.int64) * M5
    w = formation_window(ts, 50 * M5 + 1, days=10 * M5 / DAY_MS, interval_ms=M5)
    assert ts[w.stop - 1] == 50 * M5


def test_window_before_data_is_empty() -> None:
    ts = np.arange(100, 200, dtype=np.int64) * M5
    w = formation_window(ts, 50 * M5, days=1.0, interval_ms=M5)
    assert w.stop - w.start == 0


def test_refresh_schedule_aligned_to_utc() -> None:
    h = 3_600_000
    start = 1_759_276_800_000 + 90 * 60_000  # 2025-10-01 01:30 UTC
    cut = refresh_schedule(start, start + 24 * h, 4)
    assert np.all(cut % (4 * h) == 0)
    assert cut[0] == 1_759_276_800_000 + 4 * h
    assert cut.size == 6
    assert np.all(np.diff(cut) == 4 * h)
