"""Formation penceresi dilimleme ve refresh takvimi (RAPOR §12.1, CLAUDE.md invariant #1, #2).

Bir formation penceresi `(cutoff − days, cutoff]` aralığındaki bar'lardır: `ts_ms` bar'ın kapanış
anıdır, dolayısıyla `ts_ms == cutoff` olan bar cutoff anında tamamlanmıştır ve kullanılabilir.
`ts_ms > cutoff` olan hiçbir bar formation'a girmez.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DAY_MS = 86_400_000
HOUR_MS = 3_600_000


@dataclass(frozen=True, slots=True)
class FormationWindow:
    start: int  # panel satır indeksi (dahil)
    stop: int  # panel satır indeksi (hariç); stop-1 = cutoff'taki/öncesindeki son bar
    expected_bars: int  # pencere tam dolu olsaydı bar sayısı

    @property
    def rows(self) -> slice:
        return slice(self.start, self.stop)


def formation_window(
    ts_ms: npt.NDArray[np.int64], cutoff_ms: int, days: float, interval_ms: int
) -> FormationWindow:
    """`(cutoff − days, cutoff]` aralığındaki satırlar. `ts_ms` artan sıralı olmalı."""
    span = round(days * DAY_MS)
    lo = int(np.searchsorted(ts_ms, cutoff_ms - span, side="right"))
    hi = int(np.searchsorted(ts_ms, cutoff_ms, side="right"))
    return FormationWindow(lo, hi, span // interval_ms)


def refresh_schedule(start_ms: int, end_ms: int, refresh_hours: int) -> npt.NDArray[np.int64]:
    """`[start_ms, end_ms)` içinde UTC epoch'a hizalı refresh cutoff'ları (00/04/08/... UTC)."""
    step = refresh_hours * HOUR_MS
    first = -(-start_ms // step) * step
    return np.arange(first, end_ms, step, dtype=np.int64)
