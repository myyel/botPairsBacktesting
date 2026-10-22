"""TimeSynchronizer — sembolleri ortak bar grid'ine hizalar (RAPOR §7, CLAUDE.md §7, invariant #1).

- Yalnız forward-fill; en fazla `max_ffill_bars` bar. Aşılırsa (veya ilk bar'dan önce) hücre `stale`
  sayılır ve fiyat NaN kalır — o sembol için sinyal üretilmez. bfill / center / gelecek verisi YOK.
- ffill edilen bar'larda hacim 0'dır (o dakikada işlem gözlenmedi).
- Çıktı numpy dizileridir (hot loop'ta DataFrame yok, CLAUDE.md §5). Semboller sıralıdır (#7).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SyncedPanel:
    ts_ms: npt.NDArray[np.int64]  # (T,) bar kapanış zamanları
    symbols: tuple[str, ...]  # (N,)
    close: FloatArr  # (T, N); stale hücrelerde NaN
    quote_volume: FloatArr  # (T, N); gözlenmeyen bar'da 0
    observed: BoolArr  # (T, N) gerçek bar var
    stale: BoolArr  # (T, N) ffill limiti aşıldı / henüz veri yok


class TimeSynchronizer:
    def __init__(self, interval_ms: int, max_ffill_bars: int) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms pozitif olmalı")
        if max_ffill_bars < 0:
            raise ValueError("max_ffill_bars negatif olamaz")
        self.interval_ms = interval_ms
        self.max_ffill_bars = max_ffill_bars

    def grid(self, start_ms: int, end_ms: int) -> npt.NDArray[np.int64]:
        """`[start_ms, end_ms)` içindeki, interval'e hizalı bar kapanış zamanları."""
        first = -(-start_ms // self.interval_ms) * self.interval_ms
        return np.arange(first, end_ms, self.interval_ms, dtype=np.int64)

    def align(self, frames: Mapping[str, pl.DataFrame], start_ms: int, end_ms: int) -> SyncedPanel:
        grid = self.grid(start_ms, end_ms)
        symbols = tuple(sorted(frames))
        shape = (grid.size, len(symbols))
        close = np.full(shape, np.nan)
        qvol = np.zeros(shape)
        observed = np.zeros(shape, dtype=bool)
        stale = np.ones(shape, dtype=bool)
        if grid.size == 0:
            return SyncedPanel(grid, symbols, close, qvol, observed, stale)
        rows = np.arange(grid.size)
        for j, sym in enumerate(symbols):
            df = frames[sym]
            ts = df["ts_ms"].to_numpy().astype(np.int64)
            if np.any(ts % self.interval_ms != 0):
                raise ValueError(f"{sym}: bar'lar {self.interval_ms} ms grid'ine hizalı değil")
            in_range = (ts >= grid[0]) & (ts <= grid[-1])
            pos = ((ts[in_range] - grid[0]) // self.interval_ms).astype(np.intp)
            obs_close = np.full(grid.size, np.nan)
            obs_close[pos] = df["close"].to_numpy()[in_range]
            qvol[pos, j] = df["quote_volume"].to_numpy()[in_range]
            obs = np.zeros(grid.size, dtype=bool)
            obs[pos] = True
            # son gözlem indeksi (yalnız geçmiş): causal running max
            last = np.maximum.accumulate(np.where(obs, rows, -1))
            valid = (last >= 0) & (rows - last <= self.max_ffill_bars)
            close[valid, j] = obs_close[last[valid]]
            observed[:, j] = obs
            stale[:, j] = ~valid
        return SyncedPanel(grid, symbols, close, qvol, observed, stale)
