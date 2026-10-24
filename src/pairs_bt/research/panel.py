"""Formation paneli: repository'den hizalı numpy panel + memmap önbellek.

Panel bir kez yüklenir; walk-forward boyunca her refresh yalnız `(cutoff − pencere, cutoff]`
satırlarını dilimler (research.window). Yükleme sembol sembol yapılır (tüm evren için tek seferde
DataFrame tutulmaz). Önbellek `np.save` dosyalarıdır; `open_panel` bunları `mmap_mode="r"` ile
açar, böylece paralel worker'lar aynı sayfaları paylaşır.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pairs_bt.config import DatasetName
from pairs_bt.data.repository import DataRepository
from pairs_bt.data.synchronizer import SyncedPanel, TimeSynchronizer

_ARRAYS = ("ts_ms", "close", "quote_volume", "observed", "stale")


def load_panel(
    repo: DataRepository,
    dataset: DatasetName,
    symbols: Sequence[str],
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    max_ffill_bars: int,
) -> SyncedPanel:
    """`[start_ms, end_ms)` grid'inde, semboller sıralı. Verisi olmayan sembol tamamen stale."""
    sync = TimeSynchronizer(interval_ms, max_ffill_bars)
    grid = sync.grid(start_ms, end_ms)
    syms = tuple(sorted(set(symbols)))
    shape = (grid.size, len(syms))
    close = np.full(shape, np.nan)
    qvol = np.zeros(shape)
    observed = np.zeros(shape, dtype=bool)
    stale = np.ones(shape, dtype=bool)
    for j, sym in enumerate(syms):
        df = repo.read_range(dataset, sym, start_ms, end_ms)
        one = sync.align({sym: df}, start_ms, end_ms)
        close[:, j] = one.close[:, 0]
        qvol[:, j] = one.quote_volume[:, 0]
        observed[:, j] = one.observed[:, 0]
        stale[:, j] = one.stale[:, 0]
    return SyncedPanel(grid, syms, close, qvol, observed, stale)


def save_panel(panel: SyncedPanel, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in _ARRAYS:
        np.save(directory / f"{name}.npy", np.ascontiguousarray(getattr(panel, name)))
    (directory / "symbols.json").write_text(json.dumps(list(panel.symbols)), encoding="utf-8")
    return directory


def open_panel(directory: Path) -> SyncedPanel:
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in _ARRAYS}
    symbols = tuple(json.loads((directory / "symbols.json").read_text(encoding="utf-8")))
    return SyncedPanel(symbols=symbols, **arrays)


def panel_cached(directory: Path) -> bool:
    return (directory / "symbols.json").exists() and all(
        (directory / f"{n}.npy").exists() for n in _ARRAYS
    )
