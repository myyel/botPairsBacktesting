"""Run provenance (CLAUDE.md §8): run_id, git commit + dirty bayrağı, paket sürümleri.

`pairs_bt` içinde bağımlılığı yoktur; script'ler ve (Faz 3) backtest orkestratörü kullanır.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

DEFAULT_PACKAGES = ("numpy", "pandas", "polars", "scipy", "statsmodels", "pyarrow", "pydantic")


def make_run_id(config_hash: str, now: datetime | None = None) -> str:
    """`<UTC tarih>_<config_hash[:8]>` (CLAUDE.md §8). Aynı gün aynı config = aynı run_id."""
    ts = now or datetime.now(UTC)
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("now tz-aware olmalı")
    return f"{ts.astimezone(UTC):%Y%m%d}_{config_hash[:8]}"


def git_info(root: Path) -> tuple[str | None, bool]:
    """(commit sha, dirty). Git yoksa (None, True) — bilinmeyen durum kirli sayılır."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, True
    return sha, bool(status.strip())


def package_versions(names: Sequence[str] = DEFAULT_PACKAGES) -> dict[str, str]:
    out: dict[str, str] = {}
    for n in sorted(names):
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "yok"
    return out
