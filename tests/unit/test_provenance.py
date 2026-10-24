from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from pairs_bt.provenance import git_info, make_run_id, package_versions

ROOT = Path(__file__).resolve().parents[2]


def test_run_id_format() -> None:
    assert make_run_id("abcdef0123456789", datetime(2026, 10, 24, 23, 0, tzinfo=UTC)) == (
        "20261024_abcdef01"
    )


def test_run_id_requires_tz() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        make_run_id("abc", datetime(2026, 1, 1))  # noqa: DTZ001


def test_git_info_in_repo() -> None:
    sha, dirty = git_info(ROOT)
    assert sha is not None
    assert len(sha) == 40
    assert isinstance(dirty, bool)


def test_git_info_outside_repo(tmp_path: Path) -> None:
    sha, dirty = git_info(tmp_path)
    assert sha is None
    assert dirty


def test_package_versions() -> None:
    v = package_versions(["numpy", "no-such-package-xyz"])
    assert v["no-such-package-xyz"] == "yok"
    assert v["numpy"][0].isdigit()
