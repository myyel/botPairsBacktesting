"""CLAUDE.md §3 bağımlılık yönü: paketler yalnız izin verilen paketleri import eder."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "pairs_bt"

# paket -> import edebileceği pairs_bt alt paketleri (kendisi her zaman serbest)
ALLOWED: dict[str, set[str]] = {
    "domain": set(),
    "models": {"domain"},
    "costs": {"domain"},
    "data": {"domain", "config"},
    "research": {"domain", "models", "data", "config", "costs"},
    "strategy": {"models", "costs", "domain", "config"},
    "execution": {"costs", "domain", "config"},
    "portfolio": {"domain", "config"},
    "risk": {"domain", "config", "costs"},
    "analytics": {"domain", "portfolio", "config"},
}


def _imports(pkg: str) -> set[str]:
    found: set[str] = set()
    for py in (SRC / pkg).rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                parts = n.split(".")
                if parts[0] == "pairs_bt" and len(parts) > 1:
                    found.add(parts[1])
    found.discard(pkg)
    return found


@pytest.mark.parametrize("pkg", sorted(ALLOWED))
def test_dependency_direction(pkg: str) -> None:
    illegal = _imports(pkg) - ALLOWED[pkg]
    assert not illegal, f"{pkg} izin verilmeyen paketleri import ediyor: {sorted(illegal)}"


def test_nobody_imports_backtest() -> None:
    offenders = [p for p in ALLOWED if "backtest" in _imports(p)]
    assert offenders == []
