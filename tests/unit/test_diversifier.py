"""NetworkDiversifier (RAPOR §8.2 adım 7–8)."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pairs_bt.domain.events import PairId
from pairs_bt.research.diversifier import NetworkDiversifier, ScoredCandidate

ASSETS = [f"S{i}USDT" for i in range(7)]
ALL_PAIRS = [PairId(a, b) for a, b in itertools.combinations(ASSETS, 2)]


@st.composite
def candidate_lists(draw: st.DrawFn) -> list[ScoredCandidate]:
    pairs = draw(st.lists(st.sampled_from(ALL_PAIRS), unique=True, max_size=len(ALL_PAIRS)))
    scores = draw(
        st.lists(st.floats(0, 1, allow_nan=False), min_size=len(pairs), max_size=len(pairs))
    )
    return [ScoredCandidate(p, s) for p, s in zip(pairs, scores, strict=True)]


@settings(max_examples=200, deadline=None)
@given(cands=candidate_lists(), cap=st.integers(1, 3), k=st.integers(1, 25))
def test_selection_properties(cands: list[ScoredCandidate], cap: int, k: int) -> None:
    div = NetworkDiversifier(cap, weight=0.1)
    sel = div.select(cands, k)
    degree: dict[str, int] = {}
    for s in sel:
        for sym in (s.pair.a, s.pair.b):
            degree[sym] = degree.get(sym, 0) + 1
    # varlık kapasitesi kesin
    assert all(d <= cap for d in degree.values())
    assert len(sel) <= k
    assert len({s.pair for s in sel}) == len(sel)
    assert [s.rank for s in sel] == list(range(1, len(sel) + 1))
    # açgözlü maksimallik: k dolmadıysa kalan her aday kapasiteye takılır
    if len(sel) < k:
        chosen = {s.pair for s in sel}
        assert all(div.is_blocked(c.pair, sel) for c in cands if c.pair not in chosen)
    # sıra bağımsızlığı (determinizm)
    assert div.select(list(reversed(cands)), k) == sel


def test_prefers_less_central_pair_on_tie() -> None:
    p_ab, p_ac, p_de = PairId("A", "B"), PairId("A", "C"), PairId("D", "E")
    cands = [ScoredCandidate(p_ab, 0.5), ScoredCandidate(p_ac, 0.5), ScoredCandidate(p_de, 0.5)]
    sel = NetworkDiversifier(2, weight=0.2).select(cands, 2)
    # ilk seçim (eşitlik → ad sırası) A|B; ikinci A'yı tekrar kullanmayan D|E
    assert [s.pair for s in sel] == [p_ab, p_de]
    assert sel[0].diversification == 1.0
    assert sel[1].diversification == 1.0


def test_diversification_penalty_value() -> None:
    p_ab, p_ac = PairId("A", "B"), PairId("A", "C")
    sel = NetworkDiversifier(2, weight=0.1).select(
        [ScoredCandidate(p_ab, 0.9), ScoredCandidate(p_ac, 0.8)], 5
    )
    assert sel[1].pair == p_ac
    assert sel[1].diversification == pytest.approx(1 - 1 / 4)
    assert sel[1].score == pytest.approx(0.8 + 0.1 * 0.75)


def test_duplicate_candidates_rejected() -> None:
    p = PairId("A", "B")
    with pytest.raises(ValueError, match="tekrar"):
        NetworkDiversifier(2, 0.1).select([ScoredCandidate(p, 0.1), ScoredCandidate(p, 0.2)], 3)


@pytest.mark.parametrize(("cap", "w"), [(0, 0.1), (2, -0.1)])
def test_invalid_params(cap: int, w: float) -> None:
    with pytest.raises(ValueError, match=r"pozitif|negatif"):
        NetworkDiversifier(cap, w)
