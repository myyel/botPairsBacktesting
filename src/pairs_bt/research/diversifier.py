"""NetworkDiversifier — varlık yoğunlaşmasını sınırlayan açgözlü pair seçimi (RAPOR §8.2 adım 7–8).

Seçilen pair'ler bir graf oluşturur (düğüm = varlık, kenar = pair). Her adımda, iki varlığı da
`max_pairs_per_asset` sınırının altında olan adaylar arasından
    score = base_score + w_div · diversification,
    diversification = 1 − (deg(a) + deg(b)) / (2 · max_pairs_per_asset)
en yüksek olan seçilir (eşitlikte pair adı — determinizm, invariant #7). Diversification seçim
anındaki graf derecesine göre hesaplanır; böylece zaten merkezi olan bir varlığı içeren pair'ler
cezalandırılır ve kapasite sınırı kesin uygulanır.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pairs_bt.domain.events import PairId


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    pair: PairId
    base_score: float  # diversification hariç ağırlıklı skor


@dataclass(frozen=True, slots=True)
class Selection:
    pair: PairId
    base_score: float
    diversification: float
    score: float
    rank: int  # 1 = ilk seçilen


class NetworkDiversifier:
    def __init__(self, max_pairs_per_asset: int, weight: float) -> None:
        if max_pairs_per_asset <= 0:
            raise ValueError("max_pairs_per_asset pozitif olmalı")
        if weight < 0:
            raise ValueError("weight negatif olamaz")
        self.max_pairs_per_asset = max_pairs_per_asset
        self.weight = weight

    def select(self, candidates: Sequence[ScoredCandidate], k: int) -> tuple[Selection, ...]:
        pool = sorted(candidates, key=lambda c: (c.pair.a, c.pair.b))
        if len({c.pair for c in pool}) != len(pool):
            raise ValueError("aday listesinde tekrar eden pair var")
        degree: dict[str, int] = {}
        chosen: list[Selection] = []
        remaining = list(pool)
        cap = self.max_pairs_per_asset
        while remaining and len(chosen) < k:
            best: tuple[float, ScoredCandidate, float] | None = None
            for c in remaining:
                da, db = degree.get(c.pair.a, 0), degree.get(c.pair.b, 0)
                if da >= cap or db >= cap:
                    continue
                div = 1.0 - (da + db) / (2.0 * cap)
                total = c.base_score + self.weight * div
                if best is None or total > best[0]:  # eşitlikte sıralı ilk aday kalır
                    best = (total, c, div)
            if best is None:
                break
            total, c, div = best
            chosen.append(Selection(c.pair, c.base_score, div, total, len(chosen) + 1))
            degree[c.pair.a] = degree.get(c.pair.a, 0) + 1
            degree[c.pair.b] = degree.get(c.pair.b, 0) + 1
            remaining.remove(c)
        return tuple(chosen)

    def is_blocked(self, pair: PairId, selected: Sequence[Selection]) -> bool:
        """Seçim bittikten sonra: bu pair varlık kapasitesi yüzünden mi eklenemezdi?"""
        degree: dict[str, int] = {}
        for s in selected:
            for sym in (s.pair.a, s.pair.b):
                degree[sym] = degree.get(sym, 0) + 1
        cap = self.max_pairs_per_asset
        return degree.get(pair.a, 0) >= cap or degree.get(pair.b, 0) >= cap
