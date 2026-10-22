---
description: Kod tabanını look-ahead, survivorship ve walk-forward sızıntısı için denetle
argument-hint: [dizin veya dosya — boşsa src/]
---

Hedef: ${ARGUMENTS:-src/}

CLAUDE.md §4 invariant #1, #2, #3, #10'a göre denetim yap. Kodu DEĞİŞTİRME, yalnızca raporla.

Ara:
- `center=True`, `shift(-`, `bfill`, `backfill`, `.iloc[-1]` ile gelecek satır erişimi,
  tüm seri üzerinden `fit`/`mean`/`std`/`quantile` (formation penceresi dışına taşan).
- Aynı bar close'unda sinyal + aynı barın OHLC'sinden fill.
- Fold sınırını aşan state (global cache, sınıf düzeyi mutable alan, modül düzeyi dict).
- Bugünkü exchange-info / sembol listesinin geçmiş tarihlere uygulanması.
- OOS verisiyle parametre seçimi.
- `CAUSAL_FUNCTIONS` listesine (tests/unit/test_lookahead_guard.py) eklenmemiş feature/sinyal fonksiyonları.

Çıktı: dosya:satır, risk seviyesi (YÜKSEK/ORTA/DÜŞÜK), açıklama, önerilen düzeltme.
