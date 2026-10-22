---
description: CLAUDE.md mimarisine uygun yeni bir sınıf/modül + testlerini oluştur
argument-hint: <katman>.<SınıfAdı>  (ör. models.KalmanHedgeModel)
---

`$ARGUMENTS` oluştur.

- Doğru katmana koy ve CLAUDE.md §3 bağımlılık yönünü kontrol et (yasak import varsa dur).
- Docstring'de ilgili RAPOR bölümünü referansla.
- Tüm eşikler config'den (gerekirse `src/pairs_bt/config.py` ve `config/strategy_v1.yaml`'a alan ekle).
- Önce `tests/unit/test_<modül>.py`: happy path + edge case'ler + (feature/sinyal ise)
  `CAUSAL_FUNCTIONS` listesine ekleme.
- Sonra implementasyon. `make check` yeşil olana kadar bitirme.
