---
description: RAPOR yol haritasındaki bir fazı planla ve uygula (ör. /phase 1)
argument-hint: <faz-no>
---

Faz $ARGUMENTS üzerinde çalışacağız.

1. `CLAUDE.md` §9'daki faz tablosunu ve `docs/design/ACKPT50_Pairs_Trading_Backtest_Teknik_Raporu.pdf`
   içindeki ilgili bölümleri oku (Faz 1: §4, §14 · Faz 2: §3, §8 · Faz 3: §5–§7, §9–§11 ·
   Faz 4: §7.2, §10 · Faz 5: §12, §13, §15).
2. Önceki fazın çıkış kriterinin karşılandığını kontrol et; karşılanmıyorsa dur ve bildir.
3. Kısa bir plan yaz: oluşturulacak modüller/sınıflar, public arayüzler, etkilenen invariant'lar
   (CLAUDE.md §4), yazılacak testler ve fixture'lar. Onayımı bekle.
4. Onaydan sonra test-first ilerle; her alt adımda `make check` çalıştır.
5. Bitince: faz çıkış kriterine karşı durum özeti ver ve CLAUDE.md'deki "Mevcut faz" satırını güncelle.
