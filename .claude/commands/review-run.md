---
description: Bir backtest run'ının sonuçlarını RAPOR §13 kabul kapılarına göre değerlendir
argument-hint: <artifacts/run_id>
---

`$ARGUMENTS` klasöründeki metrikleri, trade blotter'ı ve config'i oku.

1. Reproducibility kontrolü: config hash, git commit (dirty?), dataset manifest, seed mevcut mu?
2. RAPOR §13.2 kabul kapıları tablosu: her gate için değer / eşik / GEÇTİ-KALDI.
   (median round-trips/day ≥50, net expectancy >0, OOS PF >1.15, OOS Sharpe >1.5, MDD <%3–5,
   cost/gross-edge, rejim dayanıklılığı, parametre platosu)
3. Maliyet ayrıştırması: brüt edge vs fee / slippage / funding / leg-risk. "Model edge'i var
   ama execution'da mı ölüyor?" sorusunu yanıtla.
4. Fold-by-fold OOS PnL dağılımı; tek bir fold'a bağımlılık var mı?
5. Stres senaryoları (fee×1.5, slip×2, all-taker) sonuçları.
6. Şüpheli işaretler: gerçekçi olmayan win rate, sıfıra yakın holding, aynı-bar fill, çok yüksek Sharpe.

Sonuçta net bir hüküm ver: devam / parametre/universe yeniden tasarım / dur.
