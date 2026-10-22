# ACKPT-50 — VS Code Kurulumu ve Claude Code Prompt Rehberi

Bu dosyada iki bölüm var: önce VS Code'da kurulum adımları, sonra Claude Code paneline
sırayla yazılacak prompt'lar. Prompt'ları kod bloklarından kopyalayıp panele yapıştır.

---

## A. VS Code kurulum adımları

**1. Ön koşullar**
- VS Code 1.94 veya üstü (Help → About ile kontrol et).
- Git. Windows'ta [Git for Windows](https://git-scm.com/downloads/win) kur. Projedeki otomatik
  format hook'u bash ile çalışıyor, Git Bash gerekiyor.
- Claude aboneliği: Pro, Max, Team, Enterprise veya Console hesabı.

**2. Projeyi aç**
- Zip'i aç.
- VS Code'da **File → Open Folder** ile `pairs_backtest` klasörünü seç.
- "Do you trust the authors?" sorusuna **Yes** de.

**3. Ortamı kur**
`` Ctrl+` `` ile entegre terminali aç ve kurulum scriptini çalıştır:
```bash
bash setup.sh                                         # macOS / Linux
powershell -ExecutionPolicy Bypass -File .\setup.ps1  # Windows
```
Script `22 passed` ile bitmeli.

**4. Python yorumlayıcısını seç**
- `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv` olanı seç.
- Önerilen eklentiler: **Python**, **Ruff**, **YAML**, **Even Better TOML**.

**5. Claude Code eklentisini kur**
- `Ctrl+Shift+X` ile eklentiler görünümünü aç, "Claude Code" ara ve Anthropic'in eklentisini kur.
- Eklenti kendi CLI kopyasıyla geliyor, panel için ayrıca kurulum gerekmiyor. Terminalde
  `claude` komutunu da kullanmak istersen standalone CLI'yi ayrıca kurman gerekir.
- Eklenti görünmezse `Ctrl+Shift+P` → **Developer: Reload Window**.

**6. Paneli aç ve giriş yap**
- Herhangi bir dosya açıkken sağ üstteki **Spark ikonuna** tıkla. Alternatif: sağ alttaki
  **✻ Claude Code** yazısı.
- **Sign in** ile tarayıcıdan giriş yap.

**7. Çalışma modunu ayarla**
- Prompt kutusunun altındaki mod göstergesinden izin modunu seç.
- Başlangıç için plan modu veya düzenlemeden önce onay isteyen mod uygun.
- Plan moduna `/plan` yazarak da geçebilirsin. Claude önce plan çıkarır, onaylayınca kod yazar.

**8. Proje ayarlarının yüklendiğini kontrol et**
- `/` yaz ve slash komutları listesinde `phase`, `new-module`, `lookahead-audit`, `review-run`
  görünüyor mu bak.
- `CLAUDE.md` otomatik yüklenir.

**Faydalı kısayollar**
- **`@dosya`:** dosyayı Claude'a bağlam olarak verir.
- **Kod seçip `Alt+K`:** seçili satırları referans olarak prompt'a ekler (Mac'te `Option+K`).
- **`@terminal:<ad>`:** terminal çıktısını Claude'a verir.

---

## B. Claude Code'a sırayla yazılacak prompt'lar

**Genel kural:** Her fazı **yeni bir sohbet sekmesinde** başlat. Bağlam temiz kalır,
CLAUDE.md zaten her oturumda yükleniyor. Her fazın sonunda commit at.

### Faz 0 — Oryantasyon (plan modu, kod yok)

```
@CLAUDE.md ve @docs/design/ACKPT50_Pairs_Trading_Backtest_Teknik_Raporu.pdf dosyalarını oku.
Kod yazma. Şunları ver:
1) Projeyi 10 satırda özetle.
2) CLAUDE.md §4'teki invariant'ları kendi cümlelerinle say ve her birinin hangi modülde
   uygulanacağını belirt.
3) RAPOR ile CLAUDE.md arasında gördüğün çelişki veya eksikleri listele.
4) Faz 1'e başlamadan netleştirmem gereken soruları sor.
```

### Faz 1 — Veri katmanı

**1.1 Plan**
```
/phase 1
```

**1.2 Veri kaynağını keşfet** (plan onayından önce veya hemen sonra)
```
data.binance.vision üzerindeki USDⓈ-M futures arşiv yapısını incele: klines (1m, 5m),
aggTrades ve fundingRate için daily/monthly URL kalıpları ve .CHECKSUM dosyaları.
Geçmişe dönük exchangeInfo (tickSize, stepSize, minNotional) arşivi var mı, doğrula.
Yoksa:
- symbol lifecycle'ı arşivdeki ilk ve son veri gününden türet,
- tick/step için güncel snapshot'ı kullan,
- bu varsayımı CLAUDE.md'ye ve manifest'e yaz.
Bulgularını kodlamadan önce özetle.
```

**1.3 Downloader**
```
scripts/download_binance_data.py ve data/ katmanında BinanceArchiveDownloader'ı yaz:
- idempotent: var olan ve checksum'ı doğru dosyayı tekrar indirme
- retry + backoff, eşzamanlı indirme (sınırlı)
- raw zip'leri data/raw altına değiştirmeden kaydet
- her indirme için manifest kaydı: url, sha256, boyut, tarih
Önce testleri yaz. Ağ testleri @pytest.mark.network olsun, unit testler mock'la çalışsın.
```

**1.4 Normalize, parquet ve validator**
```
Normalizer + DataValidator + DataRepository yaz:
- raw → data/parquet/<dataset>/<symbol>/<date>.parquet, UTC int64 ms timestamp
- validator şunları kontrol etsin: duplicate, monotonic zaman, gap tespiti, OHLC tutarlılığı,
  sıfır/negatif fiyat
- sonuçları data/manifests altına JSON rapor olarak yaz
- DataRepository.load_klines / load_funding / load_aggtrades(symbols, start, end) API'si
  sadece istenen aralığı yüklesin
Önce test ve fixture'ları yaz.
```

**1.5 Küçük gerçek deneme**
```
BTCUSDT, ETHUSDT, SOLUSDT için son 14 günün 1m+5m kline ve funding verisini indir,
normalize et, validation raporunu göster. Sorun varsa düzelt.
```

**1.6 Tam veri seti ve faz kapanışı**
```
Son 12 ay için, quote volume'a göre likit ilk 60 USDⓈ-M perpetual sembolün 1m, 5m
ve funding verisini indir. Delist olanlar da dahil olsun (survivorship).
Bitince Faz 1 çıkış kriterine göre durum raporu ver ve CLAUDE.md'de mevcut fazı güncelle.
```

**1.7 Kontrol ve commit**
```
make check çalıştır, yeşil olunca "Faz 1: data katmanı" mesajıyla commit at.
```

### Faz 2 — Research ve modeller

**2.1 Plan**
```
/phase 2
```

**2.2 Kalman ve z-score modelleri**
```
/new-module models.KalmanHedgeModel
```
Ardından:
```
Kalman'ı 2-state (alpha, beta) online model olarak yaz (RAPOR §3.1).
Testler:
- sabit beta'lı sentetik çiftte beta'ya yakınsama
- beta'nın kademeli değiştiği seride takip
- NaN girdi davranışı
Sonra models.ZScoreModel'i causal EWMA olarak yaz. Her iki modelin spread ve z
çıktısını tests/unit/test_lookahead_guard.py içindeki CAUSAL_FUNCTIONS listesine ekle.
```

**2.3 İstatistik testleri**
```
research katmanına şunları yaz:
- CointegrationAnalyzer: Engle-Granger + ADF, kısa (7g) ve uzun (28g) pencere
- Hurst exponent
- OUCalibrator: kappa ve half-life (dakika)
Her biri için OU fixture, random walk ve cointegrated pair testleri olsun.
```

**2.4 Pair seçimi**
```
UniverseFilter + PairSelector + NetworkDiversifier yaz (RAPOR §8):
- correlation ön filtresi
- PairQualityScore (ağırlıklar config'den)
- varlık başına max pair kısıtı
- tek fonksiyon: select(cutoff_ts) yalnızca cutoff öncesi formation verisini kullansın
```

**2.5 Denetim**
```
/lookahead-audit src/
```

**2.6 Gerçek veride diagnostics**
```
Faz 1 verisinde rastgele 3 formation cutoff noktası için pair universe üret.
Her pair için şunları içeren bir tablo ver: p-değerleri, Hurst, half-life, beta stabilitesi,
skor. Ayrıca kaç pair'in hangi filtrede elendiğini gösteren bir funnel özeti ver.
artifacts/diagnostics/ altına kaydet.
```

**2.7 Commit**
```
make check, sonra commit. CLAUDE.md'de mevcut fazı güncelle.
```

### Faz 3 — Bar tabanlı backtest (kritik karar noktası)

**3.1 Plan**
```
/phase 3
```

**3.2 Event çekirdeği**
```
backtest katmanını yaz: EventQueue, Scheduler (4 saatlik pair refresh), çoklu sembol bar
feed'ini zaman sırasıyla birleştiren DataFeed ve TimeSynchronizer (stale kontrolü,
max_ffill_bars). Önce testleri yaz.
```

**3.3 Strateji, maliyet ve risk**
```
Şunları yaz:
- strategy: SignalEngine + PositionStateMachine (RAPOR §9; tüm state geçişleri test edilsin)
- costs: FeeModel, SlippageModel (liquidity_scaled), FundingModel, LegRiskModel
- risk: PositionSizer (beta-neutral notional, RAPOR §11.1) + RiskManager
  (max open pairs, max pairs/asset, daily soft/hard stop)
Cost gate: expected_gross > cost_multiplier × expected_cost VE net >= min_net_edge_bps.
```

**3.4 Execution ve ledger**
```
Şunları yaz:
- execution: ConservativeTakerFill. t anındaki bar close sinyali en erken t+1 bar
  open'da + slippage ile fill olsun. Aynı bar fill'i imkânsız olmalı ve bunun testi olsun.
- portfolio: PortfolioLedger. Funding event'leri, realized/unrealized PnL ve
  maliyetlerin PairTradeRecord'a ayrı alanlar olarak işlenmesi.
- hypothesis ile property testleri.
```

**3.5 Walk-forward ve raporlama**
```
WalkForwardRunner + BacktestEngine + MetricsEngine yaz:
- 28g formation / 7g OOS, fold izolasyonu
- artifacts/<run_id>/ altına CLAUDE.md §8'deki tüm reproducibility dosyaları
- metrikler: RAPOR §13.1'in tamamı, round-trips/day P10/P50/P90 ve
  reddedilen sinyal sayısı dahil
```

**3.6 İlk tam koşu**
```
Konservatif maliyetlerle tüm veri üzerinde walk-forward çalıştır.
```

**3.7 Değerlendirme**
```
/review-run artifacts/<run_id>
```

**3.8 Karar**
```
Sonuç negatifse Faz 4'e geçme. Önce zararın nereden geldiğini ayrıştır:
fee, slippage, funding ya da brüt edge yetersizliği.
Ardından 3 somut yeniden tasarım seçeneği öner.
```

### Faz 4 — AggTrade execution

**4.1 Plan**
```
/phase 4
```

**4.2 Fill modelleri**
```
AggTradeFillModel ve HybridFill yaz:
- karar zamanı + latency sonrasındaki ilk trade ile fill
- maker'da queue_haircut uygula
- maker_timeout sonrası taker'a geç
- PairOrderGroup ile iki bacağı ayrı simüle et; leg_timeout aşılınca aggress veya unwind
  (RAPOR §7.2)
- leg-risk maliyetini ayrı kaydet
```

**4.3 Karşılaştırma**
```
Aynı fold'larda bar-based ve aggTrade-based sonuçları yan yana karşılaştır.
Fark hangi maliyet kaleminden geliyor?
```

### Faz 5 — Dayanıklılık

**5.1 Plan**
```
/phase 5
```

**5.2 Stres ve hassasiyet**
```
- costs_conservative.yaml'daki stres senaryolarını çalıştır.
- entry_z, exit_z, stop_z ve min_net_edge için grid sensitivity yap. Formation verisinde
  seç, OOS'ta raporla, tüm grid dağılımını kaydet.
- Trade PnL üzerinde bootstrap güven aralıkları hesapla.
- BTC trend/volatilite rejimlerine göre ayrıştır.
```

**5.3 Final rapor**
```
/review-run ile final değerlendirme yap. Canlıya geçiş kapılarının özet tablosunu çıkar.
```

---

## Her aşamada işe yarayan kısa prompt'lar

| Durum | Prompt |
|---|---|
| Test kırıldı | `make check çalıştır, hataları düzelt. Testi gevşetme.` |
| Terminal hatası | `@terminal:bash çıktısındaki hatayı analiz et ve düzelt` |
| Belirli bir kod parçası | Kodu seç, `Alt+K`, sonra: `bu kısım look-ahead içeriyor mu?` |
| Uzun oturum | `Şu ana kadar yapılanları ve kalanları özetle` → yeni sekmede devam et |
| Faz kapanışı | `CLAUDE.md'yi bu fazda öğrendiklerimizle güncelle, sonra commit at` |

Kaynaklar: [Use Claude Code in VS Code](https://code.claude.com/docs/en/vs-code) ·
[Advanced setup](https://code.claude.com/docs/en/setup)
