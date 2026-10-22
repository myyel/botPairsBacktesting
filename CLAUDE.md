# CLAUDE.md — ACKPT-50 Pairs Trading Backtest

Bu dosya Claude Code'un bu repoda nasıl çalışacağını tanımlar. Her oturumun başında okunur.
Tasarımın kaynağı: `docs/design/ACKPT50_Pairs_Trading_Backtest_Teknik_Raporu.pdf` (bundan sonra **RAPOR**).
Bir konuda bu dosya ile RAPOR çelişirse bu dosya geçerlidir; çelişkiyi kullanıcıya bildir.

## 1. Proje özeti

ACKPT-50 (Adaptive Cost-Aware Kalman Pairs Trading), Binance USDⓈ-M Perpetual Futures üzerinde
market-neutral bir statistical arbitrage stratejisidir. Bu repo stratejinin **event-driven,
walk-forward, maliyet-duyarlı backtest altyapısını** içerir. Canlı trading kodu bu repoda YOKTUR
(Faz 6+ ayrı repo/modül olacak).

Başarı tanımı "backtest kâr etti" değildir. Başarı: tüm maliyetler dahil pozitif OOS expectancy,
makul drawdown, parametrelerde stabilite platosu, rejimler arası dayanıklılık ve normal koşullarda
medyan 50+ pair round-trip/gün **kapasitesi** (kota değil).

## 2. Komutlar

Paket yöneticisi **uv**'dir. `pip install` kullanma; bağımlılık eklemek için `uv add <paket>`
(dev için `uv add --dev <paket>`).

```bash
uv sync --all-extras          # ortamı kur / güncelle
uv run pytest                 # tüm testler
uv run pytest tests/unit -q   # sadece unit
uv run pytest -m "not slow"   # yavaş testler hariç
uv run ruff check . --fix     # lint
uv run ruff format .          # format
uv run mypy src               # tip kontrolü
make check                    # lint + type + test (commit öncesi zorunlu)
uv run python scripts/download_binance_data.py --config config/data_v1.yaml   # idempotent
uv run python scripts/build_dataset.py --config config/data_v1.yaml           # raw -> parquet + manifest
uv run pytest -m network       # gerçek arşive smoke test (varsayılan çalıştırmada hariç)
uv run python scripts/run_walk_forward.py --config config/strategy_v1.yaml
```

Bir görevi bitirmeden önce **`make check` yeşil olmalı**. Yeşil değilse görev bitmemiştir.

## 3. Mimari (RAPOR §5–§7)

```
src/pairs_bt/
├─ config.py     # pydantic config şeması (YAML -> frozen model)
├─ domain/       # immutable event/order/fill/pair-state veri tipleri. Bağımlılığı yok.
├─ data/         # archive, BinanceArchiveDownloader, normalize, DataRepository, DatasetManifest,
│                #   DataValidator, ExchangeInfoStore, SymbolLifecycle, TimeSynchronizer, pipeline
├─ research/     # UniverseFilter, PairSelector, CointegrationAnalyzer, OUCalibrator, NetworkDiversifier
├─ models/       # KalmanHedgeModel, SpreadModel, ZScoreModel (EWMA)
├─ strategy/     # SignalEngine, PositionStateMachine
├─ costs/        # FeeModel, SlippageModel, FundingModel, LegRiskModel
├─ risk/         # PositionSizer, RiskManager, ExposureGraph
├─ execution/    # ExecutionSimulator, PairOrderGroup, FillModel (Bar/AggTrade/Hybrid)
├─ portfolio/    # PortfolioLedger, PositionBook, CashLedger
├─ backtest/     # BacktestEngine, EventQueue, Scheduler, WalkForwardRunner
└─ analytics/    # MetricsEngine, TradeBlotter, ReportBuilder
```

**Bağımlılık yönü (ihlal etme):**
`domain` ← herkes. `models`, `costs` → sadece `domain`. `data` → `domain`, `config`.
`research` → `domain`, `models`, `data`.
`strategy` → `models`, `costs`, `domain`. `execution` → `costs`, `domain`. `portfolio` → `domain`.
`backtest` her şeyi orkestre eder; başka hiçbir modül `backtest`'i import etmez.
`analytics` yalnızca ledger/trade kayıtlarını okur, state değiştirmez.

Execution backend'leri ortak bir `FillModel` Protocol'ünü uygular (Bar, AggTrade, Hybrid,
ileride OrderBookReplay). Strateji kodu hangi fill modelinin kullanıldığını bilmez.

## 4. Değişmez kurallar (INVARIANTS) — asla ihlal etme

Bunlar backtest'in güvenilirliğini belirler. Bir değişiklik bunlardan birini zayıflatıyorsa
**dur ve kullanıcıya sor.**

1. **No look-ahead.** t anındaki karar yalnızca `timestamp <= t` verisini kullanır.
   - 1m bar close'unda üretilen sinyal en erken **t+1** event'inde fill olur. Aynı barın OHLC'si ile fill YASAK.
   - AggTrade replay'de karar timestamp'inden *sonraki* ilk trade kullanılır (+ latency).
   - Rolling/EWMA hesaplarında `center=True`, `shift(-k)`, `bfill` ve tüm-seriden `fit` YASAK.
2. **Walk-forward izolasyonu.** Test (OOS) penceresinin hiçbir verisi pair seçimi, threshold,
   scaler, Kalman başlangıcı veya parametre tuning için kullanılamaz. Her fold yalnızca kendi
   formation verisiyle başlar. Fold'lar arası mutable state paylaşımı YOK.
   - Yorum (kullanıcı kararı, Faz 1): OOS içinde 4 saatlik **causal** pair refresh serbesttir —
     refresh yalnız `cutoff` öncesi veriyi kullanır, canlıdaki davranışla aynıdır. Hiperparametreler,
     eşikler ve skor ağırlıkları fold başında formation'dan sabitlenir; OOS'tan fit/tuning YASAK.
3. **Survivorship bias yok.** Universe, o tarihteki symbol lifecycle manifest'inden gelir;
   bugünkü exchange-info geçmişe uygulanmaz. Delist edilen semboller veriden silinmez.
4. **İki bacak atomik değildir.** Her leg ayrı order olarak simüle edilir; partial fill ve
   leg-risk timeout modellenir. "İki fiyatı aynı anda doldur" varsayımı YASAK.
5. **Maliyet trade lifecycle'ının içindedir.** Fee, slippage, funding, leg-risk her fill/event'te
   uygulanır ve `PairTradeRecord` içinde **ayrı alanlar** olarak tutulur. Brüt PnL'dan sonradan
   toplu düşülmez.
6. **Trade sayısı kuralları bypass edemez.** "50 trade/gün" bir kota değildir; cost gate veya
   risk limiti, trade sayısını artırmak için asla gevşetilmez.
7. **Determinizm.** Aynı veri + config + seed = bit-bit aynı sonuç. Tüm rastgelelik
   `numpy.random.Generator` ile ve config'deki seed'den türetilir. `random` modülü ve global
   `np.random.*` kullanma. Dict/set iterasyon sırasına bağlı mantık kurma (sembolleri sırala).
8. **Zaman UTC.** Tüm timestamp'ler tz-aware UTC (`pandas` tarafında `datetime64[ns, UTC]`,
   iç döngüde int64 epoch-ms). Europe/Istanbul dönüşümü yalnızca raporlama katmanında.
9. **Gerçekçi emir kuralları.** Miktar/fiyat `stepSize`/`tickSize`'a yuvarlanır, `minQty`/
   `minNotional` kontrol edilir — exchange-info o tarihe ait olmalı.
10. **Parametre optimizasyonu OOS üzerinde yapılmaz** ve sadece en iyi kombinasyon raporlanmaz;
    tarama sonuçlarının tamamı (dağılım) kaydedilir.

## 5. Kodlama kuralları

- Python **3.12+** (hedef 3.13). Tam tip anotasyonu; `mypy --strict` src için geçmeli.
- Domain nesneleri `@dataclass(frozen=True, slots=True)`. Config'ler pydantic `BaseModel`
  (`model_config = ConfigDict(frozen=True, extra="forbid")`).
- Para/fiyat hesaplarında `float64` kabul; ancak ledger toplamlarında birikimli hata için
  testlerde `pytest.approx` toleransı açıkça belirt. Rounding (tick/step) için `decimal` veya
  integer tick aritmetiği kullan.
- Birimler isimde açık olsun: `fee_bps`, `notional_usdt`, `ts_ms`, `half_life_min`.
  bps = 1e-4. Yüzde ile bps'i karıştırma.
- Hot loop'ta (event başına) DataFrame oluşturma/kopyalama yok; numpy array veya önceden
  hesaplanmış yapılar kullan. `numba` yalnızca profiling bir darboğaz gösterdikten sonra.
- `print` yok; `structlog`/`logging` kullan. Log'lar run-id içerir.
- Notebook'lar yalnız keşif içindir; production mantığı `src/`'de yaşar. Notebook'tan
  `src` import edilir, tersi asla.
- Docstring ve yorumlar Türkçe veya İngilizce olabilir; tanımlayıcılar (değişken/sınıf adları)
  İngilizce.
- Sihirli sayı yok: tüm eşikler config'den gelir (RAPOR §3.5 tablosu başlangıç değerleridir).

## 6. Test kuralları (RAPOR §15)

- Yeni her public fonksiyon/sınıf için unit test. Happy path + anlamlı edge case'ler
  (boş seri, NaN, tek bar, sıfır volatilite, beta işaret değişimi, funding sınırı, partial fill).
- İstatistiksel fonksiyonlar sentetik fixture'larla test edilir: bilinen OU spread
  (beklenen half-life), random walk (ADF reddetmemeli), sabit beta'lı cointegrated çift
  (Kalman beta'ya yakınsamalı). Fixture üreticileri `tests/fixtures/` ve `tests/conftest.py`'de.
- Ledger ve rounding için `hypothesis` property testleri (ör. "kapatılan pozisyonda
  realized PnL = Σ fill PnL − Σ fee", "rounding sonrası qty % stepSize == 0").
- **Look-ahead testi zorunlu:** sinyal/feature fonksiyonlarında, t sonrası veriyi değiştirmenin
  t'deki çıktıyı değiştirmediğini doğrulayan test (bkz. `tests/unit/test_lookahead_guard.py` kalıbı).
- Regression/golden testler `tests/regression/`; golden artifact değişirse farkı açıkla,
  kullanıcı onayı olmadan golden dosyayı güncelleme.
- Yavaş testleri `@pytest.mark.slow`, ağ gerektirenleri `@pytest.mark.network` ile işaretle.
  CI/varsayılan çalıştırma ağ kullanmaz.

## 7. Veri

- Kaynak: `https://data.binance.vision` (USDⓈ-M futures: klines, aggTrades, fundingRate) +
  exchangeInfo snapshot'ları. Checksum (`.CHECKSUM`) doğrulaması zorunlu.
- Katmanlar: `data/raw` (indirilen zip, arşivle aynı yol yapısı, değiştirilmez) →
  `data/normalized` (kaynak dosya bazlı normalizasyon kayıtları: satır sayısı + düzeltmeler) →
  `data/parquet` (`dataset=/symbol=/month=YYYY-MM/part.parquet`, UTC) → `data/manifests`.
  - Partition **sembol/ay**dır (kullanıcı kararı, Faz 1): sembol/gün tam evrende ~500 bin küçük
    dosya ediyordu. Gün bazlı okuma `DataRepository.read_range` ts filtresiyle yapılır.
  - `data/manifests`: dataset manifestleri (`<dataset>.json`), `dataset_index.json`
    (`data_manifest_hash` — yalnız veri içeriği + exchangeInfo snapshot adları), `symbol_lifecycle.json`,
    `derived_ticks.json`, `validation_report.{json,md}`, `build_summary.md`, `download_report.md`.
  - Kısmi build (ör. `--datasets aggTrades`) manifest/rapor üretimini repository'deki TÜM
    dataset'ler üzerinden yeniden yapar; diğer dataset'lerin manifestlerini silmez.
- `data/` ve `artifacts/` git'e girmez. Raw veriyi asla elle düzenleme; düzeltme normalize
  adımında, loglanarak yapılır.
- Eksik veri: forward-fill süresi config'de sınırlı (`max_ffill_bars`); aşılırsa sembol o
  pencere için "stale" sayılır, sinyal üretilmez (`data.TimeSynchronizer`; bfill yok).
- Config: `config/data_v1.yaml` (`DataConfig`). Pencere 2025-09-01 → 2026-09-30: walk-forward
  `start` (2025-10-01) **ilk OOS başlangıcıdır**, öncesindeki 28 gün ilk formation içindir.

### 7.1 Veri konvansiyonları ve öğrenilenler (Faz 1)

- **Bar zamanı:** `ts_ms = open_time_ms + interval_ms` (barın tamamen kapandığı an). Binance
  `close_time` (= open + interval − 1 ms) saklanmaz. Bar'a dayalı karar en erken `ts_ms`'te verilir.
- **İşlemsiz bar'lar sentetiktir:** Binance `trades == 0` olan 1m bar'ları önceki close ile
  (OHLC sabit, hacim 0) üretir; 5m OHLC ise yalnız gerçek trade'lerden gelir. Sonuçlar:
  1m↔5m çapraz kontrol işlem görmüş bar'larla yapılır; lifecycle ilk/son veri zamanı `trades > 0`
  bar'lardan hesaplanır; validator `zero_trade_bars` uyarısı verir.
- **Delist sonrası veri:** SETTLING sembollerde arşiv, delivery sonrası ay sonuna kadar sabit
  fiyatlı işlemsiz bar yayınlar. Universe için `SymbolLifecycle.active_symbols(t)` kullan
  (delivery ve son işlem zamanını dikkate alır); ham bar varlığına güvenme.
- **Exchange-info (invariant #9) politikası:** data.binance.vision geçmiş exchangeInfo sunmaz.
  `fapi/v1/exchangeInfo` ham snapshot olarak `data/raw/exchange_info/exchangeInfo_<serverTime>.json`
  altında birikir. ts ≥ snapshot → kesin filtre. Daha önceki ts → tickSize o ayın fiyat grid'inden
  türetilir (fiyatların ≥ `tick_grid_min_fraction`'ını kapsayan en büyük `{1,2,2.5,5}×10^k`),
  step/minQty/minNotional en erken snapshot'tan; `SymbolFilters.approximate=True`. Faz 3 bu bayrağı
  raporlamalıdır. Hiçbir snapshot'ta olmayan (çok eski delist) semboller için filtre YOKTUR →
  emir üretilemez, universe'ten dışlanmalı.
- **Funding:** arşiv `funding_interval_hours` alanını içerir; aralık sembol bazında ve zamanla
  değişir (1h/4h/8h). Config'teki `interval_hours` yalnız yedek değerdir.
- **Universe:** varsayılan `contract_types: [PERPETUAL]`, quote USDT. `TRADIFI_PERPETUAL`
  (XAUUSDT, TSLAUSDT vb.) bilinçli olarak hariç (RAPOR §8.1), config'ten açılabilir. exchangeInfo'da
  hiç olmayan arşiv sembolleri survivorship için dahil edilir; `_` içeren vadeli kontratlar hariç.
- **Windows:** varsayılan encoding cp1254'tür; tüm dosya okuma/yazmalarında `encoding="utf-8"` ver.
- **Ruff exclude tuzağı:** `extend-exclude`'a tek kelimelik desen (ör. `"data"`) yazma — o adı
  taşıyan HER klasörü (ör. `src/pairs_bt/data`) lint/format dışına iter. Faz 1'de bu yüzden
  `make check` data paketini görmüyordu; pre-commit (dosya adlarını açık verdiği için) yakaladı.

### 7.2 Faz 1 veri seti (tam build, 2026-10-22)

- 689 sembol indirildi (23 381 dosya, 12.3 GB raw, 0 checksum hatası); repository'de 1m/5m için
  656 sembol (1m: ~352 M satır), funding 618 sembol. Doğrulama: **0 hata**, 1m/5m medyan kapsama %100.
  Güncel `data_manifest_hash` → `data/manifests/dataset_index.json`.
- **Ölü semboller:** pencereden ÖNCE delist edilmiş 42 sembol (AGIXUSDT, BALUSDT, ...) için arşiv
  tüm pencere boyunca sabit fiyatlı sentetik bar yayınlıyor (`trades == 0`). Lifecycle'a girmezler,
  hiçbir an aktif değildirler. Repository'de durmaları survivorship açısından zararsızdır.
- **Funding arşiv boşluğu:** 2026-06-24 04:00 funding kaydı 4 saatlik aralıktaki TÜM sembollerde
  (394) eksik; 8 saatlikler etkilenmedi. Faz 3 FundingModel eksik settlement kaydını sessizce
  "funding = 0" saymamalı; politika (ör. komşu kayıttan tahmin / trade'i işaretle) config'e konmalı
  ve kullanıcıya sorulmalı.
- **Gerçek bar boşlukları:** AIAUSDT 2026-01-20 (675 bar), LITUSDT 2025-12-23 (1 050 bar).
  Synchronizer bu aralıkları stale sayar.
- **Emir kuralları:** snapshot'lar pencereden sonra (2026-10) alındığı için pencere içindeki TÜM
  filtreler `approximate`. 49 sembolde fiyatlar güncel tickSize grid'inde değil (tick zamanla
  değişmiş) → ay bazlı türetilmiş tick şart. Hiç filtresi olmayan semboller: AERGOUSDT, BDXNUSDT,
  BTCSTUSDT, MDTUSDT, SXPUSDT (universe'ten dışla). İleriye dönük doğruluk için exchangeInfo
  snapshot'ını periyodik al (`download_binance_data.py` her çalıştırmada bir snapshot ekler).
- İşlemsiz 1m bar oranı canlı sembollerde medyan ~%1.2; illikit sembollerde çok daha yüksek —
  Faz 2 UniverseFilter likidite filtresinde `trades`/`quote_volume` kullanılmalı.
- aggTrades yalnız pilot: BTCUSDT/ETHUSDT 2025-09-01..02 (7.7 M satır). Toplu indirme Faz 4.

## 8. Reproducibility

Her backtest run'ı `artifacts/<run_id>/` altına şunları yazar: çözümlenmiş config (YAML),
config hash, git commit (dirty flag ile), dataset manifest hash, seed, paket sürümleri,
trade blotter (parquet), equity curve, metrikler (JSON), log. `run_id = <UTC tarih>_<config_hash[:8]>`.

## 9. Geliştirme yol haritası (RAPOR §16)

Sırayla ilerle; bir faz çıkış kriterini karşılamadan sonrakine geçme.
Mevcut faz: **Faz 1 — Data tamam** (13 ay 1m/5m + funding + exchange-info, 0 doğrulama hatası;
bkz. §7.2). Faz 2'ye hazır.

| Faz | Teslim | Çıkış kriteri |
|---|---|---|
| 1 Data | Downloader + validator + parquet + manifest | 6–12 ay güvenilir 1m/5m + funding + exchange-info |
| 2 Research | Universe, coint (EG/ADF), Hurst, OU, Kalman, PairSelector | Pair diagnostics + 4 saatlik pair universe üretimi |
| 3 Bar Backtest | Event engine + risk + ledger + conservative cost | Hızlı parametre/feasibility analizi |
| 4 AggTrade | Next-event fill + leg-risk + hybrid execution | Maliyet sonrası edge ve 50+ kapasite doğrulaması |
| 5 Robustness | Walk-forward + stress + sensitivity + bootstrap | Fold çoğunluğunda stabil pozitif expectancy |

**Kritik karar (RAPOR §16):** Faz 3'te konservatif maliyet modeliyle strateji negatifse Faz 4+
için kod yazmadan önce kullanıcıya rapor ver.

Faz ilerledikçe bu bölümdeki "Mevcut faz" satırını güncelle.

## 10. Çalışma şekli

- Büyük bir özelliğe başlamadan önce kısa bir plan çıkar (hangi modüller, hangi testler,
  hangi invariant'lar etkileniyor). Plan modu uygunsa kullan.
- Küçük, test edilebilir adımlarla ilerle: önce test/fixture, sonra implementasyon.
- Bir istatistiksel/finansal formülü implement ederken RAPOR'daki bölümü docstring'de referansla
  (ör. `"""RAPOR §3.4 cost-aware trade gate."""`).
- Belirsiz bir finansal varsayım (funding işareti, fee tier, beta yönü) varsa tahmin etme;
  varsayımı config'e koy, docstring'de belirt ve kullanıcıya bildir.
- Kullanıcı ile iletişim Türkçe.

## 11. Yapma

- Gerçek API key, secret, hesap bilgisi koda/config'e/test'e yazma. Gerekirse `.env`
  (git'e girmez) ve `.env.example`.
- Canlı emir gönderen kod yazma (bu repo backtest'tir).
- `data/raw` içeriğini değiştirme veya silme.
- Test geçsin diye testi gevşetme, `skip`/`xfail` ekleme — önce kullanıcıya sor.
- Golden/regression artifact'larını onaysız güncelleme.
- Look-ahead invariant'ını zayıflatan "hızlandırma" optimizasyonları.
