# ACKPT-50 — Adaptive Cost-Aware Kalman Pairs Trading Backtest

Binance USDⓈ-M Perpetual Futures için event-driven, walk-forward, maliyet-duyarlı pairs trading
backtest altyapısı. Tasarım: `docs/design/ACKPT50_Pairs_Trading_Backtest_Teknik_Raporu.pdf`.

> Bu yazılım yatırım tavsiyesi değildir. Backtest sonuçları canlı performansı garanti etmez.

## Kurulum

**Linux / macOS / WSL**
```bash
bash setup.sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Script: `uv` kurar → Python 3.13 + bağımlılıklar → git + pre-commit → `.env` → config doğrulama → testler.

Manuel:
```bash
uv sync --all-extras
uv run pre-commit install
make check
```

## Claude Code ile geliştirme

```bash
claude            # proje kökünde
```

- `CLAUDE.md` — proje kuralları, mimari, invariant'lar (Claude her oturumda okur).
- `.claude/settings.json` — izinler (ör. `data/raw` ve `.env` korumalı) ve kaydetme sonrası otomatik format.
- Slash komutları:
  - `/phase 1` — yol haritasındaki fazı planla ve uygula
  - `/new-module models.KalmanHedgeModel` — mimariye uygun modül + testler
  - `/lookahead-audit src/` — look-ahead / sızıntı denetimi
  - `/review-run artifacts/<run_id>` — sonuçları kabul kapılarına göre değerlendir

Önerilen ilk adım: `/phase 1`.

VS Code kurulumu ve faz faz kopyalanabilir prompt listesi: **`docs/PROMPTS.md`**.

## Klasör yapısı

```
config/          strateji, maliyet ve walk-forward YAML'ları
src/pairs_bt/    domain, data, research, models, strategy, costs, risk, execution, portfolio, backtest, analytics
tests/           unit / integration / regression
scripts/         veri indirme, dataset, walk-forward, rapor
data/            raw → normalized → parquet + manifests (git dışı)
artifacts/       run_id bazlı sonuçlar (git dışı)
docs/design/     tasarım raporu
docs/PROMPTS.md  VS Code kurulumu + Claude Code prompt rehberi
```

## Sık kullanılan komutlar

| Komut | Açıklama |
|---|---|
| `make check` | lint + type + test (commit öncesi) |
| `make test-fast` | yavaş testler hariç |
| `make fmt` | ruff fix + format |
| `uv run pairs-bt validate-config` | config doğrula + hash |
