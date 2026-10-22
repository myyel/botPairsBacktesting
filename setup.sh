#!/usr/bin/env bash
# ACKPT-50 geliştirme ortamı kurulumu (Linux / macOS / WSL)
# Kullanım: bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

say() { printf "\033[1;34m==>\033[0m %s\n" "$*"; }

# 1) uv
if ! command -v uv >/dev/null 2>&1; then
  say "uv kuruluyor..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
say "uv $(uv --version | awk '{print $2}')"

# 2) Python + bağımlılıklar
say "Python 3.13 ve bağımlılıklar kuruluyor (uv sync)..."
uv python install 3.13
uv sync --all-extras

# 3) git + pre-commit
if [ ! -d .git ]; then
  say "git deposu başlatılıyor..."
  git init -q -b main
fi
uv run pre-commit install >/dev/null && say "pre-commit hook kuruldu"

# 4) .env
[ -f .env ] || { cp .env.example .env; say ".env oluşturuldu (.env.example'dan)"; }

# 5) Doğrulama
say "Config doğrulanıyor..."
uv run pairs-bt validate-config config/strategy_v1.yaml
say "Testler çalıştırılıyor..."
uv run pytest -q

# 6) Claude Code
if command -v claude >/dev/null 2>&1; then
  say "Claude Code bulundu: $(claude --version 2>/dev/null || echo '?')"
else
  say "Claude Code bulunamadı. Kurulum: https://docs.claude.com/en/docs/claude-code/setup"
fi

say "Kurulum tamam. Başlamak için:  claude   (proje kökünde)"
