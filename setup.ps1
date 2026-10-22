# ACKPT-50 geliştirme ortamı kurulumu (Windows PowerShell)
# Kullanım:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# 1) uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "uv kuruluyor..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Say ("uv " + (uv --version))

# 2) Python + bağımlılıklar
Say "Python 3.13 ve bağımlılıklar kuruluyor (uv sync)..."
uv python install 3.13
uv sync --all-extras

# 3) git + pre-commit
if (-not (Test-Path .git)) { Say "git deposu başlatılıyor..."; git init -q -b main }
uv run pre-commit install | Out-Null
Say "pre-commit hook kuruldu"

# 4) .env
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Say ".env oluşturuldu" }

# 5) Doğrulama
Say "Config doğrulanıyor..."
uv run pairs-bt validate-config config/strategy_v1.yaml
Say "Testler çalıştırılıyor..."
uv run pytest -q

# 6) Claude Code
if (Get-Command claude -ErrorAction SilentlyContinue) {
    Say ("Claude Code bulundu: " + (claude --version))
} else {
    Say "Claude Code bulunamadı. Kurulum: https://docs.claude.com/en/docs/claude-code/setup"
}
Say "Kurulum tamam. Başlamak için proje kökünde:  claude"
