Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " 🌐 JRA-3Q 4D Mesoscale & 3D Spherical AI Forecaster Setup" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Python の確認
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Exit 1
}

$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "✅ Detected Python Version: $pyVer" -ForegroundColor Green

# 2. ディレクトリの入力（空白でEnterなら $HOME/jra3q-ai-forecaster）
$defaultDir = Join-Path $HOME "jra3q-ai-forecaster"
Write-Host ""
$inputDir = Read-Host "📁 Enter install directory [Press Enter for default: $defaultDir]"
if ([string]::IsNullOrWhiteSpace($inputDir)) {
    $targetDir = $defaultDir
} else {
    $targetDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($inputDir)
}

Write-Host "📂 Target directory: $targetDir" -ForegroundColor Cyan
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
}
Set-Location $targetDir

# 3. コアファイルのダウンロード
Write-Host "📥 Downloading forecaster core files..." -ForegroundColor Yellow
$baseUrl = "https://raw.githubusercontent.com/name-5096/jra3q-ai-forecaster/main"

Invoke-WebRequest -Uri "$baseUrl/converter.py" -OutFile "converter.py" -UseBasicParsing
Invoke-WebRequest -Uri "$baseUrl/local_ai.py" -OutFile "local_ai.py" -UseBasicParsing
Invoke-WebRequest -Uri "$baseUrl/requirements.txt" -OutFile "requirements.txt" -UseBasicParsing

# 4. ライブラリの存在チェック（入っているものはスキップ）
Write-Host "🔍 Checking installed dependencies..." -ForegroundColor Yellow
$checkScript = @"
import importlib.util
pkgs = {
    'streamlit': 'streamlit',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'plotly': 'plotly',
    'requests': 'requests',
    'tabulate': 'tabulate',
    'google.generativeai': 'google-generativeai',
    'netCDF4': 'netCDF4'
}
missing = [pip_name for mod, pip_name in pkgs.items() if importlib.util.find_spec(mod) is None]
print(' '.join(missing))
"@

$missingPkgs = python -c $checkScript
if ([string]::IsNullOrWhiteSpace($missingPkgs)) {
    Write-Host "⚡ All required libraries are already installed! (Skipping pip install)" -ForegroundColor Green
} else {
    Write-Host "📦 Missing packages detected: $missingPkgs" -ForegroundColor Yellow
    Write-Host "Installing missing dependencies..." -ForegroundColor Yellow
    python -m pip install $missingPkgs.Split(" ") --quiet
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " ✅ Setup Complete! Launching Forecaster UI..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# 5. Streamlit の起動
python -m streamlit run converter.py
