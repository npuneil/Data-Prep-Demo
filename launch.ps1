# ═══════════════════════════════════════════════════════
#  Data Prep Assistant — One-Click Launcher
#  For Microsoft Surface CoPilot+ PC Demo
# ═══════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "  =========================================" -ForegroundColor DarkYellow
Write-Host "   ⭐  Data Prep Assistant" -ForegroundColor Yellow
Write-Host "      On-Device AI — CoPilot+ PC Demo" -ForegroundColor DarkGray
Write-Host "  =========================================" -ForegroundColor DarkYellow
Write-Host ""

# ── Step 1: Check Python ──
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Cyan
$PythonCmd = $null

foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.\d+") {
            $PythonCmd = $cmd
            Write-Host "  ✅ Found: $ver" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $PythonCmd) {
    Write-Host "  ❌ Python 3 not found. Please install Python 3.10+ from python.org" -ForegroundColor Red
    Write-Host "     Or install from Microsoft Store: 'winget install Python.Python.3.11'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# ── Step 2: Create/Activate Virtual Environment ──
Write-Host "[2/4] Setting up virtual environment..." -ForegroundColor Cyan
$VenvPath = Join-Path $ProjectRoot ".venv"

if (-not (Test-Path $VenvPath)) {
    Write-Host "  Creating virtual environment..." -ForegroundColor DarkGray
    & $PythonCmd -m venv $VenvPath
    Write-Host "  ✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "  ✅ Virtual environment exists" -ForegroundColor Green
}

# Activate venv
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    & $ActivateScript
} else {
    Write-Host "  ⚠️  Could not activate venv, using system Python" -ForegroundColor Yellow
}

# ── Step 3: Install Dependencies ──
Write-Host "[3/4] Installing dependencies..." -ForegroundColor Cyan
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = $PythonCmd
}

$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"

# Check if packages need updating
$InstallMarker = Join-Path $VenvPath ".deps_installed"
$RequirementsHash = (Get-FileHash $RequirementsPath -Algorithm MD5).Hash

$NeedInstall = $true
if (Test-Path $InstallMarker) {
    $StoredHash = Get-Content $InstallMarker -ErrorAction SilentlyContinue
    if ($StoredHash -eq $RequirementsHash) {
        $NeedInstall = $false
        Write-Host "  ✅ Dependencies up to date" -ForegroundColor Green
    }
}

if ($NeedInstall) {
    Write-Host "  Installing packages (this may take 1-2 minutes)..." -ForegroundColor DarkGray
    & $VenvPython -m pip install --upgrade pip --quiet 2>$null
    & $VenvPython -m pip install -r $RequirementsPath --quiet
    
    if ($LASTEXITCODE -eq 0) {
        $RequirementsHash | Set-Content $InstallMarker
        Write-Host "  ✅ All dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Some packages may have failed. Trying to continue..." -ForegroundColor Yellow
        
        # Install core packages individually as fallback
        $CorePackages = @("streamlit", "requests", "beautifulsoup4", "pandas", "plotly", "pyyaml", "Pillow")
        foreach ($pkg in $CorePackages) {
            & $VenvPython -m pip install $pkg --quiet 2>$null
        }
        
        # Try ONNX Runtime separately (may fail on some architectures)
        Write-Host "  Installing ONNX Runtime DirectML..." -ForegroundColor DarkGray
        & $VenvPython -m pip install onnxruntime-directml --quiet 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ℹ️  ONNX Runtime DirectML not available — using CPU mode" -ForegroundColor Yellow
            & $VenvPython -m pip install onnxruntime --quiet 2>$null
        }
    }
}

# ── Step 4: Launch Application ──
Write-Host "[4/4] Launching Data Prep Assistant..." -ForegroundColor Cyan
Write-Host ""
Write-Host "  🌐 Opening in browser: http://localhost:8501" -ForegroundColor Green
Write-Host "  📌 Press Ctrl+C to stop the server" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  ─────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""

$AppPath = Join-Path $ProjectRoot "app.py"

# Check for Streamlit in venv
$StreamlitPath = Join-Path $VenvPath "Scripts\streamlit.exe"
if (Test-Path $StreamlitPath) {
    & $StreamlitPath run $AppPath --server.port 8501 --browser.gatherUsageStats false
} else {
    & $VenvPython -m streamlit run $AppPath --server.port 8501 --browser.gatherUsageStats false
}
