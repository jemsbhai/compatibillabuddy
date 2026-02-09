# demo/setup_broken_env.ps1
# Creates a throwaway venv with deliberately broken ML dependencies
# for demonstrating compatibillabuddy's diagnosis and repair capabilities.
#
# Usage:
#   .\demo\setup_broken_env.ps1
#
# After running this script:
#   .\.demo-env\Scripts\Activate.ps1
#   compatibuddy doctor
#   compatibuddy repair
#   compatibuddy repair --live

$ErrorActionPreference = "Stop"

$VENV_DIR = ".demo-env"

Write-Host ""
Write-Host "=== Compatibillabuddy Demo Setup ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create fresh venv
if (Test-Path $VENV_DIR) {
    Write-Host "Removing existing demo venv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VENV_DIR
}

Write-Host "Creating demo venv in $VENV_DIR..." -ForegroundColor Green
python -m venv $VENV_DIR

# Step 2: Activate and install compatibillabuddy
Write-Host "Installing compatibillabuddy[agent]..." -ForegroundColor Green
& "$VENV_DIR\Scripts\python.exe" -m pip install --upgrade pip --quiet
& "$VENV_DIR\Scripts\python.exe" -m pip install -e ".[agent]" --quiet

# Step 3: Install deliberately conflicting packages
Write-Host "Installing deliberately conflicting packages..." -ForegroundColor Yellow
Write-Host ""

# Install CPU-only torch on a CUDA system (classic CUDA mismatch)
Write-Host "  - torch (CPU-only build on CUDA system)" -ForegroundColor Yellow
& "$VENV_DIR\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet 2>$null

# Install an older numpy that may conflict
Write-Host "  - numpy 1.26.4 (potential ABI mismatch with newer packages)" -ForegroundColor Yellow
& "$VENV_DIR\Scripts\python.exe" -m pip install "numpy==1.26.4" --quiet 2>$null

# Install pandas and scikit-learn (may have numpy ABI issues)
Write-Host "  - pandas, scikit-learn (numpy ABI boundary)" -ForegroundColor Yellow
& "$VENV_DIR\Scripts\python.exe" -m pip install pandas scikit-learn --quiet 2>$null

Write-Host ""
Write-Host "=== Demo environment ready! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Now run:" -ForegroundColor White
Write-Host "  $VENV_DIR\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "  compatibuddy doctor                  # See the issues" -ForegroundColor Green
Write-Host "  compatibuddy doctor --format json     # Machine-readable" -ForegroundColor Green
Write-Host "  compatibuddy repair                   # Dry-run repair plan" -ForegroundColor Green
Write-Host "  compatibuddy repair --live            # Actually fix it" -ForegroundColor Green
Write-Host "  compatibuddy agent                    # Interactive chat" -ForegroundColor Green
Write-Host ""
