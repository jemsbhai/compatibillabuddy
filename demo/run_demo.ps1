# demo/run_demo.ps1
# Runs through the key compatibillabuddy commands for recording a demo.
# Assumes the demo venv is already set up and activated.
#
# Usage:
#   .\.demo-env\Scripts\Activate.ps1
#   .\demo\run_demo.ps1

$ErrorActionPreference = "Stop"

function Pause-Demo {
    param([string]$Message = "Press Enter to continue...")
    Write-Host ""
    Write-Host $Message -ForegroundColor DarkGray
    Read-Host
}

# Header
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Compatibillabuddy Demo" -ForegroundColor Cyan
Write-Host "  Hardware-Aware ML Dependency Repair Agent" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Demo 1: Doctor (offline, no AI needed)
Write-Host "--- Step 1: Diagnose (no AI required) ---" -ForegroundColor Yellow
Write-Host '$ compatibuddy doctor' -ForegroundColor Green
Write-Host ""
compatibuddy doctor

Pause-Demo "Step 1 complete. Press Enter for JSON output..."

# Demo 2: Doctor JSON
Write-Host "--- Step 2: JSON output for automation ---" -ForegroundColor Yellow
Write-Host '$ compatibuddy doctor --format json' -ForegroundColor Green
Write-Host ""
compatibuddy doctor --format json

Pause-Demo "Step 2 complete. Press Enter for autonomous repair..."

# Demo 3: Autonomous repair (dry run)
Write-Host "--- Step 3: Autonomous Repair (dry run) ---" -ForegroundColor Yellow
Write-Host '$ compatibuddy repair' -ForegroundColor Green
Write-Host ""
compatibuddy repair

Pause-Demo "Step 3 complete. Demo finished!"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Demo Complete!" -ForegroundColor Cyan
Write-Host "  Try: compatibuddy repair --live" -ForegroundColor Cyan
Write-Host "  Try: compatibuddy agent" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
