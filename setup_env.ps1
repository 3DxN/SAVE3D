# SAVE-3D Environment Setup Script (Windows PowerShell)
# =====================================================
# Usage: .\setup_env.ps1 [-GPU]
#
# Options:
#   -GPU    Install GPU acceleration packages (requires NVIDIA CUDA)
param(
    [switch]$GPU
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SAVE-3D Environment Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if uv is installed
$uvPath = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvPath) {
    Write-Host "[!] uv not found. Installing..." -ForegroundColor Yellow
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

Write-Host "[1/4] Installing Python 3.10..." -ForegroundColor Green
uv python install 3.10

Write-Host "[2/4] Creating virtual environment..." -ForegroundColor Green
uv venv .venv --python 3.10

Write-Host "[3/4] Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

if ($GPU) {
    Write-Host "[4/4] Installing dependencies (GPU version)..." -ForegroundColor Green
    uv pip install -r requirements-gpu.txt
} else {
    Write-Host "[4/4] Installing dependencies (CPU version)..." -ForegroundColor Green
    uv pip install -r requirements.txt
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment in the future:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To run preprocessing:" -ForegroundColor Yellow
Write-Host "  python main_preprocessing.py" -ForegroundColor White
Write-Host ""
Write-Host "To run the viewer:" -ForegroundColor Yellow
Write-Host "  python main_app.py <path_to_zarr>" -ForegroundColor White
Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Yellow
python -c "import napari; import pyvista; import zarr; print('All core packages installed successfully')"

if ($GPU) {
    python -c "import cupy; count = cupy.cuda.runtime.getDeviceCount(); print('CuPy GPU: ' + str(count) + ' device(s) available')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] CuPy GPU not available (CUDA may not be installed)" -ForegroundColor Yellow
    }
}
