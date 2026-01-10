#!/bin/bash
# SAVE-3D Environment Setup Script (macOS/Linux)
# =====================================================
# Usage: ./setup_env.sh [--gpu]
#
# Options:
#   --gpu    Install GPU acceleration packages (requires NVIDIA CUDA)

set -e

GPU_MODE=false
if [[ "$1" == "--gpu" ]]; then
    GPU_MODE=true
fi

echo "============================================"
echo "  SAVE-3D Environment Setup"
echo "============================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "[!] uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "[1/4] Installing Python 3.10..."
uv python install 3.10

echo "[2/4] Creating virtual environment..."
uv venv .venv --python 3.10

echo "[3/4] Activating virtual environment..."
source .venv/bin/activate

if $GPU_MODE; then
    echo "[4/4] Installing dependencies (GPU version)..."
    uv pip install -r requirements-gpu.txt
else
    echo "[4/4] Installing dependencies (CPU version)..."
    uv pip install -r requirements.txt
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To run preprocessing:"
echo "  python main_preprocessing.py"
echo ""
echo "To run the viewer:"
echo "  python main_app.py <path_to_zarr>"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import napari; import pyvista; import zarr; print('✓ All core packages installed successfully')"

if $GPU_MODE; then
    python -c "import cupy; print(f'✓ CuPy GPU: {cupy.cuda.runtime.getDeviceCount()} device(s) available')" 2>/dev/null || \
        echo "[!] CuPy GPU not available (CUDA may not be installed)"
fi

