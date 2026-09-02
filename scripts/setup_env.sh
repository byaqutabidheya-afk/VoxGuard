#!/usr/bin/env bash
# =============================================================================
# scripts/setup_env.sh — VoxGuard development environment bootstrap
#
# Usage:
#   bash scripts/setup_env.sh
#
# What this script does:
#   1. Creates a Python virtual environment at .venv (idempotent — skips
#      creation if .venv already exists)
#   2. Activates the venv
#   3. Upgrades pip to the latest version available in the venv
#   4. Installs all dependencies from requirements.txt
#   5. Checks that ffmpeg is on PATH and prints its version; if missing,
#      prints OS-specific installation instructions (does NOT attempt to
#      install ffmpeg itself)
#   6. Prints the CPU core count as a reminder to benchmark
#      torch.set_num_threads() before running a full local embedding
#      extraction in Phase 2
#
# Idempotency: safe to re-run at any time.  Re-running upgrades pip and
# re-syncs installed packages against requirements.txt without touching
# the venv layout or your existing package cache.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve the repository root regardless of the calling directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
REQUIREMENTS="${REPO_ROOT}/requirements.txt"

echo "=== VoxGuard environment setup ==="
echo "Repository root : ${REPO_ROOT}"
echo "Venv location   : ${VENV_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Create virtual environment (skip if it already exists)
# ---------------------------------------------------------------------------
if [ -d "${VENV_DIR}" ]; then
    echo "[1/5] Virtual environment already exists at .venv — skipping creation."
else
    echo "[1/5] Creating virtual environment at .venv …"
    python3 -m venv "${VENV_DIR}"
    echo "      Done."
fi

# ---------------------------------------------------------------------------
# 2. Activate the virtual environment
# ---------------------------------------------------------------------------
echo "[2/5] Activating virtual environment …"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
echo "      Python : $(python --version)"
echo "      pip    : $(pip --version)"

# ---------------------------------------------------------------------------
# 3. Upgrade pip
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Upgrading pip …"
pip install --quiet --upgrade pip
echo "      pip upgraded to: $(pip --version)"

# ---------------------------------------------------------------------------
# 4. Install project dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Installing dependencies from requirements.txt …"
echo "      NOTE: torch/torchaudio will be installed as CPU-only builds."
echo "            GPU work happens on Kaggle (torch+CUDA pre-installed there)."
echo ""
pip install --requirement "${REQUIREMENTS}"
echo ""
echo "      Dependencies installed successfully."

# ---------------------------------------------------------------------------
# 5. Check ffmpeg availability
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Checking ffmpeg …"
if command -v ffmpeg &>/dev/null; then
    echo "      ffmpeg found:"
    ffmpeg -version 2>&1 | head -1
else
    echo "      ⚠  WARNING: ffmpeg not found on PATH."
    echo ""
    echo "      librosa and torchaudio use ffmpeg to decode many audio formats."
    echo "      Install it for your OS before running VoxGuard:"
    echo ""
    echo "      macOS (Homebrew):"
    echo "          brew install ffmpeg"
    echo ""
    echo "      Ubuntu / Debian:"
    echo "          sudo apt update && sudo apt install -y ffmpeg"
    echo ""
    echo "      Windows (winget):"
    echo "          winget install --id Gyan.FFmpeg"
    echo ""
    echo "      Windows (Chocolatey):"
    echo "          choco install ffmpeg"
    echo ""
    echo "      After installing, open a new terminal and re-run this script"
    echo "      (or just confirm with: ffmpeg -version)"
fi

# ---------------------------------------------------------------------------
# 6. Print CPU core count
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Detecting CPU core count …"
CPU_CORES=$(python -c "import os; print(os.cpu_count())")
echo "      Logical CPU cores: ${CPU_CORES}"
echo ""
echo "      ──────────────────────────────────────────────────────────────"
echo "      Phase 2 reminder: before running a full local embedding"
echo "      extraction smoke test, benchmark torch.set_num_threads() to"
echo "      find the sweet spot for this machine (${CPU_CORES} logical cores)."
echo "      A good starting point is half the physical core count."
echo "      See Phase 2 notebook / scripts for the benchmark harness."
echo "      ──────────────────────────────────────────────────────────────"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment in a new terminal:"
echo "    source .venv/bin/activate        # macOS / Linux / Git Bash"
echo "    .venv\\Scripts\\activate           # Windows PowerShell / cmd"
echo ""
