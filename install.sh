#!/usr/bin/env bash
# Installer for SearchLibrium (latent class reproducibility release).
#
# Usage:
#   bash install.sh [--env NAME] [--python VERSION] [--venv] [--recreate] [--tmpdir DIR]
#
# Options:
#   --env NAME        Environment name (default: searchlibrium)
#   --python VERSION  Python version for a new environment (default: 3.11)
#   --venv            Use a Python venv in ./.venv instead of conda
#   --recreate        Delete and recreate the environment if it exists
#   --tmpdir DIR      Temporary directory for pip (useful on HPC)

set -euo pipefail

ENV_NAME="searchlibrium"
PY_VERSION="3.11"
USE_VENV=0
RECREATE=0
CUSTOM_TMPDIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)      ENV_NAME="$2"; shift 2 ;;
        --python)   PY_VERSION="$2"; shift 2 ;;
        --venv)     USE_VENV=1; shift ;;
        --recreate) RECREATE=1; shift ;;
        --tmpdir)   CUSTOM_TMPDIR="$2"; shift 2 ;;
        -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

info()  { echo -e "\n==> $*"; }
fail()  { echo -e "\nERROR: $*" >&2; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$REPO_DIR/pyproject.toml" ]] || fail "pyproject.toml not found. Run this script from the repository root."
cd "$REPO_DIR"

if [[ -n "$CUSTOM_TMPDIR" ]]; then
    mkdir -p "$CUSTOM_TMPDIR"
    export TMPDIR="$CUSTOM_TMPDIR"
fi

export PYTHONNOUSERSITE=1

# --- Environment -------------------------------------------------------------

if [[ "$USE_VENV" -eq 0 ]] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    env_exists() { conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null; }

    if env_exists; then
        if [[ "$RECREATE" -eq 1 ]]; then
            info "Removing existing conda environment '$ENV_NAME'"
            conda env remove -n "$ENV_NAME" -y
        else
            info "Using existing conda environment '$ENV_NAME'"
        fi
    fi

    if ! env_exists; then
        info "Creating conda environment '$ENV_NAME' (Python $PY_VERSION)"
        conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
    fi

    conda activate "$ENV_NAME"
    conda env config vars set PYTHONNOUSERSITE=1 >/dev/null
    conda deactivate
    conda activate "$ENV_NAME"
    ACTIVATE_CMD="conda activate $ENV_NAME"
else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
    [[ -n "$PYTHON_BIN" ]] || fail "Neither conda nor python3 was found."

    VENV_DIR="$REPO_DIR/.venv"
    if [[ -d "$VENV_DIR" && "$RECREATE" -eq 1 ]]; then
        info "Removing existing venv"
        rm -rf "$VENV_DIR"
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        info "Creating venv in $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        echo 'export PYTHONNOUSERSITE=1' >> "$VENV_DIR/bin/activate"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ACTIVATE_CMD="source $VENV_DIR/bin/activate"
fi

python - <<'EOF' || fail "Python >= 3.10 is required."
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

# --- Installation ------------------------------------------------------------

info "Upgrading pip"
python -m pip install --upgrade pip

info "Removing previous SearchLibrium installations (if any)"
python -m pip uninstall -y SearchLibrium >/dev/null 2>&1 || true

if [[ "$(uname -s)" == "Linux" ]]; then
    info "Installing SearchLibrium with CUDA-enabled JAX (first run downloads ~3-4 GB)"
else
    info "Installing SearchLibrium with CPU JAX"
fi
python -m pip install -e .

# --- Verification ------------------------------------------------------------

info "Verifying installation"
python - <<'EOF'
import os, sys, contextlib, io

with contextlib.redirect_stdout(io.StringIO()):
    import SearchLibrium
import jax

path = os.path.dirname(SearchLibrium.__file__)
devices = jax.devices()
backend = "GPU" if any(d.platform == "gpu" for d in devices) else "CPU"

print(f"  Package : {path}")
print(f"  Python  : {sys.version.split()[0]}")
print(f"  JAX     : {jax.__version__} ({backend})")
print(f"  Devices : {devices}")

expected = os.path.join(os.getcwd(), "src", "SearchLibrium")
if os.path.realpath(path) != os.path.realpath(expected):
    sys.exit(f"\nERROR: SearchLibrium was imported from {path}, not from this repository.")
EOF

info "Installation complete. Activate the environment with:"
echo "    $ACTIVATE_CMD"
