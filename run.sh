#!/bin/sh
set -eu
# run.sh - setup and run a short training + plotting inside Colab
# Usage in Colab after cloning your repo: `sh run.sh [REPO_DIR]`

REPO_DIR=${1:-"repo"}

if [ -d "$REPO_DIR" ]; then
  echo "Using repo dir: $REPO_DIR"
  cd "$REPO_DIR"
elif [ -f "requirements.txt" ] && [ -d "src" ]; then
  echo "Repo dir '$REPO_DIR' not found; using current directory: $(pwd)"
else
  echo "Could not find repo directory '$REPO_DIR'."
  echo "Run from /content with: sh repo/run.sh repo"
  echo "Or run inside the repo with: sh run.sh"
  exit 1
fi

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Checking for GPU (nvidia-smi)..."
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected"
  echo "Attempting to install JAX with CUDA support (best-effort)."
  # Try common JAX CUDA wheel installation; if it fails, fall back to CPU jax
  python -m pip install --upgrade "jax[cuda11_cudnn11]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || {
    echo "Failed to install GPU JAX wheel; installing CPU JAX instead."
    python -m pip install --upgrade jax jaxlib || true
  }
else
  echo "No GPU detected — installing CPU JAX"
  python -m pip install --upgrade jax jaxlib || true
fi

echo "Installing Python requirements..."
python -m pip install -r requirements.txt

# Some environments (Colab) may need a specific PyTorch wheel; install cu118 if GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Installing PyTorch (cu118 wheel) — if you prefer a different CUDA, edit this script."
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || {
    echo "PyTorch cu118 wheel failed, installing default torch wheel"
    python -m pip install torch torchvision torchaudio || true
  }
else
  echo "Installing CPU PyTorch"
  python -m pip install torch torchvision torchaudio || true
fi

echo "Preparing output directories..."
mkdir -p checkpoints
mkdir -p plots

echo "Running a short training session (2 epochs) — this may take several minutes depending on runtime."
python src/adversarial_classifier/train.py --mode standard --epochs 2 --batch-size 128 --download-if-missing --output checkpoints/standard.chk

echo "Generating plot from history CSV..."
python src/adversarial_classifier/plot_results.py checkpoints/standard_history.csv --output-dir plots

echo "Run complete. Artifacts:"
ls -l checkpoints || true
ls -l plots || true

echo "If running in Colab, display the plot with: from IPython.display import Image; Image('plots/standard_chart.png')"
