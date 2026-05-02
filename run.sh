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
  HAS_GPU=1
else
  echo "No GPU detected"
  HAS_GPU=0
fi

# Optional overrides from the caller:
# FORCE_CPU=1 sh run.sh
# FORCE_GPU=1 sh run.sh
if [ "${FORCE_CPU:-0}" = "1" ]; then
  echo "FORCE_CPU=1 -> running on CPU"
  JAX_RUN_PLATFORM=cpu
fi
if [ "${FORCE_GPU:-0}" = "1" ]; then
  if [ "$HAS_GPU" -eq 1 ]; then
    echo "FORCE_GPU=1 -> forcing GPU"
    JAX_RUN_PLATFORM=gpu
  else
    echo "FORCE_GPU=1 requested but no GPU runtime detected; keeping CPU"
  fi
fi

echo "Installing Python requirements..."
python -m pip install -r requirements.txt

echo "Installing a clean JAX stack after requirements (prevents version mismatch)..."
python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax_cuda12_plugin jax_cuda12_pjrt >/dev/null 2>&1 || true

if [ "$HAS_GPU" -eq 1 ] && [ "${FORCE_CPU:-0}" != "1" ]; then
  echo "Installing GPU JAX for Colab T4 (CUDA 12)..."
  if python -m pip install --upgrade "jax[cuda12]"; then
    echo "Running JAX GPU smoke test..."
    if python - <<'PY'
import jax
import jax.numpy as jnp
devices = jax.devices()
print("JAX devices:", devices)
assert any(d.platform == "gpu" for d in devices)
# small compute smoke test
x = jnp.ones((1024, 1024), dtype=jnp.float32)
print((x @ x).shape)
PY
    then
      echo "JAX GPU smoke test passed."
      JAX_RUN_PLATFORM=gpu
    else
      echo "JAX GPU smoke test failed. Falling back to CPU JAX."
      python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax_cuda12_plugin jax_cuda12_pjrt >/dev/null 2>&1 || true
      python -m pip install --upgrade jax jaxlib
      export JAX_PLATFORMS=cpu
      JAX_RUN_PLATFORM=cpu
    fi
  else
    echo "GPU JAX install failed. Falling back to CPU JAX."
    python -m pip install --upgrade jax jaxlib
    export JAX_PLATFORMS=cpu
    JAX_RUN_PLATFORM=cpu
  fi
else
  echo "Installing CPU JAX..."
  python -m pip install --upgrade jax jaxlib
  export JAX_PLATFORMS=cpu
  JAX_RUN_PLATFORM=cpu
fi

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

# Repo uses src-layout; expose it for module imports.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Running a short training session (2 epochs) — this may take several minutes depending on runtime."
if [ "$JAX_RUN_PLATFORM" = "gpu" ]; then
  echo "Training with GPU (JAX_PLATFORMS=cuda)"
  # On Colab T4, force CUDA backend explicitly to avoid ROCm probing errors.
  if ! JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda python -m adversarial_classifier.train --mode standard --epochs 2 --batch-size 128 --download-if-missing --output checkpoints/standard.chk; then
    echo "GPU training failed (possible JAX/CUDA runtime issue). Retrying on CPU so outputs still get generated..."
    JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -m adversarial_classifier.train --mode standard --epochs 2 --batch-size 128 --download-if-missing --output checkpoints/standard.chk
  fi
else
  echo "Training with CPU (JAX_PLATFORMS=cpu)"
  JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -m adversarial_classifier.train --mode standard --epochs 2 --batch-size 128 --download-if-missing --output checkpoints/standard.chk
fi

echo "Generating plot from history CSV..."
python -m adversarial_classifier.plot_results checkpoints/standard_history.csv --output-dir plots

echo "Run complete. Artifacts:"
ls -l checkpoints || true
ls -l plots || true

echo "If running in Colab, display the plot with: from IPython.display import Image; Image('plots/standard_chart.png')"
