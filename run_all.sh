#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-./.venv/bin/python}
ARCHIVE=${ARCHIVE:-/Users/nishad/Downloads/cifar-100-python.tar}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-128}
NUM_WORKERS=${NUM_WORKERS:-0}
LR=${LR:-0.1}
OUT_DIR=artifacts
LOG_DIR=${OUT_DIR}/logs
mkdir -p ${LOG_DIR}
mkdir -p ${OUT_DIR}

echo "=== CIFAR-10 Adversarial Classifier Training ==="
echo "Epochs: ${EPOCHS}, Batch: ${BATCH}, LR: ${LR}"
echo ""

# Train standard model
echo "📊 Training standard model for ${EPOCHS} epochs"
${PYTHON} -m adversarial_classifier.train \
  --mode standard \
  --data-root ./data \
  --archive-path "${ARCHIVE}" \
  --download-if-missing \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH} \
  --num-workers ${NUM_WORKERS} \
  --lr ${LR} \
  --output ${OUT_DIR}/standard.pt 2>&1 | tee ${LOG_DIR}/train_standard.log

# Train whitebox PGD model
echo ""
echo "📊 Training whitebox PGD model for ${EPOCHS} epochs"
${PYTHON} -m adversarial_classifier.train \
  --mode whitebox_pgd \
  --data-root ./data \
  --archive-path "${ARCHIVE}" \
  --download-if-missing \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH} \
  --num-workers ${NUM_WORKERS} \
  --lr ${LR} \
  --pgd-steps 7 \
  --pgd-eps 0.0313725 \
  --pgd-alpha 0.0078431 \
  --output ${OUT_DIR}/whitebox_pgd.pt 2>&1 | tee ${LOG_DIR}/train_whitebox_pgd.log

# Train blackbox PGD model (uses standard as surrogate)
echo ""
echo "📊 Training blackbox PGD model for ${EPOCHS} epochs (surrogate: standard)"
${PYTHON} -m adversarial_classifier.train \
  --mode blackbox_pgd \
  --surrogate-checkpoint ${OUT_DIR}/standard.pt \
  --data-root ./data \
  --archive-path "${ARCHIVE}" \
  --download-if-missing \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH} \
  --num-workers ${NUM_WORKERS} \
  --lr ${LR} \
  --pgd-steps 7 \
  --pgd-eps 0.0313725 \
  --pgd-alpha 0.0078431 \
  --output ${OUT_DIR}/blackbox_pgd.pt 2>&1 | tee ${LOG_DIR}/train_blackbox_pgd.log

# Plot training results
echo ""
echo "📈 Creating training curves"
${PYTHON} -m adversarial_classifier.plot_results \
  ${OUT_DIR}/standard_history.csv \
  ${OUT_DIR}/whitebox_pgd_history.csv \
  ${OUT_DIR}/blackbox_pgd_history.csv \
  --output-dir ${OUT_DIR} 2>&1 | tee -a ${LOG_DIR}/plot.log

# Compare and produce CSV
echo ""
echo "📋 Comparing models and producing final CSV"
${PYTHON} -m adversarial_classifier.compare_models \
  --data-root ./data \
  --archive-path "${ARCHIVE}" \
  --download-if-missing \
  --batch-size ${BATCH} \
  --standard-ckpt ${OUT_DIR}/standard.pt \
  --whitebox-ckpt ${OUT_DIR}/whitebox_pgd.pt \
  --blackbox-ckpt ${OUT_DIR}/blackbox_pgd.pt \
  --pgd-steps 20 \
  --output-csv ${OUT_DIR}/comparison_final.csv 2>&1 | tee ${LOG_DIR}/compare.log

echo ""
echo "✅ All done!"
echo "📁 Artifacts saved in: ${OUT_DIR}"
echo "   - Checkpoints: standard.pt, whitebox_pgd.pt, blackbox_pgd.pt"
echo "   - Training history CSVs: *_history.csv"
echo "   - Charts: *_chart.png"
echo "   - Comparison: comparison_final.csv"
echo "   - Logs: logs/*"

