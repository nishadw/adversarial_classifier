#!/bin/bash
set -e
REPO_URL=${1:-"<REPO_URL_HERE>"}
if [ "$REPO_URL" = "<REPO_URL_HERE>" ]; then
  echo "Please pass your repository URL as the first argument, e.g."
  echo "  bash setup_colab.sh https://github.com/yourusername/adversarial_classifier.git"
  exit 1
fi
# clone repo
git clone "$REPO_URL" repo || true
# install requirements
pip install -r repo/requirements.txt

echo "Setup complete. Run the notebook cells or: python repo/src/adversarial_classifier/train.py --epochs 1 --device cuda"
