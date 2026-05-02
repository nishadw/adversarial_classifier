Quick Colab setup for this repo

Steps

- Open the notebook `colab/setup_colab.ipynb` in Google Colab.
- In Colab: Runtime -> Change runtime type -> Hardware accelerator -> select `GPU`.
- (Optional) Use Colab Pro to get priority GPU access.
- Mount your Google Drive (first code cell in notebook).
- Either clone your GitHub repo into Colab or upload the repo to Drive and adjust the paths.

Common commands (run in notebook cells):

```bash
# clone (replace with your repo URL)
!git clone https://github.com/yourusername/adversarial_classifier.git repo
# install requirements
!pip install -r repo/requirements.txt
# run a quick training/test (example)
!python repo/src/adversarial_classifier/train.py --epochs 1 --device cuda
```

Notes

- Replace the clone URL with your repo's URL if you push this workspace to GitHub.
- If you prefer to use files from Google Drive, upload the repository folder to Drive and change paths to `/content/drive/MyDrive/...`.
- If any packages require system dependencies, install them in the notebook with apt-get before pip.
