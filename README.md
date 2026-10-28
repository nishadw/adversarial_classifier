# adversarial_classifier

**What adversarial training costs, measured.** Three CIFAR-10 classifiers trained under one
harness — clean, white-box PGD adversarial training, and transfer-based black-box
PGD against a surrogate — then evaluated against each other on clean and attacked
data.

The point is the comparison, not any single number: how much clean accuracy a
defence gives up, and how much of a white-box defence survives when the adversary
only has a surrogate to differentiate through.

| Mode | Training data | Adversary at train time |
|---|---|---|
| `standard` | clean | none |
| `whitebox_pgd` | clean + PGD examples from the model itself | full gradient access |
| `blackbox_pgd` | clean + PGD examples from a frozen surrogate | surrogate gradients only |

## The model and the attack

A six-convolution CNN with GroupNorm (128 → 256 → 512) and a three-layer head,
in PyTorch. The attack is untargeted PGD under an L∞ budget, with a random start
inside the ball and the perturbation re-projected after every step. Defaults are
ε = 8/255, α = 2/255, 7 steps, on images normalised to [−1, 1].

Both the training loss and the attack objective are computed on softmax
probabilities rather than logits, so the attack maximises exactly the quantity
training minimises.

| Path | What it is |
|---|---|
| `models.py` | `SimpleCNN` — the architecture all three modes share |
| `attacks.py` | `pgd_linf_attack` — untargeted L∞ PGD, restores `model.training` on exit |
| `data.py` | CIFAR-10 loaders, normalisation, train/val/test split |
| `train.py` | The one entry point; `--mode` picks the regime |
| `run.ipynb` | End-to-end run and the comparison plots |

## Running it

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
```

Train the three models in order — the black-box run needs the standard model as
its surrogate:

```bash
python train.py --mode standard     --epochs 50 --output artifacts/standard.pt
python train.py --mode whitebox_pgd --epochs 50 --output artifacts/whitebox_pgd.pt
python train.py --mode blackbox_pgd --epochs 50 --output artifacts/blackbox_pgd.pt \
  --surrogate-checkpoint artifacts/standard.pt
```

| Flag | Default | |
|---|---|---|
| `--mode` | required | `standard`, `whitebox_pgd`, `blackbox_pgd` |
| `--output` | required | checkpoint path |
| `--epochs` | 30 | |
| `--batch-size` | 128 | |
| `--lr` | 0.1 | |
| `--pgd-eps` | 8/255 | L∞ budget |
| `--pgd-alpha` | 2/255 | step size |
| `--pgd-steps` | 7 | |
| `--surrogate-checkpoint` | — | required for `blackbox_pgd` |
| `--data-root` | `./data` | CIFAR-10 is downloaded here if absent |
| `--seed` | 42 | |

Each run writes the checkpoint plus a per-epoch CSV beside it — train and
validation loss, accuracy and AUC — which is what the notebook plots.

## Related

Part of the same line of work as the adversarially hardened on-device vision
cascade written up at [nishad.page/projects](https://www.nishad.page/projects#atak).
