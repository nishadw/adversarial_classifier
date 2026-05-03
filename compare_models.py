from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from attacks import pgd_linf_attack
from data import build_cifar10_loaders
from models import SimpleCNN


def load_model(path: str, device) -> SimpleCNN:
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def mse_loss(logits, labels):
    return F.mse_loss(F.softmax(logits, dim=1), F.one_hot(labels, 10).float())


@torch.no_grad()
def eval_loss_acc_auc(model, loader, device):
    total_loss = total = 0
    all_probs, all_labels = [], []
    for imgs, lbs in loader:
        imgs, lbs = imgs.to(device), lbs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1)
        total_loss += mse_loss(logits, lbs).item() * imgs.size(0)
        total      += imgs.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(lbs.cpu())
    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    acc = (probs.argmax(1) == labels).float().mean().item()
    try:
        auc = float(roc_auc_score(F.one_hot(labels, 10).numpy(), probs.numpy(),
                                  average="macro", multi_class="ovr"))
    except Exception:
        auc = float("nan")
    return total_loss / max(total, 1), acc, auc


def eval_under_attack(victim, attacker, loader, device, eps, alpha, steps):
    victim.eval()
    total_correct = total = 0
    for imgs, lbs in tqdm(loader, desc="adv eval", leave=False):
        imgs, lbs = imgs.to(device), lbs.to(device)
        adv = pgd_linf_attack(attacker, imgs, lbs, eps, alpha, steps)
        total_correct += victim(adv).argmax(1).eq(lbs).sum().item()
        total         += imgs.size(0)
    return total_correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CIFAR-10 models (PyTorch)")
    p.add_argument("--data-root",     default="./data")
    p.add_argument("--batch-size",    type=int,   default=128)
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--standard-ckpt", required=True)
    p.add_argument("--whitebox-ckpt", required=True)
    p.add_argument("--blackbox-ckpt", required=True)
    p.add_argument("--pgd-eps",       type=float, default=8 / 255)
    p.add_argument("--pgd-alpha",     type=float, default=2 / 255)
    p.add_argument("--pgd-steps",     type=int,   default=20)
    p.add_argument("--output-csv",    default="./artifacts/comparison.csv")
    p.add_argument("--download",      action="store_true")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    _, _, test_loader = build_cifar10_loaders(
        args.data_root, args.batch_size, args.num_workers, download=args.download
    )

    models_dict = {
        "simple_cnn":          load_model(args.standard_ckpt, device),
        "whitebox_pgd_trained": load_model(args.whitebox_ckpt, device),
        "blackbox_pgd_trained": load_model(args.blackbox_ckpt, device),
    }
    source_attacker = models_dict["simple_cnn"]

    rows = []
    for name, model in models_dict.items():
        clean_loss, clean_acc, clean_auc = eval_loss_acc_auc(model, test_loader, device)
        wb_acc = eval_under_attack(model, model,            test_loader, device,
                                   args.pgd_eps, args.pgd_alpha, args.pgd_steps)
        tf_acc = eval_under_attack(model, source_attacker,  test_loader, device,
                                   args.pgd_eps, args.pgd_alpha, args.pgd_steps)
        rows.append(dict(model=name, test_loss=clean_loss, test_acc=clean_acc,
                         test_auc=clean_auc, whitebox_pgd_acc=wb_acc,
                         transfer_pgd_acc_from_simple=tf_acc))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Model comparison on CIFAR-10 (PyTorch)")
    print("=" * 80)
    print(f"{'model':28s} {'test_acc':>9s} {'test_auc':>9s} "
          f"{'whitebox_pgd':>14s} {'transfer_pgd':>14s}")
    print("-" * 80)
    for r in rows:
        print(f"{r['model']:28s} {r['test_acc']*100:8.2f}% {r['test_auc']*100:8.2f}% "
              f"{r['whitebox_pgd_acc']*100:13.2f}% "
              f"{r['transfer_pgd_acc_from_simple']*100:13.2f}%")
    print("=" * 80)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
