from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from attacks import pgd_linf_attack
from data import build_cifar10_loaders
from models import SimpleCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mse_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(F.softmax(logits, dim=1), F.one_hot(labels, 10).float())


def train_epoch(model, loader, optimizer, device, mode, eps, alpha, steps, surrogate=None):
    model.train()
    total_loss = total_correct = total = 0
    for imgs, lbs in tqdm(loader, desc="train", leave=False):
        imgs, lbs = imgs.to(device), lbs.to(device)
        if mode == "whitebox_pgd":
            imgs = pgd_linf_attack(model, imgs, lbs, eps, alpha, steps)
            model.train()
        elif mode == "blackbox_pgd":
            imgs = pgd_linf_attack(surrogate, imgs, lbs, eps, alpha, steps)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = mse_loss(logits, lbs)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * imgs.size(0)
        total_correct += logits.argmax(1).eq(lbs).sum().item()
        total         += imgs.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CIFAR-10 (PyTorch)")
    p.add_argument("--mode", choices=["standard", "whitebox_pgd", "blackbox_pgd"], required=True)
    p.add_argument("--data-root",            default="./data")
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--lr",          type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--pgd-eps",     type=float, default=8 / 255)
    p.add_argument("--pgd-alpha",   type=float, default=2 / 255)
    p.add_argument("--pgd-steps",   type=int, default=7)
    p.add_argument("--surrogate-checkpoint", default=None)
    p.add_argument("--output",      required=True)
    p.add_argument("--download",    action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "blackbox_pgd" and not args.surrogate_checkpoint:
        raise ValueError("--surrogate-checkpoint required for blackbox_pgd")

    set_seed(args.seed)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_cifar10_loaders(
        args.data_root, args.batch_size, args.num_workers, download=args.download
    )

    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

    surrogate = None
    if args.mode == "blackbox_pgd":
        surrogate = SimpleCNN().to(device)
        surrogate.load_state_dict(torch.load(args.surrogate_checkpoint, map_location=device))
        surrogate.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    history  = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, device, args.mode,
            args.pgd_eps, args.pgd_alpha, args.pgd_steps, surrogate,
        )
        vl_loss, vl_acc, vl_auc = evaluate(model, val_loader,  device)
        te_loss, te_acc, te_auc = evaluate(model, test_loader, device)

        row = dict(epoch=epoch,
                   train_loss=tr_loss, train_acc=tr_acc,
                   val_loss=vl_loss,   val_acc=vl_acc,   val_auc=vl_auc,
                   test_loss=te_loss,  test_acc=te_acc,  test_auc=te_auc)
        history.append(row)
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={tr_acc*100:.1f}% loss={tr_loss:.4f} | "
              f"val={vl_acc*100:.1f}% | test={te_acc*100:.1f}%")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), out_path)

    csv_path = out_path.parent / f"{out_path.stem}_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    print(f"Best test acc: {best_acc*100:.2f}%  →  {out_path}")
    print(f"History saved: {csv_path}")


if __name__ == "__main__":
    main()
