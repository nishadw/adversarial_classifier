from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from sklearn.metrics import roc_auc_score

from adversarial_classifier.attacks import pgd_linf_attack
from adversarial_classifier.data import build_cifar10_loaders
from adversarial_classifier.models import SimpleCNN


def load_checkpoint(path: str):
    """Load JAX/Flax checkpoint."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["params"]


def eval_clean(params: dict, apply_fn, batches: list) -> float:
    total_correct, total = 0, 0
    for images, labels in batches:
        logits = apply_fn({"params": params}, images, training=False)
        total_correct += jnp.sum(jnp.argmax(logits, axis=1) == labels).item()
        total += labels.shape[0]
    return total_correct / max(total, 1)


def eval_loss_acc_auc(params: dict, apply_fn, batches: list):
    total_loss = 0.0
    total = 0
    probs_list = []
    labels_list = []
    for images, labels in batches:
        logits = apply_fn({"params": params}, images, training=False)
        probs = jax.nn.softmax(logits)
        loss = jnp.mean((probs - jax.nn.one_hot(labels, 10)) ** 2)
        total_loss += float(loss) * labels.shape[0]
        total += labels.shape[0]
        probs_list.append(jnp.asarray(probs))
        labels_list.append(jnp.asarray(labels))
    avg_loss = total_loss / max(total, 1)
    probs_all = jnp.concatenate(probs_list, axis=0)
    labels_all = jnp.concatenate(labels_list, axis=0)
    try:
        auc = float(roc_auc_score(jax.nn.one_hot(labels_all, 10), probs_all, average='macro', multi_class='ovr'))
    except Exception:
        auc = float('nan')
    preds = jnp.argmax(probs_all, axis=1)
    acc = float(jnp.mean(preds == labels_all))
    return avg_loss, acc, auc


def eval_under_attack(
    params: dict,
    apply_fn,
    attacker_params: dict,
    attacker_apply_fn,
    batches: list,
    eps: float,
    alpha: float,
    steps: int,
) -> float:
    total_correct, total = 0, 0
    for images, labels in batches:
        adv = pgd_linf_attack(
            attacker_apply_fn,
            attacker_params,
            images,
            labels,
            eps,
            alpha,
            steps,
        )
        logits = apply_fn({"params": params}, adv, training=False)
        total_correct += jnp.sum(jnp.argmax(logits, axis=1) == labels).item()
        total += labels.shape[0]

    return total_correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare trained CIFAR-10 models (JAX/Flax)")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--archive-path", type=str, default=None)
    p.add_argument("--download-if-missing", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--standard-ckpt", type=str, required=True)
    p.add_argument("--whitebox-ckpt", type=str, required=True)
    p.add_argument("--blackbox-ckpt", type=str, required=True)
    p.add_argument("--pgd-eps", type=float, default=8 / 255)
    p.add_argument("--pgd-alpha", type=float, default=2 / 255)
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--output-csv", type=str, default="./artifacts/comparison.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_batches, val_batches, test_batches = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        archive_path=args.archive_path,
        download_if_missing=args.download_if_missing,
    )

    model = SimpleCNN(num_classes=10)

    models_data = {
        "simple_cnn": load_checkpoint(args.standard_ckpt),
        "whitebox_pgd_trained": load_checkpoint(args.whitebox_ckpt),
        "blackbox_pgd_trained": load_checkpoint(args.blackbox_ckpt),
    }

    source_attacker_params = models_data["simple_cnn"]

    rows = []
    for name, params in models_data.items():
        clean_loss, clean_acc, clean_auc = eval_loss_acc_auc(params, model.apply, test_batches)
        whitebox = eval_under_attack(
            params,
            model.apply,
            params,
            model.apply,
            test_batches,
            eps=args.pgd_eps,
            alpha=args.pgd_alpha,
            steps=args.pgd_steps,
        )
        transfer = eval_under_attack(
            params,
            model.apply,
            source_attacker_params,
            model.apply,
            test_batches,
            eps=args.pgd_eps,
            alpha=args.pgd_alpha,
            steps=args.pgd_steps,
        )
        row = {
            "model": name,
            "test_loss": clean_loss,
            "test_acc": clean_acc,
            "test_auc": clean_auc,
            "whitebox_pgd_acc": whitebox,
            "transfer_pgd_acc_from_simple": transfer,
        }
        rows.append(row)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "test_loss",
                "test_acc",
                "test_auc",
                "whitebox_pgd_acc",
                "transfer_pgd_acc_from_simple",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Model comparison on CIFAR-10 (JAX/Flax)")
    print("=" * 72)
    print(f"{'model':28s} {'test_acc':>9s} {'test_auc':>9s} {'whitebox_pgd':>14s} {'transfer_pgd':>14s}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['model']:28s} {r['test_acc'] * 100:8.2f}% {r['test_auc'] * 100:8.2f}% "
            f"{r['whitebox_pgd_acc'] * 100:13.2f}% {r['transfer_pgd_acc_from_simple'] * 100:13.2f}%"
        )
    print("=" * 72)
    print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    main()

