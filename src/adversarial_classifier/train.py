from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm

from adversarial_classifier.attacks import pgd_linf_attack
from adversarial_classifier.data import build_cifar10_loaders
from sklearn.metrics import roc_auc_score
from adversarial_classifier.models import SimpleCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    return key


def create_train_state(
    key: jax.random.PRNGKey, model: nn.Module, learning_rate: float
) -> train_state.TrainState:
    """Create a train state with model params and optimizer."""
    dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
    params = model.init(key, dummy_input, training=True)["params"]

    tx = optax.sgd(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray) -> tuple:
    """Train step for standard training."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x, training=True)
        loss = jnp.mean((jax.nn.softmax(logits) - jax.nn.one_hot(y, 10)) ** 2)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits


def train_step_adversarial(
    state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    eps: float,
    alpha: float,
    pgd_steps: int,
) -> tuple:
    """Train step with white-box PGD adversarial training."""
    x_adv = pgd_linf_attack(state.apply_fn, state.params, x, y, eps, alpha, pgd_steps)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x_adv, training=True)
        loss = jnp.mean((jax.nn.softmax(logits) - jax.nn.one_hot(y, 10)) ** 2)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits


def train_step_blackbox(
    state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    surrogate_params: dict,
    apply_fn,
    eps: float,
    alpha: float,
    pgd_steps: int,
) -> tuple:
    """Train step with black-box PGD (transfer attack from surrogate)."""
    x_adv = pgd_linf_attack(apply_fn, surrogate_params, x, y, eps, alpha, pgd_steps)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x_adv, training=True)
        loss = jnp.mean((jax.nn.softmax(logits) - jax.nn.one_hot(y, 10)) ** 2)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits


def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    preds = jnp.argmax(logits, axis=1)
    return float(jnp.mean(preds == labels))


def evaluate_clean(state: train_state.TrainState, batches: list) -> float:
    total_correct = 0
    total = 0
    for images, labels in batches:
        logits = state.apply_fn({"params": state.params}, images, training=False)
        total_correct += jnp.sum(jnp.argmax(logits, axis=1) == labels).item()
        total += labels.shape[0]
    return total_correct / max(total, 1)


def train_one_epoch(
    state: train_state.TrainState,
    train_batches: list,
    mode: str,
    eps: float,
    alpha: float,
    pgd_steps: int,
    surrogate_state: train_state.TrainState | None = None,
) -> tuple:
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, labels in tqdm(train_batches, desc="train", leave=False):
        if mode == "standard":
            state, loss, logits = train_step(state, images, labels)
        elif mode == "whitebox_pgd":
            state, loss, logits = train_step_adversarial(
                state, images, labels, eps, alpha, pgd_steps
            )
        elif mode == "blackbox_pgd":
            assert surrogate_state is not None
            state, loss, logits = train_step_blackbox(
                state,
                images,
                labels,
                surrogate_state.params,
                surrogate_state.apply_fn,
                eps,
                alpha,
                pgd_steps,
            )

        batch_size = labels.shape[0]
        running_loss += float(loss) * batch_size
        running_acc += accuracy(logits, labels) * batch_size
        total_samples += batch_size

    return state, running_loss / total_samples, running_acc / total_samples


def save_checkpoint(path: Path, state: train_state.TrainState, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(path, "wb") as f:
        pickle.dump({"params": state.params, "args": vars(args)}, f)


def load_checkpoint(path: Path, model: nn.Module, key: jax.random.PRNGKey) -> train_state.TrainState:
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)

    params = data["params"]
    tx = optax.sgd(learning_rate=0.1, momentum=0.9, nesterov=True)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CIFAR-10 model with optional adversarial training using JAX/Flax"
    )
    p.add_argument("--mode", choices=["standard", "whitebox_pgd", "blackbox_pgd"], required=True)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--archive-path", type=str, default=None, help="Path to local CIFAR-10 tar/tar.gz")
    p.add_argument("--download-if-missing", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pgd-eps", type=float, default=8 / 255)
    p.add_argument("--pgd-alpha", type=float, default=2 / 255)
    p.add_argument("--pgd-steps", type=int, default=7)
    p.add_argument(
        "--surrogate-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for surrogate model used in blackbox_pgd mode",
    )
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "blackbox_pgd" and not args.surrogate_checkpoint:
        raise ValueError("--surrogate-checkpoint is required in blackbox_pgd mode")

    key = set_seed(args.seed)
    key, subkey = jax.random.split(key)

    train_batches, val_batches, test_batches = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        archive_path=args.archive_path,
        download_if_missing=args.download_if_missing,
    )

    model = SimpleCNN(num_classes=10)
    state = create_train_state(subkey, model, learning_rate=args.lr)

    surrogate_state = None
    if args.mode == "blackbox_pgd":
        surrogate_state = load_checkpoint(
            Path(args.surrogate_checkpoint), model=SimpleCNN(num_classes=10), key=subkey
        )

    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        state, train_loss, train_acc = train_one_epoch(
            state=state,
            train_batches=train_batches,
            mode=args.mode,
            eps=args.pgd_eps,
            alpha=args.pgd_alpha,
            pgd_steps=args.pgd_steps,
            surrogate_state=surrogate_state,
        )
        # validation metrics
        val_acc = evaluate_clean(state, val_batches)

        # compute losses and probabilities for AUC on val and test
        def eval_loss_acc_auc(state, batches):
            total_loss = 0.0
            total = 0
            probs_list = []
            labels_list = []
            for images, labels in batches:
                logits = state.apply_fn({"params": state.params}, images, training=False)
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

        val_loss, val_acc_fine, val_auc = eval_loss_acc_auc(state, val_batches)
        test_loss, test_acc, test_auc = eval_loss_acc_auc(state, test_batches)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc_fine),
            "val_auc": float(val_auc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_auc": float(test_auc),
        }
        history.append(row)
        print(json.dumps(row))

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(Path(args.output), state=state, args=args)

    # Save history to CSV
    import csv
    csv_path = Path(args.output).parent / f"{Path(args.output).stem}_history.csv"
    if history:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        print(f"History saved to {csv_path}")

    print(f"Finished. Best clean accuracy: {best_acc:.4f}. Saved to {args.output}")


if __name__ == "__main__":
    main()

