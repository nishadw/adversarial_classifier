from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_linf_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """Untargeted PGD L-inf adversarial examples.

    Matches the original loss: -mean(sum(one_hot * softmax(logits))).
    Saves/restores model.training so callers don't need to manage it.
    """
    was_training = model.training
    model.eval()

    x = images.clone().detach()
    if random_start:
        x = torch.clamp(x + torch.empty_like(x).uniform_(-epsilon, epsilon), 0.0, 1.0)

    original = images.clone().detach()

    for _ in range(num_steps):
        x = x.detach().requires_grad_(True)
        logits = model(x)
        loss = -torch.mean(
            torch.sum(F.one_hot(labels, 10).float() * F.softmax(logits, dim=1), dim=1)
        )
        loss.backward()
        with torch.no_grad():
            x = torch.clamp(
                original + torch.clamp(x + alpha * x.grad.sign() - original, -epsilon, epsilon),
                0.0, 1.0,
            )

    if was_training:
        model.train()

    return x.detach()
