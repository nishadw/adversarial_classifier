from __future__ import annotations

import jax
import jax.numpy as jnp


def pgd_linf_attack(
    model_fn,
    params: dict,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool = True,
) -> jnp.ndarray:
    """Generate untargeted PGD adversarial examples under L-inf constraint."""
    x = images.copy()

    if random_start:
        noise = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=x.shape,
            minval=-epsilon,
            maxval=epsilon,
        )
        x = jnp.clip(x + noise, 0.0, 1.0)

    original = images.copy()

    for _ in range(num_steps):

        def loss_fn(x_adv):
            logits = model_fn({"params": params}, x_adv, training=False)
            return -jnp.mean(
                jnp.sum(
                    jax.nn.one_hot(labels, 10) * jax.nn.softmax(logits),
                    axis=1,
                )
            )

        grad = jax.grad(loss_fn)(x)
        x = x + alpha * jnp.sign(grad)
        perturbation = jnp.clip(x - original, -epsilon, epsilon)
        x = jnp.clip(original + perturbation, 0.0, 1.0)

    return x
