from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class SimpleCNN(nn.Module):
    """Improved CNN for CIFAR-10 in Flax (NHWC format), targeting >90% accuracy."""

    num_classes: int = 10
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # First conv block: 32x32 -> 32x32
        x = nn.Conv(128, (3, 3), padding=1, dtype=self.dtype)(x)
        x = nn.GroupNorm(num_groups=16, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(128, (3, 3), padding=1, dtype=self.dtype)(x)
        x = nn.GroupNorm(num_groups=16, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Second conv block: 16x16 -> 16x16
        x = nn.Conv(256, (3, 3), padding=1, dtype=self.dtype)(x)
        x = nn.GroupNorm(num_groups=32, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(256, (3, 3), padding=1, dtype=self.dtype)(x)
        x = nn.GroupNorm(num_groups=32, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Third conv block: 8x8 -> 8x8
        x = nn.Conv(512, (3, 3), padding=1, dtype=self.dtype)(x)
        x = nn.GroupNorm(num_groups=64, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(512, (3, 3), padding=1, dtype=self.dtype)(x)
        x = nn.GroupNorm(num_groups=64, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Global average pooling: 4x4x512 -> 512
        x = jnp.mean(x, axis=(1, 2))

        # Dense head
        x = nn.Dense(512, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(256, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)

        return x
