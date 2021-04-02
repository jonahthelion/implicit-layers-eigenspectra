"""A simple fully connected network."""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class FeedForwardBlock(hk.Module):
    """A 1-layer feedforward network with input injections."""

    def __init__(self,
                 # activation=jax.numpy.tanh,
                 activation=jax.nn.relu,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._activation = activation

    def __call__(self,
                 h: jnp.ndarray,
                 input_embs: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
          h: Hidden, [B, T=1, H].
          input_embs: Linear transformed inputs (Ux + b), [B, T=1, H].
        Returns:
          Array of shape [B, T=1, H].
        """
        hidden_size = h.shape[-1]
        initializer = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")

        # Note that the injected input should be linear transformed and added the bias
        h = hk.Linear(hidden_size, with_bias=False, w_init=initializer)(h) + input_embs
        h = self._activation(h)

        return h

