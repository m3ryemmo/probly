"""Flax/JAX implementation for SAPS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from .common import register


def saps_score_jax(
    probs: Array,
    label: int,
    lambda_val: float = 0.1,
    u: float | None = None,
    key: jrandom.PRNGKeyArray | None = None,
) -> float:
    """Compute SAPS Nonconformity Score for JAX arrays.

    Args:
        probs: 1D array with softmax probabilities.
        label: True label index.
        lambda_val: Lambda value for SAPS.
        u: Optional random value in [0,1). If None, generated from key.
        key: JAX random key for generating u if not provided.

    Returns:
        float: SAPS nonconformity score.
    """
    if probs.ndim == 2:
        if probs.shape[0] != 1:
            raise ValueError
        probs = probs[0]

    if probs.ndim != 1:
        raise ValueError

    if not (0 <= label < probs.shape[0]):
        raise ValueError

    if u is None:
        if key is None:
            u = float(np.random.Generator(0, 1))
        else:
            key, subkey = jrandom.split(key)
            u = float(jrandom.uniform(subkey, shape=()).item())

    max_prob = float(jnp.max(probs))

    sorted_indices = jnp.argsort(-probs)

    # find pos of label
    matches = sorted_indices == label
    pos = jnp.nonzero(matches, size=1)[0]

    if pos.size == 0:
        raise ValueError

    # convert to 1-based rank
    rank = int(pos[0].item()) + 1

    if rank == 1:
        return u * max_prob
    return max_prob + (rank - 2 + u) * lambda_val


# Optional batch helper function for JAX
def saps_score_jax_batch(
    probs: Array,
    labels: Array,
    lambda_val: float = 0.1,
    us: Array | None = None,
    key: jrandom.PRNGKeyArray | None = None,
) -> Array:
    """Batch version of SAPS Nonconformity Score for JAX arrays."""
    n_samples = probs.shape[0]

    if us is None:
        if key is None:
            us = jnp.array(np.random.Generator(0, 1, size=n_samples))
        else:
            key, subkey = jrandom.split(key)
            us = jrandom.uniform(subkey, shape=(n_samples,))

    # Vectorized computation using vmap would be more efficient but
    # for simplicity and consistency, we use a loop
    def compute_for_sample(i: int) -> float:
        return saps_score_jax(
            probs[i : i + 1],  # Keep as 2D for compatibility
            int(labels[i]),
            lambda_val=lambda_val,
            u=float(us[i]),
            key=None,  # u is already provided
        )

    # Use jax.lax.map or simple list comprehension
    scores = jnp.array([compute_for_sample(i) for i in range(n_samples)])
    return scores


# Register the implementation
register(Array, saps_score_jax)
