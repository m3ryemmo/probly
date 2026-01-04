"""Flax/JAX implementation for SAPS scores."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from .common import register


def saps_score_jax(
    softmaxprob: Array,
    label: int,
    rankweight: Array | None = None,
    u: float | None = None,
) -> float:
    """Compute SAPS Nonconformity Score for JAX arrays."""
    if softmaxprob.ndim != 1:
        raise ValueError

    if not (0 <= label < softmaxprob.shape[0]):
        raise ValueError

    if u is None:
        u = float(jnp.random.uniform(jnp.array[0.0], jnp.array[1.0]).item())

    # if given no rankweight, sort by descending probabilities
    if rankweight is None:
        rankweight = jnp.argsort(-softmaxprob)

    # find position of true label (0-based)
    pos = jnp.where(rankweight == label, size=1)[0]
    if pos.size == 0:
        raise ValueError
    o = int(pos[0]) + 1

    if o == 1:
        return u * float(softmaxprob[rankweight[0]])

    cum_prob = float(jnp.sum(softmaxprob[rankweight[: o - 1]]))
    pi_o = float(softmaxprob[rankweight[o - 1]])

    return cum_prob + (u * pi_o)


# Register the implementation
register(Array, saps_score_jax)
register("jaxlib.xla_extension.ArrayImpl", saps_score_jax)
