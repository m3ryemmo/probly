"""Torch for SAPS."""

from __future__ import annotations

import torch

from .common import register


def saps_score_torch(
    softmaxprob: torch.Tensor,
    label: int,
    rankweight: torch.Tensor | None = None,
    u: float | None = None,
) -> float:
    """Compute SAPS Nonconformity Score for torch tensors.

    Args:
        softmaxprob: 1D tensor with softmax probabilities.
        label: true index
        rankweight: optional tensor with rank indices (highest probability first).
        u: optional random value in [0,1).

    Returns:
        float: SAPS nonconformity score.
    """
    if softmaxprob.ndim != 1:
        raise ValueError

    if not (0 <= label < softmaxprob.shape[0]):
        raise ValueError

    if u is None:
        u = float(torch.rand(1).item())

    # if given no rankweight, sort by descending probabilities
    if rankweight is None:
        rankweight = torch.argsort(softmaxprob, descending=True)

    # convert label to tensor for comparison
    label_tensor = torch.tensor(label, device=softmaxprob.device)

    # find position of true label (0-based)
    pos = torch.where(rankweight == label_tensor)[0]
    if pos.numel() == 0:
        raise ValueError

    o = int(pos[0].item()) + 1

    if o == 1:
        return u * float(softmaxprob[rankweight[0]].item())

    # cumulative probability of labels ranked before the true label
    cum_prob = float(torch.sum(softmaxprob[rankweight[: o - 1]]).item())

    # probability of the true label
    pi_o = float(softmaxprob[rankweight[o - 1]].item())

    return cum_prob + (u * pi_o)


register(torch.Tensor, saps_score_torch)
