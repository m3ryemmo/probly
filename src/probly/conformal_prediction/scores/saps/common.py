"""Common for SAPS scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lazy_dispatch.isinstance import LazyType

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
from probly.conformal_prediction.methods.common import Predictor, predict_probs


@lazydispatch
def saps_score_func(
    softmaxprob: np.ndarray,
    label: int,
    rankweight: np.ndarray | None = None,
    u: float | None = None,
) -> float:
    """Compute SAPS Nonconformity Score.

    softmaxprob: 1D-Array mit Softmax-Wahrscheinlichkeiten.
    label: true index
    rankweight: optional Array mit Rang-Indizes (höchste Wahrscheinlichkeit zuerst).
    u: optionaler Zufallswert in [0,1).
    """
    if softmaxprob.ndim != 1:
        raise ValueError

    if not (0 <= label < softmaxprob.shape[0]):
        raise ValueError

    if u is None:
        u = float(np.random.Generator().random())

    # if given no rankweight, sort by descending probabilities
    if rankweight is None:
        rankweight = np.argsort(-softmaxprob)

    # rank position of true label (0-based)
    pos = np.where(rankweight == label)[0]
    if pos.size == 0:
        raise ValueError
    o = int(pos[0]) + 1

    if o == 1:
        return u * float(softmaxprob[rankweight[0]])

    cum_prob = float(np.sum(softmaxprob[rankweight[: o - 2]]))
    pi_o = float(softmaxprob[rankweight[o - 1]])

    return cum_prob + (u * pi_o)


def register(cls: LazyType, func: Callable) -> None:
    """Register a implementation for a specific type."""
    saps_score_func.register(cls=cls, func=func)


class SAPSScore:
    """Sorted Adaptive Prediction Sets (SAPS) nonconformity score."""

    def __init__(
        self,
        model: Predictor,
        rankweight: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize SAPS score with optional rank weights."""
        self.model = model
        self.rankweight = rankweight
        self.rng = np.random.default_rng(random_state)

    def calibration_nonconformity(
        self,
        x_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> np.ndarray:
        """Compute nonconformity scores for calibration data."""
        probs: npt.NDArray[np.floating] = predict_probs(self.model, x_calib)

        n_samples = probs.shape[0]
        us = self.rng.uniform(0, 1, size=n_samples)

        scores = np.empty(n_samples, dtype=float)
        for i in range(n_samples):
            scores[i] = saps_score_func(
                softmaxprob=probs[i],
                label=y_calib[i],
                rankweight=self.rankweight,
                u=us[i],
            )
        return scores

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute scores for all labels."""
        if probs is None:
            probs = predict_probs(self.model, x_test)

        n_samples, n_classes = probs.shape
        scores = np.empty((n_samples, n_classes), dtype=float)

        for i in range(n_samples):
            for label in range(n_classes):
                u = self.rng.uniform(0, 1)
                scores[i, label] = saps_score_func(
                    softmaxprob=probs[i],
                    label=label,
                    rankweight=self.rankweight,
                    u=u,
                )
        return scores
